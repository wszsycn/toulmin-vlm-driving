"""
GRPO training for pedestrian intention prediction
- 从 mp4 均匀抽 8 帧（带 bbox overlay）
- Rule-based reward functions（无 LLM judge，除方向一致性）
- 从 SFT checkpoint 开始
"""

import os, re, json, time, hashlib
from typing import Any, Dict, List, Optional
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from datasets import Dataset
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from peft import PeftModel, LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from openai import OpenAI

# =========================================================
# 0. Config
# =========================================================
BASE_MODEL_NAME  = "nvidia/Cosmos-Reason2-8B"
SFT_ADAPTER_PATH = "/workspace/outputs/Cosmos-Reason2-8B-psi-video-90f-sft"
TRAIN_JSONL      = "/workspace/PSI_change/json_mode_90/trf_train/psi_grpo_balanced.jsonl"
OUTPUT_DIR       = "/workspace/outputs/Cosmos-Reason2-8B-psi-video-90f-grpo"
CACHE_DIR        = os.path.join(OUTPUT_DIR, "judge_cache")
FRAMES_DIR       = "/workspace/PSI_change/json_mode_90/grpo_frames"

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(CACHE_DIR,   exist_ok=True)
os.makedirs(FRAMES_DIR,  exist_ok=True)
NUM_FRAMES         = 8       # 从视频抽帧数
NUM_GENERATIONS    = 2
MAX_COMPLETION_LEN = 250
LEARNING_RATE      = 5e-7
NUM_EPOCHS         = 1
MAX_SAMPLES        = None    # None = 用全部 510 条

# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(CACHE_DIR, exist_ok=True)

# =========================================================
# 1. Prompts
# =========================================================
SYSTEM_PROMPT = (
    "You are an expert autonomous driving assistant specializing in pedestrian behavior analysis. "
    "When given a driving video, you analyze the TARGET pedestrian (highlighted by a green bounding box) "
    "and predict their crossing intention using the Toulmin argument structure:\n"
    "- grounds: concrete visual observations (posture, movement, position, gaze)\n"
    "- warrant: a general principle (physical law, social norm, or traffic rule) linking observations to the conclusion\n"
    "- answer: yes or no\n\n"
    "Always output exactly 3 lines in this format:\n"
    "grounds: <observations>\n"
    "warrant: <general rule>\n"
    "answer: <yes/no>\n"
    "Do not add any extra text before or after these 3 lines."
)

USER_PROMPT = (
    "Watch the driving frames and predict: will the TARGET pedestrian "
    "attempt to cross in front of the vehicle in the next moment?"
)

# =========================================================
# 2. Extract frames from mp4
# =========================================================
def extract_and_cache_frames(video_path: str, sample_id: str,
                              n: int = NUM_FRAMES) -> List[str]:
    """
    从 mp4 均匀抽 n 帧，存为 jpg 文件，返回图片路径列表。
    已存在则直接返回，不重复解码。
    """
    out_dir = os.path.join(FRAMES_DIR, sample_id)
    os.makedirs(out_dir, exist_ok=True)

    # 检查是否已提取
    existing = sorted(Path(out_dir).glob("frame_*.jpg"))
    if len(existing) == n:
        return [str(p) for p in existing]

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    indices = [int(i * total / n) for i in range(n)]
    paths   = []
    for out_idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = os.path.join(out_dir, f"frame_{out_idx:04d}.jpg")
        cv2.imwrite(out_path, frame)  # cv2 直接写 BGR，jpg 格式
        paths.append(out_path)

    cap.release()
    return paths


def load_images_from_paths(paths: List[str]) -> List[Image.Image]:
    """读取 jpg 文件为 PIL Image（RGB）"""
    images = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"  [WARN] cannot load {p}: {e}")
    return images

# =========================================================
# 3. Dataset
# =========================================================
def build_prompt_messages(n_images: int) -> List[Dict]:
    user_content = [{"type": "image"} for _ in range(n_images)]
    user_content.append({"type": "text", "text": USER_PROMPT})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": user_content},
    ]

def build_grpo_dataset(jsonl_path: str, max_samples=None) -> Dataset:
    rows    = []
    skipped = 0
    with open(jsonl_path) as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            item       = json.loads(line)
            sample_id  = item["id"]
            video_path = item["video"]

            # 提取帧（已存在则跳过解码）
            frame_paths = extract_and_cache_frames(video_path, sample_id, NUM_FRAMES)
            if not frame_paths:
                print(f"  [SKIP] no frames: {video_path}")
                skipped += 1
                continue

            # 读取图片
            images = load_images_from_paths(frame_paths)
            if len(images) != NUM_FRAMES:
                print(f"  [SKIP] incomplete frames: {sample_id}")
                skipped += 1
                continue

            rows.append({
                "sample_id":           sample_id,
                "prompt":              build_prompt_messages(len(images)),
                "images":              images,
                "ground_truth_answer": item["answer"],
                "video_path":          video_path,
                "frame_paths":         frame_paths,
            })

    print(f"Dataset: {len(rows)} samples (skipped {skipped})")
    return Dataset.from_list(rows)

train_dataset = build_grpo_dataset(TRAIN_JSONL, max_samples=MAX_SAMPLES)

# =========================================================
# 4. Parsing helpers
# =========================================================
ANSWER_RE  = re.compile(r"answer:\s*(yes|no)\b", re.IGNORECASE)
GROUNDS_RE = re.compile(r"grounds:\s*(.*?)(?:\nwarrant:|\Z)", re.IGNORECASE | re.DOTALL)
WARRANT_RE = re.compile(r"warrant:\s*(.*?)(?:\nanswer:|\Z)", re.IGNORECASE | re.DOTALL)

def normalize(s): return re.sub(r"\s+", " ", (s or "").strip())
def extract_answer(t):  m = ANSWER_RE.search(t or "");  return m.group(1).lower() if m else "unknown"
def extract_grounds(t): m = GROUNDS_RE.search(t or ""); return normalize(m.group(1)) if m else ""
def extract_warrant(t): m = WARRANT_RE.search(t or ""); return normalize(m.group(1)) if m else ""

def to_text(c: Any) -> str:
    if isinstance(c, str): return c
    if isinstance(c, list):
        return "\n".join(str(x.get("content","")) for x in c if isinstance(x,dict)).strip()
    return str(c)

def check_format(text: str) -> bool:
    lines = [l.strip() for l in (text or "").strip().split("\n") if l.strip()]
    return (len(lines) == 3
            and lines[0].lower().startswith("grounds:")
            and lines[1].lower().startswith("warrant:")
            and lines[2].lower().startswith("answer:")
            and extract_answer(text) in {"yes", "no"})

def jaccard(a: str, b: str) -> float:
    sa = set(re.findall(r"\w+", a.lower()))
    sb = set(re.findall(r"\w+", b.lower()))
    return len(sa & sb) / max(1, len(sa | sb))

# =========================================================
# 5. Rule-based reward functions
# =========================================================

def format_reward_func(completions, **kwargs):
    return [1.0 if check_format(to_text(c)) else 0.0 for c in completions]


def answer_correct_reward_func(completions, ground_truth_answer, **kwargs):
    rewards = []
    for c, gt in zip(completions, ground_truth_answer):
        pred = extract_answer(to_text(c))
        rewards.append(1.0 if pred == gt else 0.0)
    return rewards


def warrant_type_reward_func(completions, **kwargs):
    """Physical law / social norm / traffic rule → 高分；纯行为重复 → 低分"""
    physical = ["momentum", "trajectory", "inertia", "motion", "velocity",
                "lateral", "body orientation", "committed"]
    social   = ["social", "norm", "convention", "acknowledge", "signal",
                "awareness", "expect", "assume"]
    traffic  = ["crosswalk", "right of way", "yield", "walk signal",
                "traffic rule", "marked crossing", "pedestrian signal"]

    rewards = []
    for c in completions:
        w = extract_warrant(to_text(c)).lower()
        if not w:
            rewards.append(0.0)
            continue
        if any(k in w for k in physical): rewards.append(1.0)
        elif any(k in w for k in traffic): rewards.append(1.0)
        elif any(k in w for k in social):  rewards.append(0.8)
        else:                              rewards.append(0.3)
    return rewards


def non_redundancy_reward_func(completions, **kwargs):
    """Warrant 和 grounds/answer 的词汇重叠越少越好"""
    rewards = []
    for c in completions:
        text    = to_text(c)
        grounds = extract_grounds(text)
        warrant = extract_warrant(text)
        answer  = extract_answer(text)
        if not warrant:
            rewards.append(0.0)
            continue
        overlap = max(jaccard(warrant, grounds), jaccard(warrant, answer))
        rewards.append(max(0.0, 1.0 - overlap))
    return rewards


def probabilistic_reward_func(completions, **kwargs):
    """Warrant 使用概率性语言"""
    hedges = ["typically", "tends to", "often", "usually",
              "generally", "likely", "tend to", "commonly"]
    rewards = []
    for c in completions:
        w = extract_warrant(to_text(c)).lower()
        rewards.append(1.0 if any(h in w for h in hedges) else 0.0)
    return rewards


def length_reward_func(completions, **kwargs):
    """Warrant 长度在 12-25 词"""
    rewards = []
    for c in completions:
        w = extract_warrant(to_text(c))
        n = len(w.split())
        if 12 <= n <= 25:  rewards.append(1.0)
        elif n < 12:       rewards.append(n / 12)
        else:              rewards.append(max(0.0, 1.0 - (n - 25) / 25))
    return rewards


# ── Direction reward (LLM, binary, temperature=0, cached) ──────────────────
_openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

DIRECTION_SYSTEM = """\
You are a logic checker. Answer ONLY with 0 or 1. No explanation.

answer=yes means pedestrian WILL cross.
answer=no means pedestrian will NOT cross.
Output 1 if the warrant logically supports the answer direction, 0 if it contradicts or is unrelated.

Examples:
Warrant: Pedestrians already in motion across a road tend to continue toward the far curb.
Answer: yes → 1

Warrant: Pedestrians already in motion across a road tend to continue toward the far curb.
Answer: no → 0

Warrant: People standing at the roadside without stepping forward typically wait rather than entering traffic.
Answer: no → 1

Warrant: People standing at the roadside without stepping forward typically wait rather than entering traffic.
Answer: yes → 0"""

def _direction_llm(warrant: str, answer: str) -> float:
    if not warrant:
        return 0.0
    key   = hashlib.sha256(f"{warrant}|{answer}".encode()).hexdigest()[:16]
    cp    = os.path.join(CACHE_DIR, f"dir_{key}.json")
    if os.path.exists(cp):
        with open(cp) as f:
            return json.load(f)["score"]
    try:
        resp = _openai_client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": DIRECTION_SYSTEM},
                {"role": "user",   "content": f"Warrant: {warrant}\nAnswer: {answer}\n\nOutput 0 or 1:"},
            ],
            temperature=0.0,
            max_completion_tokens=4,
        )
        raw   = resp.choices[0].message.content.strip()
        score = 1.0 if raw == "1" else 0.0
    except Exception as e:
        print(f"  [direction judge error]: {e}")
        score = 0.5  # fallback neutral
    with open(cp, "w") as f:
        json.dump({"score": score}, f)
    return score

def case_reasoning_reward(completions, ground_truth_answer, **kwargs):
    """
    对 yes 和 no 都有对称的推理质量奖励：
    - 预测正确 + warrant 支持该方向 → 1.0
    - 预测正确 + warrant 方向弱    → 0.5
    - 预测错误                     → 0.0
    """
    YES_SIGNALS = [
        "continue", "proceed", "keep crossing", "maintain",
        "already in", "committed", "moving toward", "running across",
        "mid-crossing", "lateral motion", "entering", "approaching"
    ]
    NO_SIGNALS = [
        "wait", "hesitate", "remain", "stay", "parallel",
        "stationary", "standing", "away from", "sidewalk",
        "not crossing", "avoid", "defer", "pause", "retreating"
    ]

    rewards = []
    for c, gt in zip(completions, ground_truth_answer):
        text    = to_text(c)
        pred    = extract_answer(text)
        warrant = extract_warrant(text).lower()

        if pred != gt:
            rewards.append(0.0)
            continue

        # 预测正确，看 warrant 是否支持该方向
        if gt == "yes":
            signals = YES_SIGNALS
        else:
            signals = NO_SIGNALS

        if any(s in warrant for s in signals):
            rewards.append(1.0)   # 预测对 + warrant 明确支持
        else:
            rewards.append(0.5)   # 预测对但 warrant 方向弱

    return rewards


def direction_reward_func(completions, **kwargs):
    """Warrant 方向和 answer 一致（LLM judge，binary，cached）"""
    rewards = []
    for c in completions:
        text    = to_text(c)
        warrant = extract_warrant(text)
        answer  = extract_answer(text)
        rewards.append(_direction_llm(warrant, answer))
    return rewards

def balanced_acc_reward(completions, ground_truth_answer, **kwargs):
    """
    Batch 内的 balanced accuracy 作为 reward
    鼓励模型在 yes 和 no 上都预测正确，避免偏向一类
    """
    preds = [extract_answer(to_text(c)) for c in completions]
    gts   = list(ground_truth_answer)

    yes_correct = sum(1 for p,g in zip(preds,gts) if g=="yes" and p=="yes")
    yes_total   = sum(1 for g in gts if g=="yes")
    no_correct  = sum(1 for p,g in zip(preds,gts) if g=="no"  and p=="no")
    no_total    = sum(1 for g in gts if g=="no")

    recall_yes = yes_correct / max(yes_total, 1)
    recall_no  = no_correct  / max(no_total,  1)
    bal_acc    = (recall_yes + recall_no) / 2

    return [bal_acc] * len(completions)
# =========================================================
# 6. Load model (SFT checkpoint)
# =========================================================
print("Loading processor...")
processor  = AutoProcessor.from_pretrained(BASE_MODEL_NAME)
print("Loading base model...")
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
)

# 先加载 SFT adapter，再在其上加 GRPO LoRA
print("Loading SFT adapter...")
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=False)
print("Merging adapter...")
model = model.merge_and_unload()  # 把 SFT adapter 合并进 base model
print("Adding GRPO LoRA...")
grpo_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)
model = get_peft_model(model, grpo_lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()

# 修复 warnings_issued 属性
for obj in [model, model.base_model, model.base_model.model]:
    try:
        if not hasattr(obj, "warnings_issued"):
            obj.warnings_issued = {}
    except Exception:
        pass
# =========================================================
# 7. GRPO config
# =========================================================
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    bf16=True,
    logging_steps=1,
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    max_completion_length=MAX_COMPLETION_LEN,
    num_generations=NUM_GENERATIONS,
    generation_batch_size=NUM_GENERATIONS,
    temperature=0.9,
    beta=0.01,
    dataloader_num_workers=0,
    reward_weights=[
        0.25,  # answer_correct
        0.20,  # case_reasoning
        0.15,  # balanced_acc
        0.15,  # direction (LLM)
        0.10,  # format
        0.10,  # non_redundancy
        0.05,  # probabilistic
    ],
    use_vllm=False,
    report_to="tensorboard",
    log_completions=True,
    gradient_checkpointing=False,
)

# =========================================================
# 8. Trainer
# =========================================================
print("Init trainer...")
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        answer_correct_reward_func,  # 0.25
        case_reasoning_reward,       # 0.20
        balanced_acc_reward,         # 0.15
        direction_reward_func,       # 0.15 LLM judge
        format_reward_func,          # 0.10
        non_redundancy_reward_func,  # 0.10
        probabilistic_reward_func,   # 0.05
    ],
    args=training_args,
    train_dataset=train_dataset,
    processing_class=processor,
)
print("Start training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
with open(f"{OUTPUT_DIR}/log_history.json", "w") as f:
    json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
print(f"Done. Output: {OUTPUT_DIR}")

# CUDA_VISIBLE_DEVICES=1 python train_psi_grpo.py
# CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false python train_psi_grpo.py