import os
import re
import json
import hashlib
import cv2
from typing import Any, Dict, List, Optional
from PIL import Image
from datasets import Dataset
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from peft import PeftModel, LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from openai import OpenAI

# =========================================================
# 0. Global config
# =========================================================
BASE_MODEL_NAME      = "nvidia/Cosmos-Reason2-8B"
SFT_ADAPTER_PATH     = "/workspace/outputs/Cosmos-Reason2-8B-psi-video-90f-sft"
TRAIN_JSONL          = "/workspace/PSI_change/json_mode_90/trf_train/psi_grpo_balanced.jsonl"
OUTPUT_DIR           = "/workspace/outputs/Cosmos-Reason2-8B-psi-video-90f-grpo"
OPENAI_API_KEY       = os.environ.get("OPENAI_API_KEY", "")
NUM_FRAMES           = 4       # 均匀从视频抽4帧
NUM_GENERATIONS      = 2
MAX_COMPLETION_LENGTH = 250
LEARNING_RATE        = 5e-7
NUM_EPOCHS           = 1
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACC_STEPS       = 1
MAX_SAMPLES          = None

os.makedirs(OUTPUT_DIR, exist_ok=True)
CACHE_DIR = os.path.join(OUTPUT_DIR, "judge_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# =========================================================
# 1. Prompt templates
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
    "These are sequential driving frames ordered from earliest to latest.\n"
    "The TARGET pedestrian is highlighted by the green box in each frame.\n"
    "Reason about the pedestrian's motion and change over time across the full sequence.\n"
    "Task: Predict whether the TARGET pedestrian has crossing intention at t+1."
)

# =========================================================
# 2. Parsing helpers
# =========================================================
ANSWER_RE  = re.compile(r"answer:\s*(yes|no)\b", re.IGNORECASE)
GROUNDS_RE = re.compile(r"grounds:\s*(.*?)(?:\nwarrant:|\Z)", re.IGNORECASE | re.DOTALL)
WARRANT_RE = re.compile(r"warrant:\s*(.*?)(?:\nanswer:|\Z)", re.IGNORECASE | re.DOTALL)

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def extract_answer(text: str) -> str:
    m = ANSWER_RE.search(text or "")
    return m.group(1).lower() if m else "unknown"

def extract_grounds(text: str) -> str:
    m = GROUNDS_RE.search(text or "")
    return normalize_text(m.group(1)) if m else ""

def extract_warrant(text: str) -> str:
    m = WARRANT_RE.search(text or "")
    return normalize_text(m.group(1)) if m else ""

def check_format(text: str) -> bool:
    lines = [x.strip() for x in (text or "").strip().split("\n") if x.strip()]
    if len(lines) != 3:
        return False
    if not lines[0].lower().startswith("grounds:"):
        return False
    if not lines[1].lower().startswith("warrant:"):
        return False
    if not lines[2].lower().startswith("answer:"):
        return False
    return extract_answer(text) in {"yes", "no"}

def lexical_jaccard(a: str, b: str) -> float:
    sa = set(re.findall(r"\w+", (a or "").lower()))
    sb = set(re.findall(r"\w+", (b or "").lower()))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for x in completion:
            if isinstance(x, dict) and "content" in x:
                parts.append(str(x["content"]))
        return "\n".join(parts).strip()
    return str(completion)

# =========================================================
# 3. Dataset — 从 mp4 均匀抽帧，和旧脚本结构一致
# =========================================================
def extract_frames_from_video(video_path: str, n: int = NUM_FRAMES) -> List[Image.Image]:
    """从 mp4 均匀抽 n 帧，返回 PIL Image 列表"""
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    indices = [int(i * total / n) for i in range(n)]
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

def build_prompt_messages(num_images: int) -> List[Dict[str, Any]]:
    user_content = [{"type": "image"} for _ in range(num_images)]
    user_content.append({"type": "text", "text": USER_PROMPT})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": user_content},
    ]

def build_grpo_dataset(jsonl_path: str, max_samples: Optional[int] = None) -> Dataset:
    rows    = []
    skipped = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
            item       = json.loads(line)
            sample_id  = item["id"]
            video_path = item["video"]

            images = extract_frames_from_video(video_path, NUM_FRAMES)
            if len(images) != NUM_FRAMES:
                print(f"  [SKIP] {sample_id}: got {len(images)} frames")
                skipped += 1
                continue

            ground_truth_answer = item["answer"]

            rows.append({
                "sample_id":           sample_id,
                "prompt":              build_prompt_messages(len(images)),
                "images":              images,
                "ground_truth_answer": ground_truth_answer,
            })

    print(f"Dataset: {len(rows)} samples (skipped {skipped})")
    return Dataset.from_list(rows)

train_dataset = build_grpo_dataset(TRAIN_JSONL, max_samples=MAX_SAMPLES)
print(train_dataset)
print("sample_id:", train_dataset[0]["sample_id"])
print("gt answer:", train_dataset[0]["ground_truth_answer"])
print("num images:", len(train_dataset[0]["images"]))

# =========================================================
# 4. OpenAI client (direction judge only)
# =========================================================
_openai_client = OpenAI(api_key=OPENAI_API_KEY)

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
    key = hashlib.sha256(f"{warrant}|{answer}".encode()).hexdigest()[:16]
    cp  = os.path.join(CACHE_DIR, f"dir_{key}.json")
    if os.path.exists(cp):
        with open(cp) as f:
            return json.load(f)["score"]
    try:
        resp = _openai_client.chat.completions.create(
            model="gpt-4.1",
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
        score = 0.5
    with open(cp, "w") as f:
        json.dump({"score": score}, f)
    return score

# =========================================================
# 5. Reward functions
# =========================================================
def format_reward_func(completions, **kwargs):
    texts = [completion_to_text(c) for c in completions]
    return [1.0 if check_format(t) else 0.0 for t in texts]


def answer_correct_reward_func(completions, ground_truth_answer, **kwargs):
    rewards = []
    for c, gt in zip(completions, ground_truth_answer):
        pred = extract_answer(completion_to_text(c))
        rewards.append(1.0 if pred == gt else 0.0)
    return rewards


def case_reasoning_reward(completions, ground_truth_answer, **kwargs):
    """yes/no 对称推理质量：预测对 + warrant 支持该方向 → 1.0"""
    YES_SIGNALS = ["continue", "proceed", "keep crossing", "maintain",
                   "already in", "committed", "moving toward", "running across",
                   "mid-crossing", "lateral motion", "entering", "approaching"]
    NO_SIGNALS  = ["wait", "hesitate", "remain", "stay", "parallel",
                   "stationary", "standing", "away from", "sidewalk",
                   "not crossing", "avoid", "defer", "pause", "retreating"]
    rewards = []
    for c, gt in zip(completions, ground_truth_answer):
        text    = completion_to_text(c)
        pred    = extract_answer(text)
        warrant = extract_warrant(text).lower()
        if pred != gt:
            rewards.append(0.0)
            continue
        signals = YES_SIGNALS if gt == "yes" else NO_SIGNALS
        rewards.append(1.0 if any(s in warrant for s in signals) else 0.5)
    return rewards


def balanced_acc_reward(completions, ground_truth_answer, **kwargs):
    """Batch 内 balanced accuracy，防止模型偏向一类"""
    preds = [extract_answer(completion_to_text(c)) for c in completions]
    gts   = list(ground_truth_answer)
    yes_correct = sum(1 for p,g in zip(preds,gts) if g=="yes" and p=="yes")
    yes_total   = sum(1 for g in gts if g=="yes")
    no_correct  = sum(1 for p,g in zip(preds,gts) if g=="no"  and p=="no")
    no_total    = sum(1 for g in gts if g=="no")
    recall_yes  = yes_correct / max(yes_total, 1)
    recall_no   = no_correct  / max(no_total,  1)
    bal_acc     = (recall_yes + recall_no) / 2
    return [bal_acc] * len(completions)


def direction_reward_func(completions, **kwargs):
    """Warrant 方向和 answer 一致（LLM judge，binary，cached）"""
    rewards = []
    for c in completions:
        text    = completion_to_text(c)
        warrant = extract_warrant(text)
        answer  = extract_answer(text)
        rewards.append(_direction_llm(warrant, answer))
    return rewards


def non_redundancy_reward_func(completions, **kwargs):
    """Warrant 和 grounds/answer 词汇重叠越少越好"""
    rewards = []
    for c in completions:
        text    = completion_to_text(c)
        grounds = extract_grounds(text)
        warrant = extract_warrant(text)
        answer  = extract_answer(text)
        if not warrant:
            rewards.append(0.0)
            continue
        overlap = max(lexical_jaccard(warrant, grounds), lexical_jaccard(warrant, answer))
        rewards.append(max(0.0, 1.0 - overlap))
    return rewards


def probabilistic_reward_func(completions, **kwargs):
    """Warrant 使用概率性语言"""
    hedges = ["typically", "tends to", "often", "usually",
              "generally", "likely", "tend to", "commonly"]
    rewards = []
    for c in completions:
        w = extract_warrant(completion_to_text(c)).lower()
        rewards.append(1.0 if any(h in w for h in hedges) else 0.0)
    return rewards

# =========================================================
# 6. Load model
# =========================================================
print("Loading processor...")
processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)

print("Loading base model...")
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
)

print("Loading SFT adapter...")
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=False)

print("Merging SFT adapter...")
model = model.merge_and_unload()

print("Adding GRPO LoRA...")
grpo_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, grpo_lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()

# 修复 warnings_issued
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
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    bf16=True,
    logging_steps=1,
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    max_completion_length=MAX_COMPLETION_LENGTH,
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
        direction_reward_func,       # 0.15
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
with open(f"{OUTPUT_DIR}/log_history.json", "w", encoding="utf-8") as f:
    json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
print(f"Done. Output: {OUTPUT_DIR}")

# CUDA_VISIBLE_DEVICES=1 python /workspace/train_psi_grpo_claude_new.py