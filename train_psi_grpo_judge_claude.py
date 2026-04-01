import os
import re
import json
import hashlib
from typing import Any, Dict, List, Optional

from PIL import Image
from datasets import Dataset

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer
from openai import OpenAI


# =========================================================
# 0. Global config
# =========================================================
BASE_MODEL_NAME = "nvidia/Cosmos-Reason2-8B"
SFT_ADAPTER_PATH = "/workspace/outputs/Cosmos-Reason2-8B-psi-video-sft"
TRAIN_JSONL = "/workspace/psi_llava_easy200.jsonl"
OUTPUT_DIR = "/workspace/outputs/Cosmos-Reason2-8B-psi-grpo-from-sft"


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "").strip()
OPENAI_JUDGE_MODEL = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-5.2")

MAX_SAMPLES = 200
SELECTED_FRAME_INDICES = [0, 5, 10, 15]
# SELECTED_FRAME_INDICES = None

NUM_GENERATIONS = 2
MAX_COMPLETION_LENGTH = 196
LEARNING_RATE = 1e-6
NUM_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACC_STEPS = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)
CACHE_DIR = os.path.join(OUTPUT_DIR, "judge_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# =========================================================
# 1. Prompt templates
# =========================================================
SYSTEM_PROMPT = (
    "You are an expert assistant for pedestrian-intention reasoning from sequential driving frames.\n"
    "Focus only on the TARGET pedestrian highlighted by the green box.\n"
    "Use concise and factual language.\n"
    "Grounds must describe visible evidence from the frames.\n"
    "Warrant must provide a general rule linking the observed evidence to the pedestrian's likely intention.\n"
    "Output EXACTLY 3 lines in this order:\n"
    "grounds: <1–2 sentences, concrete observations>\n"
    "warrant: <1 sentence general rule linking grounds to intention>\n"
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
ANSWER_RE = re.compile(r"answer:\s*(yes|no)\b", re.IGNORECASE)
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
# 3. Dataset
# =========================================================
def build_prompt_messages(num_images: int) -> List[Dict[str, Any]]:
    user_content = []
    for _ in range(num_images):
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": USER_PROMPT})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]


def build_grpo_dataset(
    jsonl_path: str,
    max_samples: Optional[int] = None,
    selected_frame_indices=None,
) -> Dataset:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
            item = json.loads(line)
            sample_id = item["id"]
            image_paths = item["images"]
            if selected_frame_indices is not None:
                image_paths = [image_paths[i] for i in selected_frame_indices]

            reference_completion = item["completion"][0]["content"]
            reference_grounds = extract_grounds(reference_completion)
            reference_warrant = extract_warrant(reference_completion)
            ground_truth_answer = extract_answer(reference_completion)
            images = [Image.open(p).convert("RGB") for p in image_paths]

            rows.append({
                "sample_id": sample_id,
                "prompt": build_prompt_messages(len(image_paths)),
                "image_paths": image_paths,
                "images": images,
                "reference_completion": reference_completion,
                "reference_grounds": reference_grounds,
                "reference_warrant": reference_warrant,
                "ground_truth_answer": ground_truth_answer,
            })
    return Dataset.from_list(rows)


train_dataset = build_grpo_dataset(
    TRAIN_JSONL,
    max_samples=MAX_SAMPLES,
    selected_frame_indices=SELECTED_FRAME_INDICES,
)
print(train_dataset)
print("sample_id:", train_dataset[0]["sample_id"])
print("gt answer:", train_dataset[0]["ground_truth_answer"])
print("reference grounds:", train_dataset[0]["reference_grounds"])
print("num images:", len(train_dataset[0]["images"]))


# =========================================================
# 4. OpenAI judge client with cache
# =========================================================
class OpenAIJudge:
    def __init__(self, model: str, api_key: str, base_url: str = "", cache_dir: str = "./judge_cache"):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.json")

    def score(self, system_prompt: str, user_prompt: str, default_score: float = 0.0) -> Dict[str, Any]:
        payload = {"model": self.model, "system_prompt": system_prompt, "user_prompt": user_prompt}
        cache_path = self._cache_path(payload)

        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        if not OPENAI_API_KEY:
            return {"score": default_score, "reason": "OPENAI_API_KEY not set"}

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "judge_result",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "reason": {"type": "string"}
                            },
                            "required": ["score", "reason"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                },
            )
            data = json.loads(response.output_text)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return data
        except Exception as e:
            return {"score": default_score, "reason": f"judge error: {e}"}


judge = OpenAIJudge(
    model=OPENAI_JUDGE_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    cache_dir=CACHE_DIR,
)


# =========================================================
# 5. Judge prompts
# =========================================================
GROUNDING_JUDGE_SYSTEM = """You evaluate whether predicted grounds faithfully preserve the key visible evidence described by a reference grounds statement.

Return ONLY JSON with:
{"score": <float between 0 and 1>, "reason": "<short reason>"}

Scoring:
- 1.0 = predicted grounds preserve the essential visible evidence from the reference and do not add major unsupported claims
- 0.5 = partially faithful but incomplete or somewhat distorted
- 0.0 = largely unsupported, contradictory, or hallucinated
"""

BRIDGE_JUDGE_SYSTEM = """You evaluate Toulmin-style reasoning for pedestrian intention prediction.

Return ONLY JSON with:
{"score": <float between 0 and 1>, "reason": "<short reason>"}

Score how well the warrant functions as a bridge from the grounds to the answer.

High score:
- the warrant is relevant to the grounds
- the warrant is general rather than a repetition
- the answer follows naturally from grounds + warrant

Low score:
- the warrant is irrelevant
- the warrant merely repeats the grounds
- the warrant merely restates the answer
- the warrant does not justify the answer
"""

CONSISTENCY_JUDGE_SYSTEM = """You evaluate the overall coherence of a Toulmin-style reasoning chain.

Return ONLY JSON with:
{"score": <float between 0 and 1>, "reason": "<short reason>"}

A high score means:
- grounds, warrant, and answer are mutually consistent
- the answer is supported by the warrant and grounded in the observations

A low score means:
- contradiction
- weak linkage
- answer does not follow
"""


# =========================================================
# 6. Reward functions — 单进程，无需broadcast
# =========================================================
def format_reward_func(completions, **kwargs):
    print("[debug] enter format_reward_func")
    texts = [completion_to_text(c) for c in completions]
    return [1.0 if check_format(t) else 0.0 for t in texts]


def answer_correct_reward_func(completions, ground_truth_answer, **kwargs):
    print("[debug] enter answer_correct_reward_func")
    texts = [completion_to_text(c) for c in completions]
    rewards = []
    for text, gt in zip(texts, ground_truth_answer):
        pred = extract_answer(text)
        rewards.append(1.0 if pred == gt else 0.0)
    return rewards


def grounding_faithfulness_reward_func(completions, reference_grounds, **kwargs):
    print("[debug] enter grounding_faithfulness_reward_func")
    texts = [completion_to_text(c) for c in completions]
    rewards = []
    for text, ref_g in zip(texts, reference_grounds):
        pred_g = extract_grounds(text)
        user_prompt = (
            f"Reference grounds:\n{ref_g}\n\n"
            f"Predicted grounds:\n{pred_g}\n\n"
            f"Evaluate whether the predicted grounds faithfully preserve the key visible evidence "
            f"described by the reference grounds."
        )
        result = judge.score(GROUNDING_JUDGE_SYSTEM, user_prompt, default_score=0.0)
        rewards.append(float(result["score"]))
    return rewards


def warrant_bridge_reward_func(completions, **kwargs):
    print("[debug] enter warrant_bridge_reward_func")
    texts = [completion_to_text(c) for c in completions]
    rewards = []
    for text in texts:
        grounds = extract_grounds(text)
        warrant = extract_warrant(text)
        answer = extract_answer(text)
        user_prompt = (
            f"grounds:\n{grounds}\n\nwarrant:\n{warrant}\n\nanswer:\n{answer}\n\n"
            f"Score how well the warrant functions as a bridge from the grounds to the answer."
        )
        result = judge.score(BRIDGE_JUDGE_SYSTEM, user_prompt, default_score=0.0)
        rewards.append(float(result["score"]))
    return rewards


def global_consistency_reward_func(completions, **kwargs):
    print("[debug] enter global_consistency_reward_func")
    texts = [completion_to_text(c) for c in completions]
    rewards = []
    for text in texts:
        grounds = extract_grounds(text)
        warrant = extract_warrant(text)
        answer = extract_answer(text)
        user_prompt = (
            f"grounds:\n{grounds}\n\nwarrant:\n{warrant}\n\nanswer:\n{answer}\n\n"
            f"Evaluate the overall coherence and internal consistency of this reasoning chain."
        )
        result = judge.score(CONSISTENCY_JUDGE_SYSTEM, user_prompt, default_score=0.0)
        rewards.append(float(result["score"]))
    return rewards


def nonredundancy_reward_func(completions, **kwargs):
    print("[debug] enter nonredundancy_reward_func")
    texts = [completion_to_text(c) for c in completions]
    rewards = []
    for text in texts:
        grounds = extract_grounds(text)
        warrant = extract_warrant(text)
        answer = extract_answer(text)
        if not warrant:
            rewards.append(0.0)
            continue
        overlap_ground = lexical_jaccard(warrant, grounds)
        overlap_answer = lexical_jaccard(warrant, answer)
        redundancy = max(overlap_ground, overlap_answer)
        rewards.append(max(0.0, 1.0 - redundancy))
    return rewards


# =========================================================
# 7. Load model — ✅ 用 device_map="auto" 单进程双卡，彻底避开DDP
# =========================================================
processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)

base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    # ✅ device_map="auto" 让HF自动把模型层分布到两张卡
    device_map="cuda:0",   # ✅ 固定单卡，不用auto
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
)

model = PeftModel.from_pretrained(
    base_model,
    SFT_ADAPTER_PATH,
    is_trainable=True,
)
model.enable_input_require_grads()

for obj in [base_model, model]:
    try:
        if not hasattr(obj, "warnings_issued"):
            obj.warnings_issued = {}
    except Exception:
        pass

try:
    inner = getattr(model, "base_model", None)
    if inner is not None and not hasattr(inner, "warnings_issued"):
        inner.warnings_issued = {}
except Exception:
    pass


# =========================================================
# 8. GRPO config — ✅ 单进程模式，去掉所有DDP相关参数
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
    reward_weights=[
        0.15,  # format
        0.25,  # answer correctness
        0.20,  # grounding faithfulness
        0.25,  # warrant bridge
        0.10,  # global consistency
        0.05,  # non-redundancy
    ],
    use_vllm=False,
    report_to="tensorboard",
    log_completions=True,
    gradient_checkpointing=False,
    # ✅ 不设置 ddp_find_unused_parameters 和 ddp_timeout，单进程不需要
)


# =========================================================
# 9. Trainer
# =========================================================
print("[debug] before trainer init")
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        format_reward_func,
        answer_correct_reward_func,
        grounding_faithfulness_reward_func,
        warrant_bridge_reward_func,
        global_consistency_reward_func,
        nonredundancy_reward_func,
    ],
    args=training_args,
    train_dataset=train_dataset,
    processing_class=processor,
)
print("[debug] after trainer init")
print("[debug] before trainer.train")
trainer.train()
print("[debug] after trainer.train")

trainer.save_model(OUTPUT_DIR)

with open(f"{OUTPUT_DIR}/trainer_state_log_history.json", "w", encoding="utf-8") as f:
    json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)

print(f"Saved GRPO outputs to: {OUTPUT_DIR}")
print(f"Judge model: {OPENAI_JUDGE_MODEL}")

# CUDA_VISIBLE_DEVICES=0 python train_psi_grpo_judge_claude.py