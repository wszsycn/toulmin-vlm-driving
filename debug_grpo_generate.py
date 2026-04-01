import json
from PIL import Image
import torch
from datasets import Dataset
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from peft import PeftModel

BASE_MODEL_NAME = "nvidia/Cosmos-Reason2-8B"
SFT_ADAPTER_PATH = "/workspace/outputs/Cosmos-Reason2-8B-psi-video-sft"
TRAIN_JSONL = "/workspace/psi_llava_easy200.jsonl"

SYSTEM_PROMPT = (
    "You are an expert assistant for pedestrian-intention reasoning from sequential driving frames.\n"
    "Focus only on the TARGET pedestrian highlighted by the green box.\n"
    "Use concise and factual language.\n"
    "Grounds must describe visible evidence from the frames.\n"
    "Warrant must provide a general rule linking the observed evidence to the pedestrian’s likely intention.\n"
    "Output EXACTLY 3 lines in this order:\n"
    "grounds: <1–2 sentences, concrete observations>\n"
    "warrant: <1 sentence general rule linking grounds to intention>\n"
    "answer: <yes/no>\n"
    "Do not add any extra text before or after these 3 lines."
)

USER_PROMPT = (
    "These are sequential driving frames ordered from earliest to latest.\n"
    "The TARGET pedestrian is highlighted by the green box in each frame.\n"
    "Reason about the pedestrian’s motion and change over time across the full sequence.\n"
    "Task: Predict whether the TARGET pedestrian has crossing intention at t+1."
)

SELECTED_FRAME_INDICES = [0, 5, 10, 15]

def build_prompt_messages(image_paths):
    user_content = []
    for img_path in image_paths:
        user_content.append({"type": "image", "image": img_path})
    user_content.append({"type": "text", "text": USER_PROMPT})

    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]

with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
    item = json.loads(next(f))

image_paths = item["images"]
image_paths = [image_paths[i] for i in SELECTED_FRAME_INDICES]
messages = build_prompt_messages(image_paths)

print("loading images...")
images = [Image.open(p).convert("RGB") for p in image_paths]
print("num images:", len(images))

print("loading processor...")
processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)

print("loading base model...")
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

print("loading adapter...")
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=False)
model.eval()

print("apply chat template...")
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
print(text[:500])

print("processor encode...")
inputs = processor(
    text=[text],
    images=[images],   # 注意这里是 batch 维度
    return_tensors="pt",
    padding=True,
)

for k, v in inputs.items():
    if hasattr(v, "shape"):
        print(k, v.shape)

inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}

print("start generate...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
    )

print("decode...")
decoded = processor.batch_decode(outputs, skip_special_tokens=True)
print(decoded[0])
print("done")