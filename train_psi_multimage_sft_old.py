import json
from pathlib import Path
from PIL import Image

import torch
from datasets import Dataset
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from qwen_vl_utils import process_vision_info

import os

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
is_main_process = local_rank == 0

# =========================
# 1. 配置
# =========================
jsonl_path = "/workspace/psi_llava_easy200.jsonl"
model_name = "nvidia/Cosmos-Reason2-8B"
output_dir = "/workspace/outputs/Cosmos-Reason2-8B-psi-multimage-sft"

max_samples = 200   # 先从 20 条开始，通了再加到 200


# =========================
# 2. 读 JSONL -> messages
# =========================
def build_messages_from_jsonl(jsonl_path, max_samples=None):
    samples = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break

            item = json.loads(line)

            image_paths = item["images"]
            user_text = item["prompt"][0]["content"]
            assistant_text = item["completion"][0]["content"]

            user_content = []
            for img_path in image_paths:
                user_content.append({
                    "type": "image",
                    "image": img_path
                })

            user_content.append({
                "type": "text",
                "text": user_text
            })

            messages = [
                {
                    "role": "user",
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": assistant_text
                        }
                    ]
                }
            ]

            samples.append({
                "messages": messages
            })

    return Dataset.from_list(samples)


train_dataset = build_messages_from_jsonl(jsonl_path, max_samples=max_samples)
print(train_dataset)
print(train_dataset[0]["messages"][0]["content"][:2])
print(train_dataset[0]["messages"][1]["content"])

def preview_raw_sample(dataset, idx=0):
    sample = dataset[idx]
    user_content = sample["messages"][0]["content"]
    assistant_content = sample["messages"][1]["content"]

    num_images = sum(1 for x in user_content if x["type"] == "image")
    user_texts = [x["text"] for x in user_content if x["type"] == "text"]

    print(f"\n===== Raw sample {idx} =====")
    print("num image blocks:", num_images)
    print("first 2 image paths:")
    for x in user_content[:2]:
        print(x)
    print("user text:")
    print(user_texts[0] if user_texts else None)
    print("assistant text:")
    print(assistant_content[0]["text"])


if is_main_process:
    print(train_dataset)
    preview_raw_sample(train_dataset, 0)
    preview_raw_sample(train_dataset, 1)
# =========================
# 3. 加载模型和 processor
# =========================
processor = AutoProcessor.from_pretrained(model_name)

# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     model_name,
#     dtype="auto",
#     device_map="auto",
#     quantization_config=BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#     ),
# )


import os
import torch

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="auto",
    device_map={"": local_rank},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "down_proj",
        "o_proj",
        "k_proj",
        "q_proj",
        "gate_proj",
        "up_proj",
        "v_proj",
    ],
)


# =========================
# 4. 自定义 collator
# =========================
from PIL import Image
from qwen_vl_utils import process_vision_info


class MultiImageQwenCollator:
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model

        self.image_token_id = getattr(model.config, "image_token_id", None)
        self.video_token_id = getattr(model.config, "video_token_id", None)
        self.pad_token_id = processor.tokenizer.pad_token_id

    def __call__(self, features):
        batch_messages = []

        for feature in features:
            messages = feature["messages"]

            cleaned_messages = []
            for turn in messages:
                new_turn = {
                    "role": turn["role"],
                    "content": []
                }

                for block in turn["content"]:
                    block_type = block["type"]

                    if block_type == "image":
                        img_path = block.get("image", None)
                        if img_path is None:
                            raise ValueError(f"Found image block with None image path: {block}")

                        img = Image.open(img_path).convert("RGB")
                        new_turn["content"].append({
                            "type": "image",
                            "image": img
                        })

                    elif block_type == "text":
                        text_value = block.get("text", None)
                        if text_value is None:
                            raise ValueError(f"Found text block with None text: {block}")

                        new_turn["content"].append({
                            "type": "text",
                            "text": text_value
                        })

                    else:
                        raise ValueError(f"Unsupported content type: {block_type}")

                cleaned_messages.append(new_turn)

            batch_messages.append(cleaned_messages)

        # 可选 debug：先看第一条样本是否已经被清洗成干净结构
        # print(batch_messages[0][0]["content"][:2])
        # print(batch_messages[0][1]["content"])

        # 1) 用 chat template 生成文本
        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            for messages in batch_messages
        ]

        # 2) 用官方工具提取 vision inputs
        image_inputs, video_inputs = process_vision_info(batch_messages)

        # 3) processor 编码
        batch = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        # 4) 构造 labels
        labels = batch["input_ids"].clone()

        # mask pad token
        if self.pad_token_id is not None:
            labels[labels == self.pad_token_id] = -100

        # mask image / video special tokens
        if self.image_token_id is not None:
            labels[labels == self.image_token_id] = -100

        if self.video_token_id is not None:
            labels[labels == self.video_token_id] = -100

        batch["labels"] = labels
        return batch
    
    
data_collator = MultiImageQwenCollator(processor, model)


# =========================
# 5. 训练参数
# =========================
# training_args = SFTConfig(
#     max_steps=2,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=1,
#     warmup_steps=0,
#     learning_rate=2e-4,
#     optim="adamw_8bit",
#     max_length=None,
#     output_dir=output_dir,
#     logging_steps=1,
#     report_to="none",
#     save_strategy="no",
#     dataloader_num_workers=0,
#     remove_unused_columns=False,
#     dataset_kwargs={"skip_prepare_dataset": True},
# )


training_args = SFTConfig(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_steps=0,
    learning_rate=2e-4,
    optim="adamw_8bit",
    max_length=None,
    output_dir=output_dir,
    logging_steps=1,
    report_to="tensorboard",
    save_strategy="epoch",
    save_total_limit=2,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
    ddp_find_unused_parameters=False,
    bf16=True,
    gradient_checkpointing=True,
)

# =========================
# 6. Trainer
# =========================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
    data_collator=data_collator,
    # processing_class=processor,   # 新版 TRL 参数名
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

trainer.save_model(output_dir)


# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 /workspace/train_psi_multimage_sft.py