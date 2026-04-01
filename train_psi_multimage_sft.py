import os
import csv
import json
from pathlib import Path
from PIL import Image

import torch
from datasets import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    TrainerCallback,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from qwen_vl_utils import process_vision_info


# =========================================================
# 0. DDP / runtime setup
# =========================================================
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
is_main_process = rank == 0

torch.cuda.set_device(local_rank)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =========================================================
# 1. Config
# =========================================================
jsonl_path = "/workspace/psi_llava_easy200.jsonl"
model_name = "nvidia/Cosmos-Reason2-8B"
output_dir = "/workspace/outputs/Cosmos-Reason2-8B-psi-multimage-sft"

max_samples = 200
num_epochs = 1

# 如果想提速，可以改成 8 图或 4 图
# 例如 8 图: selected_frame_indices = [0, 2, 4, 6, 9, 11, 13, 15]
# 例如 4 图: selected_frame_indices = [0, 5, 10, 15]
selected_frame_indices = None   # None 表示保留全部 16 张图

preview_dir = f"{output_dir}/previews"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(preview_dir, exist_ok=True)


# =========================================================
# 2. Build dataset: system + user + assistant
# =========================================================
def build_messages_from_jsonl(jsonl_path, max_samples=None, selected_frame_indices=None):
    system_prompt = (
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

    user_prompt = (
        "These are sequential driving frames.\n"
        "You are given a sequence of driving frames.\n"
        "The TARGET pedestrian is the one highlighted by the green box (ignore others).\n"
        "Task: Predict whether the TARGET pedestrian has crossing intention at t+1."
    )

    samples = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break

            item = json.loads(line)

            image_paths = item["images"]
            if selected_frame_indices is not None:
                image_paths = [image_paths[i] for i in selected_frame_indices]

            assistant_text = item["completion"][0]["content"]

            user_content = [{"type": "image", "image": p} for p in image_paths]
            user_content.append({"type": "text", "text": user_prompt})

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                },
            ]

            samples.append({"messages": messages})

    return Dataset.from_list(samples)


train_dataset = build_messages_from_jsonl(
    jsonl_path=jsonl_path,
    max_samples=max_samples,
    selected_frame_indices=selected_frame_indices,
)

if is_main_process:
    print(train_dataset)


# =========================================================
# 3. Example inspection helpers
# =========================================================
def print_example(dataset, idx=0, num_image_paths=4):
    ex = dataset[idx]["messages"]

    system_turn = ex[0]
    user_turn = ex[1]
    assistant_turn = ex[2]

    image_blocks = [b for b in user_turn["content"] if b["type"] == "image"]
    text_blocks = [b for b in user_turn["content"] if b["type"] == "text"]

    print("=" * 80)
    print(f"Example index: {idx}")

    print("\n[System prompt]")
    print(system_turn["content"][0]["text"])

    print("\nNumber of image blocks:", len(image_blocks))
    print("First few image paths:")
    for i, b in enumerate(image_blocks[:num_image_paths]):
        print(f"  [{i}] {b['image']}")

    print("\n[User text]")
    print(text_blocks[0]["text"])

    print("\n[Assistant text]")
    print(assistant_turn["content"][0]["text"])
    print("=" * 80)


def save_example_contact_sheet(dataset, idx=0, out_path="/workspace/example_preview.jpg", thumb_size=(224, 224)):
    ex = dataset[idx]["messages"]
    user_turn = ex[1]
    image_blocks = [b for b in user_turn["content"] if b["type"] == "image"]

    images = [Image.open(b["image"]).convert("RGB").resize(thumb_size) for b in image_blocks]

    cols = 4
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_size[0], rows * thumb_size[1]), color="white")

    for i, img in enumerate(images):
        x = (i % cols) * thumb_size[0]
        y = (i // cols) * thumb_size[1]
        sheet.paste(img, (x, y))

    sheet.save(out_path)
    print(f"Saved preview to: {out_path}")


if is_main_process:
    print_example(train_dataset, idx=0)
    print_example(train_dataset, idx=1)
    save_example_contact_sheet(train_dataset, idx=0, out_path=f"{preview_dir}/example_0.jpg")
    save_example_contact_sheet(train_dataset, idx=1, out_path=f"{preview_dir}/example_1.jpg")


# =========================================================
# 4. Load processor and model
# =========================================================
processor = AutoProcessor.from_pretrained(model_name)

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


# =========================================================
# 5. Collator
# =========================================================
class MultiImageQwenCollator:
    def __init__(self, processor, model, debug_once=True):
        self.processor = processor
        self.model = model
        self.debug_once = debug_once
        self._has_printed_debug = False

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

        if self.debug_once and (not self._has_printed_debug) and is_main_process:
            system_turn = batch_messages[0][0]
            user_turn = batch_messages[0][1]
            assistant_turn = batch_messages[0][2]
            print("\n[Collator Debug]")
            print("Batch size:", len(features))
            print("Num image blocks in first sample:", sum(1 for x in user_turn["content"] if x["type"] == "image"))
            print("System prompt:")
            print(system_turn["content"][0]["text"])
            print("User text:")
            print([x["text"] for x in user_turn["content"] if x["type"] == "text"][0])
            print("Assistant text:")
            print(assistant_turn["content"][0]["text"])
            self._has_printed_debug = True

        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            for messages in batch_messages
        ]

        image_inputs, video_inputs = process_vision_info(batch_messages)

        batch = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        labels = batch["input_ids"].clone()

        if self.pad_token_id is not None:
            labels[labels == self.pad_token_id] = -100
        if self.image_token_id is not None:
            labels[labels == self.image_token_id] = -100
        if self.video_token_id is not None:
            labels[labels == self.video_token_id] = -100

        batch["labels"] = labels
        return batch


data_collator = MultiImageQwenCollator(processor, model, debug_once=True)


# =========================================================
# 6. Callback: save per-step loss
# =========================================================
class LossCSVLoggerCallback(TrainerCallback):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        if is_main_process:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "epoch", "loss", "learning_rate", "grad_norm"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not is_main_process:
            return
        if logs is None:
            return

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                state.global_step,
                logs.get("epoch"),
                logs.get("loss"),
                logs.get("learning_rate"),
                logs.get("grad_norm"),
            ])


loss_csv_path = f"{output_dir}/step_losses.csv"
loss_callback = LossCSVLoggerCallback(loss_csv_path)


# =========================================================
# 7. Training args
# =========================================================
training_args = SFTConfig(
    num_train_epochs=num_epochs,
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


# =========================================================
# 8. Trainer
# =========================================================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
    data_collator=data_collator,
    callbacks=[loss_callback],
)


# =========================================================
# 9. Train
# =========================================================
gpu_stats = torch.cuda.get_device_properties(local_rank)
start_gpu_memory = round(torch.cuda.max_memory_reserved(local_rank) / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

if is_main_process:
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved(local_rank) / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

if is_main_process:
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    with open(f"{output_dir}/log_history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)

trainer.save_model(output_dir)



# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 /workspace/train_psi_multimage_sft.py