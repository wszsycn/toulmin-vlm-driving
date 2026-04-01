import os
import re
import csv
import json
from pathlib import Path

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
video_root = "/workspace/PSI_change/video_llama2_bbox"
model_name = "nvidia/Cosmos-Reason2-8B"
output_dir = "/workspace/outputs/Cosmos-Reason2-8B-psi-video-sft"

max_samples = 200
num_epochs = 1

os.makedirs(output_dir, exist_ok=True)


# =========================================================
# 2. Helper: derive video path from id
# =========================================================
def recover_video_path_from_id(sample_id, video_root):
    """
    Example:
    sample_id:
      video_0001_track_0_nlp_vid_12_uid_5226_v2_00155-00170_video

    target video:
      /workspace/PSI_change/video_llama2_bbox/video_0001_track_0_00155-00170.mp4
    """
    pattern = r"^(video_\d+_track_\d+)_.*?_(\d{5}-\d{5})_video$"
    m = re.match(pattern, sample_id)
    if not m:
        raise ValueError(f"Cannot recover video path from sample id: {sample_id}")

    prefix = m.group(1)       # video_0001_track_0
    frame_span = m.group(2)   # 00155-00170

    filename = f"{prefix}_{frame_span}.mp4"
    video_path = os.path.join(video_root, filename)
    return video_path


# =========================================================
# 3. Build dataset: system + user(video) + assistant
# =========================================================
def build_video_messages_from_jsonl(jsonl_path, video_root, max_samples=None):
    system_prompt = (
        "You are an expert assistant for pedestrian-intention reasoning from driving videos.\n"
        "Focus only on the TARGET pedestrian highlighted by the green box.\n"
        "Use concise and factual language.\n"
        "Grounds must describe visible evidence from the video.\n"
        "Warrant must provide a general rule linking the observed evidence to the pedestrian’s likely intention.\n"
        "Output EXACTLY 3 lines in this order:\n"
        "grounds: <1–2 sentences, concrete observations>\n"
        "warrant: <1 sentence general rule linking grounds to intention>\n"
        "answer: <yes/no>\n"
        "Do not add any extra text before or after these 3 lines."
    )

    user_prompt = (
        "You are given a short driving video.\n"
        "The TARGET pedestrian is highlighted by the green box.\n"
        "Reason about the pedestrian's motion and change over time across the full video.\n"
        "Task: Predict whether the TARGET pedestrian has crossing intention at t+1."
    )

    samples = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break

            item = json.loads(line)

            sample_id = item.get("id")
            assistant_text = item["completion"][0]["content"]

            # 如果 jsonl 里本身有 video 字段，就优先用
            if "video" in item and item["video"] is not None:
                video_path = item["video"]
            else:
                video_path = recover_video_path_from_id(sample_id, video_root)

            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video not found: {video_path}")

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": user_prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": assistant_text}
                    ],
                },
            ]

            samples.append({"messages": messages})

    return Dataset.from_list(samples)


train_dataset = build_video_messages_from_jsonl(
    jsonl_path=jsonl_path,
    video_root=video_root,
    max_samples=max_samples,
)

if is_main_process:
    print(train_dataset)


# =========================================================
# 4. Example inspection
# =========================================================
def print_example(dataset, idx=0):
    ex = dataset[idx]["messages"]

    system_turn = ex[0]
    user_turn = ex[1]
    assistant_turn = ex[2]

    video_blocks = [b for b in user_turn["content"] if b["type"] == "video"]
    text_blocks = [b for b in user_turn["content"] if b["type"] == "text"]

    print("=" * 80)
    print(f"Example index: {idx}")

    print("\n[System prompt]")
    print(system_turn["content"][0]["text"])

    print("\n[Video path]")
    print(video_blocks[0]["video"])

    print("\n[User text]")
    print(text_blocks[0]["text"])

    print("\n[Assistant text]")
    print(assistant_turn["content"][0]["text"])
    print("=" * 80)


if is_main_process:
    print_example(train_dataset, idx=0)
    print_example(train_dataset, idx=1)


# =========================================================
# 5. Load processor and model
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
# 6. Video collator
# =========================================================
class VideoQwenCollator:
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

                    if block_type == "video":
                        video_path = block.get("video", None)
                        if video_path is None:
                            raise ValueError(f"Found video block with None video path: {block}")

                        new_turn["content"].append({
                            "type": "video",
                            "video": video_path
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
            print("\n[Collator Debug]")
            print("Batch size:", len(features))
            print("First sample system:")
            print(batch_messages[0][0]["content"][0]["text"])
            print("First sample video path:")
            print(batch_messages[0][1]["content"][0]["video"])
            print("First sample user text:")
            print(batch_messages[0][1]["content"][1]["text"])
            print("First sample assistant text:")
            print(batch_messages[0][2]["content"][0]["text"])
            self._has_printed_debug = True

        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            for messages in batch_messages
        ]

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            batch_messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        video_metadatas = None
        if video_inputs is not None:
            if len(video_inputs) > 0 and isinstance(video_inputs[0], tuple):
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs = list(video_inputs)
                video_metadatas = list(video_metadatas)

        batch = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
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


data_collator = VideoQwenCollator(processor, model, debug_once=True)


# =========================================================
# 7. Callback: save per-step loss
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
# 8. Training args
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
    dataloader_num_workers=2,
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
    ddp_find_unused_parameters=False,
    bf16=True,
    gradient_checkpointing=True,
)


# =========================================================
# 9. Trainer
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
# 10. Train
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
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 /workspace/train_psi_video_sft.py