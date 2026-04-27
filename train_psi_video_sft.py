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

# os.environ["DECORD_NUM_THREADS"] = "1"
# os.environ["NCCL_DEBUG"] = "INFO"  # 看NCCL通信有没有问题
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
jsonl_path = "/workspace/PSI_change/json_mode_90/trf_train/psi_sft_llava.jsonl"
video_root = "/workspace/PSI_change/json_mode_90/videos"
model_name = "nvidia/Cosmos-Reason2-8B"
output_dir = "/workspace/outputs/Cosmos-Reason2-8B-psi-video-90f-sft"

max_samples = 500
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

    user_prompt = (
        "Watch the full video and predict: will the TARGET pedestrian "
        "attempt to cross in front of the vehicle in the next moment?"
    )

    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
            item = json.loads(line)
            sample_id = item.get("id")
            assistant_text = item["completion"][0]["content"]

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
    # device_map={"": local_rank},
    # device_map="auto",  # 自动分配到两个GPU，不用DDP
    device_map={"": 0},  # 保持0不变，CUDA_VISIBLE_DEVICES会自动映射

    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
)

# model.gradient_checkpointing_enable(
#     gradient_checkpointing_kwargs={"use_reentrant": False}
# )

peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,  # 加一点dropout防止过拟合
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
        print(f"[Collator] START batch={len(features)}", flush=True)

        batch_messages = []
        for feature in features:
            messages = feature["messages"]
            cleaned_messages = []
            for turn in messages:
                new_turn = {"role": turn["role"], "content": []}
                for block in turn["content"]:
                    block_type = block["type"]
                    if block_type == "video":
                        video_path = block.get("video")
                        if video_path is None:
                            raise ValueError(f"Video block has None path: {block}")
                        new_turn["content"].append({
                            "type":       "video",
                            "video":      video_path,
                            "nframes":    16,
                            "max_pixels": 360 * 420,
                        })
                    elif block_type == "text":
                        text_value = block.get("text")
                        if text_value is None:
                            raise ValueError(f"Text block has None text: {block}")
                        new_turn["content"].append({"type": "text", "text": text_value})
                    else:
                        raise ValueError(f"Unsupported content type: {block_type}")
                cleaned_messages.append(new_turn)
            batch_messages.append(cleaned_messages)

        if self.debug_once and not self._has_printed_debug and is_main_process:
            print("\n[Collator Debug]")
            print("Batch size:", len(features))
            print("Video path:", batch_messages[0][1]["content"][0]["video"])
            print("Assistant:", batch_messages[0][2]["content"][0]["text"])
            self._has_printed_debug = True

        print("[Collator] apply_chat_template...", flush=True)
        texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in batch_messages
        ]

        print("[Collator] process_vision_info...", flush=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            batch_messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        print("[Collator] unpacking metadata...", flush=True)
        video_metadatas = None
        if video_inputs is not None and len(video_inputs) > 0:
            if isinstance(video_inputs[0], tuple):
                frames_list, meta_list = zip(*video_inputs)
                video_inputs    = list(frames_list)
                video_metadatas = list(meta_list)

        print("[Collator] processor()...", flush=True)
        batch = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

        print("[Collator] masking labels...", flush=True)
        labels = batch["input_ids"].clone()
        if self.pad_token_id is not None:
            labels[labels == self.pad_token_id] = -100
        if self.image_token_id is not None:
            labels[labels == self.image_token_id] = -100
        if self.video_token_id is not None:
            labels[labels == self.video_token_id] = -100
        batch["labels"] = labels

        print("[Collator] DONE", flush=True)
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
# training_args = SFTConfig(
#     num_train_epochs=num_epochs,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=1,
#     warmup_steps=20,
#     learning_rate=1e-4,
#     lr_scheduler_type="cosine",
#     weight_decay=0.01,
#     max_grad_norm=1.0,
#     optim="adamw_8bit",
#     max_length=None,
#     output_dir=output_dir,
#     logging_steps=1,
#     report_to="tensorboard",
#     save_strategy="steps",
#     save_steps=50,                   # 每50步保存一次
#     save_total_limit=3,
#     dataloader_num_workers=1,
#     remove_unused_columns=False,
#     dataset_kwargs={"skip_prepare_dataset": True},
#     ddp_find_unused_parameters=False,
#     bf16=True,
#     gradient_checkpointing=False,
# )


training_args = SFTConfig(
    num_train_epochs=num_epochs,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # 先改成1
    warmup_steps=20,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    max_grad_norm=1.0,               # 加上梯度裁剪
    optim="adamw_8bit",
    max_length=None,
    output_dir=output_dir,
    logging_steps=1,
    report_to="tensorboard",               # 先关掉tensorboard
    save_strategy="epoch",
    save_total_limit=2,
    dataloader_num_workers=2,
    dataloader_prefetch_factor=2,
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
    bf16=True,
    gradient_checkpointing=True,   # 先关掉
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
# training_args = SFTConfig(
#     num_train_epochs=3,              # 从1增加到3，490条数据太少训1个epoch
#     per_device_train_batch_size=1,   # 保持
#     gradient_accumulation_steps=8,   # 从1增加到8，等效batch_size=16，更稳定
#     warmup_steps=20,                 # 从0增加，避免学习率突变
#     learning_rate=1e-4,              # 从2e-4降低，配合更多epoch
#     lr_scheduler_type="cosine",      # 加cosine decay
#     weight_decay=0.01,               # 加weight decay
#     max_grad_norm=1.0,               # 加梯度裁剪
#     optim="adamw_8bit",              # 保持
#     save_strategy="steps",           # 改成按步保存
#     save_steps=50,                   # 每50步保存一次
#     save_total_limit=3,
#     logging_steps=1,                 # 保持
#     ...
# )

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


# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
#   --master_port=29501 \
#   /workspace/train_psi_video_sft.py

# CUDA_VISIBLE_DEVICES=0,1 python /workspace/train_psi_video_sft.py
# CUDA_VISIBLE_DEVICES=1 python /workspace/train_psi_video_sft.py