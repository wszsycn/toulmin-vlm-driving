import json
from PIL import Image
from datasets import Dataset

jsonl_path = "/workspace/psi_llava_easy200.jsonl"

print("Start loading...")

samples = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        item = json.loads(line)

        print(f"Loading sample {idx}: {item.get('id')}")

        pil_images = [Image.open(img_path).convert("RGB") for img_path in item["images"]]

        sample = {
            "images": pil_images,
            "prompt": item["prompt"],
            "completion": item["completion"],
        }
        samples.append(sample)

        # if idx >= 2:   # 先只测前3条
        #     break

print("Finished loading samples.")
print("Number of samples:", len(samples))
print("Number of images in first sample:", len(samples[0]["images"]))
print("Prompt:", samples[0]["prompt"])
print("Completion:", samples[0]["completion"])

train_dataset = Dataset.from_list(samples)
# print(train_dataset)


import torch
from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration

model_name = "nvidia/Cosmos-Reason2-8B"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,  # Load the model in 4-bit precision to save memory
        bnb_4bit_compute_dtype=torch.float16,  # Data type used for internal computations in quantization
        bnb_4bit_use_double_quant=True,  # Use double quantization to improve accuracy
        bnb_4bit_quant_type="nf4",  # Type of quantization. "nf4" is recommended for recent LLMs
    ),
)

from peft import LoraConfig

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


from trl import SFTConfig

output_dir = "outputs/Cosmos-Reason2-8B-trl-sft"

# Configure training arguments using SFTConfig
training_args = SFTConfig(
    # Training schedule / optimization
    num_train_epochs=1,
    # max_steps=10,  # Number of dataset passes. For full trainings, use `num_train_epochs` instead
    per_device_train_batch_size=1,  # Batch size per GPU/CPU
    gradient_accumulation_steps=1,  # Gradients are accumulated over multiple steps → effective batch size = 4 * 8 = 32
    warmup_steps=0,  # Gradually increase LR during first N steps
    learning_rate=2e-4,  # Learning rate for the optimizer
    optim="adamw_8bit",  # Optimizer
    max_length=None,  # For VLMs, truncating may remove image tokens, leading to errors during training. max_length=None avoids it
    # Logging / reporting
    output_dir=output_dir,  # Where to save model checkpoints and logs
    logging_steps=1,  # Log training metrics every N steps
    report_to="tensorboard",  # Experiment tracking tool
)


from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
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
print(
    f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


trainer.save_model(output_dir)
