import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# =========================
# 1. 配置
# =========================
model_name = "nvidia/Cosmos-Reason2-8B"
video_path = "/workspace/PSI_change/video_llama2_bbox/video_0001_track_0_00155-00170.mp4"

# 改成你想测试的 prompt
user_text = (
    "You are given a short driving video. "
    "The TARGET pedestrian is highlighted by the green box. "
    "Describe the pedestrian's likely crossing intention."
)

# =========================
# 2. 加载 processor 和 model
# =========================
print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_name)

print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
)

print("Processor type:", type(processor))
print("Model loaded.")

# 打印一下配置里有没有 video token
print("image_token_id:", getattr(model.config, "image_token_id", None))
print("video_token_id:", getattr(model.config, "video_token_id", None))

# =========================
# 3. 构造 video-block messages
# =========================
messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are an expert assistant for pedestrian-intention reasoning in driving scenes. "
                    "Focus on the target pedestrian and answer concisely."
                ),
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
            },
            {
                "type": "text",
                "text": user_text,
            },
        ],
    },
]

print("\nMessages built successfully.")

# =========================
# 4. 测试 chat template
# =========================
print("\nApplying chat template...")
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("Chat template text preview:")
print("=" * 80)
print(text[:1000])
print("=" * 80)

# =========================
# 5. 测试 vision/video parsing
# =========================
print("\nProcessing vision info...")
image_inputs, video_inputs, video_kwargs = process_vision_info(
    messages,
    image_patch_size=16,
    return_video_kwargs=True,
    return_video_metadata=True,
)

video_metadatas = None
if video_inputs is not None:
    # Qwen3-VL 返回的视频输入通常是 [(video_tensor, metadata), ...]
    if len(video_inputs) > 0 and isinstance(video_inputs[0], tuple):
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs = list(video_inputs)
        video_metadatas = list(video_metadatas)

# print("image_inputs is None:", image_inputs is None)
# if image_inputs is not None:
#     print("num image inputs:", len(image_inputs))

# print("video_inputs is None:", video_inputs is None)
# if video_inputs is not None:
#     print("num video inputs:", len(video_inputs))

# =========================
# 6. 测试 processor 编码
# =========================
print("\nEncoding with processor...")

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    video_metadata=video_metadatas,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)

print("Encoded keys:", list(inputs.keys()))
for k, v in inputs.items():
    if hasattr(v, "shape"):
        print(f"{k}: {tuple(v.shape)}")
    else:
        print(f"{k}: {type(v)}")

# =========================
# 7. 放到模型设备上，做一次最小前向 / 生成
# =========================
device = next(model.parameters()).device
print("\nModel device:", device)

inputs = {
    k: v.to(device) if hasattr(v, "to") else v
    for k, v in inputs.items()
}

print("\nRunning generation...")
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
    )

decoded = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print("\nDecoded output:")
print("=" * 80)
for i, out in enumerate(decoded):
    print(f"[Output {i}]")
    print(out)
    print("-" * 80)

print("\nTest finished successfully.")