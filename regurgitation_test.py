#!/usr/bin/env python3
"""
regurgitation_test.py — Verify the Toulmin-SFT model reproduces training-set
predictions (greedy decoding, do_sample=False).

Separates "model didn't learn" from "test mp4 distribution shift" by feeding
training records as inference input and checking if the model agrees with the
ground-truth training labels.

Usage:
  CUDA_VISIBLE_DEVICES=1 python regurgitation_test.py \
      --config /workspace/configs/infer_toulmin_sft.yaml

Output:
  /workspace/PSI_change/json_mode_90/predictions/regurgitation_test_toulmin_sft.jsonl
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

try:
    from transformers import Qwen3VLForConditionalGeneration as _VLModelCls
except ImportError:
    from transformers import AutoModelForVision2Seq as _VLModelCls

from transformers import AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

N_RECORDS   = 30
OUTPUT_PATH = "/workspace/PSI_change/json_mode_90/predictions/regurgitation_test_toulmin_sft.jsonl"
TRAIN_PATH  = "/workspace/PSI_change/json_mode_90/trf_train/psi_sft_toulmin_matched_train.jsonl"
ANSWER_RE   = re.compile(r"(?i)answer\s*[:>\s]+\s*(yes|no)\b")

# ── Video cache ────────────────────────────────────────────────────────────────
_CACHE_DIR = Path("/tmp/video_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def load_video_cached(path: str, n_frames: int, frame_size: int) -> np.ndarray:
    key   = hashlib.md5(path.encode()).hexdigest()
    cache = _CACHE_DIR / f"{key}.npy"
    if cache.exists():
        return np.load(str(cache))
    import decord
    vr      = decord.VideoReader(path, num_threads=2)
    indices = np.linspace(0, len(vr) - 1, n_frames, dtype=int)
    frames  = vr.get_batch(indices.tolist()).asnumpy()
    if frame_size != frames.shape[1] or frame_size != frames.shape[2]:
        frames = np.stack([
            np.array(Image.fromarray(frames[i]).resize(
                (frame_size, frame_size), Image.BILINEAR))
            for i in range(len(frames))
        ])
    np.save(str(cache), frames)
    return frames


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_and_processor(cfg: dict):
    model_id     = cfg["model_id"]
    adapter_path = cfg["adapter_path"]

    print(f"Loading processor: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    print(f"Loading base model (NF4) ...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = _VLModelCls.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    base.config.use_cache = True

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()
    print("Model ready.\n")
    return model, processor


def build_inputs(rec: dict, processor, cfg: dict):
    n_frames   = cfg.get("num_video_frames", 90)
    frame_size = cfg.get("frame_size", 112)
    max_pixels = cfg.get("max_pixels_per_frame", None)

    frames     = load_video_cached(rec["video"], n_frames, frame_size)
    pil_frames = [Image.fromarray(frames[i]) for i in range(len(frames))]

    video_content: dict = {"type": "video", "video": pil_frames}
    if max_pixels is not None:
        video_content["min_pixels"] = 4 * 28 * 28
        video_content["max_pixels"] = max_pixels

    messages = [
        {"role": "system",  "content": rec["system"]},
        {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": rec["prompt"][0]["content"]},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    try:
        raw = process_vision_info(
            messages, image_patch_size=14,
            return_video_kwargs=True, return_video_metadata=True,
        )
        image_inputs, video_inputs, video_kwargs = raw
    except (TypeError, ValueError):
        raw = process_vision_info(messages, return_video_kwargs=True)
        if len(raw) == 3:
            image_inputs, video_inputs, video_kwargs = raw
        else:
            image_inputs, video_inputs = raw
            video_kwargs = {}

    video_metadatas = None
    if video_inputs and isinstance(video_inputs[0], tuple):
        frames_list, meta_list = zip(*video_inputs)
        video_inputs    = list(frames_list)
        video_metadatas = list(meta_list)

    proc_kwargs = dict(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if video_metadatas is not None:
        proc_kwargs["video_metadata"] = video_metadatas
    if video_kwargs:
        proc_kwargs.update(video_kwargs)

    return processor(**proc_kwargs), text


@torch.inference_mode()
def generate_greedy(model, processor, inputs: dict, max_new_tokens: int = 256) -> str:
    device     = next(model.parameters()).device
    inputs_gpu = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    prompt_len = inputs_gpu["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs_gpu,
        do_sample=False,
        num_return_sequences=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=(
            processor.tokenizer.pad_token_id
            or processor.tokenizer.eos_token_id
        ),
    )
    new_tokens = output_ids[0][prompt_len:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)


def parse_answer(text: str):
    matches = ANSWER_RE.findall(text)
    return matches[-1].lower() if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"Config       : {args.config}")
    print(f"Adapter      : {cfg['adapter_path']}")
    print(f"Train file   : {TRAIN_PATH}")
    print(f"N records    : {N_RECORDS}")
    print(f"Output       : {OUTPUT_PATH}")
    print()

    # Pre-flight
    if not os.path.exists(os.path.join(cfg["adapter_path"], "adapter_config.json")):
        print(f"[ABORT] Adapter not found: {cfg['adapter_path']}")
        sys.exit(1)
    if not os.path.exists(TRAIN_PATH):
        print(f"[ABORT] Train JSONL not found: {TRAIN_PATH}")
        sys.exit(1)

    # Load records
    records = []
    with open(TRAIN_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if len(records) == N_RECORDS:
                break
    print(f"Loaded {len(records)} training records.\n")

    model, processor = load_model_and_processor(cfg)

    os.makedirs(Path(OUTPUT_PATH).parent, exist_ok=True)
    results   = []
    n_correct = 0

    with open(OUTPUT_PATH, "w") as out_f:
        for i, rec in enumerate(records):
            try:
                inputs, _ = build_inputs(rec, processor, cfg)
            except Exception as e:
                print(f"  [{i+1:2d}/{N_RECORDS}] SKIP build_inputs: {e}")
                continue

            generated = generate_greedy(model, processor, inputs,
                                        max_new_tokens=cfg.get("max_new_tokens", 256))
            predicted = parse_answer(generated)
            train_ans = rec.get("answer", "")
            match     = (predicted == train_ans) if predicted else False
            if match:
                n_correct += 1

            result = {
                "id":           rec.get("id", f"rec_{i}"),
                "train_answer": train_ans,
                "predicted":    predicted,
                "match":        match,
                "generated":    generated,
            }
            results.append(result)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

            flag = "✓" if match else "✗"
            print(f"  [{i+1:2d}/{N_RECORDS}] {flag}  train={train_ans}  "
                  f"pred={predicted or 'NONE':4s}  "
                  f"id={rec.get('id','?')[:55]}")

    n_done = len(results)
    acc    = n_correct / n_done if n_done else 0.0

    print(f"\n{'='*60}")
    print(f"Records evaluated : {n_done}")
    print(f"Correct           : {n_correct}")
    print(f"Accuracy          : {acc:.3f}  ({n_correct}/{n_done})")
    print(f"{'='*60}")

    if results:
        print("\nFirst record full generation:")
        print("-" * 50)
        print(results[0]["generated"])
        print("-" * 50)

    if torch.cuda.is_available():
        print(f"\nGPU peak memory: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    print(f"\nOutput: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
