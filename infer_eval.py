#!/usr/bin/env python3
"""
infer_eval.py — Inference + evaluation for fine-tuned Cosmos-Reason2-8B on
PSI pedestrian crossing intent (psi_test_eval_v2.jsonl).

CLI:
  python infer_eval.py --config configs/infer_toulmin_sft.yaml
  python infer_eval.py --config configs/infer_toulmin_sft.yaml --smoke
  python infer_eval.py --config configs/infer_cot_sft.yaml --max-records 50
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import hashlib
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

# ── Answer parser ──────────────────────────────────────────────────────────────
ANSWER_RE = re.compile(r"(?i)answer\s*[:>\s]+\s*(yes|no)\b")

# ── Video cache (same as train_sft.py) ────────────────────────────────────────
_VIDEO_CACHE_DIR = Path("/tmp/video_cache")
_VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _video_cache_path(video_path: str) -> Path:
    md5 = hashlib.md5(video_path.encode()).hexdigest()
    return _VIDEO_CACHE_DIR / f"{md5}.npy"


def load_video_cached(path: str, n_frames: int = 90, frame_size: int = 112) -> np.ndarray:
    cache = _video_cache_path(path)
    if cache.exists():
        return np.load(str(cache))
    import decord
    vr = decord.VideoReader(path, num_threads=2)
    total = len(vr)
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = vr.get_batch(indices.tolist()).asnumpy()
    if frame_size != frames.shape[1] or frame_size != frames.shape[2]:
        frames = np.stack([
            np.array(Image.fromarray(frames[i]).resize(
                (frame_size, frame_size), Image.BILINEAR
            ))
            for i in range(len(frames))
        ])
    np.save(str(cache), frames)
    return frames


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_and_processor(cfg: dict):
    model_id     = cfg["model_id"]
    adapter_path = cfg["adapter_path"]

    print(f"Loading processor from {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    print(f"Loading base model with 4-bit NF4 ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = _VLModelCls.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    # Inference: use KV cache, no gradient checkpointing
    base.config.use_cache = True

    print(f"Loading LoRA adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()
    print("Model ready.")
    return model, processor


# ── Build vision inputs for one record ────────────────────────────────────────

def build_inputs(record: dict, processor, cfg: dict) -> dict:
    n_frames   = cfg.get("num_video_frames", 90)
    frame_size = cfg.get("frame_size", 112)
    max_pixels = cfg.get("max_pixels_per_frame", None)

    frames     = load_video_cached(record["video"], n_frames, frame_size)
    pil_frames = [Image.fromarray(frames[i]) for i in range(len(frames))]

    video_content: dict = {"type": "video", "video": pil_frames}
    if max_pixels is not None:
        video_content["min_pixels"] = 4 * 28 * 28   # 3136
        video_content["max_pixels"] = max_pixels

    messages = [
        {"role": "system",  "content": record["system"]},
        {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": record["prompt"][0]["content"]},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # process_vision_info — same compat shim as train_sft.py
    try:
        raw = process_vision_info(
            messages,
            image_patch_size=14,
            return_video_kwargs=True,
            return_video_metadata=True,
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

    proc_kwargs: dict = dict(
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

    inputs = processor(**proc_kwargs)
    return inputs, text


# ── Generate N samples for one record ─────────────────────────────────────────

@torch.inference_mode()
def generate_samples(model, processor, inputs: dict, cfg: dict) -> list[str]:
    n_samples     = cfg.get("n_samples", 5)
    temperature   = cfg.get("temperature", 0.7)
    top_p         = cfg.get("top_p", 0.95)
    max_new_tokens = cfg.get("max_new_tokens", 256)
    pad_id = (
        processor.tokenizer.pad_token_id
        or processor.tokenizer.eos_token_id
    )

    device = next(model.parameters()).device
    inputs_gpu = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    prompt_len = inputs_gpu["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs_gpu,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=n_samples,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_id,
    )

    # Slice off the prompt portion and decode each sample
    samples = []
    for seq in output_ids:
        new_tokens = seq[prompt_len:]
        text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        samples.append(text)
    return samples


# ── Parse answer from samples ──────────────────────────────────────────────────

def parse_samples(samples: list[str], n_samples_total: int) -> dict:
    n_yes = n_no = n_fail = 0
    for s in samples:
        m = ANSWER_RE.search(s)
        if m:
            if m.group(1).lower() == "yes":
                n_yes += 1
            else:
                n_no += 1
        else:
            n_fail += 1

    n_valid = n_yes + n_no
    if n_valid == 0:
        predicted_prob = 0.5
        parse_status   = "all_failed"
    else:
        predicted_prob = n_yes / n_valid
        parse_status   = f"{n_valid}/{n_samples_total}_parsed"

    predicted_hard = "yes" if predicted_prob > 0.5 else "no"
    return {
        "predicted_prob": round(predicted_prob, 4),
        "predicted_hard": predicted_hard,
        "n_samples_total": n_samples_total,
        "n_samples_parsed": n_valid,
        "parse_status": parse_status,
    }


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    from sklearn.metrics import roc_auc_score

    probs      = [r["predicted_prob"] for r in results]
    soft_true  = [r["answer_soft"] for r in results]
    hard_pred  = [r["predicted_hard"] for r in results]
    hard_true  = [r["answer_hard"] for r in results]
    n_parsed   = [r["n_samples_parsed"] for r in results]
    n_total    = [r["n_samples_total"] for r in results]

    brier = float(np.mean([(p - t) ** 2 for p, t in zip(probs, soft_true)]))
    mae   = float(np.mean([abs(p - t) for p, t in zip(probs, soft_true)]))

    hard_acc = float(np.mean([p == t for p, t in zip(hard_pred, hard_true)]))

    parsed_mask = [n > 0 for n in n_parsed]
    hard_acc_parsed = float(np.mean([
        p == t for p, t, m in zip(hard_pred, hard_true, parsed_mask) if m
    ])) if any(parsed_mask) else float("nan")

    y_true = [1 if h == "yes" else 0 for h in hard_true]
    try:
        roc_auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        roc_auc = float("nan")

    parse_success_rate = float(np.mean(parsed_mask))
    total_parse_rate   = (
        sum(n_parsed) / sum(n_total) if sum(n_total) > 0 else float("nan")
    )

    n_yes_pred = sum(1 for p in hard_pred if p == "yes")
    n_no_pred  = len(hard_pred) - n_yes_pred
    n_yes_true = sum(1 for t in hard_true if t == "yes")
    n_no_true  = len(hard_true) - n_yes_true

    return {
        "n_records": len(results),
        "brier_score": round(brier, 4),
        "mae": round(mae, 4),
        "hard_accuracy": round(hard_acc, 4),
        "hard_accuracy_parsed_only": round(hard_acc_parsed, 4) if not np.isnan(hard_acc_parsed) else "nan",
        "roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else "nan",
        "parse_success_rate": round(parse_success_rate, 4),
        "total_parse_rate": round(total_parse_rate, 4),
        "pred_yes": n_yes_pred,
        "pred_no": n_no_pred,
        "true_yes": n_yes_true,
        "true_no": n_no_true,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inference + eval on psi_test_eval_v2.jsonl"
    )
    parser.add_argument("--config",      required=True, help="Path to infer YAML config")
    parser.add_argument("--smoke",       action="store_true", help="Run on first 5 records only")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap number of records to process")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Seed ──────────────────────────────────────────────────────────────────
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Paths ─────────────────────────────────────────────────────────────────
    eval_path    = cfg["eval_path"]
    output_path  = cfg["output_path"]
    metrics_path = cfg["metrics_path"]

    if args.smoke:
        stem = Path(output_path).stem
        smoke_path = Path(output_path).with_name(stem + "_smoke.jsonl")
        smoke_metrics_path = Path(metrics_path).with_name(
            Path(metrics_path).stem + "_smoke.json"
        )
        output_path  = str(smoke_path)
        metrics_path = str(smoke_metrics_path)

    os.makedirs(Path(output_path).parent, exist_ok=True)

    # ── Load eval records ─────────────────────────────────────────────────────
    records = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n_total = len(records)
    limit = 5 if args.smoke else args.max_records
    if limit is not None:
        records = records[:limit]

    print(f"\nConfig      : {args.config}")
    print(f"Adapter     : {cfg['adapter_path']}")
    print(f"Format      : {cfg.get('format', 'toulmin')}")
    print(f"Eval path   : {eval_path}  ({n_total} records total)")
    print(f"Processing  : {len(records)} records")
    print(f"n_samples   : {cfg.get('n_samples', 5)}")
    print(f"Output      : {output_path}")
    print(f"Smoke mode  : {args.smoke}")
    print()

    # ── Pre-flight ────────────────────────────────────────────────────────────
    adapter_path = cfg["adapter_path"]
    adapter_cfg  = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_cfg):
        print(f"[ABORT] Adapter not found: {adapter_cfg}")
        sys.exit(1)
    if not os.path.exists(eval_path):
        print(f"[ABORT] Eval JSONL not found: {eval_path}")
        sys.exit(1)
    print("Pre-flight OK.")

    # ── Load model ────────────────────────────────────────────────────────────
    model, processor = load_model_and_processor(cfg)

    # ── Inference loop ────────────────────────────────────────────────────────
    results   = []
    n_samples = cfg.get("n_samples", 5)

    # Resume: skip already-done ids
    done_ids: set = set()
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["id"])
        print(f"Resuming: {len(done_ids)} records already done.")

    with open(output_path, "a", encoding="utf-8") as out_f:
        for i, rec in enumerate(records):
            if rec["id"] in done_ids:
                continue

            # ── Build inputs ────────────────────────────────────────────────
            try:
                inputs, rendered_prompt = build_inputs(rec, processor, cfg)
            except Exception as e:
                print(f"  [{i+1}/{len(records)}] SKIP (build_inputs error): {e}")
                continue

            # ── Generate ────────────────────────────────────────────────────
            try:
                samples = generate_samples(model, processor, inputs, cfg)
            except Exception as e:
                print(f"  [{i+1}/{len(records)}] SKIP (generate error): {e}")
                continue

            # ── Parse ────────────────────────────────────────────────────────
            parsed = parse_samples(samples, n_samples)

            result = {
                "id":              rec["id"],
                "video":           rec["video"],
                "answer_soft":     rec["aggregated_intent"],
                "answer_hard":     rec["answer"],
                "predicted_prob":  parsed["predicted_prob"],
                "predicted_hard":  parsed["predicted_hard"],
                "n_samples_total": parsed["n_samples_total"],
                "n_samples_parsed": parsed["n_samples_parsed"],
                "parse_status":    parsed["parse_status"],
                "raw_samples":     [s[:600] for s in samples],
                "meta":            rec.get("meta", {}),
            }
            results.append(result)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

            # ── Progress ─────────────────────────────────────────────────────
            status_str = (
                f"  [{i+1:4d}/{len(records)}]  "
                f"id={rec['id'][:40]}  "
                f"pred={parsed['predicted_hard']}({parsed['predicted_prob']:.2f})  "
                f"true={rec['answer']}({rec['aggregated_intent']:.2f})  "
                f"parse={parsed['parse_status']}"
            )
            print(status_str)

            # ── Smoke: verbose first record ──────────────────────────────────
            if args.smoke and i == 0:
                print(f"\n[smoke] Rendered prompt (first 500 chars):")
                print(f"  {rendered_prompt[:500]}")
                print(f"\n[smoke] First generated sample (first 500 chars):")
                print(f"  {samples[0][:500]}")
                print(f"[smoke] Parse result: "
                      f"n_yes={sum(1 for s in samples if ANSWER_RE.search(s) and ANSWER_RE.search(s).group(1).lower()=='yes')}  "
                      f"n_no={sum(1 for s in samples if ANSWER_RE.search(s) and ANSWER_RE.search(s).group(1).lower()=='no')}  "
                      f"n_fail={sum(1 for s in samples if not ANSWER_RE.search(s))}")
                print()

    # ── Load all results (including previously done) ──────────────────────────
    all_results = []
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_results.append(json.loads(line))

    if not all_results:
        print("No results to compute metrics on.")
        return

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(all_results)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results    : {output_path}")
    print(f"Metrics    : {metrics_path}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:<35} {v}")
    print(f"{'='*60}")

    # ── GPU memory ───────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n  GPU peak memory: {peak_gb:.1f} GB")

    # ── Smoke validation ──────────────────────────────────────────────────────
    if args.smoke:
        print("\n[smoke] Validation checks:")
        any_parsed = any(r["n_samples_parsed"] > 0 for r in all_results)
        has_toulmin = any(
            "grounds:" in s
            for r in all_results
            for s in r["raw_samples"]
        )
        peak_ok = (
            torch.cuda.max_memory_allocated() / 1e9 < 30.0
            if torch.cuda.is_available() else True
        )

        checks = [
            ("At least 1 record parsed",            any_parsed),
            ("Toulmin format detected (grounds:)",  has_toulmin),
            ("GPU peak < 30 GB",                    peak_ok),
        ]
        all_ok = True
        for name, ok in checks:
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {name}")
            if not ok:
                all_ok = False

        if not all_ok:
            print("\n[smoke] One or more checks FAILED. See above.")
            sys.exit(1)
        else:
            print("\n[smoke] All checks passed.")

    # ── tmux launch commands ───────────────────────────────────────────────────
    if args.smoke:
        print("""
Full run tmux commands (run SEQUENTIALLY on a single GPU):

  mkdir -p /workspace/logs

  tmux new-session -d -s infer_toulmin_sft \\
    "cd /workspace && CUDA_VISIBLE_DEVICES=1 python infer_eval.py \\
        --config configs/infer_toulmin_sft.yaml \\
        2>&1 | tee /workspace/logs/infer_toulmin_sft.log"

  tmux new-session -d -s infer_cot_sft \\
    "cd /workspace && CUDA_VISIBLE_DEVICES=1 python infer_eval.py \\
        --config configs/infer_cot_sft.yaml \\
        2>&1 | tee /workspace/logs/infer_cot_sft.log"

  # toulmin_dpo: launch only after DPO training completes
  # (verify /workspace/checkpoints/toulmin_dpo/adapter_config.json exists first)
  tmux new-session -d -s infer_toulmin_dpo \\
    "cd /workspace && CUDA_VISIBLE_DEVICES=1 python infer_eval.py \\
        --config configs/infer_toulmin_dpo.yaml \\
        2>&1 | tee /workspace/logs/infer_toulmin_dpo.log"

Note: all three target GPU 1 — run them sequentially, not in parallel.
If DPO is done and you want parallel, move one run to CUDA_VISIBLE_DEVICES=0.
""")


if __name__ == "__main__":
    main()
