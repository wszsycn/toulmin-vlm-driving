#!/usr/bin/env python3
"""
warrant_flip_experiment.py
==========================
Counterfactual flip-rate test for trained Toulmin models.

For each sampled test record (stratified by GT answer):
  (a) original warrant  — baseline; should reproduce the saved prediction
  (b) random warrant    — drawn from a different record in the pool
  (c) opposite-class warrant — from a record whose GT answer is the opposite class

Flip = model answer under (b)/(c) differs from model answer under (a).

DECISION RULE
  flip_rate_c > 0.30  AND  flip_rate_c > 2 * flip_rate_b  =>  SUPPORTED
  flip_rate_c < 0.20  OR   flip_rate_c <= flip_rate_b      =>  CONTRADICTED
  otherwise                                                 =>  MIXED

USAGE
  python warrant_flip_experiment.py \\
      --target-model toulmin_sft_v3 \\
      --pred-dir /workspace/PSI_change/json_mode_90/predictions \\
      --model-ckpt /workspace/checkpoints/toulmin_sft_v3 \\
      --n-samples 100 \\
      --out-json /workspace/PSI_change/json_mode_90/predictions/warrant_flip.json

NOTES
  * Random seed is fixed at 42 — do not change.
  * Temperature / do_sample settings are fixed — do not change.
  * Only data-loading / prompt-format bugs should be fixed here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── infer_eval helpers (same workspace) ──────────────────────────────────────
sys.path.insert(0, "/workspace")
from infer_eval import load_video_cached, load_model_and_processor  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402

# ── Regex ─────────────────────────────────────────────────────────────────────
_TOULMIN_RE = re.compile(
    r"\**\s*grounds\s*\**\s*[:\-]\s*(?P<grounds>.+?)"
    r"\**\s*warrant\s*\**\s*[:\-]\s*(?P<warrant>.+?)"
    r"\**\s*answer\s*\**\s*[:\-]\s*(?P<answer>yes|no)\b",
    re.IGNORECASE | re.DOTALL,
)
_ANSWER_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

# ── Model cache ───────────────────────────────────────────────────────────────
_MODEL_CACHE: dict[str, tuple] = {}

# ── Prompt constants (copied verbatim from psi_test_eval_v2.jsonl) ────────────
SYS_TOULMIN = (
    "You are an expert autonomous driving assistant specializing in pedestrian "
    "behavior analysis. When given a driving video, you analyze the TARGET "
    "pedestrian (highlighted by a green bounding box) and predict their crossing "
    "intention using the Toulmin argument structure:\n"
    "- grounds: concrete visual observations (posture, movement, position, gaze)\n"
    "- warrant: a general principle (physical law, social norm, or traffic rule) "
    "linking observations to the conclusion\n"
    "- answer: yes or no\n\n"
    "Always output exactly 3 lines in this format:\n"
    "grounds: <observations>\n"
    "warrant: <general rule>\n"
    "answer: <yes/no>\n"
    "Do not add any extra text before or after these 3 lines."
)
USR_TOULMIN = (
    "Watch the full 90-frame video and predict: will the TARGET pedestrian "
    "attempt to cross in front of the vehicle in the next moment?"
)


# ── Core inference function ───────────────────────────────────────────────────

def model_predict_answer(
    video_path: str,
    grounds: str,
    warrant: str,
    model_ckpt: str,
    *,
    n_frames: int = 90,
    frame_size: int = 112,
    max_pixels: int = 12544,
) -> str:
    """
    Forced-completion inference: feed grounds + warrant as the assistant prefix,
    then ask the model to complete with 'yes' or 'no'.

    Returns 'yes', 'no', or 'invalid'.
    Caches (model, processor) in _MODEL_CACHE keyed by model_ckpt.
    """
    global _MODEL_CACHE

    # ── Load / retrieve model ────────────────────────────────────────────────
    if model_ckpt not in _MODEL_CACHE:
        cfg = {
            "model_id": "nvidia/Cosmos-Reason2-8B",
            "adapter_path": model_ckpt,
        }
        print(f"[model_cache] Loading model from {model_ckpt} ...")
        model, processor = load_model_and_processor(cfg)
        _MODEL_CACHE[model_ckpt] = (model, processor)
    model, processor = _MODEL_CACHE[model_ckpt]

    # ── Load video ───────────────────────────────────────────────────────────
    frames = load_video_cached(video_path, n_frames, frame_size)
    pil_frames = [Image.fromarray(frames[i]) for i in range(len(frames))]

    # ── Build messages (same structure as infer_eval.py) ─────────────────────
    video_content: dict = {
        "type":       "video",
        "video":      pil_frames,
        "min_pixels": 4 * 28 * 28,
        "max_pixels": max_pixels,
    }
    messages = [
        {"role": "system", "content": SYS_TOULMIN},
        {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": USR_TOULMIN},
            ],
        },
    ]

    # ── Apply chat template → ends with "<|im_start|>assistant\n" ────────────
    base_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Append forced assistant prefix; model completes from "answer:"
    forced_prefix = f"grounds: {grounds}\nwarrant: {warrant}\nanswer:"
    full_text = base_text + forced_prefix

    # ── Process vision info (same compat shim as infer_eval.py) ──────────────
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
        text=[full_text],
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

    # ── Generate (deterministic, max 8 new tokens) ────────────────────────────
    device = next(model.parameters()).device
    inputs_gpu = {
        k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)
    }
    prompt_len = inputs_gpu["input_ids"].shape[1]
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs_gpu,
            do_sample=False,
            max_new_tokens=8,
            pad_token_id=pad_id,
        )

    new_tokens = out_ids[0][prompt_len:]
    decoded = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    m = _ANSWER_RE.search(decoded)
    if m:
        return m.group(1).lower()
    return "invalid"


# ── Prediction loader ─────────────────────────────────────────────────────────

def load_parsed_predictions(pred_dir: str, target_model: str) -> list[dict]:
    """
    Load *_predictions.jsonl for target_model, parse Toulmin, return list of
    {id, video, gt, grounds, warrant, predicted_hard}.
    Skips records that fail to parse.
    """
    path = os.path.join(pred_dir, f"{target_model}_predictions.jsonl")
    if not os.path.exists(path):
        sys.exit(f"[ABORT] Prediction file not found: {path}")

    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = ""
            rs = rec.get("raw_samples")
            if isinstance(rs, list) and rs:
                text = str(rs[0])
            if not text:
                continue
            gt = rec.get("answer_hard", "")
            if gt not in ("yes", "no"):
                continue
            m = _TOULMIN_RE.search(text)
            if not m:
                continue
            out.append({
                "id":             rec["id"],
                "video":          rec["video"],
                "gt":             gt,
                "grounds":        m.group("grounds").strip(),
                "warrant":        m.group("warrant").strip(),
                "predicted_hard": rec.get("predicted_hard", ""),
            })
    return out


# ── Stratified sample ─────────────────────────────────────────────────────────

def stratified_sample(
    records: list[dict], n: int, rng: np.random.Generator
) -> list[dict]:
    yes_pool = [r for r in records if r["gt"] == "yes"]
    no_pool  = [r for r in records if r["gt"] == "no"]
    n_yes = n // 2
    n_no  = n - n_yes
    n_yes = min(n_yes, len(yes_pool))
    n_no  = min(n_no,  len(no_pool))
    idx_y = rng.choice(len(yes_pool), size=n_yes, replace=False)
    idx_n = rng.choice(len(no_pool),  size=n_no,  replace=False)
    sampled = [yes_pool[i] for i in idx_y] + [no_pool[i] for i in idx_n]
    rng.shuffle(sampled)
    return sampled


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-model", required=True,
                    help="Model name (file stem in pred-dir, e.g. toulmin_sft_v3)")
    ap.add_argument("--pred-dir", required=True,
                    help="Directory with *_predictions.jsonl files")
    ap.add_argument("--model-ckpt", required=True,
                    help="Path to LoRA adapter checkpoint directory")
    ap.add_argument("--n-samples", type=int, default=100,
                    help="Total test samples (split ~50/50 yes/no GT)")
    ap.add_argument("--out-json", default=None,
                    help="Optional path to write full results JSON")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (do not change)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Load & parse predictions ──────────────────────────────────────────────
    print(f"Loading parsed predictions for '{args.target_model}' ...")
    all_recs = load_parsed_predictions(args.pred_dir, args.target_model)
    print(f"  Total parseable records: {len(all_recs)}")
    print(f"  yes-GT: {sum(1 for r in all_recs if r['gt']=='yes')}  "
          f"no-GT: {sum(1 for r in all_recs if r['gt']=='no')}")

    # ── Verify model checkpoint ───────────────────────────────────────────────
    adapter_cfg = os.path.join(args.model_ckpt, "adapter_config.json")
    if not os.path.exists(adapter_cfg):
        sys.exit(f"[ABORT] Adapter not found: {adapter_cfg}")
    print(f"  Checkpoint OK: {args.model_ckpt}")

    # ── Build per-class warrant pools ─────────────────────────────────────────
    warrants_yes = [r["warrant"] for r in all_recs if r["gt"] == "yes"]
    warrants_no  = [r["warrant"] for r in all_recs if r["gt"] == "no"]
    print(f"  Warrant pool — yes: {len(warrants_yes)}  no: {len(warrants_no)}")

    # ── Stratified sample ─────────────────────────────────────────────────────
    test_recs = stratified_sample(all_recs, args.n_samples, rng)
    print(f"\nTest set: {len(test_recs)} records  "
          f"(yes-GT: {sum(1 for r in test_recs if r['gt']=='yes')}  "
          f"no-GT: {sum(1 for r in test_recs if r['gt']=='no')})")

    # ── Build random-warrant pool (all warrants except current record's) ──────
    all_warrants = [(r["warrant"], r["gt"]) for r in all_recs]

    # ── Run experiment ────────────────────────────────────────────────────────
    per_record: list[dict] = []
    n_valid_a = n_flip_b = n_flip_c = 0
    n_invalid_a = n_invalid_b = n_invalid_c = 0

    for i, rec in enumerate(test_recs):
        print(f"\n[{i+1:3d}/{len(test_recs)}] id={rec['id'][:50]}  gt={rec['gt']}")

        # ── (a) original warrant ──────────────────────────────────────────────
        ans_a = model_predict_answer(
            rec["video"], rec["grounds"], rec["warrant"], args.model_ckpt
        )
        print(f"  (a) original warrant  → {ans_a}")
        if ans_a == "invalid":
            n_invalid_a += 1
            print("  (a) INVALID — skipping record")
            per_record.append({
                "id": rec["id"], "gt": rec["gt"],
                "ans_a": ans_a, "ans_b": None, "ans_c": None,
                "flip_b": None, "flip_c": None,
                "grounds": rec["grounds"],
                "warrant_orig": rec["warrant"],
                "warrant_rand": None, "warrant_opp": None,
            })
            continue
        n_valid_a += 1

        # ── (b) random warrant (different record) ─────────────────────────────
        # Pick from all warrants that are not from this record
        candidate_pool = [(w, g) for w, g in all_warrants if w != rec["warrant"]]
        rand_idx = int(rng.integers(len(candidate_pool)))
        warrant_rand, _ = candidate_pool[rand_idx]
        ans_b = model_predict_answer(
            rec["video"], rec["grounds"], warrant_rand, args.model_ckpt
        )
        flip_b = (ans_a != ans_b) if ans_b != "invalid" else None
        if ans_b == "invalid":
            n_invalid_b += 1
        elif flip_b:
            n_flip_b += 1
        print(f"  (b) random warrant    → {ans_b}  flip={flip_b}")

        # ── (c) opposite-class warrant ────────────────────────────────────────
        opp_pool = warrants_no if rec["gt"] == "yes" else warrants_yes
        opp_idx = int(rng.integers(len(opp_pool)))
        warrant_opp = opp_pool[opp_idx]
        ans_c = model_predict_answer(
            rec["video"], rec["grounds"], warrant_opp, args.model_ckpt
        )
        flip_c = (ans_a != ans_c) if ans_c != "invalid" else None
        if ans_c == "invalid":
            n_invalid_c += 1
        elif flip_c:
            n_flip_c += 1
        print(f"  (c) opposite warrant  → {ans_c}  flip={flip_c}")

        per_record.append({
            "id":           rec["id"],
            "gt":           rec["gt"],
            "ans_a":        ans_a,
            "ans_b":        ans_b,
            "ans_c":        ans_c,
            "flip_b":       flip_b,
            "flip_c":       flip_c,
            "grounds":      rec["grounds"],
            "warrant_orig": rec["warrant"],
            "warrant_rand": warrant_rand,
            "warrant_opp":  warrant_opp,
        })

    # ── Compute rates (excluding invalids) ────────────────────────────────────
    n_b_denom = n_valid_a - n_invalid_b
    n_c_denom = n_valid_a - n_invalid_c
    flip_rate_b = n_flip_b / n_b_denom if n_b_denom > 0 else float("nan")
    flip_rate_c = n_flip_c / n_c_denom if n_c_denom > 0 else float("nan")

    # ── Decision ──────────────────────────────────────────────────────────────
    if (not (flip_rate_b != flip_rate_b) and not (flip_rate_c != flip_rate_c)):
        if flip_rate_c > 0.30 and flip_rate_c > 2 * flip_rate_b:
            verdict = "SUPPORTED"
        elif flip_rate_c < 0.20 or flip_rate_c <= flip_rate_b:
            verdict = "CONTRADICTED"
        else:
            verdict = "MIXED"
    else:
        verdict = "INSUFFICIENT_DATA"

    # ── Print tables ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("WARRANT COUNTERFACTUAL FLIP RATE")
    print(f"Model: {args.target_model}   n_test={len(test_recs)}   seed={args.seed}")
    print("=" * 72)
    hdr = (f"{'Condition':<30}  {'n_valid':>7}  {'n_flips':>8}  "
           f"{'flip_rate':>10}  {'n_invalid':>10}")
    print(hdr)
    print("-" * len(hdr))
    print(f"{'(a) original warrant':<30}  {n_valid_a:>7}  {'—':>8}  "
          f"{'—':>10}  {n_invalid_a:>10}")
    print(f"{'(b) random warrant':<30}  {n_b_denom:>7}  {n_flip_b:>8}  "
          f"{flip_rate_b:>10.1%}  {n_invalid_b:>10}")
    print(f"{'(c) opposite-class warrant':<30}  {n_c_denom:>7}  {n_flip_c:>8}  "
          f"{flip_rate_c:>10.1%}  {n_invalid_c:>10}")

    print("\n" + "=" * 72)
    print("DECISION")
    print("=" * 72)
    print(f"  flip_rate_b (random warrant):         {flip_rate_b:.1%}")
    print(f"  flip_rate_c (opposite-class warrant): {flip_rate_c:.1%}")
    print(f"  flip_rate_c > 0.30:                   {flip_rate_c > 0.30}")
    print(f"  flip_rate_c > 2 * flip_rate_b:        "
          f"{flip_rate_c > 2 * flip_rate_b}  "
          f"(2x = {2*flip_rate_b:.1%})")
    print(f"\n  Verdict: flip_rate_c = {flip_rate_c:.1%} => {verdict}")

    # ── Dump JSON ─────────────────────────────────────────────────────────────
    output = {
        "model":        args.target_model,
        "n_samples":    len(test_recs),
        "seed":         args.seed,
        "n_valid_a":    n_valid_a,
        "n_invalid_a":  n_invalid_a,
        "n_flip_b":     n_flip_b,
        "n_invalid_b":  n_invalid_b,
        "flip_rate_b":  round(flip_rate_b, 4),
        "n_flip_c":     n_flip_c,
        "n_invalid_c":  n_invalid_c,
        "flip_rate_c":  round(flip_rate_c, 4),
        "verdict":      verdict,
        "per_record":   per_record,
    }
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
