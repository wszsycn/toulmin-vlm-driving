#!/usr/bin/env python3
import os
import re
import csv
import gc
import json
from typing import List, Dict, Any
import numpy as np
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

# =========================================================
# 1. Config
# =========================================================
TEST_JSONL        = "/workspace/PSI_change/json_mode_90/psi_90f_test_eval.jsonl"
BASE_MODEL_NAME   = "nvidia/Cosmos-Reason2-8B"
SFT_ADAPTER_PATH  = "/workspace/outputs/Cosmos-Reason2-8B-psi-video-90f-sft"
GRPO_ADAPTER_PATH = "/workspace/outputs/Cosmos-Reason2-8B-psi-video-90f-grpo"
OUTPUT_DIR        = "/workspace/eval_outputs/cosmos_video_compare_3models_90f"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE         = 0
MAX_SAMPLES    = 240
MAX_NEW_TOKENS = 128

# =========================================================
# 2. Prompts
# =========================================================
SYSTEM_PROMPT = (
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

USER_PROMPT = (
    "Watch the driving frames and predict: will the TARGET pedestrian "
    "attempt to cross in front of the vehicle in the next moment?"
)

# =========================================================
# 3. Helpers
# =========================================================
def extract_answer(text: str) -> str:
    if text is None:
        return "unknown"
    m = re.search(r"answer:\s*(yes|no)", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # fallback: first word
    t = text.strip().lower()
    if t.startswith("yes"): return "yes"
    if t.startswith("no"):  return "no"
    return "unknown"

def parse_three_lines(text: str) -> Dict[str, Any]:
    if text is None:
        return {"grounds": "", "warrant": "", "answer": "unknown", "format_ok": False}
    lines = [x.strip() for x in text.strip().split("\n") if x.strip()]
    result = {"grounds": "", "warrant": "", "answer": "unknown", "format_ok": False}
    if len(lines) >= 1 and lines[0].lower().startswith("grounds:"):
        result["grounds"] = lines[0][len("grounds:"):].strip()
    if len(lines) >= 2 and lines[1].lower().startswith("warrant:"):
        result["warrant"] = lines[1][len("warrant:"):].strip()
    if len(lines) >= 3 and lines[2].lower().startswith("answer:"):
        result["answer"] = extract_answer(lines[2])
    result["format_ok"] = (
        len(lines) == 3
        and lines[0].lower().startswith("grounds:")
        and lines[1].lower().startswith("warrant:")
        and lines[2].lower().startswith("answer:")
        and result["answer"] in ["yes", "no"]
    )
    return result

def load_test_samples(path: str, max_samples=None) -> List[Dict[str, Any]]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            samples.append(json.loads(line))
    return samples

def build_messages(sample: Dict[str, Any]) -> List[Dict]:
    """构造 system + user messages"""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": sample["video"],
                    "nframes": 16,
                    "max_pixels": 360 * 420,
                },
                {"type": "text", "text": USER_PROMPT},
            ]
        }
    ]

def print_example(sample: Dict[str, Any], idx: int):
    print("=" * 100)
    print(f"Example {idx}")
    print("ID:", sample["id"])
    print("Video:", sample["video"])
    print("GT answer:", sample["ground_truth_answer"])
    print("=" * 100)

# =========================================================
# 4. Model loading
# =========================================================
def load_base_model(model_name: str, device_idx: int):
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype="auto",
        device_map={"": device_idx},
        attn_implementation="flash_attention_2",
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # ),
    )
    model.eval()
    return model, processor

def load_adapter_model(base_model_name: str, adapter_path: str, device_idx: int):
    processor = AutoProcessor.from_pretrained(base_model_name)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_name,
        dtype="auto",
        device_map={"": device_idx},
        attn_implementation="flash_attention_2",
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # ),
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, processor

# =========================================================
# 5. Inference
# =========================================================
def run_inference_one(model, processor, sample: Dict[str, Any]) -> str:
    messages = build_messages(sample)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    video_metadatas = None
    if video_inputs is not None and len(video_inputs) > 0:
        if isinstance(video_inputs[0], tuple):
            frames_list, meta_list = zip(*video_inputs)
            video_inputs    = list(frames_list)
            video_metadatas = list(meta_list)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    prompt_len  = inputs["input_ids"].shape[1]
    new_tokens  = generated_ids[:, prompt_len:]
    decoded     = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return decoded

def evaluate_model(model, processor, samples: List[Dict[str, Any]], tag: str) -> List[Dict[str, Any]]:
    outputs = []
    for i, sample in enumerate(samples):
        if i % 10 == 0:
            print(f"[{tag}] {i}/{len(samples)}")
        try:
            pred_text = run_inference_one(model, processor, sample)
        except Exception as e:
            pred_text = f"__ERROR__ {e}"
        outputs.append({
            "id":                  sample["id"],
            "video":               sample["video"],
            "ground_truth_answer": sample["ground_truth_answer"],
            "pred_text":           pred_text,
            "pred_answer":         extract_answer(pred_text),
            "format_ok":           parse_three_lines(pred_text)["format_ok"],
            "grounds":             parse_three_lines(pred_text)["grounds"],
            "warrant":             parse_three_lines(pred_text)["warrant"],
        })
    return outputs

# =========================================================
# 6. Metrics (answer only, no reasoning cosine)
# =========================================================
def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    y_true = [r["ground_truth_answer"] for r in results]
    y_pred = [r["pred_answer"] for r in results]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=["yes", "no"],
        average=None,
        zero_division=0,
    )
    return {
        "num_samples":              len(results),
        "format_compliance_rate":   float(sum(r["format_ok"] for r in results) / len(results)),
        "accuracy":                 float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy":        float(balanced_accuracy_score(y_true, y_pred)),
        "precision_yes":            float(precision[0]),
        "recall_yes":               float(recall[0]),
        "f1_yes":                   float(f1[0]),
        "support_yes":              int(support[0]),
        "precision_no":             float(precision[1]),
        "recall_no":                float(recall[1]),
        "f1_no":                    float(f1[1]),
        "support_no":               int(support[1]),
    }

# =========================================================
# 7. Save helpers
# =========================================================
def save_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def save_csv(items, path, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for item in items:
            w.writerow({k: item.get(k, "") for k in fieldnames})

# =========================================================
# 8. Main
# =========================================================
def main():
    samples = load_test_samples(TEST_JSONL, max_samples=MAX_SAMPLES)
    print(f"Loaded {len(samples)} test samples.")
    print_example(samples[0], 0)
    print_example(samples[1], 1)

    # # ---------- Base model ----------
    # print("\n=== Evaluating BASE model ===")
    # base_model, base_processor = load_base_model(BASE_MODEL_NAME, DEVICE)
    # base_outputs = evaluate_model(base_model, base_processor, samples, tag="BASE")
    # base_metrics = compute_metrics(base_outputs)
    # print("\n[BASE metrics]")
    # print(json.dumps(base_metrics, indent=2))
    # del base_model, base_processor
    # gc.collect(); torch.cuda.empty_cache()

    # ---------- SFT model ----------
    # print("\n=== Evaluating SFT model ===")
    # sft_model, sft_processor = load_adapter_model(BASE_MODEL_NAME, SFT_ADAPTER_PATH, DEVICE)
    # sft_outputs = evaluate_model(sft_model, sft_processor, samples, tag="SFT")
    # sft_metrics = compute_metrics(sft_outputs)
    # print("\n[SFT metrics]")
    # print(json.dumps(sft_metrics, indent=2))
    # del sft_model, sft_processor
    # gc.collect(); torch.cuda.empty_cache()

    # ---------- GRPO model ----------
    print("\n=== Evaluating GRPO model ===")
    grpo_model, grpo_processor = load_adapter_model(BASE_MODEL_NAME, GRPO_ADAPTER_PATH, DEVICE)
    grpo_outputs = evaluate_model(grpo_model, grpo_processor, samples, tag="GRPO")
    grpo_metrics = compute_metrics(grpo_outputs)
    print("\n[GRPO metrics]")
    print(json.dumps(grpo_metrics, indent=2))
    del grpo_model, grpo_processor
    gc.collect(); torch.cuda.empty_cache()

    # ---------- Merge & save ----------
    # merged = []
    # for base_r, sft_r, grpo_r in zip(base_outputs, sft_outputs, grpo_outputs):
    #     merged.append({
    #         "id":                  base_r["id"],
    #         "video":               base_r["video"],
    #         "ground_truth_answer": base_r["ground_truth_answer"],
    #         "base_pred_answer":    base_r["pred_answer"],
    #         "base_format_ok":      base_r["format_ok"],
    #         "base_grounds":        base_r["grounds"],
    #         "base_warrant":        base_r["warrant"],
    #         "base_pred_text":      base_r["pred_text"],
    #         "sft_pred_answer":     sft_r["pred_answer"],
    #         "sft_format_ok":       sft_r["format_ok"],
    #         "sft_grounds":         sft_r["grounds"],
    #         "sft_warrant":         sft_r["warrant"],
    #         "sft_pred_text":       sft_r["pred_text"],
    #         "grpo_pred_answer":    grpo_r["pred_answer"],
    #         "grpo_format_ok":      grpo_r["format_ok"],
    #         "grpo_grounds":        grpo_r["grounds"],
    #         "grpo_warrant":        grpo_r["warrant"],
    #         "grpo_pred_text":      grpo_r["pred_text"],
    #     })

    # metrics_summary = {
    #     "base_metrics":  base_metrics,
    #     "sft_metrics":   sft_metrics,
    #     "grpo_metrics":  grpo_metrics,
    # }

    # save_jsonl(merged, f"{OUTPUT_DIR}/compare_results_3models.jsonl")
    # save_csv(merged, f"{OUTPUT_DIR}/compare_results_3models.csv", fieldnames=[
    #     "id", "video", "ground_truth_answer",
    #     "base_pred_answer", "base_format_ok",
    #     "sft_pred_answer",  "sft_format_ok",
    #     "grpo_pred_answer", "grpo_format_ok",
    #     "base_pred_text", "sft_pred_text", "grpo_pred_text",
    #     "base_grounds", "base_warrant",
    #     "sft_grounds",  "sft_warrant",
    #     "grpo_grounds", "grpo_warrant",
    # ])
    # with open(f"{OUTPUT_DIR}/metrics_summary.json", "w") as f:
    #     json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    # print(f"\nDone. Results saved to: {OUTPUT_DIR}")
    # print("\n=== Summary ===")
    # for tag, m in [("BASE", base_metrics), ("SFT", sft_metrics), ("GRPO", grpo_metrics)]:
    #     print(f"{tag}: acc={m['accuracy']:.3f} bal_acc={m['balanced_accuracy']:.3f} "
    #           f"f1_yes={m['f1_yes']:.3f} f1_no={m['f1_no']:.3f} "
    #           f"format={m['format_compliance_rate']:.3f}")


# ---------- Merge & save ----------
    merged = []
    for grpo_r in grpo_outputs:
        merged.append({
            "id":                  grpo_r["id"],
            "video":               grpo_r["video"],
            "ground_truth_answer": grpo_r["ground_truth_answer"],
            "grpo_pred_answer":    grpo_r["pred_answer"],
            "grpo_format_ok":      grpo_r["format_ok"],
            "grpo_grounds":        grpo_r["grounds"],
            "grpo_warrant":        grpo_r["warrant"],
            "grpo_pred_text":      grpo_r["pred_text"],
        })

    metrics_summary = {
        "grpo_metrics": grpo_metrics,
    }

    save_jsonl(merged, f"{OUTPUT_DIR}/grpo_results.jsonl")
    save_csv(merged, f"{OUTPUT_DIR}/grpo_results.csv", fieldnames=[
        "id", "video", "ground_truth_answer",
        "grpo_pred_answer", "grpo_format_ok",
        "grpo_grounds", "grpo_warrant", "grpo_pred_text",
    ])
    with open(f"{OUTPUT_DIR}/grpo_metrics.json", "w") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Results saved to: {OUTPUT_DIR}")
    print("\n=== Summary ===")
    print(f"GRPO: acc={grpo_metrics['accuracy']:.3f} bal_acc={grpo_metrics['balanced_accuracy']:.3f} "
          f"f1_yes={grpo_metrics['f1_yes']:.3f} f1_no={grpo_metrics['f1_no']:.3f} "
          f"format={grpo_metrics['format_compliance_rate']:.3f}")



if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=1 python /workspace/cosmos_video_compare_3models_binary.py