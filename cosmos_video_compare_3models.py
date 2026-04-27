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
from sentence_transformers import SentenceTransformer


# =========================================================
# 1. Config
# =========================================================
# TEST_JSONL = "/workspace/psi_video_eval_1000.jsonl"
TEST_JSONL = "/workspace/PSI_change/json_mode_90/psi_90f_test_eval.jsonl"

BASE_MODEL_NAME = "nvidia/Cosmos-Reason2-8B"
FT_ADAPTER_PATH = "/workspace/outputs/Cosmos-Reason2-8B-psi-video-90f-sft"
GRPO_ADAPTER_PATH = "/workspace/outputs/Cosmos-Reason2-8B-psi-video-90f-grpo"   # 改成你的GRPO输出目录

OUTPUT_DIR = "/workspace/eval_outputs/cosmos_video_compare_3models_90f"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 顺序评测最稳：都放同一张 GPU，避免来回抢
BASE_DEVICE = 0
FT_DEVICE = 0
GRPO_DEVICE = 0

MAX_SAMPLES = 400   # 保证三者都用同样的前100条
MAX_NEW_TOKENS = 256

SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# =========================================================
# 2. Helpers
# =========================================================
def extract_answer(text: str) -> str:
    if text is None:
        return "unknown"
    m = re.search(r"answer:\s*(yes|no)", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
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


def reasoning_text_from_output(text: str) -> str:
    parsed = parse_three_lines(text)
    parts = []
    if parsed["grounds"]:
        parts.append(f"grounds: {parsed['grounds']}")
    if parsed["warrant"]:
        parts.append(f"warrant: {parsed['warrant']}")
    return "\n".join(parts).strip()


def load_test_samples(path: str, max_samples=None) -> List[Dict[str, Any]]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            samples.append(json.loads(line))
    return samples


def print_example(sample: Dict[str, Any], idx: int):
    print("=" * 100)
    print(f"Example {idx}")
    print("ID:", sample["id"])
    print("Video:", sample["video"])
    print("GT answer:", sample["ground_truth_answer"])
    print("\nSystem:")
    print(sample["messages"][0]["content"][0]["text"])
    print("\nUser video path:")
    print(sample["messages"][1]["content"][0]["video"])
    print("\nUser text:")
    print(sample["messages"][1]["content"][1]["text"])
    print("\nGT assistant:")
    print(sample["ground_truth_text"])
    print("=" * 100)


# =========================================================
# 3. Model loading
# =========================================================
def load_base_model_and_processor(model_name: str, device_idx: int):
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype="auto",
        device_map={"": device_idx},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    model.eval()
    return model, processor


def load_adapter_model_and_processor(base_model_name: str, adapter_path: str, device_idx: int):
    processor = AutoProcessor.from_pretrained(base_model_name)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_name,
        dtype="auto",
        device_map={"": device_idx},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, processor


# =========================================================
# 4. Inference
# =========================================================
def run_inference_one(model, processor, sample: Dict[str, Any], max_new_tokens=96) -> str:
    # 只给 system + user，不给 GT assistant
    messages = sample["messages"][:2]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
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
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, prompt_len:]

    decoded = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True
    )[0].strip()

    return decoded


def evaluate_model(model, processor, samples: List[Dict[str, Any]], tag: str, max_new_tokens=96) -> List[Dict[str, Any]]:
    outputs = []

    for i, sample in enumerate(samples):
        if i % 10 == 0:
            print(f"[{tag}] processing sample {i}/{len(samples)}")

        try:
            pred_text = run_inference_one(model, processor, sample, max_new_tokens=max_new_tokens)
        except Exception as e:
            pred_text = f"__ERROR__ {type(e).__name__}: {e}"

        outputs.append({
            "id": sample["id"],
            "video": sample["video"],
            "ground_truth_answer": sample["ground_truth_answer"],
            "ground_truth_text": sample["ground_truth_text"],
            "pred_text": pred_text,
        })

    return outputs


# =========================================================
# 5. Metrics
# =========================================================
def compute_yes_no_metrics(results: List[Dict[str, Any]], pred_key="pred_text") -> Dict[str, Any]:
    y_true = [r["ground_truth_answer"] for r in results]
    y_pred = [extract_answer(r[pred_key]) for r in results]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["yes", "no"],
        average=None,
        zero_division=0
    )

    metrics = {
        "num_samples": len(results),
        "num_valid_yesno_predictions": int(sum(p in ["yes", "no"] for p in y_pred)),
        "format_compliance_rate": float(sum(p in ["yes", "no"] for p in y_pred) / len(results)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_yes": float(precision[0]),
        "recall_yes": float(recall[0]),
        "f1_yes": float(f1[0]),
        "support_yes": int(support[0]),
        "precision_no": float(precision[1]),
        "recall_no": float(recall[1]),
        "f1_no": float(f1[1]),
        "support_no": int(support[1]),
    }
    return metrics


def compute_reasoning_cosine(results: List[Dict[str, Any]], pred_key="pred_text", sbert_model_name=SBERT_MODEL_NAME) -> Dict[str, Any]:
    model = SentenceTransformer(sbert_model_name)

    gt_texts = []
    pred_texts = []

    for r in results:
        gt_reasoning = reasoning_text_from_output(r["ground_truth_text"])
        pred_reasoning = reasoning_text_from_output(r[pred_key])

        if gt_reasoning and pred_reasoning:
            gt_texts.append(gt_reasoning)
            pred_texts.append(pred_reasoning)

    if len(gt_texts) == 0:
        return {
            "num_reasoning_pairs": 0,
            "mean_reasoning_cosine": None,
            "median_reasoning_cosine": None,
        }

    gt_emb = model.encode(gt_texts, convert_to_numpy=True, normalize_embeddings=True)
    pred_emb = model.encode(pred_texts, convert_to_numpy=True, normalize_embeddings=True)

    sims = np.sum(gt_emb * pred_emb, axis=1)

    return {
        "num_reasoning_pairs": int(len(sims)),
        "mean_reasoning_cosine": float(np.mean(sims)),
        "median_reasoning_cosine": float(np.median(sims)),
    }


# =========================================================
# 6. Save helpers
# =========================================================
def save_jsonl(items: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_csv(items: List[Dict[str, Any]], path: str, fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow({k: item.get(k) for k in fieldnames})


# =========================================================
# 7. Main
# =========================================================
def main():
    samples = load_test_samples(TEST_JSONL, max_samples=MAX_SAMPLES)
    print(f"Loaded {len(samples)} test samples.")

    # # 保险：把这100条ID保存下来，后面任何一次对比都能确认是同一批
    # with open(f"{OUTPUT_DIR}/eval_sample_ids.txt", "w", encoding="utf-8") as f:
    #     for s in samples:
    #         f.write(s["id"] + "\n")

    if len(samples) > 0:
        print_example(samples[0], 0)
    if len(samples) > 1:
        print_example(samples[1], 1)

    # # ---------- Base model ----------
    # print("\n=== Evaluating BASE model ===")
    # base_model, base_processor = load_base_model_and_processor(BASE_MODEL_NAME, BASE_DEVICE)
    # base_outputs = evaluate_model(
    #     base_model,
    #     base_processor,
    #     samples,
    #     tag="BASE",
    #     max_new_tokens=MAX_NEW_TOKENS,
    # )

    # base_metrics = compute_yes_no_metrics(base_outputs, pred_key="pred_text")
    # base_cosine = compute_reasoning_cosine(base_outputs, pred_key="pred_text")

    # print("\n[BASE metrics]")
    # print(json.dumps(base_metrics, indent=2))
    # print(json.dumps(base_cosine, indent=2))

    # del base_model
    # del base_processor
    # gc.collect()
    # torch.cuda.empty_cache()

    # # ---------- FT model ----------
    # print("\n=== Evaluating FT model ===")
    # ft_model, ft_processor = load_adapter_model_and_processor(BASE_MODEL_NAME, FT_ADAPTER_PATH, FT_DEVICE)
    # ft_outputs = evaluate_model(
    #     ft_model,
    #     ft_processor,
    #     samples,
    #     tag="FT",
    #     max_new_tokens=MAX_NEW_TOKENS,
    # )

    # ft_metrics = compute_yes_no_metrics(ft_outputs, pred_key="pred_text")
    # ft_cosine = compute_reasoning_cosine(ft_outputs, pred_key="pred_text")

    # print("\n[FT metrics]")
    # print(json.dumps(ft_metrics, indent=2))
    # print(json.dumps(ft_cosine, indent=2))

    # del ft_model
    # del ft_processor
    # gc.collect()
    # torch.cuda.empty_cache()

    # ---------- GRPO model ----------
    print("\n=== Evaluating GRPO model ===")
    grpo_model, grpo_processor = load_adapter_model_and_processor(BASE_MODEL_NAME, GRPO_ADAPTER_PATH, GRPO_DEVICE)
    grpo_outputs = evaluate_model(
        grpo_model,
        grpo_processor,
        samples,
        tag="GRPO",
        max_new_tokens=MAX_NEW_TOKENS,
    )

    grpo_metrics = compute_yes_no_metrics(grpo_outputs, pred_key="pred_text")
    grpo_cosine = compute_reasoning_cosine(grpo_outputs, pred_key="pred_text")

    print("\n[GRPO metrics]")
    print(json.dumps(grpo_metrics, indent=2))
    print(json.dumps(grpo_cosine, indent=2))

    # ---------- Merge results ----------
    merged_results = []
    for sample, base_r, ft_r, grpo_r in zip(samples, base_outputs, ft_outputs, grpo_outputs):
        merged_results.append({
            "id": sample["id"],
            "video": sample["video"],
            "ground_truth_answer": sample["ground_truth_answer"],
            "ground_truth_text": sample["ground_truth_text"],

            "base_pred_text": base_r["pred_text"],
            "base_pred_answer": extract_answer(base_r["pred_text"]),
            "base_reasoning_text": reasoning_text_from_output(base_r["pred_text"]),
            "base_format_ok": parse_three_lines(base_r["pred_text"])["format_ok"],

            "ft_pred_text": ft_r["pred_text"],
            "ft_pred_answer": extract_answer(ft_r["pred_text"]),
            "ft_reasoning_text": reasoning_text_from_output(ft_r["pred_text"]),
            "ft_format_ok": parse_three_lines(ft_r["pred_text"])["format_ok"],

            "grpo_pred_text": grpo_r["pred_text"],
            "grpo_pred_answer": extract_answer(grpo_r["pred_text"]),
            "grpo_reasoning_text": reasoning_text_from_output(grpo_r["pred_text"]),
            "grpo_format_ok": parse_three_lines(grpo_r["pred_text"])["format_ok"],
        })

    metrics_summary = {
        "base_metrics": base_metrics,
        "base_reasoning_cosine": base_cosine,
        "ft_metrics": ft_metrics,
        "ft_reasoning_cosine": ft_cosine,
        "grpo_metrics": grpo_metrics,
        "grpo_reasoning_cosine": grpo_cosine,
    }

    save_jsonl(merged_results, f"{OUTPUT_DIR}/compare_results_3models.jsonl")
    save_csv(
        merged_results,
        f"{OUTPUT_DIR}/compare_results_3models.csv",
        fieldnames=[
            "id", "video", "ground_truth_answer",
            "base_pred_answer", "base_format_ok",
            "ft_pred_answer", "ft_format_ok",
            "grpo_pred_answer", "grpo_format_ok",
            "base_pred_text", "ft_pred_text", "grpo_pred_text"
        ]
    )

    with open(f"{OUTPUT_DIR}/metrics_summary_3models.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()