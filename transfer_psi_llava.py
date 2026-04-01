#!/usr/bin/env python3
"""
Convert PSI warrant JSONL to a LLaVA-style dataset structure.

Output JSONL line example:
{
  "id": "...",
  "pair_id": "...",
  "images": ["/path/to/frame_0000.jpg", ...],
  "prompt": [{"role": "user", "content": "..."}],
  "completion": [{"role": "assistant", "content": "..."}],
  "answer": "yes",
  "aggregated_intent": 0.63,
  "difficulty_score": 0.13,
  "meta": {...}
}

This script:
1. Reads your PSI JSONL
2. Filters samples by aggregated_intent and/or answer if requested
3. Selects a subset, optionally balanced yes/no and easy-first
4. Extracts multiple frames from each video
5. Writes a LLaVA-style JSONL that your notebook can read after a small loader step

Example:
python /workspace/transfer_psi_llava.py \
  --input_jsonl /mnt/DATA/Shaozhi/PSI_change/json_mode_16/psi_vllm2_finetune_video_only_jsons/intent+grounds+warrant_answer_overlay__Qexplain.jsonl \
  --output_jsonl /mnt/DATA/Shaozhi/PSI_change/json_mode_16/psi_llava/psi_llava_easy100.jsonl \
  --frames_root /mnt/DATA/Shaozhi/PSI_change/json_mode_16/psi_llava_frames \
  --num_frames 16 \
  --subset_size 100 \
  --balance_yes_no \
  --sort_by_easy \
  --replace_video_token
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Convert PSI JSONL to LLaVA-style JSONL.")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--frames_root", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--balance_yes_no", action="store_true")
    parser.add_argument("--sort_by_easy", action="store_true")
    parser.add_argument("--min_aggregated_intent", type=float, default=None)
    parser.add_argument("--max_aggregated_intent", type=float, default=None)
    parser.add_argument("--answer_filter", type=str, default=None, choices=["yes", "no"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing_frames", action="store_true")
    parser.add_argument("--keep_video_field", action="store_true")
    parser.add_argument("--replace_video_token", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at line {line_num}: {e}") from e
    return items


def write_jsonl(items: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_conversations(conversations: List[Dict[str, str]]):
    human_text = None
    gpt_text = None
    for turn in conversations:
        speaker = turn.get("from")
        value = turn.get("value", "")
        if speaker == "human" and human_text is None:
            human_text = value
        elif speaker == "gpt" and gpt_text is None:
            gpt_text = value
    if human_text is None or gpt_text is None:
        raise ValueError("Missing human or gpt turn in conversations.")
    return human_text, gpt_text


def extract_answer_label(gpt_text: str) -> str:
    match = re.search(r"answer:\s*(yes|no)", gpt_text, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "unknown"


def get_aggregated_intent(item: Dict[str, Any]) -> Optional[float]:
    meta = item.get("meta", {})
    value = meta.get("aggregated_intent", None)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def difficulty_score(aggregated_intent: Optional[float]) -> float:
    if aggregated_intent is None:
        return -1.0
    return abs(float(aggregated_intent) - 0.5)


def filter_items(items: List[Dict[str, Any]], args):
    filtered = []
    for item in items:
        _, gpt_text = parse_conversations(item["conversations"])
        answer = extract_answer_label(gpt_text)
        agg = get_aggregated_intent(item)

        if args.answer_filter is not None and answer != args.answer_filter:
            continue
        if args.min_aggregated_intent is not None:
            if agg is None or agg < args.min_aggregated_intent:
                continue
        if args.max_aggregated_intent is not None:
            if agg is None or agg > args.max_aggregated_intent:
                continue

        item["_answer"] = answer
        item["_aggregated_intent"] = agg
        item["_difficulty_score"] = difficulty_score(agg)
        filtered.append(item)
    return filtered


def select_subset(items: List[Dict[str, Any]], args):
    rng = random.Random(args.seed)

    if args.sort_by_easy:
        items = sorted(items, key=lambda x: x["_difficulty_score"], reverse=True)
    else:
        items = items[:]
        rng.shuffle(items)

    if args.subset_size is None:
        return items

    if not args.balance_yes_no:
        return items[:args.subset_size]

    yes_items = [x for x in items if x["_answer"] == "yes"]
    no_items = [x for x in items if x["_answer"] == "no"]

    k_yes = args.subset_size // 2
    k_no = args.subset_size - k_yes

    selected = yes_items[:k_yes] + no_items[:k_no]

    if len(selected) < args.subset_size:
        remaining = [x for x in items if x not in selected]
        selected.extend(remaining[: args.subset_size - len(selected)])

    if not args.sort_by_easy:
        rng.shuffle(selected)

    return selected[:args.subset_size]


def extract_frames_from_clip(video_path: Path, out_dir: Path, num_frames: int, skip_existing_frames: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_paths = [out_dir / f"frame_{i:04d}.jpg" for i in range(num_frames)]

    if skip_existing_frames and all(p.exists() for p in expected_paths):
        return [str(p) for p in expected_paths]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.copy())

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Video has no frames: {video_path}")

    # 如果帧数多于 num_frames，只保留前 num_frames
    if len(frames) > num_frames:
        frames = frames[:num_frames]

    # 如果帧数少于 num_frames，用最后一帧补齐
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    saved_paths = []
    for i, frame in enumerate(frames):
        out_path = out_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved_paths.append(str(out_path))

    return saved_paths

def convert_item(item: Dict[str, Any], args):
    item_id = item.get("id", "unknown_id")
    pair_id = item.get("pair_id")
    video_path = Path(item["video"])
    meta = item.get("meta", {})

    human_text, gpt_text = parse_conversations(item["conversations"])
    if args.replace_video_token:
        human_text = human_text.replace("<video>", "These are sequential driving frames.").strip()

    frame_dir = Path(args.frames_root) / item_id
    image_paths = extract_frames_from_clip(
        video_path=video_path,
        out_dir=frame_dir,
        num_frames=args.num_frames,
        skip_existing_frames=args.skip_existing_frames,
    )

    output = {
        "id": item_id,
        "pair_id": pair_id,
        "images": image_paths,
        "prompt": [{"role": "user", "content": human_text}],
        "completion": [{"role": "assistant", "content": gpt_text}],
        "answer": item.get("_answer", extract_answer_label(gpt_text)),
        "aggregated_intent": item.get("_aggregated_intent", get_aggregated_intent(item)),
        "difficulty_score": item.get("_difficulty_score", difficulty_score(get_aggregated_intent(item))),
        "meta": meta,
    }

    if args.keep_video_field:
        output["video"] = str(video_path)

    return output


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    print(f"[INFO] Reading {input_jsonl}")
    items = read_jsonl(input_jsonl)
    print(f"[INFO] Raw samples: {len(items)}")

    items = filter_items(items, args)
    print(f"[INFO] After filtering: {len(items)}")

    items = select_subset(items, args)
    print(f"[INFO] After subset selection: {len(items)}")

    converted = []
    for idx, item in enumerate(items, start=1):
        try:
            converted.append(convert_item(item, args))
        except Exception as e:
            print(f"[WARN] Failed on {item.get('id', 'unknown')}: {e}")

        if idx % 50 == 0 or idx == len(items):
            print(f"[INFO] Processed {idx}/{len(items)}")

    write_jsonl(converted, output_jsonl)
    print(f"[INFO] Saved {len(converted)} samples to {output_jsonl}")


if __name__ == "__main__":
    main()
    
    
# python /workspace/cosmos2_sft/transfer_psi_llava.py \
#   --input_jsonl /workspace/PSI_change/json_mode_16/psi_vllm2_finetune_video_only_jsons/intent+grounds+warrant_answer_overlay__Qexplain.jsonl \
#   --output_jsonl /workspace/cosmos2_sft/psi_llava_easy200.jsonl \
#   --frames_root /workspace/cosmos2_sft/psi_llava_frames_easy200 \
#   --num_frames 16 \
#   --subset_size 200 \
#   --balance_yes_no \
#   --sort_by_easy \
#   --replace_video_token
  
  
  
# python /workspace/cosmos2_sft/transfer_psi_llava.py \
#   --input_jsonl /workspace/PSI_change/json_mode_16/psi_vllm2_finetune_video_only_jsons/intent+grounds+warrant_answer_overlay__Qexplain.jsonl \
#   --output_jsonl /workspace/cosmos2_sft/psi_llava_easy200.jsonl \
#   --frames_root /workspace/psi_llava_frames_easy200 \
#   --num_frames 16 \
#   --subset_size 200 \
#   --sort_by_easy \
#   --replace_video_token