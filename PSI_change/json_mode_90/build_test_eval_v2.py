#!/usr/bin/env python3
"""
build_test_eval_v2.py — Build psi_test_eval_v2.jsonl for PSI 2.0 test set.

Fixes the bug in psi_90f_test_eval.jsonl (per-annotator records with a single
annotator's intent as label).  This script instead produces ONE record per
canonical (video_id, track_id, win_start, win_end) window with:
  - aggregated_intent: fraction cross across all annotators who have a valid
    forward-fill annotation for that window  (same formula as select_1000.py)
  - hard_label: "yes" / "no" derived from aggregated_intent ≥ 0.5
  - num_annotators: how many annotators contributed
  - per_ann_intents: {ann_id: intent_str} for auditability

Records with fewer than MIN_ANNOTATORS=3 valid annotations are skipped.

The mp4 clip is looked up from test_videos/ by canonical filename
  {video_id}_{track_id}_{win_start:05d}-{win_end:05d}.mp4
Records whose mp4 is missing are skipped (with a warning).

Output video path uses /workspace/ prefix (Docker-compatible).

Run on HOST (no cv2/GPU needed):
  python build_test_eval_v2.py
  python build_test_eval_v2.py --smoke    # first 5 records only
  python build_test_eval_v2.py --check    # spot-check 5 random records
"""

import argparse
import glob
import json
import os
import random
from collections import defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ANNOT_DIR   = "/mnt/DATA/Shaozhi/PSI/PSI/PSI2.0_Test/annotations/cognitive_annotation_key_frame"
VIDEO_DIR   = "/mnt/DATA/Shaozhi/cosmos2_sft/PSI_change/json_mode_90/test_videos"
DOCKER_VIDEO_PREFIX = "/workspace/PSI_change/json_mode_90/test_videos"
OUTPUT_PATH = "/mnt/DATA/Shaozhi/cosmos2_sft/PSI_change/json_mode_90/psi_test_eval_v2.jsonl"
STATS_PATH  = "/mnt/DATA/Shaozhi/cosmos2_sft/PSI_change/json_mode_90/psi_test_eval_v2_stats.json"

WIN_SIZE       = 90
WIN_STEP       = 45
MIN_ANNOTATORS = 3

INTENT_MAP = {"cross": 1.0, "not_cross": 0.0, "not_sure": 0.5}

# Verbatim from psi_sft_toulmin_matched.jsonl
SYSTEM_PROMPT = (
    "You are an expert autonomous driving assistant specializing in pedestrian "
    "behavior analysis. When given a driving video, you analyze the TARGET pedestrian "
    "(highlighted by a green bounding box) and predict their crossing intention using "
    "the Toulmin argument structure:\n"
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
USER_PROMPT = (
    "Watch the full 90-frame video and predict: will the TARGET pedestrian attempt "
    "to cross in front of the vehicle in the next moment?"
)


# ════════════════════════════════════════════════════
# Verbatim from step_1_make_dataset.py
# ════════════════════════════════════════════════════

def get_windows(observed_frames):
    frames = sorted(observed_frames)
    if len(frames) < WIN_SIZE:
        return []
    f_min, f_max = frames[0], frames[-1]
    windows = []
    pos = f_min
    while pos + WIN_SIZE - 1 <= f_max:
        windows.append((pos, pos + WIN_SIZE - 1))
        pos += WIN_STEP
    last_start = f_max - WIN_SIZE + 1
    if not windows or windows[-1][0] != last_start:
        windows.append((last_start, f_max))
    return windows


def get_forwardfill_annotation(ann_data, observed_frames, win_end):
    key_list  = ann_data.get("key_frame",   [])
    desc_list = ann_data.get("description", [])
    int_list  = ann_data.get("intent",      [])
    prom_list = ann_data.get("promts",      [])

    best = None
    for i, fn in enumerate(observed_frames):
        if fn > win_end:
            break
        if i < len(key_list) and key_list[i] == 1:
            desc   = desc_list[i] if i < len(desc_list) else ""
            intent = int_list[i]  if i < len(int_list)  else ""
            prom   = prom_list[i] if i < len(prom_list) else {}
            if desc or intent:
                best = {
                    "description": desc,
                    "intent":      intent,
                    "promts":      prom,
                    "key_frame":   fn,
                }
    return best


# ════════════════════════════════════════════════════
# Core builder
# ════════════════════════════════════════════════════

def build_records(annot_dir: str, video_dir: str, smoke: bool = False) -> tuple:
    """
    Returns (records, stats_dict).
    Each record is a dict ready to write as a JSONL line.
    """
    cog_files = sorted(glob.glob(os.path.join(annot_dir, "video_*/pedestrian_intent.json")))
    if not cog_files:
        raise FileNotFoundError(f"No pedestrian_intent.json found under {annot_dir}")

    records = []
    skipped_short    = 0
    skipped_ann      = 0
    skipped_no_mp4   = 0
    ann_counts       = []          # num_annotators per kept record
    soft_labels      = []          # aggregated_intent per kept record

    for cf in cog_files:
        video_id = os.path.basename(os.path.dirname(cf))
        with open(cf) as f:
            data = json.load(f)

        for track_id, tdata in data.get("pedestrians", {}).items():
            observed = tdata.get("observed_frames", [])
            cog_anns = tdata.get("cognitive_annotations", {})

            windows = get_windows(observed)
            if not windows:
                skipped_short += 1
                continue

            for ws, we in windows:
                # Collect per-annotator forward-fill intents
                per_ann: dict = {}
                for ann_id, ann_data in cog_anns.items():
                    ann = get_forwardfill_annotation(ann_data, observed, we)
                    if ann is None:
                        continue
                    intent_str = ann.get("intent", "")
                    if intent_str not in INTENT_MAP:
                        continue
                    per_ann[ann_id] = intent_str

                if len(per_ann) < MIN_ANNOTATORS:
                    skipped_ann += 1
                    continue

                # Aggregated soft label
                values = [INTENT_MAP[v] for v in per_ann.values()]
                agg    = sum(values) / len(values)
                hard   = "yes" if agg >= 0.5 else "no"

                # Look up mp4
                fname    = f"{video_id}_{track_id}_{ws:05d}-{we:05d}.mp4"
                mp4_host = os.path.join(video_dir, fname)
                if not os.path.exists(mp4_host):
                    skipped_no_mp4 += 1
                    print(f"  [no-mp4] {fname}")
                    continue

                video_docker = f"{DOCKER_VIDEO_PREFIX}/{fname}"
                record_id    = f"{video_id}_{track_id}_{ws:05d}-{we:05d}"

                records.append({
                    "id":    record_id,
                    "video": video_docker,
                    "system": SYSTEM_PROMPT,
                    "prompt": [{"role": "user", "content": USER_PROMPT}],
                    "answer": hard,
                    "aggregated_intent": round(agg, 4),
                    "num_annotators": len(per_ann),
                    "per_ann_intents": per_ann,
                    "meta": {
                        "video_id":   video_id,
                        "track_id":   track_id,
                        "frame_span": [ws, we],
                        "num_frames": WIN_SIZE,
                        "label_frame": we + 1,
                    },
                })

                ann_counts.append(len(per_ann))
                soft_labels.append(agg)

                if smoke and len(records) >= 5:
                    break
            if smoke and len(records) >= 5:
                break
        if smoke and len(records) >= 5:
            break

    # Stats
    n_yes = sum(1 for r in records if r["answer"] == "yes")
    n_no  = len(records) - n_yes

    # Soft-label histogram (bins: [0,0.2), [0.2,0.4), [0.4,0.6), [0.6,0.8), [0.8,1.0])
    bins   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ["0.0-0.2","0.2-0.4","0.4-0.6","0.6-0.8","0.8-1.0"]
    hist   = defaultdict(int)
    for v in soft_labels:
        for i in range(len(bins)-1):
            if bins[i] <= v < bins[i+1]:
                hist[labels[i]] += 1
                break

    stats = {
        "total_records":     len(records),
        "hard_yes":          n_yes,
        "hard_no":           n_no,
        "skipped_short_track": skipped_short,
        "skipped_low_ann":   skipped_ann,
        "skipped_no_mp4":    skipped_no_mp4,
        "min_annotators_threshold": MIN_ANNOTATORS,
        "mean_annotators":   round(sum(ann_counts)/len(ann_counts), 2) if ann_counts else 0,
        "soft_label_histogram": dict(hist),
        "unique_videos": len({r["meta"]["video_id"] for r in records}),
        "unique_tracks": len({(r["meta"]["video_id"], r["meta"]["track_id"]) for r in records}),
    }
    return records, stats


def spot_check(records: list, n: int = 5):
    print(f"\n[spot-check] {n} random records:")
    for r in random.sample(records, min(n, len(records))):
        print(f"  id={r['id']}  agg={r['aggregated_intent']:.3f}  "
              f"label={r['answer']}  ann={r['num_annotators']}  "
              f"intents={list(r['per_ann_intents'].values())}")


def main():
    parser = argparse.ArgumentParser(
        description="Build psi_test_eval_v2.jsonl with proper per-window soft labels"
    )
    parser.add_argument("--smoke", action="store_true", help="Build only 5 records")
    parser.add_argument("--check", action="store_true", help="Spot-check 5 random records after build")
    args = parser.parse_args()

    print(f"Annot dir  : {ANNOT_DIR}")
    print(f"Video dir  : {VIDEO_DIR}")
    print(f"Output     : {OUTPUT_PATH}")
    print(f"WIN_SIZE={WIN_SIZE}  WIN_STEP={WIN_STEP}  MIN_ANNOTATORS={MIN_ANNOTATORS}")
    print()

    records, stats = build_records(ANNOT_DIR, VIDEO_DIR, smoke=args.smoke)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} records → {OUTPUT_PATH}")

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote stats → {STATS_PATH}")

    print(f"\n{'='*55}")
    for k, v in stats.items():
        print(f"  {k:<30} {v}")
    print(f"{'='*55}")

    if args.check or args.smoke:
        spot_check(records)


if __name__ == "__main__":
    main()
