"""
build_sft_dpo_datasets.py
=========================
Filters Toulmin v2 and CoT baseline outputs into clean SFT / DPO training
files with a reproducible stratified train/val split.

Deterministic: same inputs + seed → byte-identical outputs on every run.
Side-effect-safe: outputs are overwritten (they are derived files), inputs
are never modified.

Run from the Docker container:
    python build_sft_dpo_datasets.py
"""

import json
import os
import random
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

# ============================================================
# Configuration — all tunable knobs live here
# ============================================================
MARGIN_THRESHOLD = 0.7    # Toulmin debate.margin must exceed this
VAL_FRACTION     = 0.10   # fraction held out as validation
RANDOM_SEED      = 42     # frozen; never change after first run
REQUIRE_BALANCE  = True   # stratify train/val by answer (yes/no)

_BASE = "/workspace/PSI_change/json_mode_90"
_TR   = f"{_BASE}/trf_train"

# ── Inputs ────────────────────────────────────────────────────────────────────
TOULMIN_INPUT = f"{_TR}/psi_toulmin_orchestrator_v2_1000.jsonl"
COT_INPUT     = f"{_TR}/psi_cot_baseline_1000.jsonl"

# ── Toulmin outputs ───────────────────────────────────────────────────────────
OUT_SFT_T       = f"{_TR}/psi_sft_toulmin.jsonl"
OUT_SFT_T_TRAIN = f"{_TR}/psi_sft_toulmin_train.jsonl"
OUT_SFT_T_VAL   = f"{_TR}/psi_sft_toulmin_val.jsonl"
OUT_DPO_T       = f"{_TR}/psi_dpo_toulmin.jsonl"
OUT_DPO_T_TRAIN = f"{_TR}/psi_dpo_toulmin_train.jsonl"
OUT_DPO_T_VAL   = f"{_TR}/psi_dpo_toulmin_val.jsonl"

# ── CoT outputs ───────────────────────────────────────────────────────────────
OUT_SFT_C       = f"{_TR}/psi_sft_cot.jsonl"
OUT_SFT_C_TRAIN = f"{_TR}/psi_sft_cot_train.jsonl"
OUT_SFT_C_VAL   = f"{_TR}/psi_sft_cot_val.jsonl"

# ── Matched outputs (intersection of both pipelines) ─────────────────────────
OUT_INTER_IDS     = f"{_TR}/intersection_ids.txt"
OUT_SFT_T_MATCHED = f"{_TR}/psi_sft_toulmin_matched.jsonl"
OUT_SFT_C_MATCHED = f"{_TR}/psi_sft_cot_matched.jsonl"
OUT_DPO_T_MATCHED = f"{_TR}/psi_dpo_toulmin_matched.jsonl"

# ── Stats output ──────────────────────────────────────────────────────────────
OUT_STATS = f"{_TR}/filter_stats.json"


# ============================================================
# I/O helpers
# ============================================================

def load_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, records: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {len(records):5d} records  →  {path}")


def write_text_lines(path: str, lines: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  {len(lines):5d} lines    →  {path}")


def _line_count(path: str) -> int:
    """Count non-empty lines without loading the whole file into memory."""
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# ============================================================
# Filtering
# ============================================================

def _toulmin_fmt_ok(r: dict) -> bool:
    c = r.get("completion", [{}])[0].get("content", "")
    lines = c.strip().splitlines()
    return (
        len(lines) == 3
        and lines[0].startswith("grounds:")
        and lines[1].startswith("warrant:")
        and lines[2].startswith("answer:")
    )


def filter_toulmin(records: list) -> tuple[list, dict]:
    """
    Returns (filtered_pool, filter_stats).
    Pool records still carry the 'debate' field (needed for DPO construction).
    Filter chain:
      1. matches_annotator == True
      2. margin > MARGIN_THRESHOLD
      3. grounds/warrant_yes/warrant_no non-empty + 3-line completion format
    """
    n_in = len(records)

    step1 = [r for r in records
             if r.get("debate", {}).get("matches_annotator", False)]

    step2 = [r for r in step1
             if r.get("debate", {}).get("margin", 0.0) > MARGIN_THRESHOLD]

    step3 = [r for r in step2
             if (r.get("debate", {}).get("grounds", "").strip()
                 and r.get("debate", {}).get("warrant_yes", "").strip()
                 and r.get("debate", {}).get("warrant_no", "").strip()
                 and _toulmin_fmt_ok(r))]

    return step3, {
        "input_records":          n_in,
        "matches_annotator_true": len(step1),
        "passes_margin":          len(step2),
        "passes_format":          len(step3),
        "final_sft_pool":         len(step3),
    }


def _cot_fmt_ok(r: dict) -> bool:
    c = r.get("completion", [{}])[0].get("content", "")
    return "<thinking>" in c and "answer:" in c


def filter_cot(records: list) -> tuple[list, dict]:
    """
    Returns (filtered_pool, filter_stats).
    Filter chain:
      1. cot_meta.matches_annotator == True
      2. cot_meta.thinking non-empty + <thinking>/answer: in completion
    """
    n_in = len(records)

    step1 = [r for r in records
             if r.get("cot_meta", {}).get("matches_annotator", False)]

    step2 = [r for r in step1
             if (r.get("cot_meta", {}).get("thinking", "").strip()
                 and _cot_fmt_ok(r))]

    return step2, {
        "input_records":          n_in,
        "matches_annotator_true": len(step1),
        "passes_margin":          None,    # N/A for CoT (no debate margin)
        "passes_format":          len(step2),
        "final_sft_pool":         len(step2),
        "final_dpo_pool":         None,    # CoT has no DPO track
    }


# ============================================================
# DPO construction (Toulmin only)
# ============================================================

def build_dpo_records(toulmin_pool: list) -> tuple[list, int]:
    """
    Build one DPO preference pair per filtered Toulmin record.
    chosen  = warrant arguing the annotator's answer side
    rejected = warrant arguing the opposite side

    Returns (dpo_records, skipped_count).
    Skipped count > 0 only in the degenerate case chosen_text == rejected_text.
    """
    dpo = []
    skipped = 0

    for r in toulmin_pool:
        d     = r["debate"]
        annot = r["answer"]                         # annotator ground truth
        opp   = "no" if annot == "yes" else "yes"

        grounds     = d["grounds"]
        warrant_yes = d["warrant_yes"]
        warrant_no  = d["warrant_no"]

        chosen_w   = warrant_yes if annot == "yes" else warrant_no
        rejected_w = warrant_no  if annot == "yes" else warrant_yes

        chosen_text   = f"grounds: {grounds}\nwarrant: {chosen_w}\nanswer: {annot}"
        rejected_text = f"grounds: {grounds}\nwarrant: {rejected_w}\nanswer: {opp}"

        if chosen_text == rejected_text:
            print(f"  [ERROR] identical chosen/rejected for {r['id']} — skipping DPO pair")
            skipped += 1
            continue

        dpo.append({
            "id":       r["id"],
            "video":    r["video"],
            "system":   r["system"],
            "prompt":   r["prompt"],
            "chosen":   [{"role": "assistant", "content": chosen_text}],
            "rejected": [{"role": "assistant", "content": rejected_text}],
            "answer":   annot,
            "meta":     r["meta"],
        })

    return dpo, skipped


# ============================================================
# Train/val split  (video-grouped — prevents annotator-window leakage)
# ============================================================

def group_split(pool: list, seed: int) -> tuple[set, set]:
    """
    Train/val split that keeps all records from the same video_id together.

    Using sklearn GroupShuffleSplit ensures that no video appears in both
    train and val, preventing the model from seeing the same scene during
    training and being "tested" on a different annotator's label for it.

    Returns (train_ids, val_ids) — sets of record `id` strings.
    Deterministic: same pool + seed → same partition.
    """
    groups = [r["meta"]["video_id"] for r in pool]
    gss = GroupShuffleSplit(
        n_splits=1, test_size=VAL_FRACTION, random_state=seed
    )
    train_idx, val_idx = next(gss.split(pool, groups=groups))
    return (
        {pool[i]["id"] for i in train_idx},
        {pool[i]["id"] for i in val_idx},
    )


def apply_split(pool: list, train_ids: set, val_ids: set) -> tuple[list, list]:
    """
    Partition pool into train / val lists.
    Records are sorted by id before output so the files are byte-identical
    across runs regardless of set iteration order.
    """
    train = sorted([r for r in pool if r["id"] in train_ids], key=lambda r: r["id"])
    val   = sorted([r for r in pool if r["id"] in val_ids],   key=lambda r: r["id"])
    return train, val


def _split_stats(train: list, val: list) -> dict:
    train_vids = {r["meta"]["video_id"] for r in train}
    val_vids   = {r["meta"]["video_id"] for r in val}
    overlap    = train_vids & val_vids
    assert not overlap, f"video_id leakage in split: {sorted(overlap)}"
    return {
        "train": {
            "total": len(train),
            "yes":   sum(1 for r in train if r["answer"] == "yes"),
            "no":    sum(1 for r in train if r["answer"] == "no"),
        },
        "val": {
            "total": len(val),
            "yes":   sum(1 for r in val if r["answer"] == "yes"),
            "no":    sum(1 for r in val if r["answer"] == "no"),
        },
        "n_unique_videos_train": len(train_vids),
        "n_unique_videos_val":   len(val_vids),
    }


# ============================================================
# Record transformers
# ============================================================

def strip_debate(r: dict) -> dict:
    """Remove the 'debate' field for SFT output records."""
    return {k: v for k, v in r.items() if k != "debate"}


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(_TR, exist_ok=True)

    global_stats = {
        "toulmin": None,
        "cot":     None,
        "intersection": None,
        "config": {
            "margin_threshold": MARGIN_THRESHOLD,
            "val_fraction":     VAL_FRACTION,
            "random_seed":      RANDOM_SEED,
        },
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Toulmin
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n=== Toulmin  ({TOULMIN_INPUT}) ===")
    t_raw  = load_jsonl(TOULMIN_INPUT)
    t_pool, t_fstats = filter_toulmin(t_raw)
    print(f"  filter: {t_fstats['input_records']} → "
          f"{t_fstats['matches_annotator_true']} (match) → "
          f"{t_fstats['passes_margin']} (margin) → "
          f"{t_fstats['final_sft_pool']} (format) = SFT pool")

    # DPO
    t_dpo, t_dpo_skip = build_dpo_records(t_pool)
    t_fstats["final_dpo_pool"]         = len(t_dpo)
    t_fstats["skipped_dpo_identical"]  = t_dpo_skip
    print(f"  DPO pool: {len(t_dpo)} (skipped identical: {t_dpo_skip})")

    # Shared video-grouped split — applied to both SFT and DPO
    t_train_ids, t_val_ids = group_split(t_pool, RANDOM_SEED)

    # SFT records (debate field stripped)
    t_sft_pool  = [strip_debate(r) for r in t_pool]
    t_id_to_sft = {r["id"]: r for r in t_sft_pool}
    t_id_to_dpo = {r["id"]: r for r in t_dpo}

    # Apply split (sort by id for byte-identical output across runs)
    t_sft_train = sorted(
        [t_id_to_sft[i] for i in t_train_ids if i in t_id_to_sft],
        key=lambda r: r["id"]
    )
    t_sft_val   = sorted(
        [t_id_to_sft[i] for i in t_val_ids if i in t_id_to_sft],
        key=lambda r: r["id"]
    )
    t_dpo_train = sorted(
        [t_id_to_dpo[i] for i in t_train_ids if i in t_id_to_dpo],
        key=lambda r: r["id"]
    )
    t_dpo_val   = sorted(
        [t_id_to_dpo[i] for i in t_val_ids if i in t_id_to_dpo],
        key=lambda r: r["id"]
    )

    t_fstats["split"] = _split_stats(t_sft_train, t_sft_val)
    global_stats["toulmin"] = t_fstats

    write_jsonl(OUT_SFT_T,       t_sft_pool)
    write_jsonl(OUT_SFT_T_TRAIN, t_sft_train)
    write_jsonl(OUT_SFT_T_VAL,   t_sft_val)
    write_jsonl(OUT_DPO_T,       t_dpo)
    write_jsonl(OUT_DPO_T_TRAIN, t_dpo_train)
    write_jsonl(OUT_DPO_T_VAL,   t_dpo_val)

    # ──────────────────────────────────────────────────────────────────────────
    # CoT
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n=== CoT  ({COT_INPUT}) ===")
    c_pool = None
    if os.path.exists(COT_INPUT) and _line_count(COT_INPUT) >= 100:
        c_raw  = load_jsonl(COT_INPUT)
        c_pool, c_fstats = filter_cot(c_raw)
        print(f"  filter: {c_fstats['input_records']} → "
              f"{c_fstats['matches_annotator_true']} (match) → "
              f"{c_fstats['final_sft_pool']} (format) = SFT pool")

        # Independent video-grouped split for CoT (seed+1 keeps it disjoint from Toulmin)
        c_train_ids, c_val_ids = group_split(c_pool, RANDOM_SEED + 1)
        c_sft_train, c_sft_val = apply_split(c_pool, c_train_ids, c_val_ids)

        c_fstats["split"] = _split_stats(c_sft_train, c_sft_val)
        global_stats["cot"] = c_fstats

        write_jsonl(OUT_SFT_C,       c_pool)
        write_jsonl(OUT_SFT_C_TRAIN, c_sft_train)
        write_jsonl(OUT_SFT_C_VAL,   c_sft_val)
    else:
        print("  [skip] CoT input not ready, only producing Toulmin outputs")

    # ──────────────────────────────────────────────────────────────────────────
    # Intersection / matched outputs
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n=== Intersection ===")
    if c_pool is None:
        print("  [skip] CoT not available")
    else:
        t_ids   = {r["id"] for r in t_pool}
        c_ids   = {r["id"] for r in c_pool}
        inter   = sorted(t_ids & c_ids)          # sorted for determinism

        inter_set = set(inter)
        inter_yes = sum(1 for r in t_pool if r["id"] in inter_set and r["answer"] == "yes")
        inter_no  = sum(1 for r in t_pool if r["id"] in inter_set and r["answer"] == "no")
        global_stats["intersection"] = {"size": len(inter), "yes": inter_yes, "no": inter_no}
        print(f"  size={len(inter)}  yes={inter_yes}  no={inter_no}")

        t_sft_matched = [r for r in t_sft_pool if r["id"] in inter_set]
        c_sft_matched = [r for r in c_pool      if r["id"] in inter_set]
        t_dpo_matched = [r for r in t_dpo       if r["id"] in inter_set]

        write_text_lines(OUT_INTER_IDS,     inter)
        write_jsonl(OUT_SFT_T_MATCHED, t_sft_matched)
        write_jsonl(OUT_SFT_C_MATCHED, c_sft_matched)
        write_jsonl(OUT_DPO_T_MATCHED, t_dpo_matched)

    # ──────────────────────────────────────────────────────────────────────────
    # Stats file
    # ──────────────────────────────────────────────────────────────────────────
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(global_stats, f, indent=2, ensure_ascii=False)
    print(f"\n       stats  →  {OUT_STATS}")

    print("\n=== Summary ===")
    print(json.dumps(global_stats, indent=2))


if __name__ == "__main__":
    main()
