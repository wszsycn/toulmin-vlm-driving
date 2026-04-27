#!/usr/bin/env python3
"""
build_sft_dpo_v3.py
===================
Rebuilds SFT/DPO datasets with SC CoT (matched-compute baseline) replacing
vanilla CoT, plus all agreed Tricks 1/2/3/5.

Deterministic: same inputs + seed -> byte-identical outputs.
Does NOT modify trf_train/. Writes to trf_train_v3/.

Usage (from host):
    sudo python3 /home/swang226/build_sft_dpo_v3.py
"""

import json
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

# ── Config ────────────────────────────────────────────────────────────────────
MARGIN_YES  = 0.4   # Trick 1: asymmetric -- yes-winner records
MARGIN_NO   = 0.7   # Trick 1: asymmetric -- no-winner records
DPO_MARGIN  = 0.7   # DPO pool: flat threshold (ma=True AND margin > DPO_MARGIN)
VAL_FRAC    = 0.10
RANDOM_SEED = 42

_BASE = "/workspace/PSI_change/json_mode_90"
_OUT  = f"{_BASE}/trf_train_v3"

# ── Inputs ────────────────────────────────────────────────────────────────────
TOULMIN_INPUT = f"{_BASE}/trf_train/psi_toulmin_orchestrator_v2_1000.jsonl"
SC_COT_INPUT  = f"{_BASE}/psi_sc_cot_canonical_1000.jsonl"

# ── Outputs ───────────────────────────────────────────────────────────────────
OUT_SFT_T_TRAIN   = f"{_OUT}/psi_sft_toulmin_train.jsonl"
OUT_SFT_T_VAL     = f"{_OUT}/psi_sft_toulmin_val.jsonl"
OUT_DPO_T_TRAIN   = f"{_OUT}/psi_dpo_toulmin_train.jsonl"
OUT_DPO_T_VAL     = f"{_OUT}/psi_dpo_toulmin_val.jsonl"
OUT_SFT_SC_TRAIN  = f"{_OUT}/psi_sft_sc_cot_train.jsonl"
OUT_SFT_SC_VAL    = f"{_OUT}/psi_sft_sc_cot_val.jsonl"
OUT_SFT_TM_TRAIN  = f"{_OUT}/psi_sft_toulmin_matched_train.jsonl"
OUT_SFT_TM_VAL    = f"{_OUT}/psi_sft_toulmin_matched_val.jsonl"
OUT_SFT_SCM_TRAIN = f"{_OUT}/psi_sft_sc_cot_matched_train.jsonl"
OUT_SFT_SCM_VAL   = f"{_OUT}/psi_sft_sc_cot_matched_val.jsonl"
OUT_DPO_TM_TRAIN  = f"{_OUT}/psi_dpo_toulmin_matched_train.jsonl"
OUT_DPO_TM_VAL    = f"{_OUT}/psi_dpo_toulmin_matched_val.jsonl"
OUT_STATS         = f"{_OUT}/filter_stats.json"


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {len(records):5d} records  ->  {path}")


# ── Filtering helpers ─────────────────────────────────────────────────────────

def _t_fmt_ok(r):
    c = r.get("completion", [{}])[0].get("content", "")
    lines = c.strip().splitlines()
    return (
        len(lines) == 3
        and lines[0].startswith("grounds:")
        and lines[1].startswith("warrant:")
        and lines[2].startswith("answer:")
    )


def _passes_trick1(r):
    d = r.get("debate", {})
    if not d.get("matches_annotator", False):
        return False
    margin = d.get("margin", 0.0)
    winner = d.get("winner", "")
    return margin >= MARGIN_YES if winner == "yes" else margin >= MARGIN_NO


def _debate_fields_ok(r):
    d = r.get("debate", {})
    return (
        bool(d.get("grounds", "").strip())
        and bool(d.get("warrant_yes", "").strip())
        and bool(d.get("warrant_no", "").strip())
    )


# ── Trick 5: sample weight ────────────────────────────────────────────────────

def sample_weight(r):
    ai = r.get("aggregated_intent", 0.5)
    if r["answer"] == "yes":
        return round(max(0.1, 2 * ai - 1), 4)
    else:
        return round(max(0.1, 1 - 2 * ai), 4)


# ── Trick 3: curriculum stage ─────────────────────────────────────────────────

def assign_t_stage(r, is_hard_neg):
    if is_hard_neg:
        return 2
    return 1 if r.get("debate", {}).get("margin", 0.0) >= 0.7 else 2


def assign_sc_stage(r):
    vd = r.get("meta", {}).get("vote_distribution", {})
    total = sum(vd.values()) if vd else 0
    if total == 0:
        return 2
    unanimity = max(vd.get("yes", 0), vd.get("no", 0)) / total
    return 1 if unanimity >= (5.0 / 6.0) else 2   # 5/6 or 6/6 -> stage 1


# ── SC CoT sample weight (uses vote share if no aggregated_intent) ────────────

def sc_sample_weight(r):
    ai = r.get("aggregated_intent")
    if ai is not None:
        if r["answer"] == "yes":
            return round(max(0.1, 2 * ai - 1), 4)
        else:
            return round(max(0.1, 1 - 2 * ai), 4)
    vd = r.get("meta", {}).get("vote_distribution", {})
    total = sum(vd.values()) if vd else 1
    majority_share = max(vd.get("yes", 0), vd.get("no", 0)) / max(total, 1)
    return round(max(0.1, 2 * majority_share - 1), 4)


# ── Trick 2: hard negative mining ────────────────────────────────────────────

def make_hard_neg(r):
    d = r["debate"]
    correct_answer  = r["answer"]
    correct_warrant = d["warrant_yes"] if correct_answer == "yes" else d["warrant_no"]
    new_content = (
        f"grounds: {d['grounds']}\n"
        f"warrant: {correct_warrant}\n"
        f"answer: {correct_answer}"
    )
    new_meta = dict(r["meta"])
    new_meta["source"] = "hard_negative_mined"
    rec = {
        "id":              r["id"] + "_hardneg",
        "video":           r["video"],
        "system":          r["system"],
        "prompt":          r["prompt"],
        "completion":      [{"role": "assistant", "content": new_content}],
        "answer":          correct_answer,
        "aggregated_intent": r.get("aggregated_intent"),
        "meta":            new_meta,
        "stage":           2,
        "sample_weight":   sample_weight(r),
    }
    return rec


# ── DPO construction ──────────────────────────────────────────────────────────

def build_dpo_record(r):
    d     = r["debate"]
    annot = r["answer"]
    opp   = "no" if annot == "yes" else "yes"
    grounds     = d["grounds"]
    chosen_w    = d["warrant_yes"] if annot == "yes" else d["warrant_no"]
    rejected_w  = d["warrant_no"]  if annot == "yes" else d["warrant_yes"]
    chosen_txt  = f"grounds: {grounds}\nwarrant: {chosen_w}\nanswer: {annot}"
    rejected_txt = f"grounds: {grounds}\nwarrant: {rejected_w}\nanswer: {opp}"
    if chosen_txt == rejected_txt:
        return None
    return {
        "id":       r["id"],
        "video":    r["video"],
        "system":   r["system"],
        "prompt":   r["prompt"],
        "chosen":   [{"role": "assistant", "content": chosen_txt}],
        "rejected": [{"role": "assistant", "content": rejected_txt}],
        "answer":   annot,
        "meta":     r["meta"],
    }


# ── Video-grouped train/val split ─────────────────────────────────────────────

def group_split_by_video(pool, seed):
    groups = [r["meta"]["video_id"] for r in pool]
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=seed)
    train_idx, val_idx = next(gss.split(pool, groups=groups))
    return (
        {pool[i]["id"] for i in train_idx},
        {pool[i]["id"] for i in val_idx},
    )


def apply_split(pool, train_ids, val_ids):
    train = sorted([r for r in pool if r["id"] in train_ids], key=lambda r: r["id"])
    val   = sorted([r for r in pool if r["id"] in val_ids],   key=lambda r: r["id"])
    return train, val


def split_stats(train, val):
    train_vids = {r["meta"]["video_id"] for r in train}
    val_vids   = {r["meta"]["video_id"] for r in val}
    overlap = train_vids & val_vids
    assert not overlap, f"video_id leakage: {sorted(overlap)}"
    return {
        "train": {
            "total":    len(train),
            "yes":      sum(1 for r in train if r.get("answer") == "yes"),
            "no":       sum(1 for r in train if r.get("answer") == "no"),
            "n_videos": len(train_vids),
        },
        "val": {
            "total":    len(val),
            "yes":      sum(1 for r in val if r.get("answer") == "yes"),
            "no":       sum(1 for r in val if r.get("answer") == "no"),
            "n_videos": len(val_vids),
        },
    }


def weight_stats(pool):
    ws = [r["sample_weight"] for r in pool if "sample_weight" in r]
    if not ws:
        return {}
    return {
        "mean": round(float(np.mean(ws)), 4),
        "p10":  round(float(np.percentile(ws, 10)), 4),
        "p90":  round(float(np.percentile(ws, 90)), 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(_OUT, exist_ok=True)

    # ── Load inputs ──────────────────────────────────────────────────────────
    print(f"\nLoading Toulmin:  {TOULMIN_INPUT}")
    t_raw = load_jsonl(TOULMIN_INPUT)
    print(f"  {len(t_raw)} records")

    print(f"Loading SC CoT:   {SC_COT_INPUT}")
    sc_raw = load_jsonl(SC_COT_INPUT)
    print(f"  {len(sc_raw)} records")

    # ── Toulmin SFT pool ─────────────────────────────────────────────────────
    print("\n=== Toulmin SFT + DPO ===")

    t_ma_true  = [r for r in t_raw if r.get("debate", {}).get("matches_annotator", False)]
    t_ma_false = [r for r in t_raw if not r.get("debate", {}).get("matches_annotator", False)]
    print(f"  ma=True: {len(t_ma_true)}  |  ma=False: {len(t_ma_false)}")

    # Base pool: Trick 1 + format + debate fields present
    t_base = [r for r in t_ma_true
              if _passes_trick1(r) and _t_fmt_ok(r) and _debate_fields_ok(r)]
    print(f"  Trick 1 + format -> base pool: {len(t_base)}")

    # Hard negatives (Trick 2): ma=False, debate fields present
    hard_negs = []
    for r in t_ma_false:
        if _debate_fields_ok(r):
            hard_negs.append(make_hard_neg(r))
    print(f"  Hard negatives (ma=False, debate ok): {len(hard_negs)}")

    # Add stage + sample_weight to base records (Tricks 3 + 5), strip debate
    t_base_aug = []
    for r in t_base:
        rec = {k: v for k, v in r.items() if k != "debate"}
        rec["stage"]         = assign_t_stage(r, is_hard_neg=False)
        rec["sample_weight"] = sample_weight(r)
        t_base_aug.append(rec)

    # Full Toulmin SFT pool (base + hard-negs)
    t_sft_pool = t_base_aug + hard_negs

    stage1 = sum(1 for r in t_sft_pool if r.get("stage") == 1)
    stage2 = sum(1 for r in t_sft_pool if r.get("stage") == 2)
    t_yes  = sum(1 for r in t_sft_pool if r["answer"] == "yes")
    t_no   = sum(1 for r in t_sft_pool if r["answer"] == "no")
    print(f"  SFT pool total: {len(t_sft_pool)}  yes={t_yes}  no={t_no}")
    print(f"  Stage 1: {stage1}  Stage 2: {stage2}")
    print(f"  sample_weight: {weight_stats(t_sft_pool)}")

    # Toulmin DPO pool (ma=True, margin > DPO_MARGIN flat)
    t_dpo_raw  = [r for r in t_ma_true if r["debate"]["margin"] > DPO_MARGIN]
    t_dpo_pool = []
    t_dpo_skip = 0
    for r in t_dpo_raw:
        rec = build_dpo_record(r)
        if rec is None:
            t_dpo_skip += 1
        else:
            t_dpo_pool.append(rec)
    print(f"  DPO pool: {len(t_dpo_pool)}  (skipped identical: {t_dpo_skip})")

    # Video-grouped split on full SFT pool (base + hard-negs share video_id)
    t_train_ids, t_val_ids = group_split_by_video(t_sft_pool, RANDOM_SEED)
    t_sft_train, t_sft_val = apply_split(t_sft_pool, t_train_ids, t_val_ids)

    # DPO split: map each DPO record to train/val via its video_id
    t_vid_split = {}
    for r in t_sft_pool:
        vid = r["meta"]["video_id"]
        if r["id"] in t_train_ids:
            t_vid_split[vid] = "train"
        elif r["id"] in t_val_ids:
            t_vid_split[vid] = "val"

    t_dpo_train = sorted(
        [r for r in t_dpo_pool if t_vid_split.get(r["meta"]["video_id"]) == "train"],
        key=lambda r: r["id"],
    )
    t_dpo_val = sorted(
        [r for r in t_dpo_pool if t_vid_split.get(r["meta"]["video_id"]) == "val"],
        key=lambda r: r["id"],
    )

    t_sft_stats = {
        "input":                  len(t_raw),
        "ma_true":                len(t_ma_true),
        "ma_false":               len(t_ma_false),
        "sft_base_after_trick1":  len(t_base),
        "hard_negs":              len(hard_negs),
        "sft_pool_total":         len(t_sft_pool),
        "sft_pool_yes":           t_yes,
        "sft_pool_no":            t_no,
        "stage1":                 stage1,
        "stage2":                 stage2,
        "sample_weight":          weight_stats(t_sft_pool),
        "dpo_pool":               len(t_dpo_pool),
        "sft_split":              split_stats(t_sft_train, t_sft_val),
        "dpo_split":              split_stats(t_dpo_train, t_dpo_val),
    }

    # ── SC CoT SFT pool ───────────────────────────────────────────────────────
    print("\n=== SC CoT SFT ===")

    sc_pool = [r for r in sc_raw if r.get("matches_annotator", False)]
    print(f"  ma=True: {len(sc_pool)}  (out of {len(sc_raw)})")

    # Add stage + sample_weight (Tricks 3 + 5); no Trick 2 for SC CoT
    sc_pool_aug = []
    for r in sc_pool:
        rec = dict(r)
        rec["stage"]         = assign_sc_stage(r)
        rec["sample_weight"] = sc_sample_weight(r)
        sc_pool_aug.append(rec)

    sc_stage1 = sum(1 for r in sc_pool_aug if r.get("stage") == 1)
    sc_stage2 = sum(1 for r in sc_pool_aug if r.get("stage") == 2)
    sc_yes = sum(1 for r in sc_pool_aug if r["answer"] == "yes")
    sc_no  = sum(1 for r in sc_pool_aug if r["answer"] == "no")
    print(f"  SC CoT pool: {len(sc_pool_aug)}  yes={sc_yes}  no={sc_no}")
    print(f"  Stage 1: {sc_stage1}  Stage 2: {sc_stage2}")
    print(f"  sample_weight: {weight_stats(sc_pool_aug)}")

    # Independent split (seed+1 keeps video partition disjoint from Toulmin)
    sc_train_ids, sc_val_ids = group_split_by_video(sc_pool_aug, RANDOM_SEED + 1)
    sc_sft_train, sc_sft_val = apply_split(sc_pool_aug, sc_train_ids, sc_val_ids)

    sc_stats = {
        "input":         len(sc_raw),
        "pool":          len(sc_pool_aug),
        "pool_yes":      sc_yes,
        "pool_no":       sc_no,
        "stage1":        sc_stage1,
        "stage2":        sc_stage2,
        "sample_weight": weight_stats(sc_pool_aug),
        "split":         split_stats(sc_sft_train, sc_sft_val),
    }

    # ── Matched intersection ──────────────────────────────────────────────────
    print("\n=== Matched Intersection ===")

    t_base_ids = {r["id"] for r in t_base_aug}    # Toulmin base only (no _hardneg)
    sc_ids      = {r["id"] for r in sc_pool_aug}  # SC CoT ma=True
    inter_ids   = sorted(t_base_ids & sc_ids)     # sorted for determinism
    inter_set   = set(inter_ids)

    inter_yes = sum(1 for r in t_base_aug if r["id"] in inter_set and r["answer"] == "yes")
    inter_no  = sum(1 for r in t_base_aug if r["id"] in inter_set and r["answer"] == "no")
    print(f"  Toulmin base IDs: {len(t_base_ids)}")
    print(f"  SC CoT IDs:       {len(sc_ids)}")
    print(f"  Intersection:     {len(inter_ids)}  yes={inter_yes}  no={inter_no}")

    t_sft_matched  = [r for r in t_base_aug    if r["id"] in inter_set]
    sc_sft_matched = [r for r in sc_pool_aug   if r["id"] in inter_set]
    t_dpo_matched  = [r for r in t_dpo_pool    if r["id"] in inter_set]

    # Matched Toulmin SFT uses Toulmin split; matched SC CoT uses SC CoT split
    t_m_train,  t_m_val  = apply_split(t_sft_matched,  t_train_ids, t_val_ids)
    sc_m_train, sc_m_val = apply_split(sc_sft_matched, sc_train_ids, sc_val_ids)
    t_dm_train = sorted(
        [r for r in t_dpo_matched if t_vid_split.get(r["meta"]["video_id"]) == "train"],
        key=lambda r: r["id"],
    )
    t_dm_val = sorted(
        [r for r in t_dpo_matched if t_vid_split.get(r["meta"]["video_id"]) == "val"],
        key=lambda r: r["id"],
    )

    print(f"  Matched Toulmin SFT:  {len(t_sft_matched)}")
    print(f"  Matched SC CoT SFT:   {len(sc_sft_matched)}")
    print(f"  Matched DPO:          {len(t_dpo_matched)}")

    inter_stats = {
        "size":           len(inter_ids),
        "yes":            inter_yes,
        "no":             inter_no,
        "matched_t_sft":  len(t_sft_matched),
        "matched_sc_sft": len(sc_sft_matched),
        "matched_dpo":    len(t_dpo_matched),
    }

    # ── Write outputs ─────────────────────────────────────────────────────────
    print("\n=== Writing outputs ===")
    write_jsonl(OUT_SFT_T_TRAIN,   t_sft_train)
    write_jsonl(OUT_SFT_T_VAL,     t_sft_val)
    write_jsonl(OUT_DPO_T_TRAIN,   t_dpo_train)
    write_jsonl(OUT_DPO_T_VAL,     t_dpo_val)
    write_jsonl(OUT_SFT_SC_TRAIN,  sc_sft_train)
    write_jsonl(OUT_SFT_SC_VAL,    sc_sft_val)
    write_jsonl(OUT_SFT_TM_TRAIN,  t_m_train)
    write_jsonl(OUT_SFT_TM_VAL,    t_m_val)
    write_jsonl(OUT_SFT_SCM_TRAIN, sc_m_train)
    write_jsonl(OUT_SFT_SCM_VAL,   sc_m_val)
    write_jsonl(OUT_DPO_TM_TRAIN,  t_dm_train)
    write_jsonl(OUT_DPO_TM_VAL,    t_dm_val)

    # ── Stats file ────────────────────────────────────────────────────────────
    stats = {
        "config": {
            "margin_yes":   MARGIN_YES,
            "margin_no":    MARGIN_NO,
            "dpo_margin":   DPO_MARGIN,
            "val_fraction": VAL_FRAC,
            "random_seed":  RANDOM_SEED,
        },
        "toulmin":      t_sft_stats,
        "sc_cot":       sc_stats,
        "intersection": inter_stats,
    }
    os.makedirs(os.path.dirname(OUT_STATS), exist_ok=True)
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n        stats  ->  {OUT_STATS}")

    # ── Side-by-side comparison with v2 ──────────────────────────────────────
    v2_path = "/workspace/PSI_change/json_mode_90/trf_train/filter_stats.json"
    print("\n=== Side-by-side: v2 (vanilla CoT, flat 0.7) vs v3 (SC CoT, Tricks 1/2/3/5) ===")
    if os.path.exists(v2_path):
        with open(v2_path) as f:
            v2 = json.load(f)
        v2_t     = v2.get("toulmin", {})
        v2_c     = v2.get("cot", {})
        v2_inter = v2.get("intersection", {})
        col  = "{:>18}"
        lbl  = "{:35s}"
        hdr  = f"  {lbl.format('')}  {col.format('v2')}  {col.format('v3')}"
        sep  = "-" * 75
        def row(label, v2val, v3val):
            return f"  {lbl.format(label)}  {col.format(str(v2val))}  {col.format(str(v3val))}"
        print(hdr)
        print(sep)
        print(row("Toulmin SFT pool",
                  v2_t.get("final_sft_pool", "?"), len(t_sft_pool)))
        print(row("  -> base",
                  v2_t.get("final_sft_pool", "?"), len(t_base)))
        print(row("  -> hard-negs", "0", len(hard_negs)))
        print(row("  -> stage 1", "N/A", stage1))
        print(row("  -> stage 2", "N/A", stage2))
        v2_tr = v2_t.get("split", {}).get("train", {})
        print(row("  -> yes / no (total)",
                  f"{v2_tr.get('yes','?')}/{v2_tr.get('no','?')}",
                  f"{t_yes}/{t_no}"))
        print(row("Toulmin DPO pool",
                  v2_t.get("final_dpo_pool", "?"), len(t_dpo_pool)))
        v2_cr = v2_c.get("split", {}).get("train", {})
        print(row("CoT/SC-CoT SFT pool",
                  v2_c.get("final_sft_pool", "?"), len(sc_pool_aug)))
        print(row("  (method)", "vanilla CoT", "SC CoT (N=6)"))
        print(row("  -> yes / no (total)",
                  f"{v2_cr.get('yes','?')}/{v2_cr.get('no','?')}",
                  f"{sc_yes}/{sc_no}"))
        print(row("Intersection", v2_inter.get("size", "?"), len(inter_ids)))
        print(row("  -> matched DPO",
                  v2_inter.get("matched_dpo", v2_inter.get("size", "?")),
                  len(t_dpo_matched)))
    else:
        print(f"  [v2 stats not found at {v2_path}]")

    print("\n=== v3 Final Stats ===")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
