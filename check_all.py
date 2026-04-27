"""Real-time accuracy + calibration monitor for ALL inference jobs.
Usage:
    python /workspace/check_all.py                  # one-shot
    watch -n 60 python /workspace/check_all.py      # auto-refresh
"""
import json, os, sys, glob
from collections import Counter

BASE = "/workspace/PSI_change/json_mode_90/predictions"
EVAL = "/workspace/PSI_change/json_mode_90/psi_test_eval_v2.jsonl"

total_expected = sum(1 for _ in open(EVAL)) if os.path.exists(EVAL) else None

pred_files = sorted(glob.glob(os.path.join(BASE, "*_predictions.jsonl")))
pred_files = [f for f in pred_files
              if "BAD" not in os.path.basename(f).upper()
              and "OLD" not in os.path.basename(f).upper()]

if not pred_files:
    print("No prediction files found in", BASE)
    sys.exit(0)

try:
    from sklearn.metrics import roc_auc_score, f1_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

SOFT_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

def compute_ece(records, n_bins=10):
    """Expected Calibration Error: |mean(prob) - mean(true)| weighted by bin size."""
    bins = [(i/n_bins, (i+1)/n_bins) for i in range(n_bins)]
    ece = 0.0
    n = len(records)
    for lo, hi in bins:
        in_bin = [r for r in records
                  if lo <= r["predicted_prob"] < hi
                  or (hi == 1.0 and r["predicted_prob"] == 1.0)]
        if not in_bin:
            continue
        mean_pred = sum(r["predicted_prob"] for r in in_bin) / len(in_bin)
        mean_true = sum(r["answer_soft"] for r in in_bin) / len(in_bin)
        ece += (len(in_bin) / n) * abs(mean_pred - mean_true)
    return ece

def compute(path):
    records = []
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    n = len(records)
    if n == 0:
        return None
    
    correct = sum(1 for r in records if r["predicted_hard"] == r["answer_hard"])
    acc = correct / n
    brier = sum((r["predicted_prob"] - r["answer_soft"]) ** 2 for r in records) / n
    mae = sum(abs(r["predicted_prob"] - r["answer_soft"]) for r in records) / n
    
    gt = Counter(r["answer_hard"] for r in records)
    pr = Counter(r["predicted_hard"] for r in records)
    cm = Counter()
    for r in records:
        cm[(r["answer_hard"], r["predicted_hard"])] += 1
    
    def safe(a, b): return a / b if b else 0.0
    rec_yes = safe(cm[("yes","yes")], gt["yes"])
    rec_no  = safe(cm[("no","no")],   gt["no"])
    prec_yes = safe(cm[("yes","yes")], pr["yes"])
    prec_no  = safe(cm[("no","no")],   pr["no"])
    balanced_acc = (rec_yes + rec_no) / 2
    
    # F1 macro
    f1_yes = safe(2 * prec_yes * rec_yes, prec_yes + rec_yes)
    f1_no  = safe(2 * prec_no  * rec_no,  prec_no  + rec_no )
    f1_macro = (f1_yes + f1_no) / 2
    
    # AUC
    auc = None
    if SKLEARN_OK and gt["yes"] and gt["no"]:
        try:
            y_true = [1 if r["answer_hard"] == "yes" else 0 for r in records]
            y_score = [r["predicted_prob"] for r in records]
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            pass
    
    # ECE
    ece = compute_ece(records)
    
    # per-soft-bin accuracy
    soft_bins = {}
    for lo, hi in SOFT_BINS:
        bin_records = [r for r in records
                       if lo <= r["answer_soft"] < hi
                       or (hi == 1.0 and r["answer_soft"] == 1.0)]
        if not bin_records:
            soft_bins[(lo, hi)] = (0, 0, 0.0)
        else:
            bin_correct = sum(1 for r in bin_records
                              if r["predicted_hard"] == r["answer_hard"])
            soft_bins[(lo, hi)] = (
                bin_correct, len(bin_records),
                bin_correct / len(bin_records),
            )
    
    # output length stats
    lengths = [len(r["raw_samples"][0]) for r in records
               if r.get("raw_samples")]
    if lengths:
        len_mean = sum(lengths) / len(lengths)
        len_min, len_max = min(lengths), max(lengths)
    else:
        len_mean = len_min = len_max = 0
    
    parse_fails = sum(1 for r in records if r.get("n_samples_parsed", 1) == 0)
    
    return {
        "n": n, "correct": correct,
        "acc": acc, "balanced_acc": balanced_acc,
        "f1_macro": f1_macro, "f1_yes": f1_yes, "f1_no": f1_no,
        "brier": brier, "mae": mae, "ece": ece, "auc": auc,
        "gt": gt, "pr": pr, "cm": cm,
        "prec_yes": prec_yes, "prec_no": prec_no,
        "rec_yes": rec_yes, "rec_no": rec_no,
        "soft_bins": soft_bins,
        "len_mean": len_mean, "len_min": len_min, "len_max": len_max,
        "parse_fails": parse_fails,
    }

summaries = []
for path in pred_files:
    name = os.path.basename(path).replace("_predictions.jsonl", "")
    s = compute(path)
    summaries.append((name, s))

# === Top: main metrics table ===
print()
print("=" * 110)
print(f"{'Model':<24} {'Progress':<14} {'Acc':<6} {'BalAcc':<7} {'F1mac':<7} "
      f"{'Brier':<7} {'ECE':<6} {'AUC':<6}")
print("=" * 110)
for name, s in summaries:
    if s is None:
        print(f"{name:<24} {'(empty)':<14}")
        continue
    progress = f"{s['n']}/{total_expected}" if total_expected else f"{s['n']}"
    pct = f"({s['n']/total_expected*100:.0f}%)" if total_expected else ""
    prog_str = f"{progress} {pct}".strip()
    auc_str = f"{s['auc']:.3f}" if s['auc'] is not None else "n/a"
    print(f"{name:<24} {prog_str:<14} "
          f"{s['acc']:.3f}  {s['balanced_acc']:.3f}   {s['f1_macro']:.3f}   "
          f"{s['brier']:.3f}   {s['ece']:.3f}  {auc_str}")
print()

# === Per-class breakdown ===
print(f"{'Model':<24} {'P(yes)':<8} {'P(no)':<8} {'R(yes)':<8} {'R(no)':<8} "
      f"{'F1(yes)':<8} {'F1(no)':<8}")
print("-" * 80)
for name, s in summaries:
    if s is None: continue
    print(f"{name:<24} {s['prec_yes']:.3f}    {s['prec_no']:.3f}    "
          f"{s['rec_yes']:.3f}    {s['rec_no']:.3f}    "
          f"{s['f1_yes']:.3f}    {s['f1_no']:.3f}")
print()

# === Per-soft-label-bin accuracy (calibration view) ===
print("Accuracy by GT soft-label bin (lower bin = "
      "harder = ambiguous case):")
print(f"{'Model':<24} {'[0.0-0.2]':<12} {'[0.2-0.4]':<12} {'[0.4-0.6]':<12} "
      f"{'[0.6-0.8]':<12} {'[0.8-1.0]':<12}")
print("-" * 100)
for name, s in summaries:
    if s is None: continue
    bins = s["soft_bins"]
    cells = []
    for lo, hi in SOFT_BINS:
        c, t, a = bins[(lo, hi)]
        if t == 0:
            cells.append("(empty)   ")
        else:
            cells.append(f"{a:.2f}({c}/{t})")
    print(f"{name:<24} " + " ".join(f"{c:<12}" for c in cells))
print()

# === Output length + parse fails ===
print(f"{'Model':<24} {'len_mean':<10} {'len_min':<10} {'len_max':<10} "
      f"{'parse_fails':<12}")
print("-" * 80)
for name, s in summaries:
    if s is None: continue
    print(f"{name:<24} {int(s['len_mean']):<10} {s['len_min']:<10} "
          f"{s['len_max']:<10} {s['parse_fails']:<12}")
print()

# === Confusion matrices ===
for name, s in summaries:
    if s is None: continue
    cm = s["cm"]
    print(f"--- {name} confusion ---")
    print(f"             pred=yes  pred=no")
    print(f"  gt=yes      {cm[('yes','yes')]:6d}   {cm[('yes','no')]:6d}")
    print(f"  gt=no       {cm[('no','yes')]:6d}   {cm[('no','no')]:6d}")
