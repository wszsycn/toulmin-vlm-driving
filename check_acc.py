
"""Real-time accuracy monitor for inference progress.
Usage:
    python /workspace/check_acc.py                    # one-shot
    python /workspace/check_acc.py toulmin_sft        # one-shot, specify model
    watch -n 30 python /workspace/check_acc.py        # auto-refresh every 30s
"""
import json, os, sys
from collections import Counter

BASE = "/workspace/PSI_change/json_mode_90/predictions"
EVAL = "/workspace/PSI_change/json_mode_90/psi_test_eval_v2.jsonl"

# which model to check (default: toulmin_sft)
model = sys.argv[1] if len(sys.argv) > 1 else "toulmin_sft"
pred_path = f"{BASE}/{model}_predictions.jsonl"

if not os.path.exists(pred_path):
    print(f"File not found: {pred_path}")
    print(f"Available files:")
    for f in sorted(os.listdir(BASE)):
        print(f"  {f}")
    sys.exit(1)

records = []
with open(pred_path) as f:
    for line in f:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # skip last partial line if file is being written

# total expected records
total_expected = sum(1 for _ in open(EVAL)) if os.path.exists(EVAL) else None

n = len(records)
if n == 0:
    print("No records yet.")
    sys.exit(0)

# core metrics
correct = sum(1 for r in records if r["predicted_hard"] == r["answer_hard"])
acc = correct / n
brier = sum((r["predicted_prob"] - r["answer_soft"]) ** 2 for r in records) / n
mae = sum(abs(r["predicted_prob"] - r["answer_soft"]) for r in records) / n
parse_fails = sum(1 for r in records if r.get("n_samples_parsed", 1) == 0)

# confusion + distributions
cm = Counter()
for r in records:
    cm[(r["answer_hard"], r["predicted_hard"])] += 1
gt = Counter(r["answer_hard"] for r in records)
pr = Counter(r["predicted_hard"] for r in records)

# per-class metrics
def safe(a, b): return a / b if b else 0.0
pp_yes = safe(cm[("yes","yes")], pr["yes"])    # precision when predicting yes
pp_no  = safe(cm[("no","no")],   pr["no"])     # precision when predicting no
rec_yes = safe(cm[("yes","yes")], gt["yes"])   # recall on yes class
rec_no  = safe(cm[("no","no")],   gt["no"])    # recall on no class

# rough ROC-AUC (only if both classes present)
auc_str = "n/a"
try:
    from sklearn.metrics import roc_auc_score
    if gt["yes"] and gt["no"]:
        y_true = [1 if r["answer_hard"] == "yes" else 0 for r in records]
        y_score = [r["predicted_prob"] for r in records]
        auc = roc_auc_score(y_true, y_score)
        auc_str = f"{auc:.3f}"
except Exception:
    pass

progress = f"{n}/{total_expected} ({n/total_expected*100:.1f}%)" if total_expected else f"{n}"

print(f"\n=== {model} — {progress} ===")
print(f"Hard accuracy : {acc:.3f}    ({correct}/{n})")
print(f"Brier score   : {brier:.4f}")
print(f"MAE           : {mae:.4f}")
print(f"ROC-AUC       : {auc_str}")
print(f"Parse fails   : {parse_fails}")
print()
print(f"GT       : yes={gt['yes']:3d}  no={gt['no']:3d}")
print(f"Predicted: yes={pr['yes']:3d}  no={pr['no']:3d}")
print()
print("Confusion (gt × pred):")
print(f"            pred=yes  pred=no")
print(f"  gt=yes    {cm[('yes','yes')]:6d}   {cm[('yes','no')]:6d}")
print(f"  gt=no     {cm[('no','yes')]:6d}   {cm[('no','no')]:6d}")
print()
print(f"Precision: P(gt=yes | pred=yes) = {pp_yes:.3f}")
print(f"           P(gt=no  | pred=no)  = {pp_no:.3f}")
print(f"Recall:    P(pred=yes | gt=yes) = {rec_yes:.3f}")
print(f"           P(pred=no  | gt=no)  = {rec_no:.3f}")
