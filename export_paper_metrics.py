import json
from collections import Counter
from glob import glob

BASE = "/workspace/PSI_change/json_mode_90/predictions"
out = {}

for path in sorted(glob(f"{BASE}/*_predictions.jsonl")):
    name = path.split("/")[-1].replace("_predictions.jsonl", "")
    if "BAD" in name.upper() or "OLD" in name.upper():
        continue
    records = [json.loads(l) for l in open(path)]
    n = len(records)
    if n == 0: continue
    
    correct = sum(1 for r in records if r["predicted_hard"] == r["answer_hard"])
    brier = sum((r["predicted_prob"] - r["answer_soft"])**2 for r in records) / n
    gt = Counter(r["answer_hard"] for r in records)
    cm = Counter()
    for r in records:
        cm[(r["answer_hard"], r["predicted_hard"])] += 1
    rec_yes = cm[("yes","yes")] / max(gt["yes"], 1)
    rec_no = cm[("no","no")] / max(gt["no"], 1)
    
    # diversity
    texts = [r["raw_samples"][0] for r in records if r.get("raw_samples")]
    import hashlib
    full_unique = len(set(hashlib.md5(t.encode()).hexdigest() for t in texts))
    prefix_unique = len(set(t[:100] for t in texts))
    
    out[name] = {
        "n": n,
        "acc": correct/n,
        "balanced_acc": (rec_yes + rec_no)/2,
        "brier": brier,
        "rec_yes": rec_yes,
        "rec_no": rec_no,
        "n_yes_pred": sum(1 for r in records if r["predicted_hard"]=="yes"),
        "n_no_pred": sum(1 for r in records if r["predicted_hard"]=="no"),
        "full_unique_pct": 100 * full_unique / len(texts),
        "prefix_100_unique_pct": 100 * prefix_unique / len(texts),
        "n_full_unique": full_unique,
        "n_prefix_unique": prefix_unique,
    }

with open(f"{BASE}/paper_metrics_export.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"Exported {len(out)} model metrics to {BASE}/paper_metrics_export.json")
