"""Convert SC CoT predictions into canonical SFT format.
Joins with vanilla CoT to recover system/prompt/video fields,
wraps majority-vote sample text in <thinking>...</thinking>
+ appends 'answer: X' line to match vanilla CoT-SFT supervision.
"""
import json
from pathlib import Path
from collections import Counter

SC_COT  = "/workspace/PSI_change/json_mode_90/psi_sc_cot_baseline.jsonl"
VANILLA = "/workspace/PSI_change/json_mode_90/trf_train/psi_cot_baseline_1000.jsonl"
OUT     = "/workspace/PSI_change/json_mode_90/psi_sc_cot_canonical_1000.jsonl"

# 1) build vanilla lookup
vanilla = {}
with open(VANILLA) as f:
    for line in f:
        r = json.loads(line)
        vanilla[r['id']] = r
print(f"vanilla CoT lookup loaded: {len(vanilla)} records")

# 2) process SC CoT
stats = Counter()
pred_dist = Counter()
matches = 0

with open(SC_COT) as fin, open(OUT, 'w') as fout:
    for line in fin:
        sc = json.loads(line)
        stats['total'] += 1
        
        v = vanilla.get(sc['id'])
        if v is None:
            stats['missing_vanilla'] += 1
            continue
        
        sc_meta = sc['sc_meta']
        pred = sc_meta['predicted_answer']
        if pred is None:
            stats['no_prediction'] += 1
            continue
        
        # find first sample whose parsed answer == majority
        chosen = next((s for s in sc_meta['samples']
                       if s['parsed'] == pred), None)
        if chosen is None:
            stats['no_matching_sample'] += 1
            continue
        
        # wrap raw text in canonical thinking + answer format
        raw = chosen['raw'].strip()
        completion_text = (
            f"<thinking>\n{raw}\n</thinking>\nanswer: {pred}"
        )
        
        # build canonical record (same schema as vanilla CoT)
        new_r = {
            "id": sc['id'],
            "video": v['video'],
            "system": v['system'],
            "prompt": v['prompt'],
            "completion": [{"role": "assistant", "content": completion_text}],
            "answer": pred,
            "annotator_answer": sc['annotator_answer'],
            "matches_annotator": (pred == sc['annotator_answer']),
            "aggregated_intent": v.get('aggregated_intent'),
            "meta": {
                **v.get('meta', {}),
                "vote_distribution": sc_meta['votes'],
                "n_samples_voted": sc_meta['n_samples'],
                "temperature": sc_meta['temperature'],
                "source": "sc_cot",
            },
        }
        
        fout.write(json.dumps(new_r) + '\n')
        stats['built'] += 1
        pred_dist[pred] += 1
        if pred == sc['annotator_answer']:
            matches += 1

print()
print(f"=== Stats ===")
for k, v in stats.items():
    print(f"  {k}: {v}")
if stats['built'] > 0:
    print(f"  matches_annotator: {matches} "
          f"({matches/stats['built']*100:.1f}%)")
print(f"  pred distribution: {dict(pred_dist)}")
print(f"  output: {OUT}")

# spot check 3 records
print()
print("=== Spot check ===")
with open(OUT) as f:
    recs = [json.loads(l) for i, l in enumerate(f) if i < 3]
for r in recs:
    print(f"\n--- id={r['id']} ---")
    print(f"  answer={r['answer']}  matches={r['matches_annotator']}")
    print(f"  completion (first 300 chars):")
    print(f"  {r['completion'][0]['content'][:300]}")
