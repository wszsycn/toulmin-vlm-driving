"""Post-hoc text quality analysis using SBERT semantic similarity.

Compares each model's generated reasoning (grounds + warrant for Toulmin,
thinking for CoT) against:
  1. Reference text from training data (pivot per answer class)
  2. Pairwise between models (which models reason similarly?)
  3. Output diversity (intra-model: are samples diverse?)

Usage:
    python /workspace/analyze_text.py
Requires: pip install sentence-transformers
"""
import json, os, sys, glob
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("Need: pip install sentence-transformers")
    sys.exit(1)

BASE = "/workspace/PSI_change/json_mode_90/predictions"

# use a small model for speed
MODEL_NAME = "all-MiniLM-L6-v2"
print(f"Loading {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

# === collect all model outputs ===
pred_files = sorted(glob.glob(os.path.join(BASE, "*_predictions.jsonl")))
pred_files = [f for f in pred_files
              if "BAD" not in os.path.basename(f).upper()
              and "OLD" not in os.path.basename(f).upper()]

model_outputs = {}
for path in pred_files:
    name = os.path.basename(path).replace("_predictions.jsonl", "")
    records = [json.loads(l) for l in open(path)]
    model_outputs[name] = {
        r["id"]: {
            "text": r["raw_samples"][0] if r.get("raw_samples") else "",
            "predicted": r["predicted_hard"],
            "answer_hard": r["answer_hard"],
            "answer_soft": r["answer_soft"],
        }
        for r in records
    }

# === pairwise model agreement on reasoning ===
model_names = sorted(model_outputs.keys())
common_ids = set.intersection(*(set(model_outputs[n].keys())
                                for n in model_names))
print(f"\nFound {len(common_ids)} records common to all "
      f"{len(model_names)} models")
print()

print("=" * 80)
print("Pairwise SBERT similarity (averaged over common records)")
print("Higher = more similar reasoning")
print("=" * 80)
print(f"{'':24}", end="")
for n in model_names:
    print(f"{n[:14]:<16}", end="")
print()

all_embeds = {}
for n in model_names:
    texts = [model_outputs[n][rid]["text"] for rid in sorted(common_ids)]
    all_embeds[n] = model.encode(texts, batch_size=32,
                                   show_progress_bar=False)

for n1 in model_names:
    print(f"{n1[:22]:<24}", end="")
    for n2 in model_names:
        if n1 == n2:
            print(f"{'1.000':<16}", end="")
        else:
            sims = util.cos_sim(all_embeds[n1],
                                all_embeds[n2]).diag().mean().item()
            print(f"{sims:.3f}{'':<11}", end="")
    print()
print()

# === intra-model output length and uniqueness ===
print("=" * 80)
print("Per-model output stats")
print("=" * 80)
print(f"{'Model':<24} {'avg_len':<10} {'unique%':<10} "
      f"{'agree_w_yes_class':<20}")
print("-" * 80)

for n in model_names:
    outs = list(model_outputs[n].values())
    texts = [o["text"] for o in outs]
    avg_len = sum(len(t) for t in texts) / len(texts)
    # rough unique: first 100 chars hashed
    first_chunks = [t[:100] for t in texts]
    unique_pct = len(set(first_chunks)) / len(first_chunks) * 100
    
    # avg similarity to "yes"-class outputs
    yes_outs = [o["text"] for o in outs if o["predicted"] == "yes"]
    no_outs  = [o["text"] for o in outs if o["predicted"] == "no"]
    if yes_outs and no_outs:
        yes_emb = model.encode(yes_outs[:50], show_progress_bar=False)
        no_emb  = model.encode(no_outs[:50], show_progress_bar=False)
        yes_centroid = yes_emb.mean(axis=0, keepdims=True)
        no_centroid = no_emb.mean(axis=0, keepdims=True)
        yes_no_sim = float(util.cos_sim(yes_centroid, no_centroid))
        within_yes = float(util.cos_sim(yes_emb, yes_centroid).mean())
        agree_str = f"yes-class diversity {within_yes:.3f}, " \
                    f"yes vs no centroid {yes_no_sim:.3f}"
    else:
        agree_str = "n/a"
    
    print(f"{n[:22]:<24} {int(avg_len):<10} "
          f"{unique_pct:.1f}%{'':<5} {agree_str}")

print()
