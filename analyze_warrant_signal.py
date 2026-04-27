#!/usr/bin/env python3
"""
analyze_warrant_signal.py
============================================================
Two metrics that test the "Toulmin elicits the LLM's world model" framing.

METRIC 1 -- I(warrant; answer) - I(grounds; answer)
    Train an SBERT-feature logistic regression to predict the
    ground-truth answer from each segment of the model output.
    Compare the warrant-only classifier to the grounds-only classifier.
    Headline number is delta = acc(warrant) - acc(grounds).

      delta > 0  ->  warrant carries decision-relevant signal
                     beyond what is in the visible grounds.
                     (supports world-model framing)
      delta <= 0 ->  warrant adds no information beyond grounds.
                     (warrant is decoration; framing weakens)

METRIC 5 -- yes/no centroid separation in warrant-only SBERT space
    For each model, compute the cosine distance between the
    yes-class centroid and the no-class centroid in three
    SBERT subspaces: full output, grounds only, warrant only.
    A larger distance means the segment differentiates yes from no.

      warrant_dist >> full_dist -> warrant alone is doing the
                                   differentiating work
                                   (supports framing)
      warrant_dist ~~ full_dist (both small, ~0.05) -> warrant is
                                                       decorative
                                                       (framing fails)

USAGE
    python analyze_warrant_signal.py \\
        --pred-dir /workspace/PSI_change/json_mode_90/predictions \\
        --out-json /workspace/PSI_change/json_mode_90/predictions/warrant_signal.json

REQUIREMENTS
    pip install sentence-transformers scikit-learn numpy

NOTES
    * CoT and SC-CoT outputs do not have an explicit warrant span,
      so segment-level metrics are reported only for Toulmin models.
      Full-output metrics are reported for all five models so the
      reader can still compare apples-to-apples on the headline column.
    * Random seed is fixed (42) so the table is reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np


# ============================================================
# Toulmin segment parser
# ============================================================

# Permissive parser: tolerates markdown bold, varying capitalisation,
# colons or dashes, and arbitrary whitespace between sections.  Captures
# everything between "grounds" and the next "warrant" header, etc.
_TOULMIN_RE = re.compile(
    r"\**\s*grounds\s*\**\s*[:\-]\s*(?P<grounds>.+?)"
    r"\**\s*warrant\s*\**\s*[:\-]\s*(?P<warrant>.+?)"
    r"\**\s*answer\s*\**\s*[:\-]\s*(?P<answer>yes|no)\b",
    re.IGNORECASE | re.DOTALL,
)


def parse_toulmin(text: str) -> dict | None:
    """Best-effort parse.  Returns None if the schema is not detected."""
    m = _TOULMIN_RE.search(text)
    if not m:
        return None
    g = m.group("grounds").strip().strip("*").strip()
    w = m.group("warrant").strip().strip("*").strip()
    a = m.group("answer").lower().strip()
    # Sanity: each segment should be a non-empty short string
    if not g or not w:
        return None
    if len(g) > 4000 or len(w) > 4000:
        return None
    return {"grounds": g, "warrant": w, "answer": a}


# ============================================================
# Record loading
# ============================================================

def load_predictions(pred_dir: str) -> dict[str, list[dict]]:
    """Return {model_name: list of records} from *_predictions.jsonl files."""
    out: dict[str, list[dict]] = {}
    pattern = os.path.join(pred_dir, "*_predictions.jsonl")
    for path in sorted(glob(pattern)):
        if "BAD" in path or "OLD" in path:
            continue
        name = os.path.basename(path).replace("_predictions.jsonl", "")
        recs: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        out[name] = recs
    if not out:
        sys.exit(f"No *_predictions.jsonl files found under {pred_dir!r}")
    return out


def extract_gt_answer(rec: dict) -> str | None:
    """Try common field names for the ground-truth label."""
    # infer_eval.py writes the GT into answer_hard ("yes"/"no")
    for fld in ("answer_hard", "gt_answer", "annotator_answer", "label", "intent"):
        if fld in rec:
            v = rec[fld]
            if v in ("yes", "no"):
                return v
            if v == "cross":
                return "yes"
            if v == "not_cross":
                return "no"
    # raw_annotation.intent fallback (matches sc_cot_baseline_pipeline.py)
    ann = rec.get("raw_annotation", {})
    intent = ann.get("intent", "")
    if intent == "cross":
        return "yes"
    if intent == "not_cross":
        return "no"
    return None


def extract_text(rec: dict) -> str:
    """Get the model's output text from common fields."""
    for fld in ("raw_samples", "outputs"):
        v = rec.get(fld)
        if isinstance(v, list) and v:
            return str(v[0])
    for fld in ("output", "prediction", "text", "response"):
        v = rec.get(fld)
        if isinstance(v, str):
            return v
    return ""


def build_segments(model_name: str, recs: list[dict]) -> list[dict]:
    """For each record, return {gt, full, [grounds, warrant], parse_ok}."""
    is_toulmin = "toulmin" in model_name.lower()
    out: list[dict] = []
    for rec in recs:
        text = extract_text(rec)
        if not text:
            continue
        gt = extract_gt_answer(rec)
        if gt not in ("yes", "no"):
            continue
        seg: dict = {"gt": gt, "full": text}
        if is_toulmin:
            parsed = parse_toulmin(text)
            if parsed:
                seg["grounds"] = parsed["grounds"]
                seg["warrant"] = parsed["warrant"]
                seg["parse_ok"] = True
            else:
                seg["parse_ok"] = False
        else:
            seg["parse_ok"] = False
        out.append(seg)
    return out


# ============================================================
# Metric 1 -- SBERT-feature logistic regression accuracy
# ============================================================

def logreg_cv_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Stratified k-fold CV.  Returns (mean_acc, std_acc, mean_auc)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    if len(set(y.tolist())) < 2:
        return float("nan"), float("nan"), float("nan")

    accs: list[float] = []
    aucs: list[float] = []
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        prob = clf.predict_proba(X[te])[:, 1]
        accs.append(float(accuracy_score(y[te], pred)))
        try:
            aucs.append(float(roc_auc_score(y[te], prob)))
        except ValueError:
            aucs.append(float("nan"))
    return float(np.mean(accs)), float(np.std(accs)), float(np.nanmean(aucs))


# ============================================================
# Metric 5 -- yes/no centroid cosine distance
# ============================================================

def centroid_cos(X: np.ndarray, y: np.ndarray) -> float:
    """Cosine similarity between the yes-class and no-class centroids."""
    yi = np.where(y == 1)[0]
    ni = np.where(y == 0)[0]
    if len(yi) == 0 or len(ni) == 0:
        return float("nan")
    yc = X[yi].mean(axis=0)
    nc = X[ni].mean(axis=0)
    denom = (np.linalg.norm(yc) * np.linalg.norm(nc)) + 1e-12
    return float(np.dot(yc, nc) / denom)


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True,
                    help="Directory with *_predictions.jsonl files")
    ap.add_argument("--out-json", default=None,
                    help="Where to dump the results JSON")
    ap.add_argument("--sbert-model",
                    default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Sentence-Transformers model name or local path")
    ap.add_argument("--min-toulmin-parsed", type=int, default=50,
                    help="Skip warrant/grounds metrics if fewer than this "
                         "many records parse successfully (default 50)")
    args = ap.parse_args()

    print(f"Loading SBERT model: {args.sbert_model}")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(args.sbert_model)

    print(f"\nLoading predictions from: {args.pred_dir}")
    preds = load_predictions(args.pred_dir)
    print(f"Found {len(preds)} models: {sorted(preds.keys())}")

    results: dict[str, dict] = {}

    for model_name, recs in sorted(preds.items()):
        segs = build_segments(model_name, recs)
        n_total = len(segs)
        n_parsed = sum(1 for s in segs if s.get("parse_ok"))
        is_toulmin = "toulmin" in model_name.lower()

        tag = (f"Toulmin parse_ok: {n_parsed}/{n_total}"
               if is_toulmin else "non-Toulmin")
        print(f"\n--- {model_name}: {n_total} valid records ({tag}) ---")

        if n_total == 0:
            print("  no valid records; skipping")
            continue

        y_full = np.array([1 if s["gt"] == "yes" else 0 for s in segs], dtype=np.int64)
        full_texts = [s["full"] for s in segs]
        full_emb = sbert.encode(
            full_texts, show_progress_bar=False, convert_to_numpy=True,
            normalize_embeddings=False,
        )

        result: dict = {
            "n_total": n_total,
            "n_parsed_toulmin": n_parsed if is_toulmin else None,
            "is_toulmin": is_toulmin,
        }

        # ---- Metric 1: full-output logreg ----
        acc, std, auc = logreg_cv_accuracy(full_emb, y_full)
        result["full_logreg_acc"] = acc
        result["full_logreg_acc_std"] = std
        result["full_logreg_auc"] = auc

        # ---- Metric 5: full-output centroid ----
        cos_full = centroid_cos(full_emb, y_full)
        result["full_centroid_cos"] = cos_full
        result["full_centroid_dist"] = 1.0 - cos_full

        # ---- Toulmin-only segment metrics ----
        if is_toulmin and n_parsed >= args.min_toulmin_parsed:
            parsed_segs = [s for s in segs if s.get("parse_ok")]
            y_p = np.array([1 if s["gt"] == "yes" else 0 for s in parsed_segs], dtype=np.int64)
            grounds_emb = sbert.encode(
                [s["grounds"] for s in parsed_segs],
                show_progress_bar=False, convert_to_numpy=True,
            )
            warrant_emb = sbert.encode(
                [s["warrant"] for s in parsed_segs],
                show_progress_bar=False, convert_to_numpy=True,
            )

            g_acc, g_std, g_auc = logreg_cv_accuracy(grounds_emb, y_p)
            w_acc, w_std, w_auc = logreg_cv_accuracy(warrant_emb, y_p)
            result.update({
                "grounds_logreg_acc": g_acc,
                "grounds_logreg_acc_std": g_std,
                "grounds_logreg_auc": g_auc,
                "warrant_logreg_acc": w_acc,
                "warrant_logreg_acc_std": w_std,
                "warrant_logreg_auc": w_auc,
                "delta_acc_warrant_minus_grounds": w_acc - g_acc,
            })

            cos_g = centroid_cos(grounds_emb, y_p)
            cos_w = centroid_cos(warrant_emb, y_p)
            result.update({
                "grounds_centroid_cos": cos_g,
                "grounds_centroid_dist": 1.0 - cos_g,
                "warrant_centroid_cos": cos_w,
                "warrant_centroid_dist": 1.0 - cos_w,
            })

        results[model_name] = result

    # --------------------------------------------------------
    # Print Metric 1 table
    # --------------------------------------------------------
    print("\n" + "=" * 88)
    print("METRIC 1 -- SBERT-feature logistic regression accuracy "
          "(stratified 5-fold CV)")
    print("Higher = the segment carries more decision-relevant signal")
    print("=" * 88)
    hdr = (f"{'Model':<26}{'n':>6}  {'full_acc':>14}  "
           f"{'grounds_acc':>14}  {'warrant_acc':>14}  {'delta(w-g)':>11}")
    print(hdr)
    print("-" * len(hdr))
    for name, r in sorted(results.items()):
        full = f"{r['full_logreg_acc']:.3f} +/- {r['full_logreg_acc_std']:.3f}"
        if "warrant_logreg_acc" in r:
            g = f"{r['grounds_logreg_acc']:.3f} +/- {r['grounds_logreg_acc_std']:.3f}"
            w = f"{r['warrant_logreg_acc']:.3f} +/- {r['warrant_logreg_acc_std']:.3f}"
            delta = f"{r['delta_acc_warrant_minus_grounds']:+.3f}"
        else:
            g = w = delta = "      --"
        print(f"{name:<26}{r['n_total']:>6}  {full:>14}  {g:>14}  {w:>14}  {delta:>11}")

    # --------------------------------------------------------
    # Print Metric 5 table
    # --------------------------------------------------------
    print("\n" + "=" * 88)
    print("METRIC 5 -- yes/no centroid separation in SBERT space")
    print("Distance = 1 - cos_sim(yes_centroid, no_centroid).  Larger = "
          "more yes/no differentiation")
    print("=" * 88)
    hdr = (f"{'Model':<26}  {'full_dist':>10}  "
           f"{'grounds_dist':>13}  {'warrant_dist':>13}")
    print(hdr)
    print("-" * len(hdr))
    for name, r in sorted(results.items()):
        full = f"{r['full_centroid_dist']:.4f}"
        if "warrant_centroid_dist" in r:
            g = f"{r['grounds_centroid_dist']:.4f}"
            w = f"{r['warrant_centroid_dist']:.4f}"
        else:
            g = w = "    --"
        print(f"{name:<26}  {full:>10}  {g:>13}  {w:>13}")

    # --------------------------------------------------------
    # Interpretation cheat-sheet
    # --------------------------------------------------------
    print("\n" + "=" * 88)
    print("INTERPRETATION (world-model framing)")
    print("=" * 88)
    print("""
For Toulmin models, the framing is *supported* if:
  * delta(w - g) > 0          -- the warrant carries signal beyond grounds
  * warrant_dist > full_dist  -- the warrant alone differentiates yes/no
                                 better than the full output
  * warrant_dist > 0.10       -- the warrant differentiation is real
                                 (vs the ~0.05-0.07 noise floor we see in
                                 the full output)

The framing is *contradicted* if:
  * delta(w - g) <= 0   -- warrant adds nothing beyond visible grounds
  * warrant_dist ~ full_dist (both small, ~0.05-0.07)
                        -- the warrant text is decorative; the binary
                           decision lives in the final answer token
""".rstrip())

    # --------------------------------------------------------
    # Dump JSON
    # --------------------------------------------------------
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()