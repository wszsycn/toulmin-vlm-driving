"""
sc_cot_baseline_pipeline.py — Self-Consistency CoT Baseline
============================================================
Matched-compute control condition for the Toulmin v2 ablation.

Runs the same single-agent CoT prompt N=6 times per record at
temperature=0.7, then majority-votes the parsed answers.  This mirrors
the ~6 Gemini calls per record that orchestrator_toulmin_v2.py uses, so
the two pipelines are compute-matched for a fair comparison.

Leakage profile is identical to orchestrator_toulmin_v2.py and
cot_baseline_pipeline.py:
  sees: video + annotator description + promts
  does NOT see: the annotator's answer

Infrastructure (upload, cache, retry logic) is imported from
cot_baseline_pipeline.py.  We do NOT modify that file.
"""

import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import google.generativeai as genai
import google.api_core.exceptions

# ── Import shared infrastructure from cot_baseline_pipeline ─────────────────
# Inserting the script's directory ensures the sibling module is importable
# regardless of CWD.  The import also runs cot_baseline_pipeline's module-level
# code: _require_env("GEMINI_API_KEY") and genai.configure() — so both API-key
# validation and SDK configuration happen for free.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cot_baseline_pipeline as _cot   # noqa: E402  (must follow sys.path tweak)

_require_env      = _cot._require_env
_md5              = _cot._md5
upload_video      = _cot.upload_video
delete_video_safe = _cot.delete_video_safe
_BASE             = _cot._BASE
SYS_COT           = _cot.SYS_COT
remap_path        = _cot.remap_path
promts_to_text    = _cot.promts_to_text
_RETRYABLE_ERRORS = _cot._RETRYABLE_ERRORS
_strip_fences     = _cot._strip_fences
GEMINI_MODEL      = _cot.GEMINI_MODEL


# ============================================================
# SC-CoT constants
# ============================================================

N_SAMPLES    = 6      # Gemini calls per record
TEMPERATURE  = 0.7    # diversity temperature for each sample
MAX_WORKERS  = 2      # record-level concurrency (same as v2)
RERUN_FAILED = False  # True → reprocess records where matches_annotator=False

INPUT_JSONL = os.path.join(_BASE, "psi_90f_1000.jsonl")
OUT_PATH    = os.path.join(_BASE, "psi_sc_cot_baseline.jsonl")
CACHE_DIR   = os.path.join(_BASE, "agent_cache_sc_cot")

# Module-level override; --smoke CLI flag sets this to 3 at startup.
_MAX_RECORDS: "int | None" = None

os.makedirs(CACHE_DIR, exist_ok=True)

# Own GenerativeModel instance — same model, separate Python object.
# (cot_baseline_pipeline creates its own _gemini_model at import time;
#  sharing it across threads is not safe, so we create one here.)
_sc_model = genai.GenerativeModel(model_name=GEMINI_MODEL)


# ============================================================
# Cache — own CACHE_DIR so SC and vanilla CoT caches never mix
# ============================================================

def _cache_get(key: str):
    p = Path(CACHE_DIR) / f"{key}.txt"
    return p.read_text(encoding="utf-8") if p.exists() else None


def _cache_set(key: str, value: str):
    (Path(CACHE_DIR) / f"{key}.txt").write_text(value, encoding="utf-8")


# ============================================================
# parse_answer — extract "yes" / "no" from a Gemini JSON response
# ============================================================

def parse_answer(result) -> "str | None":
    """
    Accepts either a dict (from sc_gemini_call's JSON parse) or a raw string.
    Returns "yes", "no", or None if the answer cannot be determined.
    """
    if isinstance(result, dict):
        raw = result.get("answer", "")
    else:
        raw = str(result)
    raw = raw.strip().lower()
    if raw in ("yes", "no"):
        return raw
    # Looser fallback: find the first token that is "yes" or "no".
    for token in raw.replace(",", " ").replace(".", " ").split():
        if token in ("yes", "no"):
            return token
    return None


# ============================================================
# Gemini call with configurable temperature
# ============================================================

def _sc_gemini_raw(handle, prompt: str, temperature: float) -> str:
    """
    Single Gemini generation with up to 3 retries on transient errors.
    Mirrors cot_baseline_pipeline._gemini_raw but exposes temperature.
    """
    for attempt in range(3):
        try:
            resp = _sc_model.generate_content(
                [handle, prompt],
                generation_config={"temperature": temperature},
                request_options={"timeout": 120},
            )
            return resp.text.strip()
        except _RETRYABLE_ERRORS as exc:
            if attempt == 2:
                raise
            wait = 15 * (attempt + 1)   # 15 s, 30 s
            print(f"  [gemini] {type(exc).__name__}, sleeping {wait}s...")
            time.sleep(wait)


def sc_gemini_call(
    handle,
    prompt: str,
    sample_idx: int,
    temperature: float,
) -> dict:
    """
    SC-CoT Gemini call.  Cache key includes sample_idx so each of the N
    samples has its own cache file — without this all N samples would read
    the same cached response and majority voting degenerates to N=1.

    Key components: video handle name, full prompt (includes annotation text),
    sample index.  The prompt already embeds SYS_COT so no need to repeat it.
    """
    key = _md5(handle.name, prompt, f"sample={sample_idx}")
    cached = _cache_get(key)
    if cached is not None:
        try:
            return json.loads(_strip_fences(cached))
        except Exception:
            pass   # corrupt/stale cache entry — fall through

    raw = _sc_gemini_raw(handle, prompt, temperature)
    _cache_set(key, raw)

    try:
        return json.loads(_strip_fences(raw))
    except Exception:
        # Retry once with an explicit JSON instruction
        retry_prompt = (
            prompt
            + "\n\nYour previous response was not valid JSON. "
            "Output ONLY the JSON object with no markdown fences or explanation."
        )
        retry_key = _md5(handle.name, prompt, f"sample={sample_idx}", "json_retry")
        retry_cached = _cache_get(retry_key)
        if retry_cached is not None:
            raw2 = retry_cached
        else:
            raw2 = _sc_gemini_raw(handle, retry_prompt, temperature)
            _cache_set(retry_key, raw2)
        return json.loads(_strip_fences(raw2))   # propagate if still broken


# ============================================================
# Prompt builder (reuses SYS_COT and promts_to_text from cot_baseline)
# ============================================================

def _build_prompt(ann: dict) -> str:
    description = ann.get("description", "").strip()
    promts_text = promts_to_text(ann.get("promts", {}))
    return (
        f"{SYS_COT}\n\n"
        f"--- Annotator description ---\n{description}\n\n"
        f"--- Structured annotator observations ---\n{promts_text}"
    )


# ============================================================
# Per-record processing
# ============================================================

def process_record_sc(rec: dict) -> dict:
    """
    Upload the video once, run N_SAMPLES CoT calls sequentially, majority-vote.
    Returns the output record dict.
    """
    ann        = rec.get("raw_annotation", {})
    intent     = ann.get("intent", "")
    video_path = remap_path(rec.get("video", ""))

    if intent not in ("cross", "not_cross"):
        raise ValueError(f"Unsupported intent: {intent!r}")
    if not video_path or not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    annotator_answer = "yes" if intent == "cross" else "no"
    prompt = _build_prompt(ann)
    meta   = rec.get("meta", {})

    handle = None
    try:
        handle = upload_video(video_path)

        samples = []
        votes_yes = votes_no = votes_invalid = 0

        for sample_idx in range(N_SAMPLES):
            try:
                result = sc_gemini_call(handle, prompt, sample_idx, TEMPERATURE)
                parsed = parse_answer(result)
                thinking = result.get("thinking", "") if isinstance(result, dict) else ""
            except Exception as exc:
                print(f"    [sample {sample_idx}] ERROR: {exc}")
                parsed   = None
                thinking = ""

            if parsed == "yes":
                votes_yes += 1
            elif parsed == "no":
                votes_no += 1
            else:
                votes_invalid += 1

            samples.append({
                "sample_idx": sample_idx,
                "raw":        thinking,
                "parsed":     parsed,
            })

        # ── Majority vote ──────────────────────────────────────────────────
        if votes_yes > votes_no:
            predicted_answer = "yes"
        elif votes_no > votes_yes:
            predicted_answer = "no"
        else:
            predicted_answer = None   # tie — abstain

        matches = (
            predicted_answer == annotator_answer
            if predicted_answer is not None
            else False
        )

        return {
            "id":               rec["id"],
            "video_id":         meta.get("video_id", ""),
            "annotator_answer": annotator_answer,
            "sc_meta": {
                "n_samples":         N_SAMPLES,
                "temperature":       TEMPERATURE,
                "samples":           samples,
                "votes":             {
                    "yes":     votes_yes,
                    "no":      votes_no,
                    "invalid": votes_invalid,
                },
                "predicted_answer":  predicted_answer,
                "matches_annotator": matches,
            },
        }

    finally:
        if handle is not None:
            delete_video_safe(handle)


# ============================================================
# Resume helpers
# ============================================================

def load_done_ids(output_path: str, rerun_failed: bool) -> set:
    """
    Build the set of record IDs to skip on resume.
    Uses the `id` field (full annotator-window ID, unique per record) —
    not `video_id` (which is shared across annotator windows).

    rerun_failed=False: skip everything already written.
    rerun_failed=True:  skip only records where matches_annotator=True;
                        failed records are reprocessed and appended (deduplicate
                        downstream by keeping the last line per id).
    """
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj    = json.loads(line)
                rec_id = obj.get("id")
                if not rec_id:
                    continue
                if rerun_failed:
                    if obj.get("sc_meta", {}).get("matches_annotator", False):
                        done.add(rec_id)
                else:
                    done.add(rec_id)
            except Exception:
                pass
    return done


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Self-Consistency CoT baseline pipeline for PSI pedestrian intent"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Process only the first 3 records and print per-sample votes, "
            "so you can verify temperature is actually producing variance "
            "before launching the full 1000-record run."
        ),
    )
    args = parser.parse_args()

    max_records = 3 if args.smoke else _MAX_RECORDS

    # ── Load input ──────────────────────────────────────────────────────────
    with open(INPUT_JSONL, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"Input records: {len(records)}")

    if max_records is not None:
        records = records[:max_records]
        print(f"Limited to {len(records)} records (smoke={args.smoke})")

    # ── Resume ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)), exist_ok=True)
    done_ids  = load_done_ids(OUT_PATH, RERUN_FAILED)
    pending   = [r for r in records if r["id"] not in done_ids]
    skip_cnt  = len(records) - len(pending)
    print(
        f"Already done: {len(done_ids)} | "
        f"Pending: {len(pending)} | "
        f"RERUN_FAILED={RERUN_FAILED}\n"
    )

    # ── Run ─────────────────────────────────────────────────────────────────
    ok_cnt = err_cnt = 0
    written: list = []
    write_lock = threading.Lock()

    with open(OUT_PATH, "a", encoding="utf-8") as out_f:

        def process_and_write(rec):
            result = process_record_sc(rec)
            with write_lock:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                written.append(result)
            return result

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_and_write, rec): rec for rec in pending}
            total = len(pending)
            done_count = 0

            for future in as_completed(futures):
                rec = futures[future]
                done_count += 1
                try:
                    result = future.result()
                    ok_cnt += 1
                    sm    = result["sc_meta"]
                    votes = sm["votes"]
                    print(
                        f"  [{done_count}/{total}] {rec['id'][:55]}"
                        f"  votes={votes}"
                        f"  pred={sm['predicted_answer']}"
                        f"  match={sm['matches_annotator']}"
                    )
                    if args.smoke:
                        per_sample = [
                            (s["sample_idx"], s["parsed"]) for s in sm["samples"]
                        ]
                        print(f"    per-sample: {per_sample}")
                except Exception as exc:
                    err_cnt += 1
                    print(f"  [{done_count}/{total}] ERROR {rec['id']}: {exc}")

    # ── End-of-run summary ───────────────────────────────────────────────────
    matches_yes = sum(
        1 for r in written
        if r["sc_meta"]["matches_annotator"] and r["annotator_answer"] == "yes"
    )
    matches_no = sum(
        1 for r in written
        if r["sc_meta"]["matches_annotator"] and r["annotator_answer"] == "no"
    )
    ties = sum(1 for r in written if r["sc_meta"]["predicted_answer"] is None)
    total_written = len(written)
    total_match   = matches_yes + matches_no

    print(f"\n=== Done ===")
    print(f"OK: {ok_cnt} | Skipped: {skip_cnt} | Errors: {err_cnt}")
    print(
        f"matches_annotator: {total_match} / {total_written} "
        f"(yes: {matches_yes}, no: {matches_no}) | ties: {ties}"
    )
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
