"""
cot_baseline_pipeline.py — Single-Agent CoT Baseline
=====================================================
Control condition for a Toulmin-vs-CoT ablation study.

Same leakage profile as orchestrator_toulmin_v2.py:
  sees: video + annotator description + promts (structured observations)
  does NOT see: the annotator's answer

Single Gemini agent produces free-form chain-of-thought reasoning then
predicts yes/no. No multi-agent debate, no Toulmin structure.

Output is in LLaVA format with a `cot_meta` field instead of `debate`.
For SFT training use only records where cot_meta.matches_annotator=True.

Paths use /mnt/DATA/ (host filesystem), not /workspace/ (Docker).
"""

import json
import os
import re
import sys
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import google.generativeai as genai
import google.api_core.exceptions


# ============================================================
# Config
# ============================================================

def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        sys.exit(
            f"ERROR: environment variable {name} is not set.\n"
            f"  export {name}=<your-key>"
        )
    return val


GEMINI_API_KEY = _require_env("GEMINI_API_KEY")

GEMINI_MODEL  = "gemini-3-flash-preview"

MAX_VIDEOS    = None       # smoke test — revert to None before full run
MAX_WORKERS   = 2       # concurrent record threads
UPLOAD_POLL   = 2       # seconds between upload state polls
REQUEST_DELAY = 0.0     # seconds between record submissions (0 = fire immediately)
RERUN_FAILED  = False   # if True, rerun records where matches_annotator=False

_BASE        = "/workspace/PSI_change/json_mode_90"
INPUT_JSONL  = f"{_BASE}/psi_90f_1000.jsonl"
OUTPUT_JSONL = f"{_BASE}/trf_train/psi_cot_baseline_1000.jsonl"
CACHE_DIR    = f"{_BASE}/agent_cache_cot"

# Source prefix in input JSONL video paths (Docker container mount)
_WORKSPACE_PREFIX = "/workspace/"
_HOST_PREFIX      = "/workspace/"


# ============================================================
# SFT output format constants
# ============================================================

# CoT-appropriate system prompt — describes the <thinking>/answer format
# that the fine-tuned model should produce at inference time.
# (v2 uses a Toulmin-specific prompt; this one is intentionally different
# so the two datasets train distinct output formats for the ablation.)
SYSTEM_PROMPT = (
    "You are an expert autonomous driving assistant specializing in pedestrian behavior analysis. "
    "When given a driving video, you analyze the TARGET pedestrian (highlighted by a green bounding box) "
    "and predict their crossing intention using step-by-step chain-of-thought reasoning.\n\n"
    "Always output in this exact format:\n"
    "<thinking>\n"
    "[3-6 sentences covering: pedestrian posture and motion, scene context, "
    "and your inference about likely intent]\n"
    "</thinking>\n"
    "answer: <yes or no>\n\n"
    "Do not output anything outside this structure."
)

# Copied verbatim from v2 (same question, same video framing)
PROMPT_TEXT = (
    "Watch the full 90-frame video and predict: will the TARGET pedestrian "
    "attempt to cross in front of the vehicle in the next moment?"
)


# ============================================================
# Agent prompt
# ============================================================

SYS_COT = """\
You are an expert autonomous driving assistant analyzing pedestrian behavior in a driving video.

Watch the driving video showing a TARGET pedestrian (highlighted by a green bounding box) and reason
step-by-step about whether they will attempt to cross in front of the vehicle in the next moment.

You will also be given the annotator's textual description of the scene as additional context.
Use both the video and the description.

Output format (strict, exactly this structure):

<thinking>
[Free-form chain-of-thought reasoning, 3-6 sentences. Cover: pedestrian's posture and motion,
scene context, and your inference about likely intent.]
</thinking>
answer: <yes or no>

Do not output anything outside this structure. Do not include the word "thinking" or "answer:"
inside the thinking block.

Return your final answer as JSON:
{"thinking": "<the contents of the thinking block>", "answer": "<yes or no>"}"""


# ============================================================
# Initialization
# ============================================================

genai.configure(api_key=GEMINI_API_KEY)
_gemini_model = genai.GenerativeModel(model_name=GEMINI_MODEL)
os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# Cache  (copied verbatim from orchestrator_toulmin_v2.py)
# ============================================================

def _md5(*parts: str) -> str:
    return hashlib.md5("||".join(parts).encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return Path(CACHE_DIR) / f"{key}.txt"


def cache_get(key: str):
    p = _cache_path(key)
    return p.read_text(encoding="utf-8") if p.exists() else None


def cache_set(key: str, value: str):
    _cache_path(key).write_text(value, encoding="utf-8")


# ============================================================
# Video upload  (copied verbatim from v2)
# ============================================================

def upload_video(video_path: str):
    """Upload to Gemini File API; 3 attempts with exponential backoff."""
    for attempt in range(3):
        try:
            print(f"  [upload] {Path(video_path).name} (attempt {attempt + 1})")
            fh = genai.upload_file(path=video_path)
            while fh.state.name == "PROCESSING":
                time.sleep(UPLOAD_POLL)
                fh = genai.get_file(fh.name)
            if fh.state.name == "FAILED":
                raise ValueError(f"Gemini upload failed for {video_path}")
            print(f"  [upload] ok: {fh.name}")
            return fh
        except Exception as exc:
            if attempt == 2:
                raise
            wait = 4 * (2 ** attempt)   # 4s, 8s
            print(f"  [upload] error ({exc}), retrying in {wait}s...")
            time.sleep(wait)


def delete_video_safe(handle):
    try:
        genai.delete_file(handle.name)
        print(f"  [cleanup] deleted {handle.name}")
    except Exception as exc:
        print(f"  [cleanup] warning — could not delete {handle.name}: {exc}")


# ============================================================
# Gemini call  (enhanced _gemini_raw vs v2: adds DeadlineExceeded,
# ServiceUnavailable retry and request timeout)
# ============================================================

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


_RETRYABLE_ERRORS = (
    google.api_core.exceptions.ResourceExhausted,
    google.api_core.exceptions.DeadlineExceeded,
    google.api_core.exceptions.ServiceUnavailable,
)


def _gemini_raw(handle, prompt: str) -> str:
    """
    Single Gemini call with up to 3 retries on transient errors.
    Retries on: ResourceExhausted, DeadlineExceeded, ServiceUnavailable.
    Uses a 120-second per-request timeout.
    """
    for attempt in range(3):
        try:
            resp = _gemini_model.generate_content(
                [handle, prompt],
                generation_config={"temperature": 0.3},
                request_options={"timeout": 120},
            )
            return resp.text.strip()
        except _RETRYABLE_ERRORS as exc:
            if attempt == 2:
                raise
            wait = 15 * (attempt + 1)   # 15s, 30s
            print(f"  [gemini] {type(exc).__name__}, sleeping {wait}s...")
            time.sleep(wait)


def gemini_call(handle, agent_name: str, prompt: str,
                round_: int, feedback_suffix: str = "") -> dict:
    """
    Gemini call expecting JSON. Cache key includes agent_name, round_, feedback_suffix.
    On JSON parse failure, retries once with an explicit JSON instruction (different key).
    """
    key = _md5(handle.name, agent_name, prompt, f"r{round_}", feedback_suffix)
    cached = cache_get(key)
    if cached is not None:
        try:
            return json.loads(_strip_fences(cached))
        except Exception:
            pass   # stale/corrupt cache entry — fall through to live call

    raw = _gemini_raw(handle, prompt)
    cache_set(key, raw)

    try:
        return json.loads(_strip_fences(raw))
    except Exception:
        retry_prompt = (
            prompt
            + "\n\nYour previous response was not valid JSON. "
            "Output ONLY the JSON object with no markdown fences or explanation."
        )
        retry_key = _md5(handle.name, agent_name, prompt,
                         f"r{round_}", feedback_suffix, "json_retry")
        retry_cached = cache_get(retry_key)
        if retry_cached is not None:
            raw2 = retry_cached
        else:
            raw2 = _gemini_raw(handle, retry_prompt)
            cache_set(retry_key, raw2)
        return json.loads(_strip_fences(raw2))   # propagate if still broken


# ============================================================
# Utility  (copied verbatim from v2)
# ============================================================

def promts_to_text(promts: dict) -> str:
    labels = {
        "pedestrian":       "Pedestrian behavior",
        "goalRelated":      "Goal/destination",
        "roadUsersRelated": "Surrounding road users",
        "roadFactors":      "Road environment",
        "norms":            "Social norms",
    }
    parts = []
    for k, label in labels.items():
        val = promts.get(k, "").strip()
        if val:
            parts.append(f"  {label}: {val}")
    return "\n".join(parts) if parts else "  (none)"


def remap_path(p: str) -> str:
    """Translate /workspace/ Docker paths to /mnt/DATA/ host paths."""
    if p.startswith(_WORKSPACE_PREFIX):
        return _HOST_PREFIX + p[len(_WORKSPACE_PREFIX):]
    return p


# ============================================================
# Single agent
# ============================================================

def run_cot_agent(handle, ann: dict) -> dict:
    """
    Single Gemini CoT agent. Sees video + annotator description + promts.
    Does NOT see the annotator's answer (same leakage profile as v2).
    """
    description = ann.get("description", "").strip()
    promts_text = promts_to_text(ann.get("promts", {}))
    prompt = (
        f"{SYS_COT}\n\n"
        f"--- Annotator description ---\n{description}\n\n"
        f"--- Structured annotator observations ---\n{promts_text}"
    )
    result = gemini_call(handle, "cot_single", prompt, round_=0)
    return {
        "thinking": result.get("thinking", "").strip(),
        "answer":   result.get("answer", "").strip().lower(),
    }


# ============================================================
# Build LLaVA output record
# ============================================================

def build_llava_record(rec: dict, cot_result: dict) -> dict:
    ann    = rec.get("raw_annotation", {})
    meta   = rec.get("meta", {})
    intent = ann.get("intent", "")
    annotator_answer = "yes" if intent == "cross" else "no"

    predicted = cot_result["answer"]
    if predicted not in ("yes", "no"):
        predicted = ""
    matches = (predicted == annotator_answer) if predicted else False

    # Completion uses Gemini's thinking + the annotator's answer.
    # For records where matches_annotator=False, downstream SFT prep should filter.
    completion_text = (
        f"<thinking>\n{cot_result['thinking']}\n</thinking>\n"
        f"answer: {annotator_answer}"
    )

    return {
        "id":      rec["id"],
        "pair_id": rec["id"],
        "video":   remap_path(rec["video"]),
        "system":  SYSTEM_PROMPT,
        "prompt":  [{"role": "user", "content": PROMPT_TEXT}],
        "completion": [{"role": "assistant", "content": completion_text}],
        "answer":            annotator_answer,
        "aggregated_intent": meta.get("aggregated_intent", 0.5),
        "cot_meta": {
            "predicted_answer":  predicted,
            "matches_annotator": matches,
            "thinking":          cot_result["thinking"],
        },
        "meta": {
            "video_id":          meta.get("video_id", ""),
            "track_id":          meta.get("track_id", ""),
            "annotator_id":      meta.get("annotator_id", ""),
            "frame_span":        meta.get("frame_span", []),
            "num_frames":        90,
            "label_frame":       meta.get("label_frame", 0),
            "aggregated_intent": meta.get("aggregated_intent", 0.5),
        },
    }


# ============================================================
# Process one record
# ============================================================

def process_record(rec: dict) -> dict:
    ann        = rec.get("raw_annotation", {})
    intent     = ann.get("intent", "")
    video_path = remap_path(rec.get("video", ""))

    if intent not in ("cross", "not_cross"):
        raise ValueError(f"Unsupported intent: {intent!r}")
    if not video_path or not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    handle = None
    try:
        handle = upload_video(video_path)
        cot_result = run_cot_agent(handle, ann)
        return build_llava_record(rec, cot_result)
    finally:
        if handle is not None:
            delete_video_safe(handle)


# ============================================================
# Resume / done_ids  (adapted from v2: checks cot_meta instead of debate)
# ============================================================

def load_done_ids(output_path: str, rerun_failed: bool) -> set:
    """
    Returns the set of record IDs to skip.
    - rerun_failed=False: skip all IDs already in the output file.
    - rerun_failed=True: skip only records where cot_meta.matches_annotator=True;
      non-matching records get reprocessed and a new line appended
      (deduplicate downstream by keeping the last line per ID).
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
                obj = json.loads(line)
                rec_id = obj.get("id")
                if not rec_id:
                    continue
                if rerun_failed:
                    if obj.get("cot_meta", {}).get("matches_annotator", False):
                        done.add(rec_id)
                else:
                    done.add(rec_id)
            except Exception:
                pass
    return done


# ============================================================
# Main  (ThreadPoolExecutor + Lock for concurrent record processing)
# ============================================================

def main():
    with open(INPUT_JSONL, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"Input records: {len(records)}")

    if MAX_VIDEOS is not None:
        records = records[:MAX_VIDEOS]
        print(f"Limited to {len(records)} records (MAX_VIDEOS={MAX_VIDEOS})")

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    done_ids = load_done_ids(OUTPUT_JSONL, RERUN_FAILED)
    print(f"Already done: {len(done_ids)} (RERUN_FAILED={RERUN_FAILED})\n")

    pending = [r for r in records if r["id"] not in done_ids]
    skip_cnt = len(records) - len(pending)

    ok_cnt = err_cnt = 0
    written = []          # accumulate successful results for end-of-run summary
    write_lock = threading.Lock()

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f:

        def process_and_write(rec):
            result = process_record(rec)
            with write_lock:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                written.append(result)
            return result

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for rec in pending:
                futures[executor.submit(process_and_write, rec)] = rec
                if REQUEST_DELAY > 0:
                    time.sleep(REQUEST_DELAY)

            total = len(pending)
            done_count = 0
            for future in as_completed(futures):
                rec = futures[future]
                done_count += 1
                try:
                    result = future.result()
                    ok_cnt += 1
                    cm = result.get("cot_meta", {})
                    print(
                        f"  [{done_count}/{total}] {rec['id'][:60]}"
                        f"  pred={cm.get('predicted_answer', '?')}"
                        f"  match={cm.get('matches_annotator')}"
                    )
                except Exception as exc:
                    err_cnt += 1
                    print(f"  [{done_count}/{total}] ERROR {rec['id']}: {exc}")

    # ── End-of-run summary ──────────────────────────────────────────────
    matches_yes = sum(
        1 for r in written
        if r["cot_meta"]["matches_annotator"] and r["answer"] == "yes"
    )
    matches_no = sum(
        1 for r in written
        if r["cot_meta"]["matches_annotator"] and r["answer"] == "no"
    )
    total_written = len(written)
    total_match   = matches_yes + matches_no

    print(f"\n=== Done ===")
    print(f"OK: {ok_cnt} | Skipped: {skip_cnt} | Errors: {err_cnt}")
    print(
        f"matches_annotator: {total_match} / {total_written} "
        f"(yes: {matches_yes}, no: {matches_no})"
    )
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
