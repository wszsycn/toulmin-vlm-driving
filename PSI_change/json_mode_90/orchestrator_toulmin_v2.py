"""
orchestrator_toulmin_v2.py — Dual-Warrant Debate Architecture
==============================================================
A   (Gemini): pedestrian motion analysis       — runs once per video
B   (Gemini): scene context analysis           — runs once per video
C   (Gemini): grounds synthesizer              — in retry loop
D_A (Gemini): warrant generator (argues YES)   — in retry loop, independent of D_B
D_B (Gemini): warrant generator (argues NO)    — in retry loop, independent of D_A
E   (Gemini): debate judge                     — in retry loop; sees video
F   (GPT-5.4): critic (format + consistency)   — in retry loop
Orch (GPT-5.4): retry orchestrator             — decides retry targets

Key design properties vs v1:
- Two independent warrants are generated without knowing the annotator answer.
- The judge (video-aware Gemini) picks the winner, not a GPT agent.
- Agent C does NOT see the annotator answer (no label leakage into grounds).
- Both warrants are preserved in the output debate field.
- orch_decision is initialized before the retry loop (fixes v1 UnboundLocalError).
- Gemini upload has 3-retry exponential backoff (fixes v1 single-attempt failure).
- API keys are read from env vars only (never hardcoded).
"""

import json
import os
import re
import sys
import time
import hashlib
from pathlib import Path

import google.generativeai as genai
import google.api_core.exceptions
from openai import OpenAI


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
OPENAI_API_KEY = _require_env("OPENAI_API_KEY")

GEMINI_MODEL  = "gemini-3-flash-preview"
GPT_MODEL     = "gpt-5.4"

MAX_ROUNDS    = 3       # max orchestrator retry rounds per record
MAX_VIDEOS    = None    # None = process all records
REQUEST_DELAY = 2.0     # seconds between records
UPLOAD_POLL   = 8       # seconds between Gemini upload state polls
RERUN_FAILED  = False   # if True, reprocess records where agent_ok=False

_BASE        = "/workspace/PSI_change/json_mode_90"
INPUT_JSONL  = f"{_BASE}/psi_90f_1000.jsonl"
OUTPUT_JSONL = f"{_BASE}/trf_train/psi_toulmin_orchestrator_v2_1000.jsonl"
CACHE_DIR    = f"{_BASE}/agent_cache_orch_v2"


# ============================================================
# SFT output format constants (match to_llava_format.py)
# ============================================================

SYSTEM_PROMPT = (
    "You are an expert autonomous driving assistant specializing in pedestrian behavior analysis. "
    "When given a driving video, you analyze the TARGET pedestrian (highlighted by a green bounding box) "
    "and predict their crossing intention using the Toulmin argument structure:\n"
    "- grounds: concrete visual observations (posture, movement, position, gaze)\n"
    "- warrant: a general principle (physical law, social norm, or traffic rule) linking observations to the conclusion\n"
    "- answer: yes or no\n\n"
    "Always output exactly 3 lines in this format:\n"
    "grounds: <observations>\n"
    "warrant: <general rule>\n"
    "answer: <yes/no>\n"
    "Do not add any extra text before or after these 3 lines."
)

PROMPT_TEXT = (
    "Watch the full 90-frame video and predict: will the TARGET pedestrian "
    "attempt to cross in front of the vehicle in the next moment?"
)


# ============================================================
# Agent prompts
# ============================================================

SYS_MOTION = """\
You are a driving-scene analyst. Watch the video and describe ONLY the target
pedestrian's motion: body orientation, limb movement, head direction, walking
speed, and trajectory trend. Do NOT predict whether they will cross. Do NOT
describe the scene or other road users. 2-4 sentences. Output ONLY the
description; no preamble.

Return your analysis as JSON:
{"thinking": "<brief step-by-step CoT>", "output": "<2-4 sentence description>"}"""

SYS_SCENE = """\
You are a driving-scene analyst. Watch the video and describe the road context:
road type (intersection / straight road / parking lot), presence of crosswalk
or traffic signal, lane count, other vehicles/road users relevant to the target
pedestrian's path. Do NOT predict the pedestrian's intent. 2-4 sentences.
Output ONLY the description.

Return your analysis as JSON:
{"thinking": "<brief step-by-step CoT>", "output": "<2-4 sentence description>"}"""

SYS_GROUNDS = """\
You are a grounds synthesizer for a Toulmin argument. Combine the provided
motion observations, scene observations, and the annotator's description into
a single "grounds" paragraph.

Rules:
- Present tense only.
- 1-4 sentences.
- Visual observations only — no predictions ("will cross", "about to"), no
  guesses, no numeric distances ("2 meters").
- When motion/scene/annotator sources conflict, trust the annotator description
  (they observed ground truth).
- Do NOT include any warrant or answer.

Return strict JSON:
{"thinking": "<short CoT>", "grounds": "<final grounds>"}"""

SYS_WARRANT_YES = """\
You are a warrant generator. Your job is to argue, as compellingly as possible,
that the target pedestrian WILL CROSS, based on the video and the given grounds.

You are NOT told whether this is actually true. Treat this as a debate: build
the strongest defensible case for "yes".

Rules for the warrant:
- One sentence, 12-25 words.
- A general principle, not a specific observation (the specifics are in grounds).
- Must be one of: social_norm, physical_law, traffic_rule.
- Use probabilistic language ("tends to", "usually", "is likely to").
- No restating of grounds.

Return strict JSON:
{"thinking": "...", "warrant_type": "...", "warrant": "...", "confidence": <float in [0,1]>}"""

SYS_WARRANT_NO = """\
You are a warrant generator. Your job is to argue, as compellingly as possible,
that the target pedestrian will NOT CROSS, based on the video and the given grounds.

You are NOT told whether this is actually true. Treat this as a debate: build
the strongest defensible case for "no".

Rules for the warrant:
- One sentence, 12-25 words.
- A general principle, not a specific observation (the specifics are in grounds).
- Must be one of: social_norm, physical_law, traffic_rule.
- Use probabilistic language ("tends to", "usually", "is likely to").
- No restating of grounds.

Return strict JSON:
{"thinking": "...", "warrant_type": "...", "warrant": "...", "confidence": <float in [0,1]>}"""

SYS_JUDGE = """\
You are a debate judge. You will see the video, the grounds, and two competing
warrants: one arguing the pedestrian WILL cross, one arguing they will NOT.

Your job is to decide which warrant is better supported by the video and the
grounds. You are NOT told the ground-truth answer.

Judging criteria:
1. Which warrant's general principle is more clearly instantiated by what you
   see in the video?
2. Which warrant is more specific and falsifiable vs. vague?
3. Which warrant better aligns with the grounds without contradiction?

Margin meaning: 0 = complete toss-up, 1 = one side clearly dominates.

Return strict JSON:
{
  "thinking": "<comparison reasoning>",
  "winner": "<yes or no>",
  "margin": <float in [0, 1]>,
  "reason": "<1-2 sentence summary>"
}"""

SYS_CRITIC = """\
You are a critic. You will see: grounds, winning_warrant, answer (winner's side).

Evaluate ONLY these two properties:
1. format_ok:
   - warrant is 12-25 words
   - no predictions / future-tense assertions in grounds
   - warrant uses probabilistic language
   - warrant_type is one of social_norm / physical_law / traffic_rule
2. consistency_ok:
   - grounds and winning_warrant do not contradict each other
   - winning_warrant actually supports the winning answer direction

Do NOT re-judge the debate. Do NOT rewrite grounds or warrant.

Return strict JSON (no markdown fences):
{"format_ok": <bool>, "consistency_ok": <bool>, "ok": <bool>, "reason": "<short explanation if not ok>"}"""

SYS_ORCH = """\
You are a retry orchestrator. The critic rejected the record. Decide which
agents need to rerun and what feedback each should receive.

You may choose any subset of: retry_grounds, retry_warrant_yes,
retry_warrant_no, retry_judge.

Minimize reruns — only rerun what is actually broken. Write concrete, actionable
feedback pointing at the specific flaw.

Return strict JSON (no markdown fences):
{
  "retry_grounds": <bool>,
  "retry_warrant_yes": <bool>,
  "retry_warrant_no": <bool>,
  "retry_judge": <bool>,
  "grounds_feedback": "<str>",
  "warrant_yes_feedback": "<str>",
  "warrant_no_feedback": "<str>",
  "judge_feedback": "<str>"
}"""


# ============================================================
# Initialization
# ============================================================

genai.configure(api_key=GEMINI_API_KEY)
_gemini_model = genai.GenerativeModel(model_name=GEMINI_MODEL)
_openai_client = OpenAI(api_key=OPENAI_API_KEY)
os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# Cache
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
# Video upload with retry
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
# Gemini call with rate-limit retry and JSON retry
# ============================================================

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _gemini_raw(handle, prompt: str) -> str:
    """Single Gemini call; retries once on ResourceExhausted."""
    for attempt in range(2):
        try:
            resp = _gemini_model.generate_content(
                [handle, prompt],
                generation_config={"temperature": 0.3},
            )
            return resp.text.strip()
        except google.api_core.exceptions.ResourceExhausted:
            if attempt == 1:
                raise
            print("  [gemini] rate limit hit, sleeping 15s...")
            time.sleep(15)


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
        # one retry with explicit JSON instruction; different cache key
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
# GPT call
# ============================================================

def gpt_call(agent_name: str, messages: list, temperature: float = 0.1) -> dict:
    """
    GPT call expecting JSON. Uses json_object response format.
    Cache key: md5(messages + model + temperature + agent_name).
    """
    key = _md5(
        json.dumps(messages, sort_keys=True, ensure_ascii=False),
        GPT_MODEL,
        str(temperature),
        agent_name,
    )
    cached = cache_get(key)
    if cached is not None:
        try:
            return json.loads(_strip_fences(cached))
        except Exception:
            pass

    for attempt in range(3):
        try:
            resp = _openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            cache_set(key, raw)
            return json.loads(_strip_fences(raw))
        except Exception as exc:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


# ============================================================
# Utility
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


def _with_feedback(prompt: str, feedback: str) -> str:
    if not feedback or not feedback.strip():
        return prompt
    return prompt + f"\n\n<feedback>\n{feedback.strip()}\n</feedback>"


def _safe_str(obj, key: str, default: str = "") -> str:
    if isinstance(obj, dict):
        return str(obj.get(key, default))
    return default


# ============================================================
# Agent wrappers
# ============================================================

def run_agent_motion(handle) -> dict:
    print("    [A] pedestrian motion...")
    result = gemini_call(handle, "A_motion", SYS_MOTION, round_=0)
    return {
        "thinking": result.get("thinking", ""),
        "output":   result.get("output", ""),
    }


def run_agent_scene(handle) -> dict:
    print("    [B] scene context...")
    result = gemini_call(handle, "B_scene", SYS_SCENE, round_=0)
    return {
        "thinking": result.get("thinking", ""),
        "output":   result.get("output", ""),
    }


def run_agent_grounds(handle, motion: dict, scene: dict,
                      ann: dict, feedback: str, round_: int) -> dict:
    """
    Agent C: synthesizes grounds from motion + scene + annotator text.
    Does NOT receive the annotator answer — avoids label leakage into grounds.
    """
    print(f"    [C] grounds (round {round_})...")
    description = ann.get("description", "").strip()
    promts_text = promts_to_text(ann.get("promts", {}))

    prompt = (
        f"{SYS_GROUNDS}\n\n"
        f"--- Annotator description ---\n{description}\n\n"
        f"--- Structured annotator observations ---\n{promts_text}\n\n"
        f"--- Pedestrian motion (video) ---\n{motion.get('output', '')}\n\n"
        f"--- Scene context (video) ---\n{scene.get('output', '')}"
    )
    prompt = _with_feedback(prompt, feedback)

    result = gemini_call(handle, "C_grounds", prompt, round_,
                         feedback_suffix=feedback[:40])
    return {
        "thinking": result.get("thinking", ""),
        "grounds":  result.get("grounds", ""),
    }


def run_agent_warrant(handle, grounds: dict, side: str,
                      feedback: str, round_: int) -> dict:
    """
    Agent D_A (side="yes") or D_B (side="no").
    Two independent calls with distinct cache keys (agent_name = "D_yes" / "D_no").
    Neither call is told the annotator answer.
    """
    assert side in ("yes", "no"), f"side must be 'yes' or 'no', got {side!r}"
    print(f"    [D_{side}] warrant ({side}) (round {round_})...")

    sys_prompt = SYS_WARRANT_YES if side == "yes" else SYS_WARRANT_NO
    agent_name = f"D_{side}"
    grounds_text = _safe_str(grounds, "grounds")

    prompt = f"{sys_prompt}\n\n--- Grounds ---\n{grounds_text}"
    prompt = _with_feedback(prompt, feedback)

    result = gemini_call(handle, agent_name, prompt, round_,
                         feedback_suffix=feedback[:40])
    return {
        "thinking":     result.get("thinking", ""),
        "warrant_type": result.get("warrant_type", ""),
        "warrant":      result.get("warrant", ""),
        "confidence":   float(result.get("confidence", 0.5)),
    }


def run_agent_judge(handle, grounds: dict, w_yes: dict, w_no: dict,
                    feedback: str, round_: int) -> dict:
    """
    Agent E: Gemini debate judge. Sees video + grounds + both warrants.
    NOT told the annotator answer.
    """
    print(f"    [E] judge (round {round_})...")
    grounds_text = _safe_str(grounds, "grounds")
    warrant_yes  = _safe_str(w_yes, "warrant")
    warrant_no   = _safe_str(w_no, "warrant")

    prompt = (
        f"{SYS_JUDGE}\n\n"
        f"--- Grounds ---\n{grounds_text}\n\n"
        f"--- Warrant A (arguing WILL cross) ---\n{warrant_yes}\n\n"
        f"--- Warrant B (arguing will NOT cross) ---\n{warrant_no}"
    )
    prompt = _with_feedback(prompt, feedback)

    result = gemini_call(handle, "E_judge", prompt, round_,
                         feedback_suffix=feedback[:40])

    winner = result.get("winner", "").strip().lower()
    if winner not in ("yes", "no"):
        winner = "yes"   # fallback; critic will flag if wrong
    return {
        "thinking": result.get("thinking", ""),
        "winner":   winner,
        "margin":   float(result.get("margin", 0.5)),
        "reason":   result.get("reason", ""),
    }


def run_agent_critic(grounds_text: str, winner_warrant: str,
                     winner: str) -> dict:
    """
    Agent F: GPT critic. Evaluates format_ok and consistency_ok only.
    Does NOT re-judge the debate. Receives judge["winner"], not the annotator answer.
    """
    print("    [F] critic...")
    messages = [
        {"role": "system", "content": SYS_CRITIC},
        {"role": "user", "content": (
            f"Grounds: {grounds_text}\n"
            f"Winning warrant: {winner_warrant}\n"
            f"Winner (answer direction): {winner}"
        )},
    ]
    result = gpt_call("F_critic", messages, temperature=0.1)
    return {
        "format_ok":      bool(result.get("format_ok", False)),
        "consistency_ok": bool(result.get("consistency_ok", False)),
        "ok":             bool(result.get("ok", False)),
        "reason":         result.get("reason", ""),
    }


def run_orchestrator(critic_reason: str, grounds: dict,
                     w_yes: dict, w_no: dict, judge: dict) -> dict:
    """Orchestrator: GPT-5.4. Decides which of {C, D_A, D_B, E} to retry."""
    print("    [Orch] deciding retry strategy...")
    messages = [
        {"role": "system", "content": SYS_ORCH},
        {"role": "user", "content": (
            f"Critic reason: {critic_reason}\n"
            f"Current grounds: {_safe_str(grounds, 'grounds')}\n"
            f"Current warrant_yes: {_safe_str(w_yes, 'warrant')}\n"
            f"Current warrant_no: {_safe_str(w_no, 'warrant')}\n"
            f"Judge reason: {_safe_str(judge, 'reason')}"
        )},
    ]
    result = gpt_call("Orch", messages, temperature=0.1)
    return {
        "retry_grounds":        bool(result.get("retry_grounds", False)),
        "retry_warrant_yes":    bool(result.get("retry_warrant_yes", False)),
        "retry_warrant_no":     bool(result.get("retry_warrant_no", False)),
        "retry_judge":          bool(result.get("retry_judge", False)),
        "grounds_feedback":     result.get("grounds_feedback", ""),
        "warrant_yes_feedback": result.get("warrant_yes_feedback", ""),
        "warrant_no_feedback":  result.get("warrant_no_feedback", ""),
        "judge_feedback":       result.get("judge_feedback", ""),
    }


# ============================================================
# Build LLaVA output record
# ============================================================

def build_llava_record(rec: dict, motion: dict, scene: dict,
                       grounds: dict, w_yes: dict, w_no: dict,
                       judge: dict, rounds_needed: int,
                       agent_ok: bool,
                       final_failure_reason=None) -> dict:
    meta = rec.get("meta", {})
    ann  = rec.get("raw_annotation", {})
    intent = ann.get("intent", "")
    annotator_answer = "yes" if intent == "cross" else "no"

    grounds_text  = _safe_str(grounds, "grounds") if grounds else ""
    warrant_yes   = _safe_str(w_yes, "warrant") if w_yes else ""
    warrant_no    = _safe_str(w_no, "warrant") if w_no else ""
    winner        = judge.get("winner", "") if judge else ""
    winning_warrant = warrant_yes if winner == "yes" else warrant_no

    # matches_annotator: did the judge agree with the human annotator?
    # Records with matches_annotator=False should be filtered before SFT training.
    matches_annotator = (winner == annotator_answer) if winner else False

    # Completion uses the winning warrant with the annotator's answer.
    # For matches_annotator=True records this is fully coherent for SFT.
    # For matches_annotator=False records, downstream consumers should filter.
    completion_text = (
        f"grounds: {grounds_text}\n"
        f"warrant: {winning_warrant}\n"
        f"answer: {annotator_answer}"
    )

    return {
        "id":      rec["id"],
        "pair_id": rec["id"],
        "video":   rec["video"],
        "system":  SYSTEM_PROMPT,
        "prompt":  [{"role": "user", "content": PROMPT_TEXT}],
        "completion": [{"role": "assistant", "content": completion_text}],
        "answer":            annotator_answer,
        "aggregated_intent": meta.get("aggregated_intent", 0.5),
        "debate": {
            "grounds":              grounds_text,
            "warrant_yes":          warrant_yes,
            "warrant_no":           warrant_no,
            "winner":               winner,
            "margin":               judge.get("margin", 0.0) if judge else 0.0,
            "judge_reason":         judge.get("reason", "") if judge else "",
            "matches_annotator":    matches_annotator,
            "rounds_needed":        rounds_needed,
            "agent_ok":             agent_ok,
            "final_failure_reason": final_failure_reason,
            "cot": {
                "motion":          motion.get("thinking", "") if motion else "",
                "scene":           scene.get("thinking", "") if scene else "",
                "grounds_cot":     grounds.get("thinking", "") if grounds else "",
                "warrant_yes_cot": w_yes.get("thinking", "") if w_yes else "",
                "warrant_no_cot":  w_no.get("thinking", "") if w_no else "",
                "judge_cot":       judge.get("thinking", "") if judge else "",
            },
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
    video_path = rec.get("video", "")

    if intent not in ("cross", "not_cross"):
        raise ValueError(f"Unsupported intent: {intent!r}")
    if not video_path or not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    handle = None
    try:
        handle = upload_video(video_path)

        # A and B run once per video — not subject to orchestrator retry
        motion = run_agent_motion(handle)
        scene  = run_agent_scene(handle)

        # Initialize orch_decision before the loop to avoid UnboundLocalError
        # (v1 bug: orch_decision was first assigned inside the loop body)
        orch_decision = {
            "retry_grounds":        True,
            "retry_warrant_yes":    True,
            "retry_warrant_no":     True,
            "retry_judge":          True,
            "grounds_feedback":     "",
            "warrant_yes_feedback": "",
            "warrant_no_feedback":  "",
            "judge_feedback":       "",
        }

        grounds = w_yes = w_no = judge = critic = None

        for round_ in range(1, MAX_ROUNDS + 1):
            print(f"  === Round {round_}/{MAX_ROUNDS} ===")

            if orch_decision["retry_grounds"] or grounds is None:
                grounds = run_agent_grounds(
                    handle, motion, scene, ann,
                    orch_decision["grounds_feedback"], round_,
                )
            if orch_decision["retry_warrant_yes"] or w_yes is None:
                w_yes = run_agent_warrant(
                    handle, grounds, "yes",
                    orch_decision["warrant_yes_feedback"], round_,
                )
            if orch_decision["retry_warrant_no"] or w_no is None:
                w_no = run_agent_warrant(
                    handle, grounds, "no",
                    orch_decision["warrant_no_feedback"], round_,
                )
            if orch_decision["retry_judge"] or judge is None:
                judge = run_agent_judge(
                    handle, grounds, w_yes, w_no,
                    orch_decision["judge_feedback"], round_,
                )

            winner_warrant = w_yes if judge["winner"] == "yes" else w_no
            critic = run_agent_critic(
                grounds["grounds"],
                winner_warrant["warrant"],
                judge["winner"],
            )

            if critic["ok"]:
                print(f"  [ok] critic passed on round {round_}")
                return build_llava_record(
                    rec, motion, scene, grounds, w_yes, w_no, judge,
                    rounds_needed=round_, agent_ok=True,
                )

            if round_ < MAX_ROUNDS:
                orch_decision = run_orchestrator(
                    critic["reason"], grounds, w_yes, w_no, judge,
                )
            else:
                print(f"  [fail] all {MAX_ROUNDS} rounds exhausted")

        # Max rounds reached without critic approval
        return build_llava_record(
            rec, motion, scene, grounds, w_yes, w_no, judge,
            rounds_needed=MAX_ROUNDS, agent_ok=False,
            final_failure_reason=critic["reason"] if critic else "max rounds reached",
        )

    finally:
        if handle is not None:
            delete_video_safe(handle)


# ============================================================
# Resume / done_ids
# ============================================================

def load_done_ids(output_path: str, rerun_failed: bool) -> set:
    """
    Returns the set of record IDs to skip.
    - rerun_failed=False: skip all IDs already in the output file.
    - rerun_failed=True: skip only IDs where agent_ok=True; failed records
      get reprocessed and a new line appended (deduplicate downstream by
      keeping the last line per ID).
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
                    if obj.get("debate", {}).get("agent_ok", False):
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
    with open(INPUT_JSONL, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"Input records: {len(records)}")

    if MAX_VIDEOS is not None:
        records = records[:MAX_VIDEOS]
        print(f"Limited to {len(records)} records (MAX_VIDEOS={MAX_VIDEOS})")

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    done_ids = load_done_ids(OUTPUT_JSONL, RERUN_FAILED)
    print(f"Already done: {len(done_ids)} (RERUN_FAILED={RERUN_FAILED})\n")

    ok_cnt = skip_cnt = err_cnt = 0

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f:
        for i, rec in enumerate(records, 1):
            rec_id = rec["id"]
            if rec_id in done_ids:
                skip_cnt += 1
                continue

            print(f"\n[{i}/{len(records)}] {rec_id}")
            try:
                result = process_record(rec)
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                ok_cnt += 1
                d = result.get("debate", {})
                winner = d.get("winner", "")
                print(f"  grounds:           {d.get('grounds', '')[:80]}...")
                print(f"  winning warrant:   {d.get('warrant_yes' if winner == 'yes' else 'warrant_no', '')}")
                print(f"  winner:            {winner}  margin={d.get('margin', 0):.2f}")
                print(f"  matches_annotator: {d.get('matches_annotator')}")
            except Exception as exc:
                print(f"  [error] {exc}")
                err_cnt += 1

            if i < len(records):
                time.sleep(REQUEST_DELAY)

    print(f"\n=== Done ===")
    print(f"OK: {ok_cnt} | Skipped: {skip_cnt} | Errors: {err_cnt}")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
