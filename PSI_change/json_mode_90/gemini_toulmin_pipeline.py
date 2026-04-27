"""
Multi-modal Toulmin Pipeline
Step 1 (Gemini): 视频 + PSI文字信息 → 高质量 grounds + answer
Step 2 (GPT-4.1): grounds + answer → warrant
Step 3 (GPT-4.1): Critic 自我反思 → 校验/修正 grounds + warrant
"""

import json
import os
import re
import time
import hashlib
from pathlib import Path

import google.generativeai as genai
from openai import OpenAI

# ============================================================
# 配置区
# ============================================================

GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

INPUT_JSONL     = "/workspace/PSI_change/json_mode_90/trf_train/psi_90f_toulmin_1000.jsonl"
OUTPUT_JSONL    = "/workspace/PSI_change/json_mode_90/trf_train/psi_toulmin_gemini.jsonl"
CACHE_DIR       = "/workspace/PSI_change/json_mode_90/agent_cache_gemini"

GEMINI_MODEL    = "gemini-2.5-flash"   # 或 "gemini-3-flash-preview" 你用成功的那个
GPT_MODEL       = "gpt-5.4"

MAX_VIDEOS      = 5          # 设为 5 先跑几个试试，None 跑全部
REQUEST_DELAY   = 2.0           # 每条记录之间的间隔（秒）
UPLOAD_POLL_SEC = 8             # 等待 Gemini 视频处理的轮询间隔

# ============================================================
# 初始化
# ============================================================

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(model_name=GEMINI_MODEL)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# 缓存（按输入内容 hash，避免重复调用）
# ============================================================

def cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def cache_get(key: str):
    p = Path(CACHE_DIR) / f"{key}.txt"
    return p.read_text(encoding="utf-8") if p.exists() else None

def cache_set(key: str, value: str):
    p = Path(CACHE_DIR) / f"{key}.txt"
    p.write_text(value, encoding="utf-8")


# ============================================================
# 工具函数
# ============================================================

def promts_to_text(promts: dict) -> str:
    """把结构化 promts 拼成可读文本"""
    parts = []
    labels = {
        "pedestrian":       "Pedestrian behavior",
        "goalRelated":      "Goal/destination",
        "roadUsersRelated": "Surrounding road users",
        "roadFactors":      "Road environment",
        "norms":            "Social norms",
    }
    for key, label in labels.items():
        val = promts.get(key, "").strip()
        if val:
            parts.append(f"{label}: {val}")
    return "\n".join(parts)

def upload_video(video_path: str):
    """上传视频到 Gemini File API，等待处理完成"""
    video_file = genai.upload_file(path=video_path)
    while video_file.state.name == "PROCESSING":
        time.sleep(UPLOAD_POLL_SEC)
        video_file = genai.get_file(video_file.name)
    if video_file.state.name == "FAILED":
        raise ValueError(f"Gemini video processing failed: {video_path}")
    return video_file

def gpt_call(messages: list, temperature: float = 0.3) -> str:
    """调用 GPT-4.1，带重试"""
    for attempt in range(3):
        try:
            resp = openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

def gpt_cached(messages: list, temperature: float = 0.3) -> str:
    key = cache_key(json.dumps(messages, ensure_ascii=False))
    cached = cache_get(key)
    if cached is not None:
        return cached
    result = gpt_call(messages, temperature)
    cache_set(key, result)
    return result


# ============================================================
# Agent 1: Gemini Visual Grounder
# 输入：视频 + description + promts + answer
# 输出：高质量 grounds（纯视觉事实）
# ============================================================

GEMINI_SYSTEM = """\
You are an expert analyst for pedestrian crossing intention in driving videos.
The TARGET pedestrian is highlighted with a GREEN bounding box.

Your task: generate high-quality GROUNDS for a Toulmin argument.

GROUNDS = concrete, observable visual evidence from the video that supports the answer.

Rules for GROUNDS:
- Describe only what is VISUALLY OBSERVABLE in the video frames
- Integrate both the video observations AND the provided text annotations
- Use factual, present-tense language ("The pedestrian is...", "The pedestrian faces...")
- Include: body posture, gaze direction, movement, position relative to road/curb, momentum
- Do NOT include: predictions, distances in meters/feet, subjective judgments
- Do NOT say "will cross" or "won't cross" — describe observable facts only
- Length: 2-4 concise sentences

Output ONLY the grounds text, nothing else."""

def agent_gemini_grounder(video_path: str, description: str, promts: dict, answer: str) -> str:
    """Gemini: 视频 + PSI文字 → grounds"""
    promts_text = promts_to_text(promts)
    user_text = f"""Annotator description: {description}

Structured observations:
{promts_text}

Crossing answer: {"YES - the pedestrian will cross" if answer == "yes" else "NO - the pedestrian will NOT cross"}

Watch the video carefully and generate GROUNDS:"""

    # 检查缓存（用视频路径+文字内容作为key）
    key = cache_key(video_path + user_text)
    cached = cache_get(key)
    if cached is not None:
        print(f"  [Gemini cache hit]")
        return cached

    # 上传视频
    video_file = upload_video(video_path)
    try:
        response = gemini_model.generate_content(
            [video_file, GEMINI_SYSTEM + "\n\n" + user_text],
            generation_config={"temperature": 0.2}
        )
        result = response.text.strip()
    finally:
        genai.delete_file(video_file.name)

    cache_set(key, result)
    return result


# ============================================================
# Agent 2: GPT Warrant Generator
# 输入：grounds + answer
# 输出：通用规则 warrant（12-25词）
# ============================================================

WARRANT_SYSTEM = """\
You are an expert in Toulmin argumentation for autonomous driving scenarios.
Your task: given GROUNDS (observed visual facts) and the crossing/not crossing ANSWER, generate a WARRANT.

The WARRANT is the GENERAL RULE or PRINCIPLE that explains WHY the grounds logically
support the claim. It is NOT a restatement of the grounds.

The warrant must come from one of these three categories:
  1. SOCIAL NORM — shared behavioral expectations between pedestrians and drivers
  2. PHYSICAL LAW — momentum, trajectory, body mechanics
  3. TRAFFIC RULE — codified road conventions

Rules:
- General principle, not a description of this specific scene
- Probabilistic language: "typically", "tends to", "often", "usually"
- Length: 12-25 words exactly
- NO specific scene details (no locations, distances, traffic light states)
- NO repetition of words or phrases from the grounds
- NO explicit answer (yes/no, cross/not cross, will/won't)
- NOT about driver behavior — about pedestrian behavior patterns

Output only the warrant sentence, nothing else."""

def agent_warrant(grounds: str, answer: str) -> str:
    user = f"""Grounds: {grounds}
Answer: {answer}

Generate WARRANT:"""
    return gpt_cached([
        {"role": "system", "content": WARRANT_SYSTEM},
        {"role": "user",   "content": user},
    ])


# ============================================================
# Agent 3: GPT Critic (自我反思)
# 输入：grounds + warrant + answer
# 输出：修正后的 JSON
# ============================================================

CRITIC_SYSTEM = """\
You are a strict Toulmin argument quality critic for autonomous driving.
Review GROUNDS and WARRANT, correct if needed, and return JSON.

Checklist:
GROUNDS:
  - Only observable visual facts? (no predictions, no distances in units)
  - Present-tense factual language?

WARRANT:
  - General rule (physical law / social norm / traffic rule)?
  - About PEDESTRIAN behavior/tendency — NOT about what the driver should do?
  - Probabilistic language (typically/tends to/often/usually)?
  - Does NOT repeat words or phrases from the grounds?
  - Does NOT state the answer explicitly (no yes/no/will cross/won't cross)?
  - 12-25 words?

CONSISTENCY — MOST IMPORTANT:
  - answer=yes: warrant must support WHY the pedestrian WILL cross
  - answer=no:  warrant must support WHY the pedestrian will NOT cross
  - If direction contradicts the answer, REWRITE the warrant entirely.

Output ONLY valid JSON, no markdown fences:
{"grounds": "...", "warrant": "...", "ok": true/false, "reason": "..."}"""

def agent_critic(grounds: str, warrant: str, answer: str) -> dict:
    user = f"""Grounds: {grounds}
Warrant: {warrant}
Answer: {answer}

Review and correct:"""
    raw = gpt_cached([
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user",   "content": user},
    ], temperature=0.1)

    try:
        clean = re.sub(r"```[a-z]*\n?", "", raw).strip()
        return json.loads(clean)
    except Exception:
        return {
            "grounds": grounds,
            "warrant": warrant,
            "ok": False,
            "reason": f"parse error: {raw[:120]}"
        }


# ============================================================
# Main
# ============================================================

def main():
    # 读输入
    with open(INPUT_JSONL, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"Total records: {len(records)}")

    if MAX_VIDEOS is not None:
        records = records[:MAX_VIDEOS]
        print(f"Limited to: {len(records)} records")

    # 断点续跑
    done_ids = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["id"])
    print(f"Already done: {len(done_ids)}\n")

    ok_cnt = skip_cnt = err_cnt = 0

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f:
        for i, rec in enumerate(records, 1):
            rec_id = rec["id"]

            if rec_id in done_ids:
                skip_cnt += 1
                continue

            print(f"[{i}/{len(records)}] {rec_id}")

            raw_ann     = rec.get("raw_annotation", {})
            description = raw_ann.get("description", "").strip()
            promts      = raw_ann.get("promts", {})
            intent      = raw_ann.get("intent", "")
            video_path  = rec.get("video", "")

            # answer
            if intent == "cross":
                answer = "yes"
            elif intent == "not_cross":
                answer = "no"
            else:
                print(f"  ⚠️  Skipping intent='{intent}'")
                err_cnt += 1
                continue

            if not video_path or not Path(video_path).exists():
                print(f"  ❌ Video not found: {video_path}")
                err_cnt += 1
                continue

            try:
                # Agent 1: Gemini grounder
                print(f"  [1/3] Gemini grounding...")
                grounds = agent_gemini_grounder(video_path, description, promts, answer)
                print(f"  grounds: {grounds[:80]}...")

                # Agent 2: Warrant generator
                print(f"  [2/3] GPT warrant...")
                warrant = agent_warrant(grounds, answer)
                print(f"  warrant: {warrant}")

                # Agent 3: Critic
                print(f"  [3/3] GPT critic...")
                critic_result = agent_critic(grounds, warrant, answer)
                grounds = critic_result.get("grounds", grounds).strip()
                warrant = critic_result.get("warrant", warrant).strip()
                ok_flag = critic_result.get("ok", False)
                print(f"  ✅ ok={ok_flag} | answer={answer}")

                # 输出记录
                out = dict(rec)
                out["toulmin"] = {
                    "grounds": grounds,
                    "warrant": warrant,
                    "answer":  answer,
                    "critic_ok": ok_flag,
                    "critic_reason": critic_result.get("reason", ""),
                }
                out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
                out_f.flush()
                ok_cnt += 1

            except Exception as e:
                print(f"  ❌ Error: {e}")
                err_cnt += 1

            if i < len(records):
                time.sleep(REQUEST_DELAY)

    print(f"\n=== Done ===")
    print(f"OK: {ok_cnt} | Skipped: {skip_cnt} | Error: {err_cnt}")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()