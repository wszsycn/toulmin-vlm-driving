"""
Orchestrator Multi-Agent Toulmin Pipeline
==========================================
Agent A (Gemini): 行人动作分析         — CoT
Agent B (Gemini): 场景环境分析         — CoT
Agent C (Gemini): Grounds Synthesizer  — CoT，复用同一 file_handle
Agent D (GPT-4.1): Warrant Generator   — CoT
Agent E (GPT-4.1): Critic              — 结构化 JSON 输出
Orchestrator (GPT-4.1): 控制迭代，最多 MAX_ROUNDS 轮

视频只上传一次，所有 Gemini agent 复用同一 file handle。
CoT 思考过程保存到输出 jsonl，但不传给下游 agent。
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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

INPUT_JSONL  = "/workspace/PSI_change/json_mode_90/trf_train/psi_90f_toulmin_1000.jsonl"
OUTPUT_JSONL = "/workspace/PSI_change/json_mode_90/trf_train/psi_toulmin_orchestrator.jsonl"
CACHE_DIR    = "/workspace/PSI_change/json_mode_90/agent_cache_orch"

GEMINI_MODEL = "gemini-2.5-flash"   # 换成你能用的模型
GPT_MODEL    = "gpt-5.4"

MAX_VIDEOS   = 5        # 先试跑，None 跑全部
MAX_ROUNDS   = 3        # Orchestrator 最多重试轮数
REQUEST_DELAY = 2.0     # 每条记录之间的间隔（秒）
UPLOAD_POLL  = 8        # 等待视频上传完成的轮询间隔（秒）


# ============================================================
# 初始化
# ============================================================

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(model_name=GEMINI_MODEL)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# 缓存
# ============================================================

def cache_key(*parts: str) -> str:
    return hashlib.md5("|".join(parts).encode()).hexdigest()

def cache_get(key: str):
    p = Path(CACHE_DIR) / f"{key}.txt"
    return p.read_text(encoding="utf-8") if p.exists() else None

def cache_set(key: str, value: str):
    (Path(CACHE_DIR) / f"{key}.txt").write_text(value, encoding="utf-8")


# ============================================================
# 工具函数
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
    for key, label in labels.items():
        val = promts.get(key, "").strip()
        if val:
            parts.append(f"  {label}: {val}")
    return "\n".join(parts) if parts else "  (none)"

def extract_tag(text: str, tag: str) -> str:
    """从 <tag>...</tag> 中提取内容，去首尾空白"""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()

def upload_video(video_path: str):
    """上传视频，等待处理完成，返回 file 对象"""
    print(f"  📤 上传视频: {Path(video_path).name}")
    f = genai.upload_file(path=video_path)
    while f.state.name == "PROCESSING":
        time.sleep(UPLOAD_POLL)
        f = genai.get_file(f.name)
    if f.state.name == "FAILED":
        raise ValueError(f"Video upload failed: {video_path}")
    print(f"  ✅ 上传完成: {f.name}")
    return f

def gemini_call(file_handle, prompt: str, cache_suffix: str = "") -> str:
    """调用 Gemini，复用已上传的 file_handle"""
    key = cache_key(file_handle.name, prompt, cache_suffix)
    cached = cache_get(key)
    if cached is not None:
        return cached
    resp = gemini_model.generate_content(
        [file_handle, prompt],
        generation_config={"temperature": 0.3}
    )
    result = resp.text.strip()
    cache_set(key, result)
    return result

def gpt_call(messages: list, temperature: float = 0.3) -> str:
    """调用 GPT-4.1，带简单重试"""
    key = cache_key(json.dumps(messages, ensure_ascii=False), str(temperature))
    cached = cache_get(key)
    if cached is not None:
        return cached
    for attempt in range(3):
        try:
            resp = openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                temperature=temperature,
            )
            result = resp.choices[0].message.content.strip()
            cache_set(key, result)
            return result
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


# ============================================================
# Agent A: 行人动作分析 (Gemini + CoT)
# ============================================================

AGENT_A_PROMPT = """\
You are analyzing a driving video to understand pedestrian behavior.
The TARGET pedestrian is highlighted with a GREEN bounding box.

<thinking>
Think step by step about the TARGET pedestrian:
1. What is their body posture? (upright, leaning, crouching)
2. Which direction are they facing?
3. Are they moving? If so, in which direction and at what pace?
4. What is their momentum — are they accelerating, decelerating, or steady?
5. What are their limbs doing? (arms swinging, legs striding)
6. Is there any hesitation or pausing visible?
</thinking>

<output>
Summarize ONLY the observable pedestrian motion facts in 2-3 sentences.
Use present tense. No predictions. No distances in meters/feet.
</output>

Respond using exactly the <thinking> and <output> XML tags above."""

def agent_A(file_handle) -> dict:
    print("    [Agent A] 行人动作分析...")
    raw = gemini_call(file_handle, AGENT_A_PROMPT, "agentA")
    return {
        "thinking": extract_tag(raw, "thinking"),
        "output":   extract_tag(raw, "output"),
        "raw":      raw,
    }


# ============================================================
# Agent B: 场景环境分析 (Gemini + CoT)
# ============================================================

AGENT_B_PROMPT = """\
You are analyzing a driving video to understand the road environment.
The TARGET pedestrian is highlighted with a GREEN bounding box.

<thinking>
Think step by step about the scene context:
1. What type of road is this? (intersection, crosswalk, mid-block, etc.)
2. Is there a crosswalk or traffic signal visible?
3. Where is the pedestrian positioned relative to the road? (curb, lane, median)
4. Are there other vehicles? How close and moving in which direction?
5. Are there other pedestrians? What are they doing?
6. What are the road and lighting conditions?
</thinking>

<output>
Summarize ONLY the observable scene facts relevant to crossing intention in 2-3 sentences.
Use present tense. No predictions.
</output>

Respond using exactly the <thinking> and <output> XML tags above."""

def agent_B(file_handle) -> dict:
    print("    [Agent B] 场景环境分析...")
    raw = gemini_call(file_handle, AGENT_B_PROMPT, "agentB")
    return {
        "thinking": extract_tag(raw, "thinking"),
        "output":   extract_tag(raw, "output"),
        "raw":      raw,
    }


# ============================================================
# Agent C: Grounds Synthesizer (Gemini + CoT)
# ============================================================

def agent_C_prompt(description: str, promts_text: str,
                   motion_obs: str, scene_obs: str, answer: str,
                   feedback: str = "") -> str:
    feedback_section = f"\nPrevious attempt feedback (fix these issues):\n{feedback}\n" if feedback else ""
    return f"""\
You are synthesizing visual evidence into high-quality Toulmin GROUNDS.
The TARGET pedestrian is highlighted with a GREEN bounding box.

You have the following information:
--- Annotator text description ---
{description}

--- Structured annotator observations ---
{promts_text}

--- Visual motion analysis (from video) ---
{motion_obs}

--- Visual scene analysis (from video) ---
{scene_obs}

--- Crossing answer ---
{"YES — the pedestrian WILL cross" if answer == "yes" else "NO — the pedestrian will NOT cross"}
{feedback_section}
<thinking>
Cross-reference the video observations with the annotator text:
1. Which facts appear in BOTH video and text? (most reliable, use these first)
2. Which facts come only from the video? (supplementary visual detail)
3. Are there any contradictions between text and video?
   - If yes: ALWAYS trust the PSI annotator text over the video analysis.
     The annotator watched the full video carefully; video analysis may be wrong.
   - State the contradiction explicitly before deciding what to keep.
4. What is the single strongest visual signal supporting the answer?
5. Only add video-only facts if they do NOT contradict the annotator text.
</thinking>

<output>
Write the final GROUNDS in 2-4 sentences.
Rules:
- Present-tense observable facts ONLY (no predictions)
- PSI annotator text is ground truth — if video analysis contradicts it, follow the text
- Enrich with video-only details ONLY if they are consistent with the annotator text
- No distances in meters/feet
- No subjective judgments
- Must be consistent with the answer ({answer})
</output>

Respond using exactly the <thinking> and <output> XML tags above."""

def agent_C(file_handle, description: str, promts_text: str,
            motion_obs: str, scene_obs: str, answer: str,
            feedback: str = "", round_num: int = 0) -> dict:
    print(f"    [Agent C] Grounds 合成 (round {round_num})...")
    prompt = agent_C_prompt(description, promts_text, motion_obs, scene_obs, answer, feedback)
    raw = gemini_call(file_handle, prompt, f"agentC_r{round_num}")
    return {
        "thinking": extract_tag(raw, "thinking"),
        "output":   extract_tag(raw, "output"),
        "raw":      raw,
    }


# ============================================================
# Agent D: Warrant Generator (GPT-4.1 + CoT)
# ============================================================

WARRANT_SYSTEM = """\
You are an expert in Toulmin argumentation for autonomous driving scenarios.

Given GROUNDS (observed visual facts) and the crossing ANSWER, generate a WARRANT.

The WARRANT is the GENERAL RULE or PRINCIPLE explaining WHY the grounds logically
support the answer. It must come from one of:
  1. SOCIAL NORM — shared behavioral expectations between pedestrians and drivers
  2. PHYSICAL LAW — momentum, trajectory, body mechanics
  3. TRAFFIC RULE — codified road conventions

Format your response with XML tags:

<thinking>
1. What category of rule best fits these grounds? (social norm / physical law / traffic rule)
2. What is the core mechanism linking the grounds to the answer?
3. Draft a warrant — is it truly general (not scene-specific)?
4. Does it use probabilistic language? Is it 12-25 words?
5. Does it repeat any words from the grounds? If so, rephrase.
6. Does it match the direction of the answer?
</thinking>

<output>
The single warrant sentence (12-25 words, probabilistic language, no scene details,
no explicit yes/no answer, not about driver behavior).
</output>"""

def agent_D(grounds: str, answer: str, feedback: str = "", round_num: int = 0) -> dict:
    print(f"    [Agent D] Warrant 生成 (round {round_num})...")
    feedback_section = f"\nPrevious attempt feedback (fix these issues):\n{feedback}" if feedback else ""
    user = f"""Grounds: {grounds}
Answer: {answer}{feedback_section}

Generate WARRANT:"""
    raw = gpt_call([
        {"role": "system", "content": WARRANT_SYSTEM},
        {"role": "user",   "content": user},
    ])
    return {
        "thinking": extract_tag(raw, "thinking"),
        "output":   extract_tag(raw, "output"),
        "raw":      raw,
    }


# ============================================================
# Agent E: Critic (GPT-4.1)
# ============================================================

CRITIC_SYSTEM = """\
You are a strict Toulmin argument quality critic for autonomous driving.
Review GROUNDS and WARRANT, correct if needed, output JSON.

Checklist:
GROUNDS:
  - Only present-tense observable visual facts?
  - No predictions, no distances in units, no subjective judgments?

WARRANT:
  - General rule (physical law / social norm / traffic rule)?
  - About PEDESTRIAN behavior — NOT about driver behavior?
  - Probabilistic language (typically/tends to/often/usually)?
  - Does NOT repeat words or phrases from grounds?
  - Does NOT state the answer explicitly?
  - Exactly 12-25 words?

CONSISTENCY (most important):
  - answer=yes → warrant supports WHY pedestrian WILL cross
  - answer=no  → warrant supports WHY pedestrian will NOT cross
  - If direction is wrong, set ok=false and explain clearly.

Output ONLY valid JSON, no markdown:
{"grounds": "...", "warrant": "...", "ok": true/false, "reason": "..."}"""

def agent_E(grounds: str, warrant: str, answer: str) -> dict:
    print(f"    [Agent E] Critic 校验...")
    user = f"""Grounds: {grounds}
Warrant: {warrant}
Answer: {answer}

Review and correct:"""
    raw = gpt_call([
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user",   "content": user},
    ], temperature=0.1)
    try:
        clean = re.sub(r"```[a-z]*\n?", "", raw).strip()
        return json.loads(clean)
    except Exception:
        return {"grounds": grounds, "warrant": warrant,
                "ok": False, "reason": f"parse error: {raw[:120]}"}


# ============================================================
# Orchestrator
# ============================================================

ORCH_SYSTEM = """\
You are an orchestrator for a Toulmin annotation pipeline.
Given a Critic's feedback, decide what needs to be fixed.

Output JSON only:
{
  "retry_grounds": true/false,   // re-run Agent C (grounds synthesis)
  "retry_warrant": true/false,   // re-run Agent D (warrant generation)
  "grounds_feedback": "...",     // specific instruction for Agent C
  "warrant_feedback": "..."      // specific instruction for Agent D
}"""

def orchestrator(critic_result: dict, answer: str) -> dict:
    print(f"    [Orchestrator] 分析失败原因，决定重试策略...")
    user = f"""Critic feedback: {critic_result.get('reason', '')}
Current grounds: {critic_result.get('grounds', '')}
Current warrant: {critic_result.get('warrant', '')}
Answer: {answer}

Decide what to retry and provide specific feedback:"""
    raw = gpt_call([
        {"role": "system", "content": ORCH_SYSTEM},
        {"role": "user",   "content": user},
    ], temperature=0.1)
    try:
        clean = re.sub(r"```[a-z]*\n?", "", raw).strip()
        return json.loads(clean)
    except Exception:
        # 默认两个都重试
        return {
            "retry_grounds": True,
            "retry_warrant": True,
            "grounds_feedback": critic_result.get("reason", ""),
            "warrant_feedback": critic_result.get("reason", ""),
        }


# ============================================================
# 单条记录处理
# ============================================================

def process_record(rec: dict) -> dict:
    raw_ann     = rec.get("raw_annotation", {})
    description = raw_ann.get("description", "").strip()
    promts      = raw_ann.get("promts", {})
    intent      = raw_ann.get("intent", "")
    video_path  = rec.get("video", "")
    promts_text = promts_to_text(promts)

    if intent == "cross":
        answer = "yes"
    elif intent == "not_cross":
        answer = "no"
    else:
        raise ValueError(f"Unsupported intent: {intent}")

    if not video_path or not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # 上传视频一次，所有 Gemini agent 复用
    file_handle = upload_video(video_path)

    try:
        # Agent A & B 只跑一次（不受 Orchestrator 重试影响）
        result_A = agent_A(file_handle)
        result_B = agent_B(file_handle)
        motion_obs = result_A["output"]
        scene_obs  = result_B["output"]

        grounds_feedback = ""
        warrant_feedback = ""
        rounds_log = []
        final_grounds = ""
        final_warrant = ""
        critic_result = {}

        for round_num in range(1, MAX_ROUNDS + 1):
            print(f"  === Round {round_num}/{MAX_ROUNDS} ===")

            # Agent C: Grounds（第一轮或被要求重试时运行）
            if round_num == 1 or orch_decision.get("retry_grounds", True):
                result_C = agent_C(
                    file_handle, description, promts_text,
                    motion_obs, scene_obs, answer,
                    feedback=grounds_feedback, round_num=round_num
                )
                final_grounds = result_C["output"]
            else:
                result_C = {"thinking": "(skipped)", "output": final_grounds, "raw": ""}

            # Agent D: Warrant（第一轮或被要求重试时运行）
            if round_num == 1 or orch_decision.get("retry_warrant", True):
                result_D = agent_D(
                    final_grounds, answer,
                    feedback=warrant_feedback, round_num=round_num
                )
                final_warrant = result_D["output"]
            else:
                result_D = {"thinking": "(skipped)", "output": final_warrant, "raw": ""}

            # Agent E: Critic
            critic_result = agent_E(final_grounds, final_warrant, answer)
            final_grounds = critic_result.get("grounds", final_grounds).strip()
            final_warrant = critic_result.get("warrant", final_warrant).strip()

            rounds_log.append({
                "round": round_num,
                "agent_C_thinking": result_C["thinking"],
                "agent_D_thinking": result_D["thinking"],
                "grounds":  final_grounds,
                "warrant":  final_warrant,
                "critic_ok":     critic_result.get("ok", False),
                "critic_reason": critic_result.get("reason", ""),
            })

            if critic_result.get("ok", False):
                print(f"  ✅ Critic 通过！(round {round_num})")
                break

            if round_num < MAX_ROUNDS:
                # Orchestrator 决定下一步
                orch_decision = orchestrator(critic_result, answer)
                grounds_feedback = orch_decision.get("grounds_feedback", "")
                warrant_feedback = orch_decision.get("warrant_feedback", "")
            else:
                print(f"  ⚠️  达到最大轮数，使用最后一轮结果")
                orch_decision = {}  # 不再重试

    finally:
        # 统一删除上传的视频
        genai.delete_file(file_handle.name)
        print(f"  🗑️  已删除云端文件: {file_handle.name}")

    return {
        "grounds": final_grounds,
        "warrant": final_warrant,
        "answer":  answer,
        "critic_ok":     critic_result.get("ok", False),
        "critic_reason": critic_result.get("reason", ""),
        "agent_A_thinking": result_A["thinking"],
        "agent_B_thinking": result_B["thinking"],
        "rounds": rounds_log,
    }


# ============================================================
# Main
# ============================================================

def main():
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

            print(f"\n[{i}/{len(records)}] {rec_id}")

            try:
                toulmin = process_record(rec)
                out = dict(rec)
                out["toulmin"] = toulmin
                out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
                out_f.flush()
                ok_cnt += 1
                print(f"  grounds: {toulmin['grounds'][:80]}...")
                print(f"  warrant: {toulmin['warrant']}")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                err_cnt += 1

            if i < len(records):
                time.sleep(REQUEST_DELAY)

    print(f"\n=== 完成 ===")
    print(f"OK: {ok_cnt} | Skipped: {skip_cnt} | Error: {err_cnt}")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()