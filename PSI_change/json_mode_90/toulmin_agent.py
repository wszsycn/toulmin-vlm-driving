"""
Step 2: Multi-agent Toulmin annotation
读取 psi_90f_raw.jsonl，对每条记录用三个 agent 生成 grounds + warrant
Agent 1: Extractor  → 从 description + promts 提取结构化 grounds
Agent 2: Warrant    → 生成通用规则 warrant
Agent 3: Critic     → 自我反思，验证并修正
"""

import json, os, re, time, hashlib
from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────
INPUT_JSONL  = "/workspace/PSI_change/json_mode_90/psi_90f_1000.jsonl"
OUTPUT_JSONL = "/workspace/PSI_change/json_mode_90/trf_train/psi_90f_toulmin_1000.jsonl"
CACHE_DIR    = "/workspace/PSI_change/json_mode_90/agent_cache"
MODEL        = "gpt-5.2"
MAX_RETRIES  = 3
SLEEP        = 0.3
# ──────────────────────────────────────────────────────

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)


# ════════════════════════════════════════════════════
# Cache
# ════════════════════════════════════════════════════

def cached_call(messages: list, temperature=0.3) -> str:
    key = json.dumps(messages, ensure_ascii=False)
    h   = hashlib.sha256(key.encode()).hexdigest()[:16]
    cp  = os.path.join(CACHE_DIR, f"{h}.json")

    if os.path.exists(cp):
        with open(cp) as f:
            return json.load(f)["response"]

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=512,
            )
            text = resp.choices[0].message.content.strip()
            with open(cp, "w") as f:
                json.dump({"response": text}, f)
            time.sleep(SLEEP)
            return text
        except Exception as e:
            print(f"  [API error attempt {attempt+1}]: {e}")
            time.sleep(2 ** attempt)
    return ""


# ════════════════════════════════════════════════════
# Promts → readable text
# ════════════════════════════════════════════════════

def promts_to_text(promts) -> str:
    if not promts or not isinstance(promts, dict):
        return ""
    labels = {
        "pedestrian":       "Pedestrian behavior",
        "goalRelated":      "Goal/position",
        "roadUsersRelated": "Relative to vehicle",
        "roadFactors":      "Road environment",
        "norms":            "Social norms/observations",
    }
    parts = []
    for k, label in labels.items():
        v = promts.get(k, "").strip()
        if v:
            parts.append(f"{label}: {v}")
    return "\n".join(parts)


# ════════════════════════════════════════════════════
# Agent 1: Extractor
# ════════════════════════════════════════════════════

EXTRACTOR_SYSTEM = """\
You are a precise visual evidence extractor for autonomous driving.
Given a driver's textual observation of a pedestrian, extract a concise GROUNDS statement.

Rules:
- 1-2 sentences of concrete, observable visual facts only
  (posture, movement direction, position relative to road, gaze direction)
- NO predictions or conclusions
- NO specific distances in units (feet/meters) — use relative terms (close, far, nearby)
- NO crossing decision (yes/no, will/won't cross)
- Max 50 words
Output only the grounds text, nothing else."""

def agent_extractor(description: str, promts_text: str, answer: str) -> str:
    user = f"""Driver's description:
{description}

Structured observations:
{promts_text}

Answer (for context only, do NOT include): {answer}

Extract GROUNDS:"""
    return cached_call([
        {"role": "system", "content": EXTRACTOR_SYSTEM},
        {"role": "user",   "content": user},
    ])


# ════════════════════════════════════════════════════
# Agent 2: Warrant Generator
# ════════════════════════════════════════════════════

WARRANT_SYSTEM = """\
You are an expert in Toulmin argumentation applied to pedestrian behavior analysis.

Your task: given GROUNDS (observed visual facts) and the crossing ANSWER, generate a WARRANT.

In Toulmin's model, the WARRANT is the GENERAL RULE or PRINCIPLE that explains
WHY the grounds logically support the claim. It is NOT a restatement of the grounds.
It is an independent, general truth that applies to any similar situation.

Think of it as answering: "Under what general principle does this evidence imply this conclusion?"

The warrant must come from one of these three categories:
  1. SOCIAL NORM — shared behavioral expectations between pedestrians and drivers
     e.g. "Pedestrians who have established visual contact with a driver typically assume the driver will yield and proceed."
  2. PHYSICAL LAW — momentum, trajectory, body mechanics
     e.g. "A body already in lateral motion across a road tends to continue along that trajectory unless actively stopped."
  3. TRAFFIC RULE — codified road conventions
     e.g. "Pedestrians who enter a marked crosswalk have legal right of way and typically proceed without stopping."

Rules:
- The warrant must be a GENERAL principle, not a description of this specific scene
- Use probabilistic language: "typically", "tends to", "often", "usually"
- Length: 12-25 words exactly
- NO specific scene details (no locations, distances, traffic light colors, frame numbers)
- NO repetition of words or phrases from the grounds
- NO explicit answer (yes/no, cross/not cross, will/won't)
- NOT about driver behavior (yielding, slowing down, braking)

Examples:

  [Social norm — yes]
  Grounds: A pedestrian at the roadside makes eye contact with the driver and steps forward.
  Answer: yes
  Warrant: Pedestrians who acknowledge an oncoming vehicle but continue forward typically assume the driver will yield to them.

  [Physical law — yes]
  Grounds: A pedestrian is already mid-road, body oriented laterally, moving across multiple lanes.
  Answer: yes
  Warrant: A body already committed to lateral motion across a road tends to continue unless an external force causes it to stop.

  [Traffic rule — yes]
  Grounds: A pedestrian steps off the curb at a marked crosswalk with the walk signal active.
  Answer: yes
  Warrant: Pedestrians entering a crosswalk under a walk signal have legal priority and typically proceed through without pausing.

  [Social norm — no]
  Grounds: A pedestrian on the sidewalk glances at the vehicle, then steps back and waits.
  Answer: no
  Warrant: Pedestrians who retreat from the curb after seeing an approaching vehicle typically defer crossing until the vehicle passes.

  [Physical law — no]
  Grounds: A pedestrian is walking parallel to the road, body oriented away from the opposite sidewalk.
  Answer: no
  Warrant: A person whose body trajectory runs parallel to the road rather than perpendicular typically lacks the momentum to initiate a crossing.

  [Traffic rule — no]
  Grounds: A pedestrian is standing at the curb facing the road while the pedestrian signal shows a red hand.
  Answer: no
  Warrant: Pedestrians facing a do-not-walk signal typically wait at the curb rather than entering the roadway.

Output only the warrant sentence, nothing else."""

def agent_warrant(grounds: str, answer: str) -> str:
    user = f"""Grounds: {grounds}
Answer: {answer}

Generate WARRANT:"""
    return cached_call([
        {"role": "system", "content": WARRANT_SYSTEM},
        {"role": "user",   "content": user},
    ])


# ════════════════════════════════════════════════════
# Agent 3: Critic (self-reflection)
# ════════════════════════════════════════════════════

CRITIC_SYSTEM = """\
You are a strict Toulmin argument quality critic for autonomous driving.
Review GROUNDS and WARRANT, correct if needed, and return JSON.

Checklist:
GROUNDS:
  - Only observable visual facts? (no predictions, no unit distances, no driver behavior)

WARRANT:
  - General rule (physical law / social norm / traffic rule)?
  - About PEDESTRIAN behavior/tendency — NOT about what the driver should do?
  - Uses probabilistic language (typically, tends to, often, usually)?
  - Does NOT repeat words or phrases from the grounds?
  - Does NOT state the answer explicitly (no yes/no, cross/not cross, will/won't)?
  - 12-25 words?
  - No driver behavior at the end (no "so drivers should...", "absent intervention", "path remains in conflict")?

CONSISTENCY — MOST IMPORTANT:
  - If answer is "yes": warrant must support WHY the pedestrian WILL cross
  - If answer is "no": warrant must support WHY the pedestrian will NOT cross
  - If warrant direction contradicts the answer, you MUST rewrite the warrant entirely

Output ONLY valid JSON, no markdown fences:
{"grounds": "...", "warrant": "...", "ok": true/false, "reason": "..."}"""

def agent_critic(grounds: str, warrant: str, answer: str) -> dict:
    user = f"""Grounds: {grounds}
Warrant: {warrant}
Answer: {answer}

Review and correct:"""
    raw = cached_call([
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user",   "content": user},
    ], temperature=0.1)

    try:
        clean = re.sub(r"```[a-z]*\n?", "", raw).strip()
        return json.loads(clean)
    except Exception:
        return {"grounds": grounds, "warrant": warrant,
                "ok": False, "reason": f"parse error: {raw[:100]}"}


# ════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════

def main():
    # 读输入
    with open(INPUT_JSONL) as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"Input records: {len(records)}")

    # 断点续跑
    done_ids = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["id"])
    print(f"Already done: {len(done_ids)}")

    out_f      = open(OUTPUT_JSONL, "a")
    ok_cnt     = 0
    skip_cnt   = 0
    err_cnt    = 0

    for i, rec in enumerate(records):
        if rec["id"] in done_ids:
            skip_cnt += 1
            continue

        raw_ann = rec.get("raw_annotation", {})
        description = raw_ann.get("description", "").strip()
        promts      = raw_ann.get("promts", {})
        intent      = raw_ann.get("intent", "")
        promts_text = promts_to_text(promts)

        # answer
        if intent == "cross":        answer = "yes"
        elif intent == "not_cross":  answer = "no"
        else:
            err_cnt += 1
            continue

        if not description and not promts_text:
            err_cnt += 1
            continue

        # Agent 1
        grounds = agent_extractor(description, promts_text, answer)
        if not grounds:
            err_cnt += 1
            continue

        # Agent 2
        warrant = agent_warrant(grounds, answer)
        if not warrant:
            err_cnt += 1
            continue

        # Agent 3
        result  = agent_critic(grounds, warrant, answer)
        grounds = result.get("grounds", grounds).strip()
        warrant = result.get("warrant", warrant).strip()

        # 写出：保留原始记录所有字段，替换 conversations
        out = dict(rec)
        out["conversations"] = [
            {
                "from":  "human",
                "value": (
                    "<video>\n"
                    "You are given a driving video showing a pedestrian.\n"
                    "The TARGET pedestrian is highlighted by the green bounding box.\n\n"
                    "Task: Predict whether the TARGET pedestrian has a crossing "
                    "intention in the NEXT frame (t+1).\n"
                    "Format strictly as:\n"
                    "grounds: <visual observations>\n"
                    "warrant: <general rule linking observations to intention>\n"
                    "answer: yes | no"
                )
            },
            {
                "from":  "gpt",
                "value": (
                    f"grounds: {grounds}\n"
                    f"warrant: {warrant}\n"
                    f"answer: {answer}"
                )
            }
        ]
        out["meta"]["agent_ok"]     = result.get("ok", False)
        out["meta"]["agent_reason"] = result.get("reason", "")

        out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
        out_f.flush()
        ok_cnt += 1

        if (ok_cnt + err_cnt) % 100 == 0:
            print(f"  [{i+1}/{len(records)}] ok={ok_cnt} skip={skip_cnt} err={err_cnt}")

    out_f.close()
    print(f"\nDone. ok={ok_cnt} skip={skip_cnt} err={err_cnt}")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()