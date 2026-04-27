"""
Toulmin agent — TEST SET 前 20 条
输入: psi_90f_test_eval.jsonl (已有 id / video / ground_truth_answer / ground_truth_text)
输出: psi_90f_test_toulmin_20.jsonl
      字段: id, video, ground_truth_answer, agent_grounds, agent_warrant, agent_ok
"""

import json, os, re, time, hashlib
from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────
INPUT_JSONL  = "/workspace/PSI_change/json_mode_90/psi_90f_test_eval.jsonl"
OUTPUT_JSONL = "/workspace/PSI_change/json_mode_90/psi_90f_test_toulmin_20.jsonl"
CACHE_DIR    = "/workspace/PSI_change/json_mode_90/agent_cache_test"
MODEL        = "gpt-5.4"          # 改成你实际用的 model string
MAX_RETRIES  = 3
SLEEP        = 0.3
MAX_RECORDS  = 20                 # 只跑前 20 条
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
# 从 ground_truth_text 解析 description 给 agent 用
# ground_truth_text 格式: "grounds: ...\nwarrant: ...\nanswer: ..."
# 我们把 grounds 部分当做原始 annotator 描述送给 extractor
# ════════════════════════════════════════════════════

def parse_gt_text(gt_text: str) -> dict:
    """解析 psi_90f_test_eval.jsonl 里的 ground_truth_text"""
    result = {"grounds": "", "warrant": "", "answer": ""}
    if not gt_text:
        return result
    for line in gt_text.split("\n"):
        if line.startswith("grounds:"):
            result["grounds"] = line[len("grounds:"):].strip()
        elif line.startswith("warrant:"):
            result["warrant"] = line[len("warrant:"):].strip()
        elif line.startswith("answer:"):
            result["answer"]  = line[len("answer:"):].strip()
    return result


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

def agent_extractor(description: str, answer: str) -> str:
    user = f"""Driver's description of the pedestrian:
{description}

Answer (for context only, do NOT include in grounds): {answer}

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
- NO specific scene details
- NO repetition of words or phrases from the grounds
- NO explicit answer (yes/no, cross/not cross, will/won't)
- NOT about driver behavior

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
# Agent 3: Critic
# ════════════════════════════════════════════════════

CRITIC_SYSTEM = """\
You are a strict Toulmin argument quality critic for autonomous driving.
Review GROUNDS and WARRANT, correct if needed, and return JSON.

Checklist:
GROUNDS: Only observable visual facts? (no predictions, no unit distances)
WARRANT:
  - General rule (physical law / social norm / traffic rule)?
  - About PEDESTRIAN behavior/tendency — NOT about what the driver should do?
  - Probabilistic language?
  - Does NOT repeat words from grounds?
  - Does NOT state the answer explicitly?
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
    with open(INPUT_JSONL) as f:
        records = [json.loads(l) for l in f if l.strip()]

    # 只取前 20 条
    records = records[:MAX_RECORDS]
    print(f"Processing first {len(records)} records from test set")

    # 断点续跑
    done_ids = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL) as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
    print(f"Already done: {len(done_ids)}")

    out_f    = open(OUTPUT_JSONL, "a")
    ok_cnt   = 0
    skip_cnt = 0
    err_cnt  = 0

    for i, rec in enumerate(records):
        rid = rec["id"]

        if rid in done_ids:
            skip_cnt += 1
            print(f"  [{i+1}/{len(records)}] SKIP {rid}")
            continue

        # 从 ground_truth_text 拿原始 annotator 描述作为 agent 输入
        gt_parsed   = parse_gt_text(rec.get("ground_truth_text", ""))
        description = gt_parsed["grounds"]   # annotator 的原始 grounds 描述
        answer      = rec.get("ground_truth_answer", "").strip()

        if answer not in ("yes", "no"):
            print(f"  [{i+1}] SKIP {rid} — invalid answer: {answer!r}")
            err_cnt += 1
            continue

        if not description:
            print(f"  [{i+1}] SKIP {rid} — empty description")
            err_cnt += 1
            continue

        print(f"  [{i+1}/{len(records)}] {rid}")

        # Agent 1: Extractor
        grounds = agent_extractor(description, answer)
        if not grounds:
            print(f"    ✗ extractor failed")
            err_cnt += 1
            continue

        # Agent 2: Warrant
        warrant = agent_warrant(grounds, answer)
        if not warrant:
            print(f"    ✗ warrant failed")
            err_cnt += 1
            continue

        # Agent 3: Critic
        result  = agent_critic(grounds, warrant, answer)
        grounds = result.get("grounds", grounds).strip()
        warrant = result.get("warrant", warrant).strip()

        print(f"    grounds: {grounds[:60]}...")
        print(f"    warrant: {warrant}")
        print(f"    ok={result.get('ok')} | {result.get('reason','')[:60]}")

        # 输出格式：与 grpo_results.jsonl 和 psi_90f_test_eval.jsonl 兼容
        # 用 agent_ 前缀区分，方便 HTML viewer 做三列对比
        out = {
            "id":                   rid,
            "video":                rec.get("video", ""),
            "ground_truth_answer":  answer,
            "agent_grounds":        grounds,
            "agent_warrant":        warrant,
            "agent_ok":             result.get("ok", False),
            "agent_reason":         result.get("reason", ""),
            # 保留原始 gt 文本方便对比
            "ground_truth_text":    rec.get("ground_truth_text", ""),
        }

        out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
        out_f.flush()
        ok_cnt += 1

    out_f.close()
    print(f"\nDone. ok={ok_cnt} skip={skip_cnt} err={err_cnt}")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()