"""
批量检查 toulmin jsonl 的质量
对每条记录检查：
1. warrant 和 answer 方向是否一致
2. warrant 类型（social norm / physical law / traffic rule / other）
3. warrant 是否重复 grounds 内容
4. warrant 长度是否在 12-25 词
5. grounds 是否包含非视觉信息

输出：
- 质量报告 CSV
- 问题记录单独存一个 jsonl
"""

import json, os, re, time, hashlib, csv
from openai import OpenAI

INPUT_JSONL   = "/workspace/PSI_change/json_mode_90/trf_train/psi_90f_toulmin_1000.jsonl"
REPORT_CSV    = "/workspace/PSI_change/json_mode_90/quality_report.csv"
PROBLEMS_JSONL = "/workspace/PSI_change/json_mode_90/quality_problems.jsonl"
CACHE_DIR     = "/workspace/PSI_change/json_mode_90/quality_cache"
MODEL         = "gpt-5.2"
SLEEP         = 0.2

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Cache ──
def cached_call(messages, temperature=0.1):
    key = json.dumps(messages, ensure_ascii=False)
    h   = hashlib.sha256(key.encode()).hexdigest()[:16]
    cp  = os.path.join(CACHE_DIR, f"{h}.json")
    if os.path.exists(cp):
        with open(cp) as f:
            return json.load(f)["response"]
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=256,
            )
            text = resp.choices[0].message.content.strip()
            with open(cp, "w") as f:
                json.dump({"response": text}, f)
            time.sleep(SLEEP)
            return text
        except Exception as e:
            # 400 content policy — don't retry, raise immediately
            if hasattr(e, "status_code") and e.status_code == 400:
                raise
            print(f"  [API error]: {e}")
            time.sleep(2 ** attempt)
    return ""

# ── Checker prompt ──
CHECKER_SYSTEM = """\
You are a Toulmin argument quality checker for autonomous driving.
Given GROUNDS, WARRANT, and ANSWER, evaluate the warrant quality.

Return ONLY valid JSON (no markdown):
{
  "direction_ok": true/false,        // warrant supports the answer direction
  "warrant_type": "physical_law" | "social_norm" | "traffic_rule" | "other",
  "repeats_grounds": true/false,     // warrant repeats key phrases from grounds
  "word_count_ok": true/false,       // warrant is 12-25 words
  "grounds_clean": true/false,       // grounds contains only observable visual facts
  "overall_ok": true/false,          // all checks pass
  "issues": "brief description of problems, or empty string if none"
}"""

def check_record(grounds, warrant, answer):
    user = f"Grounds: {grounds}\nWarrant: {warrant}\nAnswer: {answer}\n\nEvaluate:"
    raw = cached_call([
        {"role": "system", "content": CHECKER_SYSTEM},
        {"role": "user",   "content": user},
    ])
    try:
        clean = re.sub(r"```[a-z]*\n?", "", raw).strip()
        return json.loads(clean)
    except Exception:
        return {"overall_ok": False, "issues": f"parse error: {raw[:80]}",
                "direction_ok": None, "warrant_type": "unknown",
                "repeats_grounds": None, "word_count_ok": None, "grounds_clean": None}

def parse_conversation(rec):
    gpt_val = rec["conversations"][1]["value"]
    grounds = warrant = answer = ""
    for line in gpt_val.split("\n"):
        line = line.strip()
        if line.startswith("grounds:"):
            grounds = line[len("grounds:"):].strip()
        elif line.startswith("warrant:"):
            warrant = line[len("warrant:"):].strip()
        elif line.startswith("answer:"):
            answer  = line[len("answer:"):].strip()
    return grounds, warrant, answer

# ── Main ──
def main():
    with open(INPUT_JSONL) as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"Checking {len(records)} records...")

    # 断点续跑
    done = {}
    if os.path.exists(REPORT_CSV):
        with open(REPORT_CSV) as f:
            for row in csv.DictReader(f):
                done[row["id"]] = row

    rows = []
    problems = []

    for i, rec in enumerate(records):
        rid = rec["id"]

        if rid in done:
            rows.append(done[rid])
            continue

        grounds, warrant, answer = parse_conversation(rec)
        if not grounds or not warrant or not answer:
            result = {"overall_ok": False, "issues": "missing fields",
                      "direction_ok": False, "warrant_type": "unknown",
                      "repeats_grounds": False, "word_count_ok": False,
                      "grounds_clean": False}
        else:
            try:
                result = check_record(grounds, warrant, answer)
            except Exception as e:
                print(f"  [SKIP] {rid}: {e}")
                result = {"overall_ok": False, "issues": f"api_skip: {str(e)[:60]}",
                          "direction_ok": None, "warrant_type": "unknown",
                          "repeats_grounds": None, "word_count_ok": None,
                          "grounds_clean": None}

        word_count = len(warrant.split())
        row = {
            "id":             rid,
            "answer":         answer,
            "warrant_type":   result.get("warrant_type", "unknown"),
            "direction_ok":   result.get("direction_ok", ""),
            "repeats_grounds":result.get("repeats_grounds", ""),
            "word_count":     word_count,
            "word_count_ok":  result.get("word_count_ok", ""),
            "grounds_clean":  result.get("grounds_clean", ""),
            "overall_ok":     result.get("overall_ok", ""),
            "issues":         result.get("issues", ""),
            "grounds":        grounds[:80],
            "warrant":        warrant[:120],
        }
        rows.append(row)

        if not result.get("overall_ok", True):
            rec["quality_issues"] = result.get("issues", "")
            problems.append(rec)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(records)}] problems so far: {len(problems)}")

    # 写 CSV
    fields = ["id","answer","warrant_type","direction_ok","repeats_grounds",
              "word_count","word_count_ok","grounds_clean","overall_ok","issues",
              "grounds","warrant"]
    with open(REPORT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # 写问题记录
    with open(PROBLEMS_JSONL, "w") as f:
        for r in problems:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计
    total       = len(rows)
    ok          = sum(1 for r in rows if str(r["overall_ok"]) == "True")
    dir_fail    = sum(1 for r in rows if str(r["direction_ok"]) == "False")
    repeat_fail = sum(1 for r in rows if str(r["repeats_grounds"]) == "True")
    len_fail    = sum(1 for r in rows if str(r["word_count_ok"]) == "False")
    dirty_gnd   = sum(1 for r in rows if str(r["grounds_clean"]) == "False")

    type_counts = {}
    for r in rows:
        t = r.get("warrant_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n{'='*50}")
    print(f"Total:              {total}")
    print(f"Overall OK:         {ok} ({ok/total*100:.1f}%)")
    print(f"Direction mismatch: {dir_fail}")
    print(f"Repeats grounds:    {repeat_fail}")
    print(f"Wrong word count:   {len_fail}")
    print(f"Dirty grounds:      {dirty_gnd}")
    print(f"\nWarrant type distribution:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:15s}: {c}")
    print(f"\nReport: {REPORT_CSV}")
    print(f"Problems: {PROBLEMS_JSONL} ({len(problems)} records)")

if __name__ == "__main__":
    main()