"""
从 psi_90f_toulmin_1000.jsonl 中提取 overall_ok=False 的 510 条
转换为 GRPO 格式（只需要 video + prompt + ground truth answer）
"""
import csv, json, os

TOULMIN_JSONL = "/workspace/PSI_change/json_mode_90/trf_train/psi_90f_toulmin_1000.jsonl"
QUALITY_CSV   = "/workspace/PSI_change/json_mode_90/quality_report.csv"
OUTPUT_JSONL  = "/workspace/PSI_change/json_mode_90/trf_train/psi_grpo_510.jsonl"

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

USER_PROMPT = (
    "Watch the full video and predict: will the TARGET pedestrian "
    "attempt to cross in front of the vehicle in the next moment?"
)

# ── 读 quality report，找 overall_ok=False 的 id ──
bad_ids = set()
with open(QUALITY_CSV) as f:
    for row in csv.DictReader(f):
        if row["overall_ok"] == "False":
            bad_ids.add(row["id"])
print(f"Bad records in quality report: {len(bad_ids)}")

# ── 从 toulmin jsonl 提取并转换格式 ──
records = []
with open(TOULMIN_JSONL) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec["id"] not in bad_ids:
            continue

        meta   = rec["meta"]
        intent = meta.get("intent", "")

        # ground truth answer
        if intent == "cross":        answer = "yes"
        elif intent == "not_cross":  answer = "no"
        else:
            continue  # skip not_sure

        grpo_rec = {
            "id":    rec["id"],
            "video": rec["video"],
            "system": SYSTEM_PROMPT,
            "prompt": [
                {"role": "user", "content": USER_PROMPT}
            ],
            "answer": answer,   # ground truth，用于 reward 计算
            "meta": {
                "video_id":          meta["video_id"],
                "track_id":          meta["track_id"],
                "annotator_id":      meta["annotator_id"],
                "frame_span":        meta["frame_span"],
                "label_frame":       meta["label_frame"],
                "aggregated_intent": meta.get("aggregated_intent", 0.5),
            }
        }
        records.append(grpo_rec)

print(f"GRPO records: {len(records)}")

# yes/no 分布
yes = sum(1 for r in records if r["answer"] == "yes")
no  = sum(1 for r in records if r["answer"] == "no")
print(f"  yes: {yes} ({yes/len(records)*100:.1f}%)")
print(f"  no:  {no}  ({no/len(records)*100:.1f}%)")

os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
with open(OUTPUT_JSONL, "w") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"Output: {OUTPUT_JSONL}")


import json, random

records = []
with open("/workspace/PSI_change/json_mode_90/trf_train/psi_grpo_510.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

yes_recs = [r for r in records if r["answer"] == "yes"]  # 206
no_recs  = [r for r in records if r["answer"] == "no"]   # 297

random.seed(42)

# 目标：和 SFT 类似，约 70% yes 30% no
# 但受限于 yes 只有 206 条
# 取全部 206 yes + 88 no → 总 294 条，比例 70/30

no_sample = random.sample(no_recs, 88)
final = yes_recs + no_sample
random.shuffle(final)

with open("/workspace/PSI_change/json_mode_90/trf_train/psi_grpo_balanced.jsonl", "w") as f:
    for r in final:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Total: {len(final)}")
print(f"yes: {sum(1 for r in final if r['answer']=='yes')}")
print(f"no:  {sum(1 for r in final if r['answer']=='no')}")