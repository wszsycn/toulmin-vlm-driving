"""
过滤 overall_ok=True 的记录，输出 SFT 数据集，并统计数据分布
"""
import json, csv
from collections import defaultdict

TOULMIN_JSONL = "/workspace/PSI_change/json_mode_90/trf_train/psi_90f_toulmin_1000.jsonl"
QUALITY_CSV   = "/workspace/PSI_change/json_mode_90/quality_report.csv"
OUTPUT_JSONL  = "/workspace/PSI_change/json_mode_90/trf_train/psi_sft_final.jsonl"

# ── 读 quality report，找 overall_ok=True 的 id ──
ok_ids = set()
ok_meta = {}  # id -> row
with open(QUALITY_CSV) as f:
    for row in csv.DictReader(f):
        if row["overall_ok"] == "True":
            ok_ids.add(row["id"])
            ok_meta[row["id"]] = row

print(f"Overall OK records: {len(ok_ids)}")

# ── 过滤 toulmin jsonl ──
records = []
with open(TOULMIN_JSONL) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec["id"] in ok_ids:
            records.append(rec)

print(f"Filtered records: {len(records)}")

# ── 写出 SFT 数据 ──
with open(OUTPUT_JSONL, "w") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"Output: {OUTPUT_JSONL}")

# ════════════════════════════════════════════════════
# 统计
# ════════════════════════════════════════════════════
print(f"\n{'='*50}")
print(f"SFT Dataset Statistics")
print(f"{'='*50}")

# 1. 总量
total = len(records)
print(f"\nTotal records: {total}")

# 2. yes/no 分布
yes_cnt = sum(1 for r in records if ok_meta[r["id"]]["answer"] == "yes")
no_cnt  = total - yes_cnt
print(f"\nAnswer distribution:")
print(f"  yes: {yes_cnt} ({yes_cnt/total*100:.1f}%)")
print(f"  no:  {no_cnt}  ({no_cnt/total*100:.1f}%)")

# 3. warrant 类型分布
print(f"\nWarrant type distribution:")
type_counts = defaultdict(int)
for rid in ok_ids:
    t = ok_meta[rid]["warrant_type"]
    type_counts[t] += 1
for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {t:15s}: {c:4d} ({c/total*100:.1f}%)")

# 4. word count 分布
word_counts = [int(ok_meta[r["id"]]["word_count"]) for r in records]
print(f"\nWarrant word count:")
print(f"  mean:   {sum(word_counts)/len(word_counts):.1f}")
print(f"  min:    {min(word_counts)}")
print(f"  max:    {max(word_counts)}")
print(f"  12-25:  {sum(1 for w in word_counts if 12 <= w <= 25)} ({sum(1 for w in word_counts if 12 <= w <= 25)/total*100:.1f}%)")

# 5. video/track 分布
video_counts = defaultdict(int)
track_counts = defaultdict(int)
for rec in records:
    meta = rec["meta"]
    video_counts[meta["video_id"]] += 1
    track_counts[(meta["video_id"], meta["track_id"])] += 1

print(f"\nCoverage:")
print(f"  Unique videos:      {len(video_counts)}")
print(f"  Unique tracks:      {len(track_counts)}")
print(f"  Avg records/video:  {total/len(video_counts):.1f}")
print(f"  Avg records/track:  {total/len(track_counts):.1f}")
print(f"  Max records/track:  {max(track_counts.values())}")

# 6. aggregated_intent 分布
intents = [rec["meta"].get("aggregated_intent", 0.5) for rec in records]
print(f"\nAggregated intent:")
print(f"  mean:       {sum(intents)/len(intents):.3f}")
print(f"  >= 0.7:     {sum(1 for v in intents if v >= 0.7)} ({sum(1 for v in intents if v >= 0.7)/total*100:.1f}%)")
print(f"  <= 0.3:     {sum(1 for v in intents if v <= 0.3)} ({sum(1 for v in intents if v <= 0.3)/total*100:.1f}%)")
print(f"  0.3-0.7:    {sum(1 for v in intents if 0.3 < v < 0.7)} ({sum(1 for v in intents if 0.3 < v < 0.7)/total*100:.1f}%)")

# 7. agent_ok 分布
agent_ok = sum(1 for rec in records if rec["meta"].get("agent_ok", False))
print(f"\nAgent critic:")
print(f"  Passed first time (agent_ok=True):  {agent_ok} ({agent_ok/total*100:.1f}%)")
print(f"  Needed correction (agent_ok=False): {total-agent_ok} ({(total-agent_ok)/total*100:.1f}%)")