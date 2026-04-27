"""
从 psi_90f_raw.jsonl 选 1000 条：
1. 计算每个 (video_id, track_id, win_start, win_end) 的 aggregated_intent
   cross=1, not_cross=0, not_sure=0.5
2. 过滤掉 aggregated_intent 接近 0.5 的窗口
3. yes/no 平衡随机采样 1000 条（每条是一个 annotator 的记录）
"""

import json, random
from collections import defaultdict

INPUT_JSONL  = "/workspace/PSI_change/json_mode_90/psi_90f_raw.jsonl"
OUTPUT_JSONL = "/workspace/PSI_change/json_mode_90/psi_90f_1000.jsonl"
TARGET_N     = 1000
INTENT_THRESH = 0.2   # |aggregated_intent - 0.5| > 此值才保留
MAX_PER_TRACK = 5     # 每个 (video_id, track_id) 最多保留几条
SEED         = 42
random.seed(SEED)

INTENT_MAP = {"cross": 1.0, "not_cross": 0.0, "not_sure": 0.5}

# ── 读数据 ──
with open(INPUT_JSONL) as f:
    records = [json.loads(l) for l in f if l.strip()]
print(f"Input: {len(records)}")

# ── 计算每个窗口的 aggregated_intent ──
# 同一 (video_id, track_id, win_start, win_end) 的所有 annotator 记录
window_groups = defaultdict(list)
for rec in records:
    meta = rec["meta"]
    wkey = (meta["video_id"], meta["track_id"],
            meta["frame_span"][0], meta["frame_span"][1])
    window_groups[wkey].append(rec)

# 计算 aggregated_intent 并标注到每条记录
filtered = []
for wkey, recs in window_groups.items():
    intents = [INTENT_MAP.get(r["raw_annotation"]["intent"], 0.5) for r in recs]
    agg = sum(intents) / len(intents)

    # 过滤掉接近 0.5 的
    if abs(agg - 0.5) <= INTENT_THRESH:
        continue

    # 给每条记录加上 aggregated_intent
    for rec in recs:
        rec["meta"]["aggregated_intent"] = round(agg, 4)
        filtered.append(rec)

print(f"After intent filter (|agg - 0.5| > {INTENT_THRESH}): {len(filtered)}")

# ── 每个 (video_id, track_id) 最多保留 MAX_PER_TRACK 条 ──
# 在同一 track 的多条里随机选，避免某个 track 占太多
track_buckets = defaultdict(list)
for rec in filtered:
    meta = rec["meta"]
    tk   = (meta["video_id"], meta["track_id"])
    track_buckets[tk].append(rec)

filtered2 = []
for tk, recs in track_buckets.items():
    random.shuffle(recs)
    filtered2.extend(recs[:MAX_PER_TRACK])

print(f"After per-track limit ({MAX_PER_TRACK}): {len(filtered2)}")
filtered = filtered2

# ── yes/no 分池 ──
# aggregated_intent > 0.5 → yes pool，< 0.5 → no pool
yes_pool = [r for r in filtered if r["meta"]["aggregated_intent"] > 0.5]
no_pool  = [r for r in filtered if r["meta"]["aggregated_intent"] < 0.5]
print(f"  yes pool: {len(yes_pool)}  no pool: {len(no_pool)}")

random.shuffle(yes_pool)
random.shuffle(no_pool)

half    = TARGET_N // 2
yes_sel = yes_pool[:min(half, len(yes_pool))]
no_sel  = no_pool[:min(half, len(no_pool))]

# 不足则从另一侧补
if len(yes_sel) < half:
    no_sel = no_pool[:TARGET_N - len(yes_sel)]
elif len(no_sel) < half:
    yes_sel = yes_pool[:TARGET_N - len(no_sel)]

final = yes_sel + no_sel
random.shuffle(final)

print(f"\n最终选出: {len(final)}")
print(f"  yes: {sum(1 for r in final if r['meta']['aggregated_intent'] > 0.5)}")
print(f"  no:  {sum(1 for r in final if r['meta']['aggregated_intent'] < 0.5)}")
print(f"  agg_intent 均值: {sum(r['meta']['aggregated_intent'] for r in final)/len(final):.3f}")
print(f"  agg_intent 最小距离0.5: {min(abs(r['meta']['aggregated_intent']-0.5) for r in final):.3f}")

with open(OUTPUT_JSONL, "w") as f:
    for rec in final:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"\nOutput: {OUTPUT_JSONL}")