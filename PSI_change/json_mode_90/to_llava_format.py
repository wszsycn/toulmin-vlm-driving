"""
将 psi_sft_final.jsonl 转换为 LLaVA 格式（保留 mp4，不提取帧）
"""

import json, os

# ── 配置 ──────────────────────────────────────────────
INPUT_JSONL  = "/workspace/PSI_change/json_mode_90/trf_train/psi_sft_final.jsonl"
OUTPUT_JSONL = "/workspace/PSI_change/json_mode_90/trf_train/psi_sft_llava.jsonl"
NUM_FRAMES   = 90
# ──────────────────────────────────────────────────────

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


def parse_gpt_value(text: str) -> tuple:
    grounds = warrant = answer = ""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("grounds:"):
            grounds = line[len("grounds:"):].strip()
        elif line.startswith("warrant:"):
            warrant = line[len("warrant:"):].strip()
        elif line.startswith("answer:"):
            answer  = line[len("answer:"):].strip()
    return grounds, warrant, answer


def main():
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    with open(INPUT_JSONL) as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"Input: {len(records)} records")

    out_f   = open(OUTPUT_JSONL, "w")
    ok_cnt  = 0
    err_cnt = 0

    for rec in records:
        meta       = rec["meta"]
        rec_id     = rec["id"]
        video_path = rec["video"]

        if not os.path.exists(video_path):
            print(f"  [WARN] video not found: {video_path}")
            err_cnt += 1
            continue

        gpt_val = rec["conversations"][1]["value"]
        grounds, warrant, answer = parse_gpt_value(gpt_val)
        if not grounds or not warrant or not answer:
            err_cnt += 1
            continue

        llava_rec = {
            "id":      rec_id,
            "pair_id": rec_id,
            "video":   video_path,
            "system":  SYSTEM_PROMPT,
            "prompt": [
                {"role": "user", "content": PROMPT_TEXT}
            ],
            "completion": [
                {"role": "assistant", "content": (
                    f"grounds: {grounds}\n"
                    f"warrant: {warrant}\n"
                    f"answer: {answer}"
                )}
            ],
            "answer":            answer,
            "aggregated_intent": meta.get("aggregated_intent", 0.5),
            "meta": {
                "video_id":          meta["video_id"],
                "track_id":          meta["track_id"],
                "annotator_id":      meta["annotator_id"],
                "frame_span":        meta["frame_span"],
                "num_frames":        NUM_FRAMES,
                "label_frame":       meta["label_frame"],
                "aggregated_intent": meta.get("aggregated_intent", 0.5),
            }
        }

        out_f.write(json.dumps(llava_rec, ensure_ascii=False) + "\n")
        ok_cnt += 1

    out_f.close()
    print(f"\nDone. ok={ok_cnt} err={err_cnt}")
    print(f"Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()