import json
import re
from pathlib import Path


INPUT_JSONL = "/mnt/DATA/Shaozhi/PSI_change/json_mode_16/psi_vllm2_finetune_video_only_infer_jsons/intent+grounds+warrant_answer_overlay__Qexplain_1000.jsonl"
OUTPUT_JSONL = "/mnt/DATA/Shaozhi/cosmos2_sft/psi_video_eval_1000.jsonl"


SYSTEM_PROMPT = (
    "You are an expert assistant for pedestrian-intention reasoning from driving videos.\n"
    "Focus only on the TARGET pedestrian highlighted by the green box.\n"
    "Use concise and factual language.\n"
    "Grounds must describe visible evidence from the video.\n"
    "Warrant must provide a general rule linking the observed evidence to the pedestrian’s likely intention.\n"
    "Output EXACTLY 3 lines in this order:\n"
    "grounds: <1–2 sentences, concrete observations>\n"
    "warrant: <1 sentence general rule linking grounds to intention>\n"
    "answer: <yes/no>\n"
    "Do not add any extra text before or after these 3 lines."
)

USER_PROMPT = (
    "You are given a short driving video.\n"
    "The TARGET pedestrian is highlighted by the green box.\n"
    "Reason about the pedestrian's motion and change over time across the full video.\n"
    "Task: Predict whether the TARGET pedestrian has crossing intention at t+1."
)


def extract_answer(text: str) -> str:
    m = re.search(r"answer:\s*(yes|no)", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return "unknown"


def convert_one(item: dict) -> dict:
    assistant_text = item["conversations"][1]["value"]
    gt_answer = extract_answer(assistant_text)

    out = {
        "id": item["id"],
        "pair_id": item.get("pair_id"),
        "video_path": item["video"],
        "aggregated_intent": item.get("meta", {}).get("aggregated_intent"),
        "video_id": item.get("meta", {}).get("video_id"),
        "track_id": item.get("meta", {}).get("track_id"),
        "label_frame": item.get("meta", {}).get("label_frame"),
        "annotator_id": item.get("meta", {}).get("annotator_id"),
        "frame_span": item.get("meta", {}).get("frame_span"),
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": item["video"]},
                    {"type": "text", "text": USER_PROMPT}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_text}
                ]
            }
        ],
        "ground_truth_text": assistant_text,
        "ground_truth_answer": gt_answer,
    }
    return out


def main():
    input_path = Path(INPUT_JSONL)
    output_path = Path(OUTPUT_JSONL)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            out = convert_one(item)
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1

    print(f"Saved {count} samples to: {output_path}")


if __name__ == "__main__":
    main()