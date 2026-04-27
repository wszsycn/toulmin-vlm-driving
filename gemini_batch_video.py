"""
Gemini 1.5 Pro - 批量视频分析脚本
用途：批量处理本地视频，输出 grounds / warrant / answer
"""

import os
import time
import csv
import json
from pathlib import Path
import google.generativeai as genai

# ============================================================
# 配置区（按需修改）
# ============================================================

API_KEY = os.environ.get("GEMINI_API_KEY", "")  # set via: export GEMINI_API_KEY=...
VIDEO_DIR = "/workspace/PSI_change/json_mode_90/videos"                  # 本地视频文件夹路径
OUTPUT_CSV = "/workspace/Gemini_Output/results.csv"            # 输出结果文件
MODEL_NAME = "gemini-3-flash-preview"          # 模型名称
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

"""
Gemini 1.5 Pro - 批量视频分析脚本
用途：批量处理本地视频，输出 grounds / warrant / answer
"""


SYSTEM_PROMPT = """You are an expert assistant for pedestrian-intention reasoning from driving videos.
Focus only on the TARGET pedestrian highlighted by the green box.
Use concise and factual language.
Grounds must describe visible evidence from the video.
Warrant must provide a general rule linking the observed evidence to the pedestrian's likely intention.
Output EXACTLY 3 lines in this order:
grounds: <1-2 sentences, concrete observations>
warrant: <1 sentence general rule linking grounds to intention>
answer: <yes/no>
Do not add any extra text before or after these 3 lines."""

UPLOAD_WAIT_SECONDS = 10   # 等待视频上传完成的轮询间隔（秒）
REQUEST_DELAY_SECONDS = 2  # 每个请求之间的间隔，避免触发限流
MAX_VIDEOS = 5                          # 最多处理几个视频，设为 None 则处理全部

# ============================================================
# 初始化
# ============================================================

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name=MODEL_NAME)


def upload_video(video_path: str):
    """上传视频到 Gemini File API，等待处理完成后返回 file 对象"""
    print(f"  上传中: {video_path}")
    video_file = genai.upload_file(path=video_path)

    # 等待视频处理完成
    while video_file.state.name == "PROCESSING":
        print(f"  等待视频处理... (状态: {video_file.state.name})")
        time.sleep(UPLOAD_WAIT_SECONDS)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(f"视频处理失败: {video_path}")

    print(f"  上传完成: {video_file.name}")
    return video_file


def parse_response(text: str) -> dict:
    """解析模型输出，提取 grounds / warrant / answer"""
    result = {"grounds": "", "warrant": "", "answer": "", "raw": text}
    for line in text.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("grounds:"):
            result["grounds"] = line[len("grounds:"):].strip()
        elif line.lower().startswith("warrant:"):
            result["warrant"] = line[len("warrant:"):].strip()
        elif line.lower().startswith("answer:"):
            result["answer"] = line[len("answer:"):].strip()
    return result


def analyze_video(video_path: str) -> dict:
    """上传并分析单个视频，返回解析结果"""
    video_file = upload_video(video_path)
    try:
        response = model.generate_content(
            [video_file, SYSTEM_PROMPT],
            generation_config={"temperature": 0.2}
        )
        parsed = parse_response(response.text)
    finally:
        # 分析完成后删除上传的文件（节省配额）
        genai.delete_file(video_file.name)
        print(f"  已删除云端文件: {video_file.name}")

    return parsed


def main():
    video_dir = Path(VIDEO_DIR)
    if not video_dir.exists():
        print(f"错误：视频文件夹不存在: {VIDEO_DIR}")
        return

    # 收集所有支持的视频文件
    video_files = sorted([
        p for p in video_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])
    if MAX_VIDEOS is not None:
        video_files = video_files[:MAX_VIDEOS]

    if not video_files:
        print(f"没有找到视频文件（支持格式: {SUPPORTED_EXTENSIONS}）")
        return

    print(f"找到 {len(video_files)} 个视频，开始处理...\n")

    # 写入 CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["filename", "grounds", "warrant", "answer", "raw", "error"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, video_path in enumerate(video_files, 1):
            print(f"[{i}/{len(video_files)}] 处理: {video_path.name}")
            row = {"filename": video_path.name, "grounds": "", "warrant": "",
                   "answer": "", "raw": "", "error": ""}
            try:
                result = analyze_video(str(video_path))
                row.update(result)
                print(f"  ✅ answer: {result['answer']}")
            except Exception as e:
                row["error"] = str(e)
                print(f"  ❌ 错误: {e}")

            writer.writerow(row)
            csvfile.flush()  # 每条结果立即写入，防止中断丢失

            if i < len(video_files):
                time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\n全部完成！结果保存至: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()