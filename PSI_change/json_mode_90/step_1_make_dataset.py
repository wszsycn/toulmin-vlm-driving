"""
Step 1: 从原始帧和 pedestrian_intent.json 生成视频 + jsonl
每条记录包含：
- 90帧视频（带 bbox overlay）
- 该窗口内每个 annotator forward-fill 的最新 description + intent
"""

import json, os, subprocess, tempfile
from collections import defaultdict
import cv2

# ── 配置 ──────────────────────────────────────────────
# FRAME_DIR    = "/workspace/PSI/PSI/PSI2.0_videos_Train-Val/frames"
# ANNOT_DIR    = "/workspace/PSI/PSI/PSI2.0_TrainVal/annotations/cognitive_annotation_key_frame"
# OUTPUT_VIDEO = "/workspace/PSI_change/json_mode_90/videos"
# OUTPUT_JSONL = "/workspace/PSI_change/json_mode_90/psi_90f_raw.jsonl"
# WIN_SIZE     = 90
# WIN_STEP     = 45
# FPS          = 15


FRAME_DIR    = "/workspace/PSI/PSI/PSI2.0_videos_Train-Val/frames"
ANNOT_DIR    = "/workspace/PSI/PSI/PSI2.0_TrainVal/annotations/cognitive_annotation_key_frame"
OUTPUT_VIDEO = "/workspace/PSI_change/json_mode_90/videos"
OUTPUT_JSONL = "/workspace/PSI_change/json_mode_90/psi_90f_raw.jsonl"
WIN_SIZE     = 90
WIN_STEP     = 45
FPS          = 15
# ──────────────────────────────────────────────────────


# ════════════════════════════════════════════════════
# 窗口切分
# ════════════════════════════════════════════════════

def get_windows(observed_frames):
    frames = sorted(observed_frames)
    if len(frames) < WIN_SIZE:
        return []
    f_min, f_max = frames[0], frames[-1]
    windows = []
    pos = f_min
    while pos + WIN_SIZE - 1 <= f_max:
        windows.append((pos, pos + WIN_SIZE - 1))
        pos += WIN_STEP
    # 最后一个窗口从 f_max 往前数
    last_start = f_max - WIN_SIZE + 1
    if not windows or windows[-1][0] != last_start:
        windows.append((last_start, f_max))
    return windows


# ════════════════════════════════════════════════════
# Forward-fill annotation
# ════════════════════════════════════════════════════

def get_forwardfill_annotation(ann_data, observed_frames, win_end):
    """
    对该 annotator，在 win_end 帧之前，
    找最新的 key_frame=1 的 description + intent（forward fill 语义）。
    """
    key_list  = ann_data.get("key_frame",   [])
    desc_list = ann_data.get("description", [])
    int_list  = ann_data.get("intent",      [])
    prom_list = ann_data.get("promts",      [])

    best = None
    for i, fn in enumerate(observed_frames):
        if fn > win_end:
            break
        if i < len(key_list) and key_list[i] == 1:
            desc   = desc_list[i] if i < len(desc_list) else ""
            intent = int_list[i]  if i < len(int_list)  else ""
            prom   = prom_list[i] if i < len(prom_list) else {}
            if desc or intent:
                best = {
                    "description": desc,
                    "intent":      intent,
                    "promts":      prom,
                    "key_frame":   fn,
                }
    return best


# ════════════════════════════════════════════════════
# 视频合成
# ════════════════════════════════════════════════════

def safe_bbox(b, W, H):
    x1,y1,x2,y2 = [int(round(float(v))) for v in b]
    x1=max(0,min(x1,W-1)); x2=max(0,min(x2,W-1))
    y1=max(0,min(y1,H-1)); y2=max(0,min(y2,H-1))
    if x2<x1: x1,x2=x2,x1
    if y2<y1: y1,y2=y2,y1
    return x1,y1,x2,y2

def draw_label(img, text, pos=(12,42)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw,th),bl = cv2.getTextSize(text, font, 0.8, 2)
    x,y = pos
    cv2.rectangle(img,(x-6,y-th-6),(x+tw+6,y+bl+6),(0,0,0),-1)
    cv2.putText(img,text,(x,y),font,0.8,(255,255,255),2,cv2.LINE_AA)

def make_video(video_id, win_start, win_end, bbox_map, output_path) -> bool:
    """从原始帧合成带 bbox overlay 的 mp4"""
    vframe_dir = os.path.join(FRAME_DIR, video_id)
    if not os.path.isdir(vframe_dir):
        return False

    available = sorted(
        int(os.path.splitext(f)[0])
        for f in os.listdir(vframe_dir)
        if f.endswith(".jpg") and os.path.splitext(f)[0].isdigit()
    )
    if not available:
        return False

    avail_set   = set(available)
    bbox_frames = sorted(bbox_map.keys())
    frame_nums  = list(range(win_start, win_end + 1))
    T           = len(frame_nums)

    with tempfile.TemporaryDirectory() as tmpdir:
        written = []
        for idx, fn in enumerate(frame_nums):
            actual = fn if fn in avail_set else min(available, key=lambda x: abs(x-fn))
            jpg    = os.path.join(vframe_dir, f"{actual:03d}.jpg")
            img    = cv2.imread(jpg)
            if img is None:
                continue
            H, W = img.shape[:2]

            # bbox overlay
            if bbox_frames:
                near = min(bbox_frames, key=lambda x: abs(x-fn))
                if abs(near - fn) <= 10:
                    x1,y1,x2,y2 = safe_bbox(bbox_map[near], W, H)
                    ov = img.copy()
                    cv2.rectangle(ov,(x1,y1),(x2,y2),(0,255,0),-1)
                    cv2.addWeighted(ov,0.2,img,0.8,0,img)
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3)

            draw_label(img, f"frame {idx+1}/{T}")
            out_jpg = os.path.join(tmpdir, f"{idx:05d}.jpg")
            cv2.imwrite(out_jpg, img)
            written.append(out_jpg)

        if not written:
            return False

        list_file = os.path.join(tmpdir, "list.txt")
        with open(list_file, "w") as lf:
            for p in written:
                lf.write(f"file '{p}'\n")
                lf.write(f"duration {1/FPS:.6f}\n")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
               "-i", list_file, "-vcodec", "mpeg4", "-q:v", "5",
               "-pix_fmt", "yuv420p", "-an", output_path]
        ret = subprocess.run(cmd, capture_output=True)
        if ret.returncode != 0:
            print(f"  [ffmpeg error] {ret.stderr.decode()[-200:]}")
            return False

    return True


# ════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_VIDEO, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    # 断点续跑
    done_ids = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["id"])
    print(f"Already done: {len(done_ids)}")

    video_ids = sorted(
        d for d in os.listdir(ANNOT_DIR)
        if os.path.isdir(os.path.join(ANNOT_DIR, d))
    )
    print(f"Found {len(video_ids)} videos")

    out_f   = open(OUTPUT_JSONL, "a")
    ok_cnt  = 0
    err_cnt = 0

    for video_id in video_ids:
        annot_path = os.path.join(ANNOT_DIR, video_id, "pedestrian_intent.json")
        if not os.path.exists(annot_path):
            continue

        with open(annot_path) as f:
            data = json.load(f)

        for track_id, ped_data in data.get("pedestrians", {}).items():
            observed = ped_data.get("observed_frames", [])
            bboxes   = ped_data.get("cv_annotations", {}).get("bboxes", [])
            cog      = ped_data.get("cognitive_annotations", {})

            if not observed or not cog:
                continue

            bbox_map = {fn: bb for fn, bb in zip(observed, bboxes)}
            windows  = get_windows(observed)
            if not windows:
                continue

            for win_start, win_end in windows:
                # 视频：每个 (video_id, track_id, window) 只生成一次
                vid_name = f"{video_id}_{track_id}_{win_start:05d}-{win_end:05d}.mp4"
                vid_path = os.path.join(OUTPUT_VIDEO, vid_name)
                if not os.path.exists(vid_path):
                    ok = make_video(video_id, win_start, win_end, bbox_map, vid_path)
                    if not ok:
                        err_cnt += 1
                        continue

                # 每个 annotator 生成一条 jsonl 记录
                for ann_id, ann_data in cog.items():
                    rec_id = f"{video_id}_{track_id}_{ann_id}_{win_start:05d}-{win_end:05d}"
                    if rec_id in done_ids:
                        continue

                    ann = get_forwardfill_annotation(ann_data, observed, win_end)
                    if ann is None:
                        err_cnt += 1
                        continue

                    # skip not_sure
                    intent = ann["intent"]
                    if intent not in ("cross", "not_cross"):
                        err_cnt += 1
                        continue

                    rec = {
                        "id":    rec_id,
                        "video": vid_path,
                        "meta": {
                            "video_id":     video_id,
                            "track_id":     track_id,
                            "annotator_id": ann_id,
                            "frame_span":   [win_start, win_end],
                            "label_frame":  win_end + 1,
                            "intent":       intent,
                            "key_frame":    ann["key_frame"],
                        },
                        # 原始 annotation，供 step2 agent 使用
                        "raw_annotation": {
                            "description": ann["description"],
                            "promts":      ann["promts"],
                            "intent":      intent,
                        },
                    }

                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    ok_cnt += 1

        print(f"  {video_id}: ok={ok_cnt} err={err_cnt}")

    out_f.close()
    print(f"\nDone. ok={ok_cnt} err={err_cnt}")
    print(f"Output JSONL: {OUTPUT_JSONL}")
    print(f"Output VIDEO: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()