#!/usr/bin/env python3
"""
build_test_videos.py — Build PSI 2.0 test-set mp4 clips with per-track
canonical windowing.

Encodes test clips in EXACTLY the same format as training videos in
PSI_change/json_mode_90/videos/:
  codec      : mpeg4  (-vcodec mpeg4 -q:v 5)
  container  : mp4
  fps        : 15
  resolution : native source frame resolution (no resize)
  overlay    : green semi-transparent bbox fill (alpha=0.2) + solid green
               outline (thickness=3), plus "frame N/T" counter label
               (white text, black background, top-left corner)

Windowing difference vs step_1_make_dataset.py
-----------------------------------------------
step_1 (training): windows are computed PER ANNOTATOR — each annotator's
  `observed_frames` may differ, yielding different windows per annotator for
  the same track. One JSONL record is written per (track, window, annotator).
  One mp4 is written per (track, window) — shared across annotators.

THIS SCRIPT (test): windows are computed PER TRACK using the track's own
  `observed_frames` from pedestrian_intent.json. This is the union of all
  frames with bbox annotations. One mp4 per (track, window), no annotator
  split. A downstream eval-jsonl builder will read these mp4s.

safe_bbox(), draw_label(), and make_video() are copied VERBATIM from
step_1_make_dataset.py. Only FRAME_DIR differs (test instead of train/val).

Run on HOST (not inside Docker) — PSI source frames are not mounted in the
container. Requires: opencv-python, ffmpeg.
  conda create -n build_test python=3.11
  conda install -c conda-forge opencv ffmpeg
  python build_test_videos.py
  python build_test_videos.py --smoke   # first 5 windows
  python build_test_videos.py --verify  # ffprobe spot-check

Input paths (host):
  Frames  : /mnt/DATA/Shaozhi/PSI/PSI/PSI2.0_Test/frames/<video_id>/<frame>.jpg
  Annots  : /mnt/DATA/Shaozhi/PSI/PSI/PSI2.0_Test/annotations/
                cognitive_annotation_key_frame/<video_id>/pedestrian_intent.json

Output (host path — accessible inside Docker as /workspace/PSI_change/...):
  /mnt/DATA/Shaozhi/cosmos2_sft/PSI_change/json_mode_90/test_videos/
  filename: video_<ID>_track_<T>_<SSSSS>-<EEEEE>.mp4
"""

import argparse
import glob
import json
import os
import random
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2

# ── Paths (host absolute — script runs on host, not inside Docker) ────────────
# Output is inside cosmos2_sft so Docker sees it as /workspace/PSI_change/...
FRAME_DIR  = "/mnt/DATA/Shaozhi/PSI/PSI/PSI2.0_Test/frames"
ANNOT_DIR  = "/mnt/DATA/Shaozhi/PSI/PSI/PSI2.0_Test/annotations/cognitive_annotation_key_frame"
OUTPUT_DIR = "/mnt/DATA/Shaozhi/cosmos2_sft/PSI_change/json_mode_90/test_videos"

WIN_SIZE = 90
WIN_STEP = 45
FPS      = 15


# ════════════════════════════════════════════════════════════════════════════
# Verbatim copy from step_1_make_dataset.py — DO NOT MODIFY
# (safe_bbox, draw_label, make_video)
# ════════════════════════════════════════════════════════════════════════════

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
        # H.264 + explicit -r 15 so every ffmpeg version produces exactly 90
        # frames at 15 fps regardless of host defaults.
        # mpeg4/mp4v (used by step_1) is not supported by modern macOS and
        # silently shows green; libx264 is universally compatible.
        # decord decodes both codecs identically — no impact on model training.
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
               "-i", list_file,
               "-vcodec", "libx264", "-crf", "18", "-preset", "fast",
               "-r", str(FPS), "-pix_fmt", "yuv420p", "-an", output_path]
        ret = subprocess.run(cmd, capture_output=True)
        if ret.returncode != 0:
            print(f"  [ffmpeg error] {ret.stderr.decode()[-200:]}")
            return False

    return True


# ════════════════════════════════════════════════════
# Per-track canonical windowing
# ════════════════════════════════════════════════════

def get_windows_for_track(observed_frames: list) -> list:
    """
    Same stride logic as step_1_make_dataset.get_windows(), applied to the
    full track observed_frames (canonical, annotator-independent).
    Returns list of (win_start, win_end). Empty if track < WIN_SIZE frames.
    """
    frames = sorted(observed_frames)
    if len(frames) < WIN_SIZE:
        return []
    f_min, f_max = frames[0], frames[-1]
    windows = []
    pos = f_min
    while pos + WIN_SIZE - 1 <= f_max:
        windows.append((pos, pos + WIN_SIZE - 1))
        pos += WIN_STEP
    # Trailing window anchored at f_max (same as step_1)
    last_start = f_max - WIN_SIZE + 1
    if not windows or windows[-1][0] != last_start:
        windows.append((last_start, f_max))
    return windows


# ════════════════════════════════════════════════════
# Enumerate all canonical windows from test annotations
# ════════════════════════════════════════════════════

def enumerate_windows(annot_dir: str) -> list:
    """
    Reads all pedestrian_intent.json files under annot_dir.
    Returns list of (video_id, track_id, win_start, win_end, bbox_map).
    """
    cog_files = sorted(glob.glob(
        os.path.join(annot_dir, "video_*/pedestrian_intent.json")
    ))
    result = []
    short_skipped = 0
    for cf in cog_files:
        video_id = os.path.basename(os.path.dirname(cf))
        with open(cf) as f:
            data = json.load(f)
        for track_id, tdata in data.get("pedestrians", {}).items():
            observed = tdata.get("observed_frames", [])
            bboxes   = tdata.get("cv_annotations", {}).get("bboxes", [])
            bbox_map = {fn: bb for fn, bb in zip(observed, bboxes)}
            wins = get_windows_for_track(observed)
            if not wins:
                short_skipped += 1
                print(f"  [skip-short] {video_id}/{track_id}: "
                      f"{len(observed)} frames < {WIN_SIZE}")
                continue
            for ws, we in wins:
                result.append((video_id, track_id, ws, we, bbox_map))
    if short_skipped:
        print(f"Skipped {short_skipped} tracks with fewer than {WIN_SIZE} frames")
    return result


# ════════════════════════════════════════════════════
# Verification helpers
# ════════════════════════════════════════════════════

def _ffprobe_frames(mp4_path: str):
    """Return (n_frames, width, height, fps_str) via ffprobe, or None on error."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate",
        "-of", "json", mp4_path,
    ]
    ret = subprocess.run(cmd, capture_output=True, timeout=30)
    if ret.returncode != 0:
        return None
    info = json.loads(ret.stdout)
    s = info.get("streams", [{}])[0]
    return (
        int(s.get("nb_read_frames", -1)),
        int(s.get("width", -1)),
        int(s.get("height", -1)),
        s.get("r_frame_rate", "?"),
    )


def run_verify(output_dir: str, n_sample: int = 3):
    """Sample n_sample built mp4s and run ffprobe on each."""
    mp4s = sorted(glob.glob(os.path.join(output_dir, "*.mp4")))
    if not mp4s:
        print("[verify] No mp4s found in output_dir.")
        return
    print(f"\n[verify] Sampling {n_sample} of {len(mp4s)} mp4s via ffprobe ...")
    sample = random.sample(mp4s, min(n_sample, len(mp4s)))
    all_ok = True
    for p in sample:
        result = _ffprobe_frames(p)
        if result is None:
            print(f"  FAIL (ffprobe error): {os.path.basename(p)}")
            all_ok = False
            continue
        n_frames, w, h, fps = result
        ok = (n_frames == WIN_SIZE)
        status = "OK" if ok else f"FAIL (frames={n_frames}, expected {WIN_SIZE})"
        print(f"  {status}  frames={n_frames}  {w}x{h}  fps={fps}  "
              f"{os.path.basename(p)}")
        if not ok:
            all_ok = False
    print("[verify]", "All sampled mp4s OK." if all_ok else "Some mp4s FAILED verification.")


# ════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Build PSI 2.0 test mp4 clips with canonical per-track windowing"
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Build only the first 5 windows (sanity check)")
    parser.add_argument("--verify", action="store_true",
                        help="Run ffprobe verification on 3 random mp4s after build")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Frame dir  : {FRAME_DIR}")
    print(f"Annot dir  : {ANNOT_DIR}")
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"WIN_SIZE={WIN_SIZE}  WIN_STEP={WIN_STEP}  FPS={FPS}")
    print()

    # ── Step 2 — Source video discovery ──────────────────────────────────────
    test_frame_dirs = sorted(
        d for d in os.listdir(FRAME_DIR)
        if os.path.isdir(os.path.join(FRAME_DIR, d))
    ) if os.path.isdir(FRAME_DIR) else []
    print(f"Test source frame directories found: {len(test_frame_dirs)}")
    print(f"  path: {FRAME_DIR}")
    print()

    # ── Step 3 — Enumerate canonical per-track windows ───────────────────────
    windows = enumerate_windows(ANNOT_DIR)
    unique_vids   = len({w[0] for w in windows})
    unique_tracks = len({(w[0], w[1]) for w in windows})
    print(f"Unique videos : {unique_vids}")
    print(f"Unique tracks : {unique_tracks}")
    print(f"Total windows : {len(windows)}")
    print()

    if args.smoke:
        windows = windows[:5]
        print(f"[smoke] limited to {len(windows)} windows")
        print()

    # ── Step 4 — Build mp4s ───────────────────────────────────────────────────
    built = skipped = failed = 0
    failed_reasons: dict = defaultdict(int)

    for i, (video_id, track_id, ws, we, bbox_map) in enumerate(windows):
        fname    = f"{video_id}_{track_id}_{ws:05d}-{we:05d}.mp4"
        out_path = os.path.join(OUTPUT_DIR, fname)

        if os.path.exists(out_path):
            skipped += 1
        else:
            ok = make_video(video_id, ws, we, bbox_map, out_path)
            if ok:
                built += 1
            else:
                failed += 1
                vdir = os.path.join(FRAME_DIR, video_id)
                reason = "no_frame_dir" if not os.path.isdir(vdir) else "ffmpeg_error"
                failed_reasons[reason] += 1
                print(f"  [FAIL] {fname}  ({reason})")

        if (i + 1) % 50 == 0 or (i + 1) == len(windows):
            print(f"  [{i+1:4d}/{len(windows)}]  "
                  f"built={built}  skipped={skipped}  failed={failed}")

    # ── Step 5 — Summary ──────────────────────────────────────────────────────
    mp4s_on_disk = len(glob.glob(os.path.join(OUTPUT_DIR, "*.mp4")))
    print(f"\n{'='*60}")
    print(f"unique_videos  : {unique_vids}")
    print(f"unique_tracks  : {unique_tracks}")
    print(f"total_windows  : {len(windows)}")
    print(f"mp4s_built     : {built}")
    print(f"mp4s_skipped   : {skipped}  (already existed)")
    print(f"mp4s_failed    : {failed}")
    if failed_reasons:
        for r, c in failed_reasons.items():
            print(f"  {r}: {c}")
    print(f"mp4s_on_disk   : {mp4s_on_disk}")
    print(f"output_dir     : {OUTPUT_DIR}")
    print(f"{'='*60}")

    if args.verify or args.smoke:
        run_verify(OUTPUT_DIR, n_sample=3)

    print()
    print("Test mp4s built. Now run the eval-jsonl prompt to produce "
          "psi_test_eval_v2.jsonl pointing to these mp4s.")


if __name__ == "__main__":
    main()
