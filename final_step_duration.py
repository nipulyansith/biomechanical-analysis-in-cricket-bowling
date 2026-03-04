"""
step_duration.py  (formerly: final_step_duration.py)

✅ Requirement covered:
- Step Duration in the Final Run-Up and Delivery Stride (last 5 contacts before release)
- Outputs:
  1) <out_dir>/<trial_id>/<metric_name>_frames.csv
  2) <out_dir>/<trial_id>/<metric_name>_summary.json
  3) Annotated video (optional): <out_dir>/<trial_id>/<metric_name>_annotated.mp4
  4) Keypoints cache (reused by other scripts later): <out_dir>/<trial_id>/keypoints.csv

✅ Fix included:
- Detect foot contacts from LEFT and RIGHT ankle separately
- Merge + clean events to avoid same leg twice consecutively
"""

import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.signal import savgol_filter, find_peaks


# =========================
# KEYPOINT DEFINITIONS
# =========================
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

SELECTED_POINTS = [
    "left_wrist", "right_wrist",
    "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]


# =========================
# SIGNAL HELPERS
# =========================
def make_window(n: int, length: int) -> int:
    """Make a valid odd window length for Savitzky-Golay filter."""
    w = n if n % 2 == 1 else n + 1
    if w >= length:
        w = length - 1 if (length - 1) % 2 == 1 else length - 2
    return max(3, w)

def smooth(sig: np.ndarray, win: int, poly: int) -> np.ndarray:
    if len(sig) < 3:
        return sig.copy()
    w = make_window(win, len(sig))
    return savgol_filter(sig, window_length=w, polyorder=min(poly, w - 1))

def detect_ankle_peaks(smoothed_y: np.ndarray, fps: float, prominence: float, min_dist_s: float = 0.06):
    """
    Foot contact proxy: local maxima in ankle_y (y increases downward).
    """
    min_distance_frames = max(3, int(min_dist_s * fps))
    peaks, props = find_peaks(smoothed_y, distance=min_distance_frames, prominence=prominence)
    prominences = props.get("prominences", np.zeros_like(peaks, dtype=float))
    return peaks, prominences

def clean_step_events(events, fps: float, min_gap_s: float = 0.10):
    """
    events: list of dict {idx, foot('L'/'R'), prom, y}
    Cleans:
      1) merges duplicates too close in time (keep stronger)
      2) removes same-foot consecutive steps (keep stronger)
    """
    if not events:
        return []

    events = sorted(events, key=lambda e: e["idx"])

    min_gap = int(min_gap_s * fps)

    # 1) Remove too-close events (keep stronger)
    filtered = [events[0]]
    for e in events[1:]:
        if e["idx"] - filtered[-1]["idx"] < min_gap:
            prev = filtered[-1]
            if (e["prom"] > prev["prom"]) or (e["prom"] == prev["prom"] and e["y"] > prev["y"]):
                filtered[-1] = e
        else:
            filtered.append(e)

    # 2) Enforce alternation (no same foot twice in row)
    cleaned = [filtered[0]]
    for e in filtered[1:]:
        prev = cleaned[-1]
        if e["foot"] == prev["foot"]:
            if (e["prom"] > prev["prom"]) or (e["prom"] == prev["prom"] and e["y"] > prev["y"]):
                cleaned[-1] = e
        else:
            cleaned.append(e)

    return cleaned


# =========================
# KEYPOINT EXTRACTION (cached)
# =========================
def extract_or_load_keypoints(video_path: Path, keypoints_csv: Path, model_path: str):
    """
    Extract pose keypoints for all frames (once) and cache to keypoints_csv.
    If keypoints_csv exists, it is loaded instead (fast).
    """
    if keypoints_csv.exists():
        df = pd.read_csv(keypoints_csv)
        # basic sanity check
        if "frame" in df.columns and "left_ankle_y" in df.columns:
            return df

    model = YOLO(model_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"🎥 Frames={total_frames}, FPS={fps:.2f}")
    print("⏳ Extracting YOLO keypoints (cached)...")

    rows = []
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        data = {"frame": frame_num}
        for name in SELECTED_POINTS:
            data[f"{name}_x"] = np.nan
            data[f"{name}_y"] = np.nan

        results = model.predict(frame, verbose=False)

        if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints) > 0:
            kps = results[0].keypoints.xy
            if kps is not None and len(kps) > 0:
                pts = kps[0].cpu().numpy()  # (17,2)
                for i, name in enumerate(KEYPOINT_NAMES):
                    if name in SELECTED_POINTS:
                        data[f"{name}_x"] = float(pts[i, 0])
                        data[f"{name}_y"] = float(pts[i, 1])

        rows.append(data)

        if frame_num % 150 == 0:
            print(f"  Processed {frame_num}/{total_frames}...")

    cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(keypoints_csv, index=False)
    print(f"✅ Saved keypoints CSV: {keypoints_csv}")
    return df


# =========================
# ANNOTATION UTIL
# =========================
def draw_center_box(img, lines, width, height, font_scale=1.8, y_center=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 4
    line_gap = int(55 * font_scale)

    # measure max width
    max_w = 0
    for t in lines:
        (tw, th), _ = cv2.getTextSize(t, font, font_scale, thickness)
        max_w = max(max_w, tw)

    total_h = line_gap * len(lines)
    x = (width - max_w) // 2
    if y_center is None:
        y = 80
    else:
        y = int(y_center - total_h / 2)

    pad = 18
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x - pad, y - int(40 * font_scale)),
        (x + max_w + pad, y + total_h - int(20 * font_scale)),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    for i, t in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(t, font, font_scale, thickness)
        tx = (width - tw) // 2
        ty = y + i * line_gap
        cv2.putText(img, t, (tx, ty), font, font_scale, (255, 255, 255), thickness)


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--trial_id", required=True, help="Trial ID e.g. B01_T03")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--view_mode", default="SIDE", choices=["SIDE", "FRONT"])
    parser.add_argument("--bowling_arm", default="RIGHT", choices=["LEFT", "RIGHT"])
    parser.add_argument("--metric_name", default="step_duration")
    parser.add_argument("--model", default="yolov8n-pose.pt")

    # tuning knobs
    parser.add_argument("--smooth_window", type=int, default=7)
    parser.add_argument("--smooth_poly", type=int, default=2)
    parser.add_argument("--ankle_prom_main", type=float, default=5.0)
    parser.add_argument("--ankle_prom_relaxed", type=float, default=1.0)
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    trial_dir = out_dir / args.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)

    metric_name = args.metric_name

    # Output paths (contract with orchestrator)
    keypoints_csv = trial_dir / "keypoints.csv"
    frames_out = trial_dir / f"{metric_name}_frames.csv"
    summary_out = trial_dir / f"{metric_name}_summary.json"
    annotated_out = trial_dir / f"{metric_name}_annotated.mp4"
    debug_out = trial_dir / f"{metric_name}_debug.csv"

    # 1) Load/extract keypoints
    df = extract_or_load_keypoints(video_path, keypoints_csv, args.model)

    # Fill missing
    df = df.copy()
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # FPS (read from video to avoid missing)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 2) Release detection (bowling wrist highest => min y)
    wrist_col = "right_wrist_y" if args.bowling_arm.upper() == "RIGHT" else "left_wrist_y"
    wrist_y = df[wrist_col].values.astype(float)
    wrist_y_sm = smooth(wrist_y, args.smooth_window, args.smooth_poly)

    release_idx = int(np.nanargmin(wrist_y_sm))
    release_frame = int(df.loc[release_idx, "frame"])
    print(f"🎯 Release (arm={args.bowling_arm}): idx={release_idx}, frame={release_frame}")

    # 3) Step detection (LEFT + RIGHT separately)
    l_y = df["left_ankle_y"].values.astype(float)
    r_y = df["right_ankle_y"].values.astype(float)

    l_sm = smooth(l_y, args.smooth_window, args.smooth_poly)
    r_sm = smooth(r_y, args.smooth_window, args.smooth_poly)

    l_peaks, l_prom = detect_ankle_peaks(l_sm, fps, prominence=args.ankle_prom_main)
    r_peaks, r_prom = detect_ankle_peaks(r_sm, fps, prominence=args.ankle_prom_main)

    # before release only
    l_mask = l_peaks < release_idx
    r_mask = r_peaks < release_idx
    l_peaks, l_prom = l_peaks[l_mask], l_prom[l_mask]
    r_peaks, r_prom = r_peaks[r_mask], r_prom[r_mask]

    # relax if needed
    if len(l_peaks) < 3 or len(r_peaks) < 3:
        l2, lp2 = detect_ankle_peaks(l_sm, fps, prominence=args.ankle_prom_relaxed)
        r2, rp2 = detect_ankle_peaks(r_sm, fps, prominence=args.ankle_prom_relaxed)

        l2 = l2[l2 < release_idx]
        r2 = r2[r2 < release_idx]

        # merge peaks (prominence fallback to y if mismatch)
        l_peaks = np.unique(np.concatenate([l_peaks, l2]))
        r_peaks = np.unique(np.concatenate([r_peaks, r2]))

    # build events list with foot labels
    events = []
    # use prominence where available; otherwise use y-value as strength
    for i, p in enumerate(l_peaks):
        yv = float(l_sm[p])
        pv = float(l_prom[i]) if i < len(l_prom) else yv
        events.append({"idx": int(p), "foot": "L", "prom": pv, "y": yv})

    for i, p in enumerate(r_peaks):
        yv = float(r_sm[p])
        pv = float(r_prom[i]) if i < len(r_prom) else yv
        events.append({"idx": int(p), "foot": "R", "prom": pv, "y": yv})

    events_clean = clean_step_events(events, fps, min_gap_s=0.10)

    # Need last 5 steps before release
    if len(events_clean) < 5:
        print(f"⚠️ Not enough clean step events (found {len(events_clean)}). Saving partial outputs.")
        last5 = events_clean
    else:
        last5 = events_clean[-5:]

    last5_idx = [e["idx"] for e in last5]
    last5_frames = df.loc[last5_idx, "frame"].astype(int).tolist() if last5_idx else []
    last5_feet = [e["foot"] for e in last5]

    # compute intervals
    last5_intervals_s = []
    if len(last5_frames) >= 2:
        for a, b in zip(last5_frames[:-1], last5_frames[1:]):
            last5_intervals_s.append(float((b - a) / fps))

    step_mean = float(np.mean(last5_intervals_s)) if last5_intervals_s else None
    step_std = float(np.std(last5_intervals_s, ddof=1)) if len(last5_intervals_s) >= 2 else None
    step_cv = float(step_std / step_mean) if (step_mean and step_std is not None and step_mean != 0) else None
    total_duration_s = float((last5_frames[-1] - last5_frames[0]) / fps) if len(last5_frames) >= 2 else None

    print("✅ Last5 frames:", last5_frames)
    print("✅ Last5 feet:  ", last5_feet)
    if total_duration_s is not None:
        print(f"✅ Total duration (first→last of last5): {total_duration_s:.3f}s")

    # 4) Write FRAME-LEVEL metrics CSV for this requirement
    frames_df = pd.DataFrame({
        "frame": df["frame"].astype(int),
        "left_ankle_y": df["left_ankle_y"].astype(float),
        "right_ankle_y": df["right_ankle_y"].astype(float),
        "left_ankle_y_sm": l_sm.astype(float),
        "right_ankle_y_sm": r_sm.astype(float),
        "bowling_wrist_y": df[wrist_col].astype(float),
        "bowling_wrist_y_sm": wrist_y_sm.astype(float),
    })

    # flags
    frames_df["is_release"] = 0
    if 1 <= release_frame <= len(frames_df):
        frames_df.loc[frames_df["frame"] == release_frame, "is_release"] = 1

    frames_df["step_event"] = 0
    frames_df["step_number"] = np.nan
    frames_df["step_foot"] = np.nan

    for i, (fr, foot) in enumerate(zip(last5_frames, last5_feet), start=1):
        frames_df.loc[frames_df["frame"] == fr, "step_event"] = 1
        frames_df.loc[frames_df["frame"] == fr, "step_number"] = i
        frames_df.loc[frames_df["frame"] == fr, "step_foot"] = foot

    frames_df.to_csv(frames_out, index=False)
    print(f"📌 Saved frames CSV: {frames_out}")

    # 5) Write SUMMARY JSON (trial-level)
    summary = {
        "fps": float(fps),
        "total_frames": int(total_frames),
        "bowling_arm": args.bowling_arm.upper(),
        "view_mode": args.view_mode.upper(),
        "release_frame": int(release_frame),

        "last5_step_frames": last5_frames,     # list[int]
        "last5_step_feet": last5_feet,         # list['L'/'R']
        "last5_step_intervals_s": last5_intervals_s,  # list[float] length 4 typically

        "step_interval_mean_s": step_mean,
        "step_interval_std_s": step_std,
        "step_interval_cv": step_cv,
        "last5_total_duration_s": total_duration_s,
    }

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"📌 Saved summary JSON: {summary_out}")

    # 6) Save debug (optional but useful)
    dbg = pd.DataFrame({
        "frame": df["frame"].astype(int),
        "left_ankle_sm": l_sm,
        "right_ankle_sm": r_sm,
        "wrist_sm": wrist_y_sm,
    })
    dbg.to_csv(debug_out, index=False)
    print(f"🧪 Saved debug CSV: {debug_out}")

    # 7) Annotated trimmed video (last5 start → release)
    if len(last5_frames) >= 2:
        start_frame = int(last5_frames[0])
        end_frame = int(release_frame)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("⚠️ Could not reopen video for annotation.")
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(annotated_out), fourcc, float(fps), (width, height))

        step_label_map = {fr: f"STEP {i} ({foot})" for i, (fr, foot) in enumerate(zip(last5_frames, last5_feet), start=1)}

        frame_no = start_frame
        print(f"🎬 Writing annotated video: {annotated_out.name}  frames {start_frame}→{end_frame}")

        while frame_no <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = frame.copy()

            # step pause
            if frame_no in step_label_map:
                idx = last5_frames.index(frame_no)
                lines = [
                    step_label_map[frame_no],
                    f"Frame: {frame_no}",
                    f"Time: {frame_no / fps:.2f}s",
                ]
                if idx > 0:
                    dt = (frame_no - last5_frames[idx - 1]) / fps
                    lines.append(f"Δ since last: {dt:.2f}s")

                draw_center_box(annotated, lines, width, height, font_scale=2.0, y_center=height // 2)

                # pause ~1.0s
                for _ in range(int(fps * 1.0)):
                    out.write(annotated)
            else:
                out.write(annotated)

            # release pause
            if frame_no == release_frame:
                lines = [
                    "BALL RELEASE",
                    f"Frame: {release_frame}",
                    f"Time: {release_frame / fps:.2f}s",
                ]
                if total_duration_s is not None:
                    lines.append(f"Last5 duration: {total_duration_s:.2f}s")
                draw_center_box(annotated, lines, width, height, font_scale=2.2, y_center=height // 3)

                for _ in range(int(fps * 1.2)):
                    out.write(annotated)

            frame_no += 1

        cap.release()
        out.release()
        print(f"✅ Saved annotated video: {annotated_out}")
    else:
        print("⚠️ Skipping annotation: not enough step frames detected.")

if __name__ == "__main__":
    main()