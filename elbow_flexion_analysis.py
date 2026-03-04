"""
elbow_flexion.py  (converted from: elbow_flexion_analysis.py)

✅ Requirement covered:
- Elbow Flexion-Extension from Arm-Back to Ball Release
- Outputs (contract with run_all.py):
  1) <out_dir>/<trial_id>/<metric_name>_frames.csv
  2) <out_dir>/<trial_id>/<metric_name>_summary.json
  3) <out_dir>/<trial_id>/<metric_name>_annotated.mp4

✅ Major fixes for your pipeline:
- NO input() / no auto video search. Uses --video, --trial_id, --out_dir.
- Reuses cached pose keypoints: <out_dir>/<trial_id>/keypoints.csv (from step_duration.py).
- Uses the SAME release frame as other scripts (from step_duration_summary.json),
  so all parameters align in time.
- Adds a PRACTICAL arm-back detection (simple + robust):
    Arm-back frame = frame BEFORE the main “upswing” of bowling wrist:
      we find the last local maximum of wrist_y (lowest point) before release within a window,
      then take a few frames after that as arm-back start.
  (This is usable for a thesis; and consistent across trials.)
- Computes elbow angle time series and angular velocity.
- Computes delta angle between arm-back and release.

✅ NOTE about “arm back”:
- In side-view, the arm-back phase typically occurs when bowling wrist is down/behind the body
  (wrist_y is high, because y increases downward).
- We approximate arm-back as the last “wrist down” (local max wrist_y) before the rapid rise to release.

If you later want “manual override”, we can add --arm_back_frame.
"""

import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks


# =========================
# Helpers
# =========================
def calculate_angle(a, b, c):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return np.nan
    cosine_angle = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))

def make_window(n: int, length: int) -> int:
    w = n if n % 2 == 1 else n + 1
    if w >= length:
        w = length - 1 if (length - 1) % 2 == 1 else length - 2
    return max(3, w)

def smooth(sig: np.ndarray, win: int, poly: int) -> np.ndarray:
    if len(sig) < 3:
        return sig.copy()
    w = make_window(win, len(sig))
    return savgol_filter(sig, window_length=w, polyorder=min(poly, w - 1))

def safe_xy(df: pd.DataFrame, name: str, idx: int):
    x = float(df.loc[idx, f"{name}_x"])
    y = float(df.loc[idx, f"{name}_y"])
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return np.array([x, y], dtype=float)

def draw_overlay(frame, elbow_angle, ang_vel, label_lines, arm_pts=None):
    """
    arm_pts: tuple(shoulder_xy, elbow_xy, wrist_xy) (np arrays)
    """
    h, w = frame.shape[:2]

    if arm_pts is not None:
        s, e, wr = arm_pts
        s = tuple(s.astype(int))
        e = tuple(e.astype(int))
        wr = tuple(wr.astype(int))
        cv2.line(frame, s, e, (255, 255, 0), 3)
        cv2.line(frame, e, wr, (255, 255, 0), 3)
        cv2.circle(frame, s, 8, (0, 255, 0), -1)
        cv2.circle(frame, e, 8, (255, 0, 0), -1)
        cv2.circle(frame, wr, 8, (0, 0, 255), -1)

    # metrics box
    x0, y0 = 25, 40
    box_w, box_h = 520, 190
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"Elbow angle: {elbow_angle:6.1f} deg", (x0 + 15, y0 + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Ang vel:     {ang_vel:6.1f} deg/s", (x0 + 15, y0 + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    yy = y0 + 135
    for line in label_lines[:2]:
        cv2.putText(frame, line, (x0 + 15, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        yy += 40


# =========================
# Arm-back detection heuristic
# =========================
def detect_arm_back_idx(wrist_y_sm: np.ndarray, release_idx: int, fps: float):
    """
    Heuristic:
    - Look in a window before release (e.g., 1.5 seconds).
    - Find last local MAX of wrist_y (wrist is lowest in image = arm down/back).
    - That peak indicates the end of the backswing-down position.
    - Arm-back frame is shortly after that peak (a few frames), when the arm starts rising.
    """
    n = len(wrist_y_sm)
    if n < 10:
        return max(0, release_idx - 5)

    lookback = int(1.5 * fps)
    start = max(0, release_idx - lookback)
    end = max(start + 5, release_idx)

    segment = wrist_y_sm[start:end]
    # local maxima
    min_dist = max(3, int(0.08 * fps))
    peaks, _ = find_peaks(segment, distance=min_dist)
    if len(peaks) == 0:
        # fallback: take frame where wrist_y is highest in window (lowest wrist position)
        peak_rel = int(np.nanargmax(segment))
    else:
        peak_rel = int(peaks[-1])  # last peak before release

    peak_idx = start + peak_rel

    # small offset forward so we are at "arm starting to come up"
    offset = max(1, int(0.03 * fps))  # ~30ms
    arm_back_idx = min(release_idx, peak_idx + offset)
    return int(arm_back_idx)


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--trial_id", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--view_mode", default="SIDE", choices=["SIDE", "FRONT"])
    parser.add_argument("--bowling_arm", default="RIGHT", choices=["LEFT", "RIGHT"])
    parser.add_argument("--metric_name", default="elbow_flexion")

    # tuning
    parser.add_argument("--smooth_window", type=int, default=9)
    parser.add_argument("--smooth_poly", type=int, default=2)

    # optional override
    parser.add_argument("--arm_back_frame", type=int, default=None,
                        help="Optional manual override for arm-back frame (1-based video frame)")

    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    trial_dir = out_dir / args.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)

    metric = args.metric_name

    # expected cached inputs
    keypoints_csv = trial_dir / "keypoints.csv"
    step_summary = trial_dir / "step_duration_summary.json"

    if not keypoints_csv.exists():
        raise FileNotFoundError(
            f"Missing {keypoints_csv}. Run step_duration.py first to create cached keypoints."
        )

    # outputs
    frames_out = trial_dir / f"{metric}_frames.csv"
    summary_out = trial_dir / f"{metric}_summary.json"
    annotated_out = trial_dir / f"{metric}_annotated.mp4"

    # fps + dimensions
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"🎥 Frames={total_frames}, FPS={fps:.2f}")

    df = pd.read_csv(keypoints_csv)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # choose joints by bowling arm
    if args.bowling_arm.upper() == "RIGHT":
        shoulder_kp, elbow_kp, wrist_kp = "right_shoulder", "right_elbow", "right_wrist"
    else:
        shoulder_kp, elbow_kp, wrist_kp = "left_shoulder", "left_elbow", "left_wrist"

    # release frame: prefer step_duration summary to align with other scripts
    if step_summary.exists():
        with open(step_summary, "r", encoding="utf-8") as f:
            ss = json.load(f)
        release_frame = int(ss.get("release_frame"))
    else:
        # fallback: wrist highest (min y)
        wrist_y = df[f"{wrist_kp}_y"].astype(float).to_numpy()
        wrist_y_sm = smooth(wrist_y, args.smooth_window, args.smooth_poly)
        release_idx = int(np.nanargmin(wrist_y_sm))
        release_frame = int(df.loc[release_idx, "frame"])

    # indices
    release_idx = int(df.index[df["frame"] == release_frame][0])

    # detect arm-back
    wrist_y = df[f"{wrist_kp}_y"].astype(float).to_numpy()
    wrist_y_sm = smooth(wrist_y, args.smooth_window, args.smooth_poly)

    if args.arm_back_frame is not None:
        arm_back_frame = int(args.arm_back_frame)
        if arm_back_frame < 1 or arm_back_frame > int(df["frame"].max()):
            raise ValueError("arm_back_frame out of range")
        arm_back_idx = int(df.index[df["frame"] == arm_back_frame][0])
    else:
        arm_back_idx = detect_arm_back_idx(wrist_y_sm, release_idx, fps)
        arm_back_frame = int(df.loc[arm_back_idx, "frame"])

    print(f"🟠 Arm-back frame (estimated): {arm_back_frame}")
    print(f"🏏 Release frame: {release_frame}")

    # compute elbow angle + angular velocity (full clip OR at least from arm-back→release)
    angles = np.full(len(df), np.nan, dtype=float)
    for i in range(len(df)):
        s = safe_xy(df, shoulder_kp, i)
        e = safe_xy(df, elbow_kp, i)
        w = safe_xy(df, wrist_kp, i)
        if s is None or e is None or w is None:
            continue
        angles[i] = calculate_angle(s, e, w)

    # smooth angle slightly for velocity stability
    angle_sm = angles.copy()
    finite_mask = np.isfinite(angle_sm)
    if finite_mask.sum() > 10:
        # fill gaps quickly for smoothing then restore NaNs
        tmp = pd.Series(angle_sm).interpolate().bfill().ffill().to_numpy()
        tmp_sm = smooth(tmp, args.smooth_window, args.smooth_poly)
        angle_sm = tmp_sm
        angle_sm[~finite_mask] = np.nan

    # angular velocity (deg/s)
    ang_vel = np.full(len(df), np.nan, dtype=float)
    for i in range(1, len(df)):
        if np.isfinite(angle_sm[i]) and np.isfinite(angle_sm[i - 1]):
            ang_vel[i] = (angle_sm[i] - angle_sm[i - 1]) * float(fps)

    # key values at arm-back and release
    ang_arm_back = float(angle_sm[arm_back_idx]) if np.isfinite(angle_sm[arm_back_idx]) else float(angles[arm_back_idx])
    ang_release = float(angle_sm[release_idx]) if np.isfinite(angle_sm[release_idx]) else float(angles[release_idx])
    delta_theta = float(ang_arm_back - ang_release) if (np.isfinite(ang_arm_back) and np.isfinite(ang_release)) else None

    # -------------------------
    # FRAME-LEVEL OUTPUT
    # -------------------------
    frames_df = pd.DataFrame({
        "frame": df["frame"].astype(int),
        "time_s": df["frame"].astype(float) / float(fps),

        f"{shoulder_kp}_x": df[f"{shoulder_kp}_x"].astype(float),
        f"{shoulder_kp}_y": df[f"{shoulder_kp}_y"].astype(float),
        f"{elbow_kp}_x": df[f"{elbow_kp}_x"].astype(float),
        f"{elbow_kp}_y": df[f"{elbow_kp}_y"].astype(float),
        f"{wrist_kp}_x": df[f"{wrist_kp}_x"].astype(float),
        f"{wrist_kp}_y": df[f"{wrist_kp}_y"].astype(float),

        "elbow_angle_deg": angles.astype(float),
        "elbow_angle_deg_sm": angle_sm.astype(float),
        "elbow_ang_vel_deg_s": ang_vel.astype(float),

        "is_arm_back": 0,
        "is_release": 0,
    })
    frames_df.loc[frames_df["frame"] == arm_back_frame, "is_arm_back"] = 1
    frames_df.loc[frames_df["frame"] == release_frame, "is_release"] = 1

    frames_df.to_csv(frames_out, index=False)
    print(f"📌 Saved frames CSV: {frames_out}")

    # -------------------------
    # SUMMARY JSON
    # -------------------------
    summary = {
        "fps": float(fps),
        "bowling_arm": args.bowling_arm.upper(),
        "view_mode": args.view_mode.upper(),

        "arm_back_frame": int(arm_back_frame),
        "release_frame": int(release_frame),

        "elbow_angle_arm_back_deg": float(ang_arm_back) if np.isfinite(ang_arm_back) else None,
        "elbow_angle_release_deg": float(ang_release) if np.isfinite(ang_release) else None,
        "delta_elbow_extension_deg": float(delta_theta) if delta_theta is not None else None,

        "peak_elbow_ang_vel_deg_s": float(np.nanmax(ang_vel)) if np.isfinite(ang_vel).any() else None,
    }

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"📌 Saved summary JSON: {summary_out}")

    # -------------------------
    # ANNOTATED VIDEO (arm-back → release segment, with pauses)
    # -------------------------
    start_frame = int(arm_back_frame)
    end_frame = int(release_frame)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("⚠️ Could not reopen video for annotation.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - 1))
    out = cv2.VideoWriter(
        str(annotated_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height)
    )

    pause_arm_back = int(1.0 * fps)
    pause_release = int(1.2 * fps)
    slow_factor = 4  # duplicate frames to slow down

    frame_no = start_frame
    while frame_no <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        idx = int(df.index[df["frame"] == frame_no][0])

        s = safe_xy(df, shoulder_kp, idx)
        e = safe_xy(df, elbow_kp, idx)
        wpt = safe_xy(df, wrist_kp, idx)

        angle_now = float(angle_sm[idx]) if np.isfinite(angle_sm[idx]) else (float(angles[idx]) if np.isfinite(angles[idx]) else np.nan)
        vel_now = float(ang_vel[idx]) if np.isfinite(ang_vel[idx]) else 0.0

        labels = []
        if frame_no == arm_back_frame:
            labels.append("EVENT: ARM-BACK")
        if frame_no == release_frame:
            labels.append("EVENT: RELEASE")

        arm_pts = (s, e, wpt) if (s is not None and e is not None and wpt is not None) else None
        draw_overlay(frame, angle_now, vel_now, labels, arm_pts=arm_pts)

        # repeats
        repeats = slow_factor
        if frame_no == arm_back_frame:
            repeats += pause_arm_back
        if frame_no == release_frame:
            repeats += pause_release

        for _ in range(repeats):
            out.write(frame)

        frame_no += 1

    cap.release()
    out.release()
    print(f"🎥 Annotated video saved: {annotated_out}")
    print("✅ Done.")


if __name__ == "__main__":
    main()