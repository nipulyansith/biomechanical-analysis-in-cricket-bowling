"""
front_knee.py  (converted from your front knee flexion script)

✅ Requirement covered:
- Front Knee Flexion-Extension (FFC → Release)
- Outputs (contract with run_all.py):
  1) <out_dir>/<trial_id>/<metric_name>_frames.csv
  2) <out_dir>/<trial_id>/<metric_name>_summary.json
  3) <out_dir>/<trial_id>/<metric_name>_annotated.mp4

✅ Pipeline fixes:
- NO YOLO in this script. It reuses: <out_dir>/<trial_id>/keypoints.csv
- NO input() selection of knee side.
  We choose front leg side based on bowling arm:
    RIGHT-arm bowler -> front leg = LEFT
    LEFT-arm bowler  -> front leg = RIGHT
- Uses the SAME FFC frame as delivery_stride.py summary (recommended), so alignment is perfect.
- Uses the SAME release frame as step_duration_summary.json.
- Computes knee angle per frame (hip-knee-ankle).
- Computes knee angular velocity deg/s.
- Computes:
    knee_angle_at_ffc
    knee_angle_at_release
    delta_knee_angle (release - ffc)  (you can flip sign if you prefer)
    peak_knee_extension_velocity (max +ve ang vel if angle increases)
    peak_knee_flexion_velocity (min -ve ang vel)

✅ Annotated video:
- Trims FFC → Release
- Draws hip-knee-ankle lines + knee angle + velocity
- Pauses at FFC and Release, slow motion via duplication

"""

import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# =========================
# Helpers
# =========================
def angle_deg(a, b, c):
    """Angle at point b (a-b-c) in degrees."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-9:
        return np.nan
    cosang = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))

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

def draw_overlay(frame, knee_angle, knee_vel, label_lines, pts=None):
    """
    pts: tuple(hip_xy, knee_xy, ankle_xy)
    """
    h, w = frame.shape[:2]

    if pts is not None:
        hip, knee, ankle = pts
        hip = tuple(hip.astype(int))
        knee = tuple(knee.astype(int))
        ankle = tuple(ankle.astype(int))
        cv2.line(frame, hip, knee, (255, 255, 255), 2)
        cv2.line(frame, knee, ankle, (255, 255, 255), 2)
        cv2.circle(frame, hip, 7, (0, 255, 0), -1)
        cv2.circle(frame, knee, 9, (255, 0, 0), -1)
        cv2.circle(frame, ankle, 7, (0, 0, 255), -1)

    # text box
    x0, y0 = 25, 35
    box_w, box_h = 560, 185
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"Knee angle: {knee_angle:6.1f} deg", (x0 + 15, y0 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    cv2.putText(frame, f"Ang vel:    {knee_vel:6.1f} deg/s", (x0 + 15, y0 + 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

    yy = y0 + 150
    for line in label_lines[:2]:
        cv2.putText(frame, line, (x0 + 15, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)
        yy += 35


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
    parser.add_argument("--metric_name", default="front_knee")

    # tuning
    parser.add_argument("--smooth_window", type=int, default=9)
    parser.add_argument("--smooth_poly", type=int, default=2)

    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    trial_dir = out_dir / args.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)

    metric = args.metric_name

    # expected cached inputs
    keypoints_csv = trial_dir / "keypoints.csv"
    step_summary = trial_dir / "step_duration_summary.json"
    stride_summary = trial_dir / "delivery_stride_summary.json"  # produced by delivery_stride.py

    if not keypoints_csv.exists():
        raise FileNotFoundError(
            f"Missing {keypoints_csv}. Run step_duration.py first to create cached keypoints."
        )

    if not step_summary.exists():
        raise FileNotFoundError(
            f"Missing {step_summary}. Run step_duration.py first (release frame needed)."
        )

    if not stride_summary.exists():
        raise FileNotFoundError(
            f"Missing {stride_summary}. Run delivery_stride.py first (FFC frame needed)."
        )

    # outputs (contract)
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

    # load keypoints
    df = pd.read_csv(keypoints_csv)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # load release + ffc
    with open(step_summary, "r", encoding="utf-8") as f:
        ss = json.load(f)
    release_frame = int(ss["release_frame"])

    with open(stride_summary, "r", encoding="utf-8") as f:
        ds = json.load(f)
    ffc_frame = int(ds["ffc_frame"])

    if ffc_frame >= release_frame:
        # fallback: if something weird, allow short segment
        print("⚠️ FFC >= Release. Adjusting segment to (release-10 frames → release).")
        ffc_frame = max(1, release_frame - 10)

    # decide front leg side from bowling arm
    # RIGHT-arm bowler => front leg LEFT, LEFT-arm bowler => front leg RIGHT
    front_side = "left" if args.bowling_arm.upper() == "RIGHT" else "right"

    hip_kp = f"{front_side}_hip"
    knee_kp = f"{front_side}_knee"
    ankle_kp = f"{front_side}_ankle"

    print(f"🦵 Front leg side: {front_side.upper()} -> using {hip_kp}, {knee_kp}, {ankle_kp}")
    print(f"✅ FFC frame: {ffc_frame}")
    print(f"✅ Release frame: {release_frame}")

    # indices
    ffc_idx = int(df.index[df["frame"] == ffc_frame][0]) if (df["frame"] == ffc_frame).any() else None
    rel_idx = int(df.index[df["frame"] == release_frame][0]) if (df["frame"] == release_frame).any() else None
    if ffc_idx is None or rel_idx is None:
        raise RuntimeError("FFC/Release frame not found in keypoints.csv (frame mismatch).")

    # knee angle over full clip (so plots later are easy)
    knee_angles = np.full(len(df), np.nan, dtype=float)
    for i in range(len(df)):
        hip = safe_xy(df, hip_kp, i)
        knee = safe_xy(df, knee_kp, i)
        ankle = safe_xy(df, ankle_kp, i)
        if hip is None or knee is None or ankle is None:
            continue
        knee_angles[i] = angle_deg(hip, knee, ankle)

    # smooth for velocity stability
    knee_sm = knee_angles.copy()
    finite = np.isfinite(knee_sm)
    if finite.sum() > 10:
        tmp = pd.Series(knee_sm).interpolate().bfill().ffill().to_numpy()
        tmp_sm = smooth(tmp, args.smooth_window, args.smooth_poly)
        knee_sm = tmp_sm
        knee_sm[~finite] = np.nan

    # angular velocity (deg/s)
    knee_vel = np.full(len(df), np.nan, dtype=float)
    for i in range(1, len(df)):
        if np.isfinite(knee_sm[i]) and np.isfinite(knee_sm[i - 1]):
            knee_vel[i] = (knee_sm[i] - knee_sm[i - 1]) * float(fps)

    # values at events
    knee_at_ffc = float(knee_sm[ffc_idx]) if np.isfinite(knee_sm[ffc_idx]) else float(knee_angles[ffc_idx])
    knee_at_rel = float(knee_sm[rel_idx]) if np.isfinite(knee_sm[rel_idx]) else float(knee_angles[rel_idx])

    delta_knee = None
    if np.isfinite(knee_at_ffc) and np.isfinite(knee_at_rel):
        # release - ffc : positive means angle increased (more extension), negative means more flexion
        delta_knee = float(knee_at_rel - knee_at_ffc)

    # peak velocities in the segment (FFC->Release)
    seg_vel = knee_vel[ffc_idx:rel_idx + 1]
    peak_ext_vel = float(np.nanmax(seg_vel)) if np.isfinite(seg_vel).any() else None
    peak_flex_vel = float(np.nanmin(seg_vel)) if np.isfinite(seg_vel).any() else None

    # -------------------------
    # FRAME-LEVEL OUTPUT
    # -------------------------
    frames_df = pd.DataFrame({
        "frame": df["frame"].astype(int),
        "time_s": df["frame"].astype(float) / float(fps),

        f"{hip_kp}_x": df[f"{hip_kp}_x"].astype(float),
        f"{hip_kp}_y": df[f"{hip_kp}_y"].astype(float),
        f"{knee_kp}_x": df[f"{knee_kp}_x"].astype(float),
        f"{knee_kp}_y": df[f"{knee_kp}_y"].astype(float),
        f"{ankle_kp}_x": df[f"{ankle_kp}_x"].astype(float),
        f"{ankle_kp}_y": df[f"{ankle_kp}_y"].astype(float),

        "front_knee_angle_deg": knee_angles.astype(float),
        "front_knee_angle_deg_sm": knee_sm.astype(float),
        "front_knee_ang_vel_deg_s": knee_vel.astype(float),

        "is_ffc": 0,
        "is_release": 0,
    })
    frames_df.loc[frames_df["frame"] == ffc_frame, "is_ffc"] = 1
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

        "front_leg": front_side.upper(),

        "ffc_frame": int(ffc_frame),
        "release_frame": int(release_frame),

        "knee_angle_ffc_deg": float(knee_at_ffc) if np.isfinite(knee_at_ffc) else None,
        "knee_angle_release_deg": float(knee_at_rel) if np.isfinite(knee_at_rel) else None,
        "delta_knee_angle_deg": float(delta_knee) if delta_knee is not None else None,

        "peak_knee_extension_velocity_deg_s": peak_ext_vel,
        "peak_knee_flexion_velocity_deg_s": peak_flex_vel,
    }

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"📌 Saved summary JSON: {summary_out}")

    # -------------------------
    # ANNOTATED VIDEO (FFC → Release)
    # -------------------------
    start_frame = int(ffc_frame)
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

    pause_ffc = int(1.0 * fps)
    pause_rel = int(1.2 * fps)
    slow_factor = 4

    frame_no = start_frame
    while frame_no <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        idx = int(df.index[df["frame"] == frame_no][0])

        hip = safe_xy(df, hip_kp, idx)
        knee = safe_xy(df, knee_kp, idx)
        ankle = safe_xy(df, ankle_kp, idx)

        ang_now = float(knee_sm[idx]) if np.isfinite(knee_sm[idx]) else (float(knee_angles[idx]) if np.isfinite(knee_angles[idx]) else np.nan)
        vel_now = float(knee_vel[idx]) if np.isfinite(knee_vel[idx]) else 0.0

        labels = []
        if frame_no == ffc_frame:
            labels.append("EVENT: FFC")
        if frame_no == release_frame:
            labels.append("EVENT: RELEASE")

        pts = (hip, knee, ankle) if (hip is not None and knee is not None and ankle is not None) else None
        draw_overlay(frame, ang_now, vel_now, labels, pts=pts)

        repeats = slow_factor
        if frame_no == ffc_frame:
            repeats += pause_ffc
        if frame_no == release_frame:
            repeats += pause_rel

        for _ in range(repeats):
            out.write(frame)

        frame_no += 1

    cap.release()
    out.release()
    print(f"🎥 Annotated video saved: {annotated_out}")
    print("✅ Done.")


if __name__ == "__main__":
    main()