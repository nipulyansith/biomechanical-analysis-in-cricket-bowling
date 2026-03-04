"""
wrist_velocity.py  (converted to match the SAME pipeline contract)

✅ Requirement covered:
- Wrist Joint Velocity and Ball Release Speed (Mode A proxy)

✅ Contract with run_all.py:
Inputs (cached):
  - <out_dir>/<trial_id>/keypoints.csv                (from step_duration.py)
  - <out_dir>/<trial_id>/step_duration_summary.json   (release_frame)
  - <out_dir>/<trial_id>/delivery_stride_summary.json (meters_per_pixel + front_ankle, events)

Outputs:
  1) <out_dir>/<trial_id>/wrist_velocity_frames.csv
  2) <out_dir>/<trial_id>/wrist_velocity_summary.json
  3) <out_dir>/<trial_id>/wrist_velocity_annotated.mp4

✅ Major changes vs your original:
- NO YOLO in this script (uses cached keypoints.csv)
- NO calibration clicks (reuses meters_per_pixel from delivery_stride_summary.json)
- Release frame is NOT re-detected here:
    it reuses release_frame from step_duration_summary.json
  (keeps ALL scripts aligned to the same events)
- Robust derivative: central difference
- Uses smoothing before derivative + also smooths speed & omega
- Writes per-frame series + summary near release window

Notes:
- Mode B (ball detector) intentionally removed here for dataset consistency.
  If you later add a ball model, make it a separate optional module.

"""

import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ---------------- Helpers ----------------
def make_window(n: int, length: int) -> int:
    w = n if n % 2 == 1 else n + 1
    if length <= 3:
        return 3
    if w >= length:
        w = length - 1 if (length - 1) % 2 == 1 else length - 2
    return max(3, w)


def smooth_signal(sig, win=9, poly=2):
    sig = np.asarray(sig, dtype=float)
    if len(sig) < 3:
        return sig.copy()
    w = make_window(win, len(sig))
    return savgol_filter(sig, window_length=w, polyorder=min(poly, w - 1))


def central_diff(sig, dt: float):
    sig = np.asarray(sig, dtype=float)
    out = np.zeros_like(sig, dtype=float)
    if len(sig) < 2:
        return out
    out[0] = (sig[1] - sig[0]) / dt
    out[-1] = (sig[-1] - sig[-2]) / dt
    if len(sig) > 2:
        out[1:-1] = (sig[2:] - sig[:-2]) / (2.0 * dt)
    return out


def unwrap_angle(theta):
    return np.unwrap(np.asarray(theta, dtype=float))


def pick_arm_points(bowling_arm: str):
    bowling_arm = bowling_arm.upper().strip()
    if bowling_arm not in ["RIGHT", "LEFT"]:
        raise ValueError("bowling_arm must be RIGHT or LEFT")
    wrist = "right_wrist" if bowling_arm == "RIGHT" else "left_wrist"
    elbow = "right_elbow" if bowling_arm == "RIGHT" else "left_elbow"
    return elbow, wrist


def frame_to_idx(df, frame_no: int):
    hits = df.index[df["frame"] == int(frame_no)].tolist()
    return hits[0] if hits else None


def draw_overlay(frame, frame_no, label_lines, wrist_xy, elbow_xy, speed_mps, omega_rads):
    h, w = frame.shape[:2]

    if elbow_xy is not None:
        cv2.circle(frame, tuple(elbow_xy.astype(int)), 6, (255, 0, 0), -1)
    if wrist_xy is not None:
        cv2.circle(frame, tuple(wrist_xy.astype(int)), 6, (0, 255, 0), -1)
    if elbow_xy is not None and wrist_xy is not None:
        cv2.line(frame, tuple(elbow_xy.astype(int)), tuple(wrist_xy.astype(int)), (255, 255, 255), 2)

    # HUD
    x0, y0 = 25, 30
    box_w, box_h = 720, 205
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"Frame: {frame_no}", (x0 + 15, y0 + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

    yy = y0 + 85
    for s in label_lines[:2]:
        cv2.putText(frame, s, (x0 + 15, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)
        yy += 35

    cv2.putText(frame, f"Wrist speed: {speed_mps:.2f} m/s  ({speed_mps*3.6:.1f} km/h)",
                (x0 + 15, y0 + 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 255, 255), 2)

    cv2.putText(frame, f"Forearm omega: {omega_rads:.2f} rad/s",
                (x0 + 15, y0 + 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 255, 255), 2)


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--trial_id", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--bowling_arm", default="RIGHT", choices=["LEFT", "RIGHT"])
    parser.add_argument("--metric_name", default="wrist_velocity")
    parser.add_argument("--smooth_window", type=int, default=9)
    parser.add_argument("--smooth_poly", type=int, default=2)
    parser.add_argument("--near_ms", type=int, default=120)  # ± window around release for "near release" peaks
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    trial_dir = out_dir / args.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)

    metric = args.metric_name

    # cached inputs
    keypoints_csv = trial_dir / "keypoints.csv"
    step_summary = trial_dir / "step_duration_summary.json"
    stride_summary = trial_dir / "delivery_stride_summary.json"

    if not keypoints_csv.exists():
        raise FileNotFoundError(f"Missing {keypoints_csv}. Run step_duration.py first.")
    if not step_summary.exists():
        raise FileNotFoundError(f"Missing {step_summary}. Run step_duration.py first.")
    if not stride_summary.exists():
        raise FileNotFoundError(f"Missing {stride_summary}. Run delivery_stride.py first.")

    # outputs
    frames_out = trial_dir / f"{metric}_frames.csv"
    summary_out = trial_dir / f"{metric}_summary.json"
    annotated_out = trial_dir / f"{metric}_annotated.mp4"

    # open video for fps + size
    cap0 = cv2.VideoCapture(str(args.video))
    if not cap0.isOpened():
        raise RuntimeError("Cannot open video")
    fps = float(cap0.get(cv2.CAP_PROP_FPS) or 60.0)
    width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
    cap0.release()

    dt = 1.0 / fps
    print(f"🎥 Frames={total_frames}, FPS={fps:.2f}")

    # load cached keypoints
    df = pd.read_csv(keypoints_csv)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    elbow_name, wrist_name = pick_arm_points(args.bowling_arm)

    # load events + scale
    with open(step_summary, "r", encoding="utf-8") as f:
        ss = json.load(f)
    release_frame = int(ss["release_frame"])

    with open(stride_summary, "r", encoding="utf-8") as f:
        ds = json.load(f)
    meters_per_pixel = ds.get("meters_per_pixel", None)
    if meters_per_pixel is None:
        cm_per_pixel = ds.get("cm_per_pixel", None)
        if cm_per_pixel is None:
            raise RuntimeError("No scale found in delivery_stride_summary.json")
        meters_per_pixel = float(cm_per_pixel) / 100.0
    meters_per_pixel = float(meters_per_pixel)

    # optional: segment for annotation (BFC/FFC → Release)
    ffc_frame = int(ds["ffc_frame"])
    bfc_frame = ds.get("bfc_frame", None)
    bfc_frame = int(bfc_frame) if bfc_frame is not None else None
    start_frame = bfc_frame if bfc_frame is not None else ffc_frame
    end_frame = release_frame

    rel_idx = frame_to_idx(df, release_frame)
    if rel_idx is None:
        raise RuntimeError("release_frame not found in keypoints.csv (frame mismatch).")

    # smooth elbow + wrist
    wx = smooth_signal(df[f"{wrist_name}_x"].astype(float).to_numpy(), args.smooth_window, args.smooth_poly)
    wy = smooth_signal(df[f"{wrist_name}_y"].astype(float).to_numpy(), args.smooth_window, args.smooth_poly)
    ex = smooth_signal(df[f"{elbow_name}_x"].astype(float).to_numpy(), args.smooth_window, args.smooth_poly)
    ey = smooth_signal(df[f"{elbow_name}_y"].astype(float).to_numpy(), args.smooth_window, args.smooth_poly)

    # linear velocity (m/s)
    vx_px = central_diff(wx, dt)
    vy_px = central_diff(wy, dt)
    vx_m = vx_px * meters_per_pixel
    vy_m = vy_px * meters_per_pixel
    speed_mps = np.sqrt(vx_m * vx_m + vy_m * vy_m)
    speed_mps_sm = smooth_signal(speed_mps, args.smooth_window, args.smooth_poly)

    # angular velocity of forearm segment (rad/s)
    theta = np.arctan2((wy - ey), (wx - ex))
    theta_u = unwrap_angle(theta)
    omega = central_diff(theta_u, dt)
    omega_sm = smooth_signal(omega, args.smooth_window, args.smooth_poly)

    # time axis
    time_s = (df["frame"].astype(float).to_numpy() - 1.0) / fps

    # peak overall + near release window
    peak_idx = int(np.nanargmax(speed_mps_sm))
    peak_frame = int(df.loc[peak_idx, "frame"])
    peak_speed = float(speed_mps_sm[peak_idx])

    win = int(round((args.near_ms / 1000.0) * fps))
    a = max(0, rel_idx - win)
    b = min(len(df) - 1, rel_idx + win)

    near_idx = a + int(np.nanargmax(speed_mps_sm[a:b+1]))
    near_frame = int(df.loc[near_idx, "frame"])
    near_peak = float(speed_mps_sm[near_idx])

    # peak abs omega near release
    omega_slice = omega_sm[a:b+1]
    near_omega_idx = a + int(np.nanargmax(np.abs(omega_slice)))
    near_omega_frame = int(df.loc[near_omega_idx, "frame"])
    near_omega_peak = float(omega_sm[near_omega_idx])

    # at release
    speed_at_release = float(speed_mps_sm[rel_idx])
    omega_at_release = float(omega_sm[rel_idx])

    # ---------------- Frame-level CSV ----------------
    frames_df = pd.DataFrame({
        "frame": df["frame"].astype(int),
        "time_s": time_s,

        "wrist_x_sm": wx,
        "wrist_y_sm": wy,
        "elbow_x_sm": ex,
        "elbow_y_sm": ey,

        "wrist_vx_mps": vx_m,
        "wrist_vy_mps": vy_m,
        "wrist_speed_mps_sm": speed_mps_sm,
        "wrist_speed_kmh_sm": speed_mps_sm * 3.6,

        "forearm_angle_rad": theta_u,
        "forearm_omega_rads_sm": omega_sm,

        "is_release": 0
    })
    frames_df.loc[frames_df["frame"] == int(release_frame), "is_release"] = 1
    frames_df.to_csv(frames_out, index=False)
    print(f"📌 Saved frames CSV: {frames_out}")

    # ---------------- Summary JSON ----------------
    summary = {
        "video": os.path.basename(args.video),
        "fps": float(fps),
        "bowling_arm": args.bowling_arm.upper(),
        "meters_per_pixel": float(meters_per_pixel),

        "release_frame": int(release_frame),

        "wrist_speed": {
            "at_release_mps": float(speed_at_release),
            "at_release_kmh": float(speed_at_release * 3.6),

            "peak_overall_mps": float(peak_speed),
            "peak_overall_frame": int(peak_frame),

            "peak_near_release_mps": float(near_peak),
            "peak_near_release_frame": int(near_frame),

            "near_window_ms": int(args.near_ms)
        },
        "forearm_angular_velocity": {
            "omega_at_release_rads": float(omega_at_release),
            "peak_abs_near_release_rads": float(near_omega_peak),
            "peak_abs_near_release_frame": int(near_omega_frame),
            "near_window_ms": int(args.near_ms),
            "note": "Omega from 2D forearm segment angle (elbow->wrist)."
        },
        "ball_release_speed_proxy": {
            "method": "Mode A proxy (wrist speed at release)",
            "proxy_mps": float(speed_at_release),
            "proxy_kmh": float(speed_at_release * 3.6),
            "note": "Proxy only. Wrist speed correlates with ball speed but is not equal."
        }
    }

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"📌 Saved summary JSON: {summary_out}")

    # ---------------- Annotated video ----------------
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video for annotation")

    # clamp range
    start_frame = max(1, int(start_frame))
    end_frame = min(int(end_frame), total_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

    out = cv2.VideoWriter(
        str(annotated_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height)
    )

    slow_factor = 4
    pause_rel = int(1.2 * fps)
    pause_peak = int(0.9 * fps)

    frame_no = start_frame
    while frame_no <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        idx = frame_to_idx(df, frame_no)
        if idx is None:
            break

        label_lines = []
        if frame_no == release_frame:
            label_lines.append("EVENT: RELEASE")
        if frame_no == near_frame:
            label_lines.append("EVENT: PEAK SPEED (near release)")

        wrist_xy = np.array([wx[idx], wy[idx]], dtype=float) if np.isfinite([wx[idx], wy[idx]]).all() else None
        elbow_xy = np.array([ex[idx], ey[idx]], dtype=float) if np.isfinite([ex[idx], ey[idx]]).all() else None

        draw_overlay(
            frame=frame,
            frame_no=frame_no,
            label_lines=label_lines,
            wrist_xy=wrist_xy,
            elbow_xy=elbow_xy,
            speed_mps=float(speed_mps_sm[idx]) if np.isfinite(speed_mps_sm[idx]) else 0.0,
            omega_rads=float(omega_sm[idx]) if np.isfinite(omega_sm[idx]) else 0.0
        )

        repeats = slow_factor
        if frame_no == release_frame:
            repeats += pause_rel
        if frame_no == near_frame:
            repeats += pause_peak

        for _ in range(repeats):
            out.write(frame)

        frame_no += 1

    cap.release()
    out.release()
    print(f"🎥 Saved annotated video: {annotated_out}")
    print("✅ Done.")


if __name__ == "__main__":
    main()