"""
head_com.py  (converted from your head position script)

✅ Requirement covered:
- Center of Mass (COM) of the Head Relative to the Front Foot

✅ Outputs (contract with run_all.py):
  1) <out_dir>/<trial_id>/<metric_name>_frames.csv
  2) <out_dir>/<trial_id>/<metric_name>_summary.json
  3) <out_dir>/<trial_id>/<metric_name>_annotated.mp4

✅ Major changes vs your original:
- NO YOLO in this script. It reuses cached: <out_dir>/<trial_id>/keypoints.csv
- NO calibration clicks. It reuses cached calibration from delivery_stride_summary.json
  (so you calibrate only once in the pipeline).
- NO separate BFC/FFC detection logic here. It reuses cached:
    - ffc_frame and bfc_frame from delivery_stride_summary.json
    - release_frame from step_duration_summary.json
- Head proxy = mean(nose, eyes, ears) each frame
- Computes head offset relative to front foot:
    Dx, Dy, D in pixels + meters
- Also computes “midline” head offset relative to pelvis midpoint (optional stability proxy)

✅ Annotated video:
- Trims BFC → Release (or FFC → Release if BFC missing)
- Draws head point, front foot point, line, and Dx in meters
- Pauses at BFC / FFC / Release

"""

import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


FACE_POINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]


def safe_mean_xy(df: pd.DataFrame, points, idx: int):
    xs, ys = [], []
    for p in points:
        x = df.loc[idx, f"{p}_x"] if f"{p}_x" in df.columns else np.nan
        y = df.loc[idx, f"{p}_y"] if f"{p}_y" in df.columns else np.nan
        if np.isfinite(x) and np.isfinite(y):
            xs.append(float(x))
            ys.append(float(y))
    if not xs:
        return np.nan, np.nan
    return float(np.mean(xs)), float(np.mean(ys))


def draw_overlay(frame, frame_no, label_lines, head_xy, foot_xy, dx_m, dy_m, d_m):
    h, w = frame.shape[:2]

    if head_xy is not None:
        cv2.circle(frame, tuple(head_xy.astype(int)), 6, (0, 255, 0), -1)  # head
    if foot_xy is not None:
        cv2.circle(frame, tuple(foot_xy.astype(int)), 7, (0, 0, 255), -1)  # front foot
    if head_xy is not None and foot_xy is not None:
        cv2.line(frame, tuple(head_xy.astype(int)), tuple(foot_xy.astype(int)), (255, 255, 255), 2)

    # HUD box
    x0, y0 = 25, 30
    box_w, box_h = 640, 200
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

    cv2.putText(frame, f"Dx: {dx_m:+.3f} m   Dy: {dy_m:+.3f} m   D: {d_m:.3f} m",
                (x0 + 15, y0 + 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(frame, "Green=head (proxy)  |  Red=front foot", (25, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--trial_id", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--view_mode", default="SIDE", choices=["SIDE", "FRONT"])
    parser.add_argument("--bowling_arm", default="RIGHT", choices=["LEFT", "RIGHT"])
    parser.add_argument("--metric_name", default="head_com")

    args = parser.parse_args()

    video_path = Path(args.video)
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

    # outputs (contract)
    frames_out = trial_dir / f"{metric}_frames.csv"
    summary_out = trial_dir / f"{metric}_summary.json"
    annotated_out = trial_dir / f"{metric}_annotated.mp4"

    # open video for meta
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"🎥 Frames={total_frames}, FPS={fps:.2f}")

    # load cached frames
    df = pd.read_csv(keypoints_csv)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # load events + calibration
    with open(step_summary, "r", encoding="utf-8") as f:
        ss = json.load(f)
    release_frame = int(ss["release_frame"])

    with open(stride_summary, "r", encoding="utf-8") as f:
        ds = json.load(f)

    ffc_frame = int(ds["ffc_frame"])
    bfc_frame = ds.get("bfc_frame", None)
    bfc_frame = int(bfc_frame) if bfc_frame is not None else None

    meters_per_pixel = ds.get("meters_per_pixel", None)
    if meters_per_pixel is None:
        # fallback (very unlikely if delivery_stride.py was used)
        cm_per_pixel = ds.get("cm_per_pixel", None)
        if cm_per_pixel is None:
            raise RuntimeError("No scale found in delivery_stride_summary.json (need meters_per_pixel or cm_per_pixel).")
        meters_per_pixel = float(cm_per_pixel) / 100.0
    meters_per_pixel = float(meters_per_pixel)

    # choose front ankle from delivery stride script (best)
    front_ankle = ds.get("front_ankle", None)
    if front_ankle not in ["left_ankle", "right_ankle"]:
        # deterministic fallback:
        # RIGHT-arm bowler => front leg LEFT, LEFT-arm bowler => front leg RIGHT
        front_ankle = "left_ankle" if args.bowling_arm.upper() == "RIGHT" else "right_ankle"

    # resolve indices
    def frame_to_idx(frame_no: int):
        hits = df.index[df["frame"] == int(frame_no)].tolist()
        return hits[0] if hits else None

    ffc_idx = frame_to_idx(ffc_frame)
    rel_idx = frame_to_idx(release_frame)
    bfc_idx = frame_to_idx(bfc_frame) if bfc_frame is not None else None

    if ffc_idx is None or rel_idx is None:
        raise RuntimeError("FFC/Release not found in keypoints.csv (frame mismatch).")
    if bfc_frame is not None and bfc_idx is None:
        print("⚠️ BFC frame not found in keypoints.csv; will annotate from FFC instead.")
        bfc_frame = None
        bfc_idx = None

    start_frame = int(bfc_frame) if bfc_frame is not None else int(ffc_frame)
    end_frame = int(release_frame)

    if start_frame >= end_frame:
        start_frame = max(1, end_frame - 10)

    print(f"🦶 Front foot: {front_ankle.upper()}")
    print(f"✅ Start frame: {start_frame} ({'BFC' if bfc_frame is not None else 'FFC'})")
    print(f"✅ FFC frame:   {ffc_frame}")
    print(f"✅ Release:     {release_frame}")
    print(f"📏 meters_per_pixel: {meters_per_pixel:.6f}")

    # compute head proxy per frame
    head_x = np.full(len(df), np.nan, dtype=float)
    head_y = np.full(len(df), np.nan, dtype=float)
    for i in range(len(df)):
        hx, hy = safe_mean_xy(df, FACE_POINTS, i)
        head_x[i] = hx
        head_y[i] = hy

    # pelvis midpoint (optional stability proxy)
    hip_mid_x = (df["left_hip_x"].astype(float).to_numpy() + df["right_hip_x"].astype(float).to_numpy()) / 2.0
    hip_mid_y = (df["left_hip_y"].astype(float).to_numpy() + df["right_hip_y"].astype(float).to_numpy()) / 2.0

    # front foot coords
    foot_x = df[f"{front_ankle}_x"].astype(float).to_numpy()
    foot_y = df[f"{front_ankle}_y"].astype(float).to_numpy()

    # offsets head -> foot
    dx_px = head_x - foot_x
    dy_px = head_y - foot_y
    d_px = np.sqrt(dx_px**2 + dy_px**2)

    dx_m = dx_px * meters_per_pixel
    dy_m = dy_px * meters_per_pixel
    d_m = d_px * meters_per_pixel

    # midline offsets head -> pelvis midpoint
    mdx_px = head_x - hip_mid_x
    mdy_px = head_y - hip_mid_y
    md_px = np.sqrt(mdx_px**2 + mdy_px**2)

    mdx_m = mdx_px * meters_per_pixel
    mdy_m = mdy_px * meters_per_pixel
    md_m = md_px * meters_per_pixel

    # event values
    def at(idx):
        if idx is None:
            return None
        return {
            "dx_m": float(dx_m[idx]) if np.isfinite(dx_m[idx]) else None,
            "dy_m": float(dy_m[idx]) if np.isfinite(dy_m[idx]) else None,
            "d_m":  float(d_m[idx]) if np.isfinite(d_m[idx]) else None,
            "dx_px": float(dx_px[idx]) if np.isfinite(dx_px[idx]) else None,
            "dy_px": float(dy_px[idx]) if np.isfinite(dy_px[idx]) else None,
            "d_px":  float(d_px[idx]) if np.isfinite(d_px[idx]) else None,
        }

    def mid_at(idx):
        if idx is None:
            return None
        return {
            "dx_m": float(mdx_m[idx]) if np.isfinite(mdx_m[idx]) else None,
            "dy_m": float(mdy_m[idx]) if np.isfinite(mdy_m[idx]) else None,
            "d_m":  float(md_m[idx]) if np.isfinite(md_m[idx]) else None,
        }

    ffc_vals = at(ffc_idx)
    rel_vals = at(rel_idx)
    bfc_vals = at(bfc_idx) if bfc_idx is not None else None

    # -------------------------
    # FRAME-LEVEL OUTPUT
    # -------------------------
    frames_df = pd.DataFrame({
        "frame": df["frame"].astype(int),
        "time_s": df["frame"].astype(float) / float(fps),

        "head_x": head_x,
        "head_y": head_y,

        f"{front_ankle}_x": foot_x,
        f"{front_ankle}_y": foot_y,

        "head_to_front_dx_px": dx_px,
        "head_to_front_dy_px": dy_px,
        "head_to_front_d_px": d_px,
        "head_to_front_dx_m": dx_m,
        "head_to_front_dy_m": dy_m,
        "head_to_front_d_m": d_m,

        "head_to_mid_dx_m": mdx_m,
        "head_to_mid_dy_m": mdy_m,
        "head_to_mid_d_m": md_m,

        "is_bfc": 0,
        "is_ffc": 0,
        "is_release": 0,
    })
    if bfc_frame is not None:
        frames_df.loc[frames_df["frame"] == int(bfc_frame), "is_bfc"] = 1
    frames_df.loc[frames_df["frame"] == int(ffc_frame), "is_ffc"] = 1
    frames_df.loc[frames_df["frame"] == int(release_frame), "is_release"] = 1

    frames_df.to_csv(frames_out, index=False)
    print(f"📌 Saved frames CSV: {frames_out}")

    # -------------------------
    # SUMMARY JSON
    # -------------------------
    summary = {
        "fps": float(fps),
        "view_mode": args.view_mode.upper(),
        "bowling_arm": args.bowling_arm.upper(),
        "front_ankle": front_ankle,

        "meters_per_pixel": float(meters_per_pixel),

        "bfc_frame": int(bfc_frame) if bfc_frame is not None else None,
        "ffc_frame": int(ffc_frame),
        "release_frame": int(release_frame),

        "head_to_front": {
            "at_bfc": bfc_vals,
            "at_ffc": ffc_vals,
            "at_release": rel_vals,
            "note": "SIDE view: Dx ~ forward/back (lean). FRONT view: Dx ~ lateral deviation."
        },
        "head_to_midline": {
            "at_bfc": mid_at(bfc_idx) if bfc_idx is not None else None,
            "at_ffc": mid_at(ffc_idx),
            "at_release": mid_at(rel_idx),
            "note": "Useful stability proxy if front foot detection is noisy."
        }
    }

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"📌 Saved summary JSON: {summary_out}")

    # -------------------------
    # ANNOTATED VIDEO (BFC/FFC → Release)
    # -------------------------
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - 1))
    out = cv2.VideoWriter(
        str(annotated_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height)
    )

    slow_factor = 4
    pause_bfc = int(1.0 * fps)
    pause_ffc = int(1.2 * fps)
    pause_rel = int(1.4 * fps)

    frame_no = start_frame
    while frame_no <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        idx = frame_to_idx(frame_no)
        if idx is None:
            break

        label_lines = []
        if bfc_frame is not None and frame_no == bfc_frame:
            label_lines.append("EVENT: BFC")
        if frame_no == ffc_frame:
            label_lines.append("EVENT: FFC")
        if frame_no == release_frame:
            label_lines.append("EVENT: RELEASE")

        hx = head_x[idx]
        hy = head_y[idx]
        fx = foot_x[idx]
        fy = foot_y[idx]

        head_xy = np.array([hx, hy], dtype=float) if (np.isfinite(hx) and np.isfinite(hy)) else None
        foot_xy = np.array([fx, fy], dtype=float) if (np.isfinite(fx) and np.isfinite(fy)) else None

        dxm = float(dx_m[idx]) if np.isfinite(dx_m[idx]) else 0.0
        dym = float(dy_m[idx]) if np.isfinite(dy_m[idx]) else 0.0
        dm  = float(d_m[idx])  if np.isfinite(d_m[idx])  else 0.0

        draw_overlay(frame, frame_no, label_lines, head_xy, foot_xy, dxm, dym, dm)

        repeats = slow_factor
        if bfc_frame is not None and frame_no == bfc_frame:
            repeats += pause_bfc
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