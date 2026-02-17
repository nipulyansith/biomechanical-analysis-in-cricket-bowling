"""
FAST BOWLING – Requirement: Wrist Joint Velocity + Ball Release Speed (2D)

What this script does (single video):
✅ YOLOv8 Pose extraction (Ultralytics)
✅ Pixel → meters calibration (stump height: click top & bottom)
✅ Supports RIGHT-arm and LEFT-arm bowlers (BOWLING_ARM variable)
✅ Detects release frame (bowling wrist highest OR peak wrist speed)
✅ Computes:
   1) Wrist linear velocity (m/s) + peak wrist speed near release
   2) Wrist angular velocity (rad/s) of forearm (elbow→wrist angle) + peak
✅ Optional ball release speed estimation:
   - Mode A (default, robust): Ball speed ≈ wrist speed at release (proxy)
   - Mode B (optional if you have a ball detector model): Track ball for N frames after release and compute speed

Important notes:
- This is 2D projected motion; results depend on camera angle and fps.
- Wrist “angular velocity” is computed from the forearm segment angle (elbow->wrist).
- Ball speed from Mode A is a proxy (literature says correlated, but not equal).
- For real ball speed, use Mode B with a cricket-ball detection model.

Dependencies:
pip install ultralytics opencv-python numpy pandas scipy

"""

import os
import cv2
import json
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.signal import savgol_filter

# =========================
# USER SETTINGS
# =========================
VIDEO_PATH = "side 1.MOV"

# RIGHT or LEFT arm bowler (IMPORTANT)
BOWLING_ARM = "LEFT"  # "RIGHT" or "LEFT"

# Release detection mode:
# - "WRIST_HIGHEST": release frame = bowling wrist highest (min y)
# - "PEAK_WRIST_SPEED": release frame = peak wrist speed (often near release)
RELEASE_DETECT_MODE = "WRIST_HIGHEST"  # "WRIST_HIGHEST" or "PEAK_WRIST_SPEED"

MODEL_PATH = "yolov8n-pose.pt"

OUT_DIR = "output_wrist_speed"
KEYPOINT_CSV = os.path.join(OUT_DIR, "yolo_keypoints.csv")
METRICS_JSON = os.path.join(OUT_DIR, "wrist_ball_metrics.json")
TIMESERIES_CSV = os.path.join(OUT_DIR, "wrist_timeseries.csv")

# Smoothing (for derivatives)
SMOOTH_WINDOW = 9   # odd preferred
SMOOTH_POLY = 2

# Calibration (ICC stumps ~ 71.1 cm)
STUMP_HEIGHT_M = 0.711

# ============= OPTIONAL BALL SPEED MODE B (requires a ball detector) =============
# If you have a YOLO model that detects the cricket ball, put its path here.
# Set to None to disable Mode B and use proxy speed only.
BALL_MODEL_PATH = None  # e.g. "cricket_ball_yolo.pt"

# How many frames after release to track ball for speed (Mode B)
BALL_TRACK_FRAMES = 12

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
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist"
]

# =========================
# HELPERS
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def make_window(n, length):
    w = n if n % 2 == 1 else n + 1
    if length <= 3:
        return 3
    if w >= length:
        w = length - 1 if (length - 1) % 2 == 1 else length - 2
    return max(3, w)

def smooth_signal(sig, win, poly=2):
    if len(sig) < 3:
        return sig.copy()
    w = make_window(win, len(sig))
    return savgol_filter(sig, window_length=w, polyorder=min(poly, w - 1))

def calibrate_pixels_to_m(video_path, stump_height_m=0.711):
    """
    Click TOP and BOTTOM of stump on first frame. Returns meters_per_pixel.
    """
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read video for calibration.")

    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration", on_mouse)

    print("\nCALIBRATION:")
    print("Click 1) TOP of stump, 2) BOTTOM of stump. Press ESC to cancel.")

    while True:
        show = frame.copy()
        for i, (cx, cy) in enumerate(clicks):
            cv2.circle(show, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(show, str(i + 1), (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.putText(show, "Click TOP then BOTTOM of stump (2 clicks). ESC to cancel.",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Calibration", show)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyWindow("Calibration")
            raise RuntimeError("Calibration cancelled.")
        if len(clicks) >= 2:
            break

    cv2.destroyWindow("Calibration")

    (x1, y1), (x2, y2) = clicks[0], clicks[1]
    px_dist = float(np.hypot(x2 - x1, y2 - y1))
    if px_dist < 2:
        raise RuntimeError("Calibration failed: clicked points too close.")

    meters_per_pixel = float(stump_height_m / px_dist)
    print(f"✅ Calibration: stump_px={px_dist:.2f}px  -> meters_per_pixel={meters_per_pixel:.6f}")
    return meters_per_pixel

def central_diff(sig, dt):
    """
    Central difference derivative (more stable than forward difference).
    For ends, uses forward/backward.
    """
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
        raise ValueError("BOWLING_ARM must be 'RIGHT' or 'LEFT'")
    wrist = "right_wrist" if bowling_arm == "RIGHT" else "left_wrist"
    elbow = "right_elbow" if bowling_arm == "RIGHT" else "left_elbow"
    return elbow, wrist

def find_release_frame(df, fps, wrist_col_y_sm, wrist_speed_sm):
    """
    Returns release_idx (0-based) and release_frame (1-based frame number).
    """
    if RELEASE_DETECT_MODE.upper() == "PEAK_WRIST_SPEED":
        idx = int(np.nanargmax(wrist_speed_sm))
    else:
        idx = int(np.nanargmin(wrist_col_y_sm))
    frame_num = int(df.loc[idx, "frame"])
    return idx, frame_num

# ----------------- Optional Ball Mode B helpers -----------------
def detect_ball_center_yolo(frame, ball_model):
    """
    Return (x,y,conf) for the highest-confidence ball detection in this frame.
    If no detection -> (None,None,None)
    """
    res = ball_model.predict(frame, verbose=False)
    if len(res) == 0 or res[0].boxes is None or len(res[0].boxes) == 0:
        return None, None, None
    boxes = res[0].boxes
    confs = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy()
    j = int(np.argmax(confs))
    x1, y1, x2, y2 = xyxy[j]
    cx = float((x1 + x2) / 2.0)
    cy = float((y1 + y2) / 2.0)
    return cx, cy, float(confs[j])

def compute_ball_speed_mode_b(video_path, release_frame, meters_per_pixel, fps, ball_model, n_frames=12):
    """
    Track ball for n_frames after release. Compute average speed (m/s) and peak speed (m/s).
    Requires BALL_MODEL_PATH (ball detector).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, release_frame - 1))
    pts = []
    frames_used = 0

    while frames_used < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        cx, cy, conf = detect_ball_center_yolo(frame, ball_model)
        if cx is not None and cy is not None:
            pts.append((cx, cy))
        else:
            pts.append((np.nan, np.nan))
        frames_used += 1

    cap.release()

    pts = np.asarray(pts, dtype=float)
    if np.all(~np.isfinite(pts)):
        return None

    # smooth and differentiate
    x = smooth_signal(pts[:, 0], SMOOTH_WINDOW, SMOOTH_POLY)
    y = smooth_signal(pts[:, 1], SMOOTH_WINDOW, SMOOTH_POLY)

    dt = 1.0 / float(fps)
    vx = central_diff(x, dt) * meters_per_pixel
    vy = central_diff(y, dt) * meters_per_pixel
    speed = np.sqrt(vx * vx + vy * vy)

    # ignore NaNs
    speed_valid = speed[np.isfinite(speed)]
    if len(speed_valid) == 0:
        return None

    return {
        "ball_speed_avg_mps": float(np.nanmean(speed_valid)),
        "ball_speed_peak_mps": float(np.nanmax(speed_valid)),
        "ball_speed_avg_kmh": float(np.nanmean(speed_valid) * 3.6),
        "ball_speed_peak_kmh": float(np.nanmax(speed_valid) * 3.6),
        "tracked_frames": int(frames_used),
        "note": "Mode B: ball speed computed from detected ball centers after release."
    }

def stabilize_left_right_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix left/right flipping using ONLY columns that exist.
    For this wrist_velocity script, we stabilize using:
      left_elbow, right_elbow, left_wrist, right_wrist
    If you later include more joints, this will automatically use them too.
    """

    # Candidate paired joints in priority order
    candidate_pairs = [
        ("left_wrist", "right_wrist"),
        ("left_elbow", "right_elbow"),
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
        ("left_eye", "right_eye"),
        ("left_ear", "right_ear"),
    ]

    # Keep only pairs that actually exist in df
    pairs = []
    for L, R in candidate_pairs:
        if f"{L}_x" in df.columns and f"{L}_y" in df.columns and f"{R}_x" in df.columns and f"{R}_y" in df.columns:
            pairs.append((L, R))

    # If nothing to stabilize, return as-is
    if len(pairs) == 0 or len(df) < 2:
        return df

    def get_xy(row, name):
        return np.array([row[f"{name}_x"], row[f"{name}_y"]], dtype=float)

    def frame_cost(prev_row, cur_row, swapped: bool) -> float:
        total = 0.0
        for L, R in pairs:
            # choose current L/R depending on swap
            if not swapped:
                Lc = get_xy(cur_row, L)
                Rc = get_xy(cur_row, R)
            else:
                Lc = get_xy(cur_row, R)
                Rc = get_xy(cur_row, L)

            Lp = get_xy(prev_row, L)
            Rp = get_xy(prev_row, R)

            if np.all(np.isfinite(Lc)) and np.all(np.isfinite(Lp)):
                total += float(np.linalg.norm(Lc - Lp))
            if np.all(np.isfinite(Rc)) and np.all(np.isfinite(Rp)):
                total += float(np.linalg.norm(Rc - Rp))
        return total

    def apply_swap(row):
        row = row.copy()
        for L, R in pairs:
            row[f"{L}_x"], row[f"{R}_x"] = row[f"{R}_x"], row[f"{L}_x"]
            row[f"{L}_y"], row[f"{R}_y"] = row[f"{R}_y"], row[f"{L}_y"]
        return row

    rows = df.to_dict(orient="records")
    stabilized = [rows[0]]

    for i in range(1, len(rows)):
        prev = stabilized[-1]
        cur = rows[i]

        cost_keep = frame_cost(prev, cur, swapped=False)
        cost_swap = frame_cost(prev, cur, swapped=True)

        if cost_swap < cost_keep:
            stabilized.append(apply_swap(cur))
        else:
            stabilized.append(cur)

    return pd.DataFrame(stabilized)

def write_wrist_velocity_annotated_video(
    video_path,
    df,
    fps,
    out_path,
    release_frame,
    peak_frame,
    meters_per_pixel,
    slow_mo_factor=4
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video for annotation.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, float(fps), (width, height))

    print("\n🎬 Writing annotated wrist velocity video...")

    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        idx = frame_no - 1
        if idx >= len(df):
            break

        # Get smoothed values
        wx = int(df.loc[idx, "wrist_x_sm"])
        wy = int(df.loc[idx, "wrist_y_sm"])
        ex = int(df.loc[idx, "elbow_x_sm"])
        ey = int(df.loc[idx, "elbow_y_sm"])

        speed_mps = df.loc[idx, "wrist_speed_mps_sm"]
        speed_kmh = speed_mps * 3.6
        omega = df.loc[idx, "wrist_angular_vel_rads_sm"]

        # Draw elbow + wrist
        cv2.circle(frame, (wx, wy), 6, (0, 255, 0), -1)
        cv2.circle(frame, (ex, ey), 6, (255, 0, 0), -1)

        # Draw forearm line
        cv2.line(frame, (ex, ey), (wx, wy), (255, 255, 255), 2)

        # Draw velocity vector (scaled for visibility)
        scale = 0.05
        vx = df.loc[idx, "wrist_vx_mps"]
        vy = df.loc[idx, "wrist_vy_mps"]
        end_x = int(wx + vx / meters_per_pixel * scale)
        end_y = int(wy + vy / meters_per_pixel * scale)

        cv2.arrowedLine(frame, (wx, wy), (end_x, end_y), (0, 255, 255), 3)

        y0 = 35
        cv2.putText(frame, f"Frame: {frame_no}", (20, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.putText(frame, f"Wrist Speed: {speed_mps:.2f} m/s ({speed_kmh:.1f} km/h)",
                    (20, y0+35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.putText(frame, f"Angular Velocity: {omega:.2f} rad/s",
                    (20, y0+70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        if frame_no == release_frame:
            cv2.putText(frame, "RELEASE", (width-220, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)

        if frame_no == peak_frame:
            cv2.putText(frame, "PEAK WRIST SPEED", (width-350, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

        # Slow motion by duplicating frames
        for _ in range(slow_mo_factor):
            out.write(frame)

        if frame_no % 100 == 0:
            print(f"  Processed {frame_no}")

    cap.release()
    out.release()
    print("✅ Annotated wrist velocity video saved.")

    

# =========================
# MAIN
# =========================
def main():
    ensure_dir(OUT_DIR)

    # 1) Calibration
    meters_per_pixel = calibrate_pixels_to_m(VIDEO_PATH, STUMP_HEIGHT_M)

    # 2) Load pose model
    pose_model = YOLO(MODEL_PATH)

    # 3) Video info
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    dt = 1.0 / fps

    elbow_name, wrist_name = pick_arm_points(BOWLING_ARM)
    print(f"\n🎥 Frames={total_frames}, FPS={fps:.2f}")
    print(f"🧤 Bowling arm={BOWLING_ARM} -> elbow={elbow_name}, wrist={wrist_name}")
    print(f"🎯 Release detect mode: {RELEASE_DETECT_MODE}")

    # 4) Extract elbow + wrist keypoints
    rows = []
    frame_num = 0
    print("\n⏳ Extracting wrist/elbow keypoints with YOLO pose...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_num += 1

        data = {"frame": frame_num}
        for p in SELECTED_POINTS:
            data[f"{p}_x"] = np.nan
            data[f"{p}_y"] = np.nan

        res = pose_model.predict(frame, verbose=False)
        if len(res) > 0 and res[0].keypoints is not None and len(res[0].keypoints) > 0:
            kps = res[0].keypoints.xy
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
    df.to_csv(KEYPOINT_CSV, index=False)
    print(f"\n✅ Saved keypoints CSV: {KEYPOINT_CSV}")

    # 5) Fill missing (simple; keeps script robust)
    df = pd.read_csv(KEYPOINT_CSV)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    df = stabilize_left_right_labels(df)

    # 6) Smooth wrist & elbow coords
    wx = smooth_signal(df[f"{wrist_name}_x"].values.astype(float), SMOOTH_WINDOW, SMOOTH_POLY)
    wy = smooth_signal(df[f"{wrist_name}_y"].values.astype(float), SMOOTH_WINDOW, SMOOTH_POLY)
    ex = smooth_signal(df[f"{elbow_name}_x"].values.astype(float), SMOOTH_WINDOW, SMOOTH_POLY)
    ey = smooth_signal(df[f"{elbow_name}_y"].values.astype(float), SMOOTH_WINDOW, SMOOTH_POLY)

    df["wrist_x_sm"] = wx
    df["wrist_y_sm"] = wy
    df["elbow_x_sm"] = ex
    df["elbow_y_sm"] = ey

    # 7) Wrist linear velocity (m/s)
    vx_px = central_diff(wx, dt)
    vy_px = central_diff(wy, dt)

    vx_m = vx_px * meters_per_pixel
    vy_m = vy_px * meters_per_pixel
    wrist_speed_mps = np.sqrt(vx_m * vx_m + vy_m * vy_m)

    # smooth speed slightly (helps peak stability)
    wrist_speed_mps_sm = smooth_signal(wrist_speed_mps, SMOOTH_WINDOW, SMOOTH_POLY)

    df["wrist_vx_mps"] = vx_m
    df["wrist_vy_mps"] = vy_m
    df["wrist_speed_mps"] = wrist_speed_mps
    df["wrist_speed_mps_sm"] = wrist_speed_mps_sm
    df["wrist_speed_kmh_sm"] = wrist_speed_mps_sm * 3.6

    # 8) Wrist angular velocity (rad/s) of forearm segment (elbow->wrist)
    # angle of vector (wrist - elbow)
    theta = np.arctan2((wy - ey), (wx - ex))  # radians
    theta_u = unwrap_angle(theta)
    omega = central_diff(theta_u, dt)         # rad/s
    omega_sm = smooth_signal(omega, SMOOTH_WINDOW, SMOOTH_POLY)

    df["forearm_angle_rad"] = theta_u
    df["wrist_angular_vel_rads"] = omega
    df["wrist_angular_vel_rads_sm"] = omega_sm

    # 9) Detect release frame
    release_idx, release_frame = find_release_frame(
        df, fps,
        wrist_col_y_sm=df["wrist_y_sm"].values.astype(float),
        wrist_speed_sm=wrist_speed_mps_sm
    )

    # Peak wrist velocity (in full clip) + near release window
    peak_idx = int(np.nanargmax(wrist_speed_mps_sm))
    peak_frame = int(df.loc[peak_idx, "frame"])
    peak_speed_mps = float(wrist_speed_mps_sm[peak_idx])

    # “peak near release”: within ±N frames of release (better for delivery analysis)
    window = int(round(0.12 * fps))  # ~120 ms each side
    a = max(0, release_idx - window)
    b = min(len(df) - 1, release_idx + window)
    near_idx = a + int(np.nanargmax(wrist_speed_mps_sm[a:b+1]))
    near_frame = int(df.loc[near_idx, "frame"])
    near_peak_mps = float(wrist_speed_mps_sm[near_idx])

    # Peak angular vel near release
    near_omega = omega_sm[a:b+1]
    near_omega_idx = a + int(np.nanargmax(np.abs(near_omega)))
    near_omega_frame = int(df.loc[near_omega_idx, "frame"])
    near_omega_peak = float(omega_sm[near_omega_idx])

    # Wrist speed at release (proxy)
    wrist_at_release_mps = float(wrist_speed_mps_sm[release_idx])
    wrist_at_release_kmh = wrist_at_release_mps * 3.6

    print(f"\n🎯 Release frame: {release_frame} (idx={release_idx})")
    print(f"🧤 Wrist speed @ release: {wrist_at_release_mps:.2f} m/s ({wrist_at_release_kmh:.1f} km/h)")
    print(f"📈 Peak wrist speed (overall): {peak_speed_mps:.2f} m/s at frame {peak_frame}")
    print(f"📈 Peak wrist speed (near release): {near_peak_mps:.2f} m/s at frame {near_frame}")
    print(f"🌀 Peak |angular| velocity near release: {near_omega_peak:.2f} rad/s at frame {near_omega_frame}")

    # 10) Ball speed estimation
    # Mode A (proxy): ball speed ≈ wrist speed at release (CORRELATED, not equal)
    ball_speed_proxy = {
        "method": "Mode A (proxy)",
        "ball_speed_proxy_mps": wrist_at_release_mps,
        "ball_speed_proxy_kmh": wrist_at_release_kmh,
        "note": (
            "Proxy estimate using wrist speed at detected release. "
            "Literature: wrist velocity correlates with ball release speed, but not identical."
        )
    }

    # Mode B (if ball detector is provided)
    ball_speed_mode_b = None
    if BALL_MODEL_PATH is not None:
        print("\n⚾ Ball speed Mode B enabled (ball detector model provided).")
        ball_model = YOLO(BALL_MODEL_PATH)
        ball_speed_mode_b = compute_ball_speed_mode_b(
            video_path=VIDEO_PATH,
            release_frame=release_frame,
            meters_per_pixel=meters_per_pixel,
            fps=fps,
            ball_model=ball_model,
            n_frames=BALL_TRACK_FRAMES
        )
        if ball_speed_mode_b is None:
            print("⚠️ Could not compute Mode B ball speed (no reliable detections).")
        else:
            print(f"⚾ Ball speed (Mode B avg): {ball_speed_mode_b['ball_speed_avg_kmh']:.1f} km/h")
            print(f"⚾ Ball speed (Mode B peak): {ball_speed_mode_b['ball_speed_peak_kmh']:.1f} km/h")

    # 11) Save time series CSV (so you can plot in Excel / Python later)
    df_out = df[[
        "frame",
        "wrist_x_sm", "wrist_y_sm",
        "elbow_x_sm", "elbow_y_sm",
        "wrist_vx_mps", "wrist_vy_mps",
        "wrist_speed_mps_sm", "wrist_speed_kmh_sm",
        "forearm_angle_rad",
        "wrist_angular_vel_rads_sm",
    ]].copy()
    df_out.to_csv(TIMESERIES_CSV, index=False)
    print(f"\n📌 Saved wrist time-series: {TIMESERIES_CSV}")

    # 12) Save summary metrics JSON
    metrics = {
        "video": os.path.basename(VIDEO_PATH),
        "fps": fps,
        "bowling_arm": BOWLING_ARM,
        "release_detect_mode": RELEASE_DETECT_MODE,
        "calibration": {
            "stump_height_m": float(STUMP_HEIGHT_M),
            "meters_per_pixel": float(meters_per_pixel),
        },
        "release": {
            "release_idx": int(release_idx),
            "release_frame": int(release_frame),
        },
        "wrist_linear_velocity": {
            "wrist_speed_at_release_mps": wrist_at_release_mps,
            "wrist_speed_at_release_kmh": wrist_at_release_kmh,
            "peak_wrist_speed_overall_mps": peak_speed_mps,
            "peak_wrist_speed_overall_frame": int(peak_frame),
            "peak_wrist_speed_near_release_mps": near_peak_mps,
            "peak_wrist_speed_near_release_frame": int(near_frame),
        },
        "wrist_angular_velocity": {
            "peak_abs_angular_vel_near_release_rads": float(near_omega_peak),
            "peak_abs_angular_vel_near_release_frame": int(near_omega_frame),
            "note": "Angular velocity computed from forearm segment angle (elbow->wrist) in 2D."
        },
        "ball_release_speed_estimates": {
            "proxy_mode_a": ball_speed_proxy,
            "mode_b_ball_tracking": ball_speed_mode_b
        }
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"📌 Saved metrics JSON: {METRICS_JSON}")

    

    print("\n✅ Done.")

        # 13) Optional: write annotated video with wrist velocity vectors
    annotated_path = os.path.join(OUT_DIR, "wrist_velocity_annotated.mp4")
    write_wrist_velocity_annotated_video(
        video_path=VIDEO_PATH,
        df=df,
        fps=fps,
        out_path=annotated_path,
        release_frame=release_frame,
        peak_frame=near_frame,
        meters_per_pixel=meters_per_pixel,
        slow_mo_factor=4
    )

    print(f"📁 Annotated video saved: {annotated_path}")

if __name__ == "__main__":
    main()