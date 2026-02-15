"""
Fast Bowling: Head position + (optional) 2D projected COM proxy analysis
Single script that works for either:
- SIDE view (90° to bowler)  -> forward/back head offset relative to front foot (Dx)
- FRONT view (0° / face-on)  -> lateral head offset relative to front foot (Dx)

Requirements covered:
✅ Pose extraction (YOLOv8 pose)
✅ Pixel→cm calibration using stump height (click top & bottom)
✅ Detect release frame (BOWLING wrist highest => min y)
✅ Detect BFC + FFC (from ankle y peaks + stable logic)
✅ Compute head position offset to front foot at BFC and FFC
✅ Report metrics in pixels + cm

Notes:
- This is 2D (projected) analysis.
- “Head COM” here is a proxy from face keypoints (nose/eyes/ears).
- Front foot is chosen automatically near FFC based on direction of motion (SIDE view) or based on later-contact (FRONT view).
- This supports BOTH right-arm and left-arm bowlers via BOWLING_ARM.
"""

import cv2
import os
import json
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.signal import savgol_filter, find_peaks

# =========================
# USER SETTINGS
# =========================
VIDEO_PATH = "front2.MOV"
VIEW_MODE = "FRONT"   # "SIDE" (90°) or "FRONT" (0°)

# NEW: set which arm the bowler uses
BOWLING_ARM = "RIGHT"  # "RIGHT" or "LEFT"

MODEL_PATH = "yolov8n-pose.pt"

OUT_DIR = "output"
KEYPOINT_CSV = os.path.join(OUT_DIR, "yolo_keypoints.csv")
METRICS_JSON = os.path.join(OUT_DIR, "head_metrics.json")
METRICS_CSV = os.path.join(OUT_DIR, "head_metrics.csv")

# Smoothing
SMOOTH_WINDOW = 9  # must be odd; will auto-fix if too large
SMOOTH_POLY = 2

# Calibration: stump height in meters (ICC stumps: 71.1 cm)
STUMP_HEIGHT_M = 0.711

# Peak detection tuning (you can adjust if needed)
ANKLE_PROMINENCE_MAIN = 5
ANKLE_PROMINENCE_RELAXED = 1

# Optional: extra robustness for release detection (recommended)
# - "ARM": use specified BOWLING_ARM wrist only
# - "AUTO": pick the wrist that reaches the highest point (lowest y) overall
RELEASE_MODE = "ARM"  # "ARM" or "AUTO"

# =========================
# VALIDATION + ARM MAPPING
# =========================
VIEW_MODE = VIEW_MODE.upper().strip()
if VIEW_MODE not in ["SIDE", "FRONT"]:
    raise ValueError("VIEW_MODE must be 'SIDE' or 'FRONT'")

BOWLING_ARM = BOWLING_ARM.upper().strip()
if BOWLING_ARM not in ["RIGHT", "LEFT"]:
    raise ValueError("BOWLING_ARM must be 'RIGHT' or 'LEFT'")

RELEASE_MODE = RELEASE_MODE.upper().strip()
if RELEASE_MODE not in ["ARM", "AUTO"]:
    raise ValueError("RELEASE_MODE must be 'ARM' or 'AUTO'")

BOWLING_WRIST = "right_wrist" if BOWLING_ARM == "RIGHT" else "left_wrist"
NON_BOWLING_WRIST = "left_wrist" if BOWLING_ARM == "RIGHT" else "right_wrist"

# =========================
# KEYPOINT DEFINITIONS
# =========================
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

FACE_POINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]

SELECTED_POINTS = [
    # face
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    # upper body
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    # lower body
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

# =========================
# HELPERS
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def make_window(n, length):
    """Make a valid odd window length for Savitzky-Golay filter."""
    w = n if n % 2 == 1 else n + 1
    if length <= 2:
        return 3
    if w >= length:
        w = length - 1 if (length - 1) % 2 == 1 else length - 2
    return max(3, w)

def smooth_signal(sig, win, poly=2):
    w = make_window(win, len(sig))
    if len(sig) < 3:
        return sig.copy()
    return savgol_filter(sig, window_length=w, polyorder=min(poly, w - 1))

def safe_mean_xy(df, points, frame_idx):
    """Mean of available keypoints for head proxy on a specific frame index."""
    xs, ys = [], []
    for p in points:
        x = df.loc[frame_idx, f"{p}_x"]
        y = df.loc[frame_idx, f"{p}_y"]
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x); ys.append(y)
    if len(xs) == 0:
        return np.nan, np.nan
    return float(np.mean(xs)), float(np.mean(ys))

def midpoint(df, a, b):
    """Return midpoint arrays (x,y) of two keypoints."""
    mx = (df[f"{a}_x"].values + df[f"{b}_x"].values) / 2.0
    my = (df[f"{a}_y"].values + df[f"{b}_y"].values) / 2.0
    return mx, my

def pick_direction(hip_mid_x):
    """
    Estimate run direction in image x-axis:
    +1 means moving to the right, -1 means moving to the left.
    Uses early vs late hip midpoint x.
    """
    n = len(hip_mid_x)
    if n < 10:
        return 1
    early = np.nanmean(hip_mid_x[: max(5, n // 10)])
    late = np.nanmean(hip_mid_x[-max(5, n // 10):])
    return 1 if (late - early) >= 0 else -1

def detect_release_arm(df, fps, bowling_wrist: str):
    """
    Release frame proxy: bowling wrist highest => minimum y in image coordinates.
    Returns release_idx (0-based), release_frame (1-based frame number), wrist_y_sm.
    """
    wrist_y = df[f"{bowling_wrist}_y"].values.astype(float)
    wrist_y_sm = smooth_signal(wrist_y, SMOOTH_WINDOW, SMOOTH_POLY)
    release_idx = int(np.nanargmin(wrist_y_sm))
    release_frame = int(df.loc[release_idx, "frame"])
    return release_idx, release_frame, bowling_wrist, wrist_y_sm

def detect_release_auto(df, fps):
    """
    Auto-selects the wrist that reaches the highest point (lowest y) overall,
    then uses that wrist to choose the release frame proxy.
    """
    ly = df["left_wrist_y"].values.astype(float)
    ry = df["right_wrist_y"].values.astype(float)
    ly_sm = smooth_signal(ly, SMOOTH_WINDOW, SMOOTH_POLY)
    ry_sm = smooth_signal(ry, SMOOTH_WINDOW, SMOOTH_POLY)

    if np.nanmin(ly_sm) < np.nanmin(ry_sm):
        wrist = "left_wrist"
        sig = ly_sm
    else:
        wrist = "right_wrist"
        sig = ry_sm

    release_idx = int(np.nanargmin(sig))
    release_frame = int(df.loc[release_idx, "frame"])
    return release_idx, release_frame, wrist, sig

def detect_peaks(sig, fps, prominence):
    """
    Foot contact proxy: ankle y peaks (y increases downward; foot on ground often max y).
    """
    min_distance_frames = max(3, int(0.06 * fps))  # ~60ms
    peaks, props = find_peaks(sig, distance=min_distance_frames, prominence=prominence)
    return peaks, props

# =========================
# CALIBRATION (CLICK STUMP TOP + BOTTOM)
# =========================
def calibrate_pixels_to_cm(video_path, stump_height_m=0.711):
    """
    Shows first frame. User clicks:
      1) top of stump
      2) bottom of stump
    Returns: pixels_per_meter, cm_per_pixel
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

    disp = frame.copy()
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration", on_mouse)

    print("\nCALIBRATION:")
    print("Click 1) TOP of stump, then 2) BOTTOM of stump. Press ESC to cancel.")

    while True:
        show = disp.copy()
        for i, (cx, cy) in enumerate(clicks):
            cv2.circle(show, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(show, f"{i+1}", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(show, "Click TOP then BOTTOM of stump (2 clicks). ESC to cancel.",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Calibration", show)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            cv2.destroyWindow("Calibration")
            raise RuntimeError("Calibration cancelled.")
        if len(clicks) >= 2:
            break

    cv2.destroyWindow("Calibration")

    (x1, y1), (x2, y2) = clicks[0], clicks[1]
    pix_dist = float(np.hypot(x2 - x1, y2 - y1))
    if pix_dist < 2:
        raise RuntimeError("Calibration failed: clicked points too close.")

    pixels_per_meter = pix_dist / float(stump_height_m)
    cm_per_pixel = (100.0 / pixels_per_meter)
    print(f"✅ Calibration done. stump_px={pix_dist:.2f} px, pixels_per_meter={pixels_per_meter:.2f}, cm_per_pixel={cm_per_pixel:.4f}")
    return pixels_per_meter, cm_per_pixel

def write_annotated_trimmed_video(
    video_path: str,
    df: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    fps: float,
    out_path: str,
    front_ankle: str,
    ffc_frame: int,
    bfc_frame: int | None,
    release_frame: int,
    cm_per_pixel: float,
    slow_mo_factor: int = 6,              # 6x slower overall
    pause_seconds_bfc: float = 1.5,       # pause duration
    pause_seconds_ffc: float = 2.0,
    pause_seconds_release: float = 2.0
):
    """
    Writes a trimmed annotated video from start_frame -> end_frame (inclusive),
    with slow motion and pauses at key events.
    Slow motion is done by duplicating frames (safe & simple).
    Pauses are done by duplicating the event frames more times.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video for annotation output.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Keep the SAME fps, but duplicate frames to slow down
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, float(fps), (width, height))

    # Convert pause seconds into number of repeated frames
    pause_frames_bfc = int(round(pause_seconds_bfc * fps))
    pause_frames_ffc = int(round(pause_seconds_ffc * fps))
    pause_frames_release = int(round(pause_seconds_release * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - 1))

    print(f"\n🎬 Writing SLOW annotated video: {out_path}")
    print(f"   Range: frames {start_frame} → {end_frame}")
    print(f"   Slow-mo factor: {slow_mo_factor}x")
    print(f"   Pauses: BFC={pause_seconds_bfc}s, FFC={pause_seconds_ffc}s, RELEASE={pause_seconds_release}s")

    frame_no = start_frame
    while frame_no <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        idx = frame_no - 1
        if idx < 0 or idx >= len(df):
            break

        # Read points
        hx = float(df.loc[idx, "head_x"])
        hy = float(df.loc[idx, "head_y"])
        fx = float(df.loc[idx, f"{front_ankle}_x"])
        fy = float(df.loc[idx, f"{front_ankle}_y"])

        # Compute offsets
        dx_px = hx - fx
        dy_px = hy - fy
        d_px = float(np.hypot(dx_px, dy_px))
        dx_cm = dx_px * cm_per_pixel
        dy_cm = dy_px * cm_per_pixel
        d_cm = d_px * cm_per_pixel

        # Event label
        label = ""
        if bfc_frame is not None and frame_no == bfc_frame:
            label = "BFC"
        elif frame_no == ffc_frame:
            label = "FFC"
        elif frame_no == release_frame:
            label = "RELEASE"

        # Draw overlays
        cv2.circle(frame, (int(hx), int(hy)), 6, (0, 255, 0), -1)    # head
        cv2.circle(frame, (int(fx), int(fy)), 6, (0, 0, 255), -1)    # front foot
        cv2.line(frame, (int(hx), int(hy)), (int(fx), int(fy)), (255, 255, 255), 2)

        y0 = 35
        cv2.putText(frame, f"Frame: {frame_no}", (20, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if label:
            cv2.putText(frame, f"EVENT: {label}", (20, y0 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)

        cv2.putText(frame, f"Dx: {dx_cm:.1f} cm   Dy: {dy_cm:.1f} cm   D: {d_cm:.1f} cm",
                    (20, y0 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, "Green: Head  |  Red: Front foot", (20, height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Determine how many times to write this frame:
        repeats = max(1, int(slow_mo_factor))

        # Extra repeats to pause at events
        if bfc_frame is not None and frame_no == bfc_frame:
            repeats += pause_frames_bfc
        if frame_no == ffc_frame:
            repeats += pause_frames_ffc
        if frame_no == release_frame:
            repeats += pause_frames_release

        for _ in range(repeats):
            out.write(frame)

        if frame_no % 10 == 0:
            print(f"  → processed frame {frame_no}/{end_frame}")

        frame_no += 1

    cap.release()
    out.release()
    print("✅ Slow + paused annotated video saved.")

# =========================
# MAIN
# =========================
def main():
    ensure_dir(OUT_DIR)

    # 1) Calibrate
    pixels_per_meter, cm_per_pixel = calibrate_pixels_to_cm(VIDEO_PATH, STUMP_HEIGHT_M)

    # 2) Load model and video
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"\n🎥 Total frames: {total_frames}, FPS: {fps:.2f}, VIEW_MODE: {VIEW_MODE}, BOWLING_ARM: {BOWLING_ARM}, RELEASE_MODE: {RELEASE_MODE}")

    # 3) Extract keypoints for all frames
    rows = []
    frame_num = 0
    print("\n⏳ Extracting keypoints with YOLO pose...")

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
                pts = kps[0].cpu().numpy()  # (17, 2)
                for i, name in enumerate(KEYPOINT_NAMES):
                    if name in SELECTED_POINTS:
                        data[f"{name}_x"] = float(pts[i, 0])
                        data[f"{name}_y"] = float(pts[i, 1])

        rows.append(data)
        if frame_num % 100 == 0:
            print(f"  Processed {frame_num}/{total_frames}...")

    cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(KEYPOINT_CSV, index=False)
    print(f"\n✅ Saved keypoints CSV: {KEYPOINT_CSV}")

    # 4) Fill missing values (simple)
    df = pd.read_csv(KEYPOINT_CSV)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # 5) Compute head proxy point per frame (mean of face points)
    head_x, head_y = [], []
    for i in range(len(df)):
        hx, hy = safe_mean_xy(df, FACE_POINTS, i)
        head_x.append(hx)
        head_y.append(hy)
    df["head_x"] = head_x
    df["head_y"] = head_y

    # 6) Hip midpoint (for direction estimation and optional midline)
    hip_mid_x, hip_mid_y = midpoint(df, "left_hip", "right_hip")
    df["hip_mid_x"] = hip_mid_x
    df["hip_mid_y"] = hip_mid_y

    # 7) Release frame detection (arm-aware)
    if RELEASE_MODE == "AUTO":
        release_idx, release_frame, used_wrist, wrist_y_sm = detect_release_auto(df, fps)
    else:
        release_idx, release_frame, used_wrist, wrist_y_sm = detect_release_arm(df, fps, BOWLING_WRIST)

    df[f"{used_wrist}_y_sm"] = wrist_y_sm
    print(f"\n🎯 Release estimate: idx={release_idx} frame={release_frame} (using {used_wrist})")

    # 8) Detect ankle peaks for each ankle (foot contacts)
    l_ank_y = df["left_ankle_y"].values.astype(float)
    r_ank_y = df["right_ankle_y"].values.astype(float)

    l_ank_sm = smooth_signal(l_ank_y, SMOOTH_WINDOW, SMOOTH_POLY)
    r_ank_sm = smooth_signal(r_ank_y, SMOOTH_WINDOW, SMOOTH_POLY)
    df["left_ankle_y_sm"] = l_ank_sm
    df["right_ankle_y_sm"] = r_ank_sm

    l_peaks, _ = detect_peaks(l_ank_sm, fps, ANKLE_PROMINENCE_MAIN)
    r_peaks, _ = detect_peaks(r_ank_sm, fps, ANKLE_PROMINENCE_MAIN)

    l_peaks = l_peaks[l_peaks < release_idx]
    r_peaks = r_peaks[r_peaks < release_idx]

    # Relax if too few
    if len(l_peaks) < 2:
        l_peaks2, _ = detect_peaks(l_ank_sm, fps, ANKLE_PROMINENCE_RELAXED)
        l_peaks = l_peaks2[l_peaks2 < release_idx]
    if len(r_peaks) < 2:
        r_peaks2, _ = detect_peaks(r_ank_sm, fps, ANKLE_PROMINENCE_RELAXED)
        r_peaks = r_peaks2[r_peaks2 < release_idx]

    if len(l_peaks) == 0 or len(r_peaks) == 0:
        print("\n⚠️ Could not find enough ankle peaks for both feet before release.")
        print("   Try increasing video quality, fps, or lowering prominence.")
        metrics = {
            "view_mode": VIEW_MODE,
            "fps": float(fps),
            "bowling_arm": BOWLING_ARM,
            "release_mode": RELEASE_MODE,
            "release_wrist_used": used_wrist,
            "calibration": {
                "stump_height_m": float(STUMP_HEIGHT_M),
                "pixels_per_meter": float(pixels_per_meter),
                "cm_per_pixel": float(cm_per_pixel)
            },
            "release": {"idx": int(release_idx), "frame": int(release_frame)},
            "warning": "Insufficient ankle peaks for BFC/FFC detection."
        }
        with open(METRICS_JSON, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Saved partial metrics: {METRICS_JSON}")
        return

    # 9) Choose delivery stride window: last X steps before release (optional helpful)
    combined_peaks = np.sort(np.unique(np.concatenate([l_peaks, r_peaks])))
    last5 = combined_peaks[-5:] if len(combined_peaks) >= 5 else combined_peaks
    last5_frames = df.loc[last5, "frame"].astype(int).tolist()

    # 10) Determine direction (SIDE view) and find front foot at FFC
    direction = pick_direction(df["hip_mid_x"].values)

    # Candidate "late" contacts: last peak from each foot
    l_last = int(l_peaks[-1])
    r_last = int(r_peaks[-1])

    # Propose FFC as the later of the two last contacts
    if l_last > r_last:
        ffc_idx_candidate = l_last
        ffc_foot_candidate = "left"
    else:
        ffc_idx_candidate = r_last
        ffc_foot_candidate = "right"

    lx = df.loc[ffc_idx_candidate, "left_ankle_x"]
    rx = df.loc[ffc_idx_candidate, "right_ankle_x"]

    if VIEW_MODE == "SIDE":
        # front foot is farther in direction of motion
        if direction == 1:
            front_is_left = (lx >= rx)
        else:
            front_is_left = (lx <= rx)
    else:  # FRONT
        # Use later-contact foot as "front foot" proxy for delivery stride
        front_is_left = (ffc_foot_candidate == "left")

    front_ankle = "left_ankle" if front_is_left else "right_ankle"
    back_ankle = "right_ankle" if front_is_left else "left_ankle"

    # 11) Define FFC and BFC frames:
    front_peaks = l_peaks if front_is_left else r_peaks
    back_peaks = r_peaks if front_is_left else l_peaks

    front_peaks_before_release = front_peaks[front_peaks < release_idx]
    if len(front_peaks_before_release) == 0:
        raise RuntimeError("No front foot peaks before release for FFC.")
    ffc_idx = int(front_peaks_before_release[-1])

    back_peaks_before_ffc = back_peaks[back_peaks < ffc_idx]
    if len(back_peaks_before_ffc) == 0:
        print("\n⚠️ Could not find a back foot peak before FFC -> BFC not detected reliably.")
        bfc_idx = None
    else:
        bfc_idx = int(back_peaks_before_ffc[-1])

    ffc_frame = int(df.loc[ffc_idx, "frame"])
    bfc_frame = int(df.loc[bfc_idx, "frame"]) if bfc_idx is not None else None

    print(f"\n🦶 Detected front ankle: {front_ankle} (direction={direction}, view={VIEW_MODE})")
    print(f"✅ FFC: idx={ffc_idx}, frame={ffc_frame}")
    if bfc_idx is not None:
        print(f"✅ BFC: idx={bfc_idx}, frame={bfc_frame}")
    else:
        print("⚠️ BFC: not reliably detected")

    # 12) Compute head offsets relative to front foot at BFC and FFC
    def offsets_at(idx):
        hx = df.loc[idx, "head_x"]
        hy = df.loc[idx, "head_y"]
        fx = df.loc[idx, f"{front_ankle}_x"]
        fy = df.loc[idx, f"{front_ankle}_y"]
        dx = float(hx - fx)
        dy = float(hy - fy)
        d = float(np.hypot(dx, dy))
        return {
            "head_px": [float(hx), float(hy)],
            "front_foot_px": [float(fx), float(fy)],
            "Dx_px": dx,
            "Dy_px": dy,
            "D_px": d,
            "Dx_cm": dx * cm_per_pixel,
            "Dy_cm": dy * cm_per_pixel,
            "D_cm": d * cm_per_pixel,
        }

    ffc_metrics = offsets_at(ffc_idx)
    bfc_metrics = offsets_at(bfc_idx) if bfc_idx is not None else None

    # Optional: head relative to pelvis midline
    def head_vs_midline(idx):
        hx = df.loc[idx, "head_x"]
        hy = df.loc[idx, "head_y"]
        mx = df.loc[idx, "hip_mid_x"]
        my = df.loc[idx, "hip_mid_y"]
        dx = float(hx - mx)
        dy = float(hy - my)
        d = float(np.hypot(dx, dy))
        return {
            "midline_px": [float(mx), float(my)],
            "Dx_px": dx,
            "Dy_px": dy,
            "D_px": d,
            "Dx_cm": dx * cm_per_pixel,
            "Dy_cm": dy * cm_per_pixel,
            "D_cm": d * cm_per_pixel,
        }

    ffc_midline = head_vs_midline(ffc_idx)
    bfc_midline = head_vs_midline(bfc_idx) if bfc_idx is not None else None

    # 13) Save metrics
    metrics = {
        "view_mode": VIEW_MODE,
        "bowling_arm": BOWLING_ARM,
        "release_mode": RELEASE_MODE,
        "release_wrist_used": used_wrist,
        "fps": float(fps),
        "calibration": {
            "stump_height_m": float(STUMP_HEIGHT_M),
            "pixels_per_meter": float(pixels_per_meter),
            "cm_per_pixel": float(cm_per_pixel),
        },
        "events": {
            "release": {"idx": int(release_idx), "frame": int(release_frame)},
            "FFC": {"idx": int(ffc_idx), "frame": int(ffc_frame), "front_ankle": front_ankle},
            "BFC": {"idx": int(bfc_idx) if bfc_idx is not None else None,
                    "frame": int(bfc_frame) if bfc_frame is not None else None,
                    "back_ankle": back_ankle},
            "last5_contact_frames_before_release": last5_frames,
        },
        "head_relative_to_front_foot": {
            "FFC": ffc_metrics,
            "BFC": bfc_metrics,
            "interpretation_note": (
                "In SIDE view, Dx is forward/back (lean). "
                "In FRONT view, Dx is lateral (off to side)."
            ),
        },
        "head_relative_to_pelvis_midline": {
            "FFC": ffc_midline,
            "BFC": bfc_midline,
            "why": "Useful stability proxy; especially for FRONT view lateral head deviation."
        }
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n📌 Saved metrics JSON: {METRICS_JSON}")

    # 14) Save a tidy CSV summary (one row)
    row = {
        "video": os.path.basename(VIDEO_PATH),
        "view_mode": VIEW_MODE,
        "bowling_arm": BOWLING_ARM,
        "release_mode": RELEASE_MODE,
        "release_wrist_used": used_wrist,
        "fps": float(fps),
        "cm_per_pixel": float(cm_per_pixel),

        "release_frame": int(release_frame),
        "ffc_frame": int(ffc_frame),
        "bfc_frame": int(bfc_frame) if bfc_frame is not None else None,

        "front_ankle": front_ankle,
        "back_ankle": back_ankle,

        # FFC offsets to front foot
        "ffc_Dx_cm": ffc_metrics["Dx_cm"],
        "ffc_Dy_cm": ffc_metrics["Dy_cm"],
        "ffc_D_cm": ffc_metrics["D_cm"],

        # FFC head vs midline
        "ffc_midline_Dx_cm": ffc_midline["Dx_cm"],
        "ffc_midline_Dy_cm": ffc_midline["Dy_cm"],
        "ffc_midline_D_cm": ffc_midline["D_cm"],
    }

    if bfc_metrics is not None:
        row.update({
            "bfc_Dx_cm": bfc_metrics["Dx_cm"],
            "bfc_Dy_cm": bfc_metrics["Dy_cm"],
            "bfc_D_cm": bfc_metrics["D_cm"],
        })
    else:
        row.update({"bfc_Dx_cm": None, "bfc_Dy_cm": None, "bfc_D_cm": None})

    if bfc_midline is not None:
        row.update({
            "bfc_midline_Dx_cm": bfc_midline["Dx_cm"],
            "bfc_midline_Dy_cm": bfc_midline["Dy_cm"],
            "bfc_midline_D_cm": bfc_midline["D_cm"],
        })
    else:
        row.update({"bfc_midline_Dx_cm": None, "bfc_midline_Dy_cm": None, "bfc_midline_D_cm": None})

    pd.DataFrame([row]).to_csv(METRICS_CSV, index=False)
    print(f"📌 Saved metrics CSV: {METRICS_CSV}")

    # 15) Annotated trimmed video: BFC → Release (or FFC → Release if BFC missing)
    start_frame = bfc_frame if bfc_frame is not None else ffc_frame
    end_frame = release_frame

    annotated_path = os.path.join(OUT_DIR, "annotated_delivery_phase.mp4")

    write_annotated_trimmed_video(
        video_path=VIDEO_PATH,
        df=df,
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        fps=float(fps),
        out_path=annotated_path,
        front_ankle=front_ankle,
        ffc_frame=int(ffc_frame),
        bfc_frame=int(bfc_frame) if bfc_frame is not None else None,
        release_frame=int(release_frame),
        cm_per_pixel=float(cm_per_pixel)
    )

    print(f"📁 Annotated trimmed video: {annotated_path}")
    print("\n✅ Done.")
    print("Tip: run once for SIDE video and once for FRONT video (change VIEW_MODE and VIDEO_PATH).")

if __name__ == "__main__":
    main()