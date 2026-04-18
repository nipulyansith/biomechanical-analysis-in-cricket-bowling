"""
FAST BOWLING — UNIFIED BIOMECHANICS PIPELINE + DATASET WRITER
==============================================================
FIXES in this version:
  FIX A: stump_calibration — robust video opening with retries, codec fallback,
          and helpful error messages instead of a bare RuntimeError crash.

  FIX B: Release frame detection completely rewritten.
          Uses a 4-signal ensemble with confidence voting:
            1. Wrist peak speed (in delivery window)
            2. Wrist highest elevation (min-y in delivery window)
            3. Shoulder-to-wrist vector angle peak (arm fully extended)
            4. Post-peak deceleration magnitude
          Each signal nominates a candidate frame and votes with a weight.
          The frame with the highest weighted vote wins inside a tight
          ±fps/3 consensus window around the median nominee.

  FIX C: Last-5 foot contacts rewritten with a 3-pass approach:
          Pass 1 — per-ankle peak detection with adaptive parameters
          Pass 2 — cross-ankle cluster deduplication (keeps the most
                   grounded candidate inside each MIN_CLUSTER_GAP window)
          Pass 3 — forward-walking alternation enforcer: builds the
                   longest valid alternating sequence that ends at or
                   before release, then picks the final 5.
          FFC injection is still guaranteed (unchanged from original Fix 4).

  FIX D: FRAME NUMBERING CONSISTENCY — all 6 video annotation modules
          now use the same frame-tracking mechanism:
            • New helper _open_and_advance() uses cap.grab() (no-decode)
              for frame-accurate seeking instead of cap.set(CAP_PROP_POS_FRAMES)
              which is unreliable on MOV/H.264 (seeks to nearest keyframe only).
            • Every annotation loop derives frame_no from
              cap.get(cv2.CAP_PROP_POS_FRAMES) after each cap.read() call,
              so the counter always reflects the actual decoded frame, not
              a manually-maintained offset that can drift from the seek target.
            • idx = frame_no - 1 is used uniformly across all 6 loops for
              0-based df_xy row lookups, matching how df_xy was built in
              extract_keypoints().
"""

import os, json, datetime
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from scipy.signal import savgol_filter, find_peaks
import subprocess
import openpyxl

# =============================================================================
# ── SETTINGS
# =============================================================================
VIDEO_PATH   = r"C:\Nipul\videos\B-10_T-10.MOV"
MODEL_PATH   = "yolov8l-pose.pt"
BASE_OUT_DIR = "output"

FRAMES_DATASET_PATH = "frames.xlsx"
MASTER_DATASET_PATH = "master.xlsx"

STUMP_HEIGHT_M  = 0.711
SMOOTH_WINDOW   = 9
SMOOTH_POLY     = 2
OUTPUT_SCALE    = 0.5

VIEW_MODE            = "SIDE"
BALL_MODEL_PATH      = None   # set to a ball-detector .pt path to enable Mode B
BALL_TRACK_FRAMES    = 12
KNEE_MAX_JUMP_LEGLEN = 0.60
KNEE_CONF_TH         = 0.25
ANKLE_PROMINENCE_MAIN    = 5
ANKLE_PROMINENCE_RELAXED = 1

# =============================================================================
# ── OUTPUT DIRECTORY HELPERS
# =============================================================================
OUT_DIR = BASE_OUT_DIR

def setup_output_dir(video_path: str, base_dir: str) -> str:
    stem     = os.path.splitext(os.path.basename(video_path))[0]
    base_try = os.path.join(base_dir, stem)
    folder   = base_try
    counter  = 2
    while os.path.exists(folder) and os.listdir(folder):
        folder = f"{base_try}_run{counter}"
        counter += 1
    os.makedirs(folder, exist_ok=True)
    return folder

def out_path(*parts):
    return os.path.join(OUT_DIR, *parts)

# =============================================================================
# ── VIDEO COMPRESSION HELPER
# =============================================================================
def compress_video(input_path: str, crf: int = 28) -> str:
    if not os.path.exists(input_path):
        return input_path
    stem    = os.path.splitext(input_path)[0]
    out_vid = stem + "_c.mp4"
    vf      = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    cmd = ["ffmpeg", "-y", "-i", input_path,
           "-vcodec", "libx264", "-crf", str(crf),
           "-preset", "fast", "-vf", vf, "-an", out_vid]
    try:
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(input_path)
        os.rename(out_vid, input_path)
        print(f"  🗜️  Compressed → {os.path.basename(input_path)}  "
              f"({os.path.getsize(input_path)/1e6:.1f} MB)")
    except (subprocess.CalledProcessError, FileNotFoundError):
        if os.path.exists(out_vid):
            os.remove(out_vid)
        print("  ⚠️  ffmpeg not found or failed — video kept uncompressed.")
    return input_path

# =============================================================================
# ── COCO KEYPOINT DEFINITIONS
# =============================================================================
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
KP_IDX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
FACE_POINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]

SKELETON_CONNECTIONS = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
]

KEYPOINT_COLORS = {
    "left_wrist":      (0, 255, 255),  "right_wrist":     (0, 200, 255),
    "left_elbow":      (0, 255, 150),  "right_elbow":     (0, 150, 255),
    "left_shoulder":   (255, 100, 0),  "right_shoulder":  (255, 0, 100),
    "left_hip":        (200, 0, 255),  "right_hip":       (150, 0, 200),
    "left_knee":       (0, 200, 100),  "right_knee":      (0, 100, 200),
    "left_ankle":      (100, 255, 0),  "right_ankle":     (50, 200, 0),
}

# =============================================================================
# ── SECTION 1 : USER INPUT
# =============================================================================
def get_user_inputs():
    print("\n" + "="*60)
    print("  FAST BOWLING — UNIFIED BIOMECHANICS PIPELINE")
    print("="*60)
    hand = input("\nBowling hand (R / L): ").strip().upper()
    if hand not in ["R", "L"]:
        raise ValueError("Enter R or L only.")
    knee_side  = "L" if hand == "R" else "R"
    knee_label = "Left" if knee_side == "L" else "Right"
    print(f"✅ Front knee auto-selected: {knee_label} knee")
    return hand, knee_side

# =============================================================================
# ── SECTION 2 : STUMP CALIBRATION  (FIX A — robust video opening)
# =============================================================================
def _open_video_robust(video_path: str, max_retries: int = 3):
    """
    Try to open a video with cv2.VideoCapture.
    Attempts multiple backends (default, FFMPEG, MSMF) and retries
    to handle transient file-lock or codec issues common with .MOV files.
    Returns (cap, frame) or raises a descriptive RuntimeError.
    """
    if not os.path.exists(video_path):
        raise RuntimeError(
            f"Video file not found: '{video_path}'\n"
            "Please check VIDEO_PATH at the top of the script."
        )

    backends = [
        (cv2.CAP_ANY,   "default"),
        (cv2.CAP_FFMPEG, "FFMPEG"),
    ]
    # On Windows also try MSMF
    if hasattr(cv2, "CAP_MSMF"):
        backends.append((cv2.CAP_MSMF, "MSMF"))

    last_error = ""
    for backend, name in backends:
        for attempt in range(1, max_retries + 1):
            try:
                cap = cv2.VideoCapture(video_path, backend)
                if not cap.isOpened():
                    cap.release()
                    last_error = f"Backend {name}: cap.isOpened() returned False"
                    continue

                # Give the demuxer a moment on large MOV files
                import time; time.sleep(0.1)

                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    print(f"  ✅ Video opened via backend={name} (attempt {attempt})")
                    return cap, frame

                cap.release()
                last_error = (f"Backend {name} attempt {attempt}: "
                              "cap.read() returned empty frame")
            except Exception as exc:
                last_error = f"Backend {name} attempt {attempt}: {exc}"

    raise RuntimeError(
        f"Could not read first frame from '{video_path}'.\n"
        f"Last error: {last_error}\n"
        "Possible causes:\n"
        "  • File path contains non-ASCII characters or spaces — try moving the file\n"
        "  • File is still being written / copied\n"
        "  • Missing codec for .MOV — install K-Lite or use ffmpeg to convert to .mp4\n"
        "  • File is corrupt — verify it plays in VLC\n"
        "  • OpenCV built without FFMPEG support — reinstall: pip install opencv-python"
    )


# =============================================================================
# ── FIX D: FRAME-ACCURATE SEEK HELPER
# =============================================================================
def _open_and_advance(video_path: str, start_frame_1based: int) -> cv2.VideoCapture:
    """
    Open the video and advance to start_frame_1based using cap.grab() for
    frame-accurate positioning.

    cap.set(CAP_PROP_POS_FRAMES) is NOT used for seeking because on MOV/H.264
    files it snaps to the nearest keyframe but cap.get() still reports the
    *requested* position — so the verification check passes falsely and the
    2-frame (or N-frame) offset is never caught.

    Instead we always seek from frame 0 using cap.grab() (no decode overhead).
    After target_0based grabs, the next cap.read() returns exactly the frame
    numbered start_frame_1based (1-based), matching extract_keypoints().
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for annotation: {video_path}")

    target_0based = max(0, start_frame_1based - 1)

    if target_0based == 0:
        return cap   # already at the start — nothing to skip

    # Always use grab() from position 0 — never trust cap.set() on MOV files.
    # grab() is fast (no decode); for a 600-frame seek at 60fps this takes
    # well under a second.
    for _ in range(target_0based):
        if not cap.grab():
            break   # hit end of file unexpectedly

    return cap


def _read_frame_with_number(cap: cv2.VideoCapture):
    """
    Read the next frame and return (ret, frame, frame_no_1based).

    frame_no_1based is derived from cap.get(CAP_PROP_POS_FRAMES) AFTER
    the read, which equals the 0-based index of the frame just decoded + 1,
    i.e. the 1-based frame number.  This is the single authoritative source
    of frame numbering used by ALL six annotation loops (FIX D).
    """
    ret, frame = cap.read()
    if not ret:
        return False, None, -1
    frame_no_1based = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    return True, frame, frame_no_1based


def stump_calibration(video_path, stump_height_m=0.711):
    # ── FIX A: use robust opener ─────────────────────────────────────────────
    cap, frame = _open_video_robust(video_path)
    cap.release()

    orig_h, orig_w = frame.shape[:2]
    max_disp_w = 900
    disp_scale = min(1.0, max_disp_w / orig_w)
    disp_w = int(orig_w * disp_scale)
    disp_h = int(orig_h * disp_scale)
    clicks_display = []
    win_name = "CALIBRATION - Click TOP then BOTTOM of stump"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks_display) < 2:
            clicks_display.append((x, y))
            print(f"Clicked point {len(clicks_display)}: ({x}, {y})")

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, disp_w, disp_h)
    cv2.imshow(win_name, cv2.resize(frame, (disp_w, disp_h)))
    cv2.waitKey(1)
    cv2.setMouseCallback(win_name, on_mouse)
    print("\n" + "=" * 60)
    print("CALIBRATION")
    print("1) Click TOP of stump\n2) Click BOTTOM of stump\n3) Press SPACE to confirm")
    print("Press R to reset | Press ESC to cancel")
    print("=" * 60)

    while True:
        canvas = cv2.resize(frame, (disp_w, disp_h)).copy()
        cv2.rectangle(canvas, (0, 0), (disp_w, 85), (0, 0, 0), -1)
        cv2.putText(canvas,
                    "Click TOP and BOTTOM of stump | SPACE=confirm | R=reset | ESC=cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, f"Points selected: {len(clicks_display)}/2",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for i, (cx, cy) in enumerate(clicks_display):
            label = "TOP" if i == 0 else "BOTTOM"
            cv2.circle(canvas, (cx, cy), 8, (0, 255, 0), -1)
            cv2.circle(canvas, (cx, cy), 10, (255, 255, 255), 2)
            cv2.putText(canvas, label, (cx + 12, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(clicks_display) == 2:
            cv2.line(canvas, clicks_display[0], clicks_display[1], (0, 255, 255), 2)
        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyWindow(win_name)
            raise RuntimeError("Calibration cancelled by user.")
        elif key in (ord('r'), ord('R')):
            clicks_display.clear()
            print("Calibration points reset.")
        elif key == 32:
            if len(clicks_display) < 2:
                print("Please click both TOP and BOTTOM first.")
            else:
                break

    cv2.destroyWindow(win_name)
    (dx1, dy1), (dx2, dy2) = clicks_display
    x1, y1 = dx1 / disp_scale, dy1 / disp_scale
    x2, y2 = dx2 / disp_scale, dy2 / disp_scale
    pix_dist = float(np.hypot(x2 - x1, y2 - y1))
    if pix_dist < 5:
        raise RuntimeError("Calibration failed: clicked points are too close together.")
    pixels_per_meter = pix_dist / stump_height_m
    meters_per_pixel = stump_height_m / pix_dist
    cm_per_pixel     = 100.0 / pixels_per_meter
    print(f"\nCalibration complete.")
    print(f"Stump pixel distance : {pix_dist:.2f} px")
    print(f"Pixels per meter     : {pixels_per_meter:.2f}")
    print(f"CM per pixel         : {cm_per_pixel:.4f}")
    return pixels_per_meter, meters_per_pixel, cm_per_pixel

# =============================================================================
# ── SECTION 3 : KEYPOINT EXTRACTION
# =============================================================================
def extract_keypoints(video_path, model):
    # Use robust opener so we always get consistent behaviour
    cap_test, _ = _open_video_robust(video_path)
    fps          = float(cap_test.get(cv2.CAP_PROP_FPS) or 30.0)
    width        = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_test.release()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

    print(f"\n🎥 Video: {total_frames} frames @ {fps:.2f} fps  ({width}×{height})")
    print("⏳ Extracting keypoints (single pass)…")
    xy_rows, conf_rows = [], []
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        xy_row   = {"frame": frame_no}
        conf_row = {"frame": frame_no}
        for name in KEYPOINT_NAMES:
            xy_row[f"{name}_x"]      = np.nan
            xy_row[f"{name}_y"]      = np.nan
            conf_row[f"{name}_conf"] = np.nan
        results = model.predict(frame, verbose=False)
        if (results and results[0].keypoints is not None
                and len(results[0].keypoints.xy) > 0):
            kps_xy   = results[0].keypoints.xy.cpu().numpy()
            kps_conf = results[0].keypoints.conf.cpu().numpy()
            boxes    = results[0].boxes.xywh.cpu().numpy()
            areas    = boxes[:, 2] * boxes[:, 3]
            idx      = int(np.argmax(areas))
            kp = kps_xy[idx]; cf = kps_conf[idx]
            for i, name in enumerate(KEYPOINT_NAMES):
                xy_row[f"{name}_x"]      = float(kp[i, 0])
                xy_row[f"{name}_y"]      = float(kp[i, 1])
                conf_row[f"{name}_conf"] = float(cf[i])
        xy_rows.append(xy_row)
        conf_rows.append(conf_row)
        if frame_no % 200 == 0:
            print(f"  … {frame_no}/{total_frames}")
    cap.release()
    print(f"✅ Extracted {frame_no} frames.")
    df_xy   = pd.DataFrame(xy_rows)
    df_conf = pd.DataFrame(conf_rows)
    for df in (df_xy, df_conf):
        df.interpolate(inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
    df_xy["time_s"] = (df_xy["frame"].values.astype(float) - 1.0) / fps
    return df_xy, df_conf, fps, width, height, total_frames

# =============================================================================
# ── SECTION 4 : SHARED SIGNAL UTILITIES
# =============================================================================
def make_window(n, length):
    w = n if n % 2 == 1 else n + 1
    if length <= 2:
        return 3
    if w >= length:
        w = length - 1 if (length - 1) % 2 == 1 else length - 2
    return max(3, w)

def smooth(sig, win=SMOOTH_WINDOW, poly=SMOOTH_POLY):
    sig = np.asarray(sig, dtype=float)
    if len(sig) < 3:
        return sig.copy()
    w = make_window(win, len(sig))
    return savgol_filter(sig, window_length=w, polyorder=min(poly, w - 1))

def central_diff(sig, dt):
    sig = np.asarray(sig, dtype=float)
    out = np.zeros_like(sig)
    if len(sig) < 2:
        return out
    out[0]  = (sig[1]  - sig[0])  / dt
    out[-1] = (sig[-1] - sig[-2]) / dt
    if len(sig) > 2:
        out[1:-1] = (sig[2:] - sig[:-2]) / (2.0 * dt)
    return out

def sized_writer(path, fps, width, height):
    ow     = int(width  * OUTPUT_SCALE)
    oh     = int(height * OUTPUT_SCALE)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (ow, oh)), ow, oh

def angle_deg(a, b, c):
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc  = a - b, c - b
    denom   = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1, 1))))

# =============================================================================
# ── SECTION 4b : RELEASE FRAME DETECTION  (FIX B — 4-signal ensemble)
# =============================================================================
def detect_release_frame_biomech(df_xy, bowling_wrist, fps, video_path=None):
    """
    4-Signal Ensemble Release Detector
    ════════════════════════════════════════════════════════════════════════════

    Four independent biomechanical signals each nominate a candidate frame.
    The final release frame is the weighted-median of their nominations,
    then refined within a tight consensus window by a composite score.

    Signal definitions
    ──────────────────
    S1  WRIST SPEED PEAK
        The bowling wrist reaches its maximum speed just as the ball is
        transferred to the fingers.  We locate the dominant speed peak in
        the delivery phase (last 55% of the clip).
        Weight: 0.35

    S2  WRIST ELEVATION MINIMUM  (highest physical point)
        In pixel coordinates, smaller y = higher position.  The arm is
        typically at its apex at or fractionally after release.
        Weight: 0.25

    S3  SHOULDER-TO-WRIST VECTOR ANGLE PEAK
        We compute the angle of the vector from the bowling shoulder to
        the bowling wrist.  At full arm extension / release this angle
        passes through a characteristic maximum (or crosses from
        increasing to decreasing).  We locate that turning-point.
        Weight: 0.25

    S4  POST-PEAK DECELERATION ONSET
        After release, the wrist decelerates sharply because the ball is
        gone and gravity/muscle damping dominate.  We find the frame
        where deceleration (speed drop over next 3 frames) is largest,
        constrained to start no earlier than S1-3 frames.
        Weight: 0.15

    Consensus window & final scoring
    ──────────────────────────────────
    The four nominees are sorted.  We compute a weighted-median nominee
    and define a consensus window of ±(fps/3) frames around it.  Inside
    that window every frame is scored:

        score = 0.40 * norm_speed(i)
              + 0.25 * proximity_to_miny(i)
              + 0.20 * decel_score(i)
              + 0.15 * arm_angle_score(i)

    The frame with the highest score is returned as the release frame.

    Returns: release_idx (0-based), release_frame (1-based), method_used
    """
    dt  = 1.0 / fps
    arm = "right" if bowling_wrist == "right_wrist" else "left"
    n   = len(df_xy)

    # ── 1. Smooth trajectories ───────────────────────────────────────────────
    wx = smooth(df_xy[f"{arm}_wrist_x"].values.astype(float))
    wy = smooth(df_xy[f"{arm}_wrist_y"].values.astype(float))
    sx = smooth(df_xy[f"{arm}_shoulder_x"].values.astype(float))
    sy = smooth(df_xy[f"{arm}_shoulder_y"].values.astype(float))

    vx  = central_diff(wx, dt)
    vy  = central_diff(wy, dt)
    spd = np.sqrt(vx**2 + vy**2)
    spd_sm = smooth(spd)          # second-pass smooth for stability

    # Shoulder→Wrist angle (degrees, 0 = rightward horizontal)
    sw_angle = np.degrees(np.arctan2(wy - sy, wx - sx))
    sw_angle_sm = smooth(sw_angle)

    # ── 2. Delivery search window: last 55% of clip ──────────────────────────
    search_start = int(n * 0.45)
    del_spd  = spd_sm[search_start:]
    del_wy   = wy[search_start:]
    del_ang  = sw_angle_sm[search_start:]

    # ── S1: wrist speed peak ─────────────────────────────────────────────────
    spd_threshold = np.nanpercentile(del_spd, 55)
    min_dist_pk   = max(3, int(0.05 * fps))
    peaks_s1, props_s1 = find_peaks(
        del_spd,
        height=spd_threshold,
        distance=min_dist_pk,
        prominence=np.nanstd(del_spd) * 0.3,
    )
    s1_idx = None
    if len(peaks_s1) > 0:
        # Best = highest speed
        best_local = peaks_s1[int(np.argmax(del_spd[peaks_s1]))]
        s1_idx     = search_start + int(best_local)

    # ── S2: wrist elevation minimum (highest point) ──────────────────────────
    s2_idx = search_start + int(np.nanargmin(del_wy))

    # ── S3: shoulder→wrist angle turning point ───────────────────────────────
    # We look for the last local maximum of the angle in the delivery window
    # (arm sweeps upward = angle increases, then flicks over at release).
    ang_peaks, _ = find_peaks(
        del_ang,
        distance=max(3, int(0.04 * fps)),
        prominence=5.0,
    )
    s3_idx = None
    if len(ang_peaks) > 0:
        s3_idx = search_start + int(ang_peaks[-1])  # last peak before some decline
    else:
        # Fallback: gradient sign change (first derivative zero-crossing)
        grad_ang = np.gradient(del_ang)
        sign_changes = np.where(np.diff(np.sign(grad_ang)) < 0)[0]
        if len(sign_changes) > 0:
            s3_idx = search_start + int(sign_changes[-1])

    # ── S4: largest deceleration onset (constrained to ≥ S1) ─────────────────
    s4_idx = None
    start_s4 = s1_idx if s1_idx is not None else search_start
    # decel[i] = spd_sm[i] - spd_sm[i+3]  (positive = slowing down)
    if start_s4 < n - 3:
        decel = np.zeros(n)
        for i in range(start_s4, n - 3):
            decel[i] = spd_sm[i] - spd_sm[i + 3]
        # Only look past s1 to avoid pre-delivery noise
        decel_window = decel[start_s4:]
        best_decel   = int(np.argmax(decel_window))
        s4_idx       = start_s4 + best_decel

    # ── Diagnostic log ───────────────────────────────────────────────────────
    def _f(idx): return int(df_xy.loc[idx, "frame"]) if idx is not None else None
    print(f"  📊 Release ensemble nominees:")
    print(f"       S1 speed-peak      : frame {_f(s1_idx)}  (idx {s1_idx})")
    print(f"       S2 wrist-elevation : frame {_f(s2_idx)}  (idx {s2_idx})")
    print(f"       S3 arm-angle peak  : frame {_f(s3_idx)}  (idx {s3_idx})")
    print(f"       S4 decel-onset     : frame {_f(s4_idx)}  (idx {s4_idx})")

    # ── 3. Weighted-median consensus ─────────────────────────────────────────
    nominees = []
    weights  = []
    for idx_nom, w in [(s1_idx, 0.35), (s2_idx, 0.25), (s3_idx, 0.25), (s4_idx, 0.15)]:
        if idx_nom is not None:
            nominees.append(idx_nom)
            weights.append(w)

    if len(nominees) == 0:
        # Absolute fallback — last 30% of delivery window
        fallback = search_start + int(len(del_spd) * 0.7)
        print(f"  ⚠️  No nominees found — using fallback idx {fallback}")
        return int(fallback), int(df_xy.loc[fallback, "frame"]), "fallback_no_nominees"

    # Weighted median
    nominees_arr = np.array(nominees, float)
    weights_arr  = np.array(weights,  float)
    weights_arr /= weights_arr.sum()
    sorted_idx   = np.argsort(nominees_arr)
    nominees_arr = nominees_arr[sorted_idx]
    weights_arr  = weights_arr[sorted_idx]
    cum_w        = np.cumsum(weights_arr)
    median_idx   = int(nominees_arr[np.searchsorted(cum_w, 0.5)])

    # ── 4. Consensus window: median ± fps/3 ──────────────────────────────────
    half_win = max(4, int(fps / 3.0))
    win_lo   = max(search_start, median_idx - half_win)
    win_hi   = min(n - 1,        median_idx + half_win)

    print(f"       Weighted-median idx: {median_idx}  "
          f"→ consensus window [{win_lo}, {win_hi}]")

    # Pre-compute normalisation
    max_speed = max(float(np.nanmax(spd_sm)), 1e-9)
    min_y_val = float(np.nanmin(wy[win_lo:win_hi+1]))
    max_y_val = float(np.nanmax(wy[win_lo:win_hi+1]))
    y_range   = max(max_y_val - min_y_val, 1e-9)

    min_ang_val = float(np.nanmin(sw_angle_sm[win_lo:win_hi+1]))
    max_ang_val = float(np.nanmax(sw_angle_sm[win_lo:win_hi+1]))
    ang_range   = max(max_ang_val - min_ang_val, 1e-9)

    best_score = -1.0
    best_idx   = median_idx  # guaranteed fallback

    for i in range(win_lo, win_hi + 1):
        # Component 1: wrist speed
        c_speed = spd_sm[i] / max_speed

        # Component 2: wrist elevation (lower y = higher = better)
        c_elev  = 1.0 - (wy[i] - min_y_val) / y_range

        # Component 3: deceleration (speed drop 3 frames ahead)
        if i + 3 <= n - 1:
            c_decel = max(0.0, spd_sm[i] - spd_sm[i + 3]) / max_speed
        else:
            c_decel = 0.0

        # Component 4: arm angle proximity to its own maximum in the window
        c_ang = (sw_angle_sm[i] - min_ang_val) / ang_range

        score = (0.40 * c_speed + 0.25 * c_elev +
                 0.20 * c_decel + 0.15 * c_ang)

        if score > best_score:
            best_score = score
            best_idx   = i

    final_idx   = int(best_idx)
    method_used = "ensemble_4signal"
    final_frame = int(df_xy.loc[final_idx, "frame"])
    print(f"  ✅ Selected release: frame {final_frame}  "
          f"(idx {final_idx},  score={best_score:.4f},  method={method_used})")
    return final_idx, final_frame, method_used


# =============================================================================
# ── SECTION 4c : SHARED EVENT DETECTION
# =============================================================================
def detect_ankle_peaks(df_xy, fps, side, release_idx):
    col  = f"{side}_ankle_y"
    sig  = smooth(df_xy[col].values.astype(float))
    min_dist = max(3, int(0.06 * fps))
    peaks, _ = find_peaks(sig, distance=min_dist, prominence=ANKLE_PROMINENCE_MAIN)
    peaks    = peaks[peaks < release_idx]
    if len(peaks) < 2:
        peaks2, _ = find_peaks(sig, distance=min_dist, prominence=ANKLE_PROMINENCE_RELAXED)
        peaks     = peaks2[peaks2 < release_idx]
    return peaks, sig


def _find_foot_contact_velocity(df_xy, side, ref_idx, search_before=True):
    col = f"{side}_ankle_y_s"
    if col not in df_xy.columns:
        df_xy[col] = smooth(df_xy[f"{side}_ankle_y"].values.astype(float))
    y_sig    = df_xy[col].values
    velocity = np.gradient(y_sig)
    peaks, _ = find_peaks(y_sig, distance=10)
    if search_before:
        valid    = peaks[peaks <= ref_idx]
        peak_idx = int(valid[-1]) if len(valid) > 0 else ref_idx
    else:
        valid    = peaks[peaks < ref_idx]
        peak_idx = int(valid[-1]) if len(valid) > 0 else max(0, ref_idx - 10)
    refined = peak_idx
    for i in range(peak_idx, max(0, peak_idx - 10), -1):
        if velocity[i] < 0.5:
            refined = i
        else:
            break
    return refined


def detect_shared_events(df_xy, fps, bowling_wrist, video_path=None):
    """Compute ALL shared event frames ONCE."""
    # Release — upgraded 4-signal ensemble
    release_idx, release_frame, release_method = detect_release_frame_biomech(
        df_xy, bowling_wrist, fps, video_path=video_path)

    # FFC / BFC
    for side in ("left", "right"):
        df_xy[f"{side}_ankle_y_s"] = smooth(
            df_xy[f"{side}_ankle_y"].values.astype(float))

    f_side = "left"  if bowling_wrist == "right_wrist" else "right"
    b_side = "right" if f_side == "left" else "left"

    ffc_idx = _find_foot_contact_velocity(df_xy, f_side, release_idx, True)
    bfc_idx = _find_foot_contact_velocity(df_xy, b_side, ffc_idx,     False)

    ffc_frame = int(df_xy.loc[ffc_idx, "frame"])
    bfc_frame = int(df_xy.loc[bfc_idx, "frame"])

    # Arm-back frame
    arm = "right" if bowling_wrist == "right_wrist" else "left"
    wrist_y_sm = smooth(df_xy[f"{arm}_wrist_y"].values.astype(float))
    arm_back_peaks, _ = find_peaks(wrist_y_sm, distance=max(5, int(0.1 * fps)))
    arm_back_candidates = arm_back_peaks[arm_back_peaks < release_idx]

    if len(arm_back_candidates) > 0:
        arm_back_idx   = int(arm_back_candidates[-1])
        arm_back_frame = int(df_xy.loc[arm_back_idx, "frame"])
    else:
        fallback_offset = max(1, int(0.4 * fps))
        arm_back_idx    = max(0, release_idx - fallback_offset)
        arm_back_frame  = int(df_xy.loc[arm_back_idx, "frame"])
        print(f"  ⚠️  No arm-back peak found — using fallback frame {arm_back_frame}")

    print(f"\n📍 SHARED EVENTS  (release method: {release_method})")
    print(f"   BFC     : frame {bfc_frame}  (idx {bfc_idx})")
    print(f"   FFC     : frame {ffc_frame}  (idx {ffc_idx})")
    print(f"   Arm-back: frame {arm_back_frame}  (idx {arm_back_idx})")
    print(f"   Release : frame {release_frame}  (idx {release_idx})")

    return {
        "release_idx":      release_idx,
        "release_frame":    release_frame,
        "release_method":   release_method,
        "ffc_idx":          ffc_idx,
        "ffc_frame":        ffc_frame,
        "bfc_idx":          bfc_idx,
        "bfc_frame":        bfc_frame,
        "f_side":           f_side,
        "b_side":           b_side,
        "arm_back_idx":     arm_back_idx,
        "arm_back_frame":   arm_back_frame,
    }

# =============================================================================
# ── SHARED VIDEO ANNOTATION HELPERS
# =============================================================================
def draw_info_box(img, lines, x=10, y=10, font_scale=0.6,
                  thickness=2, bg_alpha=0.6, color=(255, 255, 255)):
    font   = cv2.FONT_HERSHEY_SIMPLEX
    pad    = 8
    line_h = int((font_scale * 30) + 8)
    max_w  = max(cv2.getTextSize(l, font, font_scale, thickness)[0][0]
                 for l in lines) if lines else 100
    box_h  = line_h * len(lines) + pad * 2
    box_w  = max_w + pad * 2
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
    for i, line in enumerate(lines):
        ty = y + pad + (i + 1) * line_h - 4
        cv2.putText(img, line, (x + pad, ty),
                    font, font_scale, color, thickness, cv2.LINE_AA)

def draw_angle_arc(img, vertex, p1, p2, angle, color=(0, 255, 255),
                   radius=40, thickness=2):
    v    = np.array(vertex, float)
    a1   = np.array(p1, float) - v
    a2   = np.array(p2, float) - v
    ang1 = float(np.degrees(np.arctan2(a1[1], a1[0])))
    ang2 = float(np.degrees(np.arctan2(a2[1], a2[0])))
    if ang1 > ang2: ang1, ang2 = ang2, ang1
    if ang2 - ang1 > 180: ang1, ang2 = ang2, ang1 + 360
    cv2.ellipse(img, (int(v[0]), int(v[1])), (radius, radius),
                0, ang1, ang2, color, thickness, cv2.LINE_AA)
    mid_ang = np.radians((ang1 + ang2) / 2)
    lx = int(v[0] + (radius + 20) * np.cos(mid_ang))
    ly = int(v[1] + (radius + 20) * np.sin(mid_ang))
    cv2.putText(img, f"{angle:.1f}", (lx - 20, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

def draw_skeleton_full(img, kp_xy, scale=1.0, highlight_joints=None):
    kp_dict = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        x, y = kp_xy[i]
        if x > 0 and y > 0:
            kp_dict[name] = (int(x * scale), int(y * scale))
    for p1, p2 in SKELETON_CONNECTIONS:
        if p1 in kp_dict and p2 in kp_dict:
            c1   = KEYPOINT_COLORS.get(p1, (180, 180, 180))
            c2   = KEYPOINT_COLORS.get(p2, (180, 180, 180))
            bone = tuple((a + b) // 2 for a, b in zip(c1, c2))
            cv2.line(img, kp_dict[p1], kp_dict[p2], bone,
                     max(1, int(3 * scale)), cv2.LINE_AA)
    for name, (px, py) in kp_dict.items():
        color = KEYPOINT_COLORS.get(name, (255, 255, 255))
        r     = max(4, int(7 * scale))
        if highlight_joints and name in highlight_joints:
            cv2.circle(img, (px, py), r + 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), r + 3, highlight_joints[name], -1, cv2.LINE_AA)
        else:
            cv2.circle(img, (px, py), r + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), r, color, -1, cv2.LINE_AA)

# =============================================================================
# ── SECTION 5 : MODULE 1 — STEP CADENCE  (BFC=step4, FFC=step5 fixed)
# =============================================================================

def run_step_cadence(df_xy, fps, events, video_path, width, height):
    """
    Step Cadence — BFC and FFC are always steps 4 and 5.
    ════════════════════════════════════════════════════════════════════════
    Steps 1-3  are the last three foot contacts *before* BFC, enforcing
               strict alternation working BACKWARDS from BFC.

               Required sequence (right-arm example):
                 Step1=opposite(BFC), Step2=BFC_foot, Step3=opposite(BFC),
                 Step4=BFC, Step5=FFC

    Step 4     = BFC frame  (from shared events — never moved)
    Step 5     = FFC frame  (from shared events — never moved)
    """
    print("\n" + "─"*60)
    print("MODULE 1 — STEP CADENCE")
    print("─"*60)

    release_frame = events["release_frame"]
    ffc_frame     = events["ffc_frame"]
    ffc_foot      = events["f_side"]
    bfc_frame     = events["bfc_frame"]
    bfc_foot      = events["b_side"]

    MIN_CLUSTER_GAP = max(4, int(0.15 * fps))
    min_dist        = max(6, int(0.20 * fps))

    r_sig = smooth(df_xy["right_ankle_y"].values.astype(float))
    l_sig = smooth(df_xy["left_ankle_y"].values.astype(float))
    ankle_frames = df_xy["frame"].values

    # ── PASS 1: adaptive prominence — search ONLY before BFC ─────────────────
    def find_peaks_adaptive(sig, min_distance, prominence_levels):
        for prom in prominence_levels:
            pks, _ = find_peaks(sig, distance=min_distance, prominence=prom)
            if len(pks) >= 3:
                return pks
        pks, _ = find_peaks(sig, distance=min_distance)
        return pks

    prom_levels = [30, 15, 8, 3]
    r_peaks_all = find_peaks_adaptive(r_sig, min_dist, prom_levels)
    l_peaks_all = find_peaks_adaptive(l_sig, min_dist, prom_levels)

    # Convert to frame numbers and keep only those strictly before BFC
    r_peaks = [p for p in r_peaks_all if int(ankle_frames[p]) < bfc_frame]
    l_peaks = [p for p in l_peaks_all if int(ankle_frames[p]) < bfc_frame]

    print(f"  Raw peaks before BFC — right: {len(r_peaks)}, left: {len(l_peaks)}")

    # Build raw event list before BFC: (frame, foot, ankle_y)
    step_events_raw = (
        [(int(ankle_frames[p]), "right", float(r_sig[p])) for p in r_peaks] +
        [(int(ankle_frames[p]), "left",  float(l_sig[p])) for p in l_peaks]
    )
    step_events_raw.sort(key=lambda x: x[0])

    # ── PASS 2: cross-ankle cluster deduplication ────────────────────────────
    def merge_contact_clusters(evts, gap):
        if not evts:
            return []
        merged = [evts[0]]
        for evt in evts[1:]:
            prev = merged[-1]
            if evt[0] - prev[0] <= gap:
                if evt[2] > prev[2]:        # larger Y = more grounded
                    merged[-1] = evt
            else:
                merged.append(evt)
        return [(f, foot) for f, foot, _ in merged]

    pre_bfc_events = merge_contact_clusters(step_events_raw, MIN_CLUSTER_GAP)
    print(f"  After cluster-merge (pre-BFC): {len(pre_bfc_events)} events")

    # ── PASS 3: enforce strict alternation working BACKWARDS from BFC ────────
    def opposite(foot):
        return "left" if foot == "right" else "right"

    required_feet = [
        opposite(bfc_foot),   # step 1
        bfc_foot,             # step 2
        opposite(bfc_foot),   # step 3
    ]

    def pick_last_before(candidates, max_frame_exclusive, required_foot):
        matched = [(f, ft) for f, ft in candidates
                   if ft == required_foot and f < max_frame_exclusive]
        return matched[-1] if matched else None

    step3 = pick_last_before(pre_bfc_events, bfc_frame,              required_feet[2])
    step2 = pick_last_before(pre_bfc_events, step3[0] if step3 else bfc_frame, required_feet[1])
    step1 = pick_last_before(pre_bfc_events, step2[0] if step2 else bfc_frame, required_feet[0])

    steps_1_to_3 = [s for s in [step1, step2, step3] if s is not None]

    if len(steps_1_to_3) < 3:
        print(f"  ⚠️  Only {len(steps_1_to_3)} alternating steps found before BFC — padding.")

        known_frames = [s[0] for s in steps_1_to_3] + [bfc_frame]
        if len(known_frames) >= 2:
            diffs = [known_frames[i+1] - known_frames[i]
                     for i in range(len(known_frames)-1)]
            est_interval = int(np.median(diffs))
        else:
            est_interval = max(6, int(0.20 * fps))

        full_steps = [None, None, None]
        for s in steps_1_to_3:
            for idx_r, req_foot in enumerate(required_feet):
                if s[1] == req_foot and full_steps[idx_r] is None:
                    full_steps[idx_r] = s
                    break

        ref_frame = bfc_frame
        for i in range(2, -1, -1):
            if full_steps[i] is None:
                next_frame = (full_steps[i+1][0] if i < 2 and full_steps[i+1] is not None
                              else ref_frame)
                synth_frame = max(1, next_frame - est_interval)
                full_steps[i] = (synth_frame, required_feet[i])
                print(f"     Synthesised step {i+1} @ frame {synth_frame} "
                      f"({required_feet[i]} foot, interval={est_interval}f)")
            ref_frame = full_steps[i][0]

        steps_1_to_3 = full_steps

    step4 = (bfc_frame, bfc_foot)
    step5 = (ffc_frame, ffc_foot)

    last5 = list(steps_1_to_3) + [step4, step5]

    last5_frames = [s[0] for s in last5]
    foot_labels  = [s[1] for s in last5]

    print(f"\n✅ Last 5 foot contacts (frames): {last5_frames}")
    print(f"   Feet: {[f.upper() for f in foot_labels]}")
    for i in range(1, 5):
        if foot_labels[i] == foot_labels[i-1]:
            print(f"   ⚠️  Alternation violation at step {i+1}: "
                  f"{foot_labels[i-1].upper()} → {foot_labels[i].upper()} (same foot!)")
    for i in range(1, 5):
        if last5_frames[i] <= last5_frames[i-1]:
            print(f"   ⚠️  Frame order issue at step {i+1}: "
                  f"frame {last5_frames[i]} <= {last5_frames[i-1]}")

    times        = [f / fps for f in last5_frames]
    duration     = times[-1] - times[0]
    intervals_s  = [(last5_frames[i] - last5_frames[i-1]) / fps for i in range(1, 5)]

    print(f"   Step 4 = BFC : frame {bfc_frame}  ({bfc_foot} foot)  [FIXED]")
    print(f"   Step 5 = FFC : frame {ffc_frame}  ({ffc_foot} foot)  [FIXED]")
    print(f"   Times (s): {[f'{t:.3f}' for t in times]}")
    print(f"   Duration (step 1→5): {duration:.3f} s")
    print(f"   Avg step interval:   {duration/4:.3f} s")

    # ── Video annotation ──────────────────────────────────────────────────────
    # FIX D: Use _open_and_advance() + cap.get() for frame-accurate numbering.
    out_vid    = out_path("step_cadence_annotated.mp4")
    vw, ow, oh = sized_writer(out_vid, fps, width, height)
    step_foot_map = {last5_frames[i]: foot_labels[i] for i in range(5)}
    model_local   = YOLO(MODEL_PATH)

    # Open video at the exact start frame using grab()-based seek (FIX D)
    cap = _open_and_advance(video_path, last5_frames[0])

    print(f"\n🎬 Writing step cadence video…")
    while True:
        ret, frame, frame_idx = _read_frame_with_number(cap)   # FIX D
        if not ret:
            break
        if frame_idx > release_frame:
            break

        results   = model_local.predict(frame, verbose=False)
        annotated = cv2.resize(frame, (ow, oh))
        if (results and results[0].keypoints is not None
                and len(results[0].keypoints.xy) > 0):
            kps = results[0].keypoints.xy[0].cpu().numpy()
            draw_skeleton_full(annotated, kps, scale=OUTPUT_SCALE)
            la = kps[KP_IDX["left_ankle"]]
            ra = kps[KP_IDX["right_ankle"]]
            if frame_idx in last5_frames:
                dom = step_foot_map[frame_idx].upper()
                lnd = None
                if   dom == "LEFT"  and la[1] > 0: lnd = la
                elif dom == "RIGHT" and ra[1] > 0: lnd = ra
                elif la[1] > 0 and ra[1] > 0:
                    lnd = la if la[1] > ra[1] else ra
                if lnd is not None:
                    cx     = int(lnd[0] * OUTPUT_SCALE)
                    cy     = int(lnd[1] * OUTPUT_SCALE)
                    phase  = (frame_idx % 30) / 30.0
                    r_anim = int((50 + 20 * np.sin(phase * 2 * np.pi)) * OUTPUT_SCALE)
                    ov2    = annotated.copy()
                    cv2.circle(ov2, (cx, cy), r_anim + int(15*OUTPUT_SCALE),
                               (0, 255, 255), -1)
                    cv2.circle(ov2, (cx, cy), r_anim, (0, 255, 0),
                               max(2, int(4*OUTPUT_SCALE)))
                    cv2.addWeighted(ov2, 0.55, annotated, 0.45, 0, annotated)

        step_num = None
        if frame_idx in last5_frames:
            step_num = last5_frames.index(frame_idx) + 1

        step_label = ""
        if step_num == 4:
            step_label = f">>> STEP 4/5  ({bfc_foot.upper()} FOOT) — BFC [FIXED] <<<"
        elif step_num == 5:
            step_label = f">>> STEP 5/5  ({ffc_foot.upper()} FOOT) — FFC [FIXED] <<<"
        elif step_num:
            step_label = (f">>> STEP {step_num}/5  "
                          f"({step_foot_map[frame_idx].upper()} FOOT) <<<")

        bar_y  = oh - int(30 * OUTPUT_SCALE)
        bar_x0 = int(20 * OUTPUT_SCALE)
        bar_x1 = int((ow - 20) * OUTPUT_SCALE)
        cv2.rectangle(annotated, (bar_x0, bar_y),
                      (bar_x1, bar_y + int(12*OUTPUT_SCALE)), (60, 60, 60), -1)
        if duration > 0:
            prog = min(1.0, (frame_idx - last5_frames[0]) /
                       max(1, release_frame - last5_frames[0]))
            cv2.rectangle(annotated, (bar_x0, bar_y),
                          (bar_x0 + int(prog*(bar_x1-bar_x0)),
                           bar_y + int(12*OUTPUT_SCALE)), (0, 200, 255), -1)

        info = ["MODULE: STEP CADENCE",
                f"Frame: {frame_idx}   Time: {frame_idx/fps:.2f}s",
                f"Duration (5 steps): {duration:.3f}s",
                f"Avg interval: {duration/4:.3f}s"]
        if step_label:
            info.insert(1, step_label)
            if step_num and step_num > 1:
                dt_step = (frame_idx - last5_frames[step_num-2]) / fps
                info.append(f"Since last step: {dt_step:.3f}s")
        if frame_idx == release_frame:
            info.insert(1, ">>> BALL RELEASE <<<")
        draw_info_box(annotated, info, x=10, y=10,
                      font_scale=0.55 * OUTPUT_SCALE / 0.5)
        repeats = 1
        if frame_idx in last5_frames: repeats = int(fps * 1.2)
        if frame_idx == release_frame: repeats = int(fps * 1.5)
        for _ in range(repeats):
            vw.write(annotated)

    cap.release(); vw.release()
    compress_video(out_vid)
    print(f"✅ Step cadence video: {out_vid}")

    return {
        "last5_frames":   last5_frames,
        "foot_labels":    foot_labels,
        "duration_s":     duration,
        "avg_interval_s": duration / 4,
        "intervals_s":    intervals_s,
    }

# =============================================================================
# ── SECTION 6 : MODULE 2 — DELIVERY STRIDE
# =============================================================================
def run_delivery_stride(df_xy, fps, events, meters_per_pixel, video_path, width, height):
    print("\n" + "─"*60)
    print("MODULE 2 — DELIVERY STRIDE")
    print("─"*60)

    release_frame = events["release_frame"]
    ffc_idx       = events["ffc_idx"]
    ffc_frame     = events["ffc_frame"]
    bfc_idx       = events["bfc_idx"]
    bfc_frame     = events["bfc_frame"]
    f_side        = events["f_side"]
    b_side        = events["b_side"]

    bx = float(df_xy.loc[bfc_idx, f"{b_side}_ankle_x"])
    by = float(df_xy.loc[bfc_idx, f"{b_side}_ankle_y"])
    fx = float(df_xy.loc[ffc_idx, f"{f_side}_ankle_x"])
    fy = float(df_xy.loc[ffc_idx, f"{f_side}_ankle_y"])

    stride_px  = float(np.hypot(fx - bx, fy - by))
    stride_m   = stride_px * meters_per_pixel
    duration_s = (ffc_frame - bfc_frame) / fps

    print(f"  BFC Frame:       {bfc_frame}")
    print(f"  FFC Frame:       {ffc_frame}")
    print(f"  Release Frame:   {release_frame}")
    print(f"  Stride Length:   {stride_m:.2f} m")
    print(f"  Stride Duration: {duration_s:.3f} s")

    out_vid    = out_path("delivery_stride_annotated.mp4")
    vw, ow, oh = sized_writer(out_vid, fps, width, height)
    # FIX D: Sequential read from frame 1 — no seeking, uses cap.get() for frame_no
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    bfc_drawn  = ffc_drawn = False

    print(f"\n🎬 Writing delivery stride video…")
    while True:
        ret, frame, frame_no = _read_frame_with_number(cap)   # FIX D
        if not ret:
            break
        ann = cv2.resize(frame, (ow, oh)); s = OUTPUT_SCALE
        if frame_no >= bfc_frame: bfc_drawn = True
        if frame_no >= ffc_frame: ffc_drawn = True
        if bfc_drawn:
            cv2.circle(ann, (int(bx*s), int(by*s)), int(14*s), (0, 0, 255), -1)
            cv2.circle(ann, (int(bx*s), int(by*s)), int(16*s), (255, 255, 255), 2)
            cv2.putText(ann, "BFC", (int(bx*s)+int(18*s), int(by*s)+int(6*s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7*s/0.5, (0, 0, 255), 2, cv2.LINE_AA)
        if ffc_drawn:
            cv2.circle(ann, (int(fx*s), int(fy*s)), int(14*s), (0, 255, 0), -1)
            cv2.circle(ann, (int(fx*s), int(fy*s)), int(16*s), (255, 255, 255), 2)
            cv2.putText(ann, "FFC", (int(fx*s)+int(18*s), int(fy*s)+int(6*s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7*s/0.5, (0, 255, 0), 2, cv2.LINE_AA)
        if bfc_drawn and ffc_drawn:
            cv2.line(ann, (int(bx*s), int(by*s)), (int(fx*s), int(fy*s)),
                     (255, 255, 0), max(2, int(3*s)))
            mid_x = int((bx+fx)/2*s); mid_y = int((by+fy)/2*s) - int(20*s)
            cv2.putText(ann, f"STRIDE: {stride_m:.2f} m", (mid_x-int(60*s), mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7*s/0.5, (255, 255, 0), 2, cv2.LINE_AA)
        event = ("BFC — Back Foot Contact"  if frame_no == bfc_frame else
                 "FFC — Front Foot Contact" if frame_no == ffc_frame else
                 "RELEASE"                  if frame_no == release_frame else "")
        info = ["MODULE: DELIVERY STRIDE",
                f"Frame: {frame_no}   Time: {frame_no/fps:.2f}s",
                f"BFC frame: {bfc_frame}  FFC frame: {ffc_frame}",
                f"Stride length: {stride_m:.2f} m",
                f"Stride duration: {duration_s:.3f} s"]
        if event: info.insert(1, f">>> {event} <<<")
        draw_info_box(ann, info, x=10, y=10,
                      font_scale=0.55 * OUTPUT_SCALE / 0.5)
        repeats = 1
        if frame_no in (bfc_frame, ffc_frame, release_frame):
            repeats = int(fps * 1.5)
        for _ in range(repeats): vw.write(ann)

    cap.release(); vw.release()
    compress_video(out_vid)
    print(f"✅ Delivery stride video: {out_vid}")

    return {
        "bfc_frame":         bfc_frame,
        "ffc_frame":         ffc_frame,
        "stride_m":          stride_m,
        "stride_duration_s": duration_s,
        "f_side":            f_side,
        "b_side":            b_side,
    }

# =============================================================================
# ── SECTION 7 : MODULE 3 — ELBOW FLEXION
# =============================================================================
def run_elbow_flexion(df_xy, fps, events, bowling_wrist, video_path, width, height):
    print("\n" + "─"*60)
    print("MODULE 3 — ELBOW FLEXION")
    print("─"*60)

    release_frame  = events["release_frame"]
    arm_back_frame = events["arm_back_frame"]

    arm          = "right" if bowling_wrist == "right_wrist" else "left"
    shoulder_col = f"{arm}_shoulder"
    elbow_col    = f"{arm}_elbow"
    wrist_col    = f"{arm}_wrist"
    arm_label    = arm.upper()

    angles, times, frames = [], [], []
    for idx in range(len(df_xy)):
        s_pt = (float(df_xy.loc[idx, f"{shoulder_col}_x"]),
                float(df_xy.loc[idx, f"{shoulder_col}_y"]))
        e_pt = (float(df_xy.loc[idx, f"{elbow_col}_x"]),
                float(df_xy.loc[idx, f"{elbow_col}_y"]))
        w_pt = (float(df_xy.loc[idx, f"{wrist_col}_x"]),
                float(df_xy.loc[idx, f"{wrist_col}_y"]))
        if all(np.isfinite(v) for pt in (s_pt, e_pt, w_pt) for v in pt):
            ang = angle_deg(s_pt, e_pt, w_pt)
            if np.isfinite(ang):
                angles.append(ang)
                times.append(float(df_xy.loc[idx, "time_s"]))
                frames.append(int(df_xy.loc[idx, "frame"]))

    if len(angles) == 0:
        print("⚠️  No valid elbow angles detected.")
        return None

    vel = [0.0]
    for i in range(1, len(angles)):
        dt_i = times[i] - times[i - 1]
        vel.append((angles[i] - angles[i-1]) / dt_i if dt_i > 0 else 0.0)

    df_elbow = pd.DataFrame({"frame": frames, "time_s": times,
                              "elbow_angle": angles, "angular_velocity": vel})
    csv_path = out_path("elbow_full_analysis.csv")
    df_elbow.to_csv(csv_path, index=False)

    frame_to_angle = {int(r["frame"]): float(r["elbow_angle"])
                      for _, r in df_elbow.iterrows()}

    elbow_at_release  = frame_to_angle.get(release_frame,  np.nan)
    elbow_at_arm_back = frame_to_angle.get(arm_back_frame, np.nan)
    if np.isfinite(elbow_at_release) and np.isfinite(elbow_at_arm_back):
        elbow_extension = elbow_at_release - elbow_at_arm_back
    else:
        elbow_extension = np.nan

    if np.isfinite(elbow_at_arm_back):
        print(f"  Elbow @ arm-back (frame {arm_back_frame}): {elbow_at_arm_back:.1f}°")
    else:
        print(f"  Elbow @ arm-back: N/A")
    if np.isfinite(elbow_at_release):
        print(f"  Elbow @ release  (frame {release_frame}): {elbow_at_release:.1f}°")
    else:
        print(f"  Elbow @ release: N/A")
    if np.isfinite(elbow_extension):
        print(f"  Elbow extension: {elbow_extension:.1f}°")

    rel_t      = release_frame  / fps
    arm_back_t = arm_back_frame / fps

    png_path = out_path("elbow_full_plots.png")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(times, angles, linewidth=1.5, color="deepskyblue")
    axes[0].axvline(x=arm_back_t, linestyle="--", color="orange",
                    label=f"Arm-back (f{arm_back_frame})")
    axes[0].axvline(x=rel_t, linestyle="--", color="red",
                    label=f"Release (f{release_frame})")
    axes[0].set_ylabel("Elbow Angle (deg)")
    axes[0].set_title(f"{arm_label} ARM — Elbow Angle (Full Clip)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(times, vel, linewidth=1.5, color="orange")
    axes[1].axvline(x=arm_back_t, linestyle="--", color="orange")
    axes[1].axvline(x=rel_t, linestyle="--", color="red")
    axes[1].axhline(y=0, alpha=0.3)
    axes[1].set_ylabel("Angular Velocity (deg/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title(f"{arm_label} ARM — Elbow Angular Velocity (Full Clip)")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(png_path, dpi=200); plt.close()
    print(f"📊 Elbow CSV:  {csv_path}")
    print(f"📈 Elbow plot: {png_path}")

    out_vid    = out_path("elbow_flexion_annotated.mp4")
    vw, ow, oh = sized_writer(out_vid, fps, width, height)
    # FIX D: Sequential read from frame 1, cap.get() for frame number
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

    def elbow_color(ang):
        t = max(0.0, min(1.0, (180.0 - ang) / 120.0))
        return (0, int(255*(1-t)), int(255*t))

    print(f"\n🎬 Writing elbow flexion video…")
    while True:
        ret, frame, frame_no = _read_frame_with_number(cap)   # FIX D
        if not ret:
            break
        idx = frame_no - 1   # 0-based df_xy row index — consistent with extract_keypoints
        ann = cv2.resize(frame, (ow, oh)); s = OUTPUT_SCALE
        sh = (float(df_xy.loc[idx, f"{shoulder_col}_x"]),
              float(df_xy.loc[idx, f"{shoulder_col}_y"]))
        el = (float(df_xy.loc[idx, f"{elbow_col}_x"]),
              float(df_xy.loc[idx, f"{elbow_col}_y"]))
        wr = (float(df_xy.loc[idx, f"{wrist_col}_x"]),
              float(df_xy.loc[idx, f"{wrist_col}_y"]))
        ang = frame_to_angle.get(frame_no, np.nan)
        if all(np.isfinite(v) for pt in (sh, el, wr) for v in pt):
            col  = elbow_color(ang) if np.isfinite(ang) else (200, 200, 200)
            sh_d = (int(sh[0]*s), int(sh[1]*s))
            el_d = (int(el[0]*s), int(el[1]*s))
            wr_d = (int(wr[0]*s), int(wr[1]*s))
            cv2.line(ann, sh_d, el_d, col, max(3, int(5*s)), cv2.LINE_AA)
            cv2.line(ann, el_d, wr_d, col, max(3, int(5*s)), cv2.LINE_AA)
            cv2.circle(ann, sh_d, int(9*s), (255,255,255), -1, cv2.LINE_AA)
            cv2.circle(ann, sh_d, int(7*s), (255,100,0),   -1, cv2.LINE_AA)
            cv2.circle(ann, el_d, int(11*s),(255,255,255),  -1, cv2.LINE_AA)
            cv2.circle(ann, el_d, int(9*s), col,            -1, cv2.LINE_AA)
            cv2.circle(ann, wr_d, int(9*s), (255,255,255),  -1, cv2.LINE_AA)
            cv2.circle(ann, wr_d, int(7*s), (0,200,255),    -1, cv2.LINE_AA)
            cv2.putText(ann, "S", (sh_d[0]+int(8*s), sh_d[1]-int(8*s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5*s/0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(ann, "E", (el_d[0]+int(8*s), el_d[1]-int(8*s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5*s/0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(ann, "W", (wr_d[0]+int(8*s), wr_d[1]-int(8*s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5*s/0.5, (255,255,255), 1, cv2.LINE_AA)
            if np.isfinite(ang):
                draw_angle_arc(ann, el_d, sh_d, wr_d, ang, color=col, radius=int(45*s))
        if np.isfinite(ang):
            gauge_x  = ow - int(40*s)
            gauge_y0 = int(80*s); gauge_y1 = oh - int(80*s); gauge_h = gauge_y1 - gauge_y0
            cv2.rectangle(ann, (gauge_x,gauge_y0), (gauge_x+int(18*s),gauge_y1), (50,50,50), -1)
            fill = int(gauge_h * max(0, min(1, ang/180.0)))
            cv2.rectangle(ann, (gauge_x,gauge_y1-fill), (gauge_x+int(18*s),gauge_y1), elbow_color(ang), -1)
            cv2.rectangle(ann, (gauge_x,gauge_y0), (gauge_x+int(18*s),gauge_y1), (200,200,200), 1)
            cv2.putText(ann, "180", (gauge_x-int(38*s),gauge_y0+int(8*s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45*s/0.5, (200,200,200), 1)
            cv2.putText(ann, "0", (gauge_x-int(18*s),gauge_y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45*s/0.5, (200,200,200), 1)
        event = (">>> ARM-BACK <<<" if frame_no == arm_back_frame else
                 ">>> BALL RELEASE <<<" if frame_no == release_frame else "")
        info  = [f"MODULE: {arm_label} ARM ELBOW FLEXION",
                 f"Frame: {frame_no}   Time: {frame_no/fps:.2f}s",
                 f"Elbow angle: {ang:.1f} deg" if np.isfinite(ang) else "Elbow angle: --",
                 "(180=straight  <180=flexed)"]
        if np.isfinite(elbow_extension):
            info.append(f"Extension: {elbow_extension:.1f} deg")
        if event: info.insert(1, event)
        draw_info_box(ann, info, x=10, y=10, font_scale=0.55 * OUTPUT_SCALE / 0.5)
        vw.write(ann)

    cap.release(); vw.release()
    compress_video(out_vid)
    print(f"✅ Elbow flexion video: {out_vid}")

    return {
        "df_elbow":          df_elbow,
        "release_frame":     release_frame,
        "arm":               arm,
        "frame_to_angle":    frame_to_angle,
        "elbow_at_release":  elbow_at_release,
        "elbow_at_arm_back": elbow_at_arm_back,
        "elbow_extension":   elbow_extension,
    }

# =============================================================================
# ── SECTION 8 : MODULE 4 — KNEE FLEXION
# =============================================================================
class KneeLockTracker:
    def __init__(self, side, fps, max_jump_leglen=0.60, conf_th=0.25):
        self.side = side; self.fps = fps
        self.conf_th = conf_th; self.max_jump = max_jump_leglen
        self.prev_knee = self.prev_leglen = None
        if side == "L": self.iH, self.iK, self.iA = 11, 13, 15
        else:           self.iH, self.iK, self.iA = 12, 14, 16

    def update(self, kp_xy, kp_conf):
        if kp_xy is None: return False, None, None, None, np.nan
        H  = kp_xy[self.iH]; K  = kp_xy[self.iK]; A  = kp_xy[self.iA]
        cH = kp_conf[self.iH]; cK = kp_conf[self.iK]; cA = kp_conf[self.iA]
        if cH < self.conf_th or cK < self.conf_th or cA < self.conf_th:
            return False, None, None, None, np.nan
        leglen = float(np.linalg.norm(H-K) + np.linalg.norm(K-A))
        if self.prev_knee is not None and self.prev_leglen is not None:
            fps_scale = max(0.7, min(2.0, 30.0/max(self.fps,1e-6)))
            if float(np.linalg.norm(K-self.prev_knee)) > self.max_jump*self.prev_leglen*fps_scale:
                return False, None, None, None, np.nan
        ang = angle_deg(tuple(H), tuple(K), tuple(A))
        self.prev_knee = K.copy(); self.prev_leglen = leglen
        return True, H, K, A, ang


def run_knee_flexion(df_xy, df_conf, fps, events, knee_side, video_path, width, height):
    print("\n" + "─"*60)
    print("MODULE 4 — KNEE FLEXION")
    print("─"*60)

    release_frame = events["release_frame"]
    ffc_frame     = events["ffc_frame"]

    side_label = "LEFT (front)" if knee_side == "L" else "RIGHT (front)"
    kside      = "left"         if knee_side == "L" else "right"
    print(f"  Tracking: {side_label} knee  (shared FFC={ffc_frame}, release={release_frame})")

    tracker = KneeLockTracker(knee_side, fps, KNEE_MAX_JUMP_LEGLEN, KNEE_CONF_TH)
    rows = []; ankle_y_list = []; wrist_xy_list = []; last_good = None
    iW = KP_IDX["left_wrist"] if knee_side == "L" else KP_IDX["right_wrist"]

    for i in range(len(df_xy)):
        kp_xy   = np.array([[df_xy.loc[i, f"{name}_x"],
                             df_xy.loc[i, f"{name}_y"]]
                            for name in KEYPOINT_NAMES], dtype=float)
        kp_conf = np.array([df_conf.loc[i, f"{name}_conf"]
                            for name in KEYPOINT_NAMES], dtype=float)
        ok, H, K, A, ang = tracker.update(kp_xy, kp_conf)
        if ok:
            last_good = (H, K, A, ang)
            ankle_y_list.append(float(A[1]))
            wrist_xy_list.append((float(kp_xy[iW,0]), float(kp_xy[iW,1])))
        else:
            if last_good:
                _, _, A_lg, ang = last_good
                ankle_y_list.append(float(A_lg[1]))
            else:
                ankle_y_list.append(np.nan); ang = np.nan
            wrist_xy_list.append(wrist_xy_list[-1] if wrist_xy_list else (0.0,0.0))
        rows.append({"frame": int(df_xy.loc[i,"frame"]),
                     "time_s": float(df_xy.loc[i,"time_s"]),
                     "knee_angle_deg": ang})

    df_knee = pd.DataFrame(rows)
    frame_to_knee_angle = {int(r["frame"]): float(r["knee_angle_deg"])
                           for _, r in df_knee.iterrows()}

    df_knee["is_ffc"]     = df_knee["frame"] == ffc_frame
    df_knee["is_release"] = df_knee["frame"] == release_frame

    csv_path = out_path("knee_angles.csv"); df_knee.to_csv(csv_path, index=False)

    ffc_df_idx = df_knee[df_knee["frame"] == ffc_frame].index
    rel_df_idx = df_knee[df_knee["frame"] == release_frame].index
    if len(ffc_df_idx) > 0 and len(rel_df_idx) > 0:
        f0 = ffc_df_idx[0]; f1 = rel_df_idx[0]
        if f1 > f0:
            df_knee.loc[f0:f1, ["frame","time_s","knee_angle_deg"]].to_csv(
                out_path("knee_angles_ffc_to_release.csv"), index=False)

    ffc_t_row = df_knee[df_knee["frame"] == ffc_frame]
    rel_t_row = df_knee[df_knee["frame"] == release_frame]
    ffc_t_val = float(ffc_t_row["time_s"].values[0]) if len(ffc_t_row) > 0 else None
    rel_t_val = float(rel_t_row["time_s"].values[0]) if len(rel_t_row) > 0 else None

    png_path = out_path("knee_flexion_analysis.png")
    plt.figure(figsize=(10,4))
    plt.plot(df_knee["time_s"], df_knee["knee_angle_deg"],
             linewidth=1.5, color="limegreen")
    if ffc_t_val is not None:
        plt.axvline(ffc_t_val, linestyle="--", color="blue", label=f"FFC (f{ffc_frame})")
    if rel_t_val is not None:
        plt.axvline(rel_t_val, linestyle=":", color="red", label=f"Release (f{release_frame})")
    plt.xlabel("Time (s)"); plt.ylabel("Knee Angle (deg)")
    plt.title(f"{side_label} Knee Flexion — Full Clip")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(png_path, dpi=200); plt.close()
    print(f"📊 Knee CSV:  {csv_path}")
    print(f"📈 Knee plot: {png_path}")

    frame_to_knee_angle = {int(r["frame"]): float(r["knee_angle_deg"])
                           for _, r in df_knee.iterrows()}

    out_vid    = out_path("knee_flexion_annotated.mp4")
    vw, ow, oh = sized_writer(out_vid, fps, width, height)
    # FIX D: Sequential read from frame 1, cap.get() for frame number
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    tracker2   = KneeLockTracker(knee_side, fps, KNEE_MAX_JUMP_LEGLEN, KNEE_CONF_TH)
    last_good2 = None

    def knee_color(ang):
        t = max(0.0, min(1.0, (180.0-ang)/120.0))
        return (0, int(255*(1-t)), int(255*t))

    print(f"\n🎬 Writing knee flexion video…")
    while True:
        ret, frame, frame_no = _read_frame_with_number(cap)   # FIX D
        if not ret:
            break
        idx = frame_no - 1   # 0-based df_xy row index
        if idx >= len(df_xy):
            break
        ann = cv2.resize(frame, (ow, oh)); s = OUTPUT_SCALE
        kp_xy   = np.array([[df_xy.loc[idx, f"{name}_x"],
                             df_xy.loc[idx, f"{name}_y"]]
                            for name in KEYPOINT_NAMES], dtype=float)
        kp_conf = np.array([df_conf.loc[idx, f"{name}_conf"]
                            for name in KEYPOINT_NAMES], dtype=float)
        ok2, H2, K2, A2, ang2 = tracker2.update(kp_xy, kp_conf)
        if ok2: last_good2 = (H2, K2, A2, ang2)
        ang_vid = frame_to_knee_angle.get(frame_no, np.nan)
        if last_good2 is not None:
            H3, K3, A3, _ = last_good2
            col = knee_color(ang_vid) if np.isfinite(ang_vid) else (200,200,200)
            H_d=(int(H3[0]*s),int(H3[1]*s)); K_d=(int(K3[0]*s),int(K3[1]*s)); A_d=(int(A3[0]*s),int(A3[1]*s))
            cv2.line(ann,H_d,K_d,col,max(3,int(5*s)),cv2.LINE_AA); cv2.line(ann,K_d,A_d,col,max(3,int(5*s)),cv2.LINE_AA)
            cv2.circle(ann,H_d,int(9*s),(255,255,255),-1,cv2.LINE_AA); cv2.circle(ann,H_d,int(7*s),(200,0,255),-1,cv2.LINE_AA)
            cv2.circle(ann,K_d,int(11*s),(255,255,255),-1,cv2.LINE_AA); cv2.circle(ann,K_d,int(9*s),col,-1,cv2.LINE_AA)
            cv2.circle(ann,A_d,int(9*s),(255,255,255),-1,cv2.LINE_AA); cv2.circle(ann,A_d,int(7*s),(100,255,0),-1,cv2.LINE_AA)
            cv2.putText(ann,"HIP",(H_d[0]+int(8*s),H_d[1]-int(8*s)),cv2.FONT_HERSHEY_SIMPLEX,0.45*s/0.5,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(ann,"KNEE",(K_d[0]+int(8*s),K_d[1]-int(8*s)),cv2.FONT_HERSHEY_SIMPLEX,0.45*s/0.5,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(ann,"ANKLE",(A_d[0]+int(8*s),A_d[1]-int(8*s)),cv2.FONT_HERSHEY_SIMPLEX,0.45*s/0.5,(255,255,255),1,cv2.LINE_AA)
            if np.isfinite(ang_vid):
                draw_angle_arc(ann,K_d,H_d,A_d,ang_vid,color=col,radius=int(50*s))
        if np.isfinite(ang_vid):
            gauge_x=ow-int(40*s); gauge_y0=int(80*s); gauge_y1=oh-int(80*s); gauge_h=gauge_y1-gauge_y0
            col_g=knee_color(ang_vid)
            cv2.rectangle(ann,(gauge_x,gauge_y0),(gauge_x+int(18*s),gauge_y1),(50,50,50),-1)
            fill=int(gauge_h*max(0,min(1,ang_vid/180.0)))
            cv2.rectangle(ann,(gauge_x,gauge_y1-fill),(gauge_x+int(18*s),gauge_y1),col_g,-1)
            cv2.rectangle(ann,(gauge_x,gauge_y0),(gauge_x+int(18*s),gauge_y1),(200,200,200),1)
            cv2.putText(ann,"180",(gauge_x-int(38*s),gauge_y0+int(8*s)),cv2.FONT_HERSHEY_SIMPLEX,0.4*s/0.5,(200,200,200),1)
            cv2.putText(ann,"0",(gauge_x-int(18*s),gauge_y1),cv2.FONT_HERSHEY_SIMPLEX,0.4*s/0.5,(200,200,200),1)
        is_ffc = frame_no == ffc_frame
        is_rel = frame_no == release_frame
        info = [f"MODULE: {side_label.upper()} KNEE FLEXION",
                f"Frame: {frame_no}   Time: {frame_no/fps:.2f}s",
                f"Knee angle: {ang_vid:.1f} deg" if np.isfinite(ang_vid) else "Knee angle: --",
                "(180=straight  <180=flexed)"]
        if is_ffc: info.insert(1, ">>> FRONT FOOT CONTACT (FFC) <<<")
        if is_rel: info.insert(1, ">>> BALL RELEASE <<<")
        draw_info_box(ann,info,x=10,y=10,font_scale=0.55*OUTPUT_SCALE/0.5)
        repeats = 1
        if is_ffc or is_rel: repeats = int(fps*1.5)
        for _ in range(repeats): vw.write(ann)

    cap.release(); vw.release()
    compress_video(out_vid)
    print(f"✅ Knee flexion video: {out_vid}")

    return {
        "df_knee":             df_knee,
        "frame_to_knee_angle": frame_to_knee_angle,
        "kside":               kside,
        "knee_angle_at_ffc":     frame_to_knee_angle.get(ffc_frame,     np.nan),
        "knee_angle_at_release": frame_to_knee_angle.get(release_frame, np.nan),
    }

# =============================================================================
# ── SECTION 9 : MODULE 5 — HEAD / COM POSITION
# =============================================================================
def safe_head_xy(df_xy, idx):
    xs, ys = [], []
    for p in FACE_POINTS:
        x = df_xy.loc[idx, f"{p}_x"]; y = df_xy.loc[idx, f"{p}_y"]
        if np.isfinite(x) and np.isfinite(y): xs.append(x); ys.append(y)
    if not xs: return np.nan, np.nan
    return float(np.mean(xs)), float(np.mean(ys))


def run_head_position(df_xy, fps, events, bowling_wrist,
                      cm_per_pixel, pixels_per_meter, video_path, width, height):
    print("\n" + "─"*60)
    print("MODULE 5 — HEAD / COM POSITION")
    print("─"*60)

    release_frame = events["release_frame"]
    release_idx   = events["release_idx"]
    ffc_frame     = events["ffc_frame"]
    ffc_idx       = events["ffc_idx"]
    bfc_frame     = events["bfc_frame"]
    bfc_idx       = events["bfc_idx"]
    f_side        = events["f_side"]

    front_ankle = f"{f_side}_ankle"

    head_x = [safe_head_xy(df_xy,i)[0] for i in range(len(df_xy))]
    head_y = [safe_head_xy(df_xy,i)[1] for i in range(len(df_xy))]
    df_xy["head_x"] = head_x; df_xy["head_y"] = head_y
    df_xy["hip_mid_x"] = (df_xy["left_hip_x"].values + df_xy["right_hip_x"].values)/2
    df_xy["hip_mid_y"] = (df_xy["left_hip_y"].values + df_xy["right_hip_y"].values)/2

    for side in ("left", "right"):
        _, sig = detect_ankle_peaks(df_xy, fps, side, release_idx)
        df_xy[f"{side}_ankle_y_sm"] = sig

    print(f"  Front ankle : {front_ankle}  (from shared events, view={VIEW_MODE})")
    print(f"  FFC frame   : {ffc_frame}")
    print(f"  BFC frame   : {bfc_frame}")

    df_xy["head_x_sm"]           = smooth(np.array(head_x, float))
    df_xy[f"{front_ankle}_x_sm"] = smooth(df_xy[f"{front_ankle}_x"].values.astype(float))
    df_xy["videoDx_px"]          = df_xy["head_x_sm"] - df_xy[f"{front_ankle}_x_sm"]
    df_xy["videoDx_cm"]          = df_xy["videoDx_px"] * cm_per_pixel

    seg = df_xy.iloc[ffc_idx:release_idx+1].copy()
    seg[["frame","time_s","videoDx_px","videoDx_cm"]].to_csv(
        out_path("headDx_vs_time.csv"), index=False)

    plt.figure(figsize=(10,4))
    plt.plot(seg["time_s"], seg["videoDx_cm"], linewidth=1.5, color="gold")
    plt.axhline(0, linestyle="--", color="gray", alpha=0.5)
    plt.xlabel("Time (s)"); plt.ylabel("Head Dx (cm)")
    plt.title("Head Horizontal Offset relative to Front Foot (FFC → Release)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_path("headDx_vs_time.png"), dpi=200); plt.close()

    def offsets_at(i, fa):
        hx2 = float(df_xy.loc[i,"head_x"]); hy2 = float(df_xy.loc[i,"head_y"])
        fx2 = float(df_xy.loc[i,f"{fa}_x"]); fy2 = float(df_xy.loc[i,f"{fa}_y"])
        dx, dy = hx2-fx2, hy2-fy2; d = float(np.hypot(dx,dy))
        return {"Dx_cm": dx*cm_per_pixel, "Dy_cm": dy*cm_per_pixel, "D_cm": d*cm_per_pixel}

    ffc_off = offsets_at(ffc_idx, front_ankle)
    bfc_off = offsets_at(bfc_idx, front_ankle)

    print(f"\n  Head @ FFC — Dx:{ffc_off['Dx_cm']:.1f}cm  "
          f"Dy:{ffc_off['Dy_cm']:.1f}cm  D:{ffc_off['D_cm']:.1f}cm")
    print(f"  Head @ BFC — Dx:{bfc_off['Dx_cm']:.1f}cm  "
          f"Dy:{bfc_off['Dy_cm']:.1f}cm  D:{bfc_off['D_cm']:.1f}cm")

    metrics = {"view_mode": VIEW_MODE, "front_ankle": front_ankle,
               "ffc_frame": ffc_frame, "bfc_frame": bfc_frame,
               "head_offset_at_FFC": ffc_off, "head_offset_at_BFC": bfc_off}
    with open(out_path("head_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([{"ffc_frame": ffc_frame, "bfc_frame": bfc_frame, **ffc_off}]).to_csv(
        out_path("head_metrics.csv"), index=False)
    print(f"📌 Head metrics: {out_path('head_metrics.json')}")
    print(f"📈 Head Dx plot: {out_path('headDx_vs_time.png')}")

    out_vid    = out_path("head_position_annotated.mp4")
    vw, ow, oh = sized_writer(out_vid, fps, width, height)
    # FIX D: Use _open_and_advance() with grab() for frame-accurate seek
    cap = _open_and_advance(video_path, bfc_frame)

    print(f"\n🎬 Writing head position video…")
    while True:
        ret, frame, frame_no = _read_frame_with_number(cap)   # FIX D
        if not ret:
            break
        if frame_no > release_frame:
            break
        idx2 = frame_no - 1   # 0-based df_xy row index
        ann = cv2.resize(frame, (ow, oh)); s = OUTPUT_SCALE
        hx2 = float(df_xy.loc[idx2,"head_x"]) if idx2<len(df_xy) else np.nan
        hy2 = float(df_xy.loc[idx2,"head_y"]) if idx2<len(df_xy) else np.nan
        fx2 = float(df_xy.loc[idx2,f"{front_ankle}_x"]) if idx2<len(df_xy) else np.nan
        fy2 = float(df_xy.loc[idx2,f"{front_ankle}_y"]) if idx2<len(df_xy) else np.nan
        dx_cm2 = dy_cm2 = np.nan
        if all(np.isfinite(v) for v in (hx2,hy2,fx2,fy2)):
            dx_cm2=(hx2-fx2)*cm_per_pixel; dy_cm2=(hy2-fy2)*cm_per_pixel
            H_d=(int(hx2*s),int(hy2*s)); F_d=(int(fx2*s),int(fy2*s))
            cv2.circle(ann,F_d,int(14*s),(0,0,255),-1,cv2.LINE_AA)
            cv2.circle(ann,F_d,int(16*s),(255,255,255),2,cv2.LINE_AA)
            cv2.putText(ann,"FRONT FOOT",(F_d[0]+int(18*s),F_d[1]+int(6*s)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5*s/0.5,(0,150,255),2,cv2.LINE_AA)
            cv2.circle(ann,H_d,int(12*s),(0,255,0),-1,cv2.LINE_AA)
            cv2.circle(ann,H_d,int(14*s),(255,255,255),2,cv2.LINE_AA)
            cv2.putText(ann,"HEAD",(H_d[0]+int(14*s),H_d[1]-int(10*s)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5*s/0.5,(0,255,0),2,cv2.LINE_AA)
            intensity=min(1.0,abs(dx_cm2)/30.0)
            lc=(0,int(255*(1-intensity)),int(255*intensity))
            cv2.line(ann,H_d,F_d,lc,max(2,int(3*s)),cv2.LINE_AA)
            cv2.arrowedLine(ann,F_d,(H_d[0],F_d[1]),(255,255,0),max(2,int(3*s)),cv2.LINE_AA,tipLength=0.05)
            mid_dx=((F_d[0]+H_d[0])//2,F_d[1]-int(15*s))
            cv2.putText(ann,f"Dx={dx_cm2:.1f}cm",mid_dx,cv2.FONT_HERSHEY_SIMPLEX,0.55*s/0.5,(255,255,0),2,cv2.LINE_AA)
        event_label=("BFC" if frame_no==bfc_frame else "FFC" if frame_no==ffc_frame else
                     "RELEASE" if frame_no==release_frame else "")
        info=["MODULE: HEAD / COM POSITION",
              f"Frame: {frame_no}   Time: {frame_no/fps:.2f}s",
              f"Head Dx: {dx_cm2:.1f} cm" if np.isfinite(dx_cm2) else "Head Dx: --",
              f"Head Dy: {dy_cm2:.1f} cm" if np.isfinite(dy_cm2) else "Head Dy: --",
              "Green=Head  Red=Front foot"]
        if event_label: info.insert(1, f">>> {event_label} <<<")
        draw_info_box(ann,info,x=10,y=10,font_scale=0.55*OUTPUT_SCALE/0.5)
        repeats=6
        if frame_no in [bfc_frame, ffc_frame, release_frame]:
            repeats += int(fps*2.0)
        for _ in range(repeats): vw.write(ann)

    cap.release(); vw.release()
    compress_video(out_vid)
    print(f"✅ Head position video: {out_vid}")

    return {
        "ffc_frame":   ffc_frame,
        "bfc_frame":   bfc_frame,
        "ffc_idx":     ffc_idx,
        "front_ankle": front_ankle,
        "ffc_offset":  ffc_off,
        "bfc_offset":  bfc_off,
    }

# =============================================================================
# ── SECTION 10 : MODULE 6 — WRIST VELOCITY
# =============================================================================
def stabilize_lr_non_bowling(df: pd.DataFrame, bowling_arm: str) -> pd.DataFrame:
    all_pairs = [
        ("left_wrist","right_wrist"), ("left_elbow","right_elbow"),
        ("left_shoulder","right_shoulder"), ("left_hip","right_hip"),
        ("left_knee","right_knee"), ("left_ankle","right_ankle"),
    ]
    bowling_joints = {f"{bowling_arm}_wrist", f"{bowling_arm}_elbow", f"{bowling_arm}_shoulder"}
    pairs = [(L, R) for L, R in all_pairs
             if L not in bowling_joints and R not in bowling_joints
             and f"{L}_x" in df.columns and f"{R}_x" in df.columns]
    if not pairs or len(df) < 2:
        return df

    def cost(prev, cur, swapped):
        total = 0.0
        for L, R in pairs:
            Lc = np.array([cur[f"{R if swapped else L}_x"], cur[f"{R if swapped else L}_y"]], float)
            Rc = np.array([cur[f"{L if swapped else R}_x"], cur[f"{L if swapped else R}_y"]], float)
            Lp = np.array([prev[f"{L}_x"], prev[f"{L}_y"]], float)
            Rp = np.array([prev[f"{R}_x"], prev[f"{R}_y"]], float)
            if np.all(np.isfinite(Lc)) and np.all(np.isfinite(Lp)): total += float(np.linalg.norm(Lc-Lp))
            if np.all(np.isfinite(Rc)) and np.all(np.isfinite(Rp)): total += float(np.linalg.norm(Rc-Rp))
        return total

    def swap_row(row):
        row = row.copy()
        for L, R in pairs:
            row[f"{L}_x"], row[f"{R}_x"] = row[f"{R}_x"], row[f"{L}_x"]
            row[f"{L}_y"], row[f"{R}_y"] = row[f"{R}_y"], row[f"{L}_y"]
        return row

    rows = df.to_dict("records"); out = [rows[0]]
    for i in range(1, len(rows)):
        ck = cost(out[-1], rows[i], False); cs = cost(out[-1], rows[i], True)
        out.append(swap_row(rows[i]) if cs < ck else rows[i])
    return pd.DataFrame(out)


def run_wrist_velocity(df_xy, fps, events, bowling_wrist,
                       meters_per_pixel, video_path, width, height):
    print("\n" + "─"*60)
    print("MODULE 6 — WRIST VELOCITY")
    print("─"*60)

    release_frame = events["release_frame"]
    release_idx   = events["release_idx"]

    arm        = "right" if bowling_wrist == "right_wrist" else "left"
    elbow_name = f"{arm}_elbow"
    wrist_name = f"{arm}_wrist"
    dt         = 1.0 / fps

    print(f"  Bowling arm  : {arm.upper()}")
    print(f"  Wrist column : {wrist_name}")

    stabilize_lr_non_bowling(df_xy.copy(), bowling_arm=arm)

    wx = smooth(df_xy[f"{arm}_wrist_x"].values.astype(float))
    wy = smooth(df_xy[f"{arm}_wrist_y"].values.astype(float))
    ex = smooth(df_xy[f"{arm}_elbow_x"].values.astype(float))
    ey = smooth(df_xy[f"{arm}_elbow_y"].values.astype(float))

    df_xy["wrist_x_sm"] = wx; df_xy["wrist_y_sm"] = wy
    df_xy["elbow_x_sm"] = ex; df_xy["elbow_y_sm"] = ey

    vx_m   = central_diff(wx, dt) * meters_per_pixel
    vy_m   = central_diff(wy, dt) * meters_per_pixel
    spd    = np.sqrt(vx_m**2 + vy_m**2)
    spd_sm = smooth(spd)

    df_xy["wrist_vx_mps"]       = vx_m
    df_xy["wrist_vy_mps"]       = vy_m
    df_xy["wrist_speed_mps_sm"] = spd_sm
    df_xy["wrist_speed_kmh_sm"] = spd_sm * 3.6

    theta    = np.arctan2(wy-ey, wx-ex)
    theta_u  = np.unwrap(theta)
    omega    = central_diff(theta_u, dt)
    omega_sm = smooth(omega)

    df_xy["forearm_angle_rad"]         = theta_u
    df_xy["wrist_angular_vel_rads_sm"] = omega_sm

    rel_frame2 = release_frame

    pk_idx   = int(np.nanargmax(spd_sm))
    pk_frame = int(df_xy.loc[pk_idx, "frame"])
    pk_speed = float(spd_sm[pk_idx])

    win_frames = int(round(0.12*fps))
    a, b       = max(0,release_idx-win_frames), min(len(df_xy)-1,release_idx+win_frames)
    near_idx   = a + int(np.nanargmax(spd_sm[a:b+1]))
    near_frame = int(df_xy.loc[near_idx, "frame"])
    near_speed = float(spd_sm[near_idx])

    near_om_idx   = a + int(np.nanargmax(np.abs(omega_sm[a:b+1])))
    near_om_frame = int(df_xy.loc[near_om_idx, "frame"])
    near_om_peak  = float(omega_sm[near_om_idx])

    spd_at_rel   = float(spd_sm[release_idx])
    spd_at_rel_k = spd_at_rel * 3.6

    print(f"  Release frame:                   {rel_frame2}")
    print(f"  Wrist speed @ release:           {spd_at_rel:.2f} m/s  ({spd_at_rel_k:.1f} km/h)")
    print(f"  Peak wrist speed (overall):      {pk_speed:.2f} m/s  @ frame {pk_frame}")
    print(f"  Peak wrist speed (near release): {near_speed:.2f} m/s  @ frame {near_frame}")
    print(f"  Peak angular velocity (near):    {near_om_peak:.2f} rad/s @ frame {near_om_frame}")

    ball_proxy  = {"mps": spd_at_rel, "kmh": spd_at_rel_k}
    ball_mode_b = None
    if BALL_MODEL_PATH is not None:
        ball_model_b = YOLO(BALL_MODEL_PATH)
        cap_b = _open_and_advance(video_path, max(1, rel_frame2))
        pts_b = []
        for _ in range(BALL_TRACK_FRAMES):
            ok_b, frm_b, _ = _read_frame_with_number(cap_b)
            if not ok_b: break
            res_b = ball_model_b.predict(frm_b, verbose=False)
            if res_b and res_b[0].boxes is not None and len(res_b[0].boxes)>0:
                confs_b = res_b[0].boxes.conf.cpu().numpy()
                xyxy_b  = res_b[0].boxes.xyxy.cpu().numpy()
                j = int(np.argmax(confs_b)); x1_b,y1_b,x2_b,y2_b=xyxy_b[j]
                pts_b.append(((x1_b+x2_b)/2,(y1_b+y2_b)/2))
            else:
                pts_b.append((np.nan,np.nan))
        cap_b.release()
        pts_b = np.array(pts_b,float)
        if np.any(np.isfinite(pts_b)):
            bx_s=smooth(pts_b[:,0]); by_s=smooth(pts_b[:,1])
            bvx=central_diff(bx_s,dt)*meters_per_pixel
            bvy=central_diff(by_s,dt)*meters_per_pixel
            bspd=np.sqrt(bvx**2+bvy**2); bspd_v=bspd[np.isfinite(bspd)]
            if len(bspd_v):
                ball_mode_b={"avg_kmh":float(np.nanmean(bspd_v)*3.6),
                             "peak_kmh":float(np.nanmax(bspd_v)*3.6)}

    ts_cols = ["frame","time_s","wrist_x_sm","wrist_y_sm","elbow_x_sm","elbow_y_sm",
               "wrist_vx_mps","wrist_vy_mps","wrist_speed_mps_sm","wrist_speed_kmh_sm",
               "forearm_angle_rad","wrist_angular_vel_rads_sm"]
    df_xy[ts_cols].to_csv(out_path("wrist_timeseries.csv"), index=False)

    metrics_w = {
        "bowling_arm": arm.upper(), "release_frame": rel_frame2,
        "release_method": events.get("release_method", "unknown"),
        "wrist_speed_at_release_mps": spd_at_rel,
        "wrist_speed_at_release_kmh": spd_at_rel_k,
        "peak_wrist_speed_overall_mps": pk_speed,
        "peak_wrist_speed_near_release_mps": near_speed,
        "peak_angular_vel_near_release_rads": near_om_peak,
        "ball_speed_proxy": ball_proxy, "ball_speed_mode_b": ball_mode_b,
    }
    with open(out_path("wrist_ball_metrics.json"), "w") as f:
        json.dump(metrics_w, f, indent=2)

    rel_t = rel_frame2 / fps
    plt.figure(figsize=(10,4))
    plt.plot(df_xy["time_s"], df_xy["wrist_speed_kmh_sm"], linewidth=1.5, color="cyan",
             label=f"{arm.upper()} wrist speed")
    plt.axvline(x=rel_t, linestyle="--", color="red", label="Release")
    plt.axvline(x=near_frame/fps, linestyle=":", color="orange", label="Peak (near release)")
    plt.xlabel("Time (s)"); plt.ylabel("Speed (km/h)")
    plt.title(f"{arm.upper()} WRIST Speed vs Time")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_path("wrist_speed_vs_time.png"), dpi=200); plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(df_xy["time_s"], omega_sm, linewidth=1.5, color="magenta",
             label=f"{arm.upper()} forearm angular vel")
    plt.axvline(x=rel_t, linestyle="--", color="red", label="Release")
    plt.xlabel("Time (s)"); plt.ylabel("Angular velocity (rad/s)")
    plt.title(f"{arm.upper()} FOREARM Angular Velocity vs Time")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_path("wrist_omega_vs_time.png"), dpi=200); plt.close()

    out_vid       = out_path("wrist_velocity_annotated.mp4")
    vw, ow, oh    = sized_writer(out_vid, fps, width, height)
    # FIX D: Sequential read from frame 1, cap.get() for frame number
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    slow_mo = 4
    max_speed_kmh = max(float(np.nanmax(df_xy["wrist_speed_kmh_sm"])), 1.0)

    print(f"\n🎬 Writing wrist velocity video  ({arm.upper()} arm)…")
    while True:
        ret, frame, frame_no = _read_frame_with_number(cap)   # FIX D
        if not ret:
            break
        idx = frame_no - 1   # 0-based df_xy row index
        if idx >= len(df_xy):
            break
        ann = cv2.resize(frame, (ow, oh)); s = OUTPUT_SCALE
        wxi=float(df_xy.loc[idx,"wrist_x_sm"]); wyi=float(df_xy.loc[idx,"wrist_y_sm"])
        exi=float(df_xy.loc[idx,"elbow_x_sm"]); eyi=float(df_xy.loc[idx,"elbow_y_sm"])
        spd_i=float(df_xy.loc[idx,"wrist_speed_mps_sm"]); spd_k=spd_i*3.6
        omega_i=float(df_xy.loc[idx,"wrist_angular_vel_rads_sm"])
        vxi=float(df_xy.loc[idx,"wrist_vx_mps"]); vyi=float(df_xy.loc[idx,"wrist_vy_mps"])
        W_d=(int(wxi*s),int(wyi*s)); E_d=(int(exi*s),int(eyi*s))
        t_spd=min(1.0,spd_k/max_speed_kmh)
        arm_col=(0,int(255*(1-t_spd)),int(255*t_spd))
        cv2.line(ann,E_d,W_d,arm_col,max(3,int(5*s)),cv2.LINE_AA)
        cv2.circle(ann,E_d,int(9*s),(255,255,255),-1,cv2.LINE_AA); cv2.circle(ann,E_d,int(7*s),(255,100,0),-1,cv2.LINE_AA)
        cv2.circle(ann,W_d,int(9*s),(255,255,255),-1,cv2.LINE_AA); cv2.circle(ann,W_d,int(7*s),arm_col,-1,cv2.LINE_AA)
        cv2.putText(ann,"ELBOW",(E_d[0]+int(8*s),E_d[1]-int(8*s)),cv2.FONT_HERSHEY_SIMPLEX,0.45*s/0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(ann,"WRIST",(W_d[0]+int(8*s),W_d[1]-int(8*s)),cv2.FONT_HERSHEY_SIMPLEX,0.45*s/0.5,(255,255,255),1,cv2.LINE_AA)
        arrow_scale=0.08
        end_x=int(wxi*s+vxi/meters_per_pixel*arrow_scale*s)
        end_y=int(wyi*s+vyi/meters_per_pixel*arrow_scale*s)
        cv2.arrowedLine(ann,W_d,(end_x,end_y),(0,255,255),max(2,int(3*s)),cv2.LINE_AA,tipLength=0.15)
        gauge_x=ow-int(40*s); gauge_y0=int(60*s); gauge_y1=oh-int(60*s); gauge_h=gauge_y1-gauge_y0
        cv2.rectangle(ann,(gauge_x,gauge_y0),(gauge_x+int(18*s),gauge_y1),(50,50,50),-1)
        fill_g=int(gauge_h*t_spd)
        cv2.rectangle(ann,(gauge_x,gauge_y1-fill_g),(gauge_x+int(18*s),gauge_y1),arm_col,-1)
        cv2.rectangle(ann,(gauge_x,gauge_y0),(gauge_x+int(18*s),gauge_y1),(200,200,200),1)
        cv2.putText(ann,f"{max_speed_kmh:.0f}",(gauge_x-int(35*s),gauge_y0+int(8*s)),cv2.FONT_HERSHEY_SIMPLEX,0.4*s/0.5,(200,200,200),1)
        cv2.putText(ann,"0",(gauge_x-int(15*s),gauge_y1),cv2.FONT_HERSHEY_SIMPLEX,0.4*s/0.5,(200,200,200),1)
        cv2.putText(ann,"km/h",(gauge_x-int(5*s),gauge_y1+int(18*s)),cv2.FONT_HERSHEY_SIMPLEX,0.4*s/0.5,(200,200,200),1)
        event=(">>> BALL RELEASE <<<"     if frame_no==rel_frame2 else
               ">>> PEAK WRIST SPEED <<<" if frame_no==near_frame  else "")
        info=[f"MODULE: {arm.upper()} WRIST VELOCITY",
              f"Frame: {frame_no}   Time: {frame_no/fps:.2f}s",
              f"Wrist speed: {spd_i:.2f} m/s  ({spd_k:.1f} km/h)",
              f"Angular vel: {omega_i:.2f} rad/s",
              f"Ball proxy: {spd_at_rel_k:.1f} km/h"]
        if event: info.insert(1, event)
        draw_info_box(ann,info,x=10,y=10,font_scale=0.55*OUTPUT_SCALE/0.5)
        for _ in range(slow_mo): vw.write(ann)

    cap.release(); vw.release()
    compress_video(out_vid)
    print(f"✅ Wrist velocity video: {out_vid}")

    return {
        "release_frame":               rel_frame2,
        "speed_at_release_mps":        spd_at_rel,
        "peak_speed_near_release_mps": near_speed,
        "peak_speed_overall_mps":      pk_speed,
        "peak_angular_vel_rads":       near_om_peak,
        "ball_proxy_kmh":              spd_at_rel_k,
        "arm":                         arm,
    }

# =============================================================================
# ── SECTION 11 : SUMMARY PRINT
# =============================================================================
def print_summary(hand, events, r_stride, r_cadence, r_elbow, r_knee, r_head, r_wrist, fps):
    print("\n" + "="*60)
    print("  FULL PIPELINE SUMMARY")
    print("="*60)
    arm_label  = "RIGHT" if hand == "R" else "LEFT"
    knee_label = "LEFT (front)" if hand == "R" else "RIGHT (front)"
    print(f"  Bowling arm    : {arm_label}")
    print(f"  Front knee     : {knee_label}")
    print(f"  Release method : {events.get('release_method','?')}")
    print(f"  BFC frame      : {events['bfc_frame']}  ({events['bfc_frame']/fps:.2f}s)")
    print(f"  FFC frame      : {events['ffc_frame']}  ({events['ffc_frame']/fps:.2f}s)")
    print(f"  Arm-back frame : {events['arm_back_frame']}  ({events['arm_back_frame']/fps:.2f}s)")
    print(f"  Release frame  : {events['release_frame']}  ({events['release_frame']/fps:.2f}s)")
    if r_cadence:
        print(f"\n📍 Step Cadence")
        print(f"   Last 5 frames : {r_cadence['last5_frames']}")
        print(f"   Duration 1→5  : {r_cadence['duration_s']:.3f}s   Avg: {r_cadence['avg_interval_s']:.3f}s")
    if r_stride:
        print(f"\n📍 Delivery Stride")
        print(f"   Length: {r_stride['stride_m']:.2f}m   Duration: {r_stride['stride_duration_s']:.3f}s")
    if r_elbow:
        print(f"\n📍 Elbow Flexion")
        if np.isfinite(r_elbow.get("elbow_at_arm_back", np.nan)):
            print(f"   Angle @ arm-back: {r_elbow['elbow_at_arm_back']:.1f}°")
        if np.isfinite(r_elbow.get("elbow_at_release", np.nan)):
            print(f"   Angle @ release : {r_elbow['elbow_at_release']:.1f}°")
        if np.isfinite(r_elbow.get("elbow_extension", np.nan)):
            print(f"   Extension       : {r_elbow['elbow_extension']:.1f}°")
    if r_knee:
        print(f"\n📍 Knee Flexion")
        k_ffc = r_knee.get("knee_angle_at_ffc", np.nan)
        k_rel = r_knee.get("knee_angle_at_release", np.nan)
        if np.isfinite(k_ffc): print(f"   Angle @ FFC    : {k_ffc:.1f}°")
        if np.isfinite(k_rel): print(f"   Angle @ release: {k_rel:.1f}°")
    if r_head:
        o = r_head["ffc_offset"]
        print(f"\n📍 Head / COM Position  (front ankle: {r_head['front_ankle']})")
        print(f"   @ FFC — Dx:{o['Dx_cm']:.1f}cm  Dy:{o['Dy_cm']:.1f}cm  D:{o['D_cm']:.1f}cm")
        b = r_head["bfc_offset"]
        print(f"   @ BFC — Dx:{b['Dx_cm']:.1f}cm  Dy:{b['Dy_cm']:.1f}cm  D:{b['D_cm']:.1f}cm")
    if r_wrist:
        print(f"\n📍 Wrist Velocity  ({r_wrist['arm'].upper()} arm)")
        print(f"   Speed @ release : {r_wrist['speed_at_release_mps']:.2f} m/s  "
              f"({r_wrist['ball_proxy_kmh']:.1f} km/h)")
        print(f"   Peak near release: {r_wrist['peak_speed_near_release_mps']:.2f} m/s")
        print(f"   Peak angular vel : {r_wrist['peak_angular_vel_rads']:.2f} rad/s")

# =============================================================================
# ── SECTION 12 : DATASET HELPERS
# =============================================================================
def _safe_read_excel(path, expected_cols):
    if not os.path.exists(path):
        return pd.DataFrame(columns=expected_cols)
    try:
        df = pd.read_excel(path, dtype=str)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        return df
    except Exception as e:
        print(f"  ⚠️  Could not read {path}: {e} — starting fresh.")
        return pd.DataFrame(columns=expected_cols)

def _safe_write_excel(df, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        df.to_excel(path, index=False, engine="openpyxl")
        print(f"  💾  Saved → {path}  ({len(df)} rows)")
    except PermissionError:
        fallback = path.replace(".xlsx",
            f"_backup_{datetime.datetime.now().strftime('%H%M%S')}.xlsx")
        df.to_excel(fallback, index=False, engine="openpyxl")
        print(f"  ⚠️  {path} is open in Excel — saved to fallback: {fallback}")

def _drop_existing_trial(df, trial_id):
    if "trial_id" in df.columns:
        return df[df["trial_id"].astype(str) != str(trial_id)].reset_index(drop=True)
    return df

# =============================================================================
# ── SECTION 13 : FRAME DATASET WRITER
# =============================================================================
FRAME_COLS = [
    "trial_id", "view_mode", "frame",
    "left_ankle_y", "right_ankle_y", "left_ankle_y_sm", "right_ankle_y_sm",
    "bowling_shoulder_x", "bowling_shoulder_y",
    "bowling_elbow_x", "bowling_elbow_y",
    "bowling_wrist_x", "bowling_wrist_y",
    "elbow_angle_deg",
    "front_hip_x", "front_hip_y",
    "front_knee_x", "front_knee_y",
    "front_ankle_x", "front_ankle_y",
    "front_knee_angle_deg",
    "head_x", "head_y",
    "head_to_front_dx_cm", "head_to_front_dy_cm", "head_to_front_d_cm",
    "wrist_speed_m_s", "forearm_angle_deg", "wrist_angular_velocity_deg_s",
]

def build_frame_dataset(trial_id, df_xy, fps, cm_per_pixel,
                        bowling_arm, kside, front_ankle,
                        frame_to_angle, frame_to_knee_angle):
    n = len(df_xy)
    for col in ("left_ankle_y_sm","right_ankle_y_sm","head_x","head_y","videoDx_cm"):
        if col not in df_xy.columns:
            df_xy[col] = np.nan
    rows = []
    for i in range(n):
        frame_no  = int(df_xy.loc[i,"frame"])
        elbow_ang = frame_to_angle.get(frame_no, np.nan)      if frame_to_angle      else np.nan
        knee_ang  = frame_to_knee_angle.get(frame_no, np.nan) if frame_to_knee_angle else np.nan
        hx  = float(df_xy.loc[i,"head_x"])
        hy  = float(df_xy.loc[i,"head_y"])
        fax = float(df_xy.loc[i,f"{front_ankle}_x"])
        fay = float(df_xy.loc[i,f"{front_ankle}_y"])
        if all(np.isfinite(v) for v in (hx,hy,fax,fay)):
            raw_dx  = float(df_xy.loc[i,"videoDx_cm"])
            h_dx_cm = raw_dx if np.isfinite(raw_dx) else (hx-fax)*cm_per_pixel
            h_dy_cm = (fay - hy) * cm_per_pixel   # positive = head above ankle
            h_d_cm  = float(np.hypot(h_dx_cm,h_dy_cm))
        else:
            h_dx_cm = h_dy_cm = h_d_cm = np.nan
        fa_rad  = float(df_xy.loc[i,"forearm_angle_rad"]) if "forearm_angle_rad" in df_xy.columns else np.nan
        fa_deg  = np.degrees(fa_rad) if np.isfinite(fa_rad) else np.nan
        av_rads = float(df_xy.loc[i,"wrist_angular_vel_rads_sm"]) if "wrist_angular_vel_rads_sm" in df_xy.columns else np.nan
        av_degs = np.degrees(av_rads) if np.isfinite(av_rads) else np.nan
        bw_x = float(df_xy.loc[i,"wrist_x_sm"]) if "wrist_x_sm" in df_xy.columns else float(df_xy.loc[i,f"{bowling_arm}_wrist_x"])
        bw_y = float(df_xy.loc[i,"wrist_y_sm"]) if "wrist_y_sm" in df_xy.columns else float(df_xy.loc[i,f"{bowling_arm}_wrist_y"])
        rows.append({
            "trial_id": trial_id, "view_mode": VIEW_MODE, "frame": frame_no,
            "left_ankle_y":   float(df_xy.loc[i,"left_ankle_y"]),
            "right_ankle_y":  float(df_xy.loc[i,"right_ankle_y"]),
            "left_ankle_y_sm":  float(df_xy.loc[i,"left_ankle_y_sm"]),
            "right_ankle_y_sm": float(df_xy.loc[i,"right_ankle_y_sm"]),
            "bowling_shoulder_x": float(df_xy.loc[i,f"{bowling_arm}_shoulder_x"]),
            "bowling_shoulder_y": float(df_xy.loc[i,f"{bowling_arm}_shoulder_y"]),
            "bowling_elbow_x":    float(df_xy.loc[i,f"{bowling_arm}_elbow_x"]),
            "bowling_elbow_y":    float(df_xy.loc[i,f"{bowling_arm}_elbow_y"]),
            "bowling_wrist_x": bw_x, "bowling_wrist_y": bw_y,
            "elbow_angle_deg": elbow_ang,
            "front_hip_x":   float(df_xy.loc[i,f"{kside}_hip_x"]),
            "front_hip_y":   float(df_xy.loc[i,f"{kside}_hip_y"]),
            "front_knee_x":  float(df_xy.loc[i,f"{kside}_knee_x"]),
            "front_knee_y":  float(df_xy.loc[i,f"{kside}_knee_y"]),
            "front_ankle_x": fax, "front_ankle_y": fay,
            "front_knee_angle_deg": knee_ang,
            "head_x": hx, "head_y": hy,
            "head_to_front_dx_cm": h_dx_cm,
            "head_to_front_dy_cm": h_dy_cm,
            "head_to_front_d_cm":  h_d_cm,
            "wrist_speed_m_s": float(df_xy.loc[i,"wrist_speed_mps_sm"]) if "wrist_speed_mps_sm" in df_xy.columns else np.nan,
            "forearm_angle_deg": fa_deg,
            "wrist_angular_velocity_deg_s": av_degs,
        })
    return pd.DataFrame(rows, columns=FRAME_COLS)

def write_frame_dataset(df_new_frames, trial_id, path):
    print(f"\n📋 Updating frame dataset → {path}")
    df_existing = _safe_read_excel(path, FRAME_COLS)
    df_existing = _drop_existing_trial(df_existing, trial_id)
    df_combined = pd.concat([df_existing, df_new_frames], ignore_index=True)
    _safe_write_excel(df_combined, path)

# =============================================================================
# ── SECTION 14 : MASTER DATASET WRITER
# =============================================================================
MASTER_COLS = [
    "trial_id", "fps", "bowling_arm", "view_mode", "release_frame",
    "release_method",
    "last5_steps_frame", "last5_step_intervals_s",
    "step_duration_mean_s", "step_duration_std_s", "step_duration_cv",
    "final5_total_duration_s",
    "stride_duration_s", "stride_length_m", "bfc_frame", "ffc_frame",
    "arm_back_frame", "release_frame.1",
    "elbow_angle_arm_back_deg", "elbow_angle_release_deg", "elbow_extension_deg",
    "knee_angle_ffc_deg", "knee_angle_release_deg",
    "head_dx_ffc_cm", "head_dy_ffc_cm", "head_d_ffc_cm",
    "head_dx_bfc_cm", "head_dy_bfc_cm", "head_d_bfc_cm",
    "peak_wrist_speed_m_s", "wrist_speed_at_release_m_s",
]

def build_master_row(trial_id, fps, hand, events,
                     r_cadence, r_stride, r_elbow, r_knee, r_head, r_wrist):
    release_frame  = events["release_frame"]
    arm_back_frame = events["arm_back_frame"]
    bfc_frame      = events["bfc_frame"]
    ffc_frame      = events["ffc_frame"]

    if r_cadence:
        intervals = r_cadence["intervals_s"]
        last5_str = str(r_cadence["last5_frames"])
        intv_str  = str(intervals)
        mean_i    = float(np.mean(intervals))
        std_i     = float(np.std(intervals, ddof=0))
        cv_i      = std_i / mean_i if mean_i > 0 else np.nan
        total_dur = r_cadence["duration_s"]
    else:
        last5_str = intv_str = ""
        mean_i = std_i = cv_i = total_dur = np.nan

    stride_dur = r_stride["stride_duration_s"] if r_stride else np.nan
    stride_len = r_stride["stride_m"]          if r_stride else np.nan

    elbow_at_arm_back = r_elbow.get("elbow_at_arm_back", np.nan) if r_elbow else np.nan
    elbow_at_release  = r_elbow.get("elbow_at_release",  np.nan) if r_elbow else np.nan
    elbow_extension   = r_elbow.get("elbow_extension",   np.nan) if r_elbow else np.nan

    knee_at_ffc     = r_knee.get("knee_angle_at_ffc",     np.nan) if r_knee else np.nan
    knee_at_release = r_knee.get("knee_angle_at_release", np.nan) if r_knee else np.nan

    hd_ffc = r_head["ffc_offset"] if r_head else None
    hd_bfc = r_head["bfc_offset"] if r_head else None

    return {
        "trial_id":                   trial_id,
        "fps":                        fps,
        "bowling_arm":                hand,
        "view_mode":                  VIEW_MODE,
        "release_frame":              release_frame,
        "release_method":             events.get("release_method", "unknown"),
        "last5_steps_frame":          last5_str,
        "last5_step_intervals_s":     intv_str,
        "step_duration_mean_s":       mean_i,
        "step_duration_std_s":        std_i,
        "step_duration_cv":           cv_i,
        "final5_total_duration_s":    total_dur,
        "stride_duration_s":          stride_dur,
        "stride_length_m":            stride_len,
        "bfc_frame":                  bfc_frame,
        "ffc_frame":                  ffc_frame,
        "arm_back_frame":             arm_back_frame,
        "release_frame.1":            release_frame,
        "elbow_angle_arm_back_deg":   elbow_at_arm_back,
        "elbow_angle_release_deg":    elbow_at_release,
        "elbow_extension_deg":        elbow_extension,
        "knee_angle_ffc_deg":         knee_at_ffc,
        "knee_angle_release_deg":     knee_at_release,
        "head_dx_ffc_cm":             hd_ffc["Dx_cm"] if hd_ffc else np.nan,
        "head_dy_ffc_cm":             hd_ffc["Dy_cm"] if hd_ffc else np.nan,
        "head_d_ffc_cm":              hd_ffc["D_cm"]  if hd_ffc else np.nan,
        "head_dx_bfc_cm":             hd_bfc["Dx_cm"] if hd_bfc else np.nan,
        "head_dy_bfc_cm":             hd_bfc["Dy_cm"] if hd_bfc else np.nan,
        "head_d_bfc_cm":              hd_bfc["D_cm"]  if hd_bfc else np.nan,
        "peak_wrist_speed_m_s":       r_wrist["peak_speed_near_release_mps"] if r_wrist else np.nan,
        "wrist_speed_at_release_m_s": r_wrist["speed_at_release_mps"]        if r_wrist else np.nan,
    }

def write_master_dataset(master_row, trial_id, path):
    print(f"\n📋 Updating master dataset → {path}")
    df_existing = _safe_read_excel(path, MASTER_COLS)
    df_existing = _drop_existing_trial(df_existing, trial_id)
    df_new      = pd.DataFrame([master_row], columns=MASTER_COLS)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    _safe_write_excel(df_combined, path)

# =============================================================================
# ── MAIN
# =============================================================================
def main():
    global OUT_DIR

    hand, knee_side  = get_user_inputs()
    bowling_wrist    = "right_wrist" if hand == "R" else "left_wrist"
    bowling_arm_name = "right"       if hand == "R" else "left"
    print(f"🎯 Bowling wrist: {bowling_wrist}")

    OUT_DIR  = setup_output_dir(VIDEO_PATH, BASE_OUT_DIR)
    trial_id = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    print(f"📁 Output folder : {OUT_DIR}")
    print(f"🆔 Trial ID      : {trial_id}")

    pixels_per_meter, meters_per_pixel, cm_per_pixel = stump_calibration(VIDEO_PATH)

    print(f"\n⚙️  Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    df_xy, df_conf, fps, width, height, total_frames = extract_keypoints(VIDEO_PATH, model)

    # ── Compute ALL shared events ONCE ───────────────────────────────────────
    events = detect_shared_events(df_xy, fps, bowling_wrist, video_path=VIDEO_PATH)
    print(f"\n🏏 Release frame: {events['release_frame']}  "
          f"({events['release_frame']/fps:.2f}s)  [{events['release_method']}]")

    r_cadence = run_step_cadence(df_xy, fps, events, VIDEO_PATH, width, height)
    r_stride  = run_delivery_stride(df_xy, fps, events, meters_per_pixel, VIDEO_PATH, width, height)
    r_elbow   = run_elbow_flexion(df_xy, fps, events, bowling_wrist, VIDEO_PATH, width, height)
    r_knee    = run_knee_flexion(df_xy, df_conf, fps, events, knee_side, VIDEO_PATH, width, height)
    r_head    = run_head_position(df_xy, fps, events, bowling_wrist, cm_per_pixel, pixels_per_meter, VIDEO_PATH, width, height)
    r_wrist   = run_wrist_velocity(df_xy, fps, events, bowling_wrist, meters_per_pixel, VIDEO_PATH, width, height)

    print_summary(hand, events, r_stride, r_cadence, r_elbow, r_knee, r_head, r_wrist, fps)

    kside = r_knee["kside"] if r_knee and "kside" in r_knee else (
        "left" if knee_side == "L" else "right")
    front_ankle = (r_head["front_ankle"] if r_head and "front_ankle" in r_head
                   else f"{events['f_side']}_ankle")
    frame_to_angle      = r_elbow["frame_to_angle"]      if r_elbow else None
    frame_to_knee_angle = r_knee["frame_to_knee_angle"]  if r_knee  else None

    print("\n" + "─"*60)
    print("WRITING DATASETS")
    print("─"*60)

    df_frames = build_frame_dataset(
        trial_id=trial_id, df_xy=df_xy, fps=fps, cm_per_pixel=cm_per_pixel,
        bowling_arm=bowling_arm_name, kside=kside, front_ankle=front_ankle,
        frame_to_angle=frame_to_angle, frame_to_knee_angle=frame_to_knee_angle,
    )
    write_frame_dataset(df_frames, trial_id, FRAMES_DATASET_PATH)

    master_row = build_master_row(
        trial_id=trial_id, fps=fps, hand=hand, events=events,
        r_cadence=r_cadence, r_stride=r_stride, r_elbow=r_elbow,
        r_knee=r_knee, r_head=r_head, r_wrist=r_wrist,
    )
    write_master_dataset(master_row, trial_id, MASTER_DATASET_PATH)

    print("\n✅ Both datasets updated successfully.")
    print(f"   Frame dataset : {os.path.abspath(FRAMES_DATASET_PATH)}")
    print(f"   Master dataset: {os.path.abspath(MASTER_DATASET_PATH)}")


if __name__ == "__main__":
    main()