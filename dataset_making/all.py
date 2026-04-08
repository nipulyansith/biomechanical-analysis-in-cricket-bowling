"""
FAST BOWLING — UNIFIED BIOMECHANICS PIPELINE + DATASET WRITER
==============================================================
Drop-in replacement for the original pipeline.
After every run it automatically updates two Excel datasets:

  FRAMES_DATASET_PATH  → one row per frame  (frame-level signals)
  MASTER_DATASET_PATH  → one row per video  (delivery-level summary)

Duplicate-handling strategy
  • Both files are keyed on `trial_id`  (= video filename stem, e.g. "geenod").
  • On re-processing the same video the old rows are deleted and replaced.
  • New videos are simply appended.

Column mapping is documented in write_frame_dataset() and write_master_dataset().
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
VIDEO_PATH   = r"C:\Users\nipul\OneDrive\Desktop\tm\videos\B-04_T-04.MOV"
MODEL_PATH   = "yolov8l-pose.pt"
BASE_OUT_DIR = "output"

# ── Dataset paths (edit these to point to your shared Excel files) ──────────
FRAMES_DATASET_PATH = "frames.xlsx"
MASTER_DATASET_PATH = "master.xlsx"

STUMP_HEIGHT_M  = 0.711
SMOOTH_WINDOW   = 9
SMOOTH_POLY     = 2
OUTPUT_SCALE    = 0.5

VIEW_MODE            = "SIDE"
RELEASE_DETECT_MODE  = "WRIST_HIGHEST"
BALL_MODEL_PATH      = None
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
def compress_video(input_path: str, crf: int = 28, scale: float = 1.0) -> str:
    if not os.path.exists(input_path):
        return input_path
    stem    = os.path.splitext(input_path)[0]
    out_vid = stem + "_c.mp4"
    vf = f"scale=iw*{scale}:ih*{scale}" if scale < 1.0 else "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    cmd = ["ffmpeg", "-y", "-i", input_path, "-vcodec", "libx264",
           "-crf", str(crf), "-preset", "fast", "-vf", vf, "-an", out_vid]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(input_path)
        os.rename(out_vid, input_path)
        orig_mb = os.path.getsize(input_path) / 1e6
        print(f"  🗜️  Compressed → {os.path.basename(input_path)}  ({orig_mb:.1f} MB)")
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
# ── SECTION 2 : STUMP CALIBRATION
# =============================================================================
def stump_calibration(video_path, stump_height_m=0.711):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read first frame for calibration.")
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
        cv2.putText(canvas, "Click TOP and BOTTOM of stump | SPACE=confirm | R=reset | ESC=cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, f"Points selected: {len(clicks_display)}/2",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for i, (cx, cy) in enumerate(clicks_display):
            label = "TOP" if i == 0 else "BOTTOM"
            cv2.circle(canvas, (cx, cy), 8, (0, 255, 0), -1)
            cv2.circle(canvas, (cx, cy), 10, (255, 255, 255), 2)
            cv2.putText(canvas, label, (cx + 12, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps          = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        if (results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0):
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

def detect_release_frame(df_xy, bowling_wrist):
    wy  = df_xy[f"{bowling_wrist}_y"].values.astype(float)
    wys = smooth(wy)
    idx = int(np.nanargmin(wys))
    return idx, int(df_xy.loc[idx, "frame"]), wys

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
# ── SHARED VIDEO ANNOTATION HELPERS
# =============================================================================
def draw_info_box(img, lines, x=10, y=10, font_scale=0.6,
                  thickness=2, bg_alpha=0.6, color=(255, 255, 255)):
    font   = cv2.FONT_HERSHEY_SIMPLEX
    pad    = 8
    line_h = int((font_scale * 30) + 8)
    max_w  = max(cv2.getTextSize(l, font, font_scale, thickness)[0][0] for l in lines) if lines else 100
    box_h  = line_h * len(lines) + pad * 2
    box_w  = max_w + pad * 2
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
    for i, line in enumerate(lines):
        ty = y + pad + (i + 1) * line_h - 4
        cv2.putText(img, line, (x + pad, ty), font, font_scale, color, thickness, cv2.LINE_AA)

def draw_angle_arc(img, vertex, p1, p2, angle, color=(0, 255, 255), radius=40, thickness=2):
    v    = np.array(vertex, float)
    a1   = np.array(p1, float) - v
    a2   = np.array(p2, float) - v
    ang1 = float(np.degrees(np.arctan2(a1[1], a1[0])))
    ang2 = float(np.degrees(np.arctan2(a2[1], a2[0])))
    if ang1 > ang2: ang1, ang2 = ang2, ang1
    if ang2 - ang1 > 180: ang1, ang2 = ang2, ang1 + 360
    cv2.ellipse(img, (int(v[0]), int(v[1])), (radius, radius), 0, ang1, ang2, color, thickness, cv2.LINE_AA)
    mid_ang = np.radians((ang1 + ang2) / 2)
    lx = int(v[0] + (radius + 20) * np.cos(mid_ang))
    ly = int(v[1] + (radius + 20) * np.sin(mid_ang))
    cv2.putText(img, f"{angle:.1f}", (lx - 20, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

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
            cv2.line(img, kp_dict[p1], kp_dict[p2], bone, max(1, int(3 * scale)), cv2.LINE_AA)
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
# ── SECTION 5 : MODULE 1 — STEP CADENCE
# =============================================================================
def run_step_cadence(df_xy, fps, release_idx, release_frame, bowling_wrist, video_path, width, height):
    print("\n" + "─"*60)
    print("MODULE 1 — STEP CADENCE")
    print("─"*60)
    min_dist = max(6, int(0.20 * fps))
    r_sig    = smooth(df_xy["right_ankle_y"].values.astype(float))
    l_sig    = smooth(df_xy["left_ankle_y"].values.astype(float))
    r_peaks, _ = find_peaks(r_sig, distance=min_dist, prominence=30)
    l_peaks, _ = find_peaks(l_sig, distance=min_dist, prominence=30)
    ankle_frames = df_xy["frame"].values
    step_events  = (
        [(int(ankle_frames[p]), "right") for p in r_peaks] +
        [(int(ankle_frames[p]), "left")  for p in l_peaks]
    )
    step_events.sort(key=lambda x: x[0])
    step_events = [s for s in step_events if s[0] <= release_frame]
    print("\n📋 Foot-contact events before release:")
    for s in step_events:
        print(f"   Frame {s[0]:>4d} — {s[1]} foot")

    def pick_alternating_last5(events):
        selected, last_foot = [], None
        for evt in reversed(events):
            if evt[1] != last_foot:
                selected.append(evt)
                last_foot = evt[1]
                if len(selected) == 5:
                    break
        return list(reversed(selected))

    last5 = pick_alternating_last5(step_events)
    if len(last5) < 5:
        print(f"\n⚠️  Only {len(last5)} alternating steps found before release.")
        return None

    last5_frames = [s[0] for s in last5]
    foot_labels  = [s[1] for s in last5]
    times        = [f / fps for f in last5_frames]
    duration     = times[-1] - times[0]
    print(f"\n✅ Last 5 foot contacts (frames): {last5_frames}")
    print(f"   Feet: {[f.upper() for f in foot_labels]}")
    print(f"   Times (s): {[f'{t:.3f}' for t in times]}")
    print(f"   Duration (step 1→5): {duration:.3f} s")
    print(f"   Avg step interval:   {duration/4:.3f} s")

    out_vid    = out_path("step_cadence_annotated.mp4")
    vw, ow, oh = sized_writer(out_vid, fps, width, height)
    step_foot_map = {last5_frames[i]: foot_labels[i] for i in range(5)}
    model_local   = YOLO(MODEL_PATH)
    cap       = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, last5_frames[0] - 1)
    frame_idx = last5_frames[0]
    print(f"\n🎬 Writing step cadence video…")
    while frame_idx <= release_frame:
        ret, frame = cap.read()
        if not ret:
            break
        results   = model_local.predict(frame, verbose=False)
        annotated = cv2.resize(frame, (ow, oh))
        if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
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
                    cv2.circle(ov2, (cx, cy), r_anim + int(15*OUTPUT_SCALE), (0,255,255), -1)
                    cv2.circle(ov2, (cx, cy), r_anim, (0,255,0), max(2,int(4*OUTPUT_SCALE)))
                    cv2.addWeighted(ov2, 0.55, annotated, 0.45, 0, annotated)
        step_num = None
        if frame_idx in last5_frames:
            step_num = last5_frames.index(frame_idx) + 1
        bar_y  = oh - int(30 * OUTPUT_SCALE)
        bar_x0 = int(20  * OUTPUT_SCALE)
        bar_x1 = int((ow - 20) * OUTPUT_SCALE)
        cv2.rectangle(annotated, (bar_x0, bar_y), (bar_x1, bar_y + int(12*OUTPUT_SCALE)), (60,60,60), -1)
        if duration > 0:
            prog = min(1.0, (frame_idx - last5_frames[0]) / max(1, release_frame - last5_frames[0]))
            cv2.rectangle(annotated, (bar_x0, bar_y),
                          (bar_x0 + int(prog*(bar_x1-bar_x0)), bar_y + int(12*OUTPUT_SCALE)), (0,200,255), -1)
        info = ["MODULE: STEP CADENCE",
                f"Frame: {frame_idx}   Time: {frame_idx/fps:.2f}s",
                f"Duration (5 steps): {duration:.3f}s",
                f"Avg interval: {duration/4:.3f}s"]
        if step_num:
            foot_str = step_foot_map[frame_idx].upper()
            info.insert(1, f">>> STEP {step_num}/5  ({foot_str} FOOT) <<<")
            if step_num > 1:
                dt_step = (frame_idx - last5_frames[step_num-2]) / fps
                info.append(f"Since last step: {dt_step:.3f}s")
        if frame_idx == release_frame:
            info.insert(1, ">>> BALL RELEASE <<<")
        draw_info_box(annotated, info, x=10, y=10, font_scale=0.55 * OUTPUT_SCALE / 0.5)
        repeats = 1
        if frame_idx in last5_frames: repeats = int(fps * 1.2)
        if frame_idx == release_frame: repeats = int(fps * 1.5)
        for _ in range(repeats):
            vw.write(annotated)
        frame_idx += 1
    cap.release(); vw.release()
    compress_video(out_vid)
    print(f"✅ Step cadence video: {out_vid}")

    # Compute inter-step intervals for dataset
    intervals_s = [(last5_frames[i] - last5_frames[i-1]) / fps for i in range(1, 5)]
    return {
        "last5_frames":   last5_frames,
        "foot_labels":    foot_labels,
        "duration_s":     duration,
        "avg_interval_s": duration / 4,
        "intervals_s":    intervals_s,       # ← NEW: individual intervals
    }

# =============================================================================
# ── SECTION 6 : MODULE 2 — DELIVERY STRIDE
# =============================================================================
def find_foot_contact_velocity(df_xy, side, ref_idx, search_before=True):
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
        if velocity[i] < 0.5: refined = i
        else: break
    return refined

def run_delivery_stride(df_xy, fps, release_idx, release_frame, bowling_wrist,
                        meters_per_pixel, video_path, width, height):
    print("\n" + "─"*60)
    print("MODULE 2 — DELIVERY STRIDE")
    print("─"*60)
    for side in ("left", "right"):
        df_xy[f"{side}_ankle_y_s"] = smooth(df_xy[f"{side}_ankle_y"].values.astype(float))
    f_side = "left"  if bowling_wrist == "right_wrist" else "right"
    b_side = "right" if f_side == "left" else "left"
    ffc_idx = find_foot_contact_velocity(df_xy, f_side, release_idx, True)
    bfc_idx = find_foot_contact_velocity(df_xy, b_side, ffc_idx,     False)
    ffc_frame = int(df_xy.loc[ffc_idx, "frame"])
    bfc_frame = int(df_xy.loc[bfc_idx, "frame"])
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
    cap        = cv2.VideoCapture(video_path)
    frame_no   = 0
    bfc_drawn  = ffc_drawn = False
    print(f"\n🎬 Writing delivery stride video…")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_no += 1
        ann = cv2.resize(frame, (ow, oh)); s = OUTPUT_SCALE
        if frame_no >= bfc_frame: bfc_drawn = True
        if frame_no >= ffc_frame: ffc_drawn = True
        if bfc_drawn:
            cv2.circle(ann, (int(bx*s), int(by*s)), int(14*s), (0,0,255), -1)
            cv2.circle(ann, (int(bx*s), int(by*s)), int(16*s), (255,255,255), 2)
            cv2.putText(ann, "BFC", (int(bx*s)+int(18*s), int(by*s)+int(6*s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7*s/0.5, (0,0,255), 2, cv2.LINE_AA)
        if ffc_drawn:
            cv2.circle(ann, (int(fx*s), int(fy*s)), int(14*s), (0,255,0), -1)
            cv2.circle(ann, (int(fx*s), int(fy*s)), int(16*s), (255,255,255), 2)
            cv2.putText(ann, "FFC", (int(fx*s)+int(18*s), int(fy*s)+int(6*s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7*s/0.5, (0,255,0), 2, cv2.LINE_AA)
        if bfc_drawn and ffc_drawn:
            cv2.line(ann, (int(bx*s), int(by*s)), (int(fx*s), int(fy*s)), (255,255,0), max(2,int(3*s)))
            mid_x = int((bx+fx)/2*s); mid_y = int((by+fy)/2*s) - int(20*s)
            cv2.putText(ann, f"STRIDE: {stride_m:.2f} m", (mid_x-int(60*s), mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7*s/0.5, (255,255,0), 2, cv2.LINE_AA)
        event = ("BFC — Back Foot Contact"  if frame_no == bfc_frame else
                 "FFC — Front Foot Contact" if frame_no == ffc_frame else
                 "RELEASE"                  if frame_no == release_frame else "")
        info = ["MODULE: DELIVERY STRIDE",
                f"Frame: {frame_no}   Time: {frame_no/fps:.2f}s",
                f"BFC frame: {bfc_frame}  FFC frame: {ffc_frame}",
                f"Stride length: {stride_m:.2f} m",
                f"Stride duration: {duration_s:.3f} s"]
        if event: info.insert(1, f">>> {event} <<<")
        draw_info_box(ann, info, x=10, y=10, font_scale=0.55 * OUTPUT_SCALE / 0.5)
        repeats = 1
        if frame_no in (bfc_frame, ffc_frame, release_frame): repeats = int(fps * 1.5)
        for _ in range(repeats): vw.write(ann)
    cap.release(); vw.release()
    compress_video(out_vid)
    print(f"✅ Delivery stride video: {out_vid}")
    return {"bfc_frame": bfc_frame, "ffc_frame": ffc_frame,
            "stride_m": stride_m, "stride_duration_s": duration_s,
            "ffc_idx": ffc_idx, "bfc_idx": bfc_idx,
            "f_side": f_side, "b_side": b_side}

# =============================================================================
# ── SECTION 7 : MODULE 3 — ELBOW FLEXION
# =============================================================================
def run_elbow_flexion(df_xy, fps, release_idx, release_frame, bowling_wrist, video_path, width, height):
    print("\n" + "─"*60)
    print("MODULE 3 — ELBOW FLEXION")
    print("─"*60)
    arm          = "right" if bowling_wrist == "right_wrist" else "left"
    shoulder_col = f"{arm}_shoulder"
    elbow_col    = f"{arm}_elbow"
    wrist_col    = f"{arm}_wrist"
    arm_label    = arm.upper()
    angles, times, frames = [], [], []
    for idx in range(len(df_xy)):
        s_pt = (float(df_xy.loc[idx, f"{shoulder_col}_x"]), float(df_xy.loc[idx, f"{shoulder_col}_y"]))
        e_pt = (float(df_xy.loc[idx, f"{elbow_col}_x"]),   float(df_xy.loc[idx, f"{elbow_col}_y"]))
        w_pt = (float(df_xy.loc[idx, f"{wrist_col}_x"]),   float(df_xy.loc[idx, f"{wrist_col}_y"]))
        if all(np.isfinite(v) for pt in (s_pt, e_pt, w_pt) for v in pt):
            ang = angle_deg(s_pt, e_pt, w_pt)
            if np.isfinite(ang):
                angles.append(ang); times.append(float(df_xy.loc[idx, "time_s"])); frames.append(int(df_xy.loc[idx, "frame"]))
    if len(angles) == 0:
        print("⚠️  No valid elbow angles detected.")
        return None
    vel = [0.0]
    for i in range(1, len(angles)):
        dt_i = times[i] - times[i - 1]
        vel.append((angles[i] - angles[i-1]) / dt_i if dt_i > 0 else 0.0)
    df_elbow = pd.DataFrame({"frame": frames, "time_s": times, "elbow_angle": angles, "angular_velocity": vel})
    csv_path = out_path("elbow_full_analysis.csv")
    df_elbow.to_csv(csv_path, index=False)
    frame_to_angle = {int(r["frame"]): float(r["elbow_angle"]) for _, r in df_elbow.iterrows()}
    rel_t = release_frame / fps
    png_path = out_path("elbow_full_plots.png")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(times, angles, linewidth=1.5, color="deepskyblue")
    axes[0].axvline(x=rel_t, linestyle="--", color="red", label="Release")
    axes[0].set_ylabel("Elbow Angle (deg)"); axes[0].set_title(f"{arm_label} ARM — Elbow Angle (Full Clip)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(times, vel, linewidth=1.5, color="orange")
    axes[1].axvline(x=rel_t, linestyle="--", color="red"); axes[1].axhline(y=0, alpha=0.3)
    axes[1].set_ylabel("Angular Velocity (deg/s)"); axes[1].set_xlabel("Time (s)")
    axes[1].set_title(f"{arm_label} ARM — Elbow Angular Velocity (Full Clip)"); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(png_path, dpi=200); plt.close()
    print(f"📊 Elbow CSV:  {csv_path}")
    print(f"📈 Elbow plot: {png_path}")

    out_vid    = out_path("elbow_flexion_annotated.mp4")
    vw, ow, oh = sized_writer(out_vid, fps, width, height)
    cap        = cv2.VideoCapture(video_path)
    frame_no   = 0
    def elbow_color(ang):
        t = max(0.0, min(1.0, (180.0 - ang) / 120.0))
        return (0, int(255*(1-t)), int(255*t))
    print(f"\n🎬 Writing elbow flexion video…")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_no += 1; idx = frame_no - 1
        ann = cv2.resize(frame, (ow, oh)); s = OUTPUT_SCALE
        sh = (float(df_xy.loc[idx, f"{shoulder_col}_x"]), float(df_xy.loc[idx, f"{shoulder_col}_y"]))
        el = (float(df_xy.loc[idx, f"{elbow_col}_x"]),   float(df_xy.loc[idx, f"{elbow_col}_y"]))
        wr = (float(df_xy.loc[idx, f"{wrist_col}_x"]),   float(df_xy.loc[idx, f"{wrist_col}_y"]))
        ang = frame_to_angle.get(frame_no, np.nan)
        if all(np.isfinite(v) for pt in (sh, el, wr) for v in pt):
            col  = elbow_color(ang) if np.isfinite(ang) else (200,200,200)
            sh_d = (int(sh[0]*s), int(sh[1]*s)); el_d = (int(el[0]*s), int(el[1]*s)); wr_d = (int(wr[0]*s), int(wr[1]*s))
            cv2.line(ann, sh_d, el_d, col, max(3,int(5*s)), cv2.LINE_AA)
            cv2.line(ann, el_d, wr_d, col, max(3,int(5*s)), cv2.LINE_AA)
            cv2.circle(ann, sh_d, int(9*s), (255,255,255), -1, cv2.LINE_AA); cv2.circle(ann, sh_d, int(7*s), (255,100,0), -1, cv2.LINE_AA)
            cv2.circle(ann, el_d, int(11*s), (255,255,255), -1, cv2.LINE_AA); cv2.circle(ann, el_d, int(9*s), col, -1, cv2.LINE_AA)
            cv2.circle(ann, wr_d, int(9*s), (255,255,255), -1, cv2.LINE_AA); cv2.circle(ann, wr_d, int(7*s), (0,200,255), -1, cv2.LINE_AA)
            cv2.putText(ann, "S", (sh_d[0]+int(8*s), sh_d[1]-int(8*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.5*s/0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(ann, "E", (el_d[0]+int(8*s), el_d[1]-int(8*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.5*s/0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(ann, "W", (wr_d[0]+int(8*s), wr_d[1]-int(8*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.5*s/0.5, (255,255,255), 1, cv2.LINE_AA)
            if np.isfinite(ang):
                draw_angle_arc(ann, el_d, sh_d, wr_d, ang, color=col, radius=int(45*s))
        if np.isfinite(ang):
            gauge_x = ow - int(40*s); gauge_y0 = int(80*s); gauge_y1 = oh - int(80*s); gauge_h = gauge_y1 - gauge_y0
            cv2.rectangle(ann, (gauge_x, gauge_y0), (gauge_x+int(18*s), gauge_y1), (50,50,50), -1)
            fill = int(gauge_h * max(0, min(1, ang/180.0)))
            cv2.rectangle(ann, (gauge_x, gauge_y1-fill), (gauge_x+int(18*s), gauge_y1), elbow_color(ang), -1)
            cv2.rectangle(ann, (gauge_x, gauge_y0), (gauge_x+int(18*s), gauge_y1), (200,200,200), 1)
            cv2.putText(ann, "180", (gauge_x-int(38*s), gauge_y0+int(8*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.45*s/0.5, (200,200,200), 1)
            cv2.putText(ann, "0",   (gauge_x-int(18*s), gauge_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.45*s/0.5, (200,200,200), 1)
        event = ">>> BALL RELEASE <<<" if frame_no == release_frame else ""
        info  = [f"MODULE: {arm_label} ARM ELBOW FLEXION",
                 f"Frame: {frame_no}   Time: {frame_no/fps:.2f}s",
                 f"Elbow angle: {ang:.1f} deg" if np.isfinite(ang) else "Elbow angle: --",
                 "(180=straight  <180=flexed)"]
        if event: info.insert(1, event)
        draw_info_box(ann, info, x=10, y=10, font_scale=0.55 * OUTPUT_SCALE / 0.5)
        vw.write(ann)
    cap.release(); vw.release()
    compress_video(out_vid)
    print(f"✅ Elbow flexion video: {out_vid}")
    return {"df_elbow": df_elbow, "release_frame": release_frame,
            "arm": arm, "frame_to_angle": frame_to_angle}   # ← arm & map exposed

# =============================================================================
# ── SECTION 8 : MODULE 4 — KNEE FLEXION
# =============================================================================
class KneeLockTracker:
    def __init__(self, side, fps, max_jump_leglen=0.60, conf_th=0.25):
        self.side = side; self.fps = fps; self.conf_th = conf_th; self.max_jump = max_jump_leglen
        self.prev_knee = self.prev_leglen = None
        if side == "L": self.iH, self.iK, self.iA = 11, 13, 15
        else:           self.iH, self.iK, self.iA = 12, 14, 16

    def update(self, kp_xy, kp_conf):
        if kp_xy is None: return False, None, None, None, np.nan
        H  = kp_xy[self.iH]; K  = kp_xy[self.iK]; A  = kp_xy[self.iA]
        cH = kp_conf[self.iH]; cK = kp_conf[self.iK]; cA = kp_conf[self.iA]
        if cH < self.conf_th or cK < self.conf_th or cA < self.conf_th: return False, None, None, None, np.nan
        leglen = float(np.linalg.norm(H-K) + np.linalg.norm(K-A))
        if self.prev_knee is not None and self.prev_leglen is not None:
            fps_scale = max(0.7, min(2.0, 30.0/max(self.fps,1e-6)))
            if float(np.linalg.norm(K-self.prev_knee)) > self.max_jump*self.prev_leglen*fps_scale:
                return False, None, None, None, np.nan
        ang = angle_deg(tuple(H), tuple(K), tuple(A))
        self.prev_knee = K.copy(); self.prev_leglen = leglen
        return True, H, K, A, ang

def run_knee_flexion(df_xy, df_conf, fps, release_idx, release_frame, knee_side, video_path, width, height):
    print("\n" + "─"*60)
    print("MODULE 4 — KNEE FLEXION")
    print("─"*60)
    side_label = "LEFT (front)" if knee_side == "L" else "RIGHT (front)"
    print(f"  Tracking: {side_label} knee")
    tracker = KneeLockTracker(knee_side, fps, KNEE_MAX_JUMP_LEGLEN, KNEE_CONF_TH)
    rows = []; ankle_y_list = []; wrist_xy_list = []; last_good = None
    iW = KP_IDX["left_wrist"] if knee_side == "L" else KP_IDX["right_wrist"]

    # Knee side → hip/knee/ankle column names (for frame dataset)
    kside = "left" if knee_side == "L" else "right"

    for i in range(len(df_xy)):
        kp_xy   = np.array([[df_xy.loc[i, f"{name}_x"], df_xy.loc[i, f"{name}_y"]] for name in KEYPOINT_NAMES], dtype=float)
        kp_conf = np.array([df_conf.loc[i, f"{name}_conf"] for name in KEYPOINT_NAMES], dtype=float)
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
        rows.append({"frame": int(df_xy.loc[i,"frame"]), "time_s": float(df_xy.loc[i,"time_s"]), "knee_angle_deg": ang})

    df_knee = pd.DataFrame(rows)
    ay   = pd.Series(ankle_y_list).interpolate(limit_direction="both").values
    ay_s = pd.Series(ay).rolling(7, center=True, min_periods=1).mean().values
    v    = np.gradient(ay_s)
    ffc_cands = [i for i in range(2, len(ay_s)-2) if ay_s[i]<ay_s[i-1] and ay_s[i]<ay_s[i+1] and abs(v[i])<np.percentile(abs(v),35)]
    knee_ffc  = ffc_cands[0] if ffc_cands else None
    wpts      = np.array(wrist_xy_list, float)
    knee_rel  = None
    if knee_ffc is not None and len(wpts) >= knee_ffc + int(0.3*fps):
        d   = np.linalg.norm(np.diff(wpts,axis=0),axis=1)
        d   = np.r_[d[0], d]
        d_s = pd.Series(d).rolling(5,center=True,min_periods=1).mean().values
        a_w = knee_ffc + int(0.10*fps)
        b_w = min(len(d_s)-1, knee_ffc+int(1.20*fps))
        if b_w > a_w:
            knee_rel = int(a_w + np.argmax(d_s[a_w:b_w+1]))
    df_knee["is_ffc"]     = False
    df_knee["is_release"] = False
    if knee_ffc is not None and 0<=knee_ffc<len(df_knee): df_knee.loc[knee_ffc, "is_ffc"] = True
    if knee_rel is not None and 0<=knee_rel<len(df_knee): df_knee.loc[knee_rel, "is_release"] = True
    csv_path = out_path("knee_angles.csv"); df_knee.to_csv(csv_path, index=False)
    if knee_ffc is not None and knee_rel is not None and knee_rel > knee_ffc:
        df_knee.loc[knee_ffc:knee_rel, ["frame","time_s","knee_angle_deg"]].to_csv(
            out_path("knee_angles_ffc_to_release.csv"), index=False)
    png_path = out_path("knee_flexion_analysis.png")
    plt.figure(figsize=(10,4))
    plt.plot(df_knee["time_s"], df_knee["knee_angle_deg"], linewidth=1.5, color="limegreen")
    if knee_ffc is not None: plt.axvline(df_knee.loc[knee_ffc,"time_s"], linestyle="--", color="blue", label="FFC")
    if knee_rel is not None: plt.axvline(df_knee.loc[knee_rel,"time_s"], linestyle=":", color="red", label="Release")
    plt.xlabel("Time (s)"); plt.ylabel("Knee Angle (deg)"); plt.title(f"{side_label} Knee Flexion — Full Clip")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(png_path, dpi=200); plt.close()
    print(f"📊 Knee CSV:  {csv_path}")
    print(f"📈 Knee plot: {png_path}")
    frame_to_knee_angle = {int(r["frame"]): float(r["knee_angle_deg"]) for _, r in df_knee.iterrows()}
    out_vid    = out_path("knee_flexion_annotated.mp4")
    vw, ow, oh = sized_writer(out_vid, fps, width, height)
    cap        = cv2.VideoCapture(video_path)
    frame_no   = 0
    tracker2   = KneeLockTracker(knee_side, fps, KNEE_MAX_JUMP_LEGLEN, KNEE_CONF_TH)
    last_good2 = None
    def knee_color(ang):
        t = max(0.0, min(1.0, (180.0-ang)/120.0))
        return (0, int(255*(1-t)), int(255*t))
    print(f"\n🎬 Writing knee flexion video…")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_no += 1; idx = frame_no - 1
        if idx >= len(df_xy): break
        ann = cv2.resize(frame, (ow, oh)); s = OUTPUT_SCALE
        kp_xy   = np.array([[df_xy.loc[idx, f"{name}_x"], df_xy.loc[idx, f"{name}_y"]] for name in KEYPOINT_NAMES], dtype=float)
        kp_conf = np.array([df_conf.loc[idx, f"{name}_conf"] for name in KEYPOINT_NAMES], dtype=float)
        ok2, H2, K2, A2, ang2 = tracker2.update(kp_xy, kp_conf)
        if ok2: last_good2 = (H2, K2, A2, ang2)
        ang_vid = frame_to_knee_angle.get(frame_no, np.nan)
        if last_good2 is not None:
            H3, K3, A3, _ = last_good2
            col = knee_color(ang_vid) if np.isfinite(ang_vid) else (200,200,200)
            H_d = (int(H3[0]*s), int(H3[1]*s)); K_d = (int(K3[0]*s), int(K3[1]*s)); A_d = (int(A3[0]*s), int(A3[1]*s))
            cv2.line(ann, H_d, K_d, col, max(3,int(5*s)), cv2.LINE_AA); cv2.line(ann, K_d, A_d, col, max(3,int(5*s)), cv2.LINE_AA)
            cv2.circle(ann, H_d, int(9*s), (255,255,255), -1, cv2.LINE_AA); cv2.circle(ann, H_d, int(7*s), (200,0,255), -1, cv2.LINE_AA)
            cv2.circle(ann, K_d, int(11*s), (255,255,255), -1, cv2.LINE_AA); cv2.circle(ann, K_d, int(9*s), col, -1, cv2.LINE_AA)
            cv2.circle(ann, A_d, int(9*s), (255,255,255), -1, cv2.LINE_AA); cv2.circle(ann, A_d, int(7*s), (100,255,0), -1, cv2.LINE_AA)
            cv2.putText(ann, "HIP",   (H_d[0]+int(8*s), H_d[1]-int(8*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.45*s/0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(ann, "KNEE",  (K_d[0]+int(8*s), K_d[1]-int(8*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.45*s/0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(ann, "ANKLE", (A_d[0]+int(8*s), A_d[1]-int(8*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.45*s/0.5, (255,255,255), 1, cv2.LINE_AA)
            if np.isfinite(ang_vid): draw_angle_arc(ann, K_d, H_d, A_d, ang_vid, color=col, radius=int(50*s))
        if np.isfinite(ang_vid):
            gauge_x = ow - int(40*s); gauge_y0 = int(80*s); gauge_y1 = oh - int(80*s); gauge_h = gauge_y1 - gauge_y0
            col_g = knee_color(ang_vid)
            cv2.rectangle(ann, (gauge_x,gauge_y0), (gauge_x+int(18*s),gauge_y1), (50,50,50), -1)
            fill = int(gauge_h * max(0,min(1,ang_vid/180.0)))
            cv2.rectangle(ann, (gauge_x,gauge_y1-fill), (gauge_x+int(18*s),gauge_y1), col_g, -1)
            cv2.rectangle(ann, (gauge_x,gauge_y0), (gauge_x+int(18*s),gauge_y1), (200,200,200), 1)
            cv2.putText(ann,"180",(gauge_x-int(38*s),gauge_y0+int(8*s)), cv2.FONT_HERSHEY_SIMPLEX,0.4*s/0.5,(200,200,200),1)
            cv2.putText(ann,"0",(gauge_x-int(18*s),gauge_y1), cv2.FONT_HERSHEY_SIMPLEX,0.4*s/0.5,(200,200,200),1)
        is_ffc_frame = (knee_ffc is not None and frame_no == int(df_knee.loc[knee_ffc,"frame"]))
        is_rel_frame = (knee_rel is not None and frame_no == int(df_knee.loc[knee_rel,"frame"]))
        info = [f"MODULE: {side_label.upper()} KNEE FLEXION",
                f"Frame: {frame_no}   Time: {frame_no/fps:.2f}s",
                f"Knee angle: {ang_vid:.1f} deg" if np.isfinite(ang_vid) else "Knee angle: --",
                "(180=straight  <180=flexed)"]
        if is_ffc_frame: info.insert(1, ">>> FRONT FOOT CONTACT (FFC) <<<")
        if is_rel_frame: info.insert(1, ">>> BALL RELEASE <<<")
        draw_info_box(ann, info, x=10, y=10, font_scale=0.55*OUTPUT_SCALE/0.5)
        repeats = 1
        if is_ffc_frame or is_rel_frame: repeats = int(fps*1.5)
        for _ in range(repeats): vw.write(ann)
    cap.release(); vw.release()
    compress_video(out_vid)
    print(f"✅ Knee flexion video: {out_vid}")
    return {"df_knee": df_knee, "ffc_idx": knee_ffc, "release_idx": knee_rel,
            "frame_to_knee_angle": frame_to_knee_angle, "kside": kside}  # ← extras exposed

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

def run_head_position(df_xy, fps, release_idx, release_frame, bowling_wrist,
                      cm_per_pixel, pixels_per_meter, video_path, width, height):
    print("\n" + "─"*60)
    print("MODULE 5 — HEAD / COM POSITION")
    print("─"*60)
    head_x = [safe_head_xy(df_xy,i)[0] for i in range(len(df_xy))]
    head_y = [safe_head_xy(df_xy,i)[1] for i in range(len(df_xy))]
    df_xy["head_x"] = head_x; df_xy["head_y"] = head_y
    df_xy["hip_mid_x"] = (df_xy["left_hip_x"].values + df_xy["right_hip_x"].values) / 2.0
    df_xy["hip_mid_y"] = (df_xy["left_hip_y"].values + df_xy["right_hip_y"].values) / 2.0
    l_peaks, l_sig = detect_ankle_peaks(df_xy, fps, "left",  release_idx)
    r_peaks, r_sig = detect_ankle_peaks(df_xy, fps, "right", release_idx)
    df_xy["left_ankle_y_sm"]  = l_sig
    df_xy["right_ankle_y_sm"] = r_sig
    if len(l_peaks) == 0 or len(r_peaks) == 0:
        print("⚠️  Insufficient ankle peaks for head/COM analysis."); return None
    hx_arr = df_xy["hip_mid_x"].values; n = len(hx_arr)
    early  = np.nanmean(hx_arr[:max(5,n//10)]); late = np.nanmean(hx_arr[-max(5,n//10):])
    direction = 1 if late >= early else -1
    l_last = int(l_peaks[-1]); r_last = int(r_peaks[-1])
    ffc_candidate = "left" if l_last > r_last else "right"
    if VIEW_MODE == "SIDE":
        ref_i  = max(l_last, r_last)
        lx_ref = float(df_xy.loc[ref_i, "left_ankle_x"]); rx_ref = float(df_xy.loc[ref_i, "right_ankle_x"])
        front_is_left = (lx_ref >= rx_ref) if direction == 1 else (lx_ref <= rx_ref)
    else:
        front_is_left = (ffc_candidate == "left")
    front_ankle = "left_ankle"  if front_is_left else "right_ankle"
    front_peaks = l_peaks       if front_is_left else r_peaks
    back_peaks  = r_peaks       if front_is_left else l_peaks
    fp_before = front_peaks[front_peaks < release_idx]
    if len(fp_before) == 0: print("⚠️  No front-foot peak before release."); return None
    ffc_idx  = int(fp_before[-1])
    bp_before = back_peaks[back_peaks < ffc_idx]
    bfc_idx   = int(bp_before[-1]) if len(bp_before) > 0 else None
    ffc_frame_h = int(df_xy.loc[ffc_idx, "frame"])
    bfc_frame_h = int(df_xy.loc[bfc_idx, "frame"]) if bfc_idx is not None else None
    print(f"  Front ankle : {front_ankle}  (view={VIEW_MODE})")
    print(f"  FFC frame   : {ffc_frame_h}")
    print(f"  BFC frame   : {bfc_frame_h if bfc_frame_h else 'not detected'}")
    df_xy["head_x_sm"]           = smooth(np.array(head_x, float))
    df_xy[f"{front_ankle}_x_sm"] = smooth(df_xy[f"{front_ankle}_x"].values.astype(float))
    df_xy["videoDx_px"]          = df_xy["head_x_sm"] - df_xy[f"{front_ankle}_x_sm"]
    df_xy["videoDx_cm"]          = df_xy["videoDx_px"] * cm_per_pixel
    seg = df_xy.iloc[ffc_idx:release_idx+1].copy()
    seg[["frame","time_s","videoDx_px","videoDx_cm"]].to_csv(out_path("headDx_vs_time.csv"), index=False)
    plt.figure(figsize=(10,4))
    plt.plot(seg["time_s"], seg["videoDx_cm"], linewidth=1.5, color="gold")
    plt.axhline(0, linestyle="--", color="gray", alpha=0.5)
    plt.xlabel("Time (s)"); plt.ylabel("Head Dx (cm)")
    plt.title("Head Horizontal Offset relative to Front Foot (FFC → Release)")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(out_path("headDx_vs_time.png"), dpi=200); plt.close()

    def offsets_at(i, fa):
        hx2 = float(df_xy.loc[i,"head_x"]); hy2 = float(df_xy.loc[i,"head_y"])
        fx2 = float(df_xy.loc[i,f"{fa}_x"]); fy2 = float(df_xy.loc[i,f"{fa}_y"])
        dx, dy = hx2-fx2, hy2-fy2; d = float(np.hypot(dx,dy))
        return {"Dx_cm": dx*cm_per_pixel, "Dy_cm": dy*cm_per_pixel, "D_cm": d*cm_per_pixel}

    ffc_off = offsets_at(ffc_idx, front_ankle)
    bfc_off = offsets_at(bfc_idx, front_ankle) if bfc_idx is not None else None
    print(f"\n  Head @ FFC — Dx:{ffc_off['Dx_cm']:.1f}cm  Dy:{ffc_off['Dy_cm']:.1f}cm  D:{ffc_off['D_cm']:.1f}cm")
    if bfc_off:
        print(f"  Head @ BFC — Dx:{bfc_off['Dx_cm']:.1f}cm  Dy:{bfc_off['Dy_cm']:.1f}cm  D:{bfc_off['D_cm']:.1f}cm")
    metrics = {"view_mode": VIEW_MODE, "front_ankle": front_ankle,
               "ffc_frame": ffc_frame_h, "bfc_frame": bfc_frame_h,
               "head_offset_at_FFC": ffc_off, "head_offset_at_BFC": bfc_off}
    with open(out_path("head_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([{"ffc_frame": ffc_frame_h, "bfc_frame": bfc_frame_h, **ffc_off}]).to_csv(
        out_path("head_metrics.csv"), index=False)
    print(f"📌 Head metrics: {out_path('head_metrics.json')}")
    print(f"📈 Head Dx plot: {out_path('headDx_vs_time.png')}")

    out_vid    = out_path("head_position_annotated.mp4")
    vw, ow, oh = sized_writer(out_vid, fps, width, height)
    start_f    = bfc_frame_h if bfc_frame_h else ffc_frame_h
    cap        = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_f-1))
    print(f"\n🎬 Writing head position video…")
    frame_no = start_f
    while frame_no <= release_frame:
        ret, frame = cap.read()
        if not ret: break
        idx2 = frame_no - 1; ann = cv2.resize(frame, (ow, oh)); s = OUTPUT_SCALE
        hx2 = float(df_xy.loc[idx2,"head_x"]) if idx2<len(df_xy) else np.nan
        hy2 = float(df_xy.loc[idx2,"head_y"]) if idx2<len(df_xy) else np.nan
        fx2 = float(df_xy.loc[idx2,f"{front_ankle}_x"]) if idx2<len(df_xy) else np.nan
        fy2 = float(df_xy.loc[idx2,f"{front_ankle}_y"]) if idx2<len(df_xy) else np.nan
        dx_cm2 = dy_cm2 = np.nan
        if all(np.isfinite(v) for v in (hx2,hy2,fx2,fy2)):
            dx_cm2 = (hx2-fx2)*cm_per_pixel; dy_cm2 = (hy2-fy2)*cm_per_pixel
            H_d = (int(hx2*s), int(hy2*s)); F_d = (int(fx2*s), int(fy2*s))
            cv2.circle(ann, F_d, int(14*s), (0,0,255), -1, cv2.LINE_AA)
            cv2.circle(ann, F_d, int(16*s), (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(ann, "FRONT FOOT", (F_d[0]+int(18*s), F_d[1]+int(6*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.5*s/0.5, (0,150,255), 2, cv2.LINE_AA)
            cv2.circle(ann, H_d, int(12*s), (0,255,0), -1, cv2.LINE_AA)
            cv2.circle(ann, H_d, int(14*s), (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(ann, "HEAD", (H_d[0]+int(14*s), H_d[1]-int(10*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.5*s/0.5, (0,255,0), 2, cv2.LINE_AA)
            intensity = min(1.0, abs(dx_cm2)/30.0)
            lc = (0, int(255*(1-intensity)), int(255*intensity))
            cv2.line(ann, H_d, F_d, lc, max(2,int(3*s)), cv2.LINE_AA)
            cv2.arrowedLine(ann, F_d, (H_d[0],F_d[1]), (255,255,0), max(2,int(3*s)), cv2.LINE_AA, tipLength=0.05)
            mid_dx = ((F_d[0]+H_d[0])//2, F_d[1]-int(15*s))
            cv2.putText(ann, f"Dx={dx_cm2:.1f}cm", mid_dx, cv2.FONT_HERSHEY_SIMPLEX, 0.55*s/0.5, (255,255,0), 2, cv2.LINE_AA)
        event_label = ("BFC" if frame_no == bfc_frame_h else "FFC" if frame_no == ffc_frame_h else
                       "RELEASE" if frame_no == release_frame else "")
        info = ["MODULE: HEAD / COM POSITION",
                f"Frame: {frame_no}   Time: {frame_no/fps:.2f}s",
                f"Head Dx: {dx_cm2:.1f} cm" if np.isfinite(dx_cm2) else "Head Dx: --",
                f"Head Dy: {dy_cm2:.1f} cm" if np.isfinite(dy_cm2) else "Head Dy: --",
                "Green=Head  Red=Front foot"]
        if event_label: info.insert(1, f">>> {event_label} <<<")
        draw_info_box(ann, info, x=10, y=10, font_scale=0.55*OUTPUT_SCALE/0.5)
        repeats = 6
        if frame_no in ([bfc_frame_h] if bfc_frame_h else []) + [ffc_frame_h, release_frame]:
            repeats += int(fps*2.0)
        for _ in range(repeats): vw.write(ann)
        frame_no += 1
    cap.release(); vw.release()
    compress_video(out_vid)
    print(f"✅ Head position video: {out_vid}")
    return {"ffc_frame": ffc_frame_h, "bfc_frame": bfc_frame_h,
            "front_ankle": front_ankle, "ffc_offset": ffc_off, "bfc_offset": bfc_off,
            "ffc_idx": ffc_idx}  # ← ffc_idx exposed for frame dataset

# =============================================================================
# ── SECTION 10 : MODULE 6 — WRIST VELOCITY
# =============================================================================
def stabilize_lr(df):
    pairs = [(L,R) for L,R in [
        ("left_wrist","right_wrist"), ("left_elbow","right_elbow"),
        ("left_shoulder","right_shoulder"), ("left_hip","right_hip"),
        ("left_knee","right_knee"), ("left_ankle","right_ankle"),
    ] if f"{L}_x" in df.columns and f"{R}_x" in df.columns]
    if not pairs or len(df) < 2: return df
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

def run_wrist_velocity(df_xy, fps, release_idx, release_frame, bowling_wrist,
                       meters_per_pixel, video_path, width, height):
    print("\n" + "─"*60)
    print("MODULE 6 — WRIST VELOCITY")
    print("─"*60)
    arm        = "right" if bowling_wrist == "right_wrist" else "left"
    elbow_name = f"{arm}_elbow"; wrist_name = f"{arm}_wrist"; dt = 1.0 / fps
    df_stab = stabilize_lr(df_xy.copy())
    wx = smooth(df_stab[f"{wrist_name}_x"].values.astype(float))
    wy = smooth(df_stab[f"{wrist_name}_y"].values.astype(float))
    ex = smooth(df_stab[f"{elbow_name}_x"].values.astype(float))
    ey = smooth(df_stab[f"{elbow_name}_y"].values.astype(float))
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
    theta   = np.arctan2(wy-ey, wx-ex)
    theta_u = np.unwrap(theta)
    omega   = central_diff(theta_u, dt)
    omega_sm = smooth(omega)
    df_xy["forearm_angle_rad"]         = theta_u
    df_xy["wrist_angular_vel_rads_sm"] = omega_sm
    if RELEASE_DETECT_MODE == "PEAK_WRIST_SPEED":
        rel_idx2 = int(np.nanargmax(spd_sm))
    else:
        rel_idx2 = int(np.nanargmin(wy))
    rel_frame2 = int(df_xy.loc[rel_idx2, "frame"])
    pk_idx   = int(np.nanargmax(spd_sm))
    pk_frame = int(df_xy.loc[pk_idx, "frame"])
    pk_speed = float(spd_sm[pk_idx])
    win_frames = int(round(0.12*fps))
    a, b       = max(0,rel_idx2-win_frames), min(len(df_xy)-1,rel_idx2+win_frames)
    near_idx   = a + int(np.nanargmax(spd_sm[a:b+1]))
    near_frame = int(df_xy.loc[near_idx, "frame"])
    near_speed = float(spd_sm[near_idx])
    near_om_idx   = a + int(np.nanargmax(np.abs(omega_sm[a:b+1])))
    near_om_frame = int(df_xy.loc[near_om_idx, "frame"])
    near_om_peak  = float(omega_sm[near_om_idx])
    spd_at_rel    = float(spd_sm[rel_idx2])
    spd_at_rel_k  = spd_at_rel * 3.6
    print(f"  Release frame:                   {rel_frame2}")
    print(f"  Wrist speed @ release:           {spd_at_rel:.2f} m/s  ({spd_at_rel_k:.1f} km/h)")
    print(f"  Peak wrist speed (overall):      {pk_speed:.2f} m/s  @ frame {pk_frame}")
    print(f"  Peak wrist speed (near release): {near_speed:.2f} m/s  @ frame {near_frame}")
    print(f"  Peak angular velocity (near):    {near_om_peak:.2f} rad/s @ frame {near_om_frame}")

    ball_proxy  = {"mps": spd_at_rel, "kmh": spd_at_rel_k}
    ball_mode_b = None
    if BALL_MODEL_PATH is not None:
        ball_model_b = YOLO(BALL_MODEL_PATH)
        cap_b = cv2.VideoCapture(video_path)
        cap_b.set(cv2.CAP_PROP_POS_FRAMES, max(0,rel_frame2-1))
        pts_b = []
        for _ in range(BALL_TRACK_FRAMES):
            ok_b, frm_b = cap_b.read()
            if not ok_b: break
            res_b = ball_model_b.predict(frm_b, verbose=False)
            if res_b and res_b[0].boxes is not None and len(res_b[0].boxes)>0:
                confs_b = res_b[0].boxes.conf.cpu().numpy(); xyxy_b = res_b[0].boxes.xyxy.cpu().numpy()
                j = int(np.argmax(confs_b)); x1_b,y1_b,x2_b,y2_b = xyxy_b[j]
                pts_b.append(((x1_b+x2_b)/2,(y1_b+y2_b)/2))
            else:
                pts_b.append((np.nan,np.nan))
        cap_b.release()
        pts_b = np.array(pts_b,float)
        if np.any(np.isfinite(pts_b)):
            bx_s = smooth(pts_b[:,0]); by_s = smooth(pts_b[:,1])
            bvx  = central_diff(bx_s,dt)*meters_per_pixel; bvy = central_diff(by_s,dt)*meters_per_pixel
            bspd = np.sqrt(bvx**2+bvy**2); bspd_v = bspd[np.isfinite(bspd)]
            if len(bspd_v):
                ball_mode_b = {"avg_kmh": float(np.nanmean(bspd_v)*3.6), "peak_kmh": float(np.nanmax(bspd_v)*3.6)}

    ts_cols = ["frame","time_s","wrist_x_sm","wrist_y_sm","elbow_x_sm","elbow_y_sm",
               "wrist_vx_mps","wrist_vy_mps","wrist_speed_mps_sm","wrist_speed_kmh_sm",
               "forearm_angle_rad","wrist_angular_vel_rads_sm"]
    df_xy[ts_cols].to_csv(out_path("wrist_timeseries.csv"), index=False)
    metrics_w = {
        "bowling_arm": arm.upper(), "release_frame": rel_frame2,
        "wrist_speed_at_release_mps": spd_at_rel, "wrist_speed_at_release_kmh": spd_at_rel_k,
        "peak_wrist_speed_overall_mps": pk_speed,
        "peak_wrist_speed_near_release_mps": near_speed,
        "peak_angular_vel_near_release_rads": near_om_peak,
        "ball_speed_proxy": ball_proxy, "ball_speed_mode_b": ball_mode_b,
    }
    with open(out_path("wrist_ball_metrics.json"), "w") as f:
        json.dump(metrics_w, f, indent=2)
    rel_t = rel_frame2 / fps
    plt.figure(figsize=(10,4))
    plt.plot(df_xy["time_s"], df_xy["wrist_speed_kmh_sm"], linewidth=1.5, color="cyan")
    plt.axvline(x=rel_t, linestyle="--", color="red", label="Release")
    plt.axvline(x=near_frame/fps, linestyle=":", color="orange", label="Peak (near release)")
    plt.xlabel("Time (s)"); plt.ylabel("Speed (km/h)"); plt.title(f"{arm.upper()} WRIST Speed vs Time")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_path("wrist_speed_vs_time.png"), dpi=200); plt.close()
    plt.figure(figsize=(10,4))
    plt.plot(df_xy["time_s"], omega_sm, linewidth=1.5, color="magenta")
    plt.axvline(x=rel_t, linestyle="--", color="red", label="Release")
    plt.xlabel("Time (s)"); plt.ylabel("Angular velocity (rad/s)"); plt.title(f"{arm.upper()} FOREARM Angular Velocity vs Time")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_path("wrist_omega_vs_time.png"), dpi=200); plt.close()
    print(f"📊 Wrist CSV:  {out_path('wrist_timeseries.csv')}")
    print(f"📌 Metrics:    {out_path('wrist_ball_metrics.json')}")

    out_vid    = out_path("wrist_velocity_annotated.mp4")
    vw, ow, oh = sized_writer(out_vid, fps, width, height)
    cap        = cv2.VideoCapture(video_path)
    frame_no   = 0; slow_mo = 4
    max_speed_kmh = max(float(np.nanmax(df_xy["wrist_speed_kmh_sm"])), 1.0)
    print(f"\n🎬 Writing wrist velocity video…")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_no += 1; idx = frame_no - 1
        if idx >= len(df_xy): break
        ann = cv2.resize(frame, (ow, oh)); s = OUTPUT_SCALE
        wxi = float(df_xy.loc[idx,"wrist_x_sm"]); wyi = float(df_xy.loc[idx,"wrist_y_sm"])
        exi = float(df_xy.loc[idx,"elbow_x_sm"]); eyi = float(df_xy.loc[idx,"elbow_y_sm"])
        spd_i   = float(df_xy.loc[idx,"wrist_speed_mps_sm"]); spd_k = spd_i * 3.6
        omega_i = float(df_xy.loc[idx,"wrist_angular_vel_rads_sm"])
        vxi     = float(df_xy.loc[idx,"wrist_vx_mps"]); vyi = float(df_xy.loc[idx,"wrist_vy_mps"])
        W_d = (int(wxi*s), int(wyi*s)); E_d = (int(exi*s), int(eyi*s))
        t_spd = min(1.0, spd_k/max_speed_kmh)
        arm_col = (0, int(255*(1-t_spd)), int(255*t_spd))
        cv2.line(ann, E_d, W_d, arm_col, max(3,int(5*s)), cv2.LINE_AA)
        cv2.circle(ann, E_d, int(9*s), (255,255,255), -1, cv2.LINE_AA); cv2.circle(ann, E_d, int(7*s), (255,100,0), -1, cv2.LINE_AA)
        cv2.circle(ann, W_d, int(9*s), (255,255,255), -1, cv2.LINE_AA); cv2.circle(ann, W_d, int(7*s), arm_col, -1, cv2.LINE_AA)
        cv2.putText(ann, "ELBOW", (E_d[0]+int(8*s), E_d[1]-int(8*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.45*s/0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(ann, "WRIST", (W_d[0]+int(8*s), W_d[1]-int(8*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.45*s/0.5, (255,255,255), 1, cv2.LINE_AA)
        arrow_scale = 0.08
        end_x = int(wxi*s + vxi/meters_per_pixel*arrow_scale*s); end_y = int(wyi*s + vyi/meters_per_pixel*arrow_scale*s)
        cv2.arrowedLine(ann, W_d, (end_x,end_y), (0,255,255), max(2,int(3*s)), cv2.LINE_AA, tipLength=0.15)
        gauge_x = ow - int(40*s); gauge_y0 = int(60*s); gauge_y1 = oh - int(60*s); gauge_h = gauge_y1 - gauge_y0
        cv2.rectangle(ann,(gauge_x,gauge_y0),(gauge_x+int(18*s),gauge_y1),(50,50,50),-1)
        fill_g = int(gauge_h*t_spd)
        cv2.rectangle(ann,(gauge_x,gauge_y1-fill_g),(gauge_x+int(18*s),gauge_y1),arm_col,-1)
        cv2.rectangle(ann,(gauge_x,gauge_y0),(gauge_x+int(18*s),gauge_y1),(200,200,200),1)
        cv2.putText(ann,f"{max_speed_kmh:.0f}",(gauge_x-int(35*s),gauge_y0+int(8*s)), cv2.FONT_HERSHEY_SIMPLEX,0.4*s/0.5,(200,200,200),1)
        cv2.putText(ann,"0",(gauge_x-int(15*s),gauge_y1), cv2.FONT_HERSHEY_SIMPLEX,0.4*s/0.5,(200,200,200),1)
        cv2.putText(ann,"km/h",(gauge_x-int(5*s),gauge_y1+int(18*s)), cv2.FONT_HERSHEY_SIMPLEX,0.4*s/0.5,(200,200,200),1)
        event = (">>> BALL RELEASE <<<"    if frame_no == rel_frame2 else
                 ">>> PEAK WRIST SPEED <<<" if frame_no == near_frame else "")
        info  = [f"MODULE: {arm.upper()} WRIST VELOCITY",
                 f"Frame: {frame_no}   Time: {frame_no/fps:.2f}s",
                 f"Wrist speed: {spd_i:.2f} m/s  ({spd_k:.1f} km/h)",
                 f"Angular vel: {omega_i:.2f} rad/s",
                 f"Ball speed proxy: {spd_at_rel_k:.1f} km/h"]
        if event: info.insert(1, event)
        draw_info_box(ann, info, x=10, y=10, font_scale=0.55*OUTPUT_SCALE/0.5)
        for _ in range(slow_mo): vw.write(ann)
    cap.release(); vw.release()
    compress_video(out_vid)
    print(f"✅ Wrist velocity video: {out_vid}")
    return {"release_frame": rel_frame2, "speed_at_release_mps": spd_at_rel,
            "peak_speed_near_release_mps": near_speed,
            "peak_speed_overall_mps": pk_speed,
            "peak_angular_vel_rads": near_om_peak,
            "ball_proxy_kmh": spd_at_rel_k,
            "arm": arm}

# =============================================================================
# ── SECTION 11 : SUMMARY PRINT
# =============================================================================
def print_summary(hand, release_frame, fps,
                  r_stride, r_cadence, r_elbow, r_knee, r_head, r_wrist):
    print("\n" + "="*60)
    print("  FULL PIPELINE SUMMARY")
    print("="*60)
    arm_label  = "RIGHT" if hand == "R" else "LEFT"
    knee_label = "LEFT (front)" if hand == "R" else "RIGHT (front)"
    print(f"  Bowling arm  : {arm_label}")
    print(f"  Front knee   : {knee_label}")
    print(f"  Release frame: {release_frame}  ({release_frame/fps:.2f}s)")
    if r_cadence:
        print(f"\n📍 Step Cadence")
        print(f"   Last 5 frames : {r_cadence['last5_frames']}")
        print(f"   Duration 1→5  : {r_cadence['duration_s']:.3f}s   Avg interval: {r_cadence['avg_interval_s']:.3f}s")
    if r_stride:
        print(f"\n📍 Delivery Stride")
        print(f"   BFC:{r_stride['bfc_frame']}  FFC:{r_stride['ffc_frame']}")
        print(f"   Length: {r_stride['stride_m']:.2f}m   Duration: {r_stride['stride_duration_s']:.3f}s")
    if r_elbow:
        print(f"\n📍 Elbow Flexion  → elbow_full_plots.png + elbow_flexion_annotated.mp4")
    if r_knee:
        print(f"\n📍 Knee Flexion   → knee_flexion_analysis.png + knee_flexion_annotated.mp4")
    if r_head:
        o = r_head["ffc_offset"]
        print(f"\n📍 Head / COM Position")
        print(f"   Front ankle: {r_head['front_ankle']}")
        print(f"   @ FFC — Dx:{o['Dx_cm']:.1f}cm  Dy:{o['Dy_cm']:.1f}cm  D:{o['D_cm']:.1f}cm")
        if r_head["bfc_offset"]:
            b = r_head["bfc_offset"]
            print(f"   @ BFC — Dx:{b['Dx_cm']:.1f}cm  Dy:{b['Dy_cm']:.1f}cm  D:{b['D_cm']:.1f}cm")
    if r_wrist:
        print(f"\n📍 Wrist Velocity")
        print(f"   Speed @ release : {r_wrist['speed_at_release_mps']:.2f} m/s  ({r_wrist['ball_proxy_kmh']:.1f} km/h)")
        print(f"   Peak near release: {r_wrist['peak_speed_near_release_mps']:.2f} m/s")
        print(f"   Peak angular vel : {r_wrist['peak_angular_vel_rads']:.2f} rad/s")


# =============================================================================
# ── SECTION 12 : DATASET HELPERS
# =============================================================================

def _safe_read_excel(path: str, expected_cols: list) -> pd.DataFrame:
    """
    Read an existing dataset Excel file.
    If missing, returns an empty DataFrame with the expected columns.
    If present but empty, returns it as-is.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=expected_cols)
    try:
        df = pd.read_excel(path, dtype=str)   # read as str to avoid type coercion
        # Coerce back to numeric where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        return df
    except Exception as e:
        print(f"  ⚠️  Could not read {path}: {e} — starting fresh.")
        return pd.DataFrame(columns=expected_cols)


def _safe_write_excel(df: pd.DataFrame, path: str):
    """
    Write a DataFrame to Excel safely.
    Creates the parent directory if it doesn't exist.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        df.to_excel(path, index=False, engine="openpyxl")
        print(f"  💾  Saved → {path}  ({len(df)} rows)")
    except PermissionError:
        fallback = path.replace(".xlsx", f"_backup_{datetime.datetime.now().strftime('%H%M%S')}.xlsx")
        df.to_excel(fallback, index=False, engine="openpyxl")
        print(f"  ⚠️  {path} is open in Excel — saved to fallback: {fallback}")


def _drop_existing_trial(df: pd.DataFrame, trial_id: str) -> pd.DataFrame:
    """Remove all rows for `trial_id` so we can replace them cleanly."""
    if "trial_id" in df.columns:
        return df[df["trial_id"].astype(str) != str(trial_id)].reset_index(drop=True)
    return df


# =============================================================================
# ── SECTION 13 : FRAME DATASET WRITER
# =============================================================================
# Frame dataset column mapping
# ─────────────────────────────────────────────────────────────────────────────
# trial_id              → video stem (e.g. "geenod")
# view_mode             → VIEW_MODE setting constant
# frame                 → df_xy["frame"]
# left_ankle_y          → df_xy["left_ankle_y"]          (raw pixel)
# right_ankle_y         → df_xy["right_ankle_y"]         (raw pixel)
# left_ankle_y_sm       → df_xy["left_ankle_y_sm"]       (smoothed, added by head module)
# right_ankle_y_sm      → df_xy["right_ankle_y_sm"]      (smoothed, added by head module)
# bowling_shoulder_x/y  → df_xy["{arm}_shoulder_x/y"]
# bowling_elbow_x/y     → df_xy["{arm}_elbow_x/y"]
# bowling_wrist_x/y     → df_xy["wrist_x_sm"] / df_xy["wrist_y_sm"]  (stabilised+smoothed)
# elbow_angle_deg       → from elbow module frame_to_angle dict
# front_hip_x/y         → df_xy["{kside}_hip_x/y"]
# front_knee_x/y        → df_xy["{kside}_knee_x/y"]
# front_ankle_x/y       → df_xy["{front_ankle}_x/y"]
# front_knee_angle_deg  → from knee module frame_to_knee_angle dict
# head_x / head_y       → df_xy["head_x"] / df_xy["head_y"]  (set by head module)
# head_to_front_dx_cm   → df_xy["videoDx_cm"] (set by head module, NaN outside FFC→release)
# head_to_front_dy_cm   → computed per-frame using head_y − front_ankle_y, scaled
# head_to_front_d_cm    → Euclidean of Dx and Dy
# wrist_speed_m_s       → df_xy["wrist_speed_mps_sm"]
# forearm_angle_deg     → df_xy["forearm_angle_rad"] converted to degrees
# wrist_angular_vel_deg_s → df_xy["wrist_angular_vel_rads_sm"] converted to deg/s
# ─────────────────────────────────────────────────────────────────────────────

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


def build_frame_dataset(
        trial_id, df_xy, fps, cm_per_pixel,
        bowling_arm,     # "right" or "left"
        kside,           # "left" or "right"  (front knee side)
        front_ankle,     # "left_ankle" or "right_ankle"
        frame_to_angle,          # dict frame→elbow_angle (from elbow module), may be None
        frame_to_knee_angle,     # dict frame→knee_angle (from knee module), may be None
) -> pd.DataFrame:
    """
    Assembles one row per frame from the already-computed df_xy columns.
    All signal columns were written into df_xy by the individual modules,
    so this function purely selects and renames them — no re-computation.
    """
    n = len(df_xy)

    # Guard: smoothed ankle columns may not exist if head module failed
    for col in ("left_ankle_y_sm", "right_ankle_y_sm"):
        if col not in df_xy.columns:
            df_xy[col] = np.nan
    # Guard: head / Dx columns may not exist if head module failed
    for col in ("head_x", "head_y", "videoDx_cm"):
        if col not in df_xy.columns:
            df_xy[col] = np.nan

    rows = []
    for i in range(n):
        frame_no = int(df_xy.loc[i, "frame"])

        # ── Elbow angle ──────────────────────────────────────────────────────
        elbow_ang = frame_to_angle.get(frame_no, np.nan) if frame_to_angle else np.nan

        # ── Knee angle ───────────────────────────────────────────────────────
        knee_ang  = frame_to_knee_angle.get(frame_no, np.nan) if frame_to_knee_angle else np.nan

        # ── Head→frontfoot vertical offset (Dy) ──────────────────────────────
        hx  = float(df_xy.loc[i, "head_x"])
        hy  = float(df_xy.loc[i, "head_y"])
        fax = float(df_xy.loc[i, f"{front_ankle}_x"])
        fay = float(df_xy.loc[i, f"{front_ankle}_y"])
        if all(np.isfinite(v) for v in (hx, hy, fax, fay)):
            h_dx_cm = float(df_xy.loc[i, "videoDx_cm"]) if np.isfinite(df_xy.loc[i, "videoDx_cm"]) else (hx - fax) * cm_per_pixel
            h_dy_cm = (hy - fay) * cm_per_pixel
            h_d_cm  = float(np.hypot(h_dx_cm, h_dy_cm))
        else:
            h_dx_cm = h_dy_cm = h_d_cm = np.nan

        # ── Forearm angle: radians → degrees ─────────────────────────────────
        fa_rad = float(df_xy.loc[i, "forearm_angle_rad"]) if "forearm_angle_rad" in df_xy.columns else np.nan
        fa_deg = np.degrees(fa_rad) if np.isfinite(fa_rad) else np.nan

        # ── Angular velocity: rad/s → deg/s ──────────────────────────────────
        av_rads = float(df_xy.loc[i, "wrist_angular_vel_rads_sm"]) if "wrist_angular_vel_rads_sm" in df_xy.columns else np.nan
        av_degs = np.degrees(av_rads) if np.isfinite(av_rads) else np.nan

        rows.append({
            "trial_id":                  trial_id,
            "view_mode":                 VIEW_MODE,
            "frame":                     frame_no,
            # ankle raw + smoothed
            "left_ankle_y":              float(df_xy.loc[i, "left_ankle_y"]),
            "right_ankle_y":             float(df_xy.loc[i, "right_ankle_y"]),
            "left_ankle_y_sm":           float(df_xy.loc[i, "left_ankle_y_sm"]),
            "right_ankle_y_sm":          float(df_xy.loc[i, "right_ankle_y_sm"]),
            # bowling arm joints (smoothed wrist)
            "bowling_shoulder_x":        float(df_xy.loc[i, f"{bowling_arm}_shoulder_x"]),
            "bowling_shoulder_y":        float(df_xy.loc[i, f"{bowling_arm}_shoulder_y"]),
            "bowling_elbow_x":           float(df_xy.loc[i, f"{bowling_arm}_elbow_x"]),
            "bowling_elbow_y":           float(df_xy.loc[i, f"{bowling_arm}_elbow_y"]),
            "bowling_wrist_x":           float(df_xy.loc[i, "wrist_x_sm"]) if "wrist_x_sm" in df_xy.columns else float(df_xy.loc[i, f"{bowling_arm}_wrist_x"]),
            "bowling_wrist_y":           float(df_xy.loc[i, "wrist_y_sm"]) if "wrist_y_sm" in df_xy.columns else float(df_xy.loc[i, f"{bowling_arm}_wrist_y"]),
            # elbow angle
            "elbow_angle_deg":           elbow_ang,
            # front leg joints
            "front_hip_x":               float(df_xy.loc[i, f"{kside}_hip_x"]),
            "front_hip_y":               float(df_xy.loc[i, f"{kside}_hip_y"]),
            "front_knee_x":              float(df_xy.loc[i, f"{kside}_knee_x"]),
            "front_knee_y":              float(df_xy.loc[i, f"{kside}_knee_y"]),
            "front_ankle_x":             fax,
            "front_ankle_y":             fay,
            "front_knee_angle_deg":      knee_ang,
            # head position
            "head_x":                    hx,
            "head_y":                    hy,
            "head_to_front_dx_cm":       h_dx_cm,
            "head_to_front_dy_cm":       h_dy_cm,
            "head_to_front_d_cm":        h_d_cm,
            # wrist velocity
            "wrist_speed_m_s":           float(df_xy.loc[i, "wrist_speed_mps_sm"]) if "wrist_speed_mps_sm" in df_xy.columns else np.nan,
            "forearm_angle_deg":         fa_deg,
            "wrist_angular_velocity_deg_s": av_degs,
        })

    return pd.DataFrame(rows, columns=FRAME_COLS)


def write_frame_dataset(df_new_frames: pd.DataFrame, trial_id: str, path: str):
    """Upsert frame rows for `trial_id` into the frames Excel dataset."""
    print(f"\n📋 Updating frame dataset → {path}")
    df_existing = _safe_read_excel(path, FRAME_COLS)
    df_existing = _drop_existing_trial(df_existing, trial_id)
    df_combined = pd.concat([df_existing, df_new_frames], ignore_index=True)
    _safe_write_excel(df_combined, path)


# =============================================================================
# ── SECTION 14 : MASTER DATASET WRITER
# =============================================================================
# Master dataset column mapping
# ─────────────────────────────────────────────────────────────────────────────
# trial_id                    → video stem
# fps                         → fps from extract_keypoints
# bowling_arm                 → hand (R/L)
# view_mode                   → VIEW_MODE constant
# release_frame               → release_frame from detect_release_frame
# last5_steps_frame           → str(r_cadence["last5_frames"])
# last5_step_intervals_s      → str(r_cadence["intervals_s"])
# step_duration_mean_s        → mean of intervals_s
# step_duration_std_s         → std of intervals_s
# step_duration_cv            → std / mean  (coefficient of variation)
# final5_total_duration_s     → r_cadence["duration_s"]
# stride_duration_s           → r_stride["stride_duration_s"]
# stride_length_m             → r_stride["stride_m"]
# bfc_frame                   → r_stride["bfc_frame"]
# ffc_frame                   → r_stride["ffc_frame"]
# arm_back_frame              → NOT COMPUTED by pipeline — left NaN
#                               (would require detecting the arm backswing peak,
#                                a separate detection step not in current pipeline)
# release_frame.1             → duplicate of release_frame (template has two)
# elbow_angle_arm_back_deg    → NOT COMPUTED — left NaN (arm_back_frame unknown)
# elbow_angle_release_deg     → elbow angle at release frame from df_elbow
# elbow_extension_deg         → NOT COMPUTED — left NaN (no arm_back reference)
# knee_angle_ffc_deg          → knee angle at FFC frame from df_knee
# knee_angle_release_deg      → knee angle at module-detected release from df_knee
# head_dx_ffc_cm              → r_head["ffc_offset"]["Dx_cm"]
# head_dy_ffc_cm              → r_head["ffc_offset"]["Dy_cm"]
# head_d_ffc_cm               → r_head["ffc_offset"]["D_cm"]
# head_dx_bfc_cm              → r_head["bfc_offset"]["Dx_cm"]  (NaN if not detected)
# head_dy_bfc_cm              → r_head["bfc_offset"]["Dy_cm"]
# head_d_bfc_cm               → r_head["bfc_offset"]["D_cm"]
# peak_wrist_speed_m_s        → r_wrist["peak_speed_near_release_mps"]
# wrist_speed_at_release_m_s  → r_wrist["speed_at_release_mps"]
# ─────────────────────────────────────────────────────────────────────────────

MASTER_COLS = [
    "trial_id", "fps", "bowling_arm", "view_mode", "release_frame",
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


def build_master_row(
        trial_id, fps, hand, release_frame,
        r_cadence, r_stride, r_elbow, r_knee, r_head, r_wrist,
) -> dict:
    """Construct a single master-row dict from all module results."""

    # ── Cadence ──────────────────────────────────────────────────────────────
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

    # ── Stride ───────────────────────────────────────────────────────────────
    stride_dur = r_stride["stride_duration_s"] if r_stride else np.nan
    stride_len = r_stride["stride_m"]          if r_stride else np.nan
    bfc_frame  = r_stride["bfc_frame"]         if r_stride else np.nan
    ffc_frame  = r_stride["ffc_frame"]         if r_stride else np.nan

    # ── Elbow ────────────────────────────────────────────────────────────────
    elbow_at_rel = np.nan
    if r_elbow and r_elbow["frame_to_angle"]:
        elbow_at_rel = r_elbow["frame_to_angle"].get(release_frame, np.nan)

    # ── Knee ─────────────────────────────────────────────────────────────────
    knee_at_ffc = knee_at_rel = np.nan
    if r_knee:
        df_knee  = r_knee["df_knee"]
        ffc_kidx = r_knee["ffc_idx"]
        rel_kidx = r_knee["release_idx"]
        ftka     = r_knee.get("frame_to_knee_angle", {})
        if ffc_kidx is not None:
            ffc_frm = int(df_knee.loc[ffc_kidx, "frame"])
            knee_at_ffc = ftka.get(ffc_frm, np.nan)
        if rel_kidx is not None:
            rel_frm = int(df_knee.loc[rel_kidx, "frame"])
            knee_at_rel = ftka.get(rel_frm, np.nan)

    # ── Head ─────────────────────────────────────────────────────────────────
    hd_ffc = r_head["ffc_offset"] if r_head else None
    hd_bfc = r_head["bfc_offset"] if r_head else None

    return {
        "trial_id":                  trial_id,
        "fps":                       fps,
        "bowling_arm":               hand,
        "view_mode":                 VIEW_MODE,
        "release_frame":             release_frame,
        "last5_steps_frame":         last5_str,
        "last5_step_intervals_s":    intv_str,
        "step_duration_mean_s":      mean_i,
        "step_duration_std_s":       std_i,
        "step_duration_cv":          cv_i,
        "final5_total_duration_s":   total_dur,
        "stride_duration_s":         stride_dur,
        "stride_length_m":           stride_len,
        "bfc_frame":                 bfc_frame,
        "ffc_frame":                 ffc_frame,
        "arm_back_frame":            np.nan,          # NOT computed — see note above
        "release_frame.1":           release_frame,
        "elbow_angle_arm_back_deg":  np.nan,          # NOT computed — arm_back unknown
        "elbow_angle_release_deg":   elbow_at_rel,
        "elbow_extension_deg":       np.nan,          # NOT computed — arm_back unknown
        "knee_angle_ffc_deg":        knee_at_ffc,
        "knee_angle_release_deg":    knee_at_rel,
        "head_dx_ffc_cm":            hd_ffc["Dx_cm"] if hd_ffc else np.nan,
        "head_dy_ffc_cm":            hd_ffc["Dy_cm"] if hd_ffc else np.nan,
        "head_d_ffc_cm":             hd_ffc["D_cm"]  if hd_ffc else np.nan,
        "head_dx_bfc_cm":            hd_bfc["Dx_cm"] if hd_bfc else np.nan,
        "head_dy_bfc_cm":            hd_bfc["Dy_cm"] if hd_bfc else np.nan,
        "head_d_bfc_cm":             hd_bfc["D_cm"]  if hd_bfc else np.nan,
        "peak_wrist_speed_m_s":      r_wrist["peak_speed_near_release_mps"] if r_wrist else np.nan,
        "wrist_speed_at_release_m_s": r_wrist["speed_at_release_mps"]      if r_wrist else np.nan,
    }


def write_master_dataset(master_row: dict, trial_id: str, path: str):
    """Upsert one master row for `trial_id` into the master Excel dataset."""
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

    hand, knee_side   = get_user_inputs()
    bowling_wrist     = "right_wrist" if hand == "R" else "left_wrist"
    bowling_arm_name  = "right"       if hand == "R" else "left"
    print(f"🎯 Bowling wrist: {bowling_wrist}")

    OUT_DIR  = setup_output_dir(VIDEO_PATH, BASE_OUT_DIR)
    trial_id = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    print(f"📁 Output folder : {OUT_DIR}")
    print(f"🆔 Trial ID      : {trial_id}")

    pixels_per_meter, meters_per_pixel, cm_per_pixel = stump_calibration(VIDEO_PATH)

    print(f"\n⚙️  Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    df_xy, df_conf, fps, width, height, total_frames = extract_keypoints(VIDEO_PATH, model)

    release_idx, release_frame, _ = detect_release_frame(df_xy, bowling_wrist)
    print(f"\n🏏 Release frame: {release_frame}  ({release_frame/fps:.2f}s)")

    r_cadence = run_step_cadence(df_xy, fps, release_idx, release_frame, bowling_wrist, VIDEO_PATH, width, height)
    r_stride  = run_delivery_stride(df_xy, fps, release_idx, release_frame, bowling_wrist, meters_per_pixel, VIDEO_PATH, width, height)
    r_elbow   = run_elbow_flexion(df_xy, fps, release_idx, release_frame, bowling_wrist, VIDEO_PATH, width, height)
    r_knee    = run_knee_flexion(df_xy, df_conf, fps, release_idx, release_frame, knee_side, VIDEO_PATH, width, height)
    r_head    = run_head_position(df_xy, fps, release_idx, release_frame, bowling_wrist, cm_per_pixel, pixels_per_meter, VIDEO_PATH, width, height)
    r_wrist   = run_wrist_velocity(df_xy, fps, release_idx, release_frame, bowling_wrist, meters_per_pixel, VIDEO_PATH, width, height)

    print_summary(hand, release_frame, fps, r_stride, r_cadence, r_elbow, r_knee, r_head, r_wrist)

    # ── Derive helper values needed by dataset builders ───────────────────────
    # Front knee side string ("left"/"right")
    kside = r_knee["kside"] if r_knee and "kside" in r_knee else ("left" if knee_side == "L" else "right")

    # Front ankle column name (from head module, else infer)
    if r_head and "front_ankle" in r_head:
        front_ankle = r_head["front_ankle"]
    else:
        # Fallback: use same logic as head module
        front_ankle = "left_ankle" if bowling_arm_name == "right" else "right_ankle"

    frame_to_angle      = r_elbow["frame_to_angle"]      if r_elbow else None
    frame_to_knee_angle = r_knee["frame_to_knee_angle"]  if r_knee  else None

    # ── Build and write FRAME dataset ────────────────────────────────────────
    print("\n" + "─"*60)
    print("WRITING DATASETS")
    print("─"*60)

    df_frames = build_frame_dataset(
        trial_id     = trial_id,
        df_xy        = df_xy,
        fps          = fps,
        cm_per_pixel = cm_per_pixel,
        bowling_arm  = bowling_arm_name,
        kside        = kside,
        front_ankle  = front_ankle,
        frame_to_angle       = frame_to_angle,
        frame_to_knee_angle  = frame_to_knee_angle,
    )
    write_frame_dataset(df_frames, trial_id, FRAMES_DATASET_PATH)

    # ── Build and write MASTER dataset ───────────────────────────────────────
    master_row = build_master_row(
        trial_id      = trial_id,
        fps           = fps,
        hand          = hand,
        release_frame = release_frame,
        r_cadence     = r_cadence,
        r_stride      = r_stride,
        r_elbow       = r_elbow,
        r_knee        = r_knee,
        r_head        = r_head,
        r_wrist       = r_wrist,
    )
    write_master_dataset(master_row, trial_id, MASTER_DATASET_PATH)

    print("\n✅ Both datasets updated successfully.")
    print(f"   Frame dataset : {os.path.abspath(FRAMES_DATASET_PATH)}")
    print(f"   Master dataset: {os.path.abspath(MASTER_DATASET_PATH)}")


if __name__ == "__main__":
    main()