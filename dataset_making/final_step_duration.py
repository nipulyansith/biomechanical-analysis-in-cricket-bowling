import cv2
import pandas as pd
import numpy as np
import os
from ultralytics import YOLO
from scipy.signal import savgol_filter, find_peaks

# === SETTINGS ===
VIDEO_PATH = "data/seba.MOV"
OUTPUT_CSV = "output/yolo_keypoints_left.csv"
MODEL_PATH = "yolov8l-pose.pt"
SMOOTH_WINDOW = 7
SMOOTH_POLY = 2
BOWLING_ARM = "left"
OUTPUT_SCALE = 0.5

# === Load YOLO Pose Model ===
model = YOLO(MODEL_PATH)

# === Open Video ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
print(f"🎥 Total frames in video: {total_frames}, FPS: {fps:.2f}")

frame_num = 0
rows = []

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

SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

KEYPOINT_COLORS = {
    "left_wrist":     (0, 255, 255),
    "right_wrist":    (0, 200, 255),
    "left_elbow":     (0, 255, 150),
    "right_elbow":    (0, 150, 255),
    "left_shoulder":  (255, 100, 0),
    "right_shoulder": (255, 0, 100),
    "left_hip":       (200, 0, 255),
    "right_hip":      (150, 0, 200),
    "left_knee":      (0, 200, 100),
    "right_knee":     (0, 100, 200),
    "left_ankle":     (100, 255, 0),
    "right_ankle":    (50, 200, 0),
}

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

print("⏳ Processing video (extracting YOLO keypoints)...")

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

    if len(results) > 0 and len(results[0].keypoints) > 0:
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        for i, name in enumerate(KEYPOINT_NAMES):
            if name in SELECTED_POINTS:
                data[f"{name}_x"] = keypoints[i, 0]
                data[f"{name}_y"] = keypoints[i, 1]

    rows.append(data)
    if frame_num % 100 == 0:
        print(f"Processed {frame_num}/{total_frames} frames...")

cap.release()

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Keypoints extracted for all {total_frames} frames.")
print(f"📁 Saved to: {OUTPUT_CSV}")

# =============================================================================
# POST-PROCESSING  —  detect release + last 5 steps
# =============================================================================
df = pd.read_csv(OUTPUT_CSV)

# ── FIX 1: use ffill/bfill only on rows that actually had data,
#    keeping leading-NaN rows (before the bowler enters frame) as NaN.
#    We fill per-column to avoid propagating values across the NaN gap.
df = df.ffill().bfill()

def make_window(n, length):
    w = n if n % 2 == 1 else n + 1
    if w >= length:
        w = length - 1 if (length - 1) % 2 == 1 else length - 2
    return max(3, w)

# ── FIX 2: determine release frame only on frames that have valid wrist data.
wrist_col = "right_wrist_y" if BOWLING_ARM.lower() == "right" else "left_wrist_y"

# Restrict to rows where the wrist was actually detected (not backfilled from NaN)
valid_wrist_mask = pd.read_csv(OUTPUT_CSV)[wrist_col].notna()
df_valid_wrist = df[valid_wrist_mask].copy().reset_index(drop=True)

wrist_signal_valid = df_valid_wrist[wrist_col].values
win_wrist = make_window(SMOOTH_WINDOW, len(wrist_signal_valid))
smoothed_wrist_valid = savgol_filter(wrist_signal_valid, window_length=win_wrist, polyorder=SMOOTH_POLY)

# Arm highest = wrist y minimum = release moment
release_local_idx = int(np.argmin(smoothed_wrist_valid))
release_frame = int(df_valid_wrist.loc[release_local_idx, "frame"])
print(f"\n🎯 Estimated release frame ({BOWLING_ARM}-arm wrist minimum): frame {release_frame}")

# ── FIX 3: per-foot peak detection on valid ankle frames only.
#    Do NOT merge both feet into a single signal — that loses identity.
raw_df = pd.read_csv(OUTPUT_CSV)
valid_ankle_mask = raw_df["right_ankle_x"].notna() | raw_df["left_ankle_x"].notna()
df_ankle = df[valid_ankle_mask].copy().reset_index(drop=True)

right_ankle_y = df_ankle["right_ankle_y"].fillna(0).values
left_ankle_y  = df_ankle["left_ankle_y"].fillna(0).values
ankle_frames  = df_ankle["frame"].values

# ── FIX 4: minimum distance between peaks.
#    Cricket run-up cadence is ~3–5 steps/second → minimum ~0.2 s between steps.
#    Use 20 % of fps (≥ 6 frames at 30 fps) as a safe floor.
min_dist_frames = max(6, int(0.20 * fps))

win_ankle = make_window(SMOOTH_WINDOW, len(right_ankle_y))
smoothed_right = savgol_filter(right_ankle_y, window_length=win_ankle, polyorder=SMOOTH_POLY)
smoothed_left  = savgol_filter(left_ankle_y,  window_length=win_ankle, polyorder=SMOOTH_POLY)

right_peaks, _ = find_peaks(smoothed_right, distance=min_dist_frames, prominence=30)
left_peaks,  _ = find_peaks(smoothed_left,  distance=min_dist_frames, prominence=30)

# Build unified event list: (frame, foot)
step_events = (
    [(int(ankle_frames[p]), "right") for p in right_peaks] +
    [(int(ankle_frames[p]), "left")  for p in left_peaks]
)
step_events.sort(key=lambda x: x[0])

# Keep only steps that occur at or before release
step_events_before_release = [s for s in step_events if s[0] <= release_frame]

print(f"\n📋 All foot-contact events detected before release:")
for s in step_events_before_release:
    print(f"   Frame {s[0]:>4d} — {s[1]} foot")

# ── Alternating filter: walk backwards, accept each step only if
#    it alternates from the previously accepted foot.
def pick_alternating_last5(events):
    selected = []
    last_foot = None
    for evt in reversed(events):
        foot = evt[1]
        if foot != last_foot:
            selected.append(evt)
            last_foot = foot
            if len(selected) == 5:
                break
    return list(reversed(selected))

last_five = pick_alternating_last5(step_events_before_release)

if len(last_five) < 5:
    print(f"\n⚠️  Only {len(last_five)} alternating steps found before release.")
else:
    last_five_frames = [s[0] for s in last_five]
    foot_labels      = [s[1] for s in last_five]
    contact_times    = [f / fps for f in last_five_frames]
    duration         = contact_times[-1] - contact_times[0]

    print(f"\n✅ Foot contact frame numbers (last 5 before release): {last_five_frames}")
    print(f"   Foot per step: {[f.upper() for f in foot_labels]}")
    print(f"   Contact times (s): {[f'{t:.3f}' for t in contact_times]}")
    print(f"   Duration (first→last of 5): {duration:.3f} s")
    print(f"   Avg step interval: {duration / 4:.3f} s")

# ── Save debug CSV ──
debug_path = os.path.splitext(OUTPUT_CSV)[0] + "_debug.csv"
pd.DataFrame({
    "frame":            df["frame"].astype(int),
    "left_ankle_y":     df["left_ankle_y"],
    "right_ankle_y":    df["right_ankle_y"],
    "bowling_wrist_y":  df[wrist_col],
}).to_csv(debug_path, index=False)
print(f"\n📁 Debug signals saved to: {debug_path}")


# =============================================================================
# ANNOTATED VIDEO
# =============================================================================
def draw_keypoints_on_frame(img, keypoints_xy, scale=1.0):
    kp_dict = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        x, y = keypoints_xy[i]
        if x > 0 and y > 0:
            kp_dict[name] = (int(x * scale), int(y * scale))

    for (p1_name, p2_name) in SKELETON_CONNECTIONS:
        if p1_name in kp_dict and p2_name in kp_dict:
            c1 = KEYPOINT_COLORS.get(p1_name, (200, 200, 200))
            c2 = KEYPOINT_COLORS.get(p2_name, (200, 200, 200))
            bone_color = tuple((a + b) // 2 for a, b in zip(c1, c2))
            cv2.line(img, kp_dict[p1_name], kp_dict[p2_name], bone_color,
                     max(1, int(3 * scale)), cv2.LINE_AA)

    dot_radius    = max(4, int(8 * scale))
    font_scale    = max(0.3, 0.55 * scale)
    font_thickness = max(1, int(1.5 * scale))
    for name, (px, py) in kp_dict.items():
        color = KEYPOINT_COLORS.get(name, (255, 255, 255))
        cv2.circle(img, (px, py), dot_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (px, py), dot_radius, color, -1, cv2.LINE_AA)
        short = name.replace("left_", "L_").replace("right_", "R_")
        cv2.putText(img, short, (px + dot_radius + 2, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)


if len(last_five) >= 5:
    start_frame = last_five_frames[0]
    end_frame   = release_frame
    trimmed_output_path = os.path.join(os.path.dirname(OUTPUT_CSV),
                                       "yolo_annotated_trimmed_left.mp4")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error reopening video for annotation.")
        exit()

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_width   = int(orig_width  * OUTPUT_SCALE)
    out_height  = int(orig_height * OUTPUT_SCALE)
    width, height = out_width, out_height

    print(f"\n🎬 Output video: {out_width}×{out_height}  (scale={OUTPUT_SCALE})")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(trimmed_output_path, fourcc, fps, (out_width, out_height))

    step_label_map = {f: f"Step {i+1}" for i, f in enumerate(last_five_frames)}
    step_foot_map  = {last_five_frames[i]: foot_labels[i] for i in range(5)}

    def draw_stats_box(img, text_lines, font_scale=1.5, vertical_position=None):
        font         = cv2.FONT_HERSHEY_SIMPLEX
        fs           = font_scale * OUTPUT_SCALE
        thickness    = max(1, int(4 * OUTPUT_SCALE))
        color        = (255, 255, 255)
        line_spacing = int(70 * OUTPUT_SCALE)
        total_height = line_spacing * len(text_lines)
        max_width    = 0
        for text in text_lines:
            (tw, _), _ = cv2.getTextSize(text, font, fs, thickness)
            max_width = max(max_width, tw)
        start_x = (width - max_width) // 2
        start_y = (int(80 * OUTPUT_SCALE) if vertical_position is None
                   else vertical_position - (total_height // 2))
        bg_pad  = int(20 * OUTPUT_SCALE)
        overlay = img.copy()
        cv2.rectangle(overlay,
                      (start_x - bg_pad, start_y - int(40 * OUTPUT_SCALE)),
                      (start_x + max_width + bg_pad, start_y + total_height - int(20 * OUTPUT_SCALE)),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        for i, text in enumerate(text_lines):
            (tw, _), _ = cv2.getTextSize(text, font, fs, thickness)
            tx = (width - tw) // 2
            cv2.putText(img, text, (tx, start_y + i * line_spacing), font, fs, color, thickness)

    print(f"   Annotating frames {start_frame} → {end_frame} ...")
    frame_index = start_frame

    while frame_index <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        results  = model.predict(frame, verbose=False)
        annotated = cv2.resize(frame, (out_width, out_height))

        if len(results) > 0 and len(results[0].keypoints) > 0:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            draw_keypoints_on_frame(annotated, keypoints, scale=OUTPUT_SCALE)

            left_ankle  = keypoints[15]
            right_ankle = keypoints[16]

            if frame_index in last_five_frames:
                dominant = step_foot_map[frame_index].upper()
                if dominant == "LEFT" and left_ankle[1] > 0:
                    landing_ankle = left_ankle
                elif dominant == "RIGHT" and right_ankle[1] > 0:
                    landing_ankle = right_ankle
                else:
                    landing_ankle = None
                    if left_ankle[1] > 0 and right_ankle[1] > 0:
                        landing_ankle = (left_ankle if left_ankle[1] > right_ankle[1]
                                         else right_ankle)
                    elif left_ankle[1] > 0:
                        landing_ankle = left_ankle
                    elif right_ankle[1] > 0:
                        landing_ankle = right_ankle

                if landing_ankle is not None:
                    circle_phase  = (frame_index % 30) / 30.0
                    base_r        = int(60 * OUTPUT_SCALE)
                    pulse_r       = int(30 * OUTPUT_SCALE)
                    circle_radius = int(base_r + pulse_r * np.sin(circle_phase * 2 * np.pi))
                    overlay       = annotated.copy()
                    center        = (int(landing_ankle[0] * OUTPUT_SCALE),
                                     int(landing_ankle[1] * OUTPUT_SCALE))
                    cv2.circle(overlay, center, circle_radius + int(20 * OUTPUT_SCALE),
                               (0, 255, 255), -1)
                    cv2.circle(overlay, center, circle_radius,
                               (0, 255, 0), max(2, int(4 * OUTPUT_SCALE)))
                    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

        if frame_index in step_label_map:
            current_step_idx = last_five_frames.index(frame_index)
            foot_label       = step_foot_map[frame_index].upper()
            stats = [
                f"STEP {current_step_idx + 1} OF 5  ({foot_label} FOOT)",
                f"Frame: {frame_index}",
                f"Time: {frame_index / fps:.2f}s",
            ]
            if current_step_idx > 0:
                time_since_last = (frame_index - last_five_frames[current_step_idx - 1]) / fps
                stats.append(f"Time since last step: {time_since_last:.2f}s")
            draw_stats_box(annotated, stats, 2.2, vertical_position=height // 2)
            for _ in range(int(fps * 1.2)):
                out.write(annotated)
        else:
            out.write(annotated)

        if frame_index == release_frame:
            text   = "BALL RELEASE"
            fs_rel = 4.0 * OUTPUT_SCALE
            th_rel = max(2, int(6 * OUTPUT_SCALE))
            (tw, th_px), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs_rel, th_rel)
            text_x = (width - tw) // 2
            text_y = height // 3
            bg_pad = int(40 * OUTPUT_SCALE)
            bg_ov  = annotated.copy()
            cv2.rectangle(bg_ov,
                          (text_x - bg_pad, text_y - th_px - bg_pad),
                          (text_x + tw + bg_pad, text_y + bg_pad),
                          (0, 0, 0), -1)
            cv2.addWeighted(bg_ov, 0.6, annotated, 0.4, 0, annotated)
            cv2.putText(annotated, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        fs_rel, (0, 0, 0), th_rel + max(2, int(4 * OUTPUT_SCALE)))
            cv2.putText(annotated, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        fs_rel, (255, 0, 0), th_rel)
            stats = [
                f"Release Frame: {release_frame}",
                f"Release Time:  {release_frame / fps:.2f}s",
                f"Total Duration: {duration:.2f}s"
            ]
            draw_stats_box(annotated, stats, 1.8)
            for _ in range(int(fps * 1.5)):
                out.write(annotated)

        duration_stats = [
            f"Total Steps: 5",
            f"Sequence Duration: {duration:.2f}s",
            f"Avg Step Interval: {duration / 4:.2f}s"
        ]
        draw_stats_box(annotated, duration_stats, 1.5)

        if frame_index % 10 == 0:
            print(f"  → Frame {frame_index}/{end_frame}")

        frame_index += 1

    cap.release()
    out.release()
    print(f"\n✅ Annotated video saved to: {trimmed_output_path}")
else:
    print("\n⚠️  Skipping annotated video: insufficient step frames detected.")