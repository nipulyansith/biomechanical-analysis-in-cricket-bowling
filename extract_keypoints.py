import cv2
import pandas as pd
import numpy as np
import os
from ultralytics import YOLO
from scipy.signal import savgol_filter, find_peaks

# === SETTINGS ===
VIDEO_PATH = "video.mp4"                     # Input video
OUTPUT_CSV = "output/yolo_keypoints_right.csv"     # Output CSV
MODEL_PATH = "yolov8n-pose.pt"               # Model file (downloaded or cached)
SMOOTH_WINDOW = 7                            # Must be odd
SMOOTH_POLY = 2

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

# === YOLO Keypoints names (17 keypoints in COCO format) ===
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

# === Ensure output folder exists ===
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

print("⏳ Processing video (extracting YOLO keypoints)...")

# === Loop through video frames ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    data = {"frame": frame_num}

    # Initialize with NaN
    for name in SELECTED_POINTS:
        data[f"{name}_x"] = np.nan
        data[f"{name}_y"] = np.nan

    # Run YOLO inference
    results = model.predict(frame, verbose=False)

    if len(results) > 0 and len(results[0].keypoints) > 0:
        keypoints = results[0].keypoints.xy[0].cpu().numpy()  # shape (17, 2)
        for i, name in enumerate(KEYPOINT_NAMES):
            if name in SELECTED_POINTS:
                data[f"{name}_x"] = keypoints[i, 0]
                data[f"{name}_y"] = keypoints[i, 1]

    rows.append(data)
    if frame_num % 100 == 0:
        print(f"Processed {frame_num}/{total_frames} frames...")

cap.release()

# === Save keypoints CSV ===
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Keypoints extracted for all {total_frames} frames.")
print(f"📁 Saved to: {OUTPUT_CSV}")

# === Post-process: detect release and last 5-step duration ===
df = pd.read_csv(OUTPUT_CSV)
df.fillna(method="ffill", inplace=True)
df.fillna(method="bfill", inplace=True)

# --- Build ankle combined signal (higher y = foot on ground) ---
ankle_signal = np.maximum(df["left_ankle_y"].values, df["right_ankle_y"].values)

# --- Smooth ankle and wrist signals ---
def make_window(n, length):
    w = n if n % 2 == 1 else n + 1
    if w >= length:
        w = length - 1 if (length - 1) % 2 == 1 else length - 2
    return max(3, w)

win = make_window(SMOOTH_WINDOW, len(ankle_signal))
smoothed_ankle = savgol_filter(ankle_signal, window_length=win, polyorder=SMOOTH_POLY)

wrist_signal = df["right_wrist_y"].values
win_wrist = make_window(SMOOTH_WINDOW, len(wrist_signal))
smoothed_wrist = savgol_filter(wrist_signal, window_length=win_wrist, polyorder=SMOOTH_POLY)

# --- Determine release frame (wrist highest point = min y) ---
release_idx = int(np.argmin(smoothed_wrist))
release_frame = int(df.loc[release_idx, "frame"])
print(f"\n Estimated release frame: index {release_idx} (frame {release_frame})")

# --- Detect foot-contact peaks (ankle max y = ground contact) ---
min_distance_frames = max(3, int(0.03 * fps))
peaks, _ = find_peaks(smoothed_ankle, distance=min_distance_frames, prominence=5)
peaks_before_release = peaks[peaks < release_idx]

if len(peaks_before_release) < 5:
    peaks_relaxed, _ = find_peaks(smoothed_ankle, distance=min_distance_frames, prominence=1)
    peaks_before_release = peaks_relaxed[peaks_relaxed < release_idx]

if len(peaks_before_release) < 5:
    print(f" Not enough foot-contact peaks found before release (found {len(peaks_before_release)}).")
else:
    last_five_idx = peaks_before_release[-5:]
    last_five_frames = df.loc[last_five_idx, "frame"].astype(int).tolist()
    contact_times = [f / fps for f in last_five_frames]
    duration = contact_times[-1] - contact_times[0]

    print("\n Foot contact frame numbers (last 5 before release):", last_five_frames)
    print(" Foot contact times (s):", [f"{t:.3f}" for t in contact_times])
    print(f" Duration between first and last of those 5 steps: {duration:.3f} seconds")

# --- Save debug CSV ---
debug_path = os.path.splitext(OUTPUT_CSV)[0] + "_debug.csv"
pd.DataFrame({
    "frame": df["frame"].astype(int),
    "left_ankle_y": df["left_ankle_y"],
    "right_ankle_y": df["right_ankle_y"],
    "ankle_combined": ankle_signal,
    "smoothed_ankle": smoothed_ankle,
    "right_wrist_y": df["right_wrist_y"],
    "smoothed_wrist": smoothed_wrist
}).to_csv(debug_path, index=False)
print(f" Debug signals saved to: {debug_path}")

# === Annotated & Trimmed Video Output ===
if len(peaks_before_release) >= 5:
    start_frame = last_five_frames[0]
    end_frame = release_frame
    trimmed_output_path = os.path.join(os.path.dirname(OUTPUT_CSV), "yolo_annotated_trimmed.mp4")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error reopening video for annotation.")
        exit()

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(trimmed_output_path, fourcc, fps, (width, height))

    print(f"\n Generating annotated trimmed video from frame {start_frame} to {end_frame}...")

    frame_index = start_frame
    while frame_index <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw YOLO pose keypoints
        results = model.predict(frame, verbose=False)
        if len(results) > 0:
            annotated = results[0].plot()  # draws skeletons
        else:
            annotated = frame

        out.write(annotated)
        if frame_index % 10 == 0:
            print(f"  → Annotated frame {frame_index}/{end_frame}")
        frame_index += 1

    cap.release()
    out.release()
    print(f"\n Annotated trimmed video saved to: {trimmed_output_path}")
else:
    print("\n Skipping annotated trimmed video: insufficient step frames detected.")
