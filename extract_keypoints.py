import cv2
import pandas as pd
import numpy as np
import os
from ultralytics import YOLO
from scipy.signal import savgol_filter, find_peaks

# === SETTINGS ===
VIDEO_PATH = "nipul.mp4"                     # Input video
OUTPUT_CSV = "output/yolo_keypoints_left.csv"     # Output CSV
MODEL_PATH = "yolov8n-pose.pt"               # Model file
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

wrist_signal = df["left_wrist_y"].values
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

# === Annotated & Trimmed Video Output with Step Labels ===
if len(peaks_before_release) >= 5:
    start_frame = last_five_frames[0]
    end_frame = release_frame
    trimmed_output_path = os.path.join(os.path.dirname(OUTPUT_CSV), "yolo_annotated_trimmed_left.mp4")

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
    step_labels = {f: f"Step {i+1}" for i, f in enumerate(last_five_frames)}

    while frame_index <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Get keypoints without visualization
        results = model.predict(frame, verbose=False)
        annotated = frame.copy()  # Use clean frame
        
            # Draw foot contact circle if keypoints detected
        if len(results) > 0 and len(results[0].keypoints) > 0:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            
            # Get ankle positions (indices 15 and 16 for left and right ankle)
            left_ankle = keypoints[15]
            right_ankle = keypoints[16]
            
            # If this is a foot contact frame, draw highlight circle for landing foot
            if frame_index in last_five_frames:
                # Determine landing foot (the one closer to ground, i.e., larger y-coordinate)
                landing_ankle = None
                if not np.isnan(left_ankle[1]) and not np.isnan(right_ankle[1]):
                    landing_ankle = left_ankle if left_ankle[1] > right_ankle[1] else right_ankle
                elif not np.isnan(left_ankle[1]):
                    landing_ankle = left_ankle
                elif not np.isnan(right_ankle[1]):
                    landing_ankle = right_ankle
                
                if landing_ankle is not None:
                    # Create pulsing effect with larger circles
                    circle_phase = (frame_index % 30) / 30.0
                    circle_radius = int(60 + 30 * np.sin(circle_phase * 2 * np.pi))  # Increased base size and pulse range
                    
                    # Draw circles with fade-out effect
                    alpha = 0.7
                    overlay = annotated.copy()
                    
                    center = (int(landing_ankle[0]), int(landing_ankle[1]))
                    # Larger outer glow
                    cv2.circle(overlay, center, circle_radius + 20, (0, 255, 255), -1)  # Increased glow size
                    cv2.circle(overlay, center, circle_radius, (0, 255, 0), 4)  # Thicker circle line
                    
                    # Blend the circles with the original frame
                    cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)        # --- Add stats box with larger text (centered) ---
        def draw_stats_box(img, text_lines, font_scale=1.5):
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 4
            color = (255, 255, 255)
            line_spacing = 60
            
            # Calculate total height and maximum width
            total_height = line_spacing * len(text_lines)
            max_width = 0
            for text in text_lines:
                (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
                max_width = max(max_width, text_width)
            
            # Calculate centered position
            start_x = (width - max_width) // 2
            start_y = 80  # Fixed distance from top
            
            # Draw semi-transparent background for entire stats box
            bg_padding = 20
            overlay = img.copy()
            cv2.rectangle(overlay, 
                        (start_x - bg_padding, start_y - 40),
                        (start_x + max_width + bg_padding, start_y + total_height - 20),
                        (0, 0, 0),
                        -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)  # Blend with 70% opacity
            
            # Draw text
            for i, text in enumerate(text_lines):
                # Get size for this specific text for centering
                (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = (width - text_width) // 2  # Center each line independently
                cv2.putText(img, text, (text_x, start_y + (i * line_spacing)),
                          font, font_scale, color, thickness)

        # --- Annotate step frames with large text and pause ---
        if frame_index in step_labels:
            label = step_labels[frame_index]
            current_step_idx = last_five_frames.index(frame_index)
            current_time = frame_index / fps
            
            # Stats to display
            stats = [
                f"{label} (Frame {frame_index})",
                f"Time: {current_time:.2f}s",
            ]
            
            if current_step_idx > 0:
                time_since_last = (frame_index - last_five_frames[current_step_idx-1]) / fps
                stats.append(f"Time since last step: {time_since_last:.2f}s")
            
            # Draw large centered text for step number
            text = f"STEP {current_step_idx + 1}"
            font_scale = 4.0  # Increased from 3.0
            thickness = 6  # Increased from 5
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x = (width - text_width) // 2
            text_y = height // 3
            
            # Add semi-transparent background for step number
            bg_padding = 40  # Larger padding for the big text
            bg_overlay = annotated.copy()
            cv2.rectangle(bg_overlay, 
                        (text_x - bg_padding, text_y - text_height - bg_padding),
                        (text_x + text_width + bg_padding, text_y + bg_padding),
                        (0, 0, 0),
                        -1)
            # Blend with 60% opacity
            cv2.addWeighted(bg_overlay, 0.6, annotated, 0.4, 0, annotated)
            
            # Draw text with outline for extra visibility
            cv2.putText(annotated, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness + 4)
            cv2.putText(annotated, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 255, 0), thickness)
            
            # Draw stats box centered at top
            draw_stats_box(annotated, stats, 1.8)
            
            # Write the same frame multiple times to create longer pause effect
            for _ in range(int(fps * 1.2)):  # 1.2 second pause (increased from 0.5)
                out.write(annotated)
        else:
            out.write(annotated)

        # --- Annotate release frame ---
        if frame_index == release_frame:
            text = "BALL RELEASE"
            font_scale = 4.0  # Increased from 3.0
            thickness = 6  # Increased from 5
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x = (width - text_width) // 2
            text_y = height // 3
            
            # Add semi-transparent background for release text
            bg_padding = 40
            bg_overlay = annotated.copy()
            cv2.rectangle(bg_overlay, 
                        (text_x - bg_padding, text_y - text_height - bg_padding),
                        (text_x + text_width + bg_padding, text_y + bg_padding),
                        (0, 0, 0),
                        -1)
            # Blend with 60% opacity
            cv2.addWeighted(bg_overlay, 0.6, annotated, 0.4, 0, annotated)
            
            # Draw text with outline for extra visibility
            cv2.putText(annotated, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness + 4)
            cv2.putText(annotated, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 0, 0), thickness)
            
            stats = [
                f"Release Frame: {release_frame}",
                f"Release Time: {release_frame/fps:.2f}s",
                f"Total Duration: {duration:.2f}s"
            ]
            draw_stats_box(annotated, stats, 1.8)
            
            # Longer pause on release frame
            for _ in range(int(fps * 1.5)):  # 1.5 second pause (increased from 0.75)
                out.write(annotated)
        
        # Always show duration info with larger text
        duration_stats = [
            f"Total Steps: 5",
            f"Sequence Duration: {duration:.2f}s",
            f"Avg Step Interval: {duration/4:.2f}s"
        ]
        draw_stats_box(annotated, duration_stats, 1.5)

        if frame_index % 10 == 0:
            print(f"  → Annotated frame {frame_index}/{end_frame}")

        frame_index += 1

    cap.release()
    out.release()
    print(f"\n ✅ Annotated trimmed video with step labels saved to: {trimmed_output_path}")
else:
    print("\n Skipping annotated trimmed video: insufficient step frames detected.")
