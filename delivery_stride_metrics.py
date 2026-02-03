
import cv2
import pandas as pd
import numpy as np
import os
from ultralytics import YOLO
from scipy.signal import savgol_filter, find_peaks

# =========================
# SETTINGS
# =========================
VIDEO_PATH = "video.mp4"
MODEL_PATH = "yolov8n-pose.pt"
OUTPUT_DIR = "output"
ANNOTATED_VIDEO = f"{OUTPUT_DIR}/annotated_delivery_stride.mp4"

STUMP_HEIGHT_M = 0.711
SMOOTH_WINDOW = 7
SMOOTH_POLY = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# USER INPUT
# =========================
bowling_hand = input("Enter bowling hand (R/L): ").strip().upper()
if bowling_hand not in ["R", "L"]:
    raise ValueError("❌ Enter only R or L")

BOWLING_WRIST = "right_wrist" if bowling_hand == "R" else "left_wrist"
OTHER_WRIST   = "left_wrist" if bowling_hand == "R" else "right_wrist"

print(f"🎯 Bowling hand: {BOWLING_WRIST}")

# =========================
# LOAD MODEL & VIDEO
# =========================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise IOError("❌ Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"🎥 Video: {total_frames} frames @ {fps:.2f} FPS")

# =========================
# KEYPOINT SETUP
# =========================
KEYPOINT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

SELECTED = ["left_ankle","right_ankle","left_wrist","right_wrist"]

# =========================
# EXTRACT KEYPOINTS
# =========================
rows = []
frame_no = 0

print("⏳ Extracting keypoints...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1
    row = {"frame": frame_no}

    for k in SELECTED:
        row[f"{k}_x"] = np.nan
        row[f"{k}_y"] = np.nan

    results = model.predict(frame, verbose=False)

    if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        kps_all = results[0].keypoints.xy.cpu().numpy()
        boxes = results[0].boxes.xywh.cpu().numpy()

        # select main bowler
        areas = boxes[:,2] * boxes[:,3]
        idx = np.argmax(areas)
        kps = kps_all[idx]

        for i, name in enumerate(KEYPOINT_NAMES):
            if name in SELECTED:
                row[f"{name}_x"] = kps[i,0]
                row[f"{name}_y"] = kps[i,1]

    rows.append(row)

cap.release()

df = pd.DataFrame(rows)
df.interpolate(inplace=True)
df.bfill(inplace=True)
df.ffill(inplace=True)

# =========================
# SMOOTH SIGNALS
# =========================
df["left_ankle_y_s"]  = savgol_filter(df["left_ankle_y"],  SMOOTH_WINDOW, SMOOTH_POLY)
df["right_ankle_y_s"] = savgol_filter(df["right_ankle_y"], SMOOTH_WINDOW, SMOOTH_POLY)
df["wrist_y_s"] = savgol_filter(df[f"{BOWLING_WRIST}_y"], SMOOTH_WINDOW, SMOOTH_POLY)

# =========================
# BALL RELEASE (WRIST PEAK)
# =========================
release_idx = int(df["wrist_y_s"].idxmin())
release_frame = int(df.loc[release_idx,"frame"])
print(f"🏏 Ball release frame: {release_frame}")

# # =========================
# # FRONT & BACK FOOT WITH REFINED CONTACT
# # =========================
# # Front foot = opposite of bowling hand
# f_side = "left" if BOWLING_WRIST == "right_wrist" else "right"
# b_side = "right" if f_side == "left" else "left"

# left_y = df["left_ankle_y_s"].to_numpy()
# right_y = df["right_ankle_y_s"].to_numpy()

# # Detect local maxima (foot ground contacts)
# left_peaks, _ = find_peaks(left_y, distance=3)
# right_peaks, _ = find_peaks(right_y, distance=3)

# # Function to refine contact frame: move backward to first frame foot stops descending
# def refine_contact(signal, peak_idx, max_back=3):
#     idx = peak_idx
#     for i in range(max_back):
#         if idx-i-1 < 0:
#             break
#         if signal[idx-i-1] <= signal[idx-i]:  # foot stopped descending
#             idx = idx-i-1
#             break
#     return idx

# # FFC = front foot contact closest to release
# f_peaks = left_peaks if f_side == "left" else right_peaks
# f_peaks = f_peaks[f_peaks <= release_idx]  # only before or at release
# ffc_idx_raw = f_peaks[-1]
# ffc_idx = refine_contact(left_y if f_side=="left" else right_y, ffc_idx_raw)
# ffc_frame = int(df.loc[ffc_idx, "frame"])

# # BFC = previous contact of back foot
# b_peaks = right_peaks if b_side == "right" else left_peaks
# b_peaks = b_peaks[b_peaks < ffc_idx]  # only peaks before FFC
# bfc_idx_raw = b_peaks[-1]
# bfc_idx = refine_contact(left_y if b_side=="left" else right_y, bfc_idx_raw)
# bfc_frame = int(df.loc[bfc_idx, "frame"])


# =========================
# REFINED CONTACT DETECTION (Velocity Based)
# =========================
f_side = "left" if BOWLING_WRIST == "right_wrist" else "right"
b_side = "right" if f_side == "left" else "left"

def find_actual_contact(df, side, release_idx, search_back_from_release=True):
    y_signal = df[f"{side}_ankle_y_s"].to_numpy()
    
    # Calculate velocity (change in Y per frame)
    # Positive velocity = moving down (increasing Y)
    velocity = np.gradient(y_signal)
    
    # Find peaks where the foot is at its "lowest" point (max Y)
    peaks, _ = find_peaks(y_signal, distance=10)
    
    if search_back_from_release:
        # For FFC: Find the peak closest to but BEFORE release
        valid_peaks = peaks[peaks <= release_idx]
        if len(valid_peaks) == 0: return release_idx # Fallback
        peak_idx = valid_peaks[-1]
    else:
        # For BFC: Find the peak before the FFC
        valid_peaks = peaks[peaks < release_idx]
        if len(valid_peaks) == 0: return max(0, release_idx - 10) # Fallback
        peak_idx = valid_peaks[-1]

    # REFINEMENT: Move backward from the peak to find when velocity first hit ~0
    # This captures the "Initial Contact" rather than "Full Weight Bearing"
    refined_idx = peak_idx
    for i in range(peak_idx, peak_idx - 10, -1):
        if i <= 0: break
        # If velocity is very small, the foot has likely landed
        if velocity[i] < 0.5: # Threshold for 'stopped moving down'
            refined_idx = i
        else:
            break # Foot was still moving fast before this
            
    return refined_idx

# Apply the new logic
ffc_idx = find_actual_contact(df, f_side, release_idx, True)
bfc_idx = find_actual_contact(df, b_side, ffc_idx, False)

ffc_frame = int(df.loc[ffc_idx, "frame"])
bfc_frame = int(df.loc[bfc_idx, "frame"])


# =========================
# STRIDE METRICS
# =========================
bx, by = df.loc[bfc_idx,f"{b_side}_ankle_x"], df.loc[bfc_idx,f"{b_side}_ankle_y"]
fx, fy = df.loc[ffc_idx,f"{f_side}_ankle_x"], df.loc[ffc_idx,f"{f_side}_ankle_y"]

stride_px = np.sqrt((fx-bx)**2 + (fy-by)**2)
duration_s = (ffc_frame - bfc_frame) / fps


# =========================
# STUMP SCALE (Updated with Visual Feedback)
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

# Create a copy so we don't draw over the original data if we need to reset
display_frame = frame.copy()
points = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        # Draw a small green dot where the user clicked
        cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
        # Update the window with the new drawing
        cv2.imshow("Click TOP then BOTTOM of stump", display_frame)
        
        if len(points) == 2:
            print("✅ Two points captured. Press any key to continue.")

cv2.imshow("Click TOP then BOTTOM of stump", display_frame)
cv2.setMouseCallback("Click TOP then BOTTOM of stump", click)

# Wait until the user presses a key AFTER clicking
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) < 2:
    raise ValueError("❌ You must click two points (top and bottom of stump).")

# Calculate scale
scale = STUMP_HEIGHT_M / abs(points[0][1] - points[1][1])
stride_m = stride_px * scale


# =========================
# PRINT RESULTS
# =========================
print("\n🏏 DELIVERY STRIDE METRICS")
print(f"BFC Frame: {bfc_frame}")
print(f"FFC Frame: {ffc_frame}")
print(f"Release Frame: {release_frame}")
print(f"Stride Length: {stride_m:.2f} m")
print(f"Stride Duration: {duration_s:.3f} s")

# =========================
# ANNOTATED VIDEO
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
out = cv2.VideoWriter(
    ANNOTATED_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width,height)
)

frame_no = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1

    if frame_no == bfc_frame:
        cv2.circle(frame,(int(bx),int(by)),18,(0,0,255),-1)
        cv2.putText(frame,"BFC",(50,80),0,2,(0,0,255),4)

    if frame_no == ffc_frame:
        cv2.circle(frame,(int(fx),int(fy)),18,(0,255,0),-1)
        cv2.line(frame,(int(bx),int(by)),(int(fx),int(fy)),(255,255,255),3)
        cv2.putText(frame,"FFC",(50,160),0,2,(0,255,0),4)

    if frame_no == release_frame:
        cv2.putText(frame,"RELEASE",(50,240),0,2,(255,255,0),4)

    out.write(frame)

cap.release()
out.release()

print(f"\n🎥 Annotated video saved: {ANNOTATED_VIDEO}")
