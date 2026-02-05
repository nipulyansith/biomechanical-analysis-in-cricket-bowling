"""
compute_delivery_stride_auto_stump.py
Automatically detects stump pixel height using YOLO object detection.
Computes delivery stride duration & distance.
"""

import argparse
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
import cv2

KEYPOINT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]


# ------------------------------------------------------
# 1) DETECT STUMP AND RETURN PIXEL HEIGHT
# ------------------------------------------------------
def detect_stump_height(video_path, stump_model_name="yolo_stumps.pt"):
    """
    Run YOLO object detection ONCE on first frame and return stump bounding-box height.
    """

    print("\n🔍 Detecting stumps automatically using YOLO...")

    model = YOLO(stump_model_name)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to read video for stump detection")
        return None

    results = model.predict(frame, verbose=False)

    if len(results) == 0 or len(results[0].boxes) == 0:
        print("❌ No stumps detected in first frame.")
        return None

    # Pick the tallest detected stump
    heights = []
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = box.tolist()
        heights.append(y2 - y1)

    stump_pixel_height = max(heights)
    print(f"✅ Detected stump pixel height: {stump_pixel_height:.1f} px")

    return stump_pixel_height


# ------------------------------------------------------
# 2) EXTRACT KEYPOINTS (WRIST + ANKLES)
# ------------------------------------------------------
def extract_keypoints(video_path, model_name='yolov8n-pose.pt'):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(video_path)

    rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        res = model.predict(frame, verbose=False)

        if len(res) > 0 and len(res[0].keypoints) > 0:
            kps = res[0].keypoints.xy[0].cpu().numpy()

            left_ankle = kps[15].tolist()
            right_ankle = kps[16].tolist()
            right_wrist = kps[10].tolist()
        else:
            left_ankle = right_ankle = right_wrist = [np.nan, np.nan]

        rows.append({
            "frame": frame_idx,
            "left_ankle_y": left_ankle[1],
            "right_ankle_y": right_ankle[1],
            "right_ankle_x": right_ankle[0],
            "left_ankle_x": left_ankle[0],
            "right_wrist_y": right_wrist[1]
        })

    cap.release()
    return pd.DataFrame(rows)


# ------------------------------------------------------
# 3) FIND RELEASE FRAME (minimum wrist Y)
# ------------------------------------------------------
def find_release_frame(df):
    df_valid = df.dropna(subset=["right_wrist_y"])
    release_frame = df_valid.loc[df_valid["right_wrist_y"].idxmin()]["frame"]
    return int(release_frame)


# ------------------------------------------------------
# 4) FIND LAST TWO FOOT CONTACTS *before* release
# ------------------------------------------------------
def find_last_two_contacts(df, release_frame):
    df_pre = df[df["frame"] < release_frame]

    ankle_y = np.nanmax(df_pre[["left_ankle_y", "right_ankle_y"]].values, axis=1)
    smooth = pd.Series(ankle_y).rolling(9, center=True, min_periods=1).median()

    peaks = []
    for i in range(2, len(smooth)-2):
        if smooth[i] >= smooth[i-1] and smooth[i] >= smooth[i+1]:
            peaks.append(df_pre.iloc[i]["frame"])

    if len(peaks) < 2:
        return None

    return peaks[-2], peaks[-1]


# ------------------------------------------------------
# 5) PIXEL → INCH CONVERSION
# ------------------------------------------------------
def pixels_to_inches(pixel_dist, stump_pixel, stump_real_in=28):
    scale = stump_real_in / stump_pixel
    return pixel_dist * scale


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--stump_model", default="yolo_stumps.pt")
    parser.add_argument("--stump_height_in", type=float, default=28.0)
    args = parser.parse_args()

    # STEP 1: detect stump height
    stump_pixel_height = detect_stump_height(args.video, args.stump_model)
    if stump_pixel_height is None:
        print("❌ Cannot compute stride without stump height.")
        return

    # STEP 2: keypoints
    df = extract_keypoints(args.video, args.model)

    # FPS
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    # STEP 3: release
    release_frame = find_release_frame(df)
    print(f"\n🎯 Release frame detected at: {release_frame}")

    # STEP 4: last two contacts
    contacts = find_last_two_contacts(df, release_frame)
    if not contacts:
        print("❌ Could not find two valid contacts before release.")
        return

    c1, c2 = contacts
    print(f"👣 Last two contacts: {c1}, {c2}")

    # STEP 5: time difference
    t1, t2 = c1/fps, c2/fps
    duration = t2 - t1
    print(f"⏱ Duration: {duration:.3f} s")

    # STEP 6: distance
    r1 = df[df["frame"] == c1].iloc[0]
    r2 = df[df["frame"] == c2].iloc[0]

    # choose ankle with larger Y (closer to ground)
    def landing_point(row):
        if row["left_ankle_y"] > row["right_ankle_y"]:
            return np.array([row["left_ankle_x"], row["left_ankle_y"]])
        return np.array([row["right_ankle_x"], row["right_ankle_y"]])

    p1 = landing_point(r1)
    p2 = landing_point(r2)

    pixel_dist = np.linalg.norm(p2 - p1)
    print(f"📏 Pixel stride distance: {pixel_dist:.1f}px")

    inches = pixels_to_inches(pixel_dist, stump_pixel_height, args.stump_height_in)
    print(f"📐 Delivery stride distance: {inches:.2f} inches")

    # Save output
    os.makedirs("output", exist_ok=True)
    pd.DataFrame([{
        "contact1": c1,
        "contact2": c2,
        "duration_s": duration,
        "pixel_distance": pixel_dist,
        "distance_in": inches,
        "release_frame": release_frame,
        "stump_pixel_height": stump_pixel_height
    }]).to_csv("output/delivery_stride_report.csv", index=False)

    print("\n✅ Report saved to output/delivery_stride_report.csv")


if __name__ == "__main__":
    main()
