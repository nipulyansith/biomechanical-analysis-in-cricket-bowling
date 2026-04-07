"""
extract_models.py
-----------------
Runs YOLOv8-N and YOLOv8-L pose models on every 30th frame of the video
and saves their 8 selected keypoints to separate CSVs.

Also logs:
    - total extraction time per model
    - average extraction time per processed frame

Output:
    output/nmodel.csv
    output/lmodel.csv
"""

import cv2
import numpy as np
import pandas as pd
import os
import time
from ultralytics import YOLO

# ── Settings ────────────────────────────────────────────────────────────────
VIDEO_PATH  = "../data/nipul.MOV"
OUTPUT_DIR  = "output"
FRAME_STEP  = 30          # annotate / extract every Nth frame

# The 8 keypoints used throughout the pipeline
KEYPOINTS_8 = [
    "left_shoulder",  "right_shoulder",
    "left_elbow",     "right_elbow",
    "left_wrist",     "right_wrist",
    "left_ankle",     "right_ankle",
]

# Full COCO 17-keypoint name list (YOLO ordering)
COCO_KP = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP8_IDX = {name: COCO_KP.index(name) for name in KEYPOINTS_8}
# ────────────────────────────────────────────────────────────────────────────


def extract_keypoints(video_path, model_path, frame_step, label):
    model = YOLO(model_path)
    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows  = []

    print(f"\n[{label}] Extracting every {frame_step}th frame from {total} total frames ...")

    frame_idx = 0
    processed_frames = 0

    # Start timing for this model
    extraction_start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % frame_step != 0:
            continue

        processed_frames += 1
        frame_start_time = time.time()

        row = {"frame": frame_idx}
        for kp in KEYPOINTS_8:
            row[f"{kp}_x"] = np.nan
            row[f"{kp}_y"] = np.nan

        results = model.predict(frame, verbose=False)
        if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kps = results[0].keypoints.xy[0].cpu().numpy()   # (17, 2)
            for kp_name in KEYPOINTS_8:
                idx = KP8_IDX[kp_name]
                x, y = kps[idx]
                if x > 0 or y > 0:   # (0,0) = undetected
                    row[f"{kp_name}_x"] = float(x)
                    row[f"{kp_name}_y"] = float(y)

        rows.append(row)

        frame_end_time = time.time()
        frame_extract_time = frame_end_time - frame_start_time

        print(
            f"  [{label}] frame {frame_idx}/{total} | "
            f"extract time: {frame_extract_time:.4f} sec"
        )

    cap.release()

    # End timing for this model
    extraction_end_time = time.time()
    total_extract_time = extraction_end_time - extraction_start_time
    avg_time_per_frame = total_extract_time / processed_frames if processed_frames > 0 else 0

    print(f"\n[{label}] Extraction completed.")
    print(f"[{label}] Total processed frames     : {processed_frames}")
    print(f"[{label}] Total extraction time     : {total_extract_time:.4f} sec")
    print(f"[{label}] Average time per frame    : {avg_time_per_frame:.4f} sec/frame")

    return pd.DataFrame(rows), total_extract_time, avg_time_per_frame


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for label, weights in [("N-model", "yolov8n-pose.pt"),
                           ("L-model", "yolov8l-pose.pt")]:
        out_file = os.path.join(OUTPUT_DIR, f"{'nmodel' if 'N' in label else 'lmodel'}.csv")

        df, total_time, avg_time = extract_keypoints(VIDEO_PATH, weights, FRAME_STEP, label)
        df.to_csv(out_file, index=False)

        print(f"[{label}] Saved {len(df)} rows → {out_file}")
        print(f"[{label}] Final total time   : {total_time:.4f} sec")
        print(f"[{label}] Final avg/frame    : {avg_time:.4f} sec/frame\n")

    print("✅ Done. Next step: run annotate_groundtruth.py to label ground truth.")


if __name__ == "__main__":
    main()