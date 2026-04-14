"""
Biomechanical Motion Analysis & Ground Truth Validation Tool
Cricket Fast-Bowling: Manual Annotation vs YOLOv8l-Pose vs MediaPipe Pose
Backend: Flask  |  CV: OpenCV, Ultralytics YOLOv8l-Pose, MediaPipe Pose
"""

import uuid
import zipfile
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

UPLOAD_FOLDER = Path("uploads")
OUTPUT_FOLDER = Path("outputs")
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

STUMP_HEIGHT_M = 0.711

JOINT_NAMES = [
    "Head",        # 0
    "L-Shoulder",  # 1
    "R-Shoulder",  # 2
    "L-Elbow",     # 3
    "R-Elbow",     # 4
    "L-Wrist",     # 5
    "R-Wrist",     # 6
    "L-Knee",      # 7
    "R-Knee",      # 8
    "L-Ankle",     # 9
    "R-Ankle",     # 10
]

# MediaPipe Pose landmark indices
MP_JOINT_MAP = {
    "Head":       0,   # nose
    "L-Shoulder": 11,
    "R-Shoulder": 12,
    "L-Elbow":    13,
    "R-Elbow":    14,
    "L-Wrist":    15,
    "R-Wrist":    16,
    "L-Knee":     25,
    "R-Knee":     26,
    "L-Ankle":    27,
    "R-Ankle":    28,
}

# YOLOv8 COCO-Pose keypoint indices
YOLO_JOINT_MAP = {
    "Head":       0,   # nose
    "L-Shoulder": 5,
    "R-Shoulder": 6,
    "L-Elbow":    7,
    "R-Elbow":    8,
    "L-Wrist":    9,
    "R-Wrist":    10,
    "L-Knee":     13,
    "R-Knee":     14,
    "L-Ankle":    15,
    "R-Ankle":    16,
}

_yolo_model = None

def get_yolo_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("yolov8l-pose.pt")
    return _yolo_model


# ─────────────────────────────────────────────────────────────────────────────
# Video utilities
# ─────────────────────────────────────────────────────────────────────────────

def extract_sampled_frames(video_path: str, step: int = 10) -> dict:
    """Extract every `step`-th frame. Returns {frame_number: BGR ndarray}."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = {}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frames[frame_idx] = frame.copy()
        frame_idx += 1
    cap.release()
    return frames


def get_single_frame(video_path: str, frame_num: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_video_meta(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    meta = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps":          cap.get(cv2.CAP_PROP_FPS),
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate math
# ─────────────────────────────────────────────────────────────────────────────

def compute_ppm(stump_top_y: float, stump_bottom_y: float) -> float:
    """Pixels-per-metre from stump calibration clicks."""
    return abs(stump_bottom_y - stump_top_y) / STUMP_HEIGHT_M


def px_to_meters(px: float, py: float, ppm: float) -> tuple:
    return px / ppm, py / ppm


def build_dataframe(records: list, annotator_id: str, ppm: float) -> pd.DataFrame:
    """
    Convert flat list of {frame_num, joint_name, pixel_x, pixel_y}
    into the full output DataFrame with metre coords and frame-0 deltas.
    Sub-pixel float64 precision throughout.
    """
    rows = []
    for r in records:
        mx, my = px_to_meters(float(r["pixel_x"]), float(r["pixel_y"]), ppm)
        rows.append({
            "Frame_Num":    int(r["frame_num"]),
            "Joint_Name":   r["joint_name"],
            "Pixel_X":      float(r["pixel_x"]),
            "Pixel_Y":      float(r["pixel_y"]),
            "Meter_X":      mx,
            "Meter_Y":      my,
            "Delta_X":      0.0,
            "Delta_Y":      0.0,
            "Annotator_ID": annotator_id,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Frame-0 origins per joint
    min_frame = df["Frame_Num"].min()
    origin = {}
    for _, row in df[df["Frame_Num"] == min_frame].iterrows():
        origin[row["Joint_Name"]] = (row["Meter_X"], row["Meter_Y"])

    def _apply_delta(row):
        jname = row["Joint_Name"]
        if jname in origin and pd.notna(row["Meter_X"]):
            ox, oy = origin[jname]
            return pd.Series([row["Meter_X"] - ox, row["Meter_Y"] - oy])
        return pd.Series([float("nan"), float("nan")])

    df[["Delta_X", "Delta_Y"]] = df.apply(_apply_delta, axis=1)

    # Enforce strict joint ordering within each frame
    joint_order = {j: i for i, j in enumerate(JOINT_NAMES)}
    df["_j"] = df["Joint_Name"].map(joint_order)
    df = df.sort_values(["Frame_Num", "_j"]).drop(columns="_j")
    return df.round(8).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model inference
# ─────────────────────────────────────────────────────────────────────────────

def run_mediapipe(frames: dict) -> list:
    """
    MediaPipe Pose model_complexity=0 (high-accuracy) on every sampled frame.
    Returns flat list of joint records.
    """
    mp_pose = mp.solutions.pose
    records = []

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=0,
        smooth_landmarks=False,
        enable_segmentation=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as pose:
        for frame_num in sorted(frames):
            frame = frames[frame_num]
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                for joint_name, lm_idx in MP_JOINT_MAP.items():
                    lmk = lm[lm_idx]
                    records.append({
                        "frame_num":  frame_num,
                        "joint_name": joint_name,
                        "pixel_x":    float(lmk.x) * w,
                        "pixel_y":    float(lmk.y) * h,
                    })
            else:
                for joint_name in JOINT_NAMES:
                    records.append({
                        "frame_num":  frame_num,
                        "joint_name": joint_name,
                        "pixel_x":    float("nan"),
                        "pixel_y":    float("nan"),
                    })
    return records


def run_yolo(frames: dict) -> list:
    """
    YOLOv8l-Pose on every sampled frame.
    Selects the highest box-confidence person detection per frame.
    Returns flat list of joint records.
    """
    model = get_yolo_model()
    records = []

    for frame_num in sorted(frames):
        frame = frames[frame_num]
        results = model(frame, verbose=False)
        detected = False

        if results and results[0].keypoints is not None:
            kp_xy   = results[0].keypoints.xy.cpu().numpy()    # (N, 17, 2)
            box_conf = results[0].boxes.conf.cpu().numpy()      # (N,)

            if len(kp_xy) > 0:
                best = int(np.argmax(box_conf))
                kp = kp_xy[best]  # (17, 2)

                for joint_name, yolo_idx in YOLO_JOINT_MAP.items():
                    records.append({
                        "frame_num":  frame_num,
                        "joint_name": joint_name,
                        "pixel_x":    float(kp[yolo_idx][0]),
                        "pixel_y":    float(kp[yolo_idx][1]),
                    })
                detected = True

        if not detected:
            for joint_name in JOINT_NAMES:
                records.append({
                    "frame_num":  frame_num,
                    "joint_name": joint_name,
                    "pixel_x":    float("nan"),
                    "pixel_y":    float("nan"),
                })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Flask API
# ─────────────────────────────────────────────────────────────────────────────

def _find_video(session_id: str):
    for ext in [".mp4", ".mov", ".avi"]:
        p = UPLOAD_FOLDER / f"{session_id}{ext}"
        if p.exists():
            return str(p)
    return None


@app.route("/")
def index():
    with open("templates/index.html", encoding='utf-8') as f:
        return f.read()


@app.route("/api/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    ext  = Path(file.filename).suffix.lower()
    if ext not in {".mp4", ".mov", ".avi"}:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    session_id = str(uuid.uuid4())
    save_path  = UPLOAD_FOLDER / f"{session_id}{ext}"
    file.save(str(save_path))

    meta = get_video_meta(str(save_path))
    sampled = list(range(0, meta["total_frames"], 10))

    return jsonify({
        "session_id":     session_id,
        "total_frames":   meta["total_frames"],
        "fps":            meta["fps"],
        "width":          meta["width"],
        "height":         meta["height"],
        "sampled_frames": sampled,
    })


@app.route("/api/frame/<session_id>/<int:frame_num>")
def serve_frame(session_id, frame_num):
    video_path = _find_video(session_id)
    if not video_path:
        return jsonify({"error": "Session not found"}), 404

    frame = get_single_frame(video_path, frame_num)
    if frame is None:
        return jsonify({"error": "Frame not found"}), 404

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return send_file(BytesIO(buf.tobytes()), mimetype="image/jpeg")


@app.route("/api/process", methods=["POST"])
def process_video():
    """
    Run YOLOv8l-Pose + MediaPipe on sampled frames, merge with manual
    annotations, return ZIP of three CSVs.
    """
    body = request.get_json(force=True)

    session_id         = body["session_id"]
    video_id           = body.get("video_id", "video")
    annotator_name     = body.get("annotator_name", "Unknown")
    experience_level   = body.get("experience_level", "Unknown")
    ppm                = float(body["ppm"])
    manual_annotations = body["manual_annotations"]

    video_path = _find_video(session_id)
    if not video_path:
        return jsonify({"error": "Session video not found"}), 404

    annotator_id = f"{annotator_name} ({experience_level})"

    manual_df = build_dataframe(manual_annotations, annotator_id, ppm)

    frames = extract_sampled_frames(video_path, step=10)

    mp_records  = run_mediapipe(frames)
    mp_df       = build_dataframe(mp_records, "MediaPipe-Pose-complexity2", ppm)

    yolo_records = run_yolo(frames)
    yolo_df      = build_dataframe(yolo_records, "YOLOv8l-Pose", ppm)

    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{video_id}_Manual.csv",    manual_df.to_csv(index=False))
        zf.writestr(f"{video_id}_MediaPipe.csv", mp_df.to_csv(index=False))
        zf.writestr(f"{video_id}_YOLO.csv",      yolo_df.to_csv(index=False))

    zip_buf.seek(0)
    return send_file(
        zip_buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{video_id}_biomech_analysis.zip",
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)