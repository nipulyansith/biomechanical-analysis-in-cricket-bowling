"""
=============================================================================
Human Body Keypoint Annotation Tool
Research: Inter-Annotator Variability in Motion Analysis
=============================================================================
Author: Research Tool
Purpose: Collect manual joint annotations from multiple annotators for
         variability and reliability analysis of manual ground truth data.
=============================================================================

DEPENDENCIES:
    pip install opencv-python numpy pandas

USAGE:
    python annotate_keypoints.py

CONTROLS (during annotation):
    Left Click  - Place a keypoint at cursor position
    R           - Reset current frame (clear all clicks for this frame)
    SPACE       - Confirm frame and move to next
    ESC         - Save and quit (progress is saved)
=============================================================================
"""

import cv2
import numpy as np
import pandas as pd
import os
import sys
import time
from datetime import datetime

# =============================================================================
# CONFIGURATION — Edit these values as needed
# =============================================================================

VIDEO_PATH = "../videos/B-02_T-01.MOV"          # Path to the input video file
OUTPUT_DIR = "annotations"              # Directory where CSV files are saved
FRAME_STEP = 30                         # Annotate every Nth frame
STUMP_HEIGHT_METERS = 0.711            # Known real-world stump height (meters)

# Ordered list of joints to annotate per frame
JOINTS = [
    "Head",
    "Left Shoulder",
    "Right Shoulder",
    "Left Elbow",
    "Right Elbow",
    "Left Wrist",
    "Right Wrist",
    "Left Knee",
    "Right Knee",
    "Left Ankle",
    "Right Ankle",
]

# Visual style constants
POINT_COLOR      = (0, 255, 100)        # BGR: bright green
LABEL_COLOR      = (255, 255, 255)      # BGR: white
CALIB_COLOR      = (0, 165, 255)        # BGR: orange
GUIDE_BG_COLOR   = (20, 20, 20)        # BGR: near-black
CONFIRM_COLOR    = (100, 220, 100)      # BGR: light green
WARN_COLOR       = (0, 80, 220)        # BGR: red-ish (BGR order)
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_SM    = 0.50
FONT_SCALE_MD    = 0.65
FONT_SCALE_LG    = 0.80
THICKNESS        = 2

# =============================================================================
# UTILITIES
# =============================================================================

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def px_to_meters(px_x: float, px_y: float, meters_per_pixel: float) -> tuple:
    """Convert pixel coordinates to meters using calibration scale."""
    return round(px_x * meters_per_pixel, 5), round(px_y * meters_per_pixel, 5)


def draw_guide_panel(frame: np.ndarray, lines: list, y_start: int = 10,
                     x_start: int = 10, line_height: int = 22) -> None:
    """
    Overlay a semi-transparent instruction panel on the top-left corner.
    Each item in `lines` is a (text, color) tuple.
    """
    max_len = max((len(t) for t, _ in lines), default=0)
    panel_w  = max_len * 8 + 20
    panel_h  = len(lines) * line_height + 16
    overlay  = frame.copy()
    cv2.rectangle(overlay, (x_start, y_start),
                  (x_start + panel_w, y_start + panel_h), GUIDE_BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    for i, (text, color) in enumerate(lines):
        y = y_start + 14 + i * line_height
        cv2.putText(frame, text, (x_start + 8, y),
                    FONT, FONT_SCALE_SM, color, 1, cv2.LINE_AA)


def draw_crosshair(frame: np.ndarray, x: int, y: int,
                   color=(200, 200, 200), size: int = 12) -> None:
    """Draw a small crosshair to help precise clicking."""
    cv2.line(frame, (x - size, y), (x + size, y), color, 1, cv2.LINE_AA)
    cv2.line(frame, (x, y - size), (x, y + size), color, 1, cv2.LINE_AA)


def draw_placed_points(frame: np.ndarray, points: list) -> None:
    """Draw all already-placed keypoints with index labels."""
    for i, (px, py) in enumerate(points):
        label = JOINTS[i]
        cv2.circle(frame, (px, py), 6, POINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, label, (px + 10, py - 5),
                    FONT, FONT_SCALE_SM, LABEL_COLOR, 1, cv2.LINE_AA)


def show_status_bar(frame: np.ndarray, message: str, color=CONFIRM_COLOR) -> None:
    """Draw a status bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 30
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, message, (10, h - 9),
                FONT, FONT_SCALE_SM, color, 1, cv2.LINE_AA)

# =============================================================================
# STEP 1 — METADATA COLLECTION
# =============================================================================

def collect_metadata() -> dict:
    """
    Prompt annotator for their name and optional metadata via the terminal.
    Returns a dictionary with annotator_name and experience_level.
    """
    print("\n" + "=" * 60)
    print("  HUMAN KEYPOINT ANNOTATION TOOL")
    print("  Research: Inter-Annotator Variability in Motion Analysis")
    print("=" * 60)
    print()
    print("Please enter your annotator information.")
    print()

    while True:
        name = input("  Annotator name (no spaces, e.g. john_doe): ").strip()
        if name:
            break
        print("  [!] Name cannot be empty. Please try again.")

    print()
    print("  Experience level options:")
    print("    1 = Novice   (< 1 year experience with motion annotation)")
    print("    2 = Intermediate (1–3 years)")
    print("    3 = Expert   (> 3 years)")
    exp_map = {"1": "novice", "2": "intermediate", "3": "expert"}
    while True:
        exp_input = input("  Enter experience level [1/2/3]: ").strip()
        if exp_input in exp_map:
            experience = exp_map[exp_input]
            break
        print("  [!] Please enter 1, 2, or 3.")

    notes = input("  Additional notes (optional, press Enter to skip): ").strip()

    meta = {
        "annotator_name":  name,
        "experience_level": experience,
        "notes":            notes if notes else "N/A",
        "session_start":   datetime.now().isoformat(timespec="seconds"),
    }

    print()
    print(f"  ✓ Annotator : {meta['annotator_name']}")
    print(f"  ✓ Experience: {meta['experience_level']}")
    print(f"  ✓ Session   : {meta['session_start']}")
    print()
    return meta

# =============================================================================
# STEP 2 — CALIBRATION
# =============================================================================

class CalibrationSession:
    """
    Handles the two-click calibration step.
    The user clicks the top then the bottom of the stump
    to establish a pixels-per-meter scale.
    """

    def __init__(self):
        self.clicks: list = []
        self.done: bool   = False

    def mouse_callback(self, event, x, y, flags, param):
        if self.done:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.clicks) < 2:
                self.clicks.append((x, y))
            if len(self.clicks) == 2:
                self.done = True


def run_calibration(cap: cv2.VideoCapture) -> dict:
    """
    Display the first frame and ask the user to click:
        1. Top of the stump
        2. Bottom of the stump

    Returns dict with pixels_per_meter and meters_per_pixel.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read the first frame for calibration.")
        sys.exit(1)

    session  = CalibrationSession()
    win_name = "CALIBRATION — Click Top then Bottom of Stump"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, session.mouse_callback)

    print("=" * 60)
    print("  CALIBRATION STEP")
    print("=" * 60)
    print("  A window showing the first video frame will open.")
    print(f"  Known stump height: {STUMP_HEIGHT_METERS} m")
    print()
    print("  Instructions:")
    print("    Click 1 → Top of the stump")
    print("    Click 2 → Bottom of the stump")
    print()
    print("  Press R to reset clicks | Close window to abort")
    print()

    while True:
        display = frame.copy()
        n_clicks = len(session.clicks)

        # Draw already placed calibration points
        labels = ["TOP of stump", "BOTTOM of stump"]
        for i, (cx, cy) in enumerate(session.clicks):
            cv2.circle(display, (cx, cy), 7, CALIB_COLOR, -1, cv2.LINE_AA)
            cv2.circle(display, (cx, cy), 10, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display, labels[i], (cx + 12, cy - 6),
                        FONT, FONT_SCALE_MD, CALIB_COLOR, 1, cv2.LINE_AA)

        # Draw calibration line if both points are set
        if n_clicks == 2:
            cv2.line(display,
                     session.clicks[0], session.clicks[1],
                     CALIB_COLOR, 2, cv2.LINE_AA)

        # Guide panel
        if n_clicks == 0:
            next_prompt = "Click #1: TOP of the stump"
        elif n_clicks == 1:
            next_prompt = "Click #2: BOTTOM of the stump"
        else:
            next_prompt = "Calibration complete! Press SPACE to continue"

        guide = [
            ("CALIBRATION", CALIB_COLOR),
            (f"Known height: {STUMP_HEIGHT_METERS} m", LABEL_COLOR),
            (next_prompt, CONFIRM_COLOR),
            ("R = reset | SPACE = confirm", (180, 180, 180)),
        ]
        draw_guide_panel(display, guide)

        status = f"Clicks placed: {n_clicks}/2"
        show_status_bar(display, status, CALIB_COLOR)

        cv2.imshow(win_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r') or key == ord('R'):
            session.clicks.clear()
            session.done = False

        if key == 32 and session.done:   # SPACE
            break

        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Calibration window closed. Exiting.")
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyWindow(win_name)

    # Compute scale
    (x1, y1), (x2, y2) = session.clicks[0], session.clicks[1]
    pixel_distance    = np.hypot(x2 - x1, y2 - y1)
    pixels_per_meter  = pixel_distance / STUMP_HEIGHT_METERS
    meters_per_pixel  = STUMP_HEIGHT_METERS / pixel_distance

    calib = {
        "top_px":            session.clicks[0],
        "bottom_px":         session.clicks[1],
        "pixel_distance":    round(pixel_distance, 3),
        "pixels_per_meter":  round(pixels_per_meter, 4),
        "meters_per_pixel":  round(meters_per_pixel, 6),
    }

    print("  ✓ Calibration complete!")
    print(f"    Pixel distance   : {calib['pixel_distance']} px")
    print(f"    Pixels per meter : {calib['pixels_per_meter']}")
    print(f"    Meters per pixel : {calib['meters_per_pixel']}")
    print()

    return calib

# =============================================================================
# STEP 3 — FRAME ANNOTATION
# =============================================================================

class AnnotationSession:
    """Tracks click state for a single frame's annotation."""

    def __init__(self):
        self.clicks: list = []
        self.reset()

    def reset(self):
        self.clicks = []

    @property
    def n_placed(self) -> int:
        return len(self.clicks)

    @property
    def is_complete(self) -> bool:
        return len(self.clicks) == len(JOINTS)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.is_complete:
                self.clicks.append((x, y))


def annotate_frame(frame: np.ndarray,
                   frame_number: int,
                   total_frames: int,
                   win_name: str,
                   session: AnnotationSession) -> list | None:
    """
    Show `frame` in the named window and collect clicks for all JOINTS.

    Returns:
        List of (joint, px, py) tuples if confirmed.
        None if the user pressed ESC (save and quit signal).
    """
    session.reset()

    while True:
        display = frame.copy()
        n       = session.n_placed

        # Draw placed points
        draw_placed_points(display, session.clicks)

        # Highlight next joint prompt
        if not session.is_complete:
            next_joint = JOINTS[n]
            prompt     = f"Click → {next_joint}  ({n + 1}/{len(JOINTS)})"
            status_col = CONFIRM_COLOR
        else:
            prompt     = "All joints placed! Press SPACE to confirm or R to reset."
            status_col = (100, 255, 255)

        # Guide panel
        guide = [
            (f"Frame {frame_number}  [{total_frames} total]", (200, 200, 255)),
            (prompt, status_col),
            ("R = reset frame | SPACE = confirm | ESC = save & quit", (160, 160, 160)),
        ]
        draw_guide_panel(display, guide)
        show_status_bar(display, prompt, status_col)

        cv2.imshow(win_name, display)
        key = cv2.waitKey(20) & 0xFF

        # Reset
        if key == ord('r') or key == ord('R'):
            session.reset()

        # Confirm
        elif key == 32 and session.is_complete:
            result = [(JOINTS[i], cx, cy) for i, (cx, cy) in enumerate(session.clicks)]
            return result

        # Save and quit
        elif key == 27:   # ESC
            return None

        # Safety: window closed externally
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            return None


def get_frame_indices(cap: cv2.VideoCapture) -> list:
    """Return a sorted list of frame indices to annotate (every FRAME_STEP)."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = list(range(0, total, FRAME_STEP))
    return indices


def frame_to_timestamp(frame_idx: int, fps: float) -> str:
    """Convert frame index to HH:MM:SS.mmm timestamp string."""
    if fps <= 0:
        return "00:00:00.000"
    total_ms  = int((frame_idx / fps) * 1000)
    ms        = total_ms % 1000
    total_sec = total_ms // 1000
    secs      = total_sec % 60
    mins      = (total_sec // 60) % 60
    hrs       = total_sec // 3600
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"

# =============================================================================
# STEP 4 — DATA SAVING
# =============================================================================

def build_output_path(meta: dict, video_path: str) -> str:
    """Construct a unique output CSV filename per annotator + video."""
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{meta['annotator_name']}_{video_id}_{ts}.csv"
    return os.path.join(OUTPUT_DIR, filename)


def init_csv(output_path: str) -> None:
    """Write the CSV header row."""
    columns = [
        "annotator_name",
        "experience_level",
        "video_id",
        "frame_number",
        "timestamp",
        "joint_name",
        "x_pixel",
        "y_pixel",
        "x_meter",
        "y_meter",
        "session_start",
        "notes",
    ]
    df = pd.DataFrame(columns=columns)
    df.to_csv(output_path, index=False)


def append_frame_annotations(output_path: str,
                              meta: dict,
                              video_id: str,
                              frame_number: int,
                              timestamp: str,
                              joint_data: list,
                              calib: dict) -> None:
    """
    Append one row per joint for a completed frame to the CSV.

    `joint_data` is a list of (joint_name, px, py) tuples.
    """
    mpp  = calib["meters_per_pixel"]
    rows = []
    for joint_name, px, py in joint_data:
        xm, ym = px_to_meters(px, py, mpp)
        rows.append({
            "annotator_name":   meta["annotator_name"],
            "experience_level": meta["experience_level"],
            "video_id":         video_id,
            "frame_number":     frame_number,
            "timestamp":        timestamp,
            "joint_name":       joint_name,
            "x_pixel":          px,
            "y_pixel":          py,
            "x_meter":          xm,
            "y_meter":          ym,
            "session_start":    meta["session_start"],
            "notes":            meta["notes"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, mode="a", header=False, index=False)

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    # ── 1. Metadata ────────────────────────────────────────────────────────
    meta = collect_metadata()

    # ── 2. Video validation ────────────────────────────────────────────────
    if not os.path.isfile(VIDEO_PATH):
        print(f"[ERROR] Video file not found: '{VIDEO_PATH}'")
        print("        Edit VIDEO_PATH at the top of the script and retry.")
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: '{VIDEO_PATH}'")
        sys.exit(1)

    fps       = cap.get(cv2.CAP_PROP_FPS)
    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_id  = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

    print(f"  Video     : {VIDEO_PATH}")
    print(f"  FPS       : {fps:.2f}")
    print(f"  Total frm : {total_raw}")
    print(f"  Step      : every {FRAME_STEP} frames")
    print()

    # ── 3. Calibration ─────────────────────────────────────────────────────
    calib = run_calibration(cap)

    # ── 4. Output setup ────────────────────────────────────────────────────
    ensure_dir(OUTPUT_DIR)
    output_path = build_output_path(meta, VIDEO_PATH)
    init_csv(output_path)
    print(f"  ✓ Output CSV: {output_path}")
    print()

    # ── 5. Frame sampling ──────────────────────────────────────────────────
    frame_indices = get_frame_indices(cap)
    total_to_annotate = len(frame_indices)

    print(f"  Frames to annotate: {total_to_annotate}")
    print()
    print("  ANNOTATION CONTROLS")
    print("  ──────────────────────────────────────────")
    print("  Left-click  → place the highlighted joint")
    print("  R           → reset current frame")
    print("  SPACE       → confirm frame")
    print("  ESC         → save progress and quit early")
    print()
    input("  Press Enter to start annotation... ")
    print()

    # ── 6. Annotation loop ─────────────────────────────────────────────────
    win_name = "ANNOTATION — Human Keypoint Labeling"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    annotation_session = AnnotationSession()
    cv2.setMouseCallback(win_name, annotation_session.mouse_callback)

    frames_done = 0
    early_quit  = False

    for idx, frame_num in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            print(f"  [WARN] Could not read frame {frame_num}, skipping.")
            continue

        ts = frame_to_timestamp(frame_num, fps)
        print(f"  Annotating frame {frame_num}  ({idx + 1}/{total_to_annotate})  @ {ts}")

        joint_data = annotate_frame(
            frame, frame_num, total_to_annotate,
            win_name, annotation_session
        )

        if joint_data is None:
            print()
            print("  [INFO] ESC pressed — saving progress and exiting.")
            early_quit = True
            break

        # Save this frame's data
        append_frame_annotations(
            output_path, meta, video_id,
            frame_num, ts, joint_data, calib
        )
        frames_done += 1
        print(f"    ✓ Frame {frame_num} saved ({len(JOINTS)} joints)")

    # ── 7. Wrap up ─────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("  ANNOTATION SESSION COMPLETE")
    print("=" * 60)
    print(f"  Annotator   : {meta['annotator_name']}")
    print(f"  Frames done : {frames_done} / {total_to_annotate}")
    print(f"  Output CSV  : {output_path}")
    if early_quit:
        print("  (Session ended early — partial data saved)")
    print()
    print("  Thank you for your contribution to the research!")
    print()


if __name__ == "__main__":
    main()