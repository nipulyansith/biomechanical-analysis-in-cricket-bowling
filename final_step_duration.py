import cv2
import pandas as pd
import numpy as np
import os
from ultralytics import YOLO
from scipy.signal import savgol_filter, find_peaks

# =========================
# SETTINGS
# =========================
VIDEO_PATH = "data/side2.MOV"
MODEL_PATH = "yolov8n-pose.pt"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

KEYPOINTS_CSV = f"{OUTPUT_DIR}/runup_step_keypoints.csv"
DEBUG_CSV     = f"{OUTPUT_DIR}/runup_step_debug.csv"
DURATIONS_CSV = f"{OUTPUT_DIR}/final5_step_durations.csv"
EVENTS_CSV    = f"{OUTPUT_DIR}/final5_step_events.csv"
ANNOTATED_VIDEO = f"{OUTPUT_DIR}/annotated_final5_steps.mp4"

# Smoothing (must be odd)
SMOOTH_WINDOW = 9
SMOOTH_POLY = 2

# Release detection: only search in last N seconds (prevents early false "highest wrist")
RELEASE_TAIL_SECONDS = 3.5

# Peak detection:
# distance: min frames between same-foot contacts (auto from FPS)
# prominence: adaptive (auto from signal) + multiplier
PROM_MULT = 0.7

# Alternation enforcement
MIN_GAP_BETWEEN_CONTACTS_S = 0.10  # reject contacts closer than this (likely noise)

FINAL_CONTACT_COUNT = 5

# =========================
# USER INPUT
# =========================
bowling_hand = input("Enter bowling hand (R/L): ").strip().upper()
if bowling_hand not in ["R", "L"]:
    raise ValueError("❌ Enter only R or L")

BOWLING_WRIST = "right_wrist" if bowling_hand == "R" else "left_wrist"
print(f"🎯 Bowling wrist: {BOWLING_WRIST}")

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

# Auto peak distance based on fps (roughly prevents double-contacts)
PEAK_DISTANCE = int(max(6, round(0.20 * fps)))  # ~0.20s
MIN_GAP_FRAMES = int(max(2, round(MIN_GAP_BETWEEN_CONTACTS_S * fps)))

# =========================
# KEYPOINT SETUP
# =========================
KEYPOINT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]
SELECTED = ["left_ankle", "right_ankle", "left_wrist", "right_wrist"]

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

        # Pick main person (largest box)
        areas = boxes[:, 2] * boxes[:, 3]
        idx = int(np.argmax(areas))
        kps = kps_all[idx]

        for i, name in enumerate(KEYPOINT_NAMES):
            if name in SELECTED:
                row[f"{name}_x"] = float(kps[i, 0])
                row[f"{name}_y"] = float(kps[i, 1])

    rows.append(row)

cap.release()

df = pd.DataFrame(rows)

# Fill missing values (keep it, but we'll debug signal quality later)
df.interpolate(inplace=True)
df.bfill(inplace=True)
df.ffill(inplace=True)

df.to_csv(KEYPOINTS_CSV, index=False)
print(f"✅ Keypoints saved: {KEYPOINTS_CSV}")

# =========================
# SMOOTH SIGNALS
# =========================
def safe_savgol(arr, window, poly):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    w = window
    if n < 5:
        return arr
    if w >= n:
        w = n - 1 if (n - 1) % 2 == 1 else n - 2
    if w < 5:
        w = 5
    if w % 2 == 0:
        w += 1
        if w >= n:
            w -= 2
    return savgol_filter(arr, w, poly)

left_y  = df["left_ankle_y"].to_numpy()
right_y = df["right_ankle_y"].to_numpy()
wrist_y = df[f"{BOWLING_WRIST}_y"].to_numpy()

df["left_ankle_y_s"]  = safe_savgol(left_y,  SMOOTH_WINDOW, SMOOTH_POLY)
df["right_ankle_y_s"] = safe_savgol(right_y, SMOOTH_WINDOW, SMOOTH_POLY)
df["wrist_y_s"]       = safe_savgol(wrist_y, SMOOTH_WINDOW, SMOOTH_POLY)

# =========================
# RELEASE DETECTION (tail search)
# =========================
tail_frames = int(max(10, round(RELEASE_TAIL_SECONDS * fps)))
start_idx = max(0, len(df) - tail_frames)

release_idx = int(df["wrist_y_s"].iloc[start_idx:].idxmin())
release_frame = int(df.loc[release_idx, "frame"])
print(f"🏏 Release frame (tail-based): {release_frame}")

# =========================
# CONTACT DETECTION HELPERS
# =========================
def adaptive_prominence(y_signal):
    """
    Use robust variability (MAD) to set a prominence threshold automatically.
    """
    y = np.asarray(y_signal, dtype=float)
    med = np.median(y)
    mad = np.median(np.abs(y - med)) + 1e-9
    # scaled MAD as variability estimate
    return float(PROM_MULT * (1.4826 * mad))

def detect_contacts(y_signal, distance_frames):
    """
    In image coords, y increases DOWN, so ground contact is usually local MAX in ankle y.
    """
    prom = adaptive_prominence(y_signal)
    peaks, props = find_peaks(y_signal, distance=distance_frames, prominence=prom)
    return peaks, prom, props

def merge_and_clean_events(left_peaks, right_peaks, left_sig, right_sig):
    """
    Merge two lists of peaks into one timeline and remove events too close together.
    If two events collide, keep the one with higher ankle y (stronger contact).
    """
    events = [(int(i), "L") for i in left_peaks] + [(int(i), "R") for i in right_peaks]
    events.sort(key=lambda x: x[0])

    cleaned = []
    for idx, foot in events:
        if idx > release_idx:
            continue
        if not cleaned:
            cleaned.append((idx, foot))
            continue

        prev_idx, prev_foot = cleaned[-1]
        if (idx - prev_idx) <= MIN_GAP_FRAMES:
            # collision: keep stronger peak (bigger y)
            prev_y = left_sig[prev_idx] if prev_foot == "L" else right_sig[prev_idx]
            curr_y = left_sig[idx] if foot == "L" else right_sig[idx]
            if curr_y > prev_y:
                cleaned[-1] = (idx, foot)
        else:
            cleaned.append((idx, foot))

    return cleaned

def enforce_alternation(events, left_sig, right_sig):
    """
    Force L/R alternation. If same foot repeats, keep the stronger one (higher ankle y).
    """
    if not events:
        return events

    alt = [events[0]]
    for idx, foot in events[1:]:
        pidx, pfoot = alt[-1]
        if foot != pfoot:
            alt.append((idx, foot))
        else:
            # same foot repeat: keep stronger (higher y)
            prev_y = left_sig[pidx] if pfoot == "L" else right_sig[pidx]
            curr_y = left_sig[idx] if foot == "L" else right_sig[idx]
            if curr_y > prev_y:
                alt[-1] = (idx, foot)

    return alt

# =========================
# DETECT CONTACTS
# =========================
left_sig  = df["left_ankle_y_s"].to_numpy()
right_sig = df["right_ankle_y_s"].to_numpy()

left_peaks, left_prom, _ = detect_contacts(left_sig, PEAK_DISTANCE)
right_peaks, right_prom, _ = detect_contacts(right_sig, PEAK_DISTANCE)

events = merge_and_clean_events(left_peaks, right_peaks, left_sig, right_sig)
events_alt = enforce_alternation(events, left_sig, right_sig)

# Take final contacts
final_events = [e for e in events_alt if e[0] <= release_idx]
final_events = final_events[-FINAL_CONTACT_COUNT:]

if len(final_events) < FINAL_CONTACT_COUNT:
    print("\n⚠️ Not enough final contacts detected.")
    print(f"   Found {len(final_events)} contacts, need {FINAL_CONTACT_COUNT}.")
    print("   Try: increase RELEASE_TAIL_SECONDS, reduce PROM_MULT (e.g. 0.5), or reduce SMOOTH_WINDOW (e.g. 7).")

final_events_frames = [(int(df.loc[idx, "frame"]), foot, idx) for idx, foot in final_events]

print("\n🏃 FINAL 5 CONTACTS (before release)")
for fr, foot, _ in final_events_frames:
    print(f"  {foot} contact @ frame {fr}")

# =========================
# STEP DURATIONS
# =========================
step_rows = []
for j in range(len(final_events_frames) - 1):
    f1, foot1, _ = final_events_frames[j]
    f2, foot2, _ = final_events_frames[j + 1]
    dt = (f2 - f1) / fps
    step_rows.append({
        "step_index": j + 1,
        "from_foot": foot1,
        "to_foot": foot2,
        "frame_j": f1,
        "frame_j_plus_1": f2,
        "fps": float(fps),
        "step_duration_s": float(dt)
    })

dur_df = pd.DataFrame(step_rows)
dur_df.to_csv(DURATIONS_CSV, index=False)
print(f"\n✅ Saved step durations: {DURATIONS_CSV}")

events_df = pd.DataFrame([{
    "order": i + 1,
    "foot": foot,
    "frame": fr,
    "idx_in_df": idx
} for i, (fr, foot, idx) in enumerate(final_events_frames)])
events_df.to_csv(EVENTS_CSV, index=False)
print(f"✅ Saved final events: {EVENTS_CSV}")

# =========================
# DEBUG CSV (very useful)
# =========================
dbg = pd.DataFrame({
    "frame": df["frame"],
    "left_y_s": left_sig,
    "right_y_s": right_sig,
    "wrist_y_s": df["wrist_y_s"].to_numpy(),
    "is_left_peak": 0,
    "is_right_peak": 0,
    "is_final_event": 0
})

dbg.loc[left_peaks, "is_left_peak"] = 1
dbg.loc[right_peaks, "is_right_peak"] = 1
for _, _, idx in final_events_frames:
    dbg.loc[idx, "is_final_event"] = 1

dbg.to_csv(DEBUG_CSV, index=False)
print(f"🧪 Debug CSV saved: {DEBUG_CSV}")
print(f"   left prominence used:  {left_prom:.3f}")
print(f"   right prominence used: {right_prom:.3f}")

# =========================
# ANNOTATED VIDEO (slow at events)
# =========================
SLOW_FACTOR = 4
PAUSE_FRAMES = int(fps * 0.5)
EVENT_WINDOW = int(fps * 0.25)

event_frames = [fr for fr, _, _ in final_events_frames] + [release_frame]

cap = cv2.VideoCapture(VIDEO_PATH)
out = cv2.VideoWriter(
    ANNOTATED_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_no = 0
event_map = {fr: (foot, idx) for fr, foot, idx in final_events_frames}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1
    draw = frame.copy()

    # draw events
    if frame_no in event_map:
        foot, idx = event_map[frame_no]
        if foot == "L":
            x = int(df.loc[idx, "left_ankle_x"])
            y = int(df.loc[idx, "left_ankle_y"])
            color = (0, 255, 0)
            label = "L CONTACT"
        else:
            x = int(df.loc[idx, "right_ankle_x"])
            y = int(df.loc[idx, "right_ankle_y"])
            color = (0, 0, 255)
            label = "R CONTACT"

        cv2.circle(draw, (x, y), 18, color, -1)
        cv2.putText(draw, label, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

    if frame_no == release_frame:
        cv2.putText(draw, "RELEASE", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 0), 4)

    # slow near events
    is_event_zone = any(abs(frame_no - ev) <= EVENT_WINDOW for ev in event_frames)
    repeat = SLOW_FACTOR if is_event_zone else 1

    for _ in range(repeat):
        out.write(draw)

    if frame_no in event_frames:
        for _ in range(PAUSE_FRAMES):
            out.write(draw)

cap.release()
out.release()
print(f"\n🎥 Annotated video saved: {ANNOTATED_VIDEO}")