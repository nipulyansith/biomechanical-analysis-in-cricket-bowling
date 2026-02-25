import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import os

# ============================================================
# ELBOW FLEXION/EXTENSION (ARM-LOCK, NO TRAJECTORY, NO TRACKER)
# CHANGES YOU REQUESTED:
# ✅ Removed video trimming completely
# ✅ Mark BALL RELEASE at: (highest wrist frame + 1)
#    -> "just after wrist gets highest"
# ✅ Keeps arm-lock to avoid switching to the other arm
# ============================================================

# ---------------- Utilities ----------------
def calculate_angle(a, b, c):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return np.nan
    cosine_angle = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))

def get_user_hand_preference():
    print("\n" + "="*50)
    print("HAND SELECTION FOR BOWLING ANALYSIS")
    print("="*50)
    print("Which hand should be tracked for bowling analysis?")
    print("1. Right hand")
    print("2. Left hand")
    print("3. Auto-detect (may be inconsistent)")
    while True:
        choice = input("\nEnter your choice (1, 2, or 3): ").strip()
        if choice == '1':
            return 'right'
        elif choice == '2':
            return 'left'
        elif choice == '3':
            return 'auto'
        print("Invalid choice. Please enter 1, 2, or 3.")

def _finite_xy(pt):
    return pt is not None and np.isfinite(pt).all()

def _dist(a, b):
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))

# YOLOv8 pose COCO indices
def arm_indices(side):
    return (6, 8, 10) if side == "right" else (5, 7, 9)

def validate_keypoints_xy_conf(person_xy, person_conf, side, conf_th=0.15):
    s, e, w = arm_indices(side)
    shoulder = person_xy[s]
    elbow    = person_xy[e]
    wrist    = person_xy[w]
    sc, ec, wc = person_conf[s], person_conf[e], person_conf[w]

    if not (np.isfinite([shoulder, elbow, wrist]).all()):
        return None
    if (sc < conf_th) or (ec < conf_th) or (wc < conf_th):
        return None
    return shoulder, elbow, wrist, float(sc), float(ec), float(wc)

def find_best_person_for_hand_with_tracking(persons_xy, persons_conf, preferred_hand, last_person_center=None):
    if len(persons_xy) == 0:
        return None, None, None

    side = preferred_hand

    if last_person_center is None:
        for i in range(len(persons_xy)):
            kp = validate_keypoints_xy_conf(persons_xy[i], persons_conf[i], side, conf_th=0.10)
            if kp is not None:
                return i, persons_xy[i], persons_conf[i]
        return 0, persons_xy[0], persons_conf[0]

    s_idx = 6 if side == "right" else 5
    best_i, best_d = None, float("inf")
    for i in range(len(persons_xy)):
        shoulder = persons_xy[i][s_idx]
        if not np.isfinite(shoulder).all():
            continue
        d = np.linalg.norm(shoulder - last_person_center)
        if d < best_d:
            best_d = d
            best_i = i

    if best_i is None:
        return None, None, None
    return best_i, persons_xy[best_i], persons_conf[best_i]

def detect_bowling_hand_with_ball(persons_xy, persons_conf, frame_width):
    if len(persons_xy) == 0:
        return None
    best_i, best_score = 0, -1
    for i in range(len(persons_xy)):
        p = persons_xy[i]
        c = persons_conf[i]
        score = 0.0

        rs, rw = p[6], p[10]
        ls, lw = p[5], p[9]
        rsc, rwc = c[6], c[10]
        lsc, lwc = c[5], c[9]

        if np.isfinite(rs).all() and rsc > 0.10:
            dist_center = abs(rs[0] - frame_width / 2)
            score += max(0, 100 - dist_center / 10)
        if np.isfinite(rs).all() and np.isfinite(rw).all() and rsc > 0.10 and rwc > 0.10:
            score += np.linalg.norm(rw - rs) / 10
        if np.isfinite(ls).all() and np.isfinite(lw).all() and lsc > 0.10 and lwc > 0.10:
            score += np.linalg.norm(lw - ls) / 10

        if score > best_score:
            best_score = score
            best_i = i
    return best_i

def determine_bowling_arm(person_xy, person_conf, conf_th=0.10):
    rs, re, rw = person_xy[6], person_xy[8], person_xy[10]
    ls, le, lw = person_xy[5], person_xy[7], person_xy[9]
    r_ok = np.isfinite([rs, re, rw]).all() and (person_conf[6] > conf_th and person_conf[8] > conf_th and person_conf[10] > conf_th)
    l_ok = np.isfinite([ls, le, lw]).all() and (person_conf[5] > conf_th and person_conf[7] > conf_th and person_conf[9] > conf_th)
    if not r_ok and not l_ok:
        return None
    if r_ok and not l_ok:
        return "right"
    if l_ok and not r_ok:
        return "left"
    return "right" if np.linalg.norm(rw - rs) > np.linalg.norm(lw - ls) else "left"

def choose_locked_arm_with_wrist_continuity(person_xy, person_conf, preferred_side, last_wrist_xy, max_jump, conf_th=0.15):
    cand_pref = validate_keypoints_xy_conf(person_xy, person_conf, preferred_side, conf_th=conf_th)
    other_side = "left" if preferred_side == "right" else "right"
    cand_other = validate_keypoints_xy_conf(person_xy, person_conf, other_side, conf_th=conf_th)

    if cand_pref is None and cand_other is None:
        return None

    if last_wrist_xy is None or not _finite_xy(last_wrist_xy):
        return (preferred_side, cand_pref) if cand_pref is not None else (other_side, cand_other)

    def wrist_dist(cand):
        wrist = cand[2]
        return _dist(wrist, last_wrist_xy)

    dp = wrist_dist(cand_pref) if cand_pref is not None else float("inf")
    do = wrist_dist(cand_other) if cand_other is not None else float("inf")

    pref_ok = dp <= max_jump
    other_ok = do <= max_jump

    if pref_ok and other_ok:
        return (preferred_side, cand_pref) if dp <= do else (other_side, cand_other)
    if pref_ok and not other_ok:
        return (preferred_side, cand_pref)
    if (not pref_ok) and other_ok:
        return (other_side, cand_other)

    return (preferred_side, cand_pref) if cand_pref is not None else (other_side, cand_other)

def draw_skeletal_points(frame, shoulder, elbow, wrist, arm_side='right', total_detections=0):
    shoulder_color = (0, 255, 0)
    elbow_color = (255, 0, 0)
    wrist_color = (0, 0, 255)
    line_color = (255, 255, 0)

    cv2.line(frame, tuple(shoulder.astype(int)), tuple(elbow.astype(int)), line_color, 3)
    cv2.line(frame, tuple(elbow.astype(int)), tuple(wrist.astype(int)), line_color, 3)

    cv2.circle(frame, tuple(shoulder.astype(int)), 8, shoulder_color, -1)
    cv2.circle(frame, tuple(elbow.astype(int)), 8, elbow_color, -1)
    cv2.circle(frame, tuple(wrist.astype(int)), 8, wrist_color, -1)

    cv2.putText(frame, f'Detections: {total_detections}', (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def _choose_savgol_window(n, fps, desired_ms=150):
    if n < 5:
        w = n if n % 2 == 1 else max(3, n - 1)
        return max(3, w)
    desired_frames = int(round((desired_ms / 1000.0) * fps))
    desired_frames = max(desired_frames, 5)
    if desired_frames % 2 == 0:
        desired_frames += 1
    w = min(desired_frames, n if n % 2 == 1 else n - 1)
    w = max(w, 5)
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    return max(5, w)

def detect_ball_release_first_down_after_highest(
    wrist_positions,
    time_stamps,
    fps,
    start_idx=0,
    search_seconds=0.9,
    smooth_ms=150,
    down_thresh_px_per_frame=1.0,
    hold_frames=3
):
    """
    Returns: (release_sample_idx, highest_sample_idx)

    NOTE: You asked to ANNOTATE at (highest + 1), not this release index.
    We still compute highest idx here reliably.
    """
    n = len(wrist_positions)
    if n < 8:
        return None, None

    start_idx = int(max(0, min(start_idx, n - 1)))
    end_idx = int(min(n - 1, start_idx + int(search_seconds * fps)))
    if end_idx - start_idx < 6:
        end_idx = min(n - 1, start_idx + 6)

    wrist = np.array(wrist_positions, dtype=float)
    wy = wrist[:, 1].copy()

    for i in range(n):
        if not np.isfinite(wy[i]):
            wy[i] = wy[i-1] if i > 0 else np.nan
    if not np.isfinite(wy).any():
        return None, None

    win = _choose_savgol_window(n, fps, desired_ms=smooth_ms)
    try:
        wy_s = savgol_filter(wy, win, 2)
    except Exception:
        wy_s = wy

    window = wy_s[start_idx:end_idx+1]
    highest_rel = int(np.argmin(window))
    highest_idx = start_idx + highest_rel

    for i in range(highest_idx, min(end_idx - hold_frames, n - hold_frames - 1)):
        ok = True
        for k in range(hold_frames):
            dy = wy_s[i + k + 1] - wy_s[i + k]
            if dy < down_thresh_px_per_frame:
                ok = False
                break
        if ok:
            return i, highest_idx

    return highest_idx, highest_idx

def mark_event_on_video_at_point(input_video_path, output_video_path, event_frame, point_xy=None,
                                 label="BALL RELEASE", radius=18):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    event_frame = int(event_frame)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == event_frame:
            if point_xy is not None and np.isfinite(point_xy).all():
                cx, cy = int(point_xy[0]), int(point_xy[1])
            else:
                cx, cy = int(w * 0.85), int(h * 0.15)

            cv2.circle(frame, (cx, cy), radius, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), radius + 10, (0, 0, 255), 3)

            cv2.putText(frame, label, (max(10, cx - 260), min(h - 30, cy + 60)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(frame, f"Frame: {event_frame}", (max(10, cx - 260), min(h - 10, cy + 95)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

# ---------------- MAIN ----------------
os.makedirs("output", exist_ok=True)
model = YOLO("yolov8n-pose.pt")

preferred_hand = get_user_hand_preference()
print(f"✅ Will track: {preferred_hand} hand")

if preferred_hand == 'auto':
    print("⚠️  Warning: Auto-detect mode may result in inconsistent tracking between hands")
else:
    print(f"✅ Locked to {preferred_hand} hand for consistent tracking")

# Try to find an available video file
import os
possible_videos = ["kavisha.mp4"]
video_path = None

print("Checking for available video files...")
for video in possible_videos:
    if os.path.exists(video):
        test_cap = cv2.VideoCapture(video)
        if test_cap.isOpened():
            video_path = video
            test_cap.release()
            print(f"Found: {video}")
            break
        test_cap.release()

if video_path is None:
    print("\n❌ No video found in current directory.")
    exit()

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 1e-6:
    fps = 60.0
fps = float(fps)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

diag = (frame_width**2 + frame_height**2) ** 0.5
MAX_JUMP = diag * 0.08

output_full = "output/bowling_analysis_with_skeleton.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_full, fourcc, fps, (frame_width, frame_height))

elbow_angles, time_stamps, frame_numbers = [], [], []
wrist_positions = []

successful_detections = 0
last_valid_person_center = None
locked_arm_side = None
last_wrist_xy = None
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    detected_keypoints = False
    arm_side_used = None
    person_xy = None
    person_conf = None
    shoulder = elbow = wrist = None

    if results and results[0].keypoints is not None:
        persons_xy = results[0].keypoints.xy.cpu().numpy()
        persons_conf = results[0].keypoints.conf.cpu().numpy()

        if len(persons_xy) > 0:
            if preferred_hand == "auto":
                if locked_arm_side is None:
                    idx = detect_bowling_hand_with_ball(persons_xy, persons_conf, frame_width)
                    if idx is not None:
                        cand_xy = persons_xy[idx]
                        cand_conf = persons_conf[idx]
                        auto_side = determine_bowling_arm(cand_xy, cand_conf)
                        if auto_side is not None:
                            preferred_hand = auto_side
                            locked_arm_side = auto_side
                if locked_arm_side is None:
                    locked_arm_side = "right"
            else:
                if locked_arm_side is None:
                    locked_arm_side = preferred_hand

            idx, person_xy, person_conf = find_best_person_for_hand_with_tracking(
                persons_xy, persons_conf, locked_arm_side, last_valid_person_center
            )

            if person_xy is not None and person_conf is not None:
                chosen = choose_locked_arm_with_wrist_continuity(
                    person_xy, person_conf,
                    preferred_side=locked_arm_side,
                    last_wrist_xy=last_wrist_xy,
                    max_jump=MAX_JUMP,
                    conf_th=0.15
                )

                if chosen is not None:
                    arm_side_used, (shoulder, elbow, wrist, sc, ec, wc) = chosen
                    detected_keypoints = True
                    last_valid_person_center = shoulder.copy()
                    if wc >= 0.20:
                        last_wrist_xy = wrist.copy()

    if detected_keypoints and person_xy is not None:
        successful_detections += 1
        draw_skeletal_points(frame, shoulder, elbow, wrist, arm_side_used, successful_detections)

        angle = calculate_angle(shoulder, elbow, wrist)
        if np.isfinite(angle):
            elbow_angles.append(angle)
            time_stamps.append(frame_idx / fps)
            frame_numbers.append(frame_idx)
            wrist_positions.append(wrist.copy())

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

if len(elbow_angles) == 0:
    print("❌ No detections. Exiting.")
    exit()

# -------- Find highest wrist and "release just after highest" --------
# Search from 0 across full clip (or you can make a start_idx if you want)
_, highest_wrist_idx = detect_ball_release_first_down_after_highest(
    wrist_positions=wrist_positions,
    time_stamps=time_stamps,
    fps=fps,
    start_idx=0,
    search_seconds=max(0.9, len(wrist_positions) / fps),  # cover full clip safely
    smooth_ms=150,
    down_thresh_px_per_frame=1.0,
    hold_frames=3
)

if highest_wrist_idx is None or highest_wrist_idx >= len(frame_numbers):
    print("⚠️ Could not determine highest wrist frame. Marking last frame as fallback.")
    highest_wrist_frame = frame_numbers[-1]
else:
    highest_wrist_frame = int(frame_numbers[highest_wrist_idx])

# ✅ annotate at frame just AFTER highest wrist
release_frame = min(highest_wrist_frame + 1, frame_numbers[-1])

# wrist point for that annotated frame (if available)
if highest_wrist_idx is not None and (highest_wrist_idx + 1) < len(wrist_positions):
    release_wrist = np.array(wrist_positions[highest_wrist_idx + 1], dtype=float)
else:
    release_wrist = np.array(wrist_positions[highest_wrist_idx], dtype=float) if highest_wrist_idx is not None else None

print(f"\nHighest wrist frame: {highest_wrist_frame}")
print(f"Marked BALL RELEASE frame (highest + 1): {release_frame}")

# ---- Mark ball release (second pass) ----
marked_video_path = "output/bowling_analysis_with_skeleton_RELEASE_MARKED.mp4"
mark_event_on_video_at_point(
    input_video_path=output_full,
    output_video_path=marked_video_path,
    event_frame=release_frame,
    point_xy=release_wrist,
    label="BALL RELEASE (after highest wrist)"
)
print(f"✅ Release-marked video saved: {marked_video_path}")

# ---- Full-clip analysis CSV + plots ----
df = pd.DataFrame({
    "frame": frame_numbers,
    "time_sec": time_stamps,
    "elbow_angle": elbow_angles
})

# Angular velocity
vel = [0.0]
for i in range(1, len(elbow_angles)):
    dt = time_stamps[i] - time_stamps[i-1]
    vel.append((elbow_angles[i] - elbow_angles[i-1]) / dt if dt > 0 else 0.0)
df["angular_velocity"] = vel
df.to_csv("output/elbow_full_analysis.csv", index=False)

# Plots
plt.figure(figsize=(15, 9))

plt.subplot(2, 1, 1)
plt.plot(time_stamps, elbow_angles, linewidth=2, label="Elbow Angle")
plt.axvline(x=release_frame / fps, linestyle="--", linewidth=2, label="Release mark (highest+1)")
plt.xlabel("Time (s)")
plt.ylabel("Elbow Angle (deg)")
plt.title("Elbow Angle (Full Clip)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_stamps, vel, linewidth=2)
plt.axhline(y=0, alpha=0.3)
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (deg/s)")
plt.title("Angular Velocity (Full Clip)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output/elbow_full_plots.png", dpi=300, bbox_inches="tight")

print("\n✅ Done!")
print("🎥 Full annotated (NO TRAJECTORY):", output_full)
print("🎯 Release marked:", marked_video_path)
print("📊 Full CSV:", "output/elbow_full_analysis.csv")
print("📈 Full plots:", "output/elbow_full_plots.png")