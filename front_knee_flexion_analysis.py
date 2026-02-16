import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import os

# ============================================================
# FRONT KNEE FLEXION-EXTENSION ANALYSIS FOR CRICKET BOWLING
# ============================================================

# ---------------- Angle Function ----------------
def calculate_angle(a, b, c):
    """
    Calculate knee angle at point b (hip-knee-ankle)
    Returns angle in degrees
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return np.nan

    cosine_angle = np.dot(ba, bc) / denom
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle


def get_user_leg_preference():
    print("\n" + "="*50)
    print("LEG SELECTION FOR BOWLING ANALYSIS")
    print("="*50)
    print("Which leg should be tracked? (front leg = closer to batter)")
    print("1. Right leg")
    print("2. Left leg")
    print("3. Auto-detect (may be inconsistent)")

    while True:
        choice = input("\nEnter your choice (1, 2, or 3): ").strip()
        if choice == '1':
            return 'right'
        elif choice == '2':
            return 'left'
        elif choice == '3':
            return 'auto'
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def validate_keypoints(person_keypoints, preferred_leg):
    """
    COCO Keypoint indices:
    Right: hip=12 knee=14 ankle=16
    Left:  hip=11 knee=13 ankle=15
    """
    if preferred_leg == 'right':
        idxs = [12, 14, 16]
    else:
        idxs = [11, 13, 15]

    pts = person_keypoints[idxs]
    if not np.isfinite(pts).all():
        return None
    return pts[0], pts[1], pts[2]


def find_best_person_for_leg_with_tracking(persons_keypoints, preferred_leg, last_person_center=None):
    """
    Track the same person using nearest-hip to last_person_center
    """
    if len(persons_keypoints) == 0:
        return None

    hip_idx = 12 if preferred_leg == 'right' else 11

    if last_person_center is None:
        return 0

    distances = []
    for person in persons_keypoints:
        hip = person[hip_idx]
        if np.isfinite(hip).all():
            distances.append(np.linalg.norm(hip - last_person_center))
        else:
            distances.append(np.inf)

    return int(np.argmin(distances))


def detect_bowling_leg_with_motion(persons_keypoints, frame_width):
    """
    Choose person near center + valid legs
    """
    if len(persons_keypoints) == 0:
        return None

    best_person_idx = 0
    best_score = -1

    RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 12, 14, 16
    LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 11, 13, 15

    for i, person in enumerate(persons_keypoints):
        score = 0

        right_hip = person[RIGHT_HIP]
        left_hip = person[LEFT_HIP]

        # center preference
        if np.isfinite(right_hip).all():
            score += max(0, 200 - abs(right_hip[0] - frame_width/2) / 10)
        if np.isfinite(left_hip).all():
            score += max(0, 200 - abs(left_hip[0] - frame_width/2) / 10)

        # leg validity / length preference
        def leg_score(h, k, a):
            if np.isfinite(h).all() and np.isfinite(k).all() and np.isfinite(a).all():
                return np.linalg.norm(a - h) / 5
            return 0

        score += leg_score(person[RIGHT_HIP], person[RIGHT_KNEE], person[RIGHT_ANKLE])
        score += leg_score(person[LEFT_HIP], person[LEFT_KNEE], person[LEFT_ANKLE])

        if score > best_score:
            best_score = score
            best_person_idx = i

    return best_person_idx


def determine_bowling_leg(person_keypoints):
    RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 12, 14, 16
    LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 11, 13, 15

    r = (person_keypoints[RIGHT_HIP], person_keypoints[RIGHT_KNEE], person_keypoints[RIGHT_ANKLE])
    l = (person_keypoints[LEFT_HIP], person_keypoints[LEFT_KNEE], person_keypoints[LEFT_ANKLE])

    right_valid = all(np.isfinite(p).all() for p in r)
    left_valid  = all(np.isfinite(p).all() for p in l)

    if not right_valid and not left_valid:
        return None
    if right_valid and not left_valid:
        return 'right'
    if left_valid and not right_valid:
        return 'left'

    right_len = np.linalg.norm(person_keypoints[RIGHT_ANKLE] - person_keypoints[RIGHT_HIP])
    left_len  = np.linalg.norm(person_keypoints[LEFT_ANKLE] - person_keypoints[LEFT_HIP])
    return 'right' if right_len > left_len else 'left'


def draw_skeletal_points(frame, hip, knee, ankle, leg_side='right', total_detections=0):
    hip_color = (0, 255, 0)
    knee_color = (255, 0, 0)
    ankle_color = (0, 0, 255)
    line_color = (255, 255, 0)

    cv2.line(frame, tuple(hip.astype(int)), tuple(knee.astype(int)), line_color, 3)
    cv2.line(frame, tuple(knee.astype(int)), tuple(ankle.astype(int)), line_color, 3)

    cv2.circle(frame, tuple(hip.astype(int)), 8, hip_color, -1)
    cv2.circle(frame, tuple(knee.astype(int)), 8, knee_color, -1)
    cv2.circle(frame, tuple(ankle.astype(int)), 8, ankle_color, -1)

    cv2.putText(frame, f'Tracking: {leg_side.upper()} LEG', (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Detections: {total_detections}', (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


# ============================================================
# IMPROVED PHASE DETECTION (FFC + RELEASE)
# ============================================================

def compute_speed(positions, fps):
    p = np.asarray(positions, dtype=float)
    if len(p) < 2:
        return np.array([])
    dp = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return dp * fps  # px/s


def find_main_max_flexion(knee_angles, fps, late_start_ratio=0.35, smooth_w=9):
    ang = np.asarray(knee_angles, dtype=float)
    sm = savgol_filter(ang, smooth_w, 2) if len(ang) >= smooth_w else ang

    start = int(len(sm) * late_start_ratio)
    start = min(max(start, 0), len(sm)-1)

    peaks, props = find_peaks(sm[start:], prominence=max(1e-6, np.std(sm) * 0.5))
    if len(peaks) == 0:
        return int(np.argmax(sm[start:]) + start), sm

    prominences = props.get("prominences", np.ones(len(peaks)))
    best_local = peaks[int(np.argmax(prominences))]
    return int(best_local + start), sm


def detect_ffc_from_ankle(ankle_positions, fps, search_end_idx,
                         low_speed_ratio=0.25, min_stable_frames=6):
    if len(ankle_positions) < (min_stable_frames + 2):
        return None

    speed = compute_speed(ankle_positions, fps)  # N-1
    if len(speed) == 0:
        return None

    thr = max(5.0, np.median(speed) * low_speed_ratio)
    stable = speed < thr

    end = max(0, min(search_end_idx - 1, len(stable) - 1))
    if end < min_stable_frames:
        return None

    last_ffc = None
    run_len = 0
    for i in range(0, end + 1):
        if stable[i]:
            run_len += 1
            if run_len >= min_stable_frames:
                last_ffc = (i + 1) - (min_stable_frames - 1)
        else:
            run_len = 0

    return last_ffc


def detect_release_from_knee_velocity(knee_angles, fps, max_flexion_idx,
                                     max_search_seconds=0.7, smooth_w=9):
    ang = np.asarray(knee_angles, dtype=float)
    sm = savgol_filter(ang, smooth_w, 2) if len(ang) >= smooth_w else ang

    vel = np.gradient(sm) * fps  # deg/s

    start = max_flexion_idx
    end = min(len(vel) - 1, int(max_flexion_idx + max_search_seconds * fps))
    if end <= start:
        return None

    window = vel[start:end+1]
    rel_local = int(np.argmin(window))  # most negative
    return int(start + rel_local)


def detect_bowling_phases_v2(knee_angles, ankle_positions, fps):
    if len(knee_angles) < 10:
        return None, None, None

    max_flexion_idx, _ = find_main_max_flexion(knee_angles, fps)

    ffc_idx = detect_ffc_from_ankle(
        ankle_positions=ankle_positions,
        fps=fps,
        search_end_idx=max_flexion_idx,
        low_speed_ratio=0.25,
        min_stable_frames=max(4, int(0.12 * fps))
    )

    release_idx = detect_release_from_knee_velocity(
        knee_angles=knee_angles,
        fps=fps,
        max_flexion_idx=max_flexion_idx,
        max_search_seconds=0.7
    )

    # enforce ordering
    if ffc_idx is not None and release_idx is not None and ffc_idx >= release_idx:
        ffc_idx = None

    return ffc_idx, max_flexion_idx, release_idx


def calculate_extension_velocity(knee_angles, time_stamps):
    if len(knee_angles) < 2:
        return []

    angular_velocity = []
    for i in range(1, len(knee_angles)):
        dt = time_stamps[i] - time_stamps[i-1]
        dangle = knee_angles[i] - knee_angles[i-1]
        angular_velocity.append(dangle / dt if dt > 0 else 0.0)
    return angular_velocity


def calculate_range_of_motion(knee_angles, ffc_idx, ball_release_idx):
    if ffc_idx is None or ball_release_idx is None:
        return None, None, None

    max_angle = float(np.max(knee_angles[ffc_idx:ball_release_idx+1]))
    release_angle = float(knee_angles[ball_release_idx])
    rom = max_angle - release_angle
    return max_angle, release_angle, rom


# ============================================================
# MAIN
# ============================================================

print("\n" + "="*50)
print("Loading YOLO Pose Detection Model...")
print("="*50)
model = YOLO("yolov8n-pose.pt")
print("✅ Model loaded successfully")

preferred_leg = get_user_leg_preference()
print(f"✅ Will track: {preferred_leg} leg")
if preferred_leg == 'auto':
    print("⚠️  Warning: Auto-detect mode may result in inconsistent tracking between legs")
else:
    print(f"✅ Locked to {preferred_leg} leg for consistent tracking")

possible_videos = ["1.mp4", "1.mov", "nipuledit.mp4", "new.MOV", "nipul.mov", "your_video.mp4"]
video_path = None

print("\nChecking for available video files...")
for video in possible_videos:
    if os.path.exists(video):
        print(f"Found: {video}")
        cap_test = cv2.VideoCapture(video)
        if cap_test.isOpened():
            video_path = video
            cap_test.release()
            break
        cap_test.release()

if video_path is None:
    print("\n❌ No video file found in current directory.")
    for f in os.listdir("."):
        if f.endswith(('.mp4', '.MOV', '.avi', '.mov', '.mkv')):
            print(" -", f)
    raise SystemExit

print(f"✅ Using video: {video_path}")
os.makedirs("output", exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = float(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video FPS: {fps}")
print(f"Video Resolution: {frame_width}x{frame_height}")

output_path = "output/knee_flexion_analysis_with_skeleton.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# data lists (ONLY FOR DETECTED FRAMES)
knee_angles = []
time_stamps = []
hip_positions = []
knee_positions = []
ankle_positions = []
detected_leg_side = []

successful_detections = 0
last_valid_person_center = None

# For later: we also keep the output frames so we can add phase markers after detection
# (optional; if memory is an issue, you can skip storing and re-read video later)
stored_frames = []

frame_idx = 0
print("\nProcessing video frames...")
print("="*50)

locked_leg = preferred_leg  # if auto, will become 'left'/'right' after first good detection

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    detected_keypoints = False

    if results and results[0].keypoints is not None:
        kpts = results[0].keypoints.xy.cpu().numpy()  # (persons, 17, 2)

        if len(kpts) > 0:
            if locked_leg == 'auto':
                # first valid detection decides leg, then lock
                person_idx = detect_bowling_leg_with_motion(kpts, frame_width)
                if person_idx is not None:
                    person = kpts[person_idx]
                    leg_side = determine_bowling_leg(person)
                    if leg_side is not None:
                        kp = validate_keypoints(person, leg_side)
                        if kp is not None:
                            hip, knee, ankle = kp
                            detected_keypoints = True
                            locked_leg = leg_side
                            last_valid_person_center = hip.copy()
                            print(f"🔒 Locked to {leg_side} leg for consistency")
            else:
                leg_side = locked_leg
                person_idx = find_best_person_for_leg_with_tracking(kpts, locked_leg, last_valid_person_center)
                if person_idx is not None:
                    person = kpts[person_idx]
                    kp = validate_keypoints(person, locked_leg)
                    if kp is not None:
                        hip, knee, ankle = kp
                        detected_keypoints = True
                        last_valid_person_center = hip.copy()

    if detected_keypoints:
        successful_detections += 1

        draw_skeletal_points(frame, hip, knee, ankle, locked_leg, successful_detections)

        angle = calculate_angle(hip, knee, ankle)
        if np.isfinite(angle):
            knee_angles.append(float(angle))
            time_stamps.append(frame_idx / fps)

            hip_positions.append(hip)
            knee_positions.append(knee)
            ankle_positions.append(ankle)
            detected_leg_side.append(locked_leg)

            cv2.putText(frame, f'Knee Angle: {angle:.1f}°', (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    else:
        cv2.putText(frame, f'No {locked_leg if locked_leg != "auto" else "suitable"} leg detected',
                    (30, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)
    stored_frames.append(frame.copy())

    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"Processed frame {frame_idx}")

cap.release()
out.release()

if len(knee_angles) == 0:
    print("❌ No knee angles detected. Check your video and model.")
    raise SystemExit

print(f"\n✅ Total frames with detection: {len(knee_angles)}")
print(f"✅ Detection rate: {successful_detections}/{frame_idx} frames ({100*successful_detections/frame_idx:.1f}%)")
print(f"✅ Output video saved to: {output_path}")
print(f"✅ Tracking leg: {locked_leg}")

# ============================================================
# PHASE DETECTION & METRICS (MUST COME BEFORE SAVING)
# ============================================================

print("\n" + "="*50)
print("ANALYZING BOWLING PHASES...")
print("="*50)

ffc_idx, max_flexion_idx, ball_release_idx = detect_bowling_phases_v2(
    knee_angles=knee_angles,
    ankle_positions=ankle_positions,
    fps=fps
)

angular_velocity = calculate_extension_velocity(knee_angles, time_stamps)

delivery_duration = None
extension_duration = None
max_extension_velocity = None
max_flexion_angle = None
release_angle = None
rom = None

if ffc_idx is not None and ball_release_idx is not None:
    delivery_duration = time_stamps[ball_release_idx] - time_stamps[ffc_idx]

if max_flexion_idx is not None and ball_release_idx is not None:
    extension_duration = time_stamps[ball_release_idx] - time_stamps[max_flexion_idx]

if len(angular_velocity) > 0:
    max_extension_velocity = float(np.min(angular_velocity))  # most negative

if ffc_idx is not None and ball_release_idx is not None:
    max_flexion_angle, release_angle, rom = calculate_range_of_motion(
        knee_angles, ffc_idx, ball_release_idx
    )

# Print phase results
print(f"FFC idx:        {ffc_idx}")
print(f"Max flex idx:   {max_flexion_idx}")
print(f"Release idx:    {ball_release_idx}")
if ffc_idx is not None:
    print(f"FFC time:       {time_stamps[ffc_idx]:.3f}s")
if ball_release_idx is not None:
    print(f"Release time:   {time_stamps[ball_release_idx]:.3f}s")

# ============================================================
# SAVE DATA TO CSV
# ============================================================

print("\n" + "="*50)
print("SAVING ANALYSIS DATA...")
print("="*50)

data_df = pd.DataFrame({
    'time': time_stamps,
    'knee_angle': knee_angles,
    'leg_side': detected_leg_side,
    'hip_x': [pos[0] for pos in hip_positions],
    'hip_y': [pos[1] for pos in hip_positions],
    'knee_x': [pos[0] for pos in knee_positions],
    'knee_y': [pos[1] for pos in knee_positions],
    'ankle_x': [pos[0] for pos in ankle_positions],
    'ankle_y': [pos[1] for pos in ankle_positions]
})

angular_vel_padded = [0.0] + angular_velocity
# ensure same length
angular_vel_padded = angular_vel_padded[:len(data_df)]
data_df['angular_velocity'] = angular_vel_padded

data_df.to_csv('output/knee_flexion_analysis.csv', index=False)
print("✅ Detailed data saved to: output/knee_flexion_analysis.csv")

summary_rows = [
    ('Total Frames Analyzed', len(knee_angles), 'frames'),
    ('Detection Rate', f"{100*successful_detections/frame_idx:.1f}%", 'percentage'),
    ('Video FPS', fps, 'fps'),
    ('Total Video Duration', f"{frame_idx/fps:.2f}", 'seconds'),
    ('FFC Frame (detected series index)', ffc_idx if ffc_idx is not None else 'N/A', 'index'),
    ('Max Flexion Frame (detected series index)', max_flexion_idx if max_flexion_idx is not None else 'N/A', 'index'),
    ('Ball Release Frame (detected series index)', ball_release_idx if ball_release_idx is not None else 'N/A', 'index'),
    ('Total Delivery Duration', f"{delivery_duration:.3f}" if delivery_duration is not None else 'N/A', 'seconds'),
    ('Extension Phase Duration', f"{extension_duration:.3f}" if extension_duration is not None else 'N/A', 'seconds'),
    ('Angle at FFC', f"{knee_angles[ffc_idx]:.1f}" if ffc_idx is not None else 'N/A', 'degrees'),
    ('Maximum Flexion Angle', f"{max_flexion_angle:.1f}" if max_flexion_angle is not None else 'N/A', 'degrees'),
    ('Release Angle', f"{release_angle:.1f}" if release_angle is not None else 'N/A', 'degrees'),
    ('Knee ROM (Flexion-Extension)', f"{rom:.1f}" if rom is not None else 'N/A', 'degrees'),
    ('Max Extension Velocity', f"{abs(max_extension_velocity):.1f}" if max_extension_velocity is not None else 'N/A', 'deg/s'),
    ('Tracked Leg (Locked)', locked_leg, 'leg')
]

summary_df = pd.DataFrame(summary_rows, columns=['Metric', 'Value', 'Unit'])
summary_df.to_csv('output/knee_metrics_summary.csv', index=False)
print("✅ Metrics summary saved to: output/knee_metrics_summary.csv")

# ============================================================
# PLOTTING
# ============================================================

print("\nGenerating analysis plots...")

plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(time_stamps, knee_angles, linewidth=2, label='Knee Angle')
if ffc_idx is not None:
    plt.axvline(time_stamps[ffc_idx], linestyle='--', linewidth=2, label=f'FFC ({time_stamps[ffc_idx]:.3f}s)')
if max_flexion_idx is not None:
    plt.axvline(time_stamps[max_flexion_idx], linestyle='--', linewidth=2, label=f'Max Flex ({time_stamps[max_flexion_idx]:.3f}s)')
if ball_release_idx is not None:
    plt.axvline(time_stamps[ball_release_idx], linestyle='--', linewidth=2, color='orange', label=f'Release ({time_stamps[ball_release_idx]:.3f}s)')
plt.xlabel("Time (s)")
plt.ylabel("Knee Angle (deg)")
plt.title("Front Knee Flexion–Extension (Detected Frames)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 1, 2)
if len(angular_velocity) > 0:
    plt.plot(time_stamps[1:], angular_velocity, linewidth=2, label="Angular Velocity (deg/s)")
    plt.axhline(0, alpha=0.3)
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (deg/s)")
plt.title("Knee Angular Velocity (Negative = Extension)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("output/knee_flexion_analysis.png", dpi=300, bbox_inches="tight")
print("✅ Plot saved to: output/knee_flexion_analysis.png")

print(f"\n{'='*50}")
print("✅ ANALYSIS COMPLETE!")
print(f"{'='*50}")
print("Outputs:")
print(" • output/knee_flexion_analysis_with_skeleton.mp4")
print(" • output/knee_flexion_analysis.csv")
print(" • output/knee_metrics_summary.csv")
print(" • output/knee_flexion_analysis.png")
