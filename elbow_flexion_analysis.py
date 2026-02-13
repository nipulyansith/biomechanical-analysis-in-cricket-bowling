import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
import os

# ---------------- Angle Function ----------------
def calculate_angle(a, b, c):
    """
    Calculate elbow angle at point b (shoulder-elbow-wrist)
    Returns angle in degrees (0° = fully extended, 180° = fully flexed)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    return angle

def get_user_hand_preference():
    """
    Ask user which hand to track for bowling analysis
    """
    print("\n" + "="*50)
    print("HAND SELECTION FOR BOWLING ANALYSIS")
    print("="*50)
    print("Which hand should be tracked for bowling analysis?")
    print("1. Right hand")
    print("2. Left hand") 
    print("3. Auto-detect (may be inconsistent)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1, 2, or 3): ").strip()
            if choice == '1':
                return 'right'
            elif choice == '2':
                return 'left'
            elif choice == '3':
                return 'auto'
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit()
        except:
            print("Invalid input. Please enter 1, 2, or 3.")

def validate_keypoints(person_keypoints, preferred_hand):
    if preferred_hand == 'right':
        idxs = [6, 8, 10]
    else:
        idxs = [5, 7, 9]

    pts = person_keypoints[idxs]

    # Only check for NaN / Inf
    if not np.isfinite(pts).all():
        return None

    return pts[0], pts[1], pts[2]


def find_best_person_for_hand_with_tracking(persons_keypoints, preferred_hand, frame_width, last_person_center=None):
    """
    Person locking: once selected, always return the same person
    """
    if len(persons_keypoints) == 0:
        return None

    # If we already have a tracked person, reuse them
    if last_person_center is not None:
        distances = []
        for i, person in enumerate(persons_keypoints):
            shoulder = person[6] if preferred_hand == 'right' else person[5]
            if np.isfinite(shoulder).all():
                dist = np.linalg.norm(shoulder - last_person_center)
            else:
                dist = np.inf
            distances.append(dist)

        return int(np.argmin(distances))

    # First frame: just take the first detected person
    return 0


def find_best_person_for_hand(persons_keypoints, preferred_hand, frame_width):
    """
    Find the person with the best detection for the preferred hand
    """
    if len(persons_keypoints) == 0:
        return None
        
    best_person_idx = None
    best_score = -1
    
    for i, person in enumerate(persons_keypoints):
        keypoints_result = validate_keypoints(person, preferred_hand)
        if keypoints_result is not None:
            shoulder, elbow, wrist = keypoints_result
            # Calculate a quality score for this person's detection
            score = (np.linalg.norm(elbow - shoulder) + np.linalg.norm(wrist - elbow))
            if score > best_score:
                best_score = score
                best_person_idx = i
    
    return best_person_idx

# Legacy functions for auto-detect mode
def detect_bowling_hand_with_ball(persons_keypoints, frame_width):
    """
    Legacy function for auto-detect mode
    """
    if len(persons_keypoints) == 0:
        return None
    
    best_person_idx = 0
    best_score = -1
    
    for i, person in enumerate(persons_keypoints):
        score = 0
        
        # COCO keypoint indices
        RIGHT_SHOULDER = 6
        RIGHT_ELBOW = 8
        RIGHT_WRIST = 10
        LEFT_SHOULDER = 5
        LEFT_ELBOW = 7
        LEFT_WRIST = 9
        
        # Check right arm keypoints
        right_shoulder = person[RIGHT_SHOULDER]
        right_elbow = person[RIGHT_ELBOW]
        right_wrist = person[RIGHT_WRIST]
        
        # Check left arm keypoints
        left_shoulder = person[LEFT_SHOULDER]
        left_elbow = person[LEFT_ELBOW]
        left_wrist = person[LEFT_WRIST]
        
        # Prefer person closer to center of frame
        if np.all(right_shoulder > 0):
            distance_from_center = abs(right_shoulder[0] - frame_width / 2)
            score += max(0, 100 - distance_from_center / 10)
        
        # Check if right arm is more extended (typical bowling arm)
        if (np.all(right_shoulder > 0) and np.all(right_elbow > 0) and np.all(right_wrist > 0)):
            arm_length = np.linalg.norm(right_wrist - right_shoulder)
            score += arm_length / 10
        
        # Check if left arm is more extended (left-handed bowler)
        if (np.all(left_shoulder > 0) and np.all(left_elbow > 0) and np.all(left_wrist > 0)):
            arm_length = np.linalg.norm(left_wrist - left_shoulder)
            score += arm_length / 10
        
        if score > best_score:
            best_score = score
            best_person_idx = i
    
    return best_person_idx

def determine_bowling_arm(person_keypoints):
    """
    Legacy function for auto-detect mode
    """
    # COCO keypoint indices
    RIGHT_SHOULDER = 6
    RIGHT_ELBOW = 8
    RIGHT_WRIST = 10
    LEFT_SHOULDER = 5
    LEFT_ELBOW = 7
    LEFT_WRIST = 9
    
    right_shoulder = person_keypoints[RIGHT_SHOULDER]
    right_elbow = person_keypoints[RIGHT_ELBOW]
    right_wrist = person_keypoints[RIGHT_WRIST]
    
    left_shoulder = person_keypoints[LEFT_SHOULDER]
    left_elbow = person_keypoints[LEFT_ELBOW]
    left_wrist = person_keypoints[LEFT_WRIST]
    
    right_arm_valid = np.all(right_shoulder > 0) and np.all(right_elbow > 0) and np.all(right_wrist > 0)
    left_arm_valid = np.all(left_shoulder > 0) and np.all(left_elbow > 0) and np.all(left_wrist > 0)
    
    if not right_arm_valid and not left_arm_valid:
        return None
    elif right_arm_valid and not left_arm_valid:
        return 'right'
    elif left_arm_valid and not right_arm_valid:
        return 'left'
    else:
        # Both arms detected, choose the one with greater extension
        right_extension = np.linalg.norm(right_wrist - right_shoulder)
        left_extension = np.linalg.norm(left_wrist - left_shoulder)
        return 'right' if right_extension > left_extension else 'left'

def draw_skeletal_points(frame, shoulder, elbow, wrist, arm_side='right', frame_count=0, total_detections=0):
    """
    Draw skeletal points and connections on the frame
    """
    # Colors for different points
    shoulder_color = (0, 255, 0)    # Green
    elbow_color = (255, 0, 0)       # Blue  
    wrist_color = (0, 0, 255)       # Red
    line_color = (255, 255, 0)      # Cyan
    
    # Draw connections (lines)
    cv2.line(frame, tuple(shoulder.astype(int)), tuple(elbow.astype(int)), line_color, 3)
    cv2.line(frame, tuple(elbow.astype(int)), tuple(wrist.astype(int)), line_color, 3)
    
    # Draw points (circles)
    cv2.circle(frame, tuple(shoulder.astype(int)), 8, shoulder_color, -1)
    cv2.circle(frame, tuple(elbow.astype(int)), 8, elbow_color, -1)
    cv2.circle(frame, tuple(wrist.astype(int)), 8, wrist_color, -1)
    
    # Add labels
    cv2.putText(frame, 'Shoulder', (shoulder[0].astype(int) + 10, shoulder[1].astype(int) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, shoulder_color, 2)
    cv2.putText(frame, 'Elbow', (elbow[0].astype(int) + 10, elbow[1].astype(int) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, elbow_color, 2)
    cv2.putText(frame, 'Wrist', (wrist[0].astype(int) + 10, wrist[1].astype(int) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, wrist_color, 2)
    
    # Add arm side indicator and tracking info
    cv2.putText(frame, f'Tracking: {arm_side.upper()} ARM', (30, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Detections: {total_detections}', (30, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add consistency indicator
    if arm_side == 'right':
        consistency_color = (0, 255, 0)  # Green for consistent
    else:
        consistency_color = (0, 255, 255)  # Yellow for left arm
    
    cv2.circle(frame, (frame.shape[1] - 50, 50), 20, consistency_color, -1)
    cv2.putText(frame, arm_side[0].upper(), (frame.shape[1] - 60, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

def detect_bowling_phases(elbow_angles, time, smoothing_window=5):
    """
    Detect key bowling phases based on elbow angle patterns
    Returns: arm_back_time, max_flexion_time, ball_release_time
    """
    # Smooth the signal to reduce noise
    if len(elbow_angles) > smoothing_window:
        smoothed_angles = savgol_filter(elbow_angles, smoothing_window, 2)
    else:
        smoothed_angles = elbow_angles.copy()
    
    # Find peaks (maximum flexion) and valleys (maximum extension)
    peaks, _ = find_peaks(smoothed_angles, height=np.mean(smoothed_angles))
    valleys, _ = find_peaks(-np.array(smoothed_angles), height=-np.mean(smoothed_angles))
    
    arm_back_idx = None
    max_flexion_idx = None
    ball_release_idx = None
    
    if len(peaks) > 0:
        # Arm-back position: usually the first significant peak (max flexion)
        arm_back_idx = peaks[0] if len(peaks) > 0 else 0
        max_flexion_idx = peaks[np.argmax([smoothed_angles[p] for p in peaks])]
        
        # Ball release: find the steepest decline after max flexion
        if max_flexion_idx < len(smoothed_angles) - 10:
            post_flexion = smoothed_angles[max_flexion_idx:]
            # Find minimum after max flexion
            min_idx_relative = np.argmin(post_flexion)
            ball_release_idx = max_flexion_idx + min_idx_relative
    
    return arm_back_idx, max_flexion_idx, ball_release_idx
    
    return arm_back_idx, max_flexion_idx, ball_release_idx

def calculate_extension_velocity(elbow_angles, time):
    """
    Calculate angular velocity of elbow extension
    """
    if len(elbow_angles) < 2:
        return []
    
    angular_velocity = []
    for i in range(1, len(elbow_angles)):
        dt = time[i] - time[i-1]
        dangle = elbow_angles[i] - elbow_angles[i-1]
        if dt > 0:
            angular_velocity.append(dangle / dt)
        else:
            angular_velocity.append(0)
    
    return angular_velocity

def detect_ball_release_velocity(wrist_positions, time_stamps, fps=30, velocity_threshold=100):
    """
    Detect ball release frame by finding peak wrist velocity
    Returns: release_frame_idx, release_velocity
    """
    if len(wrist_positions) < 3:
        return None, None
    
    wrist_x = np.array([pos[0] for pos in wrist_positions])
    wrist_y = np.array([pos[1] for pos in wrist_positions])
    
    # Calculate displacement between frames
    dx = np.diff(wrist_x)
    dy = np.diff(wrist_y)
    
    # Calculate wrist velocity (pixels per frame)
    wrist_velocity = np.sqrt(dx**2 + dy**2)
    
    # Find peak velocity (ball release typically occurs at maximum wrist speed)
    if len(wrist_velocity) > 0:
        release_idx = np.argmax(wrist_velocity)
        peak_velocity = wrist_velocity[release_idx]
        return release_idx, peak_velocity
    
    return None, None

def check_icc_legality(elbow_angles, max_flexion_idx, ball_release_idx, icc_limit=15.0):
    """
    Check if elbow extension complies with ICC rules
    ICC rules: Elbow extension at ball release should be ≤ 15°
    
    Returns: is_legal, extension_angle, violation_amount
    """
    if max_flexion_idx is None or ball_release_idx is None:
        return None, None, None
    
    # Get angles at key positions
    max_flexion_angle = elbow_angles[max_flexion_idx]
    release_angle = elbow_angles[ball_release_idx]
    
    # Extension amount (positive = straightening)
    extension_angle = max_flexion_angle - release_angle
    
    # Check legality
    is_legal = extension_angle <= icc_limit
    violation_amount = max(0, extension_angle - icc_limit)
    
    return is_legal, extension_angle, violation_amount

# ---------------- Load YOLO Pose ----------------
model = YOLO("yolov8n-pose.pt")

# Get user preference for which hand to track
preferred_hand = get_user_hand_preference()
print(f"✅ Will track: {preferred_hand} hand")

if preferred_hand == 'auto':
    print("⚠️  Warning: Auto-detect mode may result in inconsistent tracking between hands")
else:
    print(f"✅ Locked to {preferred_hand} hand for consistent tracking")

# Try to find an available video file
import os
possible_videos = ["1.mp4", "1.mov", "nipuledit.mp4", "new.MOV", "nipul.mov", "your_video.mp4"]
video_path = None

print("Checking for available video files...")
for video in possible_videos:
    if os.path.exists(video):
        print(f"Found: {video}")
        test_cap = cv2.VideoCapture(video)
        if test_cap.isOpened():
            video_path = video
            test_cap.release()
            break
        test_cap.release()

if video_path is None:
    print("\n❌ No video file found in current directory.")
    print("Available files in current directory:")
    for file in os.listdir("."):
        if file.endswith(('.mp4', '.MOV', '.avi', '.mov', '.mkv')):
            print(f"  - {file}")
    
    print("\nPlease:")
    print("1. Copy your cricket bowling video to this directory, or")
    print("2. Update the 'possible_videos' list in the script with your video filename")
    exit()

print(f"✅ Using video: {video_path}")

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video FPS: {fps}")
print(f"Video Resolution: {frame_width}x{frame_height}")

# Setup video writer for output
output_path = 'output/bowling_analysis_with_skeleton.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

elbow_angles = []
time_stamps = []
shoulder_positions = []
elbow_positions = []
wrist_positions = []
detected_arm_side = []
successful_detections = 0
last_valid_person_center = None  # Track person position for consistency

frame_idx = 0
# ---- Arm freeze near full extension (CRITICAL FIX) ----
ANGLE_LOCK_THRESHOLD = 172.0  # degrees
arm_frozen = False
frozen_joints = None


print("Processing video frames...")
print("Debug: Logging detection details around frame 62...")
last_valid_joints = None
last_valid_person_center = None
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO pose detection
    results = model(frame, verbose=False)
    
    detected_keypoints = False
    debug_frame = (frame_idx >= 60 and frame_idx <= 65)  # Debug around frame 62

    if results[0].keypoints is not None:
        kpts = results[0].keypoints.xy.cpu().numpy()

        if debug_frame:
            print(f"\n--- FRAME {frame_idx} DEBUG ---")
            print(f"Number of detected persons: {len(kpts)}")
            print(f"Preferred hand: {preferred_hand}")
            print(f"Last valid person center: {last_valid_person_center}")

        if len(kpts) > 0:
            if preferred_hand == 'auto':
                # Auto-detect mode (legacy behavior) - but force consistency after first detection
                if len(detected_arm_side) == 0:
                    # First detection - determine arm
                    bowling_person_idx = detect_bowling_hand_with_ball(kpts, frame_width)
                    
                    if bowling_person_idx is not None:
                        person = kpts[bowling_person_idx]
                        arm_side = determine_bowling_arm(person)
                        
                        if arm_side is not None:
                            keypoints_result = validate_keypoints(person, arm_side)
                            if keypoints_result is not None:
                                shoulder, elbow, wrist = keypoints_result
                                detected_keypoints = True
                                # Lock the arm side for consistency
                                preferred_hand = arm_side
                                last_valid_person_center = shoulder.copy()
                                print(f"🔒 Locked to {arm_side} arm for consistency")
                else:
                    # Use the locked arm side
                    arm_side = preferred_hand
                    bowling_person_idx = find_best_person_for_hand_with_tracking(kpts, preferred_hand, frame_width, last_valid_person_center)
                    
                    if bowling_person_idx is not None:
                        person = kpts[bowling_person_idx]
                        keypoints_result = validate_keypoints(person, preferred_hand)
                        
                        if keypoints_result is not None:
                            shoulder, elbow, wrist = keypoints_result
                            detected_keypoints = True
                            last_valid_person_center = shoulder.copy()
            else:
                # Fixed hand mode - ensure absolute consistency
                arm_side = preferred_hand  # NEVER change this
                
                if debug_frame:
                    print(f"Fixed hand mode - arm_side FORCED to: {arm_side}")
                
                bowling_person_idx = find_best_person_for_hand_with_tracking(kpts, preferred_hand, frame_width, last_valid_person_center)
                
                if debug_frame:
                    print(f"Selected person index: {bowling_person_idx}")
                
                if bowling_person_idx is not None:
                    person = kpts[bowling_person_idx]
                    keypoints_result = validate_keypoints(person, preferred_hand)
                    
                    if keypoints_result is not None:
                        shoulder, elbow, wrist = keypoints_result
                        detected_keypoints = True
                        last_valid_person_center = shoulder.copy()

        # Handle no detections
        if not detected_keypoints:
            detected_keypoints = False
    
    if results[0].keypoints is None and debug_frame:
        print("No keypoints detected by YOLO")

    if detected_keypoints:
        successful_detections += 1
        
        # CRITICAL: Ensure arm_side is always the preferred_hand in fixed mode
        if preferred_hand != 'auto':
            arm_side = preferred_hand  # Force this assignment
        
        if debug_frame:
            print(f"Drawing with arm_side: {arm_side}")
        
        # Draw skeletal points on the frame
        draw_skeletal_points(frame, shoulder, elbow, wrist, arm_side, frame_idx, successful_detections)
        
        # Calculate angle and store data
        angle = calculate_angle(shoulder, elbow, wrist)
        elbow_angles.append(angle)
        time_stamps.append(frame_idx / fps)
        
        # Store positions for additional analysis
        shoulder_positions.append(shoulder)
        elbow_positions.append(elbow)
        wrist_positions.append(wrist)
        detected_arm_side.append(arm_side)
        
        # Add angle information to frame
        cv2.putText(frame, f'Elbow Angle: {angle:.1f}°', (30, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Frame: {frame_idx}', (30, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        if debug_frame:
            print(f"❌ No keypoints detected")
        # No detection - add warning to frame
        cv2.putText(frame, f'No {preferred_hand if preferred_hand != "auto" else "suitable"} hand detected', (30, frame_height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write frame to output video
    out.write(frame)
    
    frame_idx += 1
    
    # Print progress every 30 frames
    if frame_idx % 30 == 0:
        print(f"Processed frame {frame_idx}")

cap.release()
out.release()

if len(elbow_angles) == 0:
    print("❌ No elbow angles detected. Check your video and model.")
    print("Suggestions:")
    print("  • Try the other hand option")
    print("  • Check if the person is clearly visible in the video")
    print("  • Ensure good lighting and clear pose visibility")
    exit()

print(f"✅ Total frames with detection: {len(elbow_angles)}")
print(f"✅ Detection rate: {successful_detections}/{frame_idx} frames ({100*successful_detections/frame_idx:.1f}%)")
print(f"✅ Output video saved to: {output_path}")

# Print arm detection statistics
if detected_arm_side:
    right_count = detected_arm_side.count('right')
    left_count = detected_arm_side.count('left')
    
    # Debug: Print all detected arm sides to see if there are any inconsistencies
    print(f"\nDEBUG: All detected arm sides: {detected_arm_side}")
    print(f"DEBUG: Frames where 'left' was detected: {[i for i, side in enumerate(detected_arm_side) if side == 'left']}")
    
    if preferred_hand != 'auto':
        consistency = (max(right_count, left_count) / len(detected_arm_side)) * 100
        print(f"✅ Tracking consistency: {consistency:.1f}% ({preferred_hand} hand)")
        
        if consistency < 100:
            print(f"❌ WARNING: Found {left_count} left hand detections when tracking right hand!")
            print("This should not happen with the current fix.")
    else:
        print(f"Detected bowling arm - Right: {right_count} frames, Left: {left_count} frames")
        if right_count > 0 and left_count > 0:
            print("⚠️  Warning: Both hands detected - consider using fixed hand mode for consistency")

# ---------------- Phase Detection ----------------
arm_back_idx, max_flexion_idx, ball_release_idx = detect_bowling_phases(elbow_angles, time_stamps)

# Detect ball release using wrist velocity method as backup
release_idx_velocity, peak_wrist_velocity = detect_ball_release_velocity(wrist_positions, time_stamps, fps)

# Use velocity-based detection if it's more reliable
if release_idx_velocity is not None and (ball_release_idx is None or abs(release_idx_velocity - max_flexion_idx) < abs(ball_release_idx - max_flexion_idx)):
    ball_release_idx = release_idx_velocity

# ---------------- Calculate Metrics ----------------
angular_velocity = calculate_extension_velocity(elbow_angles, time_stamps)

# Calculate timing metrics
delivery_duration = None
extension_duration = None
max_extension_velocity = None
extension_range = None

if arm_back_idx is not None and ball_release_idx is not None:
    delivery_duration = time_stamps[ball_release_idx] - time_stamps[arm_back_idx]
    print(f"\n{'='*50}")
    print(f"BOWLING PHASE ANALYSIS")
    print(f"{'='*50}")
    print(f"Arm-back position time:      {time_stamps[arm_back_idx]:.3f}s (Frame {arm_back_idx})")
    if max_flexion_idx is not None:
        print(f"Maximum flexion time:        {time_stamps[max_flexion_idx]:.3f}s (Frame {max_flexion_idx})")
    print(f"Ball release time:           {time_stamps[ball_release_idx]:.3f}s (Frame {ball_release_idx})")
    print(f"Total delivery duration:     {delivery_duration:.3f}s")
    
    if max_flexion_idx is not None:
        extension_duration = time_stamps[ball_release_idx] - time_stamps[max_flexion_idx]
        print(f"Extension phase duration:    {extension_duration:.3f}s")
        
        # Calculate extension angle range
        max_flexion_angle = elbow_angles[max_flexion_idx]
        release_angle = elbow_angles[ball_release_idx]
        extension_range = abs(release_angle - max_flexion_angle)
        print(f"\n{'='*50}")
        print(f"ELBOW EXTENSION ANALYSIS")
        print(f"{'='*50}")
        print(f"Maximum flexion angle:       {max_flexion_angle:.1f}°")
        print(f"Release angle:               {release_angle:.1f}°")
        print(f"Elbow extension range:       {extension_range:.1f}°")
        
        # ICC Legality Check
        is_legal, extension_angle, violation = check_icc_legality(elbow_angles, max_flexion_idx, ball_release_idx, icc_limit=15.0)
        if is_legal is not None:
            status = "✅ LEGAL" if is_legal else "❌ ILLEGAL"
            print(f"\nICC Legality Check (≤15°):   {status}")
            print(f"Extension amount:            {extension_angle:.1f}°")
            if not is_legal:
                print(f"Violation amount:            {violation:.1f}° (exceeds limit by {violation:.1f}°)")

if peak_wrist_velocity is not None:
    print(f"\nPeak wrist velocity:         {peak_wrist_velocity:.1f} pixels/frame")

if len(angular_velocity) > 0:
    max_extension_velocity = min(angular_velocity)  # Most negative = fastest extension
    print(f"Maximum extension velocity:  {abs(max_extension_velocity):.1f}°/s")

# ---------------- Save Data ----------------
# Create output directory if it doesn't exist
import os
os.makedirs('output', exist_ok=True)

# Save detailed data to CSV
data_df = pd.DataFrame({
    'time': time_stamps,
    'elbow_angle': elbow_angles,
    'arm_side': detected_arm_side,
    'shoulder_x': [pos[0] for pos in shoulder_positions],
    'shoulder_y': [pos[1] for pos in shoulder_positions],
    'elbow_x': [pos[0] for pos in elbow_positions],
    'elbow_y': [pos[1] for pos in elbow_positions],
    'wrist_x': [pos[0] for pos in wrist_positions],
    'wrist_y': [pos[1] for pos in wrist_positions]
})

# Add angular velocity (one less data point)
angular_vel_padded = [0] + angular_velocity  # Pad with 0 for first frame
data_df['angular_velocity'] = angular_vel_padded

data_df.to_csv('output/elbow_flexion_analysis.csv', index=False)

# Create a summary metrics file
summary_data = {
    'Metric': [],
    'Value': [],
    'Unit': []
}

summary_data['Metric'].extend([
    'Total Frames Analyzed',
    'Detection Rate',
    'Video FPS',
    'Total Video Duration',
    'Arm-Back Frame',
    'Max Flexion Frame',
    'Ball Release Frame',
    'Total Delivery Duration',
    'Extension Phase Duration',
    'Max Flexion Angle',
    'Release Angle',
    'Elbow Extension Range',
    'Max Extension Velocity',
    'Bowling Arm (Locked)',
])

summary_data['Value'].extend([
    len(elbow_angles),
    f"{100*successful_detections/frame_idx:.1f}%",
    fps,
    f"{frame_idx/fps:.2f}s",
    arm_back_idx if arm_back_idx is not None else 'N/A',
    max_flexion_idx if max_flexion_idx is not None else 'N/A',
    ball_release_idx if ball_release_idx is not None else 'N/A',
    f"{delivery_duration:.3f}s" if delivery_duration is not None else 'N/A',
    f"{extension_duration:.3f}s" if extension_duration is not None else 'N/A',
    f"{elbow_angles[max_flexion_idx]:.1f}°" if max_flexion_idx is not None else 'N/A',
    f"{elbow_angles[ball_release_idx]:.1f}°" if ball_release_idx is not None else 'N/A',
    f"{extension_range:.1f}°" if extension_range is not None else 'N/A',
    f"{abs(max_extension_velocity):.1f}°/s" if max_extension_velocity is not None else 'N/A',
    preferred_hand if preferred_hand != 'auto' else detected_arm_side[0] if detected_arm_side else 'N/A',
])

summary_data['Unit'].extend([
    'frames',
    'percentage',
    'fps',
    'seconds',
    'frame index',
    'frame index',
    'frame index',
    'seconds',
    'seconds',
    'degrees',
    'degrees',
    'degrees',
    'degrees/second',
    'hand'
])

# Add ICC legality check results
if max_flexion_idx is not None and ball_release_idx is not None:
    is_legal, extension_angle, violation = check_icc_legality(elbow_angles, max_flexion_idx, ball_release_idx, icc_limit=15.0)
    summary_data['Metric'].extend(['ICC Legality (≤15°)', 'Extension Amount', 'Violation Amount'])
    summary_data['Value'].extend([
        'LEGAL' if is_legal else 'ILLEGAL',
        f"{extension_angle:.1f}°",
        f"{violation:.1f}°" if violation > 0 else '0°'
    ])
    summary_data['Unit'].extend(['status', 'degrees', 'degrees'])

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('output/bowling_metrics_summary.csv', index=False)

# ---------------- Enhanced Plotting ----------------
plt.figure(figsize=(15, 10))

# Plot 1: Elbow angle over time with phase markers and ICC limit reference
plt.subplot(3, 1, 1)
plt.plot(time_stamps, elbow_angles, 'b-', linewidth=2, label='Elbow Angle')

# Mark key phases
if arm_back_idx is not None:
    plt.axvline(x=time_stamps[arm_back_idx], color='green', linestyle='--', linewidth=1.5,
                label=f'Arm-back ({time_stamps[arm_back_idx]:.3f}s)')
if max_flexion_idx is not None:
    plt.axvline(x=time_stamps[max_flexion_idx], color='red', linestyle='--', linewidth=1.5,
                label=f'Max Flexion ({time_stamps[max_flexion_idx]:.3f}s)')
if ball_release_idx is not None:
    plt.axvline(x=time_stamps[ball_release_idx], color='purple', linestyle='--', linewidth=1.5,
                label=f'Ball Release ({time_stamps[ball_release_idx]:.3f}s)')

# ICC rule reference line (15° extension from max flexion)
if max_flexion_idx is not None:
    max_flexion_angle = elbow_angles[max_flexion_idx]
    icc_limit_angle = max_flexion_angle - 15.0
    plt.axhline(y=icc_limit_angle, color='orange', linestyle=':', linewidth=2, 
                label=f'ICC Legal Limit (15° ext = {icc_limit_angle:.1f}°)')

plt.xlabel("Time (s)")
plt.ylabel("Elbow Angle (degrees)")
plt.title("Elbow Flexion–Extension from Arm-Back to Ball Release")
plt.grid(True, alpha=0.3)
plt.legend(loc='best', fontsize=9)

# Plot 2: Angular velocity
plt.subplot(3, 1, 2)
if len(angular_velocity) > 0:
    plt.plot(time_stamps[1:], angular_velocity, 'r-', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.ylabel("Angular Velocity (°/s)")
    plt.xlabel("Time (s)")
    plt.title("Elbow Extension Velocity (Negative = Extension)")
    plt.grid(True, alpha=0.3)

# Plot 3: Wrist trajectory (to visualize arm motion)
plt.subplot(3, 1, 3)
wrist_x = [pos[0] for pos in wrist_positions]
wrist_y = [pos[1] for pos in wrist_positions]
plt.plot(wrist_x, wrist_y, 'g-', linewidth=2, alpha=0.7)
plt.scatter(wrist_x[0], wrist_y[0], c='green', s=100, label='Start', marker='o')
if len(wrist_x) > 1:
    plt.scatter(wrist_x[-1], wrist_y[-1], c='red', s=100, label='End', marker='x')
plt.xlabel("Wrist X Position (pixels)")
plt.ylabel("Wrist Y Position (pixels)")
plt.title("Wrist Trajectory During Bowling Action")
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates

plt.tight_layout()
plt.savefig('output/elbow_flexion_analysis.png', dpi=300, bbox_inches='tight')
# plt.show()  # Commented out to avoid matplotlib interruption

print(f"\n✅ Analysis complete!")
print(f"📊 Detailed data saved to:    output/elbow_flexion_analysis.csv")
print(f"📋 Metrics summary saved to:  output/bowling_metrics_summary.csv")
print(f"📈 Plot saved to:             output/elbow_flexion_analysis.png")
print(f"🎥 Annotated video saved to:  {output_path}")
print(f"\n{'='*50}")
print(f"OUTPUT DESCRIPTION")
print(f"{'='*50}")
print(f"The output video shows:")
print(f"  • Green circles: Shoulder position")
print(f"  • Blue circles: Elbow position") 
print(f"  • Red circles: Wrist position")
print(f"  • Cyan lines: Arm segments")
print(f"  • White text: Real-time elbow angle and frame number")
print(f"\nThe analysis plot shows:")
print(f"  • Top: Elbow angle dynamics with ICC legality limit (15°)")
print(f"  • Middle: Angular velocity during extension phase")
print(f"  • Bottom: Wrist trajectory in image coordinates")