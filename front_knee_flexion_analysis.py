import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ------------------------------
# Helpers
# ------------------------------
def angle_deg(a, b, c):
    """Angle at point b (a-b-c) in degrees."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-9:
        return np.nan
    cosang = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))

def dist(p, q):
    return float(np.linalg.norm(np.array(p, float) - np.array(q, float)))

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# COCO keypoint indices for YOLOv8 pose (17 pts)
# 11 L-hip, 13 L-knee, 15 L-ankle
# 12 R-hip, 14 R-knee, 16 R-ankle
#  9 L-wrist, 10 R-wrist
KP = {
    "L": {"hip": 11, "knee": 13, "ankle": 15, "wrist": 9},
    "R": {"hip": 12, "knee": 14, "ankle": 16, "wrist": 10},
}

def get_kpts(result):
    """
    Return keypoints array [17, 3] => (x, y, conf) for the best person in the frame.
    """
    if result.keypoints is None or len(result.keypoints) == 0:
        return None
    # choose the person with highest mean keypoint confidence
    k = result.keypoints.data.cpu().numpy()  # shape: [n, 17, 3]
    mean_conf = k[:, :, 2].mean(axis=1)
    idx = int(np.argmax(mean_conf))
    return k[idx]

def valid_triplet(kpts, side, conf_th=0.25):
    iH, iK, iA = KP[side]["hip"], KP[side]["knee"], KP[side]["ankle"]
    H, K, A = kpts[iH], kpts[iK], kpts[iA]
    if H[2] < conf_th or K[2] < conf_th or A[2] < conf_th:
        return False
    return True

def get_triplet(kpts, side):
    iH, iK, iA = KP[side]["hip"], KP[side]["knee"], KP[side]["ankle"]
    H = (float(kpts[iH][0]), float(kpts[iH][1]))
    K = (float(kpts[iK][0]), float(kpts[iK][1]))
    A = (float(kpts[iA][0]), float(kpts[iA][1]))
    return H, K, A

def get_wrist(kpts, side):
    iW = KP[side]["wrist"]
    return (float(kpts[iW][0]), float(kpts[iW][1])), float(kpts[iW][2])

# ------------------------------
# Anti-swap tracker (logic gate)
# ------------------------------
class KneeLockTracker:
    """
    Locks onto a chosen leg side (L or R) and prevents swapping by:
    - requiring plausible motion between consecutive frames (gated by leg length & fps)
    - preferring continuity (min displacement to previous knee point)
    """
    def __init__(self, side, fps, max_jump_leglen=0.60, conf_th=0.25):
        self.side = side
        self.fps = fps
        self.conf_th = conf_th
        self.max_jump_leglen = max_jump_leglen  # fraction of leg length per frame allowed
        self.prev_knee = None
        self.prev_leglen = None
        self.lost_count = 0

    def update(self, kpts):
        """
        Returns (ok, hip, knee, ankle, knee_angle, reason)
        """
        if kpts is None:
            self.lost_count += 1
            return False, None, None, None, np.nan, "no_person"

        # Need chosen side triplet
        if not valid_triplet(kpts, self.side, self.conf_th):
            self.lost_count += 1
            return False, None, None, None, np.nan, "low_conf"

        H, K, A = get_triplet(kpts, self.side)
        leglen = dist(H, K) + dist(K, A)

        # Gate motion (prevents sudden knee swap / jump)
        if self.prev_knee is not None and self.prev_leglen is not None:
            d = dist(K, self.prev_knee)
            # allow bigger jumps if fps is low; scale gently
            fps_scale = clamp(30.0 / max(self.fps, 1e-6), 0.7, 2.0)
            max_jump = self.max_jump_leglen * self.prev_leglen * fps_scale

            if d > max_jump:
                # implausible move => reject this frame (keep last)
                self.lost_count += 1
                return False, None, None, None, np.nan, f"gated_jump(d={d:.1f} > {max_jump:.1f})"

        ang = angle_deg(H, K, A)

        # Update state
        self.prev_knee = K
        self.prev_leglen = leglen
        self.lost_count = 0
        return True, H, K, A, ang, "ok"

# ------------------------------
# Simple phase heuristics: FFC + Release
# ------------------------------
def detect_ffc(ankle_y, fps):
    """
    Approx FFC: first local minimum in ankle_y (ankle lowest on screen => foot near ground)
    plus low velocity around it.
    Returns frame index or None.
    """
    if len(ankle_y) < int(fps):
        return None
    y = np.array(ankle_y, float)
    # smooth a bit
    k = 7 if len(y) > 20 else 3
    y_s = pd.Series(y).rolling(k, center=True, min_periods=1).mean().values
    v = np.gradient(y_s)

    # local minima candidates where velocity crosses from negative to positive
    cand = []
    for i in range(2, len(y_s) - 2):
        if y_s[i] < y_s[i-1] and y_s[i] < y_s[i+1] and abs(v[i]) < np.percentile(abs(v), 35):
            cand.append(i)
    return cand[0] if cand else None

def detect_release_from_wrist_speed(wrist_xy, fps, start_idx):
    """
    Approx release: peak wrist speed after FFC within a reasonable window.
    """
    if start_idx is None:
        return None
    pts = np.array(wrist_xy, float)
    if len(pts) < start_idx + int(0.3 * fps):
        return None

    # speed (px/frame)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    d = np.r_[d[0], d]  # align length
    # smooth
    d_s = pd.Series(d).rolling(5, center=True, min_periods=1).mean().values

    # search window: 0.1s to 1.2s after FFC
    a = start_idx + int(0.10 * fps)
    b = min(len(d_s) - 1, start_idx + int(1.20 * fps))
    if b <= a:
        return None

    rel = int(a + np.argmax(d_s[a:b+1]))
    return rel

# ------------------------------
# Main
# ------------------------------
def main(
    video_path="1.mov",
    out_video="knee_ffc_to_release.mp4",
    out_csv="knee_angles.csv",
    model_name="yolov8x-pose.pt"
):
    print("Loading model...")
    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path} | FPS: {fps:.2f} | {W}x{Hh}")

    print("\nSelect knee to track (locked, no swapping):")
    print("1) Left knee")
    print("2) Right knee")
    choice = input("Enter 1 or 2: ").strip()
    side = "L" if choice == "1" else "R"

    tracker = KneeLockTracker(side=side, fps=fps, max_jump_leglen=0.60, conf_th=0.25)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_video, fourcc, fps, (W, Hh))

    rows = []
    ankle_y = []
    wrist_xy = []

    frame_idx = 0
    last_good = None  # store last good (H,K,A,ang)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model(frame, verbose=False)[0]
        kpts = get_kpts(res)

        ok_leg, Hip, Knee, Ankle, ang, reason = tracker.update(kpts)

        if ok_leg:
            last_good = (Hip, Knee, Ankle, ang)
            ankle_y.append(Ankle[1])
            w, wc = get_wrist(kpts, side)
            wrist_xy.append(w)
        else:
            # if lost, carry forward last good points for arrays (keeps indices aligned)
            if last_good is not None:
                Hip, Knee, Ankle, ang = last_good
                ankle_y.append(Ankle[1])
                wrist_xy.append(wrist_xy[-1] if len(wrist_xy) else (0.0, 0.0))
            else:
                ankle_y.append(np.nan)
                wrist_xy.append((0.0, 0.0))
                ang = np.nan

        rows.append({"frame": frame_idx, "time_s": frame_idx / fps, "knee_angle_deg": ang, "status": reason})

        # Draw
        if last_good is not None:
            Hip, Knee, Ankle, ang2 = last_good
            cv2.circle(frame, (int(Knee[0]), int(Knee[1])), 8, (0, 255, 0), -1)
            cv2.line(frame, (int(Hip[0]), int(Hip[1])), (int(Knee[0]), int(Knee[1])), (255, 255, 255), 2)
            cv2.line(frame, (int(Knee[0]), int(Knee[1])), (int(Ankle[0]), int(Ankle[1])), (255, 255, 255), 2)
            cv2.putText(frame, f"{'Left' if side=='L' else 'Right'} knee angle: {ang2:.1f}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.putText(frame, f"Frame {frame_idx} | {reason}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        vw.write(frame)
        frame_idx += 1

    cap.release()
    vw.release()

    df = pd.DataFrame(rows)

    # Detect phases
    # fill nans for detection
    ay = pd.Series(ankle_y).interpolate(limit_direction="both").values
    ffc = detect_ffc(ay, fps)
    rel = detect_release_from_wrist_speed(wrist_xy, fps, ffc)

    df["is_ffc"] = False
    df["is_release"] = False
    if ffc is not None and 0 <= ffc < len(df):
        df.loc[ffc, "is_ffc"] = True
    if rel is not None and 0 <= rel < len(df):
        df.loc[rel, "is_release"] = True

    df.to_csv(out_csv, index=False)

    # Plot angles + markers
    plt.figure(figsize=(10, 4))
    plt.plot(df["time_s"], df["knee_angle_deg"])
    if ffc is not None:
        plt.axvline(df.loc[ffc, "time_s"], linestyle="--", label="FFC")
    if rel is not None:
        plt.axvline(df.loc[rel, "time_s"], linestyle=":", label="Release (heuristic)")
    plt.xlabel("Time (s)")
    plt.ylabel("Knee angle (deg)")
    plt.title("Knee flexion (locked leg) | FFC → Release")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/knee_flexion_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Export a cropped angle table (FFC->Release) if possible
    if ffc is not None and rel is not None and rel > ffc:
        seg = df.loc[ffc:rel, ["frame", "time_s", "knee_angle_deg"]].copy()
        seg.to_csv("output/knee_angles_ffc_to_release.csv", index=False)
        print("Saved segment: output/knee_angles_ffc_to_release.csv")

    print(f"\n✅ Analysis complete!")
    print(f"\nSaved outputs to 'output/' directory:")
    print(f"- Video:  {out_video}")
    print(f"- CSV:    {out_csv}")
    print(f"- Graph:  output/knee_flexion_analysis.png")
    if ffc is not None:
        print(f"\n📍 Detected FFC at frame {ffc} ({df.loc[ffc,'time_s']:.2f}s)")
    else:
        print("\n⚠️ FFC not detected (try higher-res video or adjust thresholds).")
    if rel is not None:
        print(f"📍 Detected Release at frame {rel} ({df.loc[rel,'time_s']:.2f}s)")
    else:
        print("⚠️ Release not detected (heuristic).")
    
    # Exit after all outputs are complete
    exit(0)

if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    main(
        video_path="1.mov",
        out_video="output/knee_ffc_to_release.mp4",
        out_csv="output/knee_angles.csv",
        model_name="yolov8x-pose.pt"
    )