"""
Cricket Bowling Annotation Tool  (v6 — fixed)
==============================================
Creates ground_truth.csv matching master.xlsx structure for biomechanics evaluation.

FIX 8 (this version)
─────────────────────
  cap.set(CAP_PROP_POS_FRAMES) is no longer used for navigation.
  On MOV/H.264 files it snaps to the nearest keyframe but cap.get()
  still reports the *requested* position — so frame 606 in the annotator
  was visually showing frame 604 (or similar offset) from the pipeline.

  _read_frame() now seeks by re-opening the video and advancing with
  cap.grab() (no decode) from position 0.  This is the same approach used
  in the fixed bowling pipeline (_open_and_advance), so all three tools
  — pipeline, annotator, evaluation — now refer to the same physical frame
  when they say "frame N".

  Performance note: grab() is fast (no decode).  For a typical delivery
  clip the seek target is rarely beyond frame 700, so the grab loop
  completes in well under a second.

FIXES vs v5 (carried forward)
──────────────────────────────
  FIX 7 — Frame numbering is 1-based, matching the model pipeline.
  FIX 5 — knee angles use FRONT knee only (matching KneeLockTracker).
  FIX 6 — head_bfc reference is FRONT ankle at BFC (matching model).
  FIX 1-4 — carried forward from v2/v3.

NAVIGATION
──────────
  ←  / →          : ±1 frame
  A  / D           : ±10 frames

EVENTS  (navigate to the target frame first, then press the key)
───────
  1               : mark BFC  (Back Foot Contact)
  2               : mark FFC  (Front Foot Contact)
  3               : mark Arm Back
  4               : mark Release
  F               : log a foot-contact frame (repeat for each of last 5 steps)

OTHER
─────
  C               : calibrate scale (click stump top → stump bottom)
  S               : save & finish current trial
  Q               : quit without saving

MOUSE
─────
  Left-click      : record the next expected keypoint for the active event phase.
                    The HUD always shows which point to click next.

USAGE
─────
  python cricket_annotator.py                  # interactive video picker
  python cricket_annotator.py path/to/vid.mp4  # open specific video directly
"""

import cv2
import numpy as np
import pandas as pd
import os
import sys
import math
import glob
from pathlib import Path

# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────
STUMP_HEIGHT_M  = 0.711
OUTPUT_CSV      = "ground_truth.csv"
VIDEOS_DIR      = "./../videos"

FACE_KP_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]

# UI colours (BGR)
COL_EVENT  = (0,   255, 255)
COL_KP     = (0,   165, 255)
COL_FOOT   = (255, 0,   255)
COL_CALIB  = (0,   255, 0  )
COL_TEXT   = (255, 255, 255)
COL_WARN   = (0,   0,   255)
COL_OK     = (0,   220, 80 )

# UI text sizes
TEXT_SCALE       = 0.72
TEXT_THICK       = 1
LEGEND_SCALE     = 0.62
KP_LABEL_SCALE   = 0.52
NEXT_SCALE       = 0.75
KP_RADIUS        = 8

# Layout
ROW_H   = 34
PAD     = 5


# ─────────────────────────────────────────────────────────────
#  FIX 8: grab()-based frame-accurate seek helper
# ─────────────────────────────────────────────────────────────
def _grab_seek(cap: cv2.VideoCapture, video_path: str,
               target_0based: int, total_frames: int) -> bool:
    """
    Seek to target_0based (0-indexed) by re-opening the video and advancing
    with cap.grab() from position 0.

    Why not cap.set(CAP_PROP_POS_FRAMES)?
      On MOV/H.264 it snaps to the nearest keyframe but cap.get() still
      reports the *requested* position, so the offset is silent and
      undetectable.  grab() decodes no pixels and is fast enough for
      interactive use (<1 s for a 700-frame seek).

    Modifies cap in-place by releasing and re-opening it so the internal
    position counter is always accurate.
    Returns True on success.
    """
    target_0based = max(0, min(target_0based, total_frames - 1))

    # Re-open to reset the internal state cleanly
    cap.release()
    cap.open(video_path)
    if not cap.isOpened():
        cap.open(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return False

    # Advance by grabbing (no decode) the required number of frames
    for _ in range(target_0based):
        if not cap.grab():
            break

    return True


# ─────────────────────────────────────────────────────────────
#  Text / drawing helpers
# ─────────────────────────────────────────────────────────────
def put_text(img, text, pos, scale=TEXT_SCALE, color=COL_TEXT, thickness=TEXT_THICK):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(img,
                  (x - PAD,      y - th - PAD),
                  (x + tw + PAD, y + bl + PAD),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def put_text_right(img, text, right_x, y, scale=TEXT_SCALE,
                   color=COL_TEXT, thickness=TEXT_THICK):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    x = right_x - tw
    cv2.rectangle(img,
                  (x - PAD,      y - th - PAD),
                  (x + tw + PAD, y + bl + PAD),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_panel(img, lines, x0=8, y0=14, row_h=ROW_H):
    y = y0
    for text, color in lines:
        put_text(img, text, (x0, y), color=color)
        y += row_h
    return y


# ─────────────────────────────────────────────────────────────
#  "What to do next" instruction box — top RIGHT
# ─────────────────────────────────────────────────────────────
def draw_next_instruction(img, session, calib_mode: bool):
    h, w = img.shape[:2]
    s = session
    ev = s.events

    if calib_mode:
        msg   = "CALIB: click STUMP TOP then STUMP BOTTOM"
        color = COL_CALIB
    elif s.px_per_m is None:
        msg   = "NEXT: press C to calibrate scale (stump clicks)"
        color = COL_WARN
    elif s.current_phase is not None and not s.phase_complete():
        nxt   = s.next_kp_label()
        done  = s.current_phase_idx
        total = len(ANNOTATION_PHASES[s.current_phase]["keys"])
        msg   = f"CLICK [{s.current_phase.upper()}] {nxt}  ({done}/{total} done)"
        color = COL_KP
    elif ev["bfc_frame"] is None:
        msg   = "NEXT: navigate to BFC frame → press 1"
        color = COL_TEXT
    elif ev["ffc_frame"] is None:
        msg   = "NEXT: navigate to FFC frame → press 2"
        color = COL_TEXT
    elif ev["arm_back_frame"] is None:
        msg   = "NEXT: navigate to Arm-Back frame → press 3"
        color = COL_TEXT
    elif ev["release_frame"] is None:
        msg   = "NEXT: navigate to Release frame → press 4"
        color = COL_TEXT
    elif not s.keypoints.get("release_plus1"):
        msg   = "NEXT: press → (1 frame forward) then click WRIST"
        color = COL_KP
    elif len(s.foot_contacts) < 5:
        fc    = len(s.foot_contacts)
        msg   = f"NEXT: mark foot contacts with F  ({fc}/5 done)"
        color = COL_WARN if fc == 0 else COL_TEXT
    else:
        msg   = "ALL DONE — press S to save"
        color = COL_OK

    right_x = w - 16
    put_text_right(img, msg, right_x, 28, scale=NEXT_SCALE, color=color)


# ─────────────────────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────────────────────
def angle_3pts(a, b, c):
    if None in (a, b, c):
        return None
    v1 = np.array(a, dtype=float) - np.array(b, dtype=float)
    v2 = np.array(c, dtype=float) - np.array(b, dtype=float)
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

def dist_px(p1, p2):
    if None in (p1, p2):
        return None
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def px_to_m(px_dist, px_per_m):
    if px_dist is None or not px_per_m:
        return None
    return px_dist / px_per_m

def signed_px_to_cm(dx, dy, px_per_m):
    if not px_per_m:
        return None, None, None
    dx_cm = dx / px_per_m * 100
    dy_cm = dy / px_per_m * 100
    d_cm  = math.hypot(dx_cm, dy_cm)
    return dx_cm, dy_cm, d_cm

def head_centroid(kp_dict: dict):
    xs, ys = [], []
    for name in FACE_KP_NAMES:
        pt = kp_dict.get(name)
        if pt is not None:
            xs.append(pt[0])
            ys.append(pt[1])
    if not xs:
        return None
    return (float(np.mean(xs)), float(np.mean(ys)))


# ─────────────────────────────────────────────────────────────
#  Annotation phase definitions
# ─────────────────────────────────────────────────────────────
ANNOTATION_PHASES = {
    "bfc": {
        "label": "BFC — L-hip, R-hip, L-knee, R-knee, L-ankle, R-ankle, nose",
        "keys":  ["left_hip", "right_hip", "left_knee", "right_knee",
                  "left_ankle", "right_ankle", "nose"]
    },
    "ffc": {
        "label": "FFC — L-hip, R-hip, L-knee, R-knee, L-ankle, R-ankle, nose",
        "keys":  ["left_hip", "right_hip", "left_knee", "right_knee",
                  "left_ankle", "right_ankle", "nose"]
    },
    "arm_back": {
        "label": "ARM BACK — shoulder, elbow, wrist",
        "keys":  ["shoulder", "elbow", "wrist"]
    },
    "release": {
        "label": "RELEASE — shoulder, elbow, wrist, L-hip, R-hip, L-knee, R-knee, L-ankle, R-ankle",
        "keys":  ["shoulder", "elbow", "wrist",
                  "left_hip", "right_hip",
                  "left_knee", "right_knee",
                  "left_ankle", "right_ankle"]
    },
    "release_plus1": {
        "label": "RELEASE+1 — wrist only",
        "keys":  ["wrist"]
    }
}

PHASE_ORDER = ["bfc", "ffc", "arm_back", "release", "release_plus1"]


# ─────────────────────────────────────────────────────────────
#  AnnotationSession
# ─────────────────────────────────────────────────────────────
class AnnotationSession:
    def __init__(self, trial_id: str, fps: float):
        self.trial_id   = trial_id
        self.fps        = fps
        self.events     = {k: None for k in
                           ["bfc_frame", "ffc_frame", "arm_back_frame", "release_frame"]}
        self.foot_contacts: list[int] = []
        self.keypoints: dict[str, dict] = {p: {} for p in PHASE_ORDER}
        self.calib_pts: list[tuple] = []
        self.px_per_m:  float | None = None
        self.current_phase:     str | None = None
        self.current_phase_idx: int        = 0

    def set_event(self, key: str, frame: int):
        self.events[key] = frame

    def add_foot_contact(self, frame: int):
        if frame not in self.foot_contacts:
            self.foot_contacts.append(frame)
            self.foot_contacts.sort()

    def set_calib(self, pts: list[tuple]):
        self.calib_pts = pts
        d = abs(pts[1][1] - pts[0][1])
        if d > 0:
            self.px_per_m = d / STUMP_HEIGHT_M

    def next_kp_label(self) -> str | None:
        if self.current_phase is None:
            return None
        keys = ANNOTATION_PHASES[self.current_phase]["keys"]
        if self.current_phase_idx < len(keys):
            return keys[self.current_phase_idx]
        return None

    def record_click(self, x: int, y: int):
        if self.current_phase is None:
            return
        keys = ANNOTATION_PHASES[self.current_phase]["keys"]
        if self.current_phase_idx < len(keys):
            kp = keys[self.current_phase_idx]
            self.keypoints[self.current_phase][kp] = (x, y)
            self.current_phase_idx += 1
            print(f"  [{self.current_phase.upper()}] '{kp}' → ({x}, {y})")

    def start_phase(self, phase: str):
        self.current_phase     = phase
        self.current_phase_idx = 0

    def phase_complete(self) -> bool:
        if self.current_phase is None:
            return True
        return self.current_phase_idx >= len(
            ANNOTATION_PHASES[self.current_phase]["keys"])

    # ── metric computation ───────────────────────────────────
    def compute(self, bowling_arm: str = "R", view_mode: str = "SIDE") -> dict:
        e   = self.events
        kp  = self.keypoints
        fps = self.fps
        ppm = self.px_per_m

        out: dict = {}

        # ── Identity & metadata ──────────────────────────────
        out["trial_id"]       = self.trial_id
        out["fps"]            = fps
        out["bowling_arm"]    = bowling_arm
        out["view_mode"]      = view_mode
        out["release_frame"]  = e["release_frame"]
        out["release_method"] = "manual"

        # ── Last-5 step cadence ──────────────────────────────
        rel        = e["release_frame"]
        fc_sorted  = sorted(self.foot_contacts)
        before_rel = [f for f in fc_sorted if rel is None or f <= rel]
        last5      = before_rel[-5:] if len(before_rel) >= 5 else before_rel

        out["last5_steps_frame"]      = str(last5)
        intervals = [(last5[i] - last5[i-1]) / fps for i in range(1, len(last5))]
        out["last5_step_intervals_s"] = str(intervals)

        if intervals:
            mean = float(np.mean(intervals))
            std  = float(np.std(intervals, ddof=0))
            out["step_duration_mean_s"]    = mean
            out["step_duration_std_s"]     = std
            out["step_duration_cv"]        = (std / mean) if mean else None
            out["final5_total_duration_s"] = float(sum(intervals))
        else:
            out["step_duration_mean_s"]    = None
            out["step_duration_std_s"]     = None
            out["step_duration_cv"]        = None
            out["final5_total_duration_s"] = None

        # ── Stride duration (BFC → FFC) ──────────────────────
        if e["ffc_frame"] is not None and e["bfc_frame"] is not None:
            out["stride_duration_s"] = (e["ffc_frame"] - e["bfc_frame"]) / fps
        else:
            out["stride_duration_s"] = None

        # ── Stride length ─────────────────────────────────────
        if bowling_arm == "R":
            f_ankle = kp["ffc"].get("left_ankle")
            b_ankle = kp["bfc"].get("right_ankle")
        else:
            f_ankle = kp["ffc"].get("right_ankle")
            b_ankle = kp["bfc"].get("left_ankle")
        if f_ankle is None:
            f_ankle = kp["ffc"].get("right_ankle") or kp["ffc"].get("left_ankle")
        if b_ankle is None:
            b_ankle = kp["bfc"].get("left_ankle") or kp["bfc"].get("right_ankle")
        out["stride_length_m"] = px_to_m(dist_px(b_ankle, f_ankle), ppm)

        # ── Event frame numbers ──────────────────────────────
        out["bfc_frame"]       = e["bfc_frame"]
        out["ffc_frame"]       = e["ffc_frame"]
        out["arm_back_frame"]  = e["arm_back_frame"]
        out["release_frame.1"] = e["release_frame"]

        # ── Elbow angles ─────────────────────────────────────
        sh_ab = kp["arm_back"].get("shoulder")
        el_ab = kp["arm_back"].get("elbow")
        wr_ab = kp["arm_back"].get("wrist")
        out["elbow_angle_arm_back_deg"] = angle_3pts(sh_ab, el_ab, wr_ab)

        sh_r = kp["release"].get("shoulder")
        el_r = kp["release"].get("elbow")
        wr_r = kp["release"].get("wrist")
        out["elbow_angle_release_deg"] = angle_3pts(sh_r, el_r, wr_r)

        a_ab  = out["elbow_angle_arm_back_deg"]
        a_rel = out["elbow_angle_release_deg"]
        out["elbow_extension_deg"] = (
            (a_rel - a_ab) if (a_ab is not None and a_rel is not None) else None
        )

        # ── Knee angles — FRONT knee only ────────────────────
        if bowling_arm == "R":
            out["knee_angle_ffc_deg"] = angle_3pts(
                kp["ffc"].get("left_hip"),
                kp["ffc"].get("left_knee"),
                kp["ffc"].get("left_ankle"))
            out["knee_angle_release_deg"] = angle_3pts(
                kp["release"].get("left_hip"),
                kp["release"].get("left_knee"),
                kp["release"].get("left_ankle"))
        else:
            out["knee_angle_ffc_deg"] = angle_3pts(
                kp["ffc"].get("right_hip"),
                kp["ffc"].get("right_knee"),
                kp["ffc"].get("right_ankle"))
            out["knee_angle_release_deg"] = angle_3pts(
                kp["release"].get("right_hip"),
                kp["release"].get("right_knee"),
                kp["release"].get("right_ankle"))

        # ── Head @ FFC relative to front ankle ───────────────
        head_ffc = head_centroid(kp["ffc"])
        if bowling_arm == "R":
            f_ankle_ffc = kp["ffc"].get("left_ankle")
        else:
            f_ankle_ffc = kp["ffc"].get("right_ankle")
        if f_ankle_ffc is None:
            f_ankle_ffc = kp["ffc"].get("left_ankle") or kp["ffc"].get("right_ankle")

        if head_ffc and f_ankle_ffc and ppm:
            dx = head_ffc[0] - f_ankle_ffc[0]
            dy = head_ffc[1] - f_ankle_ffc[1]
            (out["head_dx_ffc_cm"],
             out["head_dy_ffc_cm"],
             out["head_d_ffc_cm"]) = signed_px_to_cm(dx, dy, ppm)
        else:
            out["head_dx_ffc_cm"] = out["head_dy_ffc_cm"] = out["head_d_ffc_cm"] = None

        # ── Head @ BFC relative to FRONT ankle at BFC ────────
        head_bfc = head_centroid(kp["bfc"])
        if bowling_arm == "R":
            f_ankle_bfc = kp["bfc"].get("left_ankle")
        else:
            f_ankle_bfc = kp["bfc"].get("right_ankle")
        if f_ankle_bfc is None:
            f_ankle_bfc = kp["bfc"].get("left_ankle") or kp["bfc"].get("right_ankle")

        if head_bfc and f_ankle_bfc and ppm:
            dx = head_bfc[0] - f_ankle_bfc[0]
            dy = head_bfc[1] - f_ankle_bfc[1]
            (out["head_dx_bfc_cm"],
             out["head_dy_bfc_cm"],
             out["head_d_bfc_cm"]) = signed_px_to_cm(dx, dy, ppm)
        else:
            out["head_dx_bfc_cm"] = out["head_dy_bfc_cm"] = out["head_d_bfc_cm"] = None

        # ── Release speed ─────────────────────────────────────
        wr_rel  = kp["release"].get("wrist")
        wr_rel1 = kp["release_plus1"].get("wrist")
        if wr_rel and wr_rel1 and ppm and fps:
            d_m = dist_px(wr_rel, wr_rel1) / ppm
            release_spd = d_m * fps
            out["peak_wrist_speed_m_s"]       = release_spd
            out["wrist_speed_at_release_m_s"] = release_spd
        else:
            out["peak_wrist_speed_m_s"]       = None
            out["wrist_speed_at_release_m_s"] = None

        # ── Individual step frames ────────────────────────────
        for i, frm in enumerate(last5):
            out[f"step{i+1}_frame"] = frm
        for i in range(len(last5), 5):
            out[f"step{i+1}_frame"] = None

        return out


def _avg_angles(a, b):
    if a is not None and b is not None:
        return (a + b) / 2
    return a if a is not None else b


# ─────────────────────────────────────────────────────────────
#  Annotator — OpenCV UI
# ─────────────────────────────────────────────────────────────
class Annotator:

    def __init__(self, video_path: str, trial_id: str, existing_rows: pd.DataFrame):
        self.video_path = video_path          # FIX 8: needed for re-open in grab seek
        self.cap        = cv2.VideoCapture(video_path)
        self.trial_id   = trial_id
        self.total      = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps        = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.session    = AnnotationSession(trial_id, self.fps)
        self.existing   = existing_rows
        self.calib_mode = False
        self.calib_pts: list = []
        self.result      = None
        # _seek_idx: 0-based (internal position counter)
        # frame_idx: 1-based (displayed, stored in CSV, matches pipeline)
        self._seek_idx   = 0
        self.frame_idx   = 1
        self._read_frame(0)

    def _read_frame(self, seek_pos: int) -> bool:
        """
        FIX 8: Navigate to seek_pos (0-based) using grab()-based seeking.
        Never uses cap.set() to avoid the MOV keyframe-snap offset.
        Updates frame_idx to the 1-based equivalent.
        """
        seek_pos = max(0, min(seek_pos, self.total - 1))
        self._seek_idx = seek_pos
        self.frame_idx = seek_pos + 1   # 1-based, matching pipeline

        # Use grab()-based seek — accurate on MOV/H.264 (FIX 8)
        _grab_seek(self.cap, self.video_path, seek_pos, self.total)

        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
        return ret

    def _overlay(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        s = self.session

        # ── Progress bar — BOTTOM edge ───────────────────────
        bar_y   = h - 10
        prog_x  = int(self._seek_idx / max(self.total - 1, 1) * w)
        cv2.rectangle(frame, (0, bar_y), (w, h), (40, 40, 40), cv2.FILLED)
        cv2.rectangle(frame, (0, bar_y), (prog_x, h), (0, 200, 100), cv2.FILLED)
        for ev_val in s.events.values():
            if ev_val is not None:
                tx = int((ev_val - 1) / max(self.total - 1, 1) * w)
                cv2.line(frame, (tx, bar_y - 4), (tx, h), COL_EVENT, 2)

        # ── LEFT panel — status info ──────────────────────────
        calib_str = (f"OK {s.px_per_m:.1f} px/m" if s.px_per_m
                     else "NOT SET — press C")
        calib_col  = COL_OK if s.px_per_m else COL_WARN

        ev = s.events
        event_str = (
            f"BFC:{_fmt(ev['bfc_frame'])}  "
            f"FFC:{_fmt(ev['ffc_frame'])}  "
            f"ARM:{_fmt(ev['arm_back_frame'])}  "
            f"REL:{_fmt(ev['release_frame'])}"
        )
        fc_count = len(s.foot_contacts)
        fc_col   = COL_OK if fc_count >= 5 else (COL_WARN if fc_count == 0 else COL_TEXT)

        left_lines: list[tuple[str, tuple]] = [
            (f"{self.trial_id}  |  Frame {self.frame_idx}/{self.total}"
             f"  |  {self.fps:.1f} fps",  COL_TEXT),
            (f"Scale: {calib_str}",        calib_col),
            (f"Events: {event_str}",       COL_TEXT),
            (f"Foot contacts: {sorted(s.foot_contacts)}  ({fc_count}/5+)", fc_col),
        ]

        if self.calib_mode:
            left_lines.append(("CALIB MODE — click TOP then BOTTOM of stump", COL_CALIB))

        if s.current_phase:
            nxt   = s.next_kp_label()
            done  = s.current_phase_idx
            tot   = len(ANNOTATION_PHASES[s.current_phase]["keys"])
            label = ANNOTATION_PHASES[s.current_phase]["label"]
            left_lines.append((f"[{s.current_phase.upper()}]  {label}", COL_KP))
            left_lines.append((f"  → Click: {nxt}   ({done}/{tot})", COL_KP))

        draw_panel(frame, left_lines, x0=10, y0=28, row_h=ROW_H)

        # ── RIGHT panel — "what to do next" ──────────────────
        draw_next_instruction(frame, s, self.calib_mode)

        # ── Bottom legend ─────────────────────────────────────
        legend = [
            "NAV: ← → = ±1 frame   A/D = ±10 frames   C = Calibrate   S = Save   Q = Quit",
            "MARK: 1=BFC   2=FFC   3=Arm Back   4=Release   F=Foot Contact",
        ]
        for i, ln in enumerate(reversed(legend)):
            put_text(frame, ln, (10, h - 22 - i * (int(LEGEND_SCALE * 30) + 10)),
                     scale=LEGEND_SCALE, color=(200, 200, 200))

        # ── Keypoints ─────────────────────────────────────────
        active = self._phases_for_current_frame()
        for phase in active:
            for kname, pt in s.keypoints.get(phase, {}).items():
                if pt:
                    cv2.circle(frame, pt, KP_RADIUS + 2, (0, 0, 0), -1)
                    cv2.circle(frame, pt, KP_RADIUS, COL_KP, -1)
                    put_text(frame, kname[:8], (pt[0] + 10, pt[1] - 5),
                             scale=KP_LABEL_SCALE, color=COL_KP, thickness=1)

        # ── Foot-contact flash ────────────────────────────────
        if self.frame_idx in s.foot_contacts:
            cv2.putText(frame, "FC", (w // 2 - 40, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, COL_FOOT, 4, cv2.LINE_AA)

        # ── Calibration preview dots ──────────────────────────
        for cp in self.calib_pts:
            cv2.circle(frame, cp, 10, COL_CALIB, -1)
            cv2.circle(frame, cp, 12, (0, 0, 0), 1)

        return frame

    def _phases_for_current_frame(self) -> list[str]:
        s     = self.session
        fi    = self.frame_idx
        ev    = s.events
        shown = []
        phase_frame_map = {
            "bfc":           ev.get("bfc_frame"),
            "ffc":           ev.get("ffc_frame"),
            "arm_back":      ev.get("arm_back_frame"),
            "release":       ev.get("release_frame"),
            "release_plus1": (ev.get("release_frame") + 1
                              if ev.get("release_frame") is not None else None),
        }
        for phase, event_frame in phase_frame_map.items():
            if event_frame is not None and fi == event_frame:
                shown.append(phase)
        if s.current_phase and s.current_phase not in shown:
            shown.append(s.current_phase)
        return shown

    def _mouse_cb(self, event, x: int, y: int, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        s = self.session
        if self.calib_mode:
            self.calib_pts.append((x, y))
            if len(self.calib_pts) == 2:
                s.set_calib(self.calib_pts)
                self.calib_mode = False
                self.calib_pts  = []
                print(f"  Calibration set: {s.px_per_m:.2f} px/m")
            return
        s.record_click(x, y)

    def run(self) -> dict | None:
        wn = f"Cricket Annotator — {self.trial_id}"
        cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(wn, 1400, 800)
        cv2.setMouseCallback(wn, self._mouse_cb)

        _print_trial_header(self.trial_id)
        bowling_arm = input("  Bowling arm [R/L] (default R): ").strip().upper() or "R"
        view_mode   = input("  View mode [SIDE/FRONT] (default SIDE): ").strip().upper() or "SIDE"
        _print_workflow()

        while True:
            frame = self.current_frame.copy()
            frame = self._overlay(frame)
            cv2.imshow(wn, frame)
            key   = cv2.waitKeyEx(30)

            if key == -1:
                continue

            if key in (ord('q'), ord('Q')):
                print("  Quit — no save.")
                break

            elif key in (ord('s'), ord('S')):
                rec         = self.session.compute(bowling_arm, view_mode)
                self.result = rec
                print(f"  Trial '{self.trial_id}' saved.")
                break

            elif key in (ord('c'), ord('C')):
                self.calib_mode = True
                self.calib_pts  = []
                print("  Calibration mode ON — click stump TOP then BOTTOM.")

            elif key in (ord('f'), ord('F')):
                self.session.add_foot_contact(self.frame_idx)
                total = len(self.session.foot_contacts)
                print(f"  Foot contact @ frame {self.frame_idx}  (total: {total})")

            elif key == ord('1'):
                self.session.set_event("bfc_frame", self.frame_idx)
                self.session.start_phase("bfc")
                print(f"  BFC @ frame {self.frame_idx}")
                print("  Click: L-hip, R-hip, L-knee, R-knee, L-ankle, R-ankle, nose")

            elif key == ord('2'):
                self.session.set_event("ffc_frame", self.frame_idx)
                self.session.start_phase("ffc")
                print(f"  FFC @ frame {self.frame_idx}")
                print("  Click: L-hip, R-hip, L-knee, R-knee, L-ankle, R-ankle, nose")

            elif key == ord('3'):
                self.session.set_event("arm_back_frame", self.frame_idx)
                self.session.start_phase("arm_back")
                print(f"  Arm Back @ frame {self.frame_idx}")
                print("  Click: shoulder, elbow, wrist")

            elif key == ord('4'):
                self.session.set_event("release_frame", self.frame_idx)
                self.session.start_phase("release")
                print(f"  Release @ frame {self.frame_idx}")
                print("  Click: shoulder, elbow, wrist, "
                      "L-hip, R-hip, L-knee, R-knee, L-ankle, R-ankle")

            else:
                delta = self._resolve_nav_key(key)
                if delta is not None:
                    self._read_frame(self._seek_idx + delta)

            # Auto-transition release → release_plus1
            s = self.session
            if (s.current_phase == "release"
                    and s.phase_complete()
                    and not s.keypoints.get("release_plus1")):
                print("  Release keypoints done.")
                print("  → Press → to go 1 frame forward, then click the wrist.")
                s.start_phase("release_plus1")

        cv2.destroyWindow(wn)
        return self.result

    @staticmethod
    def _resolve_nav_key(key: int) -> int | None:
        NAV = {
            2424832: -1,   # Win ←
            2555904: +1,   # Win →
            65361:   -1,   # X11 ←
            65363:   +1,   # X11 →
            81:      -1,   # fallback
            83:      +1,
            ord('a'): -10,
            ord('A'): -10,
            ord('d'): +10,
            ord('D'): +10,
            2359296:  -10,
            2490368:  +10,
            0x1FF51:  -10,
            0x1FF53:  +10,
        }
        delta = NAV.get(key)
        if delta is None:
            delta = NAV.get(key & 0xFFFF)
        if delta is None:
            delta = NAV.get(key & 0xFF)
        return delta


# ─────────────────────────────────────────────────────────────
#  Video picker
# ─────────────────────────────────────────────────────────────
def pick_video(videos: list[str], gt_df: pd.DataFrame) -> str | None:
    print("\n" + "=" * 62)
    print("  AVAILABLE VIDEOS")
    print("=" * 62)
    annotated = set(gt_df["trial_id"].values) if not gt_df.empty else set()
    for i, vp in enumerate(videos):
        stem   = Path(vp).stem
        status = "✓ done " if stem in annotated else "       "
        print(f"  {i+1:>3}.  {status}  {Path(vp).name}")
    print("=" * 62)
    while True:
        raw = input("  Enter number to annotate (or 'q' to quit): ").strip()
        if raw.lower() == 'q':
            return None
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(videos):
                return videos[idx]
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(videos)}.")


# ─────────────────────────────────────────────────────────────
#  CSV I/O
# ─────────────────────────────────────────────────────────────
MASTER_COLS = [
    "trial_id", "fps", "bowling_arm", "view_mode",
    "release_frame", "release_method",
    "last5_steps_frame", "last5_step_intervals_s",
    "step_duration_mean_s", "step_duration_std_s", "step_duration_cv",
    "final5_total_duration_s", "stride_duration_s", "stride_length_m",
    "bfc_frame", "ffc_frame", "arm_back_frame", "release_frame.1",
    "elbow_angle_arm_back_deg", "elbow_angle_release_deg", "elbow_extension_deg",
    "knee_angle_ffc_deg", "knee_angle_release_deg",
    "head_dx_ffc_cm", "head_dy_ffc_cm", "head_d_ffc_cm",
    "head_dx_bfc_cm", "head_dy_bfc_cm", "head_d_bfc_cm",
    "peak_wrist_speed_m_s", "wrist_speed_at_release_m_s",
    "step1_frame", "step2_frame", "step3_frame", "step4_frame", "step5_frame",
]

def load_gt() -> pd.DataFrame:
    if os.path.exists(OUTPUT_CSV):
        return pd.read_csv(OUTPUT_CSV)
    return pd.DataFrame(columns=MASTER_COLS)

def save_gt(df: pd.DataFrame):
    for c in MASTER_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[MASTER_COLS]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved → {OUTPUT_CSV}  ({len(df)} rows)")


# ─────────────────────────────────────────────────────────────
#  Print helpers
# ─────────────────────────────────────────────────────────────
def _fmt(val) -> str:
    return str(val) if val is not None else "—"

def _print_trial_header(trial_id: str):
    print(f"\n{'='*62}")
    print(f"  Trial: {trial_id}")
    print(f"{'='*62}")

def _print_workflow():
    print()
    print("  WORKFLOW  (follow in order)")
    print("  ─────────────────────────────────────────────────")
    print("  1. C           Calibrate — click stump TOP then BOTTOM")
    print("  2. F  (×5+)    Mark each foot contact before release")
    print("  3. 1           Navigate to BFC → press 1 → click joints")
    print("  4. 2           Navigate to FFC → press 2 → click joints")
    print("  5. 3           Navigate to Arm-Back → press 3 → click joints")
    print("  6. 4           Navigate to Release → press 4 → click joints")
    print("  7. → then click  Go 1 frame forward → click wrist (speed)")
    print("  8. S           Save")
    print("  ─────────────────────────────────────────────────")
    print("  NAV: ← → = ±1 frame   A/D = ±10 frames")
    print()
    print("  RELEASE phase clicks: shoulder, elbow, wrist,")
    print("  L-hip, R-hip, L-knee, R-knee, L-ankle, R-ankle  (9 points)")
    print()


# ─────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────
def main():
    gt_df = load_gt()

    if len(sys.argv) > 1:
        vpath = sys.argv[1]
        if not os.path.exists(vpath):
            print(f"Error: file not found — {vpath}")
            sys.exit(1)
        _annotate_single(vpath, gt_df, interactive=False)
        return

    video_exts = ["*.mp4", "*.avi", "*.mov", "*.mkv",
                  "*.MP4", "*.AVI", "*.MOV", "*.MKV"]
    videos: list[str] = []
    for ext in video_exts:
        videos += glob.glob(os.path.join(VIDEOS_DIR, ext))
    videos = sorted(set(videos))

    if not videos:
        print(f"No videos found in '{VIDEOS_DIR}/'.")
        print("Place videos there, or run:  python cricket_annotator.py <video>")
        sys.exit(0)

    while True:
        vpath = pick_video(videos, gt_df)
        if vpath is None:
            break
        gt_df = _annotate_single(vpath, gt_df, interactive=True)
        again = input("\n  Annotate another video? [Y/n]: ").strip().lower()
        if again == 'n':
            break

    print("\nDone. ground_truth.csv is ready.")


def _annotate_single(vpath: str, gt_df: pd.DataFrame,
                     interactive: bool = False) -> pd.DataFrame:
    trial_id = Path(vpath).stem

    if trial_id in gt_df["trial_id"].values:
        prompt = (f"\n  Trial '{trial_id}' already annotated. "
                  f"Re-annotate? [y/N]: ")
        redo = input(prompt).strip().lower()
        if redo != 'y':
            print("  Skipped.")
            return gt_df
        gt_df = gt_df[gt_df["trial_id"] != trial_id].copy()

    print(f"\n  Opening: {vpath}")
    ann    = Annotator(vpath, trial_id, gt_df)
    result = ann.run()

    if result:
        gt_df = pd.concat([gt_df, pd.DataFrame([result])], ignore_index=True)
        save_gt(gt_df)

    return gt_df


if __name__ == "__main__":
    main()