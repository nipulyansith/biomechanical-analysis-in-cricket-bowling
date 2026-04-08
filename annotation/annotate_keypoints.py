"""
annotate_keypoints_web.py
--------------------------
Web-based Human Body Keypoint Annotation Tool
Research: Inter-Annotator Variability in Motion Analysis

CHANGES FROM PREVIOUS VERSION:
    * Calibration is now OPTIONAL and one-time — a "Calibrate Stump" button
      appears in the annotation header; any annotator can trigger it; the
      result is shared across all sessions (saved to calibration.json).
    * Annotation canvas fills most of the browser window (no fixed 1000 px cap).
    * Calibration canvas is also much larger (90 vw).
    * Pixel-to-metre conversion uses the SAME scale factor in both stages.
    * Joint dots are small (4 px radius) with NO text labels.
    * "Not Visible" button (and N key) skips a joint and marks it clearly.

DEPENDENCIES:
    pip install flask opencv-python numpy pandas

USAGE:
    python annotate_keypoints_web.py
    Then open: http://localhost:5050
"""

import os
import cv2
import json
import base64
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────
VIDEO_PATH      = "video.mp4"
OUTPUT_DIR      = "annotations"
CALIB_FILE      = "calibration.json"          # persisted calibration
FRAME_STEP      = 30
STUMP_HEIGHT_M  = 0.711

JOINTS_11 = [
    "Head",
    "Left Shoulder",  "Right Shoulder",
    "Left Elbow",     "Right Elbow",
    "Left Wrist",     "Right Wrist",
    "Left Knee",      "Right Knee",
    "Left Ankle",     "Right Ankle",
]

JOINT_COLORS = [
    "#FF6B6B","#FF9F43","#FECA57","#48DBFB","#FF9FF3",
    "#54A0FF","#5F27CD","#00D2D3","#1DD1A1","#C8D6E5","#EE5A24",
]
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = "kp_research_2025"

state = {
    "annotator":     None,
    "calibration":   None,          # loaded from file or set at runtime
    "frames":        [],
    "annotations":   {},
    "current_frame": 0,
    "video_path":    VIDEO_PATH,
    "orig_w":        1,
    "orig_h":        1,
    "fps":           30.0,
    "output_csv":    None,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_calibration():
    """Load persisted calibration from disk (if it exists)."""
    if os.path.isfile(CALIB_FILE):
        try:
            with open(CALIB_FILE) as f:
                state["calibration"] = json.load(f)
        except Exception:
            pass


def save_calibration(calib: dict):
    """Persist calibration so it survives server restarts."""
    with open(CALIB_FILE, "w") as f:
        json.dump(calib, f, indent=2)

def reset_annotation_session(keep_calibration=True):
    calib = state["calibration"] if keep_calibration else None
    state["annotator"]     = None
    state["frames"]        = state["frames"]   # keep loaded frame list
    state["annotations"]   = {}
    state["current_frame"] = 0
    state["output_csv"]    = None
    if keep_calibration:
        state["calibration"] = calib
    else:
        state["calibration"] = None       


def open_video():
    cap = cv2.VideoCapture(state["video_path"])
    if not cap.isOpened():
        raise IOError(f"Cannot open: {state['video_path']}")
    return cap


def get_frame_b64(frame_num: int):
    """
    Return (b64_jpeg, orig_h, orig_w) for a given frame number.
    We no longer downscale here — the browser handles display sizing via CSS.
    The returned b64 is the FULL-resolution frame so the pixel coordinates
    recorded always refer to the original video resolution.
    """
    cap = open_video()
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, img = cap.read()
    cap.release()
    if not ret:
        return None, None, None
    orig_h, orig_w = img.shape[:2]
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode(), orig_h, orig_w


def frame_timestamp(frame_num: int, fps: float) -> str:
    total_ms = int((frame_num / max(fps, 1)) * 1000)
    ms = total_ms % 1000
    s  = (total_ms // 1000) % 60
    m  = (total_ms // 60000) % 60
    h  = total_ms // 3600000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def build_csv_path(annotator_name: str) -> str:
    video_id = os.path.splitext(os.path.basename(state["video_path"]))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"{annotator_name}_{video_id}_{ts}.csv")


def save_csv():
    if not state["output_csv"] or not state["annotations"]:
        return
    ensure_dir(OUTPUT_DIR)
    rows  = []
    calib = state["calibration"] or {}
    mpp   = calib.get("meters_per_pixel", 0)
    ann   = state["annotator"] or {}
    for frame_num, joints in sorted(state["annotations"].items()):
        ts = frame_timestamp(frame_num, state["fps"])
        for jd in joints:
            rows.append({
                "annotator_name":   ann.get("name", ""),
                "experience_level": ann.get("experience", ""),
                "video_id":         os.path.splitext(os.path.basename(state["video_path"]))[0],
                "frame_number":     frame_num,
                "timestamp":        ts,
                "joint_name":       jd["joint"],
                "x_pixel":          jd.get("x", ""),
                "y_pixel":          jd.get("y", ""),
                "x_meter":          round(jd["x"] * mpp, 5) if not jd.get("skipped") else "",
                "y_meter":          round(jd["y"] * mpp, 5) if not jd.get("skipped") else "",
                "skipped":          jd.get("skipped", False),
                "not_visible":      jd.get("not_visible", False),
                "session_start":    ann.get("session_start", ""),
                "notes":            ann.get("notes", ""),
            })
    pd.DataFrame(rows).to_csv(state["output_csv"], index=False)


# ── HTML / JS template ────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Keypoint Annotator · Research Tool</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,400;0,500;1,400&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:       #09090f;
  --surface:  #111118;
  --surface2: #18181f;
  --border:   #242430;
  --border2:  #2e2e3e;
  --accent:   #6ee7b7;
  --accent2:  #f472b6;
  --text:     #e2e8f0;
  --muted:    #64748b;
  --danger:   #fb7185;
  --warn:     #fbbf24;
  --success:  #6ee7b7;
  --nv:       #a78bfa;   /* not-visible purple */
  --font-h:   'Syne', sans-serif;
  --font-m:   'DM Mono', monospace;
  --r:        10px;
  --sw:       260px;
}

html, body { height: 100%; background: var(--bg); color: var(--text); font-family: var(--font-m); font-size: 13px; overflow: hidden; }

body::before {
  content: ''; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
  opacity: 0.4;
}

/* Pages */
.page { position: fixed; inset: 0; z-index: 10; display: flex; flex-direction: column; align-items: center; justify-content: center; background: var(--bg); transition: opacity .3s, transform .3s; }
.page.hidden { opacity: 0; pointer-events: none; transform: translateY(10px); }

/* Card */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 36px 40px; width: 100%; max-width: 460px; box-shadow: 0 32px 80px rgba(0,0,0,.6); }
.card-title { font-family: var(--font-h); font-size: 20px; font-weight: 800; letter-spacing: -.02em; margin-bottom: 6px; }
.card-sub   { color: var(--muted); font-size: 12px; margin-bottom: 24px; line-height: 1.6; }
.wordmark   { font-family: var(--font-h); font-size: 11px; font-weight: 700; letter-spacing: .18em; text-transform: uppercase; color: var(--accent); margin-bottom: 28px; }
.wordmark span { color: var(--muted); }

/* Form */
.field { margin-bottom: 16px; }
.field label { display: block; font-size: 10px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin-bottom: 7px; }
.field input, .field select, .field textarea {
  width: 100%; background: var(--surface2); border: 1px solid var(--border2);
  border-radius: var(--r); padding: 10px 13px; color: var(--text);
  font-family: var(--font-m); font-size: 13px; outline: none;
  transition: border-color .2s, box-shadow .2s;
}
.field input:focus, .field select:focus, .field textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(110,231,183,.1); }
.field select option { background: #1a1a24; }
.field textarea { resize: none; height: 64px; }

/* Buttons */
.btn {
  display: inline-flex; align-items: center; justify-content: center; gap: 7px;
  padding: 9px 18px; border-radius: var(--r); border: 1px solid var(--border2);
  background: var(--surface2); color: var(--text); font-family: var(--font-m);
  font-size: 12px; cursor: pointer; transition: all .18s; white-space: nowrap;
}
.btn:hover   { border-color: var(--accent); background: rgba(110,231,183,.06); color: var(--accent); }
.btn:active  { transform: scale(.97); }
.btn:disabled { opacity: .3; cursor: default; pointer-events: none; }
.btn-primary { background: var(--accent); border-color: var(--accent); color: #042f2e; font-weight: 600; }
.btn-primary:hover { filter: brightness(1.1); color: #042f2e; }
.btn-danger  { border-color: var(--danger);  color: var(--danger);  }
.btn-danger:hover  { background: rgba(251,113,133,.08); }
.btn-warn    { border-color: var(--warn);    color: var(--warn);    }
.btn-warn:hover    { background: rgba(251,191,36,.08); }
.btn-success { border-color: var(--success); color: var(--success); }
.btn-success:hover { background: rgba(110,231,183,.08); }
.btn-nv      { border-color: var(--nv); color: var(--nv); }
.btn-nv:hover      { background: rgba(167,139,250,.08); }
.btn-full { width: 100%; }

/* Progress */
.prog-track { height: 3px; background: var(--border); border-radius: 99px; overflow: hidden; margin-top: 14px; }
.prog-fill  { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); border-radius: 99px; transition: width .4s cubic-bezier(.4,0,.2,1); }

/* ── Calibration MODAL (overlay on app) ────────────────────────── */
#calibModal {
  position: fixed; inset: 0; z-index: 200;
  background: rgba(9,9,15,.95); backdrop-filter: blur(8px);
  display: flex; flex-direction: column; align-items: center; justify-content: flex-start;
  padding: 20px; overflow-y: auto;
  opacity: 0; pointer-events: none; transition: opacity .3s;
}
#calibModal.open { opacity: 1; pointer-events: all; }
.calib-inner { width: 100%; max-width: 95vw; display: flex; flex-direction: column; align-items: center; gap: 14px; }
.calib-header { display: flex; align-items: center; gap: 12px; width: 100%; }
.calib-header h2 { font-family: var(--font-h); font-size: 18px; font-weight: 800; color: var(--accent); flex:1; }
.calib-hint { background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; font-size: 11px; color: var(--muted); line-height: 1.7; width: 100%; }
.calib-hint strong { color: var(--accent); }
.calib-canvas-wrap { position: relative; width: 100%; }
/* Canvas fills viewport width minus padding */
#calibCanvas {
  border-radius: 8px; border: 1px solid var(--border);
  cursor: crosshair; display: block;
  width: 100%; height: auto;   /* CSS scales display; natural size = original */
}
#calibStatus { font-size: 12px; color: var(--muted); text-align: center; min-height: 18px; }
.calib-row   { display: flex; gap: 10px; width: 100%; }

/* ── App layout ─────────────────────────────────────────────────── */
#appPage { flex-direction: row; align-items: stretch; justify-content: flex-start; padding: 0; }
#appPage.hidden { display: none; }

/* Header */
#appHeader {
  position: absolute; top: 0; left: 0; right: 0; height: 48px; z-index: 50;
  background: rgba(9,9,15,.9); backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; padding: 0 14px; gap: 10px;
}
.hdr-brand  { font-family: var(--font-h); font-weight: 800; font-size: 13px; color: var(--accent); }
.hdr-sep    { width: 1px; height: 16px; background: var(--border2); }
.hdr-pill   { background: var(--surface2); border: 1px solid var(--border2); border-radius: 99px; padding: 3px 10px; font-size: 11px; color: var(--muted); }
.hdr-pill strong { color: var(--text); }
.hdr-calib-badge { font-size: 10px; padding: 3px 10px; border-radius: 99px; border: 1px solid; }
.hdr-calib-badge.ok   { border-color: var(--success); color: var(--success); background: rgba(110,231,183,.07); }
.hdr-calib-badge.none { border-color: var(--warn);    color: var(--warn);    background: rgba(251,191,36,.07); }
.hdr-spacer { flex: 1; }
.hdr-user   { font-size: 11px; color: var(--muted); }
.hdr-user strong { color: var(--accent); }

/* Sidebar */
#sidebar {
  width: var(--sw); min-width: var(--sw); background: var(--surface);
  border-right: 1px solid var(--border); display: flex; flex-direction: column;
  padding: 54px 0 16px; overflow-y: auto; z-index: 10;
}
.sb-section  { padding: 14px 14px 0; }
.sb-lbl      { font-size: 10px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }

.frame-block { background: var(--surface2); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; margin-bottom: 12px; }
.frame-block .fb-lbl { font-size: 10px; color: var(--muted); margin-bottom: 2px; }
.frame-block .fb-val { font-family: var(--font-h); font-size: 26px; font-weight: 800; color: var(--accent); letter-spacing: -.04em; line-height: 1; }
.frame-block .fb-ts  { font-size: 10px; color: var(--muted); margin-top: 4px; }

.joint-list  { display: flex; flex-direction: column; gap: 4px; padding: 0 10px; }
.joint-item  { display: flex; align-items: center; gap: 8px; padding: 6px 8px; border-radius: 7px; border: 1px solid transparent; transition: all .15s; }
.joint-item.j-done     { border-color: rgba(110,231,183,.25); background: rgba(110,231,183,.04); }
.joint-item.j-active   { border-color: rgba(110,231,183,.5);  background: rgba(110,231,183,.08); }
.joint-item.j-novis    { border-color: rgba(167,139,250,.25); background: rgba(167,139,250,.04); opacity:.8; }
.joint-item.j-pending  { opacity: .38; }
.j-dot    { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.j-name   { flex: 1; font-size: 11px; }
.j-coords { font-size: 10px; color: var(--muted); }
.j-badge  { font-size: 10px; padding: 2px 6px; border-radius: 4px; background: var(--border); color: var(--muted); }
.j-badge.done   { background: rgba(110,231,183,.15); color: var(--success); }
.j-badge.active { background: rgba(110,231,183,.2);  color: var(--accent); }
.j-badge.novis  { background: rgba(167,139,250,.18); color: var(--nv); }

.sb-controls { padding: 12px 10px 0; display: flex; flex-direction: column; gap: 6px; }
.sb-div      { height: 1px; background: var(--border); margin: 12px 10px 0; }
.sc-list     { padding: 8px 14px 0; display: flex; flex-direction: column; gap: 5px; }
.sc-row      { display: flex; align-items: center; gap: 7px; font-size: 11px; color: var(--muted); }
.sc-key      { background: var(--surface2); border: 1px solid var(--border2); border-radius: 4px; padding: 1px 6px; font-size: 10px; color: var(--text); }

/* Main canvas area — fills all remaining space */
#main {
  flex: 1; display: flex; align-items: flex-start; justify-content: center;
  padding: 54px 16px 16px; background: #050508; overflow: auto;
}
#canvasWrap {
  position: relative; display: inline-block;
  border-radius: 6px; overflow: hidden;
  box-shadow: 0 0 0 1px var(--border), 0 16px 60px rgba(0,0,0,.8);
}
/* Fill most of the available area */
#annotCanvas {
  display: block;
  max-width:  calc(100vw - var(--sw) - 48px);
  max-height: calc(100vh - 70px);
  width:  auto;
  height: auto;
}
#cursorDot {
  position: absolute; width: 10px; height: 10px;
  border: 2px solid rgba(110,231,183,.8); border-radius: 50%;
  pointer-events: none; transform: translate(-50%,-50%);
  opacity: 0; box-shadow: 0 0 6px rgba(110,231,183,.5);
}

/* Toast */
#toast {
  position: fixed; bottom: 20px; right: 20px; z-index: 300;
  background: var(--surface2); border: 1px solid var(--border2);
  border-radius: 10px; padding: 10px 16px; font-size: 12px; color: var(--text);
  box-shadow: 0 8px 32px rgba(0,0,0,.5);
  transform: translateY(20px); opacity: 0; transition: all .25s; pointer-events: none;
}
#toast.show { transform: translateY(0); opacity: 1; }
#toast.t-success { border-color: var(--success); color: var(--success); }
#toast.t-warn    { border-color: var(--warn);    color: var(--warn); }
#toast.t-danger  { border-color: var(--danger);  color: var(--danger); }
#toast.t-nv      { border-color: var(--nv);      color: var(--nv); }

/* Done banner */
#doneBanner {
  position: fixed; inset: 0; z-index: 100;
  background: rgba(9,9,15,.92); backdrop-filter: blur(8px);
  display: flex; align-items: center; justify-content: center;
  opacity: 0; pointer-events: none; transition: opacity .4s;
}
#doneBanner.show { opacity: 1; pointer-events: all; }
.done-card { background: var(--surface); border: 1px solid var(--success); border-radius: 20px; padding: 44px 52px; text-align: center; max-width: 420px; box-shadow: 0 0 60px rgba(110,231,183,.1), 0 32px 80px rgba(0,0,0,.6); }
.done-icon  { font-size: 44px; margin-bottom: 14px; }
.done-title { font-family: var(--font-h); font-size: 24px; font-weight: 800; color: var(--success); margin-bottom: 8px; }
.done-sub   { color: var(--muted); font-size: 12px; line-height: 1.7; }

@keyframes fadeUp { from { opacity:0; transform:translateY(10px) } to { opacity:1; transform:none } }
.fade-in { animation: fadeUp .3s ease forwards; }
</style>
</head>
<body>

<!-- ═══ PAGE 1: METADATA ═══════════════════════════════════════════ -->
<div id="metaPage" class="page">
  <div class="wordmark">Keypoint <span>·</span> Research Tool</div>
  <div class="card fade-in">
    <div class="card-title">Annotator Setup</div>
    <div class="card-sub">Your annotations will be saved with your name and session details for inter-annotator variability analysis.</div>
    <div class="field">
      <label>Annotator Name <span style="color:var(--danger)">*</span></label>
      <input id="metaName" type="text" placeholder="e.g. john_doe" autocomplete="off" spellcheck="false">
    </div>
    <div class="field">
      <label>Experience Level</label>
      <select id="metaExp">
        <option value="novice">Novice — &lt; 1 year motion annotation</option>
        <option value="intermediate">Intermediate — 1–3 years</option>
        <option value="expert">Expert — &gt; 3 years</option>
      </select>
    </div>
    <div class="field">
      <label>Notes (optional)</label>
      <textarea id="metaNotes" placeholder="Any relevant context about this session…"></textarea>
    </div>
    <button class="btn btn-primary btn-full" onclick="submitMeta()">Start Annotating →</button>
  </div>
</div>

<!-- ═══ CALIBRATION MODAL (floats over annotation page) ═══════════ -->
<div id="calibModal">
  <div class="calib-inner">
    <div class="calib-header">
      <h2>Stump Calibration</h2>
      <button class="btn btn-warn" onclick="resetCalib()">Reset (R)</button>
      <button class="btn" onclick="closeCalib()">✕ Cancel</button>
    </div>
    <div class="calib-hint">
      Known stump height: <strong>{{ stump_h }} m</strong> &nbsp;·&nbsp;
      Click 1 = <strong>top of stump</strong> &nbsp;·&nbsp;
      Click 2 = <strong>bottom of stump</strong><br>
      The scale is saved globally and shared with all annotators.
    </div>
    <div class="calib-canvas-wrap">
      <canvas id="calibCanvas"></canvas>
    </div>
    <div id="calibStatus">Loading first frame…</div>
    <div class="calib-row">
      <button class="btn btn-warn" onclick="resetCalib()" style="flex:1">Reset Clicks</button>
      <button id="btnCalibOk" class="btn btn-primary" style="flex:3" disabled onclick="confirmCalib()">✓ Save Calibration</button>
    </div>
    <div class="prog-track" style="width:100%"><div class="prog-fill" id="calibProg" style="width:0%"></div></div>
    <div style="font-size:11px; color:var(--muted); text-align:center; padding-bottom:20px" id="calibInfo">0 / 2 clicks</div>
  </div>
</div>

<!-- ═══ PAGE 2: ANNOTATION APP ═════════════════════════════════════ -->
<div id="appPage" class="page hidden" style="position:relative;">

  <div id="appHeader">
    <div class="hdr-brand">KP Annotator</div>
    <div class="hdr-sep"></div>
    <div class="hdr-pill">Frame <strong id="hdrFr">—</strong></div>
    <div class="hdr-pill"><strong id="hdrDone">0</strong> / <strong id="hdrTot">—</strong></div>
    <div class="hdr-pill" id="hdrNext">—</div>
    <div id="calibBadge" class="hdr-calib-badge none" onclick="openCalib()" style="cursor:pointer" title="Click to calibrate">
      ⚙ No Calibration
    </div>
    <div class="hdr-spacer"></div>
    <div class="hdr-user">Annotator: <strong id="hdrUser">—</strong></div>
    <div class="hdr-sep"></div>
    <button class="btn" style="padding:5px 12px; font-size:11px; border-color:var(--accent); color:var(--accent)" onclick="openCalib()">Calibrate Stump</button>
    <button class="btn btn-warn" style="padding:5px 12px; font-size:11px" onclick="saveQuit()">Save &amp; Quit</button>
  </div>

  <div id="sidebar">
    <div class="sb-section">
      <div class="sb-lbl">Current Frame</div>
      <div class="frame-block">
        <div class="fb-lbl">Frame #</div>
        <div class="fb-val" id="sbFn">—</div>
        <div class="fb-ts"  id="sbTs">—</div>
      </div>
      <div class="sb-lbl" style="margin-top:2px">Overall Progress</div>
      <div class="prog-track" style="margin-bottom:12px">
        <div class="prog-fill" id="sbProg" style="width:0%"></div>
      </div>
    </div>
    <div class="sb-section" style="padding-top:0">
      <div class="sb-lbl">Joints — click in order</div>
    </div>
    <div class="joint-list" id="jointList"></div>

    <div class="sb-div"></div>
    <div class="sb-controls">
      <button class="btn btn-warn btn-full"     onclick="undoLast()">↩ Undo Last (Z)</button>
      <button id="btnNotVis" class="btn btn-nv btn-full" onclick="markNotVisible()">👁 Not Visible (N)</button>
      <button class="btn btn-danger btn-full"   onclick="resetFrame()">✕ Reset Frame (R)</button>
      <button id="btnConfirm" class="btn btn-success btn-full" disabled onclick="confirmFrame()">✓ Confirm Frame (Enter)</button>
    </div>

    <div class="sb-div"></div>
    <div class="sc-list">
      <div class="sb-lbl" style="padding:0 4px">Shortcuts</div>
      <div class="sc-row"><span class="sc-key">Z</span> Undo last click</div>
      <div class="sc-row"><span class="sc-key">N</span> Mark not visible</div>
      <div class="sc-row"><span class="sc-key">R</span> Reset frame</div>
      <div class="sc-row"><span class="sc-key">Enter</span> Confirm frame</div>
    </div>
  </div>

  <div id="main">
    <div id="canvasWrap">
      <canvas id="annotCanvas"></canvas>
      <div id="cursorDot"></div>
    </div>
  </div>

  <div id="doneBanner">
    <div class="done-card">
      <div class="done-icon">✓</div>
      <div class="done-title">All Frames Done!</div>
      <div class="done-sub" id="doneMsg">All annotations saved.</div>
      <button class="btn btn-success" style="margin-top:22px; width:100%" onclick="saveQuit()">Save &amp; Close</button>
    </div>
  </div>
</div>

<div id="toast"></div>

<!-- ══════════════ JAVASCRIPT ══════════════════════════════════════ -->
<script>
const JOINTS  = {{ joint_names  | tojson }};
const COLORS  = {{ joint_colors | tojson }};
const FRAMES  = {{ frames       | tojson }};
const STUMP_H = {{ stump_h }};

/* ── Page navigation ─────────────────────────────────────────────── */
function showPage(id) {
  ['metaPage','appPage'].forEach(p => {
    document.getElementById(p).classList.toggle('hidden', p !== id);
  });
}

/* ── Toast ───────────────────────────────────────────────────────── */
let _tt;
function toast(msg, type='') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast show' + (type ? ' t-'+type : '');
  clearTimeout(_tt);
  _tt = setTimeout(() => el.classList.remove('show'), 2800);
}

/* ── Calibration helpers ─────────────────────────────────────────── */
let calibClicks = [], calibImgB64 = null;

function updateCalibBadge(data) {
  const badge = document.getElementById('calibBadge');
  if (data && data.pixels_per_meter) {
    badge.className = 'hdr-calib-badge ok';
    badge.textContent = `⚙ ${data.pixels_per_meter.toFixed(1)} px/m`;
  } else {
    badge.className = 'hdr-calib-badge none';
    badge.textContent = '⚙ No Calibration';
  }
}

function openCalib() {
  const modal = document.getElementById('calibModal');
  modal.classList.add('open');
  if (!calibImgB64) initCalib();
}

function closeCalib() {
  document.getElementById('calibModal').classList.remove('open');
}

function initCalib() {
  fetch('/api/calibframe').then(r => r.json()).then(d => {
    calibImgB64 = d.b64;
    drawCalib();
    document.getElementById('calibStatus').textContent = 'Click 1: top of stump';
  });
}

function drawCalib() {
  const canvas = document.getElementById('calibCanvas');
  const ctx    = canvas.getContext('2d');
  const img    = new Image();
  img.onload = () => {
    /* Set the canvas NATURAL size to the original frame size.
       CSS (width:100%; height:auto) handles display scaling.
       Clicks are converted back using getBoundingClientRect. */
    canvas.width  = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);

    const lbls = ['TOP','BOTTOM'], cols = ['#6ee7b7','#f472b6'];
    calibClicks.forEach(([cx,cy], i) => {
      ctx.beginPath(); ctx.arc(cx, cy, 14, 0, Math.PI*2);
      ctx.fillStyle = cols[i]; ctx.globalAlpha = 0.85; ctx.fill();
      ctx.globalAlpha = 1;
      ctx.beginPath(); ctx.arc(cx, cy, 14, 0, Math.PI*2);
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2.5; ctx.stroke();
      // crosshair
      ctx.strokeStyle = 'rgba(255,255,255,.7)'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(cx-24,cy); ctx.lineTo(cx+24,cy); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx,cy-24); ctx.lineTo(cx,cy+24); ctx.stroke();
      // label
      ctx.font = 'bold 14px "DM Mono",monospace';
      ctx.fillStyle = '#000'; ctx.fillText(lbls[i], cx+17, cy-10);
      ctx.fillStyle = cols[i]; ctx.fillText(lbls[i], cx+16, cy-11);
    });
    if (calibClicks.length === 2) {
      const [a,b] = calibClicks;
      ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]);
      ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.setLineDash([8,5]); ctx.lineWidth=2; ctx.stroke(); ctx.setLineDash([]);
    }
  };
  img.src = 'data:image/jpeg;base64,' + calibImgB64;
}

document.getElementById('calibCanvas').addEventListener('click', e => {
  if (calibClicks.length >= 2) return;
  const canvas = document.getElementById('calibCanvas');
  const rect   = canvas.getBoundingClientRect();
  /* Scale from CSS display size → original pixel coordinates */
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top)  * scaleY;
  calibClicks.push([x, y]);
  drawCalib(); updateCalibUI();
});

function updateCalibUI() {
  const n = calibClicks.length;
  document.getElementById('calibInfo').textContent = n + ' / 2 clicks';
  document.getElementById('calibProg').style.width = (n * 50) + '%';
  document.getElementById('calibStatus').textContent =
    n===0 ? 'Click 1: top of stump' :
    n===1 ? 'Click 2: bottom of stump' :
            '✓ Both points set — confirm to save.';
  document.getElementById('btnCalibOk').disabled = n < 2;
}

function resetCalib() { calibClicks = []; updateCalibUI(); drawCalib(); }

function confirmCalib() {
  if (calibClicks.length < 2) return;
  fetch('/api/calibrate', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ top: calibClicks[0], bottom: calibClicks[1] })
  })
  .then(r => r.json())
  .then(d => {
    if (d.ok) {
      toast(`Calibration saved — ${d.pixels_per_meter.toFixed(1)} px/m`, 'success');
      updateCalibBadge(d);
      setTimeout(closeCalib, 400);
    } else toast(d.error || 'Error', 'danger');
  });
}

/* Keyboard: R resets calibration when modal is open */
document.addEventListener('keydown', e => {
  if (document.getElementById('calibModal').classList.contains('open')) {
    if (e.key==='r'||e.key==='R') resetCalib();
    if (e.key==='Escape') closeCalib();
    return;
  }
  if (!document.getElementById('appPage').classList.contains('hidden')) {
    const k = e.key;
    if ((k==='z'||k==='Z') || ((e.ctrlKey||e.metaKey)&&k==='z')) { e.preventDefault(); undoLast(); }
    else if (k==='n'||k==='N') markNotVisible();
    else if (k==='r'||k==='R') resetFrame();
    else if (k==='Enter') { if (!document.getElementById('btnConfirm').disabled) confirmFrame(); }
  }
});

/* ── PAGE 1: METADATA ────────────────────────────────────────────── */
function submitMeta() {
  const name = document.getElementById('metaName').value.trim();
  if (!name) { toast('Please enter your name.', 'warn'); return; }
  fetch('/api/setup', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      name,
      experience: document.getElementById('metaExp').value,
      notes:      document.getElementById('metaNotes').value.trim()
    })
  })
  .then(r => r.json())
  .then(d => {
    if (d.ok) {
      showPage('appPage');
      initApp(d.calibration);
    } else toast(d.error || 'Server error', 'danger');
  });
}

/* ── PAGE 2: ANNOTATION ──────────────────────────────────────────── */
let frameIdx = 0, placements = [], curIdx = 0, imgNaturalW = 1, imgNaturalH = 1;
let imgData = null;

function initApp(existingCalib) {
  document.getElementById('hdrUser').textContent = document.getElementById('metaName').value.trim();
  document.getElementById('hdrTot').textContent  = FRAMES.length;
  if (existingCalib) updateCalibBadge(existingCalib);
  const canvas = document.getElementById('annotCanvas');
  canvas.addEventListener('click', onAnnotClick);
  canvas.addEventListener('mousemove', onMove);
  canvas.addEventListener('mouseleave', () => { document.getElementById('cursorDot').style.opacity='0'; });
  fetch('/api/state').then(r=>r.json()).then(d => { frameIdx = d.current_frame || 0; loadFrame(); });
}

function loadFrame() {
  if (frameIdx >= FRAMES.length) { showDone(); return; }
  placements = []; curIdx = 0;
  const fn = FRAMES[frameIdx];
  document.getElementById('hdrFr').textContent   = fn;
  document.getElementById('sbFn').textContent    = fn;
  document.getElementById('hdrDone').textContent = frameIdx;
  document.getElementById('sbProg').style.width  = ((frameIdx / FRAMES.length) * 100) + '%';
  fetch('/api/frame/' + fn).then(r => r.json()).then(d => {
    imgData      = d.b64;
    imgNaturalW  = d.w;
    imgNaturalH  = d.h;
    document.getElementById('sbTs').textContent = d.ts || '';
    drawCanvas(); renderList();
  });
}

function onMove(e) {
  const canvas = document.getElementById('annotCanvas');
  const dot    = document.getElementById('cursorDot');
  const rect   = canvas.getBoundingClientRect();
  dot.style.opacity = '1';
  dot.style.left = (e.clientX - rect.left + canvas.offsetLeft) + 'px';
  dot.style.top  = (e.clientY - rect.top  + canvas.offsetTop)  + 'px';
}

/* Convert a click on the CSS-scaled canvas to original pixel coordinates */
function toNatural(e) {
  const canvas = document.getElementById('annotCanvas');
  const rect   = canvas.getBoundingClientRect();
  const scaleX = imgNaturalW / rect.width;
  const scaleY = imgNaturalH / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top)  * scaleY
  };
}

function onAnnotClick(e) {
  if (curIdx >= JOINTS.length) return;
  const {x, y} = toNatural(e);
  placements.push({ name: JOINTS[curIdx], x, y, skipped: false, not_visible: false });
  curIdx++;
  drawCanvas(); renderList();
  if (curIdx >= JOINTS.length) toast('All joints placed! Press Enter to confirm.', 'success');
}

function drawCanvas() {
  if (!imgData) return;
  const canvas = document.getElementById('annotCanvas');
  const ctx    = canvas.getContext('2d');
  const img    = new Image();
  img.onload = () => {
    /* Set natural size so coordinates are always in original pixels */
    canvas.width  = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);

    /* Subtle grid */
    ctx.strokeStyle = 'rgba(255,255,255,0.03)'; ctx.lineWidth = 0.5;
    for (let x=0; x<img.width;  x+=80) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,img.height); ctx.stroke(); }
    for (let y=0; y<img.height; y+=80) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(img.width,y);   ctx.stroke(); }

    /* Skeleton lines between placed joints */
    const km = {};
    placements.forEach(p => { if (!p.skipped && !p.not_visible) km[p.name] = {x:p.x, y:p.y}; });
    const links = [
      ['Left Shoulder','Right Shoulder'],
      ['Left Shoulder','Left Elbow'],  ['Left Elbow','Left Wrist'],
      ['Right Shoulder','Right Elbow'],['Right Elbow','Right Wrist'],
      ['Left Shoulder','Left Knee'],   ['Right Shoulder','Right Knee'],
      ['Left Knee','Left Ankle'],      ['Right Knee','Right Ankle'],
    ];
    ctx.lineWidth = 1.5; ctx.setLineDash([4,5]); ctx.strokeStyle = 'rgba(255,255,255,.2)';
    links.forEach(([a,b]) => {
      if (km[a] && km[b]) {
        ctx.beginPath(); ctx.moveTo(km[a].x, km[a].y); ctx.lineTo(km[b].x, km[b].y); ctx.stroke();
      }
    });
    ctx.setLineDash([]);

    /* Joint dots — small (radius 5), no text labels */
    placements.forEach((p, i) => {
      if (p.not_visible) return;   /* not-visible joints get no dot */
      if (p.skipped)     return;
      const c = COLORS[i];
      /* Small filled circle */
      ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, Math.PI*2);
      ctx.fillStyle = c; ctx.fill();
      /* Thin outer ring */
      ctx.beginPath(); ctx.arc(p.x, p.y, 8, 0, Math.PI*2);
      ctx.strokeStyle = c; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.6; ctx.stroke();
      ctx.globalAlpha = 1;
    });

    /* "Next joint" prompt — bottom-left corner, small */
    if (curIdx < JOINTS.length) {
      const c = COLORS[curIdx];
      const label = '→ ' + JOINTS[curIdx];
      ctx.font = 'bold 13px "DM Mono",monospace';
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(0,0,0,.65)';
      ctx.fillRect(8, img.height - 32, tw + 16, 26);
      ctx.fillStyle = c;
      ctx.fillText(label, 16, img.height - 14);
    }
  };
  img.src = 'data:image/jpeg;base64,' + imgData;
}

function renderList() {
  const list = document.getElementById('jointList');
  list.innerHTML = '';
  for (let i = 0; i < JOINTS.length; i++) {
    const p   = placements[i];
    const div = document.createElement('div');
    let cls = 'joint-item ', badge = '', coords = '';
    if (p) {
      if (p.not_visible) {
        cls += 'j-novis';
        badge = '<span class="j-badge novis">👁‍🗨 n/v</span>';
      } else if (p.skipped) {
        cls += 'j-novis';
        badge = '<span class="j-badge novis">skip</span>';
      } else {
        cls += 'j-done';
        badge  = '<span class="j-badge done">✓</span>';
        coords = `<span class="j-coords">${Math.round(p.x)},${Math.round(p.y)}</span>`;
      }
    } else if (i === curIdx) {
      cls += 'j-active';
      badge = '<span class="j-badge active">← next</span>';
    } else {
      cls += 'j-pending';
      badge = '<span class="j-badge">—</span>';
    }
    div.className = cls;
    div.innerHTML = `<div class="j-dot" style="background:${COLORS[i]}"></div><span class="j-name">${JOINTS[i]}</span>${coords}${badge}`;
    list.appendChild(div);
  }
  const allDone = curIdx >= JOINTS.length;
  document.getElementById('btnConfirm').disabled = !allDone;
  document.getElementById('btnNotVis').disabled  =  allDone;
  if (!allDone) {
    document.getElementById('hdrNext').innerHTML = `Next: <strong style="color:${COLORS[curIdx]}">${JOINTS[curIdx]}</strong>`;
  } else {
    document.getElementById('hdrNext').innerHTML = `<strong style="color:var(--success)">All joints ✓</strong>`;
  }
}

function undoLast() {
  if (!placements.length) { toast('Nothing to undo.', 'warn'); return; }
  placements.pop(); curIdx = placements.length; drawCanvas(); renderList();
}

function markNotVisible() {
  if (curIdx >= JOINTS.length) return;
  placements.push({ name: JOINTS[curIdx], skipped: true, not_visible: true });
  curIdx++;
  drawCanvas(); renderList();
  toast('Marked not visible: ' + JOINTS[curIdx-1], 'nv');
}

function resetFrame() { placements = []; curIdx = 0; drawCanvas(); renderList(); toast('Frame reset.', 'warn'); }

function confirmFrame() {
  if (curIdx < JOINTS.length) return;
  const fn   = FRAMES[frameIdx];
  const data = {};
  placements.forEach((p, i) => {
    data[JOINTS[i]] = (p.skipped || p.not_visible) ? null : [p.x, p.y];
  });
  /* Also send the full placement metadata so not_visible is stored */
  fetch('/api/annotate', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ frame: fn, keypoints: data, placements })
  })
  .then(r => r.json())
  .then(d => {
    if (d.ok) { toast(`Frame ${fn} saved ✓`, 'success'); frameIdx++; setTimeout(loadFrame, 200); }
    else toast(d.error || 'Save error', 'danger');
  });
}

function showDone() {
  document.getElementById('doneBanner').classList.add('show');
  document.getElementById('doneMsg').innerHTML =
    `Annotated <strong>${FRAMES.length}</strong> frames.<br>CSV saved to <code>annotations/</code>.<br><br>Thank you for contributing to the research!`;
}

function saveQuit() {
  fetch('/api/save', { method: 'POST' })
    .then(r => r.json())
    .then(d => {
      toast(d.path ? `Saved: ${d.path}` : 'Progress saved.', 'success');

      setTimeout(() => {
        // reset UI back to homepage
        document.getElementById('doneBanner').classList.remove('show');
        document.getElementById('metaName').value = '';
        document.getElementById('metaNotes').value = '';
        document.getElementById('metaExp').value = 'novice';

        frameIdx = 0;
        placements = [];
        curIdx = 0;
        imgData = null;

        showPage('metaPage');
      }, 700);
    });
}
</script>
</body>
</html>
"""


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    cap = open_video()
    state["fps"]    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total           = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    state["orig_w"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    state["orig_h"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    frames = list(range(0, total, FRAME_STEP))
    state["frames"] = frames
    load_calibration()
    return render_template_string(HTML,
        joint_names  = JOINTS_11,
        joint_colors = JOINT_COLORS,
        frames       = frames,
        stump_h      = STUMP_HEIGHT_M,
    )


@app.route("/api/setup", methods=["POST"])
def api_setup():
    data = request.get_json()
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "Name is required."})
    state["annotator"] = {
        "name":          name,
        "experience":    data.get("experience", "novice"),
        "notes":         data.get("notes", ""),
        "session_start": datetime.now().isoformat(timespec="seconds"),
    }
    state["output_csv"] = build_csv_path(name)
    # Return existing calibration so the badge updates immediately
    return jsonify({"ok": True, "calibration": state.get("calibration")})


@app.route("/api/calibframe")
def api_calibframe():
    """Return the first frame at FULL resolution for calibration."""
    b64, orig_h, orig_w = get_frame_b64(0)
    if b64 is None:
        return jsonify({"error": "Cannot read frame"}), 500
    return jsonify({"b64": b64, "h": orig_h, "w": orig_w})


@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    data   = request.get_json()
    top    = data["top"]
    bottom = data["bottom"]
    dist   = np.hypot(bottom[0] - top[0], bottom[1] - top[1])
    if dist < 5:
        return jsonify({"ok": False, "error": "Points too close — please re-click."})
    ppm = dist / STUMP_HEIGHT_M          # pixels per metre
    mpp = STUMP_HEIGHT_M / dist          # metres per pixel
    calib = {
        "top":               top,
        "bottom":            bottom,
        "pixel_distance":    round(dist, 2),
        "pixels_per_meter":  round(ppm, 4),
        "meters_per_pixel":  round(mpp, 6),
        "calibrated_at":     datetime.now().isoformat(timespec="seconds"),
    }
    state["calibration"] = calib
    save_calibration(calib)              # persist to disk
    return jsonify({"ok": True, "pixels_per_meter": ppm, "meters_per_pixel": mpp})


@app.route("/api/state")
def api_state():
    return jsonify({"current_frame": state["current_frame"]})


@app.route("/api/frame/<int:frame_num>")
def api_frame(frame_num):
    """Return a frame at FULL resolution. Browser scales via CSS."""
    b64, orig_h, orig_w = get_frame_b64(frame_num)
    if b64 is None:
        return jsonify({"error": "frame not found"}), 404
    ts = frame_timestamp(frame_num, state["fps"])
    return jsonify({"b64": b64, "h": orig_h, "w": orig_w, "ts": ts})


@app.route("/api/annotate", methods=["POST"])
def api_annotate():
    data       = request.get_json()
    frame_num  = int(data["frame"])
    keypoints  = data["keypoints"]
    raw_placements = data.get("placements", [])
    calib      = state["calibration"] or {}
    mpp        = calib.get("meters_per_pixel", 0)

    # Build a lookup for not_visible flag from raw placements
    nv_map = {p["name"]: p.get("not_visible", False) for p in raw_placements}

    joint_rows = []
    for jname in JOINTS_11:
        pt = keypoints.get(jname)
        not_vis = nv_map.get(jname, False)
        if pt:
            ox, oy = pt[0], pt[1]   # already in original pixel coords
            joint_rows.append({
                "joint":       jname,
                "x":           ox,
                "y":           oy,
                "xm":          round(ox * mpp, 5),
                "ym":          round(oy * mpp, 5),
                "skipped":     False,
                "not_visible": False,
            })
        else:
            joint_rows.append({
                "joint":       jname,
                "skipped":     True,
                "not_visible": not_vis,
            })

    state["annotations"][frame_num] = joint_rows
    try:
        state["current_frame"] = state["frames"].index(frame_num) + 1
    except ValueError:
        state["current_frame"] += 1
    save_csv()
    return jsonify({"ok": True})


@app.route("/api/save", methods=["POST"])
def api_save():
    saved_path = state.get("output_csv", "")
    save_csv()
    reset_annotation_session(keep_calibration=True)
    return jsonify({
        "ok": True,
        "path": saved_path,
        "reset": True
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.isfile(VIDEO_PATH):
        print(f"\n[WARNING] Video not found: '{VIDEO_PATH}'")
        print("          Set VIDEO_PATH at the top of the script.\n")
    ensure_dir(OUTPUT_DIR)
    load_calibration()
    print("=" * 60)
    print("  Human Keypoint Annotation Tool — Web Interface")
    print("=" * 60)
    print(f"  Video      : {VIDEO_PATH}")
    print(f"  Output dir : {OUTPUT_DIR}/")
    print(f"  Frame step : every {FRAME_STEP} frames")
    print(f"  Stump ht   : {STUMP_HEIGHT_M} m")
    if state["calibration"]:
        ppm = state["calibration"].get("pixels_per_meter", "?")
        print(f"  Calibration: {ppm} px/m (loaded from {CALIB_FILE})")
    else:
        print(f"  Calibration: not yet set (use 'Calibrate Stump' button in app)")
    print()
    print("  Open in browser:  http://localhost:5050")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5050, debug=False)