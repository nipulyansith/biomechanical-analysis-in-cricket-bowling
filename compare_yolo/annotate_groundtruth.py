"""
annotate_groundtruth.py
-----------------------
Local web app for annotating the 8 keypoints on every 30th frame.

Usage:
    python annotate_groundtruth.py

Then open  http://localhost:5000  in your browser.

Controls:
  • Click on the image to place the next keypoint in order.
  • Undo last click with the [Undo] button.
  • Skip a keypoint (mark as missing) with [Skip].
  • When all 8 keypoints are placed → [Next Frame] becomes available.
  • [Save & Quit] writes  output/groundtruth.csv  at any time.

Output:
    output/groundtruth.csv   (same schema as nmodel.csv / lmodel.csv)
"""

import os
import cv2
import base64
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

# ── Settings ────────────────────────────────────────────────────────────────
VIDEO_PATH   = "../data/geenod.MOV"
OUTPUT_CSV   = "output/groundtruth.csv"
FRAME_STEP   = 30
DISPLAY_W    = 960    # Width the image is shown at in the browser

KEYPOINTS_8  = [
    "left_shoulder",  "right_shoulder",
    "left_elbow",     "right_elbow",
    "left_wrist",     "right_wrist",
    "left_ankle",     "right_ankle",
]

KP_COLORS = [
    "#FF6B6B", "#FF9F43", "#FECA57", "#48DBFB",
    "#FF9FF3", "#54A0FF", "#5F27CD", "#00D2D3",
]
# ────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# Global state
state = {
    "frames":      [],   # list of frame numbers to annotate
    "frame_idx":   0,    # index into frames[]
    "annotations": {},   # {frame_number: {kp_name: [x,y] or None}}
    "video_path":  VIDEO_PATH,
    "display_w":   DISPLAY_W,
    "orig_w":      1,
    "orig_h":      1,
}


def load_video_meta():
    cap = cv2.VideoCapture(state["video_path"])
    if not cap.isOpened():
        raise IOError(f"Cannot open {state['video_path']}")
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    state["orig_w"] = w
    state["orig_h"] = h
    return total, w, h


def frame_to_jpeg_b64(frame_number):
    cap = cv2.VideoCapture(state["video_path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ret, img = cap.read()
    cap.release()
    if not ret:
        return None
    scale    = state["display_w"] / img.shape[1]
    new_h    = int(img.shape[0] * scale)
    img_small = cv2.resize(img, (state["display_w"], new_h))
    _, buf = cv2.imencode(".jpg", img_small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8"), new_h


def load_existing_csv():
    """Resume from a partially saved groundtruth.csv if it exists."""
    if not os.path.exists(OUTPUT_CSV):
        return
    df = pd.read_csv(OUTPUT_CSV)
    for _, row in df.iterrows():
        f = int(row["frame"])
        ann = {}
        for kp in KEYPOINTS_8:
            x = row.get(f"{kp}_x", np.nan)
            y = row.get(f"{kp}_y", np.nan)
            ann[kp] = [float(x), float(y)] if pd.notna(x) and pd.notna(y) else None
        state["annotations"][f] = ann
    # Advance frame_idx past already-annotated frames
    annotated_frames = set(state["annotations"].keys())
    for i, f in enumerate(state["frames"]):
        if f not in annotated_frames:
            state["frame_idx"] = i
            return
    state["frame_idx"] = len(state["frames"])   # all done


def save_csv():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    rows = []
    for frame_num in sorted(state["annotations"].keys()):
        ann  = state["annotations"][frame_num]
        row  = {"frame": frame_num}
        for kp in KEYPOINTS_8:
            pt = ann.get(kp)
            # Convert display coords back to original video coords
            scale = state["orig_w"] / state["display_w"]
            if pt:
                row[f"{kp}_x"] = pt[0] * scale
                row[f"{kp}_y"] = pt[1] * scale
            else:
                row[f"{kp}_x"] = np.nan
                row[f"{kp}_y"] = np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)


# ── HTML template ────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Ground Truth Annotator</title>
<style>
  :root {
    --bg: #0f0f14;
    --surface: #1a1a24;
    --border: #2e2e42;
    --accent: #7c6aff;
    --text: #e8e8f0;
    --muted: #6b6b85;
    --danger: #ff4d6d;
    --success: #3ddc84;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'JetBrains Mono', 'Fira Code', monospace; min-height: 100vh; }
  
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 14px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); }
  .progress-pill { background: var(--border); border-radius: 20px; padding: 4px 14px; font-size: 12px; color: var(--muted); }
  .progress-pill span { color: var(--text); font-weight: 600; }

  .layout { display: flex; height: calc(100vh - 53px); }

  /* Left panel */
  .sidebar { width: 280px; min-width: 280px; background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; padding: 20px 16px; gap: 12px; overflow-y: auto; }
  .sidebar h2 { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 4px; }

  .kp-list { display: flex; flex-direction: column; gap: 6px; }
  .kp-item { display: flex; align-items: center; gap: 10px; padding: 8px 10px; border-radius: 8px; border: 1px solid var(--border); font-size: 12px; transition: all .15s; }
  .kp-item.active { border-color: var(--accent); background: rgba(124,106,255,.08); }
  .kp-item.done   { border-color: var(--success); opacity: 0.7; }
  .kp-item.skipped{ border-color: var(--danger); opacity: 0.5; }
  .kp-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
  .kp-name { flex: 1; }
  .kp-coords { font-size: 10px; color: var(--muted); }
  .kp-badge { font-size: 10px; padding: 2px 6px; border-radius: 4px; }
  .kp-badge.done    { background: rgba(61,220,132,.15); color: var(--success); }
  .kp-badge.skipped { background: rgba(255,77,109,.15); color: var(--danger); }
  .kp-badge.active  { background: rgba(124,106,255,.2); color: var(--accent); }
  .kp-badge.pending { background: var(--border); color: var(--muted); }

  .btn-row { display: flex; flex-direction: column; gap: 8px; margin-top: auto; }
  button { padding: 10px 16px; border-radius: 8px; border: 1px solid var(--border); background: var(--surface); color: var(--text); font-family: inherit; font-size: 12px; cursor: pointer; transition: all .15s; text-align: center; }
  button:hover { border-color: var(--accent); background: rgba(124,106,255,.08); }
  button:disabled { opacity: 0.3; cursor: default; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
  button.primary:hover { filter: brightness(1.15); }
  button.danger { border-color: var(--danger); color: var(--danger); }
  button.danger:hover { background: rgba(255,77,109,.08); }
  button.success { border-color: var(--success); color: var(--success); }
  button.success:hover { background: rgba(61,220,132,.08); }

  /* Canvas area */
  .canvas-wrap { flex: 1; overflow: auto; display: flex; align-items: flex-start; justify-content: center; padding: 24px; background: #08080e; }
  #annotCanvas { cursor: crosshair; display: block; border-radius: 6px; border: 1px solid var(--border); }

  .instructions { font-size: 11px; color: var(--muted); line-height: 1.7; }
  .instructions strong { color: var(--text); }

  .frame-info { background: var(--border); border-radius: 8px; padding: 10px 12px; font-size: 11px; }
  .frame-info .label { color: var(--muted); margin-bottom: 2px; }
  .frame-info .val   { font-size: 16px; font-weight: 700; color: var(--accent); }

  .done-banner { background: rgba(61,220,132,.1); border: 1px solid var(--success); border-radius: 8px; padding: 12px; text-align: center; font-size: 13px; color: var(--success); display: none; }
</style>
</head>
<body>

<header>
  <h1>Ground Truth Annotator</h1>
  <div class="progress-pill">Frame <span id="frameNum">—</span> &nbsp;|&nbsp; <span id="progressText">0/0</span> annotated</div>
</header>

<div class="layout">
  <div class="sidebar">
    <div>
      <h2>Current Frame</h2>
      <div class="frame-info">
        <div class="label">Frame number</div>
        <div class="val" id="frameNumLarge">—</div>
      </div>
    </div>

    <div>
      <h2>Keypoints — click in order</h2>
      <div class="kp-list" id="kpList"></div>
    </div>

    <div>
      <h2>Instructions</h2>
      <div class="instructions">
        <strong>Click</strong> on the image to place the highlighted keypoint.<br>
        <strong>Skip</strong> marks it as missing (occluded / out of frame).<br>
        <strong>Undo</strong> removes the last placed point.<br>
        When all 8 are placed or skipped, click <strong>Next</strong>.<br><br>
        Coordinates are saved in original video resolution.
      </div>
    </div>

    <div class="done-banner" id="doneBanner">
      ✅ All frames annotated!<br>Click Save &amp; Quit.
    </div>

    <div class="btn-row">
      <button id="btnUndo"  onclick="undoLast()">⟵ Undo last</button>
      <button id="btnSkip"  onclick="skipKp()" class="danger">Skip keypoint</button>
      <button id="btnNext"  onclick="nextFrame()" class="primary" disabled>Next frame →</button>
      <button id="btnSave"  onclick="saveQuit()" class="success">💾 Save &amp; Quit</button>
    </div>
  </div>

  <div class="canvas-wrap">
    <canvas id="annotCanvas"></canvas>
  </div>
</div>

<script>
const KP_NAMES = {{ kp_names|tojson }};
const KP_COLORS = {{ kp_colors|tojson }};
const FRAMES   = {{ frames|tojson }};
const DISPLAY_W = {{ display_w }};

let frameIdx  = {{ frame_idx }};
let curKpIdx  = 0;
let placements = [];   // [{name, x, y}] or [{name, skipped: true}]
let imgNaturalW = DISPLAY_W, imgNaturalH = 400;
let imgData = null;

const canvas = document.getElementById('annotCanvas');
const ctx    = canvas.getContext('2d');

function renderSidebar() {
  const list = document.getElementById('kpList');
  list.innerHTML = '';
  placements.forEach((p, i) => {
    const div = document.createElement('div');
    const done = true;
    const skipped = p.skipped;
    div.className = 'kp-item ' + (skipped ? 'skipped' : 'done');
    div.innerHTML = `
      <div class="kp-dot" style="background:${KP_COLORS[i]}"></div>
      <span class="kp-name">${KP_NAMES[i]}</span>
      <span class="kp-badge ${skipped ? 'skipped' : 'done'}">${skipped ? 'skip' : Math.round(p.x)+','+Math.round(p.y)}</span>`;
    list.appendChild(div);
  });

  // Current keypoint
  if (curKpIdx < KP_NAMES.length) {
    const div = document.createElement('div');
    div.className = 'kp-item active';
    div.innerHTML = `
      <div class="kp-dot" style="background:${KP_COLORS[curKpIdx]}; box-shadow:0 0 6px ${KP_COLORS[curKpIdx]}"></div>
      <span class="kp-name">${KP_NAMES[curKpIdx]}</span>
      <span class="kp-badge active">← next</span>`;
    list.appendChild(div);
  }

  // Remaining
  for (let i = curKpIdx + 1; i < KP_NAMES.length; i++) {
    const div = document.createElement('div');
    div.className = 'kp-item';
    div.innerHTML = `
      <div class="kp-dot" style="background:${KP_COLORS[i]}; opacity:0.3"></div>
      <span class="kp-name" style="opacity:0.4">${KP_NAMES[i]}</span>
      <span class="kp-badge pending">—</span>`;
    list.appendChild(div);
  }

  const allDone = curKpIdx >= KP_NAMES.length;
  document.getElementById('btnNext').disabled = !allDone;
  document.getElementById('btnSkip').disabled = allDone;
  document.getElementById('frameNum').textContent = frameIdx < FRAMES.length ? FRAMES[frameIdx] : '—';
  document.getElementById('frameNumLarge').textContent = frameIdx < FRAMES.length ? FRAMES[frameIdx] : '—';

  const annotatedSoFar = frameIdx;  // frames already submitted
  document.getElementById('progressText').textContent = `${annotatedSoFar}/${FRAMES.length}`;
  document.getElementById('doneBanner').style.display = frameIdx >= FRAMES.length ? 'block' : 'none';
}

function drawCanvas() {
  if (!imgData) return;
  const img = new Image();
  img.onload = () => {
    canvas.width  = img.width;
    canvas.height = img.height;
    imgNaturalW = img.width;
    imgNaturalH = img.height;
    ctx.drawImage(img, 0, 0);

    // Draw grid crosshair guide
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 0.5;
    for (let x = 0; x < img.width; x += 80) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,img.height); ctx.stroke(); }
    for (let y = 0; y < img.height; y += 80) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(img.width,y); ctx.stroke(); }

    // Draw placed points
    placements.forEach((p, i) => {
      if (p.skipped) return;
      const r = 8;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r + 3, 0, 2*Math.PI);
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, 2*Math.PI);
      ctx.fillStyle = KP_COLORS[i];
      ctx.fill();
      // Cross lines
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(p.x - 14, p.y); ctx.lineTo(p.x + 14, p.y); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(p.x, p.y - 14); ctx.lineTo(p.x, p.y + 14); ctx.stroke();
      // Label
      ctx.fillStyle = KP_COLORS[i];
      ctx.font = 'bold 11px monospace';
      ctx.fillText(KP_NAMES[i], p.x + 12, p.y - 6);
    });

    // Highlight next keypoint name in top-left
    if (curKpIdx < KP_NAMES.length) {
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(8, 8, 260, 30);
      ctx.fillStyle = KP_COLORS[curKpIdx];
      ctx.font = 'bold 13px monospace';
      ctx.fillText('→ Click: ' + KP_NAMES[curKpIdx], 16, 28);
    }
  };
  img.src = 'data:image/jpeg;base64,' + imgData;
}

function loadFrame() {
  if (frameIdx >= FRAMES.length) {
    document.getElementById('doneBanner').style.display = 'block';
    document.getElementById('btnNext').disabled = true;
    renderSidebar();
    return;
  }
  placements = [];
  curKpIdx   = 0;
  fetch(`/frame/${FRAMES[frameIdx]}`).then(r => r.json()).then(d => {
    imgData = d.b64;
    drawCanvas();
    renderSidebar();
  });
}

canvas.addEventListener('click', e => {
  if (curKpIdx >= KP_NAMES.length) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (e.clientX - rect.left)  * scaleX;
  const y = (e.clientY - rect.top)   * scaleY;
  placements.push({ name: KP_NAMES[curKpIdx], x, y });
  curKpIdx++;
  drawCanvas();
  renderSidebar();
});

function undoLast() {
  if (placements.length === 0) return;
  placements.pop();
  curKpIdx = placements.length;
  drawCanvas();
  renderSidebar();
}

function skipKp() {
  if (curKpIdx >= KP_NAMES.length) return;
  placements.push({ name: KP_NAMES[curKpIdx], skipped: true });
  curKpIdx++;
  drawCanvas();
  renderSidebar();
}

function nextFrame() {
  const frameNum = FRAMES[frameIdx];
  const data = {};
  placements.forEach((p, i) => {
    data[KP_NAMES[i]] = p.skipped ? null : [p.x, p.y];
  });
  fetch('/annotate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ frame: frameNum, keypoints: data })
  }).then(r => r.json()).then(d => {
    frameIdx++;
    loadFrame();
  });
}

function saveQuit() {
  fetch('/save', { method: 'POST' }).then(r => r.json()).then(d => {
    alert('✅ Saved to ' + d.path + '\\nYou can close this tab.');
  });
}

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.key === 'z' && (e.ctrlKey || e.metaKey)) { undoLast(); return; }
  if (e.key === 's' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); skipKp(); return; }
  if (e.key === 'Enter') { if (!document.getElementById('btnNext').disabled) nextFrame(); return; }
});

// Init
loadFrame();
renderSidebar();
</script>
</body>
</html>
"""


# ── Flask routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    total, w, h = load_video_meta()
    frames = list(range(FRAME_STEP, total + 1, FRAME_STEP))
    state["frames"] = frames

    load_existing_csv()

    return render_template_string(
        HTML,
        kp_names   = KEYPOINTS_8,
        kp_colors  = KP_COLORS,
        frames     = frames,
        display_w  = DISPLAY_W,
        frame_idx  = state["frame_idx"],
    )


@app.route("/frame/<int:frame_num>")
def get_frame(frame_num):
    result = frame_to_jpeg_b64(frame_num)
    if result is None:
        return jsonify({"error": "frame not found"}), 404
    b64, disp_h = result
    return jsonify({"b64": b64, "h": disp_h})


@app.route("/annotate", methods=["POST"])
def annotate():
    data  = request.get_json()
    frame = int(data["frame"])
    kps   = data["keypoints"]     # {kp_name: [x,y] | null}
    state["annotations"][frame] = {k: v for k, v in kps.items()}
    save_csv()   # auto-save after every frame
    return jsonify({"ok": True, "saved": OUTPUT_CSV})


@app.route("/save", methods=["POST"])
def save():
    save_csv()
    return jsonify({"ok": True, "path": OUTPUT_CSV})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    print("=" * 60)
    print("  Ground Truth Annotator")
    print("=" * 60)
    print(f"  Video  : {VIDEO_PATH}")
    print(f"  Output : {OUTPUT_CSV}")
    print(f"  Frames : every {FRAME_STEP}th frame")
    print()
    print("  Open your browser at:  http://localhost:5000")
    print()
    print("  Keyboard shortcuts:")
    print("    Click        — place keypoint")
    print("    Ctrl+Z       — undo last")
    print("    Ctrl+S       — skip keypoint")
    print("    Enter        — next frame (when all 8 placed/skipped)")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
