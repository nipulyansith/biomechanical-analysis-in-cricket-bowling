# Cricket Bowling Annotation & Evaluation Tool
## RQ3 – Manual Ground Truth Dataset

---

## 📁 Project Structure

```
cricket_annotation/
├── annotator.py           ← Main annotation tool (OpenCV)
├── evaluate.py            ← Evaluation module (MAE / RMSE / plot)
├── sample_ground_truth.csv← Example output structure
├── master.xlsx            ← Model outputs (place here)
├── videos/                ← Place your .mp4 / .avi / .mov files here
│   ├── B-01_T-01.mp4
│   ├── B-01_T-02.mp4
│   └── ...
└── README.md
```

> **trial_id** is derived from the video filename stem (e.g. `B-01_T-01.mp4` → `trial_id = B-01_T-01`).

---

## ⚙️ Requirements

```bash
pip install opencv-python numpy pandas openpyxl matplotlib
```

Python ≥ 3.8 required.

---

## 🚀 Running the Annotation Tool

```bash
python annotator.py
```

The tool loops through all videos in `videos/`. For each:

1. You will be prompted for **bowling arm** (`R`/`L`) and **view mode** (`SIDE`/`FRONT`).
2. The OpenCV window opens.

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `←` / `→` | Move one frame |
| `Shift + ←` / `Shift + →` | Jump ±10 frames |
| `C` | Enter calibration mode |
| `1` | Mark **BFC** (Back Foot Contact) frame |
| `2` | Mark **FFC** (Front Foot Contact) frame |
| `3` | Mark **Arm Back** frame |
| `4` | Mark **Release** frame |
| `F` | Mark a foot contact (for last-5 steps) |
| `S` | Save trial and proceed |
| `Q` | Quit |

---

## 📐 Calibration (FIRST STEP every trial)

1. Navigate to a frame where the **stumps are clearly visible**.
2. Press `C` → tool enters calibration mode.
3. Click the **top of the stumps**, then the **bottom**.
4. Tool computes px/m (stump height = 0.711 m).

> Without calibration, metric distances (stride length, head offset, wrist speed) will be `None`.

---

## 🎯 Annotation Workflow per Trial

### Step 1 – Calibrate (press C, click stump top → bottom)

### Step 2 – Mark foot contacts (press F at each contact frame)
- Mark **at least 5** foot contacts before release.
- BFC and FFC must be the last two before release.
- The tool automatically selects the **last 5 contacts before release**.

### Step 3 – Mark events and annotate keypoints

For each event key (1/2/3/4), after pressing the key, **click the keypoints in order** as shown in the status bar:

| Event | Keypoints to click (in order) |
|-------|-------------------------------|
| **BFC** (1) | L-hip, R-hip, L-knee, R-knee, L-ankle, R-ankle |
| **FFC** (2) | L-hip, R-hip, L-knee, R-knee, L-ankle, R-ankle, nose, front_toe, back_toe |
| **Arm Back** (3) | shoulder, elbow, wrist |
| **Release** (4) | shoulder, elbow, wrist, L-knee, R-knee |
| **Release +1** | Navigate one frame forward → click wrist only |

> The status bar always shows which keypoint to click next.

### Step 4 – Save (press S)

---

## 📊 Computed Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| `stride_duration_s` | (release – BFC) / FPS | s |
| `stride_length_m` | front_toe to back_toe at FFC | m |
| `step_duration_mean_s` | Mean of last-5 step intervals | s |
| `step_duration_std_s` | Std of step intervals | s |
| `step_duration_cv` | CV of step intervals | — |
| `final5_total_duration_s` | Sum of last-5 step intervals | s |
| `elbow_angle_arm_back_deg` | Elbow angle at Arm Back | deg |
| `elbow_angle_release_deg` | Elbow angle at Release | deg |
| `elbow_extension_deg` | Release – Arm Back angle | deg |
| `knee_angle_ffc_deg` | Mean knee angle at FFC | deg |
| `knee_angle_release_deg` | Mean knee angle at Release | deg |
| `head_dx/dy/d_ffc_cm` | Nose vs front ankle at FFC | cm |
| `head_dx/dy/d_bfc_cm` | Nose vs BFC ankle | cm |
| `wrist_speed_at_release_m_s` | Wrist velocity: release → release+1 | m/s |

---

## 📈 Running the Evaluation

```bash
python evaluate.py --gt ground_truth.csv --model master.xlsx
```

Optional flags:
```
--gt     Path to ground truth CSV  (default: ground_truth.csv)
--model  Path to model XLSX        (default: master.xlsx)
--plot   Output plot filename      (default: evaluation_plot.png)
```

Outputs:
- **Console table** — MAE and RMSE per parameter
- `evaluation_metrics.csv` — machine-readable results
- `evaluation_plot.png` — scatter plots (manual vs model) for key parameters

---

## 📄 Output CSV Structure (`ground_truth.csv`)

Columns match `master.xlsx` exactly:

```
trial_id, fps, bowling_arm, view_mode, release_frame, release_method,
last5_steps_frame, last5_step_intervals_s,
step_duration_mean_s, step_duration_std_s, step_duration_cv, final5_total_duration_s,
stride_duration_s, stride_length_m,
bfc_frame, ffc_frame, arm_back_frame, release_frame.1,
elbow_angle_arm_back_deg, elbow_angle_release_deg, elbow_extension_deg,
knee_angle_ffc_deg, knee_angle_release_deg,
head_dx_ffc_cm, head_dy_ffc_cm, head_d_ffc_cm,
head_dx_bfc_cm, head_dy_bfc_cm, head_d_bfc_cm,
peak_wrist_speed_m_s, wrist_speed_at_release_m_s,
step1_frame, step2_frame, step3_frame, step4_frame, step5_frame
```

---

## 💡 Tips

- Annotate on a frame where the full body is visible and not motion-blurred.
- For knee angles at Release, the tool uses hip/ankle from BFC and knee clicks from Release.
- If you mis-click a keypoint, re-press the event key (1/2/3/4) to restart that phase.
- Trials already saved can be re-annotated; you'll be prompted to confirm.
