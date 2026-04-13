# 🏏 FastBowl Lab - Cricket Bowling Biomechanics Analyzer

Comprehensive real-time biomechanics analysis system for cricket fast bowling using AI-powered pose detection and biomechanical measurements.

## Project Overview

FastBowl Lab integrates a **YOLOv8 pose detection backend** with an **interactive web frontend** to analyze 6 key biomechanical metrics in cricket bowling. The system processes video of bowling deliveries and extracts detailed kinematic data about the bowler's motion.

## 🎯 Six Biomechanical Metrics Analyzed

### 1. **Step Duration in Final Run-Up**
- Time interval between consecutive foot contacts during the approach
- Measures rhythm consistency of the run-up
- Data: Intervals for 5 final steps before delivery

### 2. **Delivery Stride Duration & Length**
- **Duration**: Time from back foot contact (BFC) to front foot contact (FFC)
- **Length**: Horizontal distance covered during delivery stride
- Measures stride effectiveness and biomechanical efficiency

### 3. **Elbow Flexion-Extension**
- Angular change at the elbow joint from "arm-back" position to ball release
- Uses shoulder, elbow, and wrist coordinates
- Indicates arm acceleration pattern and extension mechanics

### 4. **Front Knee Flexion-Extension**
- Angular movement of the front knee from FFC to ball release
- Uses hip, knee, and ankle coordinates
- Measures drive phase and load management

### 5. **Center of Mass (COM) of Head**
- Positional offset between head's center and front foot position at FFC
- Measures horizontal (Dx), vertical (Dy), and total (D) distances
- Indicates balance and head stabilization during delivery

### 6. **Wrist Joint Velocity & Ball Release Speed**
- Displacement of wrist joint between frames over time
- Calculates speed at release and peak speeds near release
- Measures bowling velocity and release characteristics

---

## 🚀 Quick Start (2 minutes)

### Prerequisites
- Python 3.8+
- Virtual environment with dependencies installed
- Modern web browser

### Starting the Server

**Option 1: Using Virtual Environment Python (Recommended)**
```bash
cd /path/to/biomechanical-analysis-in-cricket-bowling
./venv/bin/python app.py
```

**Option 2: Activate Virtual Environment First**
```bash
cd /path/to/biomechanical-analysis-in-cricket-bowling
source venv/bin/activate  # or source venv/bin/activate on macOS/Linux
python app.py
```

**Expected Output:**
```
============================================================
  FastBowl Lab - Backend Server
============================================================
Server starting on http://localhost:5001
 * Running on http://127.0.0.1:5001
Press CTRL+C to quit
```

### Access the Application
- **Main Dashboard**: http://localhost:5001/dashboard.html
- **API Health Check**: http://localhost:5001/api/health

⚠️ **Important**: Always use `./venv/bin/python app.py` or activate the virtual environment first. Using bare `python3` will cause `ModuleNotFoundError`.

---

## 📋 Complete User Workflow

### 1. Dashboard (Landing Page)
- **URL**: `/dashboard.html`
- **Content**: Welcome screen, features overview, session history
- **Action**: Click "New Analysis" to start

### 2. Upload Page
- **URL**: `/upload.html`
- **Features**:
  - Select bowling arm (Right/Left)
  - Upload video via drag-drop or file picker
  - Record video from webcam
  - Video preview before proceeding
- **Supported Formats**: MP4, MOV, AVI, MKV, WebM, FLV, WMV
- **Next**: Calibration Guide

### 3. Calibration Guide (Video Instruction)
- **URL**: `/calibration-video.html`
- **Purpose**: Educational content on proper calibration
- **Shows**: Reference points and measurement technique

### 4. Calibration Marking
- **URL**: `/calibration.html`
- **Instructions**:
  - Display video first frame
  - User marks stump top point (click on video)
  - User marks stump bottom point (click on video)
  - System calculates conversion scale (pixels to meters)
- **Calibration Formula**: Stump height (0.711m) ÷ pixel distance = conversion factor
- **Next**: Processing page

### 5. Processing Page
- **URL**: `/processing.html`
- **Features**:
  - Real-time progress updates (0-100%)
  - Status messages for each analysis module
  - Animated loading indicator
- **Backend**: Asynchronous analysis via subprocess
- **Duration**: 1-3 minutes depending on video length
- **Auto-redirect**: Redirects to results when complete

### 6. Results Dashboard
- **URL**: `/results.html`
- **Displays**:
  - 6 metric cards with key values
  - Color-coded metrics (primary, secondary, tertiary)
  - Links to detailed analysis pages
  - Download buttons for each metric

### 7-12. Detailed Analysis Pages
Each metric has a dedicated analysis page:

| Metric | URL | Displays |
|--------|-----|----------|
| Step Duration | `/step-duration.html` | 5 consecutive step intervals, cadence chart |
| Delivery Stride | `/delivery-stride-new.html` | Distance (m), duration (s), annotated video |
| Elbow Flexion | `/elbow-flexion-new.html` | Angle at arm-back, at release, extension (°) |
| Knee Flexion | `/knee-flexion-new.html` | Angle at FFC, at release, extension (°) |
| Head COM | `/com-new.html` | Dx (cm), Dy (cm), total distance (cm) |
| Wrist Velocity | `/velocity-new.html` | Speed at release, peak speed (m/s), km/h |

Each page includes:
- Annotated video showing joints and measurements
- Analysis plots/graphs
- Downloadable CSV data
- Back link to results dashboard

---

## 🛠️ Installation & Setup

### Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
pip install -r requirements-backend.txt
```

### Required Packages
- **Flask** & **Flask-CORS**: Web server and cross-origin requests
- **OpenCV** (cv2): Video processing
- **Ultralytics YOLO**: Pose detection model
- **Pandas & NumPy**: Data analysis
- **SciPy**: Signal processing for event detection
- **Matplotlib**: Plotting and visualization

### Download Models
The system automatically downloads YOLO pose models on first run:
- `yolov8n-pose.pt` (40MB - fastest)
- `yolov8m-pose.pt` (97MB - balanced)
- `yolov8l-pose.pt` (189MB - most accurate)

---

## 📁 Project Structure

```
biomechanical-analysis-in-cricket-bowling/
├── app.py                          # Flask server & API endpoints
├── requirements.txt                # Main dependencies
├── requirements-backend.txt        # Backend-specific dependencies
│
├── FE/                            # Frontend (HTML/JavaScript)
│   ├── dashboard.html             # Landing page
│   ├── upload.html                # Video upload & arm selection
│   ├── calibration-video.html     # Calibration guide video
│   ├── calibration.html           # Calibration marker interface
│   ├── processing.html            # Analysis progress page
│   ├── results.html               # Results dashboard
│   ├── step-duration.html         # Step cadence details
│   ├── delivery-stride-new.html   # Stride analysis
│   ├── elbow-flexion-new.html     # Elbow angle analysis
│   ├── knee-flexion-new.html      # Knee angle analysis
│   ├── com-new.html               # Head COM analysis
│   ├── velocity-new.html          # Wrist velocity analysis
│   └── api-client.js              # Shared API client library
│
├── dataset_making/                # Analysis pipeline
│   ├── all.py                     # Main analysis module
│   ├── run_analysis_wrapper.py    # Analysis orchestrator
│   └── output/                    # Analysis outputs
│
├── uploads/                       # Temporary video uploads
├── results/                       # Analysis results (by session ID)
│
└── (Various documentation files consolidated into README.md)
```

---

## 🔌 API Endpoints

### Session Management
```
POST   /api/session/create              Create new analysis session
GET    /api/session/<session_id>        Get session details
DELETE /api/session/<session_id>        Delete session and cleanup
```

### Video Upload & Setup
```
POST   /api/upload/<session_id>         Upload video file
POST   /api/bowling-arm/<session_id>    Set bowling arm (R/L)
GET    /api/calibration/preview/<id>   Get first frame for marking
POST   /api/calibration/<session_id>    Set calibration points
```

### Analysis & Results
```
POST   /api/process/<session_id>        Start video analysis
GET    /api/progress/<session_id>       Get analysis progress
GET    /api/results/<session_id>        Get all analysis results
GET    /api/results/<sid>/stride        Get step duration data
GET    /api/results/<sid>/delivery      Get delivery stride data
GET    /api/results/<sid>/elbow         Get elbow flexion data
GET    /api/results/<sid>/knee          Get knee flexion data
GET    /api/results/<sid>/com           Get head COM data
GET    /api/results/<sid>/wrist         Get wrist velocity data
```

### Data Access
```
GET    /api/results/<sid>/video/<type>  Download annotated video
GET    /api/results/<sid>/image/<name>  Get analysis plot image
GET    /api/results/<sid>/csv/<file>    Download CSV data
GET    /api/results/<sid>/json/<file>   Download JSON data
```

---

## 📊 Output Files

For each analysis session, the system generates:

### CSV Files
- `step_cadence.csv` - Step intervals and timing
- `elbow_full_analysis.csv` - Frame-by-frame elbow angles
- `knee_angles.csv` - Frame-by-frame knee angles
- `knee_angles_ffc_to_release.csv` - Knee angles from FFC to release
- `headDx_vs_time.csv` - Head horizontal offset over time
- `wrist_timeseries.csv` - Wrist velocity over time
- `head_metrics.csv` - Summary statistics

### JSON Files
- `analysis_results.json` - Complete analysis summary (all 6 metrics)
- `head_metrics.json` - Head COM measurements
- `wrist_ball_metrics.json` - Wrist and ball release speeds

### Image Plots
- `elbow_full_plots.png` - Elbow angle and angular velocity graphs
- `knee_flexion_analysis.png` - Knee angle vs time
- `headDx_vs_time.png` - Head offset trajectory
- `wrist_speed_vs_time.png` - Wrist speed vs time
- `wrist_omega_vs_time.png` - Forearm angular velocity

### Annotated Videos
- `step_cadence_annotated.mp4` - Step detection visualized
- `delivery_stride_annotated.mp4` - Stride measurement shown
- `elbow_flexion_annotated.mp4` - Elbow angle tracking
- `knee_flexion_annotated.mp4` - Knee angle tracking
- `head_position_annotated.mp4` - Head position relative to front foot
- `wrist_velocity_annotated.mp4` - Wrist speed visualization

---

## 🔍 Keypoints Extracted

The YOLOv8 pose model extracts 17 body keypoints per frame:

**Upper Body**: Nose, Left/Right Eyes, Left/Right Ears, Left/Right Shoulders
**Arm**: Left/Right Elbows, Left/Right Wrists
**Lower Body**: Left/Right Hips, Left/Right Knees, Left/Right Ankles

These keypoints are tracked throughout the video and used to calculate all biomechanical metrics.

---

## ⚙️ Technical Details

### Event Detection
- **Back Foot Contact (BFC)**: Detected via ankle y-coordinate peak
- **Front Foot Contact (FFC)**: Detected via ankle y-coordinate peak (opposite foot)
- **Arm-Back Position**: Detected by wrist height and arm angle
- **Ball Release**: Detected by sudden velocity change in wrist

### Angle Calculation
- All angles calculated using 3-point vector geometry
- Example: Elbow angle = angle(shoulder→elbow→wrist)
- Angular velocity calculated using central difference method

### Calibration
- User marks stump top and bottom in first frame
- Pixel distance converted to real-world distance (0.711m stump height)
- Conversion factor applied to all stride and distance measurements

---

## 📝 Notes

- **Session Persistence**: Results stored in `/results/<session_id>/` directory
- **Memory Management**: Sessions loaded from disk if restarted
- **JSON Handling**: NaN values sanitized to null for JSON compatibility
- **Video Processing**: Uses OpenCV with frame-rate preservation
- **Error Handling**: Comprehensive error messages in UI and logs

---

## 🐛 Troubleshooting

### Server Won't Start
```bash
# Make sure virtual environment is active
source venv/bin/activate

# Check if port 5001 is already in use
lsof -i :5001

# Kill existing process if needed
kill -9 <PID>
```

### Video Upload Fails
- Ensure video is in supported format (MP4, MOV, AVI, etc.)
- Check file size is under 500MB
- Verify video codec is H.264/H.265

### Analysis Takes Too Long
- Analysis time depends on video length (roughly 30 seconds per minute of video)
- Longer videos (>5 min) may take several minutes
- Check browser console for error messages

---

## 📄 Original Documentation

This project integrates YOLOv8 Pose Keypoint Extraction with cricket biomechanics analysis:

- Extract pose keypoints from video using Ultralytics YOLOv8 pose model
- Keypoints exported: wrist, shoulder, elbow, knee, hip, ankle (left/right)
- CSV output with frame-by-frame pose data
- Used as foundation for biomechanical metric calculations

