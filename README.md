YOLOv8 Pose Keypoint Extractor

This small project extracts selected pose keypoints from a video using the Ultralytics YOLOv8 pose model and writes frame-by-frame CSV output.

Keypoints exported: wrist, shoulder, elbow, knee, hip, toe (left and right variants when available)

Files
- `extract_keypoints.py`: Main script. Use `--source` and `--output` to provide input video and CSV path.
- `requirements.txt`: Python dependencies.

Quick start
1. Create a virtual environment and install requirements:

   python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt

2. Run the script:

   python extract_keypoints.py --source input.mp4 --output out.csv --model yolov8n-pose.pt

Notes
- The `ultralytics` package will download model weights (e.g. `yolov8n-pose.pt`) automatically if passed as a name and not present locally.
- Device selection: pass `--device 0` to use the first CUDA GPU if available, or `--device cpu` to force CPU.
- The CSV columns: frame, person_id, then for each side: e.g. left_wrist_x,left_wrist_y,left_wrist_conf,...
