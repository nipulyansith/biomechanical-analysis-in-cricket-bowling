import cv2

video_path = "output/yolo_annotated_trimmed.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Could not open video.")
else:
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frames / fps
    print(f"🎞️ Video: {frames} frames at {fps:.3f} FPS")
    print(f"⏱️ Duration: {duration:.3f} seconds")

cap.release()
