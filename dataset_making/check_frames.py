"""
Simple Frame-by-Frame Video Viewer
====================================
Controls:
  RIGHT arrow / D  → next frame
  LEFT arrow  / A  → previous frame
  SPACE            → play / pause
  F                → jump forward 10 frames
  B                → jump back 10 frames
  G                → go to a specific frame number (type in terminal)
  Q / ESC          → quit
"""

import cv2
import sys

# ── CONFIG ────────────────────────────────────────────────────────────────────
VIDEO_PATH = r"C:\Users\nipul\OneDrive\Desktop\tm\videos\B-05_T-01.MOV"
WINDOW_W   = 1280   # display width in pixels (height scales automatically)
# ─────────────────────────────────────────────────────────────────────────────


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌  Cannot open video: {VIDEO_PATH}")
        sys.exit(1)

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Compute display height keeping aspect ratio
    disp_w = WINDOW_W
    disp_h = int(orig_h * disp_w / orig_w)

    print(f"✅  Opened: {VIDEO_PATH}")
    print(f"   {total} frames  |  {fps:.2f} fps  |  {orig_w}×{orig_h}")
    print("\nControls:")
    print("  ← / A        previous frame")
    print("  → / D        next frame")
    print("  SPACE        play / pause")
    print("  F            +10 frames")
    print("  B            -10 frames")
    print("  G            go to frame (enter number in terminal)")
    print("  Q / ESC      quit\n")

    idx   = 0        # current 0-based frame index
    playing = False  # start paused

    win = "Frame Viewer  [SPACE=play/pause  ←/→=step  Q=quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w, disp_h)

    def read_frame(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frm = cap.read()
        return frm if ok else None

    while True:
        frame = read_frame(idx)
        if frame is None:
            idx = max(0, idx - 1)
            playing = False
            frame = read_frame(idx)

        # ── Resize for display ────────────────────────────────────────────
        disp = cv2.resize(frame, (disp_w, disp_h))

        # ── Overlay: frame number & timestamp ────────────────────────────
        ts   = idx / fps
        label = f"Frame {idx + 1} / {total}   |   {ts:.3f} s   |   {'▶ PLAYING' if playing else '⏸ PAUSED'}"
        cv2.rectangle(disp, (0, 0), (disp_w, 36), (0, 0, 0), -1)
        cv2.putText(disp, label, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(win, disp)

        # ── Key handling ─────────────────────────────────────────────────
        wait_ms = max(1, int(1000 / fps)) if playing else 0
        key = cv2.waitKey(wait_ms) & 0xFF

        if key in (ord('q'), 27):           # Q or ESC → quit
            break

        elif key == ord(' '):               # SPACE → toggle play/pause
            playing = not playing

        elif key in (83, ord('d')):         # → or D → next frame
            playing = False
            idx = min(total - 1, idx + 1)

        elif key in (81, ord('a')):         # ← or A → previous frame
            playing = False
            idx = max(0, idx - 1)

        elif key in (ord('f'), ord('F')):   # F → +10 frames
            playing = False
            idx = min(total - 1, idx + 10)

        elif key in (ord('b'), ord('B')):   # B → -10 frames
            playing = False
            idx = max(0, idx - 10)

        elif key in (ord('g'), ord('G')):   # G → jump to frame
            playing = False
            try:
                target = int(input(f"  Go to frame (1–{total}): ").strip())
                idx = max(0, min(total - 1, target - 1))  # convert to 0-based
            except ValueError:
                print("  ⚠️  Invalid number — staying on current frame.")

        elif playing:                        # auto-advance during playback
            idx = min(total - 1, idx + 1)
            if idx == total - 1:
                playing = False             # stop at end

    cap.release()
    cv2.destroyAllWindows()
    print("👋  Viewer closed.")


if __name__ == "__main__":
    main()