"""Record the two files the remote tests need, into C:\\liveswap-test:

    clip.mp4   30 seconds from the webcam (the face that gets replaced)
    face.jpg   the source face: a photo you pick, or a webcam snapshot

Run it at the PC itself (it needs the webcam and a desktop). Only record and
pick photos of people who have agreed to be used for this test.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def _show(frame: np.ndarray, text: str) -> bool:
    """Show a preview with a caption. Returns False if the user pressed Esc."""
    view = frame.copy()
    cv2.rectangle(view, (0, 0), (view.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(view, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("liveswap test recording (Esc cancels)", view)
    return cv2.waitKey(1) != 27


def _pick_photo() -> str:
    """File dialog for the source photo; empty string if cancelled or unavailable."""
    try:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Source face photo (Cancel = use a webcam snapshot instead)",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        root.destroy()
        return path or ""
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=r"C:\liveswap-test", help="output folder (ASCII path)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--no-photo-dialog", action="store_true",
                        help="always use a webcam snapshot as face.jpg")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = _open_camera(args.camera)
    ok, frame = cap.read()
    if not ok:
        print(f"Cannot read from camera {args.camera}")
        return 1
    height, width = frame.shape[:2]

    # Countdown, so the person is in frame and ready before recording starts.
    start = time.perf_counter()
    while time.perf_counter() - start < 5:
        ok, frame = cap.read()
        if not ok or not _show(frame, f"Look at the camera. Recording starts in "
                                      f"{5 - int(time.perf_counter() - start)}"):
            cap.release()
            cv2.destroyAllWindows()
            print("Cancelled")
            return 1
    snapshot = frame.copy()

    clip_path = out / "clip.mp4"
    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    if not writer.isOpened():
        print(f"Cannot write {clip_path}")
        return 1
    frames = 0
    start = time.perf_counter()
    while time.perf_counter() - start < args.seconds:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frames += 1
        left = int(args.seconds - (time.perf_counter() - start))
        if not _show(frame, f"REC {left} s - turn your head slowly, then talk"):
            break
    writer.release()
    cap.release()
    cv2.destroyAllWindows()

    photo_path = "" if args.no_photo_dialog else _pick_photo()
    face_path = out / "face.jpg"
    if photo_path:
        # fromfile/imdecode and imencode/tofile handle non-ASCII paths on Windows.
        photo = cv2.imdecode(np.fromfile(photo_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if photo is None:
            print(f"Cannot read {photo_path}; using a webcam snapshot instead")
            photo = snapshot
    else:
        photo = snapshot
    cv2.imencode(".jpg", photo)[1].tofile(str(face_path))

    print(f"Saved {clip_path} ({frames} frames) and {face_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
