"""Guided recording of the two files the remote tests need, into C:\\liveswap-test:

    clip.mp4   about 30 seconds from the webcam (the face that gets replaced)
    face.jpg   the source face: a photo you pick, or a webcam snapshot

The screen works like enrolling a face on a phone: the face sits in a circle,
and a ring of marks around it turns green in whichever direction the head has
been turned, until the ring is complete. A few short timed prompts follow
(smile, open your mouth, count aloud). Only the preview carries the overlay;
the saved clip is the clean camera image.

Run it at the PC itself (it needs the webcam and a desktop). Only record and
pick photos of people who have agreed to be used for this test.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WINDOW = "liveswap"
TICKS = 60
GREEN = (89, 199, 52)        # BGR of #34C759
IDLE = (72, 72, 74)          # BGR of #4A4848, an unlit mark
TIMED_STEPS = [
    ("Улыбнитесь", 3.0),
    ("Широко откройте рот", 3.0),
    ("Скажите вслух: раз, два, три, четыре, пять", 5.0),
]


def _font(size: int, bold: bool = False):
    names = ("seguisb.ttf", "segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Screen:
    """Black canvas, the mirrored camera in a circle, a ring of marks, two lines of text."""

    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.cx, self.cy, self.r = width // 2, int(height * 0.42), int(height * 0.29)
        self._mask = np.zeros((2 * self.r, 2 * self.r), np.uint8)
        cv2.circle(self._mask, (self.r, self.r), self.r - 2, 255, -1, cv2.LINE_AA)
        self._title_font, self._hint_font = _font(38, bold=True), _font(24)

    def draw(self, frame: np.ndarray, title: str, hint: str = "",
             ticks: np.ndarray | None = None, check: bool = False) -> np.ndarray:
        canvas = np.zeros((self.h, self.w, 3), np.uint8)
        # Mirror like a phone's front camera, then show the centre square in the circle.
        mirrored = cv2.flip(frame, 1)
        side = min(mirrored.shape[:2])
        y0, x0 = (mirrored.shape[0] - side) // 2, (mirrored.shape[1] - side) // 2
        face = cv2.resize(mirrored[y0:y0 + side, x0:x0 + side], (2 * self.r, 2 * self.r),
                          interpolation=cv2.INTER_AREA)
        region = canvas[self.cy - self.r:self.cy + self.r, self.cx - self.r:self.cx + self.r]
        alpha = self._mask[..., None].astype(np.float32) / 255.0
        region[:] = (face * alpha).astype(np.uint8)

        if ticks is not None:
            for index in range(TICKS):
                angle = 2 * math.pi * index / TICKS
                lit = bool(ticks[index])
                inner, outer = self.r + 18, self.r + (50 if lit else 38)
                p0 = (int(self.cx + inner * math.cos(angle)), int(self.cy + inner * math.sin(angle)))
                p1 = (int(self.cx + outer * math.cos(angle)), int(self.cy + outer * math.sin(angle)))
                cv2.line(canvas, p0, p1, GREEN if lit else IDLE, 4, cv2.LINE_AA)
        if check:
            cv2.circle(canvas, (self.cx, self.cy), self.r, GREEN, 6, cv2.LINE_AA)
            s = self.r // 3
            points = np.array([[self.cx - s, self.cy], [self.cx - s // 4, self.cy + s * 3 // 4],
                               [self.cx + s, self.cy - s * 3 // 4]], np.int32)
            cv2.polylines(canvas, [points], False, GREEN, 14, cv2.LINE_AA)

        # Text goes through Pillow: OpenCV's built-in fonts have no Cyrillic.
        image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        pen = ImageDraw.Draw(image)
        top = self.cy + self.r + 72
        for text, font, y, color in ((title, self._title_font, top, (255, 255, 255)),
                                     (hint, self._hint_font, top + 56, (152, 152, 157))):
            if text:
                box = pen.textbbox((0, 0), text, font=font)
                pen.text(((self.w - (box[2] - box[0])) // 2, y), text, font=font, fill=color)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def head_offset(kps: np.ndarray) -> np.ndarray:
    """Where the nose points relative to the face centre, in eye-distance units.

    kps: the detector's five points (eyes, nose, mouth corners) in image pixels.
    """
    kps = np.asarray(kps, dtype=np.float64)
    centre = (kps[0] + kps[1] + kps[3] + kps[4]) / 4.0
    return (kps[2] - centre) / (np.linalg.norm(kps[1] - kps[0]) + 1e-6)


def mark_direction(ticks: np.ndarray, offset: np.ndarray, neutral: np.ndarray) -> None:
    """Light the marks on the side of the ring the head is turned towards."""
    # The preview is mirrored, so image-right is screen-left. Pitch moves the
    # nose less than yaw does, hence the smaller vertical threshold.
    dx = -(offset[0] - neutral[0]) / 0.16
    dy = (offset[1] - neutral[1]) / 0.10
    if math.hypot(dx, dy) < 1.0:
        return
    centre = round(math.atan2(dy, dx) / (2 * math.pi) * TICKS)
    for step in range(-3, 4):
        ticks[(centre + step) % TICKS] = True


def _open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def _show(view: np.ndarray) -> bool:
    """Show a preview frame. Returns False if the user pressed Esc or closed the window."""
    cv2.imshow(WINDOW, view)
    if cv2.waitKey(1) == 27:
        return False
    return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1


def _load_detector(enabled: bool):
    """The app's face detector, or None when it is unavailable (prompts are then timed)."""
    if not enabled:
        return None
    try:
        from faceswap.core.detector import detect_tracked
        detect_tracked(np.zeros((720, 1280, 3), np.uint8))     # loads the models
        return detect_tracked
    except Exception as exc:
        print(f"Face tracking unavailable ({exc}); using timed prompts")
        return None


def _pick_photo() -> str:
    """File dialog for the source photo; empty string if cancelled or unavailable."""
    try:
        import tkinter
        from tkinter import filedialog, messagebox
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showinfo(
            "liveswap",
            "Запись готова.\n\nТеперь выберите фотографию лица, на которое нужно "
            "заменить ваше.\nЧеловек на фото должен быть согласен на это.\n\n"
            "Если фотографии нет - нажмите «Отмена» в следующем окне.")
        path = filedialog.askopenfilename(
            title="Выберите фотографию лица",
            filetypes=[("Фотографии", "*.jpg *.jpeg *.png *.bmp *.webp")])
        root.destroy()
        return path or ""
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=r"C:\liveswap-test", help="output folder (ASCII path)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-photo-dialog", action="store_true",
                        help="always use a webcam snapshot as face.jpg")
    parser.add_argument("--no-face-tracking", action="store_true",
                        help="timed prompts only; do not load the face detector")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = _open_camera(args.camera)
    ok, frame = cap.read()
    if not ok:
        print(f"Cannot read from camera {args.camera}")
        return 1
    height, width = frame.shape[:2]
    screen = Screen(width, height)

    def cancelled() -> int:
        cap.release()
        cv2.destroyAllWindows()
        print("Cancelled")
        return 1

    if not _show(screen.draw(frame, "Подготовка…")):
        return cancelled()
    detect = _load_detector(not args.no_face_tracking)

    # --- 1. Position the face. Nothing is recorded yet. ---
    neutral = np.zeros(2)
    bbox = None
    steady_since = None
    started = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            return cancelled()
        now = time.perf_counter()
        hint = "Смотрите прямо в камеру"
        if detect is None:
            if now - started >= 5:
                break
        else:
            face = detect(frame, bbox)
            bbox = None if face is None else face.bbox
            centred = False
            if face is not None:
                cx, cy = (face.bbox[0] + face.bbox[2]) / 2, (face.bbox[1] + face.bbox[3]) / 2
                centred = abs(cx - width / 2) < height * 0.18 and abs(cy - height * 0.45) < height * 0.2
                if centred:
                    neutral = head_offset(face.kps)
            if not centred:
                steady_since = None
                hint = "Лицо не видно" if face is None else "Подвиньтесь, чтобы лицо было в центре"
            elif steady_since is None:
                steady_since = now
            if (steady_since is not None and now - steady_since >= 1.2) or now - started >= 25:
                break
        if not _show(screen.draw(frame, "Поместите лицо в круг", hint)):
            return cancelled()
    snapshot = frame.copy()

    clip_path = out / "clip.mp4"
    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    if not writer.isOpened():
        print(f"Cannot write {clip_path}")
        return 1
    frames = 0

    # --- 2. Turn the head around: the ring fills where the head has pointed. ---
    ticks = np.zeros(TICKS, bool)
    started = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frames += 1
        elapsed = time.perf_counter() - started
        if detect is None:
            ticks[:int(TICKS * min(1.0, elapsed / 16.0))] = True
        else:
            face = detect(frame, bbox)
            bbox = None if face is None else face.bbox
            if face is not None:
                mark_direction(ticks, head_offset(face.kps), neutral)
        if (ticks.mean() >= 0.9 and elapsed >= 6) or elapsed >= 28:
            break
        if not _show(screen.draw(frame, "Медленно двигайте головой по кругу",
                                 "Пока кольцо не станет зелёным", ticks=ticks)):
            writer.release()
            return cancelled()
    ticks[:] = True

    # --- 3. Short timed prompts for the mouth. ---
    for prompt, seconds in TIMED_STEPS:
        started = time.perf_counter()
        while time.perf_counter() - started < seconds:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            frames += 1
            if not _show(screen.draw(frame, prompt, ticks=ticks)):
                writer.release()
                return cancelled()
    writer.release()

    started = time.perf_counter()
    while time.perf_counter() - started < 1.5:
        ok, frame = cap.read()
        if not ok or not _show(screen.draw(frame, "Готово", check=True)):
            break
    cap.release()
    cv2.destroyAllWindows()

    photo_path = "" if args.no_photo_dialog else _pick_photo()
    face_path = out / "face.jpg"
    photo = None
    if photo_path:
        # fromfile/imdecode and imencode/tofile handle non-ASCII paths on Windows.
        photo = cv2.imdecode(np.fromfile(photo_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if photo is None:
            print(f"Cannot read {photo_path}; using a webcam snapshot instead")
    if photo is None:
        photo = snapshot
    cv2.imencode(".jpg", photo)[1].tofile(str(face_path))

    print(f"Saved {clip_path} ({frames} frames) and {face_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
