"""Background OpenCV capture with a latest-frame mailbox."""
from __future__ import annotations

import logging
import queue
import sys
import threading
import time

import cv2

logger = logging.getLogger(__name__)


class CaptureThread:
    def __init__(self, source=0, width=1280, height=720, fps=30):
        self.source = source
        self.width, self.height = width, height
        self.requested_fps = fps
        self.fps = 0.0
        self.dropped_frames = 0
        self._frames = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = None
        self._cap = None
        self.error = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self.error = None
        self.fps = 0.0
        self.dropped_frames = 0
        while not self._frames.empty():
            self._frames.get_nowait()
        local_windows_camera = isinstance(self.source, int) and sys.platform == "win32"
        if local_windows_camera:
            # DirectShow opens faster than the default MSMF backend and honours
            # the MJPG request below.
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise RuntimeError(
                f"Cannot open camera {self.source}. If another program is using it "
                "(Telegram, Zoom, Discord), close the camera there and press Start again.")
        if local_windows_camera:
            # Uncompressed YUY2 over USB 2 caps many webcams near 10 fps at 720p.
            # The codec must be requested before the frame size.
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        actual = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                  self._cap.get(cv2.CAP_PROP_FPS))
        if actual[:2] != (self.width, self.height) or actual[2] < self.requested_fps - 1:
            logger.warning("Camera %s delivers %dx%d @ %.0f fps, not the requested %dx%d @ %d",
                           self.source, *actual, self.width, self.height, self.requested_fps)
        self._thread = threading.Thread(target=self._capture, daemon=True)
        self._thread.start()
        return self

    def _capture(self):
        cap = self._cap
        started, count = time.monotonic(), 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                count += 1
                self.fps = count / max(time.monotonic() - started, 1e-6)
                try:
                    self._frames.put_nowait(frame)
                except queue.Full:
                    try:
                        self._frames.get_nowait()
                        self.dropped_frames += 1
                    except queue.Empty:
                        pass
                    self._frames.put_nowait(frame)
        except Exception as exc:
            self.error = exc
        finally:
            self._stop.set()
            cap.release()

    def read(self):
        while True:
            try:
                return True, self._frames.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set() or self._thread is None:
                    return False, None

    def isOpened(self):
        return self._cap is not None and not self._stop.is_set()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                # Network drivers may block read; release requests cancellation.
                self._cap.release()
                self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("Camera capture did not stop")
        self._thread = None
        self._cap = None

    release = stop
