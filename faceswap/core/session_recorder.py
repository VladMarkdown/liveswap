"""Records a running session for later analysis: camera frame and output side by side.

Writes two files into the given folder:
    session-YYYYMMDD-HHMMSS.mp4   input | output, each at half size
    session-YYYYMMDD-HHMMSS.csv   one row per delivered frame: time, work per frame

Meant for testing only; it is switched on explicitly in the app's settings.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


class SessionRecorder:
    def __init__(self, folder: str | Path, fps: float = 20.0):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.video_path = folder / f"session-{stamp}.mp4"
        self._csv = open(folder / f"session-{stamp}.csv", "w", encoding="ascii")
        self._csv.write("seconds,work_ms,swapped\n")
        self._fps = fps
        self._writer = None
        self._started = time.perf_counter()

    def write(self, camera: np.ndarray, output: np.ndarray, work_ms: float) -> None:
        """Add one delivered frame. camera and output are full-size BGR frames."""
        half = (output.shape[1] // 2, output.shape[0] // 2)
        pair = np.hstack((cv2.resize(camera, half, interpolation=cv2.INTER_AREA),
                          cv2.resize(output, half, interpolation=cv2.INTER_AREA)))
        if self._writer is None:
            self._writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                           self._fps, (pair.shape[1], pair.shape[0]))
        self._writer.write(pair)
        swapped = 0 if camera is output else 1
        self._csv.write(f"{time.perf_counter() - self._started:.3f},{work_ms:.1f},{swapped}\n")

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if not self._csv.closed:
            self._csv.close()
