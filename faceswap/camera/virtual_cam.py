"""
Virtual DirectShow webcam via tshino/softcam (MIT license).
Appears in Zoom, Teams, Meet, Discord, TikTok Live as a hardware webcam.

Setup (one-time, admin):
  1. Build softcam.dll from https://github.com/tshino/softcam
  2. Run RegisterSoftcam.bat as Administrator
  3. Place softcam.dll in faceswap/camera/ or add to PATH
"""
import ctypes
import os
import numpy as np


class VirtualCamera:
    def __init__(self, width: int, height: int, fps: float = 30.0,
                 dll_path: str | None = None):
        self._cam = None
        if width % 4 != 0 or height % 4 != 0:
            raise ValueError(f"Width/height must be multiples of 4, got {width}x{height}")

        if dll_path is None:
            dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "softcam.dll")

        try:
            self._dll = ctypes.CDLL(dll_path)
        except OSError as e:
            raise RuntimeError(
                f"Cannot load softcam.dll from {dll_path}.\n"
                "Build from https://github.com/tshino/softcam and run "
                "RegisterSoftcam.bat as Administrator."
            ) from e

        self._setup_argtypes()
        self._width = width
        self._height = height
        self._cam = self._dll.scCreateCamera(width, height, ctypes.c_float(fps))

        if not self._cam:
            raise RuntimeError(
                "scCreateCamera returned NULL.\n"
                "  1. DLL not registered — run RegisterSoftcam.bat as admin\n"
                "  2. Another softcam instance is already running\n"
                "  3. Width/height not multiples of 4"
            )

    def _setup_argtypes(self) -> None:
        d = self._dll
        d.scCreateCamera.restype  = ctypes.c_void_p
        d.scCreateCamera.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float]
        d.scDeleteCamera.restype  = None
        d.scDeleteCamera.argtypes = [ctypes.c_void_p]
        d.scSendFrame.restype     = None
        d.scSendFrame.argtypes    = [ctypes.c_void_p, ctypes.c_void_p]
        d.scWaitForConnection.restype  = ctypes.c_bool
        d.scWaitForConnection.argtypes = [ctypes.c_void_p, ctypes.c_float]
        d.scIsConnected.restype  = ctypes.c_bool
        d.scIsConnected.argtypes = [ctypes.c_void_p]

    def wait_for_connection(self, timeout: float = 0.0) -> bool:
        return bool(self._dll.scWaitForConnection(self._cam, ctypes.c_float(timeout)))

    def send_frame(self, frame_bgr: np.ndarray) -> None:
        """Push one BGR frame. frame_bgr: (H, W, 3) uint8, OpenCV native."""
        if frame_bgr.dtype != np.uint8:
            raise ValueError("Frame dtype must be uint8")
        if self._cam is None:
            raise RuntimeError("Virtual camera is closed")
        if frame_bgr.shape != (self._height, self._width, 3):
            raise ValueError(
                f"Expected ({self._height}, {self._width}, 3), got {frame_bgr.shape}"
            )
        if not frame_bgr.flags["C_CONTIGUOUS"]:
            frame_bgr = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
        self._dll.scSendFrame(self._cam, frame_bgr.ctypes.data_as(ctypes.c_void_p))

    def is_connected(self) -> bool:
        return bool(self._dll.scIsConnected(self._cam))

    def close(self) -> None:
        if getattr(self, "_cam", None):
            self._dll.scDeleteCamera(self._cam)
            self._cam = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
