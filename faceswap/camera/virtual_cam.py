"""
Virtual webcam output. Two backends, tried in this order:

1. softcam (tshino/softcam, MIT): used when its DLL is present at dll_path.
   It has no prebuilt releases; build softcam.dll from source and register it
   with RegisterSoftcam.bat as Administrator.
2. OBS Virtual Camera through pyvirtualcam: used otherwise. Needs OBS Studio
   installed (its installer registers the camera); OBS does not have to run.
   Video apps list it as "OBS Virtual Camera".
"""
import ctypes
import os
import numpy as np


class VirtualCamera:
    def __init__(self, width: int, height: int, fps: float = 30.0,
                 dll_path: str | None = None):
        self._cam = None      # softcam handle
        self._obs = None      # pyvirtualcam.Camera
        if width % 4 != 0 or height % 4 != 0:
            raise ValueError(f"Width/height must be multiples of 4, got {width}x{height}")
        self._width = width
        self._height = height

        if dll_path is None:
            dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "softcam.dll")
        if os.path.exists(dll_path):
            self._open_softcam(dll_path, fps)
        else:
            self._open_obs(fps)

    def _open_softcam(self, dll_path: str, fps: float) -> None:
        try:
            self._dll = ctypes.CDLL(dll_path)
        except OSError as e:
            raise RuntimeError(
                f"Cannot load softcam.dll from {dll_path}.\n"
                "Build from https://github.com/tshino/softcam and run "
                "RegisterSoftcam.bat as Administrator."
            ) from e

        self._setup_argtypes()
        self._cam = self._dll.scCreateCamera(self._width, self._height, ctypes.c_float(fps))

        if not self._cam:
            raise RuntimeError(
                "scCreateCamera returned NULL.\n"
                "  1. DLL not registered — run RegisterSoftcam.bat as admin\n"
                "  2. Another softcam instance is already running\n"
                "  3. Width/height not multiples of 4"
            )

    def _open_obs(self, fps: float) -> None:
        try:
            import pyvirtualcam
            self._obs = pyvirtualcam.Camera(self._width, self._height, fps,
                                            fmt=pyvirtualcam.PixelFormat.BGR)
        except Exception as e:
            raise RuntimeError(
                "No virtual camera is available. Install OBS Studio "
                "(https://obsproject.com), which provides \"OBS Virtual Camera\", "
                "and close any other program that is using it."
            ) from e

    @property
    def device_name(self) -> str:
        """Name the camera has in video apps."""
        return self._obs.device if self._obs is not None else "DirectShow Softcam"

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
        if self._obs is not None:
            return True
        return bool(self._dll.scWaitForConnection(self._cam, ctypes.c_float(timeout)))

    def send_frame(self, frame_bgr: np.ndarray) -> None:
        """Push one BGR frame. frame_bgr: (H, W, 3) uint8, OpenCV native."""
        if frame_bgr.dtype != np.uint8:
            raise ValueError("Frame dtype must be uint8")
        if self._cam is None and self._obs is None:
            raise RuntimeError("Virtual camera is closed")
        if frame_bgr.shape != (self._height, self._width, 3):
            raise ValueError(
                f"Expected ({self._height}, {self._width}, 3), got {frame_bgr.shape}"
            )
        if not frame_bgr.flags["C_CONTIGUOUS"]:
            frame_bgr = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
        if self._obs is not None:
            self._obs.send(frame_bgr)
        else:
            self._dll.scSendFrame(self._cam, frame_bgr.ctypes.data_as(ctypes.c_void_p))

    def is_connected(self) -> bool:
        if self._obs is not None:
            return True
        return bool(self._dll.scIsConnected(self._cam))

    def close(self) -> None:
        if getattr(self, "_cam", None):
            self._dll.scDeleteCamera(self._cam)
            self._cam = None
        if getattr(self, "_obs", None) is not None:
            self._obs.close()
            self._obs = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
