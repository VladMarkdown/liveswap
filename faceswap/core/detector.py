"""
Face detection via InsightFace buffalo_l (SCRFD).
Returns list of Face objects with bbox, kps (5-point), embedding, normed_embedding.
"""
from __future__ import annotations
import numpy as np
from threading import RLock


_app = None
_app_lock = RLock()


def _get_app():
    global _app
    with _app_lock:
        if _app is None:
            from insightface.app import FaceAnalysis
            import onnxruntime as ort
            available = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider",
                                     "CPUExecutionProvider") if p in available]
            options = ort.SessionOptions()
            if "DmlExecutionProvider" in providers:
                options.enable_mem_pattern = False
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            app = FaceAnalysis(name="buffalo_l", providers=providers, sess_options=options)
            app.prepare(ctx_id=0 if providers[0] != "CPUExecutionProvider" else -1,
                        det_size=(640, 640))
            _app = app
        return _app


def detect_faces(frame_bgr: np.ndarray) -> list:
    """
    Detect all faces in a BGR frame.
    Returns list of insightface Face objects sorted by bbox area (largest first).
    Each face has: .bbox, .kps (5×2), .embedding (512,), .normed_embedding (512,).
    """
    with _app_lock:
        faces = _get_app().get(frame_bgr)
    if not faces:
        return []
    return sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )


def detect_largest(frame_bgr: np.ndarray):
    """Return the largest face or None."""
    faces = detect_faces(frame_bgr)
    return faces[0] if faces else None


def get_eye_landmarks_6pt(face) -> tuple:
    """
    Extract 6-point eye landmarks for EAR from buffalo_l's 2d106det output.
    Returns (left_6, right_6) each (6, 2) float32, or (None, None) if unavailable.

    buffalo_l 2d106det eye indices (InsightFace convention):
      Left eye:  60 (l-corner), 61, 63, 64 (r-corner), 65, 67
      Right eye: 68 (l-corner), 69, 71, 72 (r-corner), 73, 75
    Landmark order matches eye_aspect_ratio:
      [l-corner, top-left, top-right, r-corner, bottom-right, bottom-left]
    """
    lm = getattr(face, 'landmark_2d_106', None)
    if lm is None or len(lm) < 106:
        return None, None
    try:
        return (lm[[60, 61, 63, 64, 65, 67]].astype(np.float32),
                lm[[68, 69, 71, 72, 73, 75]].astype(np.float32))
    except (IndexError, TypeError):
        return None, None
