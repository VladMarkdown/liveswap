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


def _area(bbox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _iou(a, b) -> float:
    inter = _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    return inter / max(_area(a) + _area(b) - inter, 1e-6)


def detect_tracked(frame_bgr: np.ndarray, prev_bbox=None):
    """
    Live-path detection: SCRFD plus the 106-point landmarks for one face only.

    Skips the recognition, gender/age and 3D-landmark models that
    FaceAnalysis.get() runs on every face; the live target needs none of them.
    Follows the face that overlaps prev_bbox (so a second, similarly sized face
    cannot steal the swap); otherwise picks the largest. Returns a Face with
    .bbox, .kps and .landmark_2d_106, or None.
    """
    from insightface.app.common import Face
    with _app_lock:
        app = _get_app()
        bboxes, kpss = app.det_model.detect(frame_bgr, max_num=0, metric='default')
        if bboxes.shape[0] == 0:
            return None
        index = int(np.argmax([_area(b) for b in bboxes]))
        if prev_bbox is not None:
            overlaps = [_iou(b, prev_bbox) for b in bboxes]
            if max(overlaps) > 0.3:
                index = int(np.argmax(overlaps))
        face = Face(bbox=bboxes[index, :4], kps=kpss[index], det_score=bboxes[index, 4])
        landmarks = app.models.get('landmark_2d_106')
        if landmarks is not None:
            landmarks.get(frame_bgr, face)
    return face


def get_eye_landmarks(face) -> tuple:
    """
    Eye contour points for the blink gate from buffalo_l's 2d106det output.
    Returns (left, right), each (10, 2) float32, or (None, None) if unavailable.

    In the 106-point layout the eyes are 33-42 and 87-96; 52-71 is the mouth.
    The order within each eye is not relied on (see eye_aspect_ratio).
    """
    lm = getattr(face, 'landmark_2d_106', None)
    if lm is None or len(lm) < 106:
        return None, None
    return lm[33:43].astype(np.float32), lm[87:97].astype(np.float32)
