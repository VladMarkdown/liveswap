"""
Occlusion detection for the face swap pipeline.

Detects objects in front of the face (hands, cups, phones) that should NOT be
overwritten by the swapped face. Returns a float32 mask where 0=visible, 1=occluded.

Two backends:
  1. ONNX model (models/occluder.onnx) — accurate, requires optional download
  2. BiSeNet heuristic fallback — no extra model needed
"""
from __future__ import annotations
import logging
from pathlib import Path
from threading import RLock

import cv2
import numpy as np

logger = logging.getLogger(__name__)
_OCCLUDER_MODEL = Path("models/occluder.onnx")
_session = None
_session_lock = RLock()


def _try_load_session():
    """Lazy-load the optional occluder ONNX model."""
    global _session
    with _session_lock:
        if _session is not None:
            return _session
        if not _OCCLUDER_MODEL.exists():
            _session = "unavailable"
            return _session
        try:
            from .ort_utils import create_session
            _session = create_session(_OCCLUDER_MODEL)
        except Exception:
            logger.exception("Occluder model failed to load; using the heuristic")
            _session = "unavailable"
        return _session


class OcclusionDetector:
    """
    Detects occluders in a 512×512 FFHQ-aligned face crop.
    Returns a float32 mask (0=face visible, 1=occluded) in 512×512 space.

    Usage:
        detector = OcclusionDetector()
        mask = detector.get_occlusion_mask(face_crop_512)  # (512, 512) float32
        swap_mask = swap_mask * (1.0 - mask)
    """

    def __init__(self):
        self._session = None  # lazy-loaded

    def get_occlusion_mask(self, face_crop_512: np.ndarray,
                           bisenet_seg: np.ndarray | None = None) -> np.ndarray:
        """
        face_crop_512: (512, 512, 3) BGR uint8
        bisenet_seg:   optional (512, 512) int32 BiSeNet label map (class 0 = background)
        Returns:       (512, 512) float32, 0=face, 1=occluder
        """
        if face_crop_512.shape[:2] != (512, 512):
            raise ValueError(f"Expected 512×512 input, got {face_crop_512.shape[:2]}")

        sess = _try_load_session()
        if sess != "unavailable":
            return self._run_onnx(face_crop_512, sess)
        return self._heuristic(face_crop_512, bisenet_seg)

    @staticmethod
    def _run_onnx(crop: np.ndarray, sess) -> np.ndarray:
        """Binary segmentation via occluder ONNX model."""
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        blob = rgb.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 512, 512)
        input_name = sess.get_inputs()[0].name
        pred = sess.run(None, {input_name: blob})[0]  # (1, 1, H, W) or (1, H, W)
        mask = pred.squeeze()
        if mask.shape != (512, 512):
            mask = cv2.resize(mask, (512, 512))
        return np.clip(mask, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _heuristic(crop: np.ndarray, seg: np.ndarray | None) -> np.ndarray:
        """
        BiSeNet heuristic: find large non-skin, non-background blobs in the face
        region that could be occluders (hands, objects, etc.).

        Without BiSeNet labels, uses skin-color exclusion in YCrCb space.
        """
        h, w = crop.shape[:2]
        occ = np.zeros((h, w), np.float32)

        # Skin detection in YCrCb
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        skin = (
            (ycrcb[..., 1] > 133) & (ycrcb[..., 1] < 177) &
            (ycrcb[..., 2] > 77)  & (ycrcb[..., 2] < 127)
        ).astype(np.uint8) * 255
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

        # Focus on the central face region (avoid border effects)
        roi_mask = np.zeros((h, w), np.uint8)
        roi_mask[h//6:5*h//6, w//8:7*w//8] = 255

        # Non-skin inside the face ROI = candidate occluder
        non_skin = cv2.bitwise_and(cv2.bitwise_not(skin), roi_mask)

        # Remove very small blobs (noise, pores) — keep large foreign objects
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(non_skin)
        min_area = (h * w) // 40  # at least 2.5% of face area

        for lbl in range(1, n_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            # Check that the blob is surrounded by skin on 3+ sides
            blob = (labels == lbl).astype(np.uint8)
            dil = cv2.dilate(blob, np.ones((25, 25), np.uint8))
            border = (dil - blob).astype(bool)
            if border.sum() == 0:
                continue
            skin_contact = float(skin[border].mean()) / 255.0
            if skin_contact > 0.35:  # ~35% of border touches skin → likely occluder
                smooth = cv2.GaussianBlur(blob.astype(np.float32), (21, 21), 5.0)
                occ = np.maximum(occ, smooth)

        if seg is not None:
            # BiSeNet background (class 0) blobs inside the face region
            bg_in_face = ((seg == 0) & (roi_mask > 0)).astype(np.uint8)
            # Erode to remove thin boundaries
            bg_in_face = cv2.erode(bg_in_face,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            if bg_in_face.any():
                occ = np.maximum(occ,
                                  cv2.GaussianBlur(bg_in_face.astype(np.float32), (21, 21), 5.0))

        return np.clip(occ, 0.0, 1.0)
