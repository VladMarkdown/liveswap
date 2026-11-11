"""
Face parsing via BiSeNet (zllrunning/face-parsing.PyTorch, CelebAMask-HQ 19 classes).

VERIFIED label map (CelebAMask-HQ checkpoint):
  0:background  1:skin        2:l_brow    3:r_brow
  4:l_eye       5:r_eye       6:glasses   7:l_ear
  8:r_ear       9:earring    10:nose     11:mouth_interior (teeth+tongue)
 12:upper_lip  13:lower_lip  14:neck     15:necklace
 16:clothing   17:hair       18:hat

SWAP region  (replace with source): [1,2,3,4,5,10,11,12,13]
PRESERVE region (keep from target): [0,6,7,8,9,14,15,16,17,18]

NOTE: class 13 = lower_lip (NOT teeth). Teeth are inside class 11 (mouth_interior).
Preserve mouth interior (11) from target for natural teeth appearance.

BiSeNet CANNOT use TRT FP16 — softmax over 19 classes is fragile. Use CUDA FP32.
"""
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path

from .ort_utils import create_session

SWAP_LABELS    = [1, 2, 3, 4, 5, 10, 11, 12, 13]
PRESERVE_LABELS = [0, 6, 7, 8, 9, 14, 15, 16, 17, 18]
HAIR_LABEL     = 17
GLASSES_LABEL  = 6
EYE_LABELS     = [4, 5]
MOUTH_INTERIOR = 11

_FP32_PROVIDERS = [
    ("CUDAExecutionProvider", {"device_id": 0}),
    "CPUExecutionProvider",
]


class FaceParser:
    def __init__(self, model_path: str | Path, logit_ema_alpha: float = 0.6):
        self._session = create_session(model_path)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._logit_ema_alpha = logit_ema_alpha
        self._prev_logits: np.ndarray | None = None

    def reset(self) -> None:
        """Reset temporal logit state (call when face is lost or subject changes)."""
        self._prev_logits = None

    def parse(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        """
        Run BiSeNet on a 512×512 face crop.

        Returns:
            seg_map: (512, 512) int32 label map, 19 classes
        """
        h, w = face_crop_bgr.shape[:2]
        # BiSeNet expects 512×512 RGB, normalized to ImageNet stats
        if h != 512 or w != 512:
            img = cv2.resize(face_crop_bgr, (512, 512))
        else:
            img = face_crop_bgr

        rgb = img[:, :, ::-1].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        blob = rgb.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 512, 512)

        logits = self._session.run([self._output_name], {self._input_name: blob})[0]

        # Temporal EMA on raw logits: suppresses per-frame boundary shimmer at hair/jaw
        if self._prev_logits is not None:
            logits = (self._logit_ema_alpha * logits
                      + (1.0 - self._logit_ema_alpha) * self._prev_logits)
        self._prev_logits = logits

        seg = np.argmax(logits[0], axis=0).astype(np.int32)  # (512, 512)
        return seg

    def build_swap_mask(self, seg_map: np.ndarray,
                        erosion_px: int = 3,
                        transition_px: int = 10) -> np.ndarray:
        """
        Build smooth alpha mask for the face swap region.

        Returns float32 (H, W) mask in range [0, 1].
        Hair and glasses are cut out (preserved from original).
        """
        binary = np.zeros(seg_map.shape, dtype=np.uint8)
        for label in SWAP_LABELS:
            binary[seg_map == label] = 255

        # Cut hair and glasses holes
        hair = (seg_map == HAIR_LABEL).astype(np.uint8) * 255
        glasses = (seg_map == GLASSES_LABEL).astype(np.uint8) * 255
        binary = cv2.bitwise_and(binary, cv2.bitwise_not(cv2.dilate(hair, None, iterations=2)))
        binary = cv2.bitwise_and(binary, cv2.bitwise_not(glasses))

        # Smoothstep alpha via distance transform
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        u = np.clip((dist - erosion_px) / max(1.0, transition_px), 0.0, 1.0)
        alpha = u * u * (3.0 - 2.0 * u)  # smoothstep
        return alpha.astype(np.float32)

    def get_region_mask(self, seg_map: np.ndarray, labels: list[int]) -> np.ndarray:
        """Binary mask for specific labels. Returns uint8 0/255."""
        mask = np.zeros(seg_map.shape, dtype=np.uint8)
        for label in labels:
            mask[seg_map == label] = 255
        return mask

    def get_skin_mask(self, seg_map: np.ndarray) -> np.ndarray:
        """Skin only (label 1), for SSS/detail injection."""
        return self.get_region_mask(seg_map, [1])

    def get_eye_mask(self, seg_map: np.ndarray) -> np.ndarray:
        """Both eyes (labels 4+5)."""
        return self.get_region_mask(seg_map, EYE_LABELS)

    def get_mouth_interior_mask(self, seg_map: np.ndarray) -> np.ndarray:
        """Mouth interior (label 11) — teeth+tongue cavity."""
        return self.get_region_mask(seg_map, [MOUTH_INTERIOR])
