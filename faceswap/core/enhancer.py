"""
Face enhancement via GPEN-BFR-512 (ONNX).

GPEN input:  (1, 3, 512, 512) float32 RGB, range [-1, 1]: x = RGB/127.5 - 1
GPEN output: (1, 3, 512, 512) float32, decode: (y+1)*127.5

GPEN requires FFHQ-512 alignment (use aligner.align_face with image_size=512).
Run color correction AFTER GPEN (GPEN shifts colors).

GPEN can use TRT FP16 safely. SCRFD also. AlphaFace/BiSeNet: CUDA FP32 only.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

from .ort_utils import create_session


class FaceEnhancer:
    """
    GPEN-BFR-512 face restorer.
    Expects 512×512 FFHQ-aligned crop.
    """

    def __init__(self, model_path: str | Path, tensorrt: bool = False):
        self._session = create_session(model_path, tensorrt_fp16=tensorrt)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def enhance(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        """
        Enhance a 512×512 FFHQ-aligned face crop.

        Args:
            face_crop_bgr: (512, 512, 3) uint8 BGR

        Returns:
            enhanced: (512, 512, 3) uint8 BGR
        """
        # BGR → RGB → float32 NCHW, range [-1, 1]
        rgb = face_crop_bgr[:, :, ::-1].astype(np.float32)
        blob = (rgb / 127.5 - 1.0).transpose(2, 0, 1)[np.newaxis]  # (1, 3, 512, 512)

        result = self._session.run([self._output_name], {self._input_name: blob})[0]

        # Decode: (y+1)*127.5, NCHW → HWC BGR
        out = (result[0].transpose(1, 2, 0) + 1.0) * 127.5
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out[:, :, ::-1]  # RGB → BGR

    def enhance_with_blend(self, face_crop_bgr: np.ndarray,
                           blend: float = 0.8) -> np.ndarray:
        """
        Enhance + blend with original (blend=1.0 → full GPEN, 0.0 → no enhancement).
        blend 0.6-0.8 preserves more identity while still improving quality.
        """
        enhanced = self.enhance(face_crop_bgr)
        if blend >= 1.0:
            return enhanced
        result = face_crop_bgr.astype(np.float32) * (1.0 - blend) + \
                 enhanced.astype(np.float32) * blend
        return np.clip(result, 0, 255).astype(np.uint8)
