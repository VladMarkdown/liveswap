"""
Face alignment: 5-point landmarks to affine transform and aligned crop.

Two alignment targets:
  - arcface_112: for swap models (AlphaFace, HyperSwap, inswapper)
  - ffhq_512:    for GPEN enhancement (requires different reference landmarks)
"""
from __future__ import annotations
import cv2
import numpy as np

# Reference landmarks for ArcFace 112 x 112 crop
_ARCFACE_REF_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

# Reference landmarks for FFHQ 512×512 crop (used by GPEN)
_FFHQ_REF_512 = np.array([
    [192.98138,  239.94708],
    [318.90277,  240.1936 ],
    [256.63416,  314.01935],
    [201.26117,  371.41043],
    [313.08905,  371.15118],
], dtype=np.float32)


def estimate_transform(kps: np.ndarray, image_size: int = 256) -> np.ndarray:
    """Estimate frame-to-crop transform; callers may smooth it before warping."""
    ref = _FFHQ_REF_512 if image_size == 512 else _ARCFACE_REF_112 * (image_size / 112.0)
    matrix, _ = cv2.estimateAffinePartial2D(
        np.asarray(kps, dtype=np.float32).reshape(5, 1, 2), ref.reshape(5, 1, 2)
    )
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError("Cannot align degenerate face landmarks")
    return matrix


def align_face(frame_bgr: np.ndarray, kps: np.ndarray,
               image_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Align to scaled ArcFace landmarks, or FFHQ landmarks for 512px GPEN."""
    matrix = estimate_transform(kps, image_size)
    crop = cv2.warpAffine(frame_bgr, matrix, (image_size, image_size),
                          borderMode=cv2.BORDER_REFLECT)
    return crop, matrix


def paste_back(frame_bgr: np.ndarray, enhanced: np.ndarray,
               M: np.ndarray, mask: np.ndarray | None = None,
               return_mask: bool = False):
    """Blend an aligned crop back with five pyramid levels.

    Optionally return (frame, frame_mask) with the warped soft face mask.
    """
    from .blend import laplacian_pyramid_blend

    h, w = frame_bgr.shape[:2]
    ih, iw = enhanced.shape[:2]
    inverse = cv2.invertAffineTransform(M)
    crop_mask = np.ones((ih, iw), np.float32) if mask is None else mask.astype(np.float32)
    if mask is not None and mask.dtype == np.uint8:
        crop_mask /= 255.0
    crop_mask = np.clip(crop_mask, 0, 1)
    frame_mask = cv2.warpAffine(crop_mask, inverse, (w, h),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    points = cv2.findNonZero((frame_mask > 0).astype(np.uint8))
    if points is None:
        result = frame_bgr.copy()
        return (result, frame_mask) if return_mask else result
    x, y, width, height = cv2.boundingRect(points)
    # Gaussian down/up sampling spreads support by < 4 * 2**levels pixels.
    padding = 128
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1, y1 = min(w, x + width + padding), min(h, y + height + padding)
    local_inverse = inverse.copy()
    local_inverse[:, 2] -= (x0, y0)
    warped = cv2.warpAffine(enhanced, local_inverse, (x1 - x0, y1 - y0),
                            borderMode=cv2.BORDER_REPLICATE)
    result = frame_bgr.copy()
    result[y0:y1, x0:x1] = laplacian_pyramid_blend(
        warped, frame_bgr[y0:y1, x0:x1], frame_mask[y0:y1, x0:x1], levels=5)
    return (result, frame_mask) if return_mask else result
