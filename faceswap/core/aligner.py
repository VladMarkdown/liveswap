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


def feather_coverage(coverage: np.ndarray,
                     erosion_px: int = 8, transition_px: int = 16) -> np.ndarray:
    """
    Apply smoothstep feathering to a hard coverage mask (from warpAffine over ones).
    Prevents hard boundary artifacts where the swap crop meets the GPEN canvas.
    """
    binary = (coverage > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    u = np.clip((dist - erosion_px) / max(1.0, float(transition_px)), 0.0, 1.0)
    return (u * u * (3.0 - 2.0 * u)).astype(np.float32)


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
    frame_mask = np.zeros((h, w), np.float32)
    # Locate the mask in crop space and project its box into the frame, so only
    # that region of the frame is warped and blended.
    bx, by, bw, bh = cv2.boundingRect((crop_mask > 0).astype(np.uint8))
    if bw == 0:
        result = frame_bgr.copy()
        return (result, frame_mask) if return_mask else result
    corners = np.array([[bx, by], [bx + bw, by], [bx, by + bh], [bx + bw, by + bh]], np.float64)
    corners = corners @ inverse[:, :2].T + inverse[:, 2]
    # A five-level pyramid spreads the mask by about 2 * 2**levels pixels.
    padding = 66
    x0 = max(0, int(np.floor(corners[:, 0].min())) - padding)
    y0 = max(0, int(np.floor(corners[:, 1].min())) - padding)
    x1 = min(w, int(np.ceil(corners[:, 0].max())) + padding)
    y1 = min(h, int(np.ceil(corners[:, 1].max())) + padding)
    if x1 <= x0 or y1 <= y0:
        result = frame_bgr.copy()
        return (result, frame_mask) if return_mask else result
    local_inverse = inverse.copy()
    local_inverse[:, 2] -= (x0, y0)
    frame_mask[y0:y1, x0:x1] = cv2.warpAffine(
        crop_mask, local_inverse, (x1 - x0, y1 - y0),
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped = cv2.warpAffine(enhanced, local_inverse, (x1 - x0, y1 - y0),
                            borderMode=cv2.BORDER_REPLICATE)
    result = frame_bgr.copy()
    result[y0:y1, x0:x1] = laplacian_pyramid_blend(
        warped, frame_bgr[y0:y1, x0:x1], frame_mask[y0:y1, x0:x1], levels=5)
    return (result, frame_mask) if return_mask else result
