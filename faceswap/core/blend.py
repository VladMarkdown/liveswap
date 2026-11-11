"""
Laplacian pyramid blending for seamless face compositing.
5-level pyramid — eliminates boundary artifacts that deepfake detectors look for.
Poisson blending avoided: causes flickering and color bleed in real-time.
"""
from __future__ import annotations
import cv2
import numpy as np

# Used to neutralize reflected/replicated edge pixels outside the face mask
# before the pyramid spreads them into the blend transition zone.
_DILATE_KERNEL = np.ones((17, 17), np.uint8)  # ~8px radius


def laplacian_pyramid_blend(src: np.ndarray, dst: np.ndarray,
                             mask: np.ndarray, levels: int = 5) -> np.ndarray:
    """
    Blend src into dst using Laplacian pyramid with smooth mask.

    Args:
        src:   source image (face region) HWC BGR uint8
        dst:   destination image (original frame) HWC BGR uint8
        mask:  blend mask HW float32 [0,1] or HW uint8 [0,255]
        levels: pyramid depth (5 recommended for face swap)

    Returns:
        blended image HWC BGR uint8
    """
    src_f = src.astype(np.float32)
    dst_f = dst.astype(np.float32)

    if mask.dtype == np.uint8:
        mask_f = mask.astype(np.float32) / 255.0
    else:
        mask_f = mask.astype(np.float32)

    if mask_f.ndim == 3:
        mask_f = mask_f[..., 0]

    # Fill src outside dilated mask with dst content so that reflected/replicated
    # edge pixels (from BORDER_REPLICATE warp) cannot bleed into the transition zone.
    mask_binary = (mask_f > 0.01).astype(np.uint8)
    dilated = cv2.dilate(mask_binary, _DILATE_KERNEL)
    cv2.copyTo(dst_f, 1 - dilated, src_f)

    # Build Gaussian pyramids
    gp_src = [src_f]
    gp_dst = [dst_f]
    gp_mask = [mask_f]

    for _ in range(levels):
        gp_src.append(cv2.pyrDown(gp_src[-1]))
        gp_dst.append(cv2.pyrDown(gp_dst[-1]))
        gp_mask.append(cv2.pyrDown(gp_mask[-1]))

    # Build Laplacian pyramids
    lp_src = []
    lp_dst = []
    for i in range(levels):
        up = cv2.pyrUp(gp_src[i + 1], dstsize=(gp_src[i].shape[1], gp_src[i].shape[0]))
        lp_src.append(gp_src[i] - up)
        up = cv2.pyrUp(gp_dst[i + 1], dstsize=(gp_dst[i].shape[1], gp_dst[i].shape[0]))
        lp_dst.append(gp_dst[i] - up)

    # Blend each level
    lp_blend = []
    for i in range(levels):
        m = gp_mask[i]
        lp_blend.append(cv2.blendLinear(lp_src[i], lp_dst[i], m, 1.0 - m))

    # Reconstruct
    top = gp_mask[levels]
    result = cv2.blendLinear(gp_src[levels], gp_dst[levels], top, 1.0 - top)
    for i in range(levels - 1, -1, -1):
        result = cv2.pyrUp(result, dstsize=(lp_blend[i].shape[1], lp_blend[i].shape[0]))
        result += lp_blend[i]

    return np.clip(result, 0, 255).astype(np.uint8)
