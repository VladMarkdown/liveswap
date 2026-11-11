"""
Color correction pipeline. EXACT ORDER (critical):
  1. Run swap
  2. Run GPEN restoration
  3. ONE masked Reinhard color match AFTER GPEN (correction confined to skin mask)
  4. Local shading correction (multiplicative in linear space, skin mask only)
  5. Skin detail injection (log-luminance HF band)
  6. SSS approximation (Jimenez separable Gaussian, linearized)
  7. Specular highlight transfer (temporal EMA on spec mask)
  8. Composite mouth interior (class 11) from target
  9. Motion-compensated temporal EMA

Reference for color correction: original target crop BEFORE enhancement.
Apply color correction again after GPEN (GPEN shifts colors — re-run once).
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# Cached pore texture tile — generated once at import, reused every frame
_PORE_TILE: np.ndarray | None = None


def _get_pore_tile(size: int = 256) -> np.ndarray:
    global _PORE_TILE
    if _PORE_TILE is None:
        rng = np.random.default_rng(42)
        noise = rng.standard_normal((size, size)).astype(np.float32)
        pore_layer = np.zeros((size, size), dtype=np.float32)
        for theta_deg in [0, 45, 90, 135]:
            kernel = cv2.getGaborKernel(
                (15, 15), sigma=2.0, theta=np.deg2rad(theta_deg),
                lambd=8.0, gamma=0.8, psi=0
            )
            pore_layer += np.abs(cv2.filter2D(noise, cv2.CV_32F, kernel))
        pore_layer = (pore_layer - pore_layer.mean()) / (pore_layer.std() + 1e-6)
        _PORE_TILE = pore_layer
    return _PORE_TILE


# Jimenez separable SSS skin profile: (weight, sigma_normalized)
_SSS_PROFILE = [
    (0.233, 0.0064), (0.100, 0.0484), (0.118, 0.1870),
    (0.113, 0.5670), (0.358, 1.9900), (0.078, 7.4100),
]


def reinhard_color_transfer(source: np.ndarray, target: np.ndarray,
                             mask: np.ndarray | None = None) -> np.ndarray:
    """
    Transfer color statistics from target to source (Reinhard et al. in LAB space).
    mask: float32 (H,W) 0-1 or uint8 0/255; statistics AND correction confined to mask.
    When mask is provided, pixels outside it are left unchanged (source values).
    """
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    result = src_lab.copy()
    for ch in range(3):
        if mask is not None:
            m = mask > (127 if mask.dtype == np.uint8 else 0.5)
            if m.sum() < 10:
                continue
            src_mean = src_lab[..., ch][m].mean()
            src_std  = src_lab[..., ch][m].std() + 1e-6
            tgt_mean = tgt_lab[..., ch][m].mean()
            tgt_std  = tgt_lab[..., ch][m].std() + 1e-6
            # Apply correction ONLY to masked pixels; outside stays as original source.
            result[..., ch][m] = (src_lab[..., ch][m] - src_mean) * (tgt_std / src_std) + tgt_mean
        else:
            src_mean, src_std = src_lab[..., ch].mean(), src_lab[..., ch].std() + 1e-6
            tgt_mean, tgt_std = tgt_lab[..., ch].mean(), tgt_lab[..., ch].std() + 1e-6
            result[..., ch] = (src_lab[..., ch] - src_mean) * (tgt_std / src_std) + tgt_mean

    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def low_freq_shading_correction(swapped: np.ndarray, target: np.ndarray,
                                sigma: float = 20.0,
                                skin_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Match low-frequency shading of swapped face to target.
    Multiplicative correction in linear (physical) light space.
    When skin_mask is provided the correction is blended in only within that region.
    """
    # Linearize: shading is a linear-light phenomenon
    sw_lin = (swapped.astype(np.float32) / 255.0) ** 2.2
    tg_lin = (target.astype(np.float32)  / 255.0) ** 2.2

    sw_low = gaussian_filter(sw_lin, sigma=[sigma, sigma, 0])
    tg_low = gaussian_filter(tg_lin, sigma=[sigma, sigma, 0])

    # Multiplicative ratio; partial 0.7 and clamp prevent extreme corrections
    ratio = tg_low / (sw_low + 1e-6)
    ratio = np.clip(1.0 + (ratio - 1.0) * 0.7, 0.3, 3.0)
    result_lin = sw_lin * ratio

    # Back to gamma
    result = np.clip(np.maximum(result_lin, 0.0) ** (1.0 / 2.2) * 255.0, 0, 255).astype(np.uint8)

    if skin_mask is not None:
        # Blend corrected result into original only within skin region
        skin_f = cv2.GaussianBlur(
            skin_mask.astype(np.float32) / 255.0, (11, 11), 3.0
        )[..., np.newaxis]
        result = np.clip(
            swapped.astype(np.float32) * (1.0 - skin_f) + result.astype(np.float32) * skin_f,
            0, 255
        ).astype(np.uint8)

    return result


def inject_skin_detail(swapped: np.ndarray, target_registered: np.ndarray,
                       skin_mask: np.ndarray, face_width: int,
                       strength: float = 0.20) -> np.ndarray:
    """
    Transfer high-frequency skin texture (pores, micro-wrinkles) from target
    to swapped face WITHOUT transferring identity features.

    Uses log-luminance band separation with edge gating.
    swapped, target_registered: HWC BGR uint8, or float32 LINEAR
    skin_mask: HW uint8 0/255
    """
    encoded = swapped.dtype == np.uint8
    if encoded:
        swapped = (swapped.astype(np.float32) / 255.0) ** 2.2
        target_registered = (target_registered.astype(np.float32) / 255.0) ** 2.2

    def luminance(img: np.ndarray) -> np.ndarray:
        # BT.709 luminance (BGR order)
        return 0.0722 * img[..., 0] + 0.7152 * img[..., 1] + 0.2126 * img[..., 2]

    sigma0 = max(0.35, 0.5 * face_width / 256)
    sigma1 = max(0.80, 2.0 * face_width / 256)

    ls = np.log(np.maximum(luminance(swapped), 1e-4))
    lt = np.log(np.maximum(luminance(target_registered), 1e-4))

    ds = gaussian_filter(ls, sigma0) - gaussian_filter(ls, sigma1)
    dt = gaussian_filter(lt, sigma0) - gaussian_filter(lt, sigma1)

    # Edge gating: suppress transfer near strong edges
    grad = np.abs(cv2.Sobel(ls.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)) + \
           np.abs(cv2.Sobel(ls.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3))
    edge_gate = np.exp(-grad * 10.0)

    safe_skin = (skin_mask > 127).astype(np.float32)
    weight = strength * safe_skin * edge_gate
    delta = np.clip(dt - ds, -0.10, 0.10)

    result = swapped * np.exp(weight * delta)[..., np.newaxis]
    if encoded:
        return np.clip(np.maximum(result, 0) ** (1 / 2.2) * 255, 0, 255).astype(np.uint8)
    return result


def apply_sss(face_bgr: np.ndarray, skin_mask: np.ndarray,
              strength: float = 0.20) -> np.ndarray:
    """
    Jimenez separable SSS approximation.
    Removes "plastic" skin look. R scatters 2mm, G ~1mm, B ~0.3mm.
    Linearizes input (gamma→linear) before blurring for physically correct scatter,
    and uses scipy gaussian_filter to handle all sigma values without kernel cap.
    """
    h, w = face_bgr.shape[:2]
    # Linearize: SSS is a linear-light phenomenon
    face_f = (face_bgr.astype(np.float32) / 255.0) ** 2.2
    channel_scale = [0.3, 0.6, 1.0]  # B, G, R
    sss_channels = []

    for ch in range(3):
        ch_layer = face_f[..., ch]
        sss_layer = np.zeros((h, w), dtype=np.float32)
        for weight, sigma_norm in _SSS_PROFILE:
            sigma_px = sigma_norm * w * 0.5 * channel_scale[ch]
            if sigma_px < 0.5:
                sss_layer += weight * ch_layer
                continue
            # scipy gaussian_filter auto-sizes the kernel — no ksize cap that
            # caused box-filter artefacts on the long-range scatter lobes.
            blurred = gaussian_filter(ch_layer, sigma_px)
            sss_layer += weight * blurred
        sss_channels.append(sss_layer)

    sss_result = np.stack(sss_channels, axis=-1)
    mask_3ch = cv2.GaussianBlur(
        np.stack([skin_mask] * 3, axis=-1).astype(np.float32) / 255.0,
        (11, 11), 3.0
    )
    out_lin = face_f + strength * mask_3ch * (sss_result - face_f)
    # Delinearize back to gamma
    out = np.clip(np.maximum(out_lin, 0.0) ** (1.0 / 2.2) * 255.0, 0, 255)
    return out.astype(np.uint8)


def apply_micro_texture(face_bgr: np.ndarray, skin_mask: np.ndarray,
                        strength: float = 0.04) -> np.ndarray:
    """Add procedural pore texture to skin region (cached Gabor tile)."""
    h, w = face_bgr.shape[:2]
    tile = cv2.resize(_get_pore_tile(256), (w, h))
    face_f = face_bgr.astype(np.float32)
    skin_norm = cv2.GaussianBlur(
        skin_mask.astype(np.float32) / 255.0, (11, 11), 3.0
    )[..., np.newaxis]
    displaced = face_f * (1.0 + strength * tile[..., np.newaxis])
    out = face_f + skin_norm * (displaced - face_f)
    return np.clip(out, 0, 255).astype(np.uint8)


def transfer_specular(target_face: np.ndarray, swapped_face: np.ndarray,
                      skin_mask: np.ndarray,
                      thresh_sat: int = 30, thresh_val: int = 200,
                      prev_spec: np.ndarray | None = None,
                      spec_alpha: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """
    Transfer specular highlights from target face to swapped face.
    Uses dichromatic model: speculars are achromatic (high brightness, low saturation).

    Applies temporal EMA (alpha=0.3) on the specular mask to suppress frame-to-frame
    flicker caused by pixels crossing the hard HSV threshold boundary.

    Returns (result_bgr, spec_mask) so the caller can store spec_mask and pass it
    back as prev_spec on the next frame.
    """
    hsv = cv2.cvtColor(target_face, cv2.COLOR_BGR2HSV).astype(np.float32)
    spec_new = ((hsv[..., 1] < thresh_sat) & (hsv[..., 2] > thresh_val)).astype(np.float32)

    # Temporal EMA to prevent boundary-crossing flicker
    if prev_spec is not None and prev_spec.shape == spec_new.shape:
        spec = spec_alpha * spec_new + (1.0 - spec_alpha) * prev_spec
    else:
        spec = spec_new

    spec_smooth = cv2.GaussianBlur(spec, (5, 5), 1.5)
    skin_norm = cv2.GaussianBlur(
        skin_mask.astype(np.float32) / 255.0, (11, 11), 3.0
    )
    combined = (spec_smooth * skin_norm)[..., np.newaxis]
    result = np.clip(
        swapped_face.astype(np.float32) * (1 - combined) +
        target_face.astype(np.float32) * combined,
        0, 255
    ).astype(np.uint8)
    return result, spec


def composite_mouth_interior(swapped: np.ndarray, target: np.ndarray,
                              mouth_mask: np.ndarray) -> np.ndarray:
    """
    Composite target's mouth interior (teeth/tongue) over swapped face.
    Preserves natural teeth appearance. mask: uint8 0/255.
    """
    alpha = cv2.GaussianBlur(mouth_mask.astype(np.float32) / 255.0, (5, 5), 1.0)
    alpha = alpha[..., np.newaxis]
    result = swapped.astype(np.float32) * (1 - alpha) + target.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)
