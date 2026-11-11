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
  9. Camera degradation matching (noise grain, sharpness, motion blur — last crop step)
 10. Motion-compensated temporal EMA

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


# Sum-of-Gaussians skin diffusion profile (d'Eon & Luebke, GPU Gems 3):
# (variance in mm², (R, G, B) weights). Each channel's weights sum to 1.
_SSS_PROFILE = [
    (0.0064, (0.233, 0.455, 0.649)),
    (0.0484, (0.100, 0.336, 0.344)),
    (0.1870, (0.118, 0.198, 0.000)),
    (0.5670, (0.113, 0.007, 0.007)),
    (1.9900, (0.358, 0.004, 0.000)),
    (7.4100, (0.078, 0.000, 0.000)),
]
# An FFHQ-512 crop puts the eyes ~126 px apart; at a ~63 mm interpupillary
# distance that is 2 px per mm, i.e. crop_width / 256.
_SSS_CROP_WIDTH_MM = 256.0

# sRGB-ish gamma 2.2 decode table for uint8 input.
_TO_LINEAR = ((np.arange(256, dtype=np.float32) / 255.0) ** 2.2).astype(np.float32)


def _to_gamma_u8(linear: np.ndarray) -> np.ndarray:
    """Linear float32 [0, 1] -> gamma-encoded uint8."""
    encoded = cv2.pow(np.clip(linear, 0.0, 1.0), 1.0 / 2.2)
    return np.clip(encoded * 255.0, 0, 255).astype(np.uint8)


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
    sw_lin = _TO_LINEAR[swapped]
    tg_lin = _TO_LINEAR[target]

    # The shading field is low-frequency by construction, so estimate it on a
    # small image: area-downsample, blur with the scaled sigma, upsample the ratio.
    h, w = swapped.shape[:2]
    factor = max(1, int(sigma // 2.5))
    small = (max(1, w // factor), max(1, h // factor))
    small_sigma = sigma / factor
    sw_low = cv2.GaussianBlur(cv2.resize(sw_lin, small, interpolation=cv2.INTER_AREA),
                              (0, 0), small_sigma)
    tg_low = cv2.GaussianBlur(cv2.resize(tg_lin, small, interpolation=cv2.INTER_AREA),
                              (0, 0), small_sigma)

    # Multiplicative ratio; partial 0.7 and clamp prevent extreme corrections
    ratio = tg_low / (sw_low + 1e-6)
    ratio = np.clip(1.0 + (ratio - 1.0) * 0.7, 0.3, 3.0)
    result_lin = sw_lin * cv2.resize(ratio, (w, h), interpolation=cv2.INTER_LINEAR)

    # Back to gamma
    result = _to_gamma_u8(result_lin)

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
    Sum-of-Gaussians SSS approximation.
    Removes "plastic" skin look: red scatters furthest, blue barely at all.
    Linearizes input (gamma→linear) before blurring for physically correct scatter.
    The profile is in millimetres; sigma = sqrt(variance) scaled to crop pixels.
    """
    w = face_bgr.shape[1]
    px_per_mm = w / _SSS_CROP_WIDTH_MM
    # Linearize: SSS is a linear-light phenomenon
    face_f = _TO_LINEAR[face_bgr]
    sss_result = np.zeros_like(face_f)

    for variance_mm2, (wr, wg, wb) in _SSS_PROFILE:
        sigma_px = float(np.sqrt(variance_mm2)) * px_per_mm
        # Below half a pixel the lobe is narrower than the sampling grid.
        layer = face_f if sigma_px < 0.5 else cv2.GaussianBlur(face_f, (0, 0), sigma_px)
        sss_result += layer * np.array([wb, wg, wr], dtype=np.float32)

    mask_3ch = cv2.GaussianBlur(
        skin_mask.astype(np.float32) / 255.0, (11, 11), 3.0
    )[..., np.newaxis]
    out_lin = face_f + strength * mask_3ch * (sss_result - face_f)
    # Delinearize back to gamma
    return _to_gamma_u8(out_lin)


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


def match_camera_degradation(
    swapped: np.ndarray,
    target: np.ndarray,
    skin_mask: np.ndarray | None = None,
    flow_vec: np.ndarray | None = None,
    noise_alpha: float = 0.7,
    sharp_alpha: float = 0.5,
    blur_alpha: float = 0.6,
) -> np.ndarray:
    """
    Match camera fingerprint of target to swapped face crop (last crop-space step).
    - Noise: estimate σ via MAD from target high-pass on flat skin; add synthetic grain.
    - Sharpness: equalize Laplacian variance between crops.
    - Motion blur: directional line kernel when |flow_vec| > 1.5 px.
    """
    h, w = swapped.shape[:2]
    result = swapped.astype(np.float32)
    target_f = target.astype(np.float32)

    # Flat-skin mask: used as the region for noise/sharpness statistics.
    tgt_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)
    flat = (np.abs(cv2.Laplacian(tgt_gray, cv2.CV_32F)) < 8.0).astype(np.float32)
    if skin_mask is not None:
        flat *= (skin_mask > 127).astype(np.float32)
    flat_mask = flat > 0

    # --- Noise matching ---
    hp = target_f - cv2.GaussianBlur(target_f, (0, 0), 1.5)
    noise_sigma = np.zeros(3, dtype=np.float32)
    if flat_mask.sum() > 100:
        for ch in range(3):
            mad = float(np.median(np.abs(hp[..., ch][flat_mask])))
            noise_sigma[ch] = float(np.clip(mad / 0.6745, 0, 15))
    grain_raw = np.random.default_rng().standard_normal((h, w, 3)).astype(np.float32)
    grain = cv2.GaussianBlur(grain_raw, (0, 0), 0.6) * noise_sigma[np.newaxis, np.newaxis, :]
    with_noise = cv2.GaussianBlur(result, (0, 0), 0.8) + grain
    result = result * (1.0 - noise_alpha) + with_noise * noise_alpha

    # --- Sharpness matching ---
    if flat_mask.sum() > 100:
        lap_sw = cv2.Laplacian(cv2.cvtColor(swapped, cv2.COLOR_BGR2GRAY).astype(np.float32),
                               cv2.CV_32F)
        lap_tg = cv2.Laplacian(tgt_gray, cv2.CV_32F)
        var_sw = float(lap_sw[flat_mask].var()) + 1e-6
        var_tg = float(lap_tg[flat_mask].var()) + 1e-6
        ratio = float(np.sqrt(var_tg / var_sw))
        if ratio < 0.9:
            sigma_b = min(2.0, (1.0 - ratio) * 3.0)
            softer = cv2.GaussianBlur(result, (0, 0), sigma_b)
            result = result * (1.0 - sharp_alpha) + softer * sharp_alpha
        elif ratio > 1.1:
            blurred = cv2.GaussianBlur(result, (0, 0), 1.0)
            sharpened = result + (result - blurred) * min(ratio - 1.0, 1.0)
            result = result * (1.0 - sharp_alpha) + sharpened * sharp_alpha

    # --- Motion blur matching ---
    if flow_vec is not None:
        flow_norm = float(np.linalg.norm(flow_vec))
        if flow_norm > 1.5:
            ksize = max(3, min(int(round(flow_norm)), 25)) | 1  # ensure odd
            half = ksize // 2
            kernel = np.zeros((ksize, ksize), dtype=np.float32)
            ang = float(np.arctan2(float(flow_vec[1]), float(flow_vec[0])))
            dx, dy = np.cos(ang), np.sin(ang)
            for i in range(-half, half + 1):
                xi, yi = int(round(half + dx * i)), int(round(half + dy * i))
                if 0 <= xi < ksize and 0 <= yi < ksize:
                    kernel[yi, xi] += 1.0
            ksum = kernel.sum()
            if ksum > 0:
                blurred_m = cv2.filter2D(result, cv2.CV_32F, kernel / ksum)
                result = result * (1.0 - blur_alpha) + blurred_m * blur_alpha

    return np.clip(result, 0, 255).astype(np.uint8)


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


def pct_harmonize(swapped: np.ndarray, target: np.ndarray,
                  mask: np.ndarray | None = None,
                  grid: int = 8) -> np.ndarray:
    """
    PCT-Net inspired spatially-varying color harmonization (Priority 4c).

    Fits a per-cell affine transform in CIE-Lab space at coarse resolution
    (grid × grid cells over a 64px thumbnail), then upsamples the per-pixel
    transform map to the full image size with bilinear interpolation.

    Handles directional lighting differences that global Reinhard misses
    (e.g. left/right illumination asymmetry).

    Args:
        swapped: (512, 512, 3) BGR uint8 — the swapped face crop.
        target:  (512, 512, 3) BGR uint8 — reference camera frame crop.
        mask:    (512, 512) uint8 optional skin mask; only masked pixels
                 contribute to the affine fits (reduces non-skin contamination).
        grid:    number of cells per dimension (default 8 → 8×8=64 cells).
    Returns:
        (H, W, 3) BGR uint8 with spatially-corrected colours.
    """
    h, w = swapped.shape[:2]
    thumb = 64  # coarse resolution for fitting

    sw_t  = cv2.resize(swapped, (thumb, thumb)).astype(np.float32)
    tgt_t = cv2.resize(target,  (thumb, thumb)).astype(np.float32)

    # Convert to Lab (float32, [0,255] → Lab)
    sw_lab  = cv2.cvtColor(sw_t.astype(np.uint8),  cv2.COLOR_BGR2Lab).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt_t.astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.float32)

    if mask is not None:
        mask_t = cv2.resize(mask, (thumb, thumb)).astype(np.float32) / 255.0
    else:
        mask_t = np.ones((thumb, thumb), np.float32)

    cell_h = thumb // grid
    cell_w = thumb // grid

    # Per-cell affine maps: shift_L, scale_a, shift_a, scale_b, shift_b
    # We build separate maps for each Lab channel
    # Simple: match mean and std per cell (affine = scale + shift)
    shift_L = np.zeros((grid, grid), np.float32)
    scale_a = np.ones((grid,  grid), np.float32)
    shift_a = np.zeros((grid, grid), np.float32)
    scale_b = np.ones((grid,  grid), np.float32)
    shift_b = np.zeros((grid, grid), np.float32)

    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = gy * cell_h, (gy + 1) * cell_h
            x0, x1 = gx * cell_w, (gx + 1) * cell_w
            m = mask_t[y0:y1, x0:x1].ravel()
            if m.sum() < 4:
                continue

            def _fit(src_ch, tgt_ch):
                s = src_ch[y0:y1, x0:x1].ravel()
                t = tgt_ch[y0:y1, x0:x1].ravel()
                mu_s = float(np.average(s, weights=m))
                mu_t = float(np.average(t, weights=m))
                std_s = float(np.sqrt(np.average((s - mu_s) ** 2, weights=m))) + 1e-6
                std_t = float(np.sqrt(np.average((t - mu_t) ** 2, weights=m))) + 1e-6
                sc = float(np.clip(std_t / std_s, 0.5, 2.0))
                sh = mu_t - sc * mu_s
                return sc, sh

            shift_L[gy, gx]        = float(np.average(
                tgt_lab[y0:y1, x0:x1, 0].ravel() - sw_lab[y0:y1, x0:x1, 0].ravel(),
                weights=m))
            scale_a[gy, gx], shift_a[gy, gx] = _fit(sw_lab[..., 1], tgt_lab[..., 1])
            scale_b[gy, gx], shift_b[gy, gx] = _fit(sw_lab[..., 2], tgt_lab[..., 2])

    # Upsample transform maps to full resolution
    def _up(arr):
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)

    sL = _up(shift_L)
    sa, sha = _up(scale_a), _up(shift_a)
    sb, shb = _up(scale_b), _up(shift_b)

    # Apply spatially-varying transform in Lab space.
    # OpenCV's uint8 Lab stores L, a and b all in 0..255 (a/b offset by 128).
    sw_full_lab = cv2.cvtColor(swapped, cv2.COLOR_BGR2Lab).astype(np.float32)
    out = sw_full_lab.copy()
    out[..., 0] = np.clip(sw_full_lab[..., 0] + sL, 0, 255)
    out[..., 1] = np.clip(sw_full_lab[..., 1] * sa + sha, 0, 255)
    out[..., 2] = np.clip(sw_full_lab[..., 2] * sb + shb, 0, 255)

    result = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_Lab2BGR)

    # Soft blend: apply only in masked skin area to avoid artefacts on boundaries
    if mask is not None:
        alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (21, 21), 5.0)[..., np.newaxis]
        result = np.clip(
            result.astype(np.float32) * alpha + swapped.astype(np.float32) * (1 - alpha),
            0, 255,
        ).astype(np.uint8)

    return result
