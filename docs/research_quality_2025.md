# Skin Quality Techniques 2025-2026

## Key Finding: StreamingSwap (arXiv 2610.26681)
**Best real-time result found:** 20 FPS on RTX 4090, stable on 10-min videos.
No other diffusion method achieves >24fps. For real-time GAN-class still wins.

## Latest Models Comparison

| Paper | arXiv | Speed | Notes |
|---|---|---|---|
| **StreamingSwap** | 2610.26681 | **20 FPS (RTX 4090)** | Identity-aware supervision |
| AlphaFace | 2601.16429 (Jan 2026) | Real-time | Our primary model |
| DreamID | 2504.14509 (Apr 2025) | 0.6s/frame | ByteDance, fastest diffusion |
| GHOST 2.0 | 2502.18417 (Feb 2025) | Slow | Full HEAD swap, FID 29.57 |
| DreamID-V | 2601.01425 (Jan 2026) | Slow | First DiT-based video swap |
| VFace | 2602.07835 (Feb 2026) | — | WACV 2026 oral, training-free |

**Survey 2026:** arXiv 2605.00883 (CASIA) — evaluates 14 methods on 4K benchmark. Quality: diffusion > StyleGAN > GAN. For real-time: GAN wins.

---

## 1. Subsurface Scattering (SSS) — Jimenez Separable

Runtime: ~1-3ms CPU / **0.2ms GPU**

```python
import numpy as np
import cv2

SSS_SKIN_PROFILE = [
    (0.233, 0.0064), (0.100, 0.0484), (0.118, 0.1870),
    (0.113, 0.5670), (0.358, 1.9900), (0.078, 7.4100),
]

def apply_sss_approximation(face_bgr, skin_mask, strength=0.25):
    """
    Jimenez separable Gaussian sum. R scatters 2mm, G ~1mm, B ~0.3mm.
    strength: 0.15-0.25 for natural look.
    """
    h, w = face_bgr.shape[:2]
    face_f = face_bgr.astype(np.float32) / 255.0
    channel_scale = [1.0, 0.6, 0.3]  # R, G, B scatter distances
    sss_channels = []

    for ch in range(3):
        ch_layer = face_f[..., ch]
        sss_layer = np.zeros((h, w), dtype=np.float32)
        for weight, sigma_norm in SSS_SKIN_PROFILE:
            sigma_px = sigma_norm * w * 0.5 * channel_scale[ch]
            if sigma_px < 0.5:
                sss_layer += weight * ch_layer
                continue
            ksize = max(3, min(int(sigma_px * 6) | 1, 51))
            blurred = cv2.GaussianBlur(ch_layer, (ksize, 1), sigma_px)
            blurred = cv2.GaussianBlur(blurred, (1, ksize), sigma_px)
            sss_layer += weight * blurred
        sss_channels.append(sss_layer)

    sss_result = np.stack(sss_channels, axis=-1)
    mask_3ch = cv2.GaussianBlur(
        np.stack([skin_mask]*3, axis=-1).astype(np.float32) / 255.0,
        (11, 11), 3.0)
    out = face_f + strength * mask_3ch * (sss_result - face_f)
    return np.clip(out * 255, 0, 255).astype(np.uint8)
```

---

## 2. Specular Highlight Transfer — Dichromatic Model

Runtime: ~0.5ms

```python
def detect_specular_mask(face_bgr, thresh_sat=30, thresh_val=200):
    """Speculars = achromatic (high brightness, low saturation)."""
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    spec = ((hsv[...,1] < thresh_sat) & (hsv[...,2] > thresh_val)).astype(np.float32)
    return cv2.GaussianBlur(spec, (5, 5), 1.5)

def transfer_specular(source_face, swapped_face, skin_mask):
    """Paste specular highlights from target-frame face onto swapped result."""
    spec_mask = detect_specular_mask(source_face)
    skin_norm = cv2.GaussianBlur(
        skin_mask.astype(np.float32) / 255.0, (11, 11), 3.0)
    combined = (spec_mask * skin_norm)[..., np.newaxis]
    src_f = source_face.astype(np.float32)
    swap_f = swapped_face.astype(np.float32)
    return np.clip(swap_f*(1-combined) + src_f*combined, 0, 255).astype(np.uint8)
```

---

## 3. Micro-Texture (Pore Synthesis) — Cached Gabor Tile

Runtime: ~0.5ms (tile cached at startup)

```python
def generate_pore_texture_tile(size=256, seed=42):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((size, size)).astype(np.float32)
    pore_layer = np.zeros((size, size), dtype=np.float32)
    for theta in [0, 45, 90, 135]:
        kernel = cv2.getGaborKernel(
            (15, 15), sigma=2.0, theta=np.deg2rad(theta),
            lambd=8.0, gamma=0.8, psi=0)
        pore_layer += np.abs(cv2.filter2D(noise, cv2.CV_32F, kernel))
    pore_layer = (pore_layer - pore_layer.mean()) / (pore_layer.std() + 1e-6)
    return pore_layer

_PORE_TILE = generate_pore_texture_tile(256)  # cache at startup

def apply_micro_texture(face_bgr, skin_mask, strength=0.04):
    """
    Blend pore texture onto skin only (excludes eyes/lips via BiSeNet).
    strength: 0.02-0.06.
    """
    h, w = face_bgr.shape[:2]
    tile = cv2.resize(_PORE_TILE, (w, h))
    face_f = face_bgr.astype(np.float32)
    skin_norm = cv2.GaussianBlur(
        skin_mask.astype(np.float32) / 255.0, (11, 11), 3.0)[..., np.newaxis]
    displaced = face_f * (1.0 + strength * tile[..., np.newaxis])
    out = face_f + skin_norm * (displaced - face_f)
    return np.clip(out, 0, 255).astype(np.uint8)
```

---

## 4. Eye/Iris Preservation

**BiSeNet labels:** 4=l_eye, 5=r_eye (whole eye region, no separate iris label)

**Simplest approach:** Exclude labels 4/5 from swap mask entirely → source iris never replaced, no color mismatch.

**Better approach:** Warp source iris into swapped face position:

```python
def get_iris_mask(face_bgr, eye_mask_binary, shrink_factor=0.55):
    contours, _ = cv2.findContours(eye_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(eye_mask_binary)
    cnt = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    iris_mask = np.zeros_like(eye_mask_binary)
    cv2.circle(iris_mask, (int(cx), int(cy)), int(radius * shrink_factor), 255, -1)
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    dark_mask = (gray < 120).astype(np.uint8) * 255
    return cv2.bitwise_and(iris_mask, dark_mask)
```

---

## Implementation Priority

| Technique | Impact | Runtime | Priority |
|---|---|---|---|
| Iris composite (source iris warp) | Highest identity | ~1ms | P1 |
| Specular transfer | Removes "plastic" | ~0.5ms | P1 |
| SSS separable blur | Removes "flat" skin | 1-3ms CPU / 0.2ms GPU | P2 |
| Micro-texture tile | Adds pore detail | ~0.5ms | P2 |
| SH relighting (Deep3DFaceRecon ONNX) | Full illumination | ~4ms GPU | P3 |

---

## Sources
- StreamingSwap: https://arxiv.org/html/2610.26681
- Jimenez SSS: https://graphics.unizar.es/papers/Separable-Subsurface-Scattering.pdf
- SynthLight (relighting 2025): https://arxiv.org/html/2501.09756v1
- GHOST 2.0: https://arxiv.org/abs/2502.18417 / https://github.com/ai-forever/ghost-2.0
- CASIA survey 2026: https://arxiv.org/pdf/2605.00883
- FLEX skin micro SIGGRAPH 2025: https://icewired-yy.github.io/papers/face_micro_detail_synthesis/
