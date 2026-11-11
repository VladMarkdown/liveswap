"""
Temporal consistency: One-Euro filter + EAR blink gate + motion-adaptive EMA.

One-Euro filter applied to similarity transform parameters (NOT raw landmarks).
Separate filter instance per parameter. Filter on normalized params (tx/W, ty/H).

FaceParamSmoother: preferred over TransformSmoother for the main pipeline.
Filters (cx, cy, log_scale, rotation) extracted from raw 5-pt landmarks and
reconstructs both M_swap-256 and M_ffhq-512 from the same filtered geometry,
ensuring rotation-translation coupling across the two transforms.

EAR blink gate: bypass temporal EMA during blinks to preserve natural eye closure.
NOTE: requires 6-pt eye landmarks; disabled in MVP (detector provides 5-pt only).

Motion-adaptive EMA: blend previous output warped by optical flow.
"""
from __future__ import annotations
import math
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# One-Euro Filter
# ---------------------------------------------------------------------------

class OneEuroFilter:
    """
    One-Euro filter for a single scalar value.

    Recommended params for face swap transform at 30fps:
      translation (tx/W, ty/H): min_cutoff=2.0, beta=8.0
      scale (log):              min_cutoff=1.5, beta=4.0
      rotation (radians):       min_cutoff=2.0, beta=2.0
      d_cutoff for all: 1.0
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0,
                 d_cutoff: float = 1.0):
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, dt: float = 1.0 / 30.0) -> float:
        if self._x_prev is None:
            self._x_prev = x
            return x

        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self._d_cutoff, dt)
        self._dx_prev = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self._min_cutoff + self._beta * abs(self._dx_prev)
        a = self._alpha(cutoff, dt)
        x_filtered = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_filtered
        return x_filtered

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0


class TransformSmoother:
    """
    Smooths similarity transform parameters using per-param One-Euro filters.
    Parameters: tx (normalized by width), ty (normalized by height), log_scale, rotation (rad).

    Prefer FaceParamSmoother for new code — it filters in landmark space and
    guarantees coupling between rotation and translation.
    """

    def __init__(self):
        self._filters = {
            "tx":    OneEuroFilter(min_cutoff=2.0, beta=8.0, d_cutoff=1.0),
            "ty":    OneEuroFilter(min_cutoff=2.0, beta=8.0, d_cutoff=1.0),
            "scale": OneEuroFilter(min_cutoff=1.5, beta=4.0, d_cutoff=1.0),
            "rot":   OneEuroFilter(min_cutoff=2.0, beta=2.0, d_cutoff=1.0),
        }

    def smooth(self, M: np.ndarray, frame_width: int,
               frame_height: int | None = None,
               dt: float = 1.0 / 30.0) -> np.ndarray:
        """
        Smooth a 2×3 affine matrix M.
        Returns smoothed 2×3 affine matrix.
        frame_height defaults to frame_width when omitted (square assumption).
        """
        _fh = frame_height if frame_height is not None else frame_width
        tx = M[0, 2] / frame_width
        ty = M[1, 2] / _fh          # Fixed: normalize by height, not width
        scale = math.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
        rot = math.atan2(M[1, 0], M[0, 0])

        tx_s    = self._filters["tx"](tx, dt)
        ty_s    = self._filters["ty"](ty, dt)
        scale_s = math.exp(self._filters["scale"](math.log(max(scale, 1e-6)), dt))
        rot_s   = self._filters["rot"](rot, dt)

        M_s = np.array([
            [scale_s * math.cos(rot_s), -scale_s * math.sin(rot_s), tx_s * frame_width],
            [scale_s * math.sin(rot_s),  scale_s * math.cos(rot_s), ty_s * _fh],
        ], dtype=np.float32)
        return M_s

    def reset(self) -> None:
        for f in self._filters.values():
            f.reset()


# ---------------------------------------------------------------------------
# Face-Space Landmark Smoother (preferred over TransformSmoother)
# ---------------------------------------------------------------------------

# ArcFace 112 reference — local copy to avoid circular import with aligner.py
_KPS_TEMPLATE_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float64)


class FaceParamSmoother:
    """
    Filter face motion as (cx, cy, log_scale, rotation) extracted from raw
    5-point landmarks, then reconstruct filtered landmarks.  Callers derive
    M_swap and M_ffhq via estimate_transform(filtered_kps, size), guaranteeing
    rotation-translation coupling across both transforms.
    """

    def __init__(self):
        self._filters = {
            "cx":    OneEuroFilter(min_cutoff=2.0, beta=8.0, d_cutoff=1.0),
            "cy":    OneEuroFilter(min_cutoff=2.0, beta=8.0, d_cutoff=1.0),
            "scale": OneEuroFilter(min_cutoff=1.5, beta=4.0, d_cutoff=1.0),
            "rot":   OneEuroFilter(min_cutoff=2.0, beta=2.0, d_cutoff=1.0),
        }
        # Template: ArcFace 112 ref centered and unit-scaled on inter-eye distance.
        # template_norm.mean() == 0, norm(template_norm[1]-template_norm[0]) == 1.
        tmpl = _KPS_TEMPLATE_112.copy()
        self._tmpl_ieye = float(np.linalg.norm(tmpl[1] - tmpl[0]))
        tmpl_centered = tmpl - tmpl.mean(axis=0)
        self._tmpl_norm = tmpl_centered / self._tmpl_ieye

    def smooth(self, kps: np.ndarray, dt: float = 1.0 / 30.0) -> np.ndarray:
        """
        Filter 5-point landmarks and return filtered (5, 2) float32 array.
        Pass result to estimate_transform(filtered_kps, size) for both transforms.
        """
        pts = np.asarray(kps, dtype=np.float64).reshape(5, 2)
        cx, cy = pts.mean(axis=0)
        eye_vec = pts[1] - pts[0]
        scale = float(np.linalg.norm(eye_vec))
        rot = float(math.atan2(eye_vec[1], eye_vec[0]))

        cx_s    = self._filters["cx"](cx, dt)
        cy_s    = self._filters["cy"](cy, dt)
        scale_s = math.exp(self._filters["scale"](math.log(max(scale, 1e-6)), dt))
        rot_s   = self._filters["rot"](rot, dt)

        c, s = math.cos(rot_s), math.sin(rot_s)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        filtered = (R @ (self._tmpl_norm.T * scale_s)).T + np.array([cx_s, cy_s])
        return filtered.astype(np.float32)

    def reset(self) -> None:
        for f in self._filters.values():
            f.reset()


# ---------------------------------------------------------------------------
# EAR Blink Gate (requires 6-pt eye landmarks; disabled in MVP)
# ---------------------------------------------------------------------------

def eye_aspect_ratio(landmarks_6: np.ndarray) -> float:
    """
    Compute EAR for one eye from 6 ordered eye landmarks.
    landmarks_6: (6, 2) array — left corner, top-left, top-right, right corner, bottom-right, bottom-left
    closed: ear < 0.60 × median_open_ear
    reopened: ear > 0.75 × median_open_ear
    """
    p = landmarks_6
    h1 = np.linalg.norm(p[1] - p[5])
    h2 = np.linalg.norm(p[2] - p[4])
    w  = np.linalg.norm(p[0] - p[3])
    return (h1 + h2) / (2.0 * max(w, 1e-6))


class BlinkGate:
    """
    Detects eye blinks and gates temporal EMA.
    During blink: bypass EMA (apply swap directly).
    After reopen: resume EMA.
    """

    def __init__(self, closed_ratio: float = 0.60, open_ratio: float = 0.75,
                 history_len: int = 30):
        self._closed_ratio = closed_ratio
        self._open_ratio = open_ratio
        self._ear_history: list[float] = []
        self._history_len = history_len
        self._is_blinking = False

    def _median_open_ear(self) -> float:
        if len(self._ear_history) < 5:
            return 0.3  # default open EAR
        return float(np.median(self._ear_history))

    def update(self, ear: float) -> bool:
        """
        Returns True if currently blinking (EMA should be bypassed).
        """
        med = self._median_open_ear()
        if not self._is_blinking and ear < self._closed_ratio * med:
            self._is_blinking = True
        elif self._is_blinking and ear > self._open_ratio * med:
            self._is_blinking = False

        if not self._is_blinking:
            self._ear_history.append(ear)
            if len(self._ear_history) > self._history_len:
                self._ear_history.pop(0)

        return self._is_blinking

    def reset(self) -> None:
        self._ear_history = []
        self._is_blinking = False


# ---------------------------------------------------------------------------
# Motion-Adaptive EMA
# ---------------------------------------------------------------------------

class TemporalEMA:
    """
    Motion-adaptive temporal EMA with optical flow warp.
    Warps previous output frame before blending to avoid ghosting.
    """

    def __init__(self, base_weight: float = 0.8, motion_threshold: float = 0.02):
        self._base_weight = base_weight
        self._motion_threshold = motion_threshold
        self._prev_frame: np.ndarray | None = None
        self._prev_gray: np.ndarray | None = None

    def blend(self, current: np.ndarray, gray: np.ndarray,
              face_bbox, is_blinking: bool = False,
              face_mask: np.ndarray | None = None) -> np.ndarray:
        """Backward-flow history sampling confined to the current face mask."""
        if (self._prev_frame is None or is_blinking or
                self._prev_frame.shape != current.shape):
            self._prev_frame = current.copy()
            self._prev_gray = gray.copy()
            return current
        h, w = gray.shape
        if face_mask is None:
            face_mask = np.zeros((h, w), np.float32)
            x1, y1, x2, y2 = map(int, face_bbox)
            x1, x2 = np.clip([x1, x2], 0, w)
            y1, y2 = np.clip([y1, y2], 0, h)
            face_mask[y1:y2, x1:x2] = 1
        else:
            face_mask = np.clip(face_mask, 0, 1)
        ys, xs = np.nonzero(face_mask > 0)
        result = current.copy()
        if xs.size:
            x1, x2 = max(0, int(xs.min()) - 8), min(w, int(xs.max()) + 9)
            y1, y2 = max(0, int(ys.min()) - 8), min(h, int(ys.max()) + 9)
            roi = np.s_[y1:y2, x1:x2]
            # Remap needs current -> previous flow, not forward flow.
            flow = cv2.calcOpticalFlowFarneback(
                gray[roi], self._prev_gray[roi], None,
                pyr_scale=0.5, levels=2, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
            rh, rw = gray[roi].shape
            xx, yy = np.meshgrid(np.arange(rw, dtype=np.float32),
                                 np.arange(rh, dtype=np.float32))
            mx, my = xx + flow[..., 0], yy + flow[..., 1]
            history = cv2.remap(self._prev_frame[roi], mx, my, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
            old_gray = cv2.remap(self._prev_gray[roi], mx, my, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
            motion = np.linalg.norm(flow, axis=2) / max(rw, rh)
            weight = self._base_weight * (1 - np.clip(
                motion / self._motion_threshold, 0, 1))
            # Reject disoccluded pixels and changes not explained by motion.
            residual = np.abs(gray[roi].astype(np.float32) - old_gray.astype(np.float32))
            weight *= np.clip(1 - residual / 30.0, 0, 1)
            weight *= ((mx >= 0) & (mx < rw - 1) & (my >= 0) & (my < rh - 1))
            weight = (weight * face_mask[roi])[..., None]
            result[roi] = np.clip(history * weight + current[roi] * (1 - weight),
                                  0, 255).astype(np.uint8)
        self._prev_frame = result.copy()
        self._prev_gray = gray.copy()
        return result

    def get_face_flow(self, current_gray: np.ndarray, face_bbox) -> np.ndarray:
        """Return median optical flow vector over the face bounding box.
        Call BEFORE blend() — prev_gray still points to the previous frame.
        Returns zero vector if no history or bbox is degenerate."""
        if self._prev_gray is None or self._prev_gray.shape != current_gray.shape:
            return np.zeros(2, dtype=np.float32)
        h, w = current_gray.shape
        x1, y1 = max(0, int(face_bbox[0])), max(0, int(face_bbox[1]))
        x2, y2 = min(w, int(face_bbox[2])), min(h, int(face_bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return np.zeros(2, dtype=np.float32)
        flow = cv2.calcOpticalFlowFarneback(
            current_gray[y1:y2, x1:x2], self._prev_gray[y1:y2, x1:x2], None,
            pyr_scale=0.5, levels=2, winsize=15,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
        return np.median(flow.reshape(-1, 2), axis=0).astype(np.float32)

    def reset(self) -> None:
        self._prev_frame = None
        self._prev_gray = None
