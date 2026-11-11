"""Real-time 256px swap and FFHQ restoration with latest-frame mailboxes."""
from __future__ import annotations

from collections import deque
import logging
import queue
import threading
import time
from typing import Callable

import cv2
import numpy as np

from .detector import detect_largest, get_eye_landmarks_6pt
from .aligner import estimate_transform, paste_back, feather_coverage
from .color import (
    reinhard_color_transfer, low_freq_shading_correction, inject_skin_detail,
    apply_sss, transfer_specular, composite_mouth_interior, match_camera_degradation,
    pct_harmonize,
)
from .occluder import OcclusionDetector
from .temporal import FaceParamSmoother, TemporalEMA, BlinkGate, eye_aspect_ratio

logger = logging.getLogger(__name__)
STOP = object()

_GPEN_BLEND_MAX  = 0.8
_GPEN_BLEND_STEP = 0.08   # ramp over ~10 frames at enable/disable transitions


def put_latest(mailbox: queue.Queue, item) -> None:
    """Publish without blocking; discard stale work, never the new frame."""
    while True:
        try:
            mailbox.put_nowait(item)
            return
        except queue.Full:
            try:
                mailbox.get_nowait()
            except queue.Empty:
                pass


class StageTimer:
    """Rolling wall-clock time per pipeline stage, in milliseconds."""
    def __init__(self, window: int = 60):
        self._window = window
        self._times = {}
        self._last = 0.0

    def start(self) -> None:
        self._last = time.perf_counter()

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        if name not in self._times:
            self._times[name] = deque(maxlen=self._window)
        self._times[name].append((now - self._last) * 1000)
        self._last = now

    def averages(self) -> dict:
        return {name: sum(v) / len(v) for name, v in self._times.items()}


def generate_profile_source_face(profile):
    """Run diffusion once at session setup, never in the frame loop."""
    from ..profile.synthesizer import generate_profile_source_face as generate
    return generate(profile)


class SwapPipeline:
    def __init__(self, swapper, enhancer, parser, source_face=None,
                 quality: str = 'ultra', frame_budget_ms: float = 33.0,
                 pixel_boost: bool = False, use_pct: bool = False,
                 use_occluder: bool = False):
        if frame_budget_ms <= 0:
            raise ValueError('frame_budget_ms must be positive')
        self._swapper = swapper
        self._enhancer = enhancer
        self._parser = parser
        self._source_face = source_face
        self._quality = quality
        self.frame_budget_ms = frame_budget_ms
        self._pixel_boost = pixel_boost
        self._use_pct = use_pct
        self._face_smoother = FaceParamSmoother()
        self._ema = TemporalEMA(base_weight=0.55)
        self._inference_times = deque(maxlen=30)
        self._base_times = deque(maxlen=30)
        self._gpen_times = deque(maxlen=10)
        self._timer = StageTimer()
        self._timed_frames = 0
        self.last_inference_ms = 0.0
        self.gpen_skipped = 0
        self._last_frame_time = None
        # GPEN hysteresis state: prevent 15 Hz flicker at budget boundary
        self._enhance_enabled: bool = True
        self._gpen_blend: float = 0.0
        # Specular mask cache for temporal EMA (fix #9)
        self._spec_mask_prev: np.ndarray | None = None
        # Blink gate: requires 2d106det landmarks (available via buffalo_l pack)
        self._blink_gate = BlinkGate()
        # Optional occlusion detector
        self._occluder: OcclusionDetector | None = OcclusionDetector() if use_occluder else None
        # Optional WebRTC streamer (set via set_webrtc_streamer)
        self._webrtc = None

    @property
    def average_inference_ms(self) -> float:
        return sum(self._inference_times) / max(1, len(self._inference_times))

    @property
    def stage_ms(self) -> dict:
        """Rolling average milliseconds per stage, in pipeline order."""
        return self._timer.averages()

    def _reset_tracking(self):
        self._face_smoother.reset()
        self._ema.reset()
        self._parser.reset()
        self._blink_gate.reset()
        self._last_frame_time = None
        self._enhance_enabled = True
        self._gpen_blend = 0.0
        self._spec_mask_prev = None

    def set_webrtc_streamer(self, streamer) -> None:
        self._webrtc = streamer

    def set_source_face(self, face) -> None:
        self._source_face = face
        self._reset_tracking()

    def set_profile(self, profile) -> None:
        if isinstance(profile, np.ndarray):
            source = _EmbeddingFace(profile)
        elif profile.lora_path is not None:
            source = generate_profile_source_face(profile)
        else:
            source = _EmbeddingFace(profile.embedding)
        self.set_source_face(source)

    def _should_enhance(self, elapsed_ms: float) -> bool:
        if self._enhancer is None:
            return False
        base = sum(self._base_times) / max(1, len(self._base_times))
        gpen = sum(self._gpen_times) / max(1, len(self._gpen_times))
        estimated = max(base, elapsed_ms) + gpen
        # Hysteresis: disable when over budget, re-enable only at 85 % of budget.
        # Eliminates the alternating on/off pattern that caused ~15 Hz texture flicker.
        if estimated > self.frame_budget_ms:
            self._enhance_enabled = False
        elif estimated < self.frame_budget_ms * 0.85:
            self._enhance_enabled = True
        return self._enhance_enabled

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._source_face is None:
            return frame_bgr
        started = time.perf_counter()
        timer = self._timer
        timer.start()
        target = detect_largest(frame_bgr)
        timer.mark('detect')
        if target is None:
            self._reset_tracking()
            return frame_bgr
        dt = 1 / 30 if self._last_frame_time is None else max(1e-3, started - self._last_frame_time)
        self._last_frame_time = started
        _, width = frame_bgr.shape[:2]

        # Single smoother filters face params from raw landmarks, then derives
        # both transforms from the same filtered geometry (fix #11: coupling).
        filtered_kps = self._face_smoother.smooth(target.kps, dt)
        M_swap = estimate_transform(filtered_kps, 256)
        M_ffhq = estimate_transform(filtered_kps, 512)

        # Extract only AFTER smoothing; these exact transforms also define paste-back.
        crop_swap = cv2.warpAffine(frame_bgr, M_swap, (256, 256), borderMode=cv2.BORDER_REFLECT)
        crop_ffhq = cv2.warpAffine(frame_bgr, M_ffhq, (512, 512), borderMode=cv2.BORDER_REFLECT)
        timer.mark('align')
        if self._pixel_boost:
            # Pixel-boost: swap at 512px resolution via 4-patch tiling
            aligned = self._swapper.swap(crop_ffhq, self._source_face, pixel_boost=True)
        else:
            swapped = self._swapper.swap(crop_swap, self._source_face)
            swap_h = np.vstack((M_swap, [0, 0, 1]))
            ffhq_h = np.vstack((M_ffhq, [0, 0, 1]))
            to_ffhq = (ffhq_h @ np.linalg.inv(swap_h))[:2]
            aligned = cv2.warpAffine(swapped, to_ffhq, (512, 512), borderMode=cv2.BORDER_REFLECT)
        # Feathered coverage mask prevents hard boundary at the swap crop edge (fix #4).
        coverage_raw = cv2.warpAffine(np.ones((256, 256), np.float32), to_ffhq, (512, 512))
        coverage = feather_coverage(coverage_raw)
        timer.mark('swap')
        gpen_ms = 0.0
        if self._should_enhance((time.perf_counter() - started) * 1000):
            # Ramp blend weight in over ~10 frames to avoid a pop at enable (fix #2).
            self._gpen_blend = min(_GPEN_BLEND_MAX, self._gpen_blend + _GPEN_BLEND_STEP)
            before = time.perf_counter()
            enhanced = self._enhancer.enhance_with_blend(aligned, blend=self._gpen_blend)
            gpen_ms = (time.perf_counter() - before) * 1000
            self._gpen_times.append(gpen_ms)
        else:
            self._gpen_blend = max(0.0, self._gpen_blend - _GPEN_BLEND_STEP)
            enhanced = aligned
            self.gpen_skipped += 1
        timer.mark('gpen')
        # Ownership masks come from the target: restoration may erase glasses
        # or change the mouth cavity before segmentation could preserve them.
        seg = self._parser.parse(crop_ffhq)
        swap_mask = self._parser.build_swap_mask(seg, erosion_px=3, transition_px=10) * coverage
        skin = self._parser.get_skin_mask(seg)
        mouth = self._parser.get_mouth_interior_mask(seg)
        # Occlusion: subtract occluded regions from swap mask so original face shows through
        if self._occluder is not None:
            occ = self._occluder.get_occlusion_mask(crop_ffhq, bisenet_seg=seg)
            swap_mask = swap_mask * (1.0 - occ)
        mask = swap_mask
        timer.mark('parse')
        # Reinhard correction now applies only within the skin mask (fix C).
        result = reinhard_color_transfer(enhanced, crop_ffhq, mask=skin)
        # PCT-Net spatial color harmonization: handles directional lighting (Priority 4c).
        # Reinhard strength reduced to 0.5 implicitly since PCT refines it.
        if self._use_pct:
            result = pct_harmonize(result, crop_ffhq, mask=skin)
        # Multiplicative shading correction in linear space, skin mask only (fix #7).
        timer.mark('color')
        result = low_freq_shading_correction(result, crop_ffhq, sigma=20.0, skin_mask=skin)
        timer.mark('shading')
        if self._quality == 'ultra':
            face_width = max(1, int(target.bbox[2] - target.bbox[0]))
            result = inject_skin_detail(result, crop_ffhq, skin, face_width, strength=0.20)
            timer.mark('detail')
            result = apply_sss(result, skin, strength=0.20)
            timer.mark('sss')
            # Use observed target detail instead of adding procedural skin texture.
            # transfer_specular returns (result, spec_mask) for temporal EMA (fix #9).
            result, self._spec_mask_prev = transfer_specular(
                crop_ffhq, result, skin, prev_spec=self._spec_mask_prev)
            timer.mark('specular')
        result = composite_mouth_interior(result, crop_ffhq, mouth)
        timer.mark('mouth')
        # Camera degradation: match noise grain, sharpness, and motion blur of the
        # target camera feed. Applied before paste_back (last crop-space quality step).
        # flow_vec comes from TemporalEMA.get_face_flow — reuses its prev_gray store.
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        flow_vec = self._ema.get_face_flow(gray, target.bbox)
        timer.mark('flow')
        result = match_camera_degradation(result, crop_ffhq, skin, flow_vec)
        timer.mark('degrade')
        full, frame_mask = paste_back(frame_bgr, result, M_ffhq, mask=mask, return_mask=True)
        timer.mark('paste')
        # BlinkGate: use 2d106det landmarks from buffalo_l; falls back gracefully to
        # is_blinking=False when landmark_2d_106 is None (e.g. on first frame).
        is_blinking = False
        left_6, right_6 = get_eye_landmarks_6pt(target)
        if left_6 is not None:
            ear = (eye_aspect_ratio(left_6) + eye_aspect_ratio(right_6)) * 0.5
            is_blinking = self._blink_gate.update(ear)
        dynamic = self._parser.get_region_mask(seg, [4, 5, 11, 12, 13])
        dynamic = cv2.warpAffine(dynamic, cv2.invertAffineTransform(M_ffhq), (width, frame_bgr.shape[0]))
        temporal_mask = frame_mask * (1.0 - dynamic.astype(np.float32) / 255.0)
        full = self._ema.blend(full, gray, target.bbox, is_blinking=is_blinking,
                               face_mask=temporal_mask)
        timer.mark('temporal')
        self._timed_frames += 1
        if self._timed_frames % 150 == 0:
            logger.info('Stage ms: %s', ' '.join(
                f'{name}={ms:.1f}' for name, ms in self.stage_ms.items()))
        self.last_inference_ms = (time.perf_counter() - started) * 1000
        self._inference_times.append(self.last_inference_ms)
        self._base_times.append(max(0.0, self.last_inference_ms - gpen_ms))
        # WebRTC phone output — non-blocking push (Priority 5)
        if self._webrtc is not None and self._webrtc.is_active():
            self._webrtc.send_frame(full)
        return full


class _EmbeddingFace:
    def __init__(self, embedding: np.ndarray):
        self.embedding = np.asarray(embedding, dtype=np.float32).reshape(512).copy()
        norm = np.linalg.norm(self.embedding)
        if not np.isfinite(self.embedding).all() or norm <= 0:
            raise ValueError('Identity embedding must be finite and nonzero')
        self.normed_embedding = self.embedding / norm


class AsyncPipelineRunner:
    """Capture, resize, inference and output workers with bounded latency."""
    def __init__(self, pipeline: SwapPipeline, output_callback: Callable,
                 target_width: int = 1280, target_height: int = 720):
        self._pipeline = pipeline
        self._callback = output_callback
        self._target_w, self._target_h = target_width, target_height
        self._raw_q = queue.Queue(maxsize=1)
        self._pre_q = queue.Queue(maxsize=1)
        self._post_q = queue.Queue(maxsize=1)
        self._threads = []
        self._stop = threading.Event()
        self._cap = None

    def start(self, cap) -> None:
        if any(t.is_alive() for t in self._threads):
            raise RuntimeError('Pipeline already running')
        self._stop.clear()
        for mailbox in (self._raw_q, self._pre_q, self._post_q):
            while not mailbox.empty():
                mailbox.get_nowait()
        self._cap = cap
        self._threads = [threading.Thread(target=fn, daemon=True) for fn in (
            self._capture_thread, self._preprocess_thread, self._infer_thread, self._output_thread)]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._cap is not None:
            self._cap.release()

    def join(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0, deadline - time.monotonic()))
        if any(t.is_alive() for t in self._threads):
            raise TimeoutError('Pipeline worker has not stopped')

    def _take(self, mailbox):
        while not self._stop.is_set():
            try:
                return mailbox.get(timeout=0.1)
            except queue.Empty:
                continue
        return STOP

    def _capture_thread(self):
        try:
            while not self._stop.is_set():
                ret, frame = self._cap.read()
                if not ret:
                    break
                put_latest(self._raw_q, frame)
        except Exception:
            logger.exception('Capture failed')
        finally:
            put_latest(self._raw_q, STOP)

    def _preprocess_thread(self):
        try:
            while True:
                item = self._take(self._raw_q)
                if item is STOP:
                    break
                if item.shape[:2] != (self._target_h, self._target_w):
                    item = cv2.resize(item, (self._target_w, self._target_h))
                put_latest(self._pre_q, item)
        except Exception:
            logger.exception('Preprocessing failed')
        finally:
            put_latest(self._pre_q, STOP)

    def _infer_thread(self):
        try:
            while True:
                item = self._take(self._pre_q)
                if item is STOP:
                    break
                try:
                    result = self._pipeline.process_frame(item)
                except Exception:
                    logger.exception('Frame processing failed')
                    result = item
                put_latest(self._post_q, result)
        finally:
            put_latest(self._post_q, STOP)

    def _output_thread(self):
        while True:
            frame = self._take(self._post_q)
            if frame is STOP:
                return
            try:
                self._callback(frame)
            except Exception:
                logger.exception('Output failed')
                self._stop.set()
                return
