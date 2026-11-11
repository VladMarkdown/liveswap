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

from .detector import detect_tracked, get_eye_landmarks
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
_GPEN_PROBE_FRAMES = 90   # while disabled, re-measure restoration cost this often
_MAX_HELD_FRAMES = 5      # reuse the last face geometry across this many missed detections


def fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Center-crop to the target aspect ratio, then resize; never stretches."""
    h, w = frame.shape[:2]
    if (h, w) == (height, width):
        return frame
    crop_w = min(w, round(h * width / height))
    crop_h = min(h, round(w * height / width))
    x0, y0 = (w - crop_w) // 2, (h - crop_h) // 2
    return cv2.resize(frame[y0:y0 + crop_h, x0:x0 + crop_w], (width, height))


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
        # Serializes process_frame against set_source_face from the GUI thread.
        self._lock = threading.Lock()
        # Last detected face, reused briefly when detection drops out.
        self._held_target = None
        self._missed = 0
        if enhancer is not None:
            # The first CUDA call is far slower than steady state; keep it out
            # of the restoration budget average.
            enhancer.enhance(np.zeros((512, 512, 3), np.uint8))

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
        self._held_target = None
        self._missed = 0

    def set_webrtc_streamer(self, streamer) -> None:
        self._webrtc = streamer

    def set_source_face(self, face) -> None:
        with self._lock:
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
        with self._lock:
            return self._process_frame(frame_bgr)

    def _process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._source_face is None:
            return frame_bgr
        started = time.perf_counter()
        timer = self._timer
        timer.start()
        target = detect_tracked(
            frame_bgr, None if self._held_target is None else self._held_target.bbox)
        timer.mark('detect')
        if target is None:
            # A brief dropout keeps the last geometry instead of flashing the
            # unprocessed frame and discarding all temporal state.
            self._missed += 1
            if self._held_target is None or self._missed > _MAX_HELD_FRAMES:
                self._reset_tracking()
                return frame_bgr
            target = self._held_target
        else:
            self._held_target = target
            self._missed = 0
        dt = 1 / 30 if self._last_frame_time is None else max(1e-3, started - self._last_frame_time)
        self._last_frame_time = started
        height, width = frame_bgr.shape[:2]

        # Single smoother filters face params from raw landmarks, then derives
        # both transforms from the same filtered geometry (fix #11: coupling).
        filtered_kps = self._face_smoother.smooth(target.kps, dt, (width, height))
        # Pixel boost swaps the same ArcFace-aligned crop at 512px (the swap
        # model's alignment, not FFHQ's); scaling the matrix scales the crop.
        swap_size = 512 if self._pixel_boost else 256
        M_swap = estimate_transform(filtered_kps, 256) * (swap_size / 256)
        M_ffhq = estimate_transform(filtered_kps, 512)

        # Extract only AFTER smoothing; these exact transforms also define paste-back.
        crop_swap = cv2.warpAffine(frame_bgr, M_swap, (swap_size, swap_size),
                                   borderMode=cv2.BORDER_REFLECT)
        crop_ffhq = cv2.warpAffine(frame_bgr, M_ffhq, (512, 512), borderMode=cv2.BORDER_REFLECT)
        timer.mark('align')
        swapped = self._swapper.swap(crop_swap, self._source_face,
                                     pixel_boost=self._pixel_boost)
        swap_h = np.vstack((M_swap, [0, 0, 1]))
        ffhq_h = np.vstack((M_ffhq, [0, 0, 1]))
        to_ffhq = (ffhq_h @ np.linalg.inv(swap_h))[:2]
        aligned = cv2.warpAffine(swapped, to_ffhq, (512, 512), borderMode=cv2.BORDER_REFLECT)
        # Feathered coverage mask prevents hard boundary at the swap crop edge (fix #4).
        coverage_raw = cv2.warpAffine(np.ones((swap_size, swap_size), np.float32),
                                      to_ffhq, (512, 512))
        coverage = feather_coverage(coverage_raw)
        timer.mark('swap')
        gpen_ms = 0.0
        # Ramp the blend weight over ~10 frames in both directions (fix #2).
        # Restoration keeps running while the weight ramps down, so disabling
        # it fades out instead of popping.
        wanted = self._should_enhance((time.perf_counter() - started) * 1000)
        step = _GPEN_BLEND_STEP if wanted else -_GPEN_BLEND_STEP
        self._gpen_blend = min(_GPEN_BLEND_MAX, max(0.0, self._gpen_blend + step))
        if self._enhancer is not None and self._gpen_blend > 1e-6:
            before = time.perf_counter()
            enhanced = self._enhancer.enhance_with_blend(aligned, blend=self._gpen_blend)
            gpen_ms = (time.perf_counter() - before) * 1000
            self._gpen_times.append(gpen_ms)
        else:
            enhanced = aligned
            self.gpen_skipped += 1
            if self.gpen_skipped % _GPEN_PROBE_FRAMES == 0:
                # Cost samples only arrive while restoration runs. Drop the
                # stale ones so the gate can re-enable and measure again.
                self._gpen_times.clear()
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
        left_eye, right_eye = get_eye_landmarks(target)
        if left_eye is not None:
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) * 0.5
            is_blinking = self._blink_gate.update(ear)
        dynamic = self._parser.get_region_mask(seg, [4, 5, 11, 12, 13])
        dynamic = cv2.warpAffine(dynamic, cv2.invertAffineTransform(M_ffhq), (width, height))
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
                put_latest(self._pre_q, fit_frame(item, self._target_w, self._target_h))
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
