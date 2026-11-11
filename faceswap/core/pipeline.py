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

from .detector import detect_largest
from .aligner import estimate_transform, paste_back
from .color import (
    reinhard_color_transfer, low_freq_shading_correction, inject_skin_detail,
    apply_sss, transfer_specular, composite_mouth_interior,
)
from .temporal import TransformSmoother, TemporalEMA

logger = logging.getLogger(__name__)
STOP = object()


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


def generate_profile_source_face(profile):
    """Run diffusion once at session setup, never in the frame loop."""
    from ..profile.synthesizer import generate_profile_source_face as generate
    return generate(profile)


class SwapPipeline:
    def __init__(self, swapper, enhancer, parser, source_face=None,
                 quality: str = 'ultra', frame_budget_ms: float = 33.0):
        if frame_budget_ms <= 0:
            raise ValueError('frame_budget_ms must be positive')
        self._swapper = swapper
        self._enhancer = enhancer
        self._parser = parser
        self._source_face = source_face
        self._quality = quality
        self.frame_budget_ms = frame_budget_ms
        self._smoother = TransformSmoother()
        self._ffhq_smoother = TransformSmoother()
        self._ema = TemporalEMA(base_weight=0.55)
        self._inference_times = deque(maxlen=30)
        self._base_times = deque(maxlen=30)
        self._gpen_times = deque(maxlen=10)
        self.last_inference_ms = 0.0
        self.gpen_skipped = 0
        self._last_frame_time = None

    @property
    def average_inference_ms(self) -> float:
        return sum(self._inference_times) / max(1, len(self._inference_times))

    def _reset_tracking(self):
        self._smoother.reset()
        self._ffhq_smoother.reset()
        self._ema.reset()
        self._last_frame_time = None

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
        if self._enhancer is None or self.last_inference_ms > self.frame_budget_ms:
            return False
        # First frame measures restoration cost. Subsequent decisions include
        # the non-restoration rolling cost, allowing recovery after overload.
        base = sum(self._base_times) / max(1, len(self._base_times))
        gpen = sum(self._gpen_times) / max(1, len(self._gpen_times))
        return max(base, elapsed_ms) + gpen <= self.frame_budget_ms

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._source_face is None:
            return frame_bgr
        started = time.perf_counter()
        target = detect_largest(frame_bgr)
        if target is None:
            self._reset_tracking()
            return frame_bgr
        dt = 1 / 30 if self._last_frame_time is None else max(1e-3, started - self._last_frame_time)
        self._last_frame_time = started
        _, width = frame_bgr.shape[:2]
        M_swap = self._smoother.smooth(estimate_transform(target.kps, 256), width, dt)
        M_ffhq = self._ffhq_smoother.smooth(estimate_transform(target.kps, 512), width, dt)
        # Extract only AFTER smoothing; these exact transforms also define paste-back.
        crop_swap = cv2.warpAffine(frame_bgr, M_swap, (256, 256), borderMode=cv2.BORDER_REFLECT)
        crop_ffhq = cv2.warpAffine(frame_bgr, M_ffhq, (512, 512), borderMode=cv2.BORDER_REFLECT)
        swapped = self._swapper.swap(crop_swap, self._source_face)
        swap_h = np.vstack((M_swap, [0, 0, 1]))
        ffhq_h = np.vstack((M_ffhq, [0, 0, 1]))
        to_ffhq = (ffhq_h @ np.linalg.inv(swap_h))[:2]
        aligned = cv2.warpAffine(swapped, to_ffhq, (512, 512), borderMode=cv2.BORDER_REFLECT)
        # Never paste reflected pixels outside the actual swap crop's coverage.
        coverage = cv2.warpAffine(np.ones((256, 256), np.float32), to_ffhq, (512, 512))
        gpen_ms = 0.0
        if self._should_enhance((time.perf_counter() - started) * 1000):
            before = time.perf_counter()
            enhanced = self._enhancer.enhance_with_blend(aligned, blend=0.8)
            gpen_ms = (time.perf_counter() - before) * 1000
            self._gpen_times.append(gpen_ms)
        else:
            enhanced = aligned
            self.gpen_skipped += 1
        # Ownership masks come from the target: restoration may erase glasses
        # or change the mouth cavity before segmentation could preserve them.
        seg = self._parser.parse(crop_ffhq)
        mask = self._parser.build_swap_mask(seg, erosion_px=3, transition_px=10) * coverage
        skin = self._parser.get_skin_mask(seg)
        mouth = self._parser.get_mouth_interior_mask(seg)
        result = reinhard_color_transfer(enhanced, crop_ffhq, mask=skin)
        result = low_freq_shading_correction(result, crop_ffhq, sigma=20.0)
        if self._quality == 'ultra':
            face_width = max(1, int(target.bbox[2] - target.bbox[0]))
            result = inject_skin_detail(result, crop_ffhq, skin, face_width, strength=0.20)
            result = apply_sss(result, skin, strength=0.20)
            # Use observed target detail instead of adding procedural skin texture.
            result = transfer_specular(crop_ffhq, result, skin)
        result = composite_mouth_interior(result, crop_ffhq, mouth)
        full, frame_mask = paste_back(frame_bgr, result, M_ffhq, mask=mask, return_mask=True)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Five-point landmarks lack eyelid contours: blink detection is disabled
        # for MVP. Preserve current eye/mouth pixels instead of inventing EAR.
        dynamic = self._parser.get_region_mask(seg, [4, 5, 11, 12, 13])
        dynamic = cv2.warpAffine(dynamic, cv2.invertAffineTransform(M_ffhq), (width, frame_bgr.shape[0]))
        temporal_mask = frame_mask * (1.0 - dynamic.astype(np.float32) / 255.0)
        full = self._ema.blend(full, gray, target.bbox, face_mask=temporal_mask)
        self.last_inference_ms = (time.perf_counter() - started) * 1000
        self._inference_times.append(self.last_inference_ms)
        self._base_times.append(max(0.0, self.last_inference_ms - gpen_ms))
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
