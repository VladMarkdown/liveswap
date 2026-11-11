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


def stamp_ai_label(frame: np.ndarray) -> None:
    """Draw a visible "AI-GENERATED FACE" label onto a swapped frame, in place.

    Every frame the pipeline alters carries it, so the preview, the virtual
    camera and the WebRTC stream all show viewers that the face is synthetic.
    """
    h, w = frame.shape[:2]
    scale = max(0.4, h / 1200.0)
    thickness = max(1, round(scale * 2))
    text = "AI-GENERATED FACE"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = max(4, th // 2)
    x0, y1 = pad * 2, h - pad * 2
    x1, y0 = min(w, x0 + tw + pad * 2), max(0, y1 - th - baseline - pad * 2)
    box = frame[y0:y1, x0:x1]
    box[:] = box // 4            # dark plate keeps the text legible on any scene
    cv2.putText(frame, text, (x0 + pad, y1 - baseline - pad), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thickness, cv2.LINE_AA)


def generate_profile_source_face(profile):
    """Run diffusion once at session setup, never in the frame loop."""
    from ..profile.synthesizer import generate_profile_source_face as generate
    return generate(profile)


class _Job:
    """One frame's data as it moves between the pipeline stages."""

    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self.output = None      # set when the frame is passed through unprocessed
        self.epoch = 0          # bumps whenever tracking restarts (face lost, new source)
        self.work_ms = 0.0      # time spent inside stages, excluding queueing
        self.gpen_ms = 0.0


class SwapPipeline:
    """The swap, as six stages that each own their state.

    geometry    detect, smooth, align, swap, reproject
    restore     optional face restoration (GPEN)
    appearance  parse, masks, color, shading
    skin        skin detail, scattering, highlights, mouth interior
    camera      face motion, camera matching
    composite   paste-back, temporal blend, label

    process_frame() runs them one after another. AsyncPipelineRunner runs each
    in its own thread, so throughput is set by the slowest stage instead of
    their sum. A stage only touches its own state; tracking restarts travel
    with the frame as an epoch number, which later stages notice and reset on.
    """

    _STAGES = ('geometry', 'restore', 'appearance', 'skin', 'camera', 'composite')

    def __init__(self, swapper, enhancer, parser, source_face=None,
                 quality: str = 'ultra', frame_budget_ms: float = 33.0,
                 pixel_boost: bool = False, use_pct: bool = False,
                 use_occluder: bool = False, force_restoration: bool = False):
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
        # Restore every frame even when that lowers the frame rate.
        self._force_restoration = force_restoration
        self._face_smoother = FaceParamSmoother()
        self._ema = TemporalEMA(base_weight=0.55)
        # Face motion for the camera stage; its flow field is handed to _ema.
        self._motion = TemporalEMA()
        self._inference_times = deque(maxlen=30)
        self._base_times = deque(maxlen=30)
        self._gpen_times = deque(maxlen=10)
        self._timers = {stage: StageTimer() for stage in self._STAGES}
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
        # Tracking epoch: owned by the geometry stage, followed by the others.
        self._epoch = 0
        self._reset_pending = False
        self._stage_epoch = dict.fromkeys(self._STAGES, 0)
        # The swap-crop coverage mask depends only on the crop-to-crop transform,
        # which is the same for every frame; (key, mask) of the last one built.
        self._coverage_cache = None
        if enhancer is not None:
            # The first CUDA call is far slower than steady state; keep it out
            # of the restoration budget average.
            enhancer.enhance(np.zeros((512, 512, 3), np.uint8))

    @property
    def average_inference_ms(self) -> float:
        return sum(self._inference_times) / max(1, len(self._inference_times))

    @property
    def stage_ms(self) -> dict:
        """Rolling average milliseconds per step, in pipeline order."""
        merged = {}
        for stage in self._STAGES:
            merged.update(self._timers[stage].averages())
        return merged

    def stage_totals_ms(self) -> dict:
        """Rolling average milliseconds per stage (the units that run in parallel)."""
        return {stage: sum(self._timers[stage].averages().values()) for stage in self._STAGES}

    def _begin_epoch(self) -> None:
        """Restart tracking. Called only from the geometry stage."""
        self._epoch += 1
        self._face_smoother.reset()
        self._last_frame_time = None
        self._held_target = None
        self._missed = 0

    def _entered_new_epoch(self, stage: str, job: _Job) -> bool:
        if self._stage_epoch[stage] == job.epoch:
            return False
        self._stage_epoch[stage] = job.epoch
        return True

    def set_webrtc_streamer(self, streamer) -> None:
        self._webrtc = streamer

    def set_source_face(self, face) -> None:
        with self._lock:
            self._source_face = face
            # Picked up by the geometry stage at its next frame.
            self._reset_pending = True

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

    def _should_enhance_parallel(self) -> bool:
        """Restoration gate when the stages run in parallel.

        Restoration is then a stage of its own, sharing the GPU with the
        others, so what it costs is its own measured time under that load. It
        runs only while that stays within 1.5 frame budgets; beyond that it
        would set the frame rate. Same hysteresis as _should_enhance.
        """
        if self._enhancer is None:
            return False
        limit = self.frame_budget_ms * 1.5
        gpen = sum(self._gpen_times) / max(1, len(self._gpen_times))
        if gpen > limit:
            self._enhance_enabled = False
        elif gpen < limit * 0.85:
            self._enhance_enabled = True
        return self._enhance_enabled

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run all stages on one frame, one after another."""
        with self._lock:
            job = self.stage_geometry(frame_bgr)
            job = self.stage_restore(job)
            job = self.stage_appearance(job)
            job = self.stage_skin(job)
            job = self.stage_camera(job)
            return self.stage_composite(job)

    def parallel_stages(self) -> list:
        """The stages in order, as callables for one thread each."""
        return [self.stage_geometry,
                lambda job: self.stage_restore(job, parallel=True),
                self.stage_appearance, self.stage_skin, self.stage_camera,
                self.stage_composite]

    @staticmethod
    def passthrough(item) -> _Job:
        """A job that outputs its frame unprocessed (used when a stage fails)."""
        job = item if isinstance(item, _Job) else _Job(item)
        job.output = job.frame
        return job

    # --- Stage 1: geometry -------------------------------------------------

    def stage_geometry(self, frame_bgr: np.ndarray) -> _Job:
        job = _Job(frame_bgr)
        if self._reset_pending:
            self._reset_pending = False
            self._begin_epoch()
        source_face = self._source_face
        job.epoch = self._epoch
        if source_face is None:
            job.output = frame_bgr
            return job
        started = time.perf_counter()
        timer = self._timers['geometry']
        timer.start()
        target = detect_tracked(
            frame_bgr, None if self._held_target is None else self._held_target.bbox)
        timer.mark('detect')
        if target is None:
            # A brief dropout keeps the last geometry instead of flashing the
            # unprocessed frame and discarding all temporal state.
            self._missed += 1
            if self._held_target is None or self._missed > _MAX_HELD_FRAMES:
                self._begin_epoch()
                job.epoch = self._epoch
                job.output = frame_bgr
                return job
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
        swapped = self._swapper.swap(crop_swap, source_face, pixel_boost=self._pixel_boost)
        swap_h = np.vstack((M_swap, [0, 0, 1]))
        ffhq_h = np.vstack((M_ffhq, [0, 0, 1]))
        to_ffhq = (ffhq_h @ np.linalg.inv(swap_h))[:2]
        job.aligned = cv2.warpAffine(swapped, to_ffhq, (512, 512), borderMode=cv2.BORDER_REFLECT)
        # Feathered coverage mask prevents hard boundary at the swap crop edge (fix #4).
        # Both crops come from the same landmarks, so to_ffhq is the same
        # template-to-template transform on every frame: build the mask once.
        key = (swap_size, tuple(np.round(to_ffhq, 2).ravel()))
        if self._coverage_cache is None or self._coverage_cache[0] != key:
            coverage_raw = cv2.warpAffine(np.ones((swap_size, swap_size), np.float32),
                                          to_ffhq, (512, 512))
            self._coverage_cache = (key, feather_coverage(coverage_raw))
        job.coverage = self._coverage_cache[1]
        timer.mark('swap')
        job.target, job.M_ffhq, job.crop_ffhq = target, M_ffhq, crop_ffhq
        job.work_ms += (time.perf_counter() - started) * 1000
        return job

    # --- Stage 2: restoration ----------------------------------------------

    def stage_restore(self, job: _Job, parallel: bool = False) -> _Job:
        if job.output is not None:
            return job
        started = time.perf_counter()
        timer = self._timers['restore']
        timer.start()
        if self._entered_new_epoch('restore', job):
            self._enhance_enabled = True
            self._gpen_blend = 0.0
        # Ramp the blend weight over ~10 frames in both directions (fix #2).
        # Restoration keeps running while the weight ramps down, so disabling
        # it fades out instead of popping.
        if self._force_restoration:
            wanted = self._enhancer is not None
        elif parallel:
            wanted = self._should_enhance_parallel()
        else:
            wanted = self._should_enhance(job.work_ms)
        step = _GPEN_BLEND_STEP if wanted else -_GPEN_BLEND_STEP
        self._gpen_blend = min(_GPEN_BLEND_MAX, max(0.0, self._gpen_blend + step))
        if self._enhancer is not None and self._gpen_blend > 1e-6:
            before = time.perf_counter()
            job.enhanced = self._enhancer.enhance_with_blend(job.aligned, blend=self._gpen_blend)
            job.gpen_ms = (time.perf_counter() - before) * 1000
            self._gpen_times.append(job.gpen_ms)
        else:
            job.enhanced = job.aligned
            self.gpen_skipped += 1
            if self.gpen_skipped % _GPEN_PROBE_FRAMES == 0:
                # Cost samples only arrive while restoration runs. Drop the
                # stale ones so the gate can re-enable and measure again.
                self._gpen_times.clear()
        timer.mark('gpen')
        job.work_ms += (time.perf_counter() - started) * 1000
        return job

    # --- Stage 3: appearance -----------------------------------------------

    def stage_appearance(self, job: _Job) -> _Job:
        if job.output is not None:
            return job
        started = time.perf_counter()
        timer = self._timers['appearance']
        timer.start()
        if self._entered_new_epoch('appearance', job):
            self._parser.reset()
        crop_ffhq = job.crop_ffhq
        # Ownership masks come from the target: restoration may erase glasses
        # or change the mouth cavity before segmentation could preserve them.
        seg = self._parser.parse(crop_ffhq)
        swap_mask = self._parser.build_swap_mask(seg, erosion_px=3, transition_px=10) * job.coverage
        skin = self._parser.get_skin_mask(seg)
        mouth = self._parser.get_mouth_interior_mask(seg)
        # Occlusion: subtract occluded regions from swap mask so original face shows through
        if self._occluder is not None:
            occ = self._occluder.get_occlusion_mask(crop_ffhq, bisenet_seg=seg)
            swap_mask = swap_mask * (1.0 - occ)
        timer.mark('parse')
        # Reinhard correction now applies only within the skin mask (fix C).
        result = reinhard_color_transfer(job.enhanced, crop_ffhq, mask=skin)
        # PCT-Net spatial color harmonization: handles directional lighting (Priority 4c).
        # Reinhard strength reduced to 0.5 implicitly since PCT refines it.
        if self._use_pct:
            result = pct_harmonize(result, crop_ffhq, mask=skin)
        # Multiplicative shading correction in linear space, skin mask only (fix #7).
        timer.mark('color')
        result = low_freq_shading_correction(result, crop_ffhq, sigma=20.0, skin_mask=skin)
        timer.mark('shading')
        job.result, job.mask, job.skin, job.seg, job.mouth = result, swap_mask, skin, seg, mouth
        job.work_ms += (time.perf_counter() - started) * 1000
        return job

    # --- Stage 4: skin -----------------------------------------------------

    def stage_skin(self, job: _Job) -> _Job:
        if job.output is not None:
            return job
        started = time.perf_counter()
        timer = self._timers['skin']
        timer.start()
        if self._entered_new_epoch('skin', job):
            self._spec_mask_prev = None
        result, crop_ffhq, skin = job.result, job.crop_ffhq, job.skin
        if self._quality == 'ultra':
            face_width = max(1, int(job.target.bbox[2] - job.target.bbox[0]))
            result = inject_skin_detail(result, crop_ffhq, skin, face_width, strength=0.20)
            timer.mark('detail')
            result = apply_sss(result, skin, strength=0.20)
            timer.mark('sss')
            # Use observed target detail instead of adding procedural skin texture.
            # transfer_specular returns (result, spec_mask) for temporal EMA (fix #9).
            result, self._spec_mask_prev = transfer_specular(
                crop_ffhq, result, skin, prev_spec=self._spec_mask_prev)
            timer.mark('specular')
        job.result = composite_mouth_interior(result, crop_ffhq, job.mouth)
        timer.mark('mouth')
        job.work_ms += (time.perf_counter() - started) * 1000
        return job

    # --- Stage 5: camera ---------------------------------------------------

    def stage_camera(self, job: _Job) -> _Job:
        if job.output is not None:
            return job
        started = time.perf_counter()
        timer = self._timers['camera']
        timer.start()
        if self._entered_new_epoch('camera', job):
            self._motion.reset()
        # Camera degradation: match noise grain, sharpness, and motion blur of the
        # target camera feed. Applied before paste_back (last crop-space quality step).
        job.gray = cv2.cvtColor(job.frame, cv2.COLOR_BGR2GRAY)
        flow_vec = self._motion.get_face_flow(job.gray, job.target.bbox)
        # The composite stage blends against the same previous frame, so it can
        # reuse this flow field instead of computing it again.
        job.flow_cache = self._motion.flow_cache
        self._motion.remember(job.gray)
        timer.mark('flow')
        job.result = match_camera_degradation(job.result, job.crop_ffhq, job.skin, flow_vec)
        timer.mark('degrade')
        job.work_ms += (time.perf_counter() - started) * 1000
        return job

    # --- Stage 6: composite ------------------------------------------------

    def stage_composite(self, job: _Job) -> np.ndarray:
        if job.output is not None:
            return job.output
        started = time.perf_counter()
        timer = self._timers['composite']
        timer.start()
        if self._entered_new_epoch('composite', job):
            self._ema.reset()
            self._blink_gate.reset()
        frame_bgr, target, M_ffhq, gray = job.frame, job.target, job.M_ffhq, job.gray
        height, width = frame_bgr.shape[:2]
        full, frame_mask = paste_back(frame_bgr, job.result, M_ffhq, mask=job.mask, return_mask=True)
        timer.mark('paste')
        if job.flow_cache is not None and job.flow_cache[0] is gray:
            self._ema.adopt_flow(job.flow_cache)
        # BlinkGate: use 2d106det landmarks from buffalo_l; falls back gracefully to
        # is_blinking=False when landmark_2d_106 is None (e.g. on first frame).
        is_blinking = False
        left_eye, right_eye = get_eye_landmarks(target)
        if left_eye is not None:
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) * 0.5
            is_blinking = self._blink_gate.update(ear)
        dynamic = self._parser.get_region_mask(job.seg, [4, 5, 11, 12, 13])
        dynamic = cv2.warpAffine(dynamic, cv2.invertAffineTransform(M_ffhq), (width, height))
        temporal_mask = frame_mask * (1.0 - dynamic.astype(np.float32) / 255.0)
        full = self._ema.blend(full, gray, target.bbox, is_blinking=is_blinking,
                               face_mask=temporal_mask)
        timer.mark('temporal')
        stamp_ai_label(full)
        self._timed_frames += 1
        if self._timed_frames % 150 == 0:
            logger.info('Stage ms: %s', ' '.join(
                f'{name}={ms:.1f}' for name, ms in self.stage_ms.items()))
        job.work_ms += (time.perf_counter() - started) * 1000
        self.last_inference_ms = job.work_ms
        self._inference_times.append(job.work_ms)
        self._base_times.append(max(0.0, job.work_ms - job.gpen_ms))
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
    """Runs capture, every pipeline stage and output each in its own thread.

    Camera frames go through latest-only mailboxes up to the first stage, so a
    slow pipeline drops stale frames instead of lagging. Between stages a frame
    is handed over and waited for: every frame that enters the pipeline is
    finished, and the output rate is set by the slowest stage rather than by
    the sum of all of them.
    """
    def __init__(self, pipeline: SwapPipeline, output_callback: Callable,
                 target_width: int = 1280, target_height: int = 720):
        self._pipeline = pipeline
        self._callback = output_callback
        self._target_w, self._target_h = target_width, target_height
        self._raw_q = queue.Queue(maxsize=1)
        self._pre_q = queue.Queue(maxsize=1)
        self._stages = pipeline.parallel_stages()
        self._stage_qs = [queue.Queue(maxsize=1) for _ in range(len(self._stages) - 1)]
        self._post_q = queue.Queue(maxsize=1)
        self._threads = []
        self._stop = threading.Event()
        self._cap = None
        # Frames that failed in a stage are passed through unprocessed;
        # these let the GUI say so instead of failing silently.
        self.error_count = 0
        self.last_error = None
        # True once the camera has stopped giving frames while the runner was running.
        self.capture_ended = False
        self._output_times = deque(maxlen=30)

    @property
    def output_fps(self) -> float:
        """Frames per second actually delivered to the output callback."""
        times = list(self._output_times)
        if len(times) < 2 or times[-1] <= times[0]:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def start(self, cap) -> None:
        if any(t.is_alive() for t in self._threads):
            raise RuntimeError('Pipeline already running')
        self._stop.clear()
        mailboxes = [self._raw_q, self._pre_q, *self._stage_qs, self._post_q]
        for mailbox in mailboxes:
            while not mailbox.empty():
                mailbox.get_nowait()
        self._output_times.clear()
        self.capture_ended = False
        self._cap = cap
        chain = [self._pre_q, *self._stage_qs, self._post_q]
        stages = [(stage, chain[i], chain[i + 1]) for i, stage in enumerate(self._stages)]
        self._threads = [threading.Thread(target=fn, daemon=True) for fn in (
            self._capture_thread, self._preprocess_thread, self._output_thread)]
        self._threads += [threading.Thread(target=self._stage_thread, args=stage, daemon=True)
                          for stage in stages]
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

    def _hand_over(self, mailbox: queue.Queue, item) -> None:
        """Pass an item to the next stage, waiting until it has taken the previous one."""
        while not self._stop.is_set():
            try:
                mailbox.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _capture_thread(self):
        try:
            while not self._stop.is_set():
                ret, frame = self._cap.read()
                if not ret:
                    if not self._stop.is_set():
                        logger.warning('The camera stopped delivering frames')
                        self.capture_ended = True
                    break
                put_latest(self._raw_q, frame)
        except Exception:
            logger.exception('Capture failed')
            self.capture_ended = True
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

    def _stage_thread(self, stage, inbox, outbox):
        try:
            while True:
                item = self._take(inbox)
                if item is STOP:
                    break
                try:
                    result = stage(item)
                except Exception as exc:
                    logger.exception('Frame processing failed')
                    self.error_count += 1
                    self.last_error = exc
                    result = self._pipeline.passthrough(item)
                    if outbox is self._post_q:
                        result = result.output
                self._hand_over(outbox, result)
        finally:
            put_latest(outbox, STOP)

    def _output_thread(self):
        while True:
            frame = self._take(self._post_q)
            if frame is STOP:
                return
            self._output_times.append(time.monotonic())
            try:
                self._callback(frame)
            except Exception:
                logger.exception('Output failed')
                self._stop.set()
                return
