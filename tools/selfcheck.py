"""Exercise every module once with the real models and report PASS / FAIL.

Needs no camera, no face and no desktop, so it can run over SSH. Each check is
independent: one failure does not stop the rest. Exit code is the number of
failed checks.
"""
from __future__ import annotations

import logging
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODELS = Path("models")
RESULTS: list[tuple[str, str, str]] = []


def check(name: str):
    """Decorator: run the function now, record PASS/FAIL/SKIP with a detail line."""
    def run(fn):
        started = time.perf_counter()
        try:
            detail = fn() or ""
            status = "SKIP" if detail.startswith("skip:") else "PASS"
        except Exception as exc:
            status = "FAIL"
            detail = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        RESULTS.append((status, name, f"{detail}  [{(time.perf_counter() - started) * 1000:.0f} ms]"))
        return fn
    return run


def _timed(fn, runs: int = 20) -> float:
    fn()                                   # warm-up
    started = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - started) * 1000 / runs


def main() -> int:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="%(name)s %(message)s")
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    crop512 = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
    state: dict = {}

    @check("onnxruntime GPU providers")
    def _():
        import onnxruntime as ort
        from faceswap.core.ort_utils import preload_gpu_libraries
        preload_gpu_libraries()
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError(f"no CUDA provider: {providers}")
        return f"onnxruntime {ort.__version__}: {providers}"

    @check("detector (InsightFace buffalo_l)")
    def _():
        from faceswap.core import detector
        if detector.detect_tracked(frame) is not None:
            raise RuntimeError("found a face in random noise")
        app = detector._get_app()
        tasks = sorted(app.models)
        if "landmark_2d_106" not in tasks or "recognition" not in tasks:
            raise RuntimeError(f"missing tasks: {tasks}")
        provider = app.det_model.session.get_providers()[0]
        return f"{provider}, {_timed(lambda: detector.detect_tracked(frame)):.1f} ms/frame, tasks {tasks}"

    @check("swap model")
    def _():
        from faceswap.core.swapper import load_default_swapper
        swapper = state["swapper"] = load_default_swapper(MODELS)
        embedding = rng.standard_normal(512).astype(np.float32)
        face = SimpleNamespace(embedding=embedding, normed_embedding=embedding / np.linalg.norm(embedding))
        crop = crop512[:256, :256].copy()
        out = swapper.swap(crop, face)
        if out.shape != (256, 256, 3) or out.dtype != np.uint8:
            raise RuntimeError(f"bad output {out.shape} {out.dtype}")
        boosted = swapper.swap(crop512, face, pixel_boost=True)
        if boosted.shape != (512, 512, 3):
            raise RuntimeError(f"bad pixel-boost output {boosted.shape}")
        outputs = [o.name for o in swapper._session.get_outputs()]
        return (f"{swapper._model_type} on {swapper._session.get_providers()[0]}, "
                f"{_timed(lambda: swapper.swap(crop, face)):.1f} ms, outputs {outputs}")

    @check("restoration model (GPEN)")
    def _():
        from faceswap.core.enhancer import FaceEnhancer
        enhancer = state["enhancer"] = FaceEnhancer(MODELS / "GPEN-BFR-512.onnx")
        out = enhancer.enhance(crop512)
        if out.shape != (512, 512, 3) or out.dtype != np.uint8:
            raise RuntimeError(f"bad output {out.shape} {out.dtype}")
        return f"{enhancer._session.get_providers()[0]}, {_timed(lambda: enhancer.enhance(crop512)):.1f} ms"

    @check("parsing model (BiSeNet)")
    def _():
        from faceswap.core.parser import FaceParser
        parser = state["parser"] = FaceParser(MODELS / "bisenet.onnx")
        seg = parser.parse(crop512)
        if seg.shape != (512, 512) or int(seg.max()) > 18:
            raise RuntimeError(f"bad label map {seg.shape}, max {seg.max()}")
        shape = parser._session.get_outputs()[0].shape
        return (f"{parser._session.get_providers()[0]}, {_timed(lambda: parser.parse(crop512)):.1f} ms, "
                f"output {shape}")

    @check("pipeline (frame without a face passes through)")
    def _():
        from faceswap.core.pipeline import SwapPipeline
        pipeline = SwapPipeline(state["swapper"], state["enhancer"], state["parser"])
        pipeline.set_source_face(SimpleNamespace(embedding=np.ones(512, np.float32),
                                                 normed_embedding=np.ones(512, np.float32) / 512 ** 0.5))
        out = pipeline.process_frame(frame)
        if not np.array_equal(out, frame):
            raise RuntimeError("frame without a face was modified")
        return "ok"

    @check("TensorRT for GPEN (optional)")
    def _():
        import onnxruntime as ort
        if "TensorrtExecutionProvider" not in ort.get_available_providers():
            return "skip: TensorRT provider not available"
        from faceswap.core.ort_utils import create_session
        session = create_session(MODELS / "GPEN-BFR-512.onnx", tensorrt_fp16=True)
        provider = session.get_providers()[0]
        if provider != "TensorrtExecutionProvider":
            return f"skip: provider listed but not usable, fell back to {provider}"
        blob = np.zeros((1, 3, 512, 512), np.float32)
        name = session.get_inputs()[0].name
        return f"{provider}, {_timed(lambda: session.run(None, {name: blob})):.1f} ms"

    @check("occluder model (optional)")
    def _():
        from faceswap.core.occluder import OcclusionDetector, _OCCLUDER_MODEL
        mask = OcclusionDetector().get_occlusion_mask(crop512, bisenet_seg=np.ones((512, 512), np.uint8))
        if mask.shape != (512, 512):
            raise RuntimeError(f"bad mask {mask.shape}")
        return "model" if _OCCLUDER_MODEL.exists() else "heuristic fallback (models/occluder.onnx not present)"

    @check("virtual camera")
    def _():
        from faceswap.camera.virtual_cam import VirtualCamera
        with VirtualCamera(1280, 720, 30.0, dll_path=str(MODELS / "softcam.dll")) as cam:
            cam.send_frame(frame)
            return f"frame sent to \"{cam.device_name}\""

    @check("WebRTC phone output")
    def _():
        from faceswap.streaming.webrtc_output import WebRTCStreamer
        streamer = WebRTCStreamer()
        streamer.start()
        try:
            url = streamer.local_url()
            streamer.send_frame(frame)
            time.sleep(1.0)
            import urllib.request
            with urllib.request.urlopen(url.replace("0.0.0.0", "127.0.0.1"), timeout=5) as response:
                status = response.status
        finally:
            stop = getattr(streamer, "stop", None)
            if stop is not None:
                stop()
        return f"{url} -> HTTP {status}"

    @check("GUI module imports")
    def _():
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from faceswap.gui import main_window
        main_window._save_settings = lambda settings: None
        main_window._LivePage._populate_cameras = lambda self: self.cam_combo.addItem("Camera 0", userData=0)
        app = QApplication.instance() or QApplication([])
        window = main_window.MainWindow()
        pages = window._stack.count()
        window.deleteLater()
        app.processEvents()
        return f"{pages} pages built"

    @check("profile mode dependencies (optional)")
    def _():
        missing = []
        for module in ("torch", "diffusers", "transformers", "peft", "arc2face"):
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        if missing:
            return f"skip: not installed: {', '.join(missing)}"
        import torch
        return f"torch {torch.__version__}, cuda {torch.cuda.is_available()}"

    print()
    for status, name, detail in RESULTS:
        print(f"{status:4}  {name}: {detail}")
    failed = sum(1 for status, _, _ in RESULTS if status == "FAIL")
    print(f"\n{len(RESULTS) - failed} of {len(RESULTS)} checks did not fail; {failed} failed")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
