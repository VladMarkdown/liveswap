"""Shared ONNX Runtime session creation for the swap, restore, parse and occluder models."""
from __future__ import annotations
import logging
from pathlib import Path

import onnxruntime as ort

logger = logging.getLogger(__name__)


def create_session(model_path: str | Path, tensorrt_fp16: bool = False) -> ort.InferenceSession:
    """Create a session on the best available provider and report which one runs.

    Sessions are FP32 unless tensorrt_fp16 is set and the TensorRT provider is
    available; only models that tolerate FP16 (GPEN) should ask for it. The
    first TensorRT run builds an engine, which takes minutes; it is cached in
    a trt_cache folder next to the model.

    ONNX Runtime silently falls back to CPU when the CUDA/cuDNN libraries cannot be
    loaded, so the active provider is logged and a CPU fallback is a warning.
    """
    available = ort.get_available_providers()
    providers = []
    if tensorrt_fp16 and "TensorrtExecutionProvider" in available:
        cache = Path(model_path).parent / "trt_cache"
        cache.mkdir(exist_ok=True)
        providers.append(("TensorrtExecutionProvider", {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache),
        }))
    if "CUDAExecutionProvider" in available:
        # Grow the arena by what is requested, not by doubling: several sessions
        # share one 8 GB card.
        providers.append(("CUDAExecutionProvider",
                          {"arena_extend_strategy": "kSameAsRequested"}))
    if "DmlExecutionProvider" in available:
        providers.append("DmlExecutionProvider")
    providers.append("CPUExecutionProvider")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if "DmlExecutionProvider" in available:
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    while True:
        try:
            session = ort.InferenceSession(str(model_path), sess_options=opts,
                                           providers=providers)
            break
        except Exception:
            if len(providers) == 1:
                raise
            providers = providers[1:]

    active = session.get_providers()[0]
    name = Path(model_path).name
    if active == "CPUExecutionProvider":
        logger.warning("%s is running on CPU (available providers: %s); "
                       "real-time speed needs onnxruntime-gpu with working CUDA/cuDNN",
                       name, available)
    else:
        logger.info("%s is running on %s", name, active)
    return session
