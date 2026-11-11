"""
Face swap via AlphaFace_256 or HyperSwap_1b_256 (ONNX).

Embedding conventions (CRITICAL — different per model):
  AlphaFace_256:    face.embedding         (raw, unnormalized)
  HyperSwap_*:      face.normed_embedding  (L2-normalized)
  inswapper_128:    projected via E_map (see prepare_inswapper_identity)

AlphaFace / HyperSwap CANNOT use TRT FP16 — documented precision loss (ORT issue #19437).
Use CUDAExecutionProvider FP32 for swap model.
"""
from __future__ import annotations
import numpy as np
import onnxruntime as ort
from pathlib import Path


def _fp32_providers():
    available = ort.get_available_providers()
    return [name for name in ("CUDAExecutionProvider", "DmlExecutionProvider",
                              "CPUExecutionProvider") if name in available]


class FaceSwapper:
    """
    Wraps AlphaFace_256 or HyperSwap_1b_256 ONNX model.

    model_type: "alphaface" | "hyperswap" | "inswapper"
    """

    def __init__(self, model_path: str | Path, model_type: str = "alphaface"):
        if model_type not in ("alphaface", "hyperswap", "hyperswap_1a",
                               "hyperswap_1b", "hyperswap_1c", "inswapper"):
            raise ValueError(f"Unsupported swap model type: {model_type}")
        self._model_type = model_type
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = _fp32_providers()
        if "DmlExecutionProvider" in providers:
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        while providers:
            try:
                self._session = ort.InferenceSession(
                    str(model_path), sess_options=opts, providers=providers)
                break
            except Exception:
                if len(providers) == 1:
                    raise
                providers = providers[1:]
        inputs = self._session.get_inputs()
        images = [item for item in inputs if len(item.shape) == 4 and item.shape[1] == 3]
        identities = [item for item in inputs if len(item.shape) == 2 and item.shape[-1] == 512]
        if len(images) != 1 or len(identities) != 1:
            raise ValueError("Swap model must have one NCHW RGB input and one 512-value identity input")
        self._image_input = images[0].name
        self._image_shape = images[0].shape
        self._identity_input = identities[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

        if model_type == "inswapper":
            self._embedding_map = self._load_inswapper_map(str(model_path))

    @staticmethod
    def _load_inswapper_map(model_path: str) -> np.ndarray:
        """Extract the 512×512 identity projection matrix from inswapper ONNX graph."""
        import onnx
        from onnx import numpy_helper
        graph = onnx.load(model_path)
        # Search by shape rather than relying on ordering (fragile across re-exports).
        for init in graph.graph.initializer:
            arr = numpy_helper.to_array(init)
            if arr.shape == (512, 512):
                return arr.astype(np.float32)
        raise ValueError(
            "Cannot find 512×512 identity projection matrix in inswapper ONNX model; "
            "model may be from an unsupported export."
        )

    def _prepare_identity(self, face) -> np.ndarray:
        """
        Extract source identity vector from InsightFace face object.
        Returns (1, 512) float32.
        """
        if self._model_type == "alphaface":
            emb = face.embedding.astype(np.float32)
        elif self._model_type in ("hyperswap", "hyperswap_1a", "hyperswap_1b", "hyperswap_1c"):
            emb = face.normed_embedding.astype(np.float32)
        elif self._model_type == "inswapper":
            unit = face.normed_embedding.astype(np.float32).reshape(1, 512)
            projected = unit @ self._embedding_map
            projected /= np.linalg.norm(projected, axis=1, keepdims=True)
            return projected.astype(np.float32)

        emb = emb.reshape(1, 512)
        return emb

    def swap(self, target_crop: np.ndarray, source_face) -> np.ndarray:
        """
        Swap source identity onto target_crop.

        Args:
            target_crop: aligned face crop, HWC BGR uint8, 256×256
            source_face: InsightFace face object from source image

        Returns:
            swapped crop: HWC BGR uint8, 256×256
        """
        if target_crop.dtype != np.uint8 or target_crop.ndim != 3 or target_crop.shape[2] != 3:
            raise ValueError("Swap crop must be HWC BGR uint8")
        for actual, required in zip(target_crop.shape[:2], self._image_shape[2:]):
            if isinstance(required, int) and required > 0 and actual != required:
                raise ValueError(f"Swap crop dimensions must match {self._image_shape[2:]}")
        identity = self._prepare_identity(source_face)

        # Preprocess: BGR → RGB, uint8 → float32, HWC → NCHW, /255
        rgb = target_crop[:, :, ::-1].astype(np.float32) / 255.0
        if self._model_type.startswith("hyperswap"):
            rgb = rgb * 2.0 - 1.0
        blob = rgb.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

        feed = {
            self._image_input: blob,
            self._identity_input: identity,
        }
        result = self._session.run(self._output_names, feed)[0]  # (1, 3, H, W)

        # Postprocess: NCHW → HWC, clamp, RGB → BGR
        out = result[0].transpose(1, 2, 0)
        if self._model_type.startswith("hyperswap"):
            out = (out + 1.0) * 127.5
        else:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out[:, :, ::-1]  # RGB → BGR
