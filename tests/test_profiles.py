"""Profile contracts without downloading inference models."""
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np
    from faceswap.profile import extractor, synthesizer
    from faceswap.core import detector
except ImportError:
    np = None


@unittest.skipIf(np is None, "numpy and Pillow are required")
class ProfileTests(unittest.TestCase):
    def test_average_keeps_raw_magnitude(self):
        embeddings = [np.full(512, 2, dtype=np.float32), np.full(512, 4, dtype=np.float32)]
        with patch.object(extractor, "extract_embedding_from_image", side_effect=embeddings):
            result = extractor.build_profile_embedding(["a.png", "b.png"])
        np.testing.assert_array_equal(result, np.full(512, 3, dtype=np.float32))

    def test_save_load_preserves_raw_embedding(self):
        embedding = np.arange(512, dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profile.npz"
            extractor.save_profile(embedding, path)
            np.testing.assert_array_equal(extractor.load_profile(path), embedding)

    def test_detector_serializes_shared_inference(self):
        active = 0
        peak = 0
        lock = threading.Lock()
        def get(frame):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return []
        with patch.object(detector, "_get_app", return_value=SimpleNamespace(get=get)):
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(detector.detect_faces, [None] * 8))
        self.assertEqual(peak, 1)

    def test_source_generation_releases_models_on_failure(self):
        with patch.object(synthesizer, "Arc2FaceSynthesizer") as factory:
            instance = factory.return_value
            instance.generate_images.side_effect = RuntimeError("generation failed")
            profile = SimpleNamespace(embedding=np.ones(512), lora_path=Path("id.safetensors"))
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                synthesizer.generate_profile_source_face(profile)
            instance._pipeline.load_lora_weights.assert_called_once_with("id.safetensors")
            instance.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
