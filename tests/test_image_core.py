import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import cv2
    import numpy as np
    import scipy
    import onnxruntime
except ImportError:
    np = None
else:
    from faceswap.core import aligner, color
    from faceswap.core.swapper import FaceSwapper


@unittest.skipIf(np is None, 'image dependencies unavailable')
class ImageCoreTests(unittest.TestCase):
    def test_unknown_model_type_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported swap model'):
            FaceSwapper('absent.onnx', 'unknown')

    def test_arcface_reference_scales_to_default_256(self):
        points = np.array([[38.2946, 51.6963], [73.5318, 51.5014],
                           [56.0252, 71.7366], [41.5493, 92.3655],
                           [70.7299, 92.2041]], np.float32)
        crop, matrix = aligner.align_face(np.zeros((256, 256, 3), np.uint8), points)
        self.assertEqual(crop.shape[:2], (256, 256))
        np.testing.assert_allclose(matrix, [[256 / 112, 0, 0], [0, 256 / 112, 0]], atol=1e-5)

    def test_paste_returns_mask_and_preserves_background(self):
        frame = np.full((512, 512, 3), 80, np.uint8)
        face = np.full((32, 32, 3), 180, np.uint8)
        result, mask = aligner.paste_back(frame, face, np.array([[1., 0, -240], [0, 1, -240]]), return_mask=True)
        self.assertEqual(mask.shape, frame.shape[:2])
        self.assertEqual(float(mask[200, 200]), 0)
        np.testing.assert_array_equal(result[0, 0], frame[0, 0])
        self.assertGreater(int(result[256, 256, 0]), 80)

    def test_alphaface_keeps_raw_embedding(self):
        swapper = FaceSwapper.__new__(FaceSwapper)
        swapper._model_type = 'alphaface'
        face = SimpleNamespace(embedding=np.full(512, 2, np.float32))
        np.testing.assert_array_equal(swapper._prepare_identity(face), face.embedding[None])

    def test_hyperswap_uses_named_roles_and_signed_rgb(self):
        feed = {}
        class Session:
            def get_inputs(self):
                return [SimpleNamespace(name='identity', shape=[1, 512]),
                        SimpleNamespace(name='pixels', shape=[1, 3, 256, 256])]
            def get_outputs(self):
                return [SimpleNamespace(name='output')]
            def run(self, names, values):
                feed.update(values)
                return [np.zeros((1, 3, 256, 256), np.float32)]
        with patch('faceswap.core.swapper.ort.InferenceSession', return_value=Session()):
            swapper = FaceSwapper('fake.onnx', 'hyperswap')
        face = SimpleNamespace(normed_embedding=np.ones(512, np.float32) / np.sqrt(512))
        out = swapper.swap(np.zeros((256, 256, 3), np.uint8), face)
        np.testing.assert_array_equal(feed['pixels'], -np.ones((1, 3, 256, 256)))
        self.assertEqual(int(out[0, 0, 0]), 127)

    def test_uint8_detail_matches_linear_calculation(self):
        rng = np.random.default_rng(1)
        face = rng.integers(60, 180, (32, 32, 3), dtype=np.uint8)
        target = rng.integers(60, 180, face.shape, dtype=np.uint8)
        mask = np.full((32, 32), 255, np.uint8)
        linear = color.inject_skin_detail((face / 255.) ** 2.2, (target / 255.) ** 2.2, mask, 256)
        expected = np.clip(linear ** (1 / 2.2) * 255, 0, 255).astype(np.uint8)
        actual = color.inject_skin_detail(face, target, mask, 256)
        self.assertEqual(actual.dtype, np.uint8)
        np.testing.assert_allclose(actual, expected, atol=1)

    def test_sss_scatters_red_more_than_blue(self):
        face = np.zeros((65, 65, 3), np.uint8)
        face[32, 32] = 255
        result = color.apply_sss(face, np.full((65, 65), 255, np.uint8), strength=1)
        self.assertGreater(int(result[32, 32, 0]), int(result[32, 32, 2]))
