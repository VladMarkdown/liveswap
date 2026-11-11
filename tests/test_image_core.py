import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import cv2
    import numpy as np
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
        # The iterative similarity fit lands within ~2e-5 px of the exact answer.
        np.testing.assert_allclose(matrix, [[256 / 112, 0, 0], [0, 256 / 112, 0]], atol=1e-4)

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
            def get_providers(self):
                return ['CPUExecutionProvider']
            def run(self, names, values):
                feed.update(values)
                return [np.zeros((1, 3, 256, 256), np.float32)]
        with patch('faceswap.core.ort_utils.ort.InferenceSession', return_value=Session()):
            swapper = FaceSwapper('fake.onnx', 'hyperswap')
        face = SimpleNamespace(normed_embedding=np.ones(512, np.float32) / np.sqrt(512))
        out = swapper.swap(np.zeros((256, 256, 3), np.uint8), face)
        np.testing.assert_array_equal(feed['pixels'], -np.ones((1, 3, 256, 256)))
        self.assertEqual(int(out[0, 0, 0]), 127)

    def test_pixel_boost_feeds_whole_face_subimages_and_reinterleaves(self):
        seen = []
        swapper = FaceSwapper.__new__(FaceSwapper)
        swapper._image_shape = [1, 3, 256, 256]
        def swap_single(crop, face):
            seen.append(crop.shape)
            return crop
        swapper._swap_single = swap_single
        crop = np.random.default_rng(0).integers(0, 255, (512, 512, 3), dtype=np.uint8)
        out = swapper.swap(crop, None, pixel_boost=True)
        self.assertEqual(seen, [(256, 256, 3)] * 4)
        np.testing.assert_array_equal(out, crop)

    def run_pipeline(self, detections, **options):
        """Drive SwapPipeline.process_frame with fake models; returns (outputs, enhancer calls)."""
        from faceswap.core import pipeline as pipeline_module
        from faceswap.core.parser import FaceParser

        class Swapper:
            def swap(self, crop, face, pixel_boost=False):
                return 255 - crop

        class Enhancer:
            calls = 0
            def enhance(self, crop):
                return crop
            def enhance_with_blend(self, crop, blend=0.8):
                Enhancer.calls += 1
                return crop

        class Session:
            def run(self, names, feed):
                logits = np.zeros((1, 19, 512, 512), np.float32)
                logits[0, 1, 128:384, 128:384] = 5   # a square of skin
                return [logits]

        parser = FaceParser.__new__(FaceParser)
        parser._session, parser._input_name, parser._output_name = Session(), 'in', 'out'
        parser._logit_ema_alpha, parser._prev_logits = 0.6, None

        pipe = pipeline_module.SwapPipeline(Swapper(), Enhancer(), parser, **options)
        pipe.set_source_face(object())
        frame = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        queue = iter(detections)
        with patch.object(pipeline_module, 'detect_tracked', lambda frame, prev: next(queue)):
            outputs = [pipe.process_frame(frame) for _ in detections]
        return frame, outputs, Enhancer.calls

    def fake_face(self):
        kps = aligner._ARCFACE_REF_112 * 3 + [450, 150]
        landmarks = np.zeros((106, 2), np.float32)
        return SimpleNamespace(kps=kps, bbox=np.array([480., 180., 800., 560.]),
                               landmark_2d_106=landmarks)

    def test_process_frame_runs_end_to_end_with_and_without_pixel_boost(self):
        for boost in (False, True):
            frame, outputs, _ = self.run_pipeline(
                [self.fake_face()] * 3, pixel_boost=boost, use_pct=True, use_occluder=True)
            self.assertEqual(outputs[-1].shape, frame.shape)
            self.assertFalse(np.array_equal(outputs[-1], frame))
            np.testing.assert_array_equal(outputs[-1][:50], frame[:50])
            # The label sits bottom-left, far from the face: only it can change those pixels.
            self.assertFalse(np.array_equal(outputs[-1][680:710, 20:200], frame[680:710, 20:200]))

    def test_frames_without_a_face_are_not_labelled(self):
        frame, outputs, _ = self.run_pipeline([None])
        np.testing.assert_array_equal(outputs[0], frame)

    def test_restoration_fades_out_instead_of_stopping_at_once(self):
        from faceswap.core import pipeline as pipeline_module
        with patch.object(pipeline_module.SwapPipeline, '_should_enhance',
                          lambda self, elapsed: self.gpen_skipped == 0 and len(self._gpen_times) < 3):
            _, _, calls = self.run_pipeline([self.fake_face()] * 8, quality='fast')
        # Three wanted frames ramp the blend to 0.24; two more run at 0.16 and 0.08
        # on the way back to 0. Stopping at once would give 3.
        self.assertEqual(calls, 5)

    def test_brief_detection_dropout_keeps_swapping_then_gives_up(self):
        face = self.fake_face()
        frame, outputs, _ = self.run_pipeline([face] + [None] * 6, quality='fast')
        for held in outputs[1:6]:
            self.assertFalse(np.array_equal(held, frame))
        np.testing.assert_array_equal(outputs[6], frame)

    def test_pct_harmonize_keeps_matching_bright_face_unchanged(self):
        face = np.full((512, 512, 3), 220, np.uint8)
        face[..., 2] = 240
        mask = np.full((512, 512), 255, np.uint8)
        out = color.pct_harmonize(face, face.copy(), mask)
        np.testing.assert_allclose(out, face, atol=2)

    def test_shading_correction_matches_target_brightness(self):
        swapped = np.full((512, 512, 3), 100, np.uint8)
        target = np.full((512, 512, 3), 140, np.uint8)
        out = color.low_freq_shading_correction(swapped, target, sigma=20.0)
        self.assertGreater(int(out[256, 256, 0]), 120)
        self.assertLess(int(out[256, 256, 0]), 140)

    def test_sss_blur_radius_stays_a_few_pixels_at_512(self):
        face = np.zeros((512, 512, 3), np.uint8)
        face[256, 256] = 255
        out = color.apply_sss(face, np.full((512, 512), 255, np.uint8), strength=1)
        self.assertGreater(int(out[256, 256, 2]), 0)
        self.assertEqual(int(out[256, 300, 2]), 0)

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
