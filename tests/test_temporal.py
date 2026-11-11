"""Numerical temporal and geometry regressions; no model downloads required."""
import unittest
try:
    import numpy as np
    import cv2
    from faceswap.core.temporal import TemporalEMA
except ImportError:
    np = None


@unittest.skipIf(np is None, 'numpy/OpenCV unavailable')
class TemporalTests(unittest.TestCase):
    def test_background_remains_current(self):
        ema = TemporalEMA()
        old = np.full((64, 64, 3), 50, np.uint8)
        gray = np.full((64, 64), 100, np.uint8)
        mask = np.zeros((64, 64), np.float32)
        mask[16:48, 16:48] = 1
        ema.blend(old, gray, [16, 16, 48, 48], face_mask=mask)
        current = np.full_like(old, 150)
        result = ema.blend(current, gray, [16, 16, 48, 48], face_mask=mask)
        np.testing.assert_array_equal(result[mask == 0], current[mask == 0])
        self.assertLess(result[32, 32, 0], 150)

    def test_resolution_change_resets_history(self):
        ema = TemporalEMA()
        ema.blend(np.zeros((32, 32, 3), np.uint8), np.zeros((32, 32), np.uint8), [0, 0, 32, 32])
        current = np.full((64, 64, 3), 120, np.uint8)
        result = ema.blend(current, current[:, :, 0], [0, 0, 64, 64])
        np.testing.assert_array_equal(result, current)

    def test_eye_aspect_ratio_ignores_point_order(self):
        from faceswap.core.temporal import eye_aspect_ratio
        eye = np.array([[0, 0], [3, -1], [7, -1], [10, 0], [7, 1], [3, 1]], float)
        self.assertAlmostEqual(eye_aspect_ratio(eye), 0.2)
        self.assertAlmostEqual(eye_aspect_ratio(eye[[4, 0, 3, 5, 1, 2]]), 0.2)

    def test_smoother_scale_uses_all_points_not_eye_distance(self):
        from faceswap.core.aligner import _ARCFACE_REF_112
        from faceswap.core.temporal import FaceParamSmoother
        frontal = _ARCFACE_REF_112 * 3 + [50, 20]
        np.testing.assert_allclose(FaceParamSmoother().smooth(frontal), frontal, atol=1e-3)
        # Yaw: the eyes move closer together, the rest of the face keeps its size.
        turned = frontal.copy()
        turned[0, 0] += 15
        turned[1, 0] -= 15
        out = FaceParamSmoother().smooth(turned)
        eye_ratio = np.linalg.norm(out[1] - out[0]) / np.linalg.norm(frontal[1] - frontal[0])
        raw_ratio = np.linalg.norm(turned[1] - turned[0]) / np.linalg.norm(frontal[1] - frontal[0])
        self.assertGreater(eye_ratio, raw_ratio + 0.1)

    def test_smoother_damps_subpixel_jitter_when_normalized(self):
        from faceswap.core.aligner import _ARCFACE_REF_112
        from faceswap.core.temporal import FaceParamSmoother
        base = _ARCFACE_REF_112 * 3 + [600, 300]
        smoother = FaceParamSmoother()
        centers = []
        for i in range(60):
            jitter = 0.5 if i % 2 else -0.5
            centers.append(smoother.smooth(base + [jitter, 0], 1 / 30, (1280, 720)).mean(axis=0)[0])
        self.assertLess(np.ptp(centers[30:]), 0.5)

    def test_flow_is_computed_once_per_frame(self):
        from unittest.mock import patch
        ema = TemporalEMA()
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        gray = frame[:, :, 0].copy()
        ema.blend(frame, gray, [16, 16, 48, 48])
        gray2 = gray.copy()
        with patch('faceswap.core.temporal.cv2.calcOpticalFlowFarneback',
                   wraps=cv2.calcOpticalFlowFarneback) as flow:
            ema.get_face_flow(gray2, [16, 16, 48, 48])
            ema.blend(frame, gray2, [16, 16, 48, 48])
        self.assertEqual(flow.call_count, 1)

    def test_homogeneous_crop_reprojection(self):
        from faceswap.core.aligner import estimate_transform, _ARCFACE_REF_112
        landmarks = _ARCFACE_REF_112 * 3 + [50, 20]
        swap = np.vstack((estimate_transform(landmarks, 256), [0, 0, 1]))
        ffhq = np.vstack((estimate_transform(landmarks, 512), [0, 0, 1]))
        transfer = ffhq @ np.linalg.inv(swap)
        points = np.column_stack((landmarks, np.ones(5))).T
        np.testing.assert_allclose(transfer @ (swap @ points), ffhq @ points, atol=1e-5)
