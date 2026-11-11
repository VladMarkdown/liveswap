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

    def test_homogeneous_crop_reprojection(self):
        from faceswap.core.aligner import estimate_transform, _ARCFACE_REF_112
        landmarks = _ARCFACE_REF_112 * 3 + [50, 20]
        swap = np.vstack((estimate_transform(landmarks, 256), [0, 0, 1]))
        ffhq = np.vstack((estimate_transform(landmarks, 512), [0, 0, 1]))
        transfer = ffhq @ np.linalg.inv(swap)
        points = np.column_stack((landmarks, np.ones(5))).T
        np.testing.assert_allclose(transfer @ (swap @ points), ffhq @ points, atol=1e-5)
