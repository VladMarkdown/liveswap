"""Capture mailbox tests use a fake camera; no hardware or OpenCV required."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class FakeCamera:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.closed = False

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def read(self):
        return next(self.frames, (False, None))

    def release(self):
        self.closed = True


class CaptureTests(unittest.TestCase):
    def load_capture(self, camera):
        fake = SimpleNamespace(VideoCapture=lambda source: camera,
                               CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
                               CAP_PROP_FPS=5)
        spec = importlib.util.spec_from_file_location(
            'capture_under_test', Path(__file__).parents[1] / 'faceswap/camera/capture.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'cv2': fake}):
            spec.loader.exec_module(module)
        return module.CaptureThread

    def test_latest_frame_replaces_stale_frames_and_end_is_readable(self):
        camera = FakeCamera([(True, 1), (True, 2), (True, 3)])
        capture = self.load_capture(camera)().start()
        capture._thread.join(timeout=1)
        self.assertFalse(capture._thread.is_alive())
        self.assertEqual(capture.read(), (True, 3))
        self.assertEqual(capture.read(), (False, None))
        self.assertEqual(capture.dropped_frames, 2)
        self.assertGreater(capture.fps, 0)
        capture.release()
        capture.release()
        self.assertTrue(camera.closed)

    def test_failed_open_releases_camera(self):
        camera = FakeCamera([])
        camera.isOpened = lambda: False
        capture = self.load_capture(camera)()
        with self.assertRaisesRegex(RuntimeError, 'Cannot open camera'):
            capture.start()
        self.assertTrue(camera.closed)
        capture.release()

    def test_backend_exception_ends_reader_and_releases(self):
        camera = FakeCamera([])
        def fail():
            raise RuntimeError('camera disconnected')
        camera.read = fail
        capture = self.load_capture(camera)().start()
        capture._thread.join(timeout=1)
        self.assertEqual(capture.read(), (False, None))
        self.assertIsInstance(capture.error, RuntimeError)
        self.assertTrue(camera.closed)
        capture.release()
