"""Exercise GUI lifecycle methods without requiring Qt/GPU dependencies."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


def window_method(name, globals=None):
    tree = ast.parse((Path(__file__).parents[1] / 'faceswap/gui/main_window.py').read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MainWindow')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    method.decorator_list = []
    namespace = {} if globals is None else globals.copy()
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<gui method>', 'exec'), namespace)
    return namespace[name]


class GuiLifecycleTests(unittest.TestCase):
    def test_output_closed_only_after_workers_join(self):
        order = []
        runner = SimpleNamespace(stop=lambda: order.append('stop'), join=lambda: order.append('join'))
        window = SimpleNamespace(_runner=runner, _cap=Mock(), _pipeline=object(), _running=True,
                                 _vcam=SimpleNamespace(close=lambda: order.append('close')),
                                 _start_btn=Mock(), statusBar=lambda: Mock())
        self.assertTrue(window_method('_stop_pipeline')(window))
        self.assertEqual(order, ['stop', 'join', 'close'])
        self.assertIsNone(window._runner)
        self.assertIsNone(window._pipeline)

    def test_timeout_keeps_output_and_runner_for_retry(self):
        runner = Mock()
        runner.join.side_effect = TimeoutError('worker busy')
        vcam = Mock()
        window = SimpleNamespace(_runner=runner, _cap=Mock(), _pipeline=object(), _running=True,
                                 _vcam=vcam, _start_btn=Mock(), statusBar=lambda: Mock())
        self.assertFalse(window_method('_stop_pipeline')(window))
        self.assertIs(window._runner, runner)
        self.assertIs(window._vcam, vcam)
        self.assertTrue(window._running)
        vcam.close.assert_not_called()

    def test_placeholder_profile_is_rejected_before_loading(self):
        status = Mock()
        window = SimpleNamespace(_starting=False, _tabs=Mock(), _profile_combo=Mock(),
                                 statusBar=lambda: status)
        window._tabs.currentIndex.return_value = 1
        window._profile_combo.currentText.return_value = '(no profiles)'
        window._profile_combo.count.return_value = 1
        window_method('_start_pipeline')(window)
        status.showMessage.assert_called_once_with('Create a profile first')

    def test_level2_generation_precedes_model_load_and_runs_once(self):
        import sys
        from unittest.mock import patch
        order = []
        profile = SimpleNamespace(lora_path='adapter.safetensors')
        generated = object()
        pipeline = Mock()
        cap = Mock()
        signals = SimpleNamespace(startup_ready=Mock(), failed=Mock())
        window = SimpleNamespace(
            _starting=False, _closing=False, _tabs=Mock(), _profile_combo=Mock(),
            _source_face=None, _cam_combo=Mock(), _quality_ultra=Mock(), _start_btn=Mock(),
            _pipeline=object(), _swapper=object(), _enhancer=object(), _parser=object(),
            _signals=signals, statusBar=lambda: Mock())
        window._tabs.currentIndex.return_value = 1
        window._profile_combo.currentText.return_value = 'alice'
        window._profile_combo.count.return_value = 1
        def generate(value):
            self.assertIs(value, profile)
            self.assertIsNone(window._pipeline)
            self.assertIsNone(window._swapper)
            order.append('generate')
            return generated
        def load_models():
            order.append('models')
            window._swapper = window._enhancer = window._parser = object()
        window._load_models = load_models
        thread = lambda target, daemon: SimpleNamespace(start=target)
        namespace = dict(load_profile_by_name=lambda name: profile,
                         generate_profile_source_face=generate,
                         SwapPipeline=lambda *args, **kwargs: pipeline,
                         threading=SimpleNamespace(Thread=thread), gc=Mock())
        fake_module = SimpleNamespace(CaptureThread=lambda *args, **kwargs: cap)
        with patch.dict(sys.modules, {'faceswap.camera.capture': fake_module}):
            window_method('_start_pipeline', namespace)(window)
        self.assertEqual(order, ['generate', 'models'])
        pipeline.set_source_face.assert_called_once_with(generated)
        pipeline.set_profile.assert_not_called()
        signals.failed.emit.assert_not_called()
        signals.startup_ready.emit.assert_called_once_with((pipeline, cap))
        # The capture must actually be started, or the runner never gets a frame.
        cap.start.assert_called_once_with()
