"""Pipeline regressions runnable without model files."""
import ast
from pathlib import Path
import queue
import unittest


def load_def(name):
    tree = ast.parse(Path('faceswap/core/pipeline.py').read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
    if not nodes:
        raise AssertionError(f'Missing {name}')
    scope = {'queue': queue}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pipeline>', 'exec'), scope)
    return scope[name]


class MailboxTests(unittest.TestCase):
    def test_newest_replaces_oldest(self):
        put = load_def('put_latest')
        mailbox = queue.Queue(maxsize=1)
        put(mailbox, 'old')
        put(mailbox, 'new')
        self.assertEqual(mailbox.get_nowait(), 'new')

    def test_stop_can_replace_full_mailbox(self):
        put = load_def('put_latest')
        mailbox = queue.Queue(maxsize=1)
        put(mailbox, 'old')
        stop = object()
        put(mailbox, stop)
        self.assertIs(mailbox.get_nowait(), stop)


if __name__ == '__main__':
    unittest.main()

class BudgetTests(unittest.TestCase):
    def budget_method(self):
        from types import SimpleNamespace
        tree = ast.parse(Path('faceswap/core/pipeline.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SwapPipeline')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_should_enhance')
        scope = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<budget>', 'exec'), scope)
        state = SimpleNamespace(_enhancer=object(), last_inference_ms=20, frame_budget_ms=33,
                                _base_times=[18, 20], _gpen_times=[10])
        return scope['_should_enhance'], state

    def test_last_frame_overrun_skips_restoration(self):
        check, state = self.budget_method()
        state.last_inference_ms = 34
        self.assertFalse(check(state, 10))

    def test_budget_recovers_without_stale_restoration(self):
        check, state = self.budget_method()
        self.assertTrue(check(state, 15))
        state._gpen_times = [20]
        self.assertFalse(check(state, 15))

    def test_current_frame_overrun_skips_restoration(self):
        check, state = self.budget_method()
        self.assertFalse(check(state, 30))
