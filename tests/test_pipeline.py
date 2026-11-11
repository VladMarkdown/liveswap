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


class StageTimerTests(unittest.TestCase):
    def test_marks_split_elapsed_time_by_stage(self):
        from collections import deque
        from types import SimpleNamespace
        clock = iter([0.0, 0.010, 0.015, 1.0, 1.030, 1.035])
        timer_cls = load_def('StageTimer')
        timer_cls.__init__.__globals__.update(
            deque=deque, time=SimpleNamespace(perf_counter=lambda: next(clock)))
        timer = timer_cls()
        for _ in range(2):
            timer.start()
            timer.mark('detect')
            timer.mark('swap')
        averages = timer.averages()
        self.assertEqual(list(averages), ['detect', 'swap'])
        self.assertAlmostEqual(averages['detect'], 20.0)
        self.assertAlmostEqual(averages['swap'], 5.0)


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
        # _enhance_enabled is the hysteresis state; starts True (GPEN enabled by default).
        state = SimpleNamespace(_enhancer=object(), last_inference_ms=20, frame_budget_ms=33,
                                _base_times=[18, 20], _gpen_times=[10], _enhance_enabled=True)
        return scope['_should_enhance'], state

    def test_single_overrun_does_not_disable_restoration(self):
        # Hysteresis fix: a single spike in last_inference_ms must NOT disable GPEN.
        # (The old hard-gate on last_inference_ms caused ~15 Hz alternating flicker.)
        check, state = self.budget_method()
        state.last_inference_ms = 50   # spike, but rolling avg is still fine
        self.assertTrue(check(state, 10))  # base=19, gpen=10 → 29 < 33 → enabled

    def test_rolling_overrun_disables_restoration(self):
        # Only a sustained overrun (rolling avg > budget) disables GPEN.
        check, state = self.budget_method()
        state._base_times = [30, 30]   # rolling avg 30, gpen=10 → 40 > 33 → disabled
        self.assertFalse(check(state, 15))

    def test_budget_recovers_without_stale_restoration(self):
        check, state = self.budget_method()
        self.assertTrue(check(state, 15))
        state._gpen_times = [20]
        self.assertFalse(check(state, 15))

    def test_current_frame_overrun_skips_restoration(self):
        check, state = self.budget_method()
        self.assertFalse(check(state, 30))
