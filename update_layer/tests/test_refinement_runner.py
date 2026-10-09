"""Exercise the real runner's artifact/report/exit boundary with a one-frame CPU backend.

No optimizer or Gaussian renderer is emulated: the fixture returns a refinement outcome at
the Backend interface, while run_chain writes actual checkpoints and run.json.
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from update_layer import run
from update_layer.eval.scenelist import SceneListTimeline
from update_layer.refinement import RefinementProgress


class _OneFrameSession:
    def __init__(self, _spec, pixel_step):
        self.ids = [0]

    def stamp_ns(self, _index):
        return 1_000_000_000

    def load(self, _index):
        return SimpleNamespace(stamp_ns=self.stamp_ns(0))


class _CompletionBackend:
    CHANGES = []

    def __init__(self, outcome, progress=None):
        self.outcome = outcome
        if progress is not None:
            self.game = SimpleNamespace(refinement_progress=progress)
        self.phase = "pre_ref"
        self.stamps = []
        self.scenes = []

    def start_session(self, *_args):
        pass

    def integrate(self, _frame):
        pass

    def elements(self):
        return []

    def snapshot(self, stamp):
        self.stamps.append(stamp)
        self.scenes.append({"phase": self.phase})

    def timeline(self):
        return SceneListTimeline(list(self.stamps), list(self.scenes))

    def end_session(self):
        return self.phase

    def prior_state(self, _final):
        return {"phase": self.phase}

    def finish_session(self, _stamp):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        self.phase = "post_ref"
        return self.outcome


class _SplitCompletionBackend(_CompletionBackend):
    consumes_state_intervals = False

    def __init__(self, measurement):
        super().__init__(None)
        self.measurement = measurement
        self.started = []
        self.final_refinements = 0
        self.restores = 0

    def start_session(self, spec, _prior):
        self.started.append(spec.name)

    def retire(self, *_args):
        pass

    def prior_from_state(self, state):
        self.restores += 1
        return state["phase"]

    def measurement_update(self, _stamp):
        self.phase = "measurement_update"
        return self.measurement

    def finish_session(self, stamp):
        self.final_refinements += 1
        return super().finish_session(stamp)


class _NoopLayer:
    def __init__(self, _cfg):
        pass

    def start_session(self, *_args):
        pass

    def motion(self, _frame):
        return None

    def observe(self, _frame):
        pass

    def decide(self, *_args, **_kwargs):
        return {"element_rule": torch.empty(0, dtype=torch.int64)}

    def session_end_memory(self, *_args):
        return torch.empty(0, dtype=torch.int64), 0

    def end_session(self, _directory):
        return None


class RefinementRunnerTests(unittest.TestCase):
    def _run(self, out, outcome, progress=None):
        cfg = SimpleNamespace(round_s=10.8, evidence_hz=5.0, pixel_step=2)
        spec = SimpleNamespace(name="synthetic_b")
        backend = _CompletionBackend(outcome, progress)
        with mock.patch.object(run, "dataset_config", return_value=(cfg, SimpleNamespace(), [spec])), \
                mock.patch.object(run, "FlatSession", _OneFrameSession), \
                mock.patch.object(run, "make_backend", return_value=backend), \
                contextlib.redirect_stdout(io.StringIO()):
            run.run_chain("points", 1, "synthetic", out, verbose=False)

    def _run_split(self, out, backend, resume=False):
        cfg = SimpleNamespace(round_s=10.8, evidence_hz=5.0, pixel_step=2)
        specs = [SimpleNamespace(name="synthetic_a"), SimpleNamespace(name="synthetic_b")]
        with mock.patch.object(run, "dataset_config", return_value=(cfg, SimpleNamespace(), specs)), \
                mock.patch.object(run, "FlatSession", _OneFrameSession), \
                mock.patch.object(run, "UpdateLayer", _NoopLayer), \
                mock.patch.object(run, "make_backend", return_value=backend), \
                contextlib.redirect_stdout(io.StringIO()):
            run.run_chain("game", 4, "synthetic", out, verbose=False, g5=True, split=True, resume=resume)

    def test_partial_artifacts_and_report_are_saved_before_exit_3(self):
        partial = {"status": "partial", "requested_iterations": 30000,
                   "completed_iterations": 500, "stop_reason": "gpu_guard"}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with self.assertRaises(SystemExit) as caught:
                self._run(out, partial)
            self.assertEqual(caught.exception.code, 3)
            for filename in ("checkpoint_b.pt", "checkpoint_b_post_ref.pt",
                             "session_b/timeline.pkl", "session_b/timeline_post_ref.pkl"):
                self.assertTrue((out / filename).is_file(), filename)
            pre = torch.load(out / "checkpoint_b.pt", weights_only=False, map_location="cpu")
            self.assertEqual(pre["backend"]["phase"], "pre_ref")
            report = json.loads((out / "session_b/run.json").read_text())["post_ref"]
            for key, expected in partial.items():
                self.assertEqual(report[key], expected)

    def test_exception_records_failure_without_stale_completion_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with self.assertRaises(SystemExit) as caught:
                self._run(out, RuntimeError())
            self.assertEqual(caught.exception.code, 3)
            self.assertTrue((out / "checkpoint_b.pt").is_file())
            self.assertFalse((out / "checkpoint_b_post_ref.pt").exists())
            report = json.loads((out / "session_b/run.json").read_text())["post_ref"]
            self.assertEqual(report["status"], "failed")
            self.assertIn("RuntimeError", report["stop_reason"])
            self.assertNotIn("requested_iterations", report)
            self.assertNotIn("completed_iterations", report)

    def test_legacy_none_return_keeps_existing_success_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            self._run(out, None)
            report = json.loads((out / "session_b/run.json").read_text())["post_ref"]
            self.assertEqual(report["status"], "ok")
            self.assertTrue((out / "checkpoint_b_post_ref.pt").is_file())

    def test_exception_preserves_current_progress_and_does_not_save_failed_model(self):
        progress = RefinementProgress(30000)
        for _ in range(500):
            progress.advance()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with self.assertRaises(SystemExit) as caught:
                self._run(out, RuntimeError("driver failed"), progress)
            self.assertEqual(caught.exception.code, 3)
            report = json.loads((out / "session_b/run.json").read_text())["post_ref"]
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["completed_iterations"], 500)
            self.assertEqual(report["requested_iterations"], 30000)
            self.assertIn("driver failed", report["stop_reason"])
            self.assertTrue((out / "checkpoint_b.pt").exists())
            self.assertFalse((out / "checkpoint_b_post_ref.pt").exists())

    def test_partial_measurement_update_is_saved_stops_chain_and_blocks_resume(self):
        partial = {"status": "partial", "requested_iterations": 30000,
                   "completed_iterations": 500, "stop_reason": "gpu_guard"}
        backend = _SplitCompletionBackend(partial)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with self.assertRaises(SystemExit) as caught:
                self._run_split(out, backend)
            self.assertEqual(caught.exception.code, 3)
            self.assertEqual(backend.started, ["synthetic_a"])
            self.assertEqual(backend.final_refinements, 0)
            for name in ("checkpoint_a.pt", "checkpoint.pt", "session_a/timeline.pkl", "session_a/run.json"):
                self.assertTrue((out / name).is_file(), name)
            self.assertFalse((out / "session_b").exists())
            report = json.loads((out / "session_a/run.json").read_text())["measurement_update"]
            self.assertEqual(report, partial)
            resumed = _SplitCompletionBackend(partial)
            with self.assertRaises(SystemExit) as caught:
                self._run_split(out, resumed, resume=True)
            self.assertEqual(caught.exception.code, 3)
            self.assertEqual(resumed.started, [])
            self.assertEqual(resumed.restores, 0)      # reject before loading the partial model into a backend
            self.assertEqual(resumed.final_refinements, 0)


if __name__ == "__main__":
    unittest.main()
