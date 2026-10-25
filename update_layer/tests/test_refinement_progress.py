"""Completion/status regression tests; standard library only, no GPU or optimizer emulation."""
import unittest

from update_layer.refinement import RefinementProgress, completion_exit_code, normalize_refinement_report


class RefinementProgressTests(unittest.TestCase):
    def test_completed_budget_is_ok(self):
        progress = RefinementProgress(3)
        for _ in range(3):
            progress.advance()
        self.assertEqual(progress.as_dict(), {"status": "ok", "requested_iterations": 3,
                                              "completed_iterations": 3, "stop_reason": None})
        self.assertEqual(completion_exit_code(progress.as_dict()), 0)

    def test_guard_before_any_step_is_partial(self):
        progress = RefinementProgress(30000)
        progress.stop("gpu_guard")
        report = normalize_refinement_report(progress.as_dict())
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["completed_iterations"], 0)
        self.assertEqual(report["stop_reason"], "gpu_guard")
        self.assertEqual(completion_exit_code(report), 3)

    def test_guard_after_500_steps_keeps_actual_count(self):
        progress = RefinementProgress(30000)
        for _ in range(500):
            progress.advance()
        progress.stop("gpu_guard")
        report = progress.as_dict()
        self.assertEqual(report["completed_iterations"], 500)
        self.assertEqual(report["requested_iterations"], 30000)
        self.assertEqual(report["status"], "partial")
        with self.assertRaises(ValueError):
            progress.advance()

    def test_no_usable_frames_is_partial(self):
        progress = RefinementProgress(100)
        progress.stop("no_usable_frames")
        self.assertEqual(progress.as_dict()["status"], "partial")
        self.assertEqual(completion_exit_code(progress.as_dict()), 3)

    def test_missing_steps_cannot_be_reported_ok(self):
        input_report = {"status": "ok", "requested_iterations": 30000, "completed_iterations": 500}
        report = normalize_refinement_report(input_report)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["stop_reason"], "iteration_budget_not_completed")
        self.assertEqual(input_report["status"], "ok")

    def test_disabled_and_legacy_backends(self):
        disabled = RefinementProgress(0).as_dict()
        self.assertEqual(disabled["status"], "disabled")
        self.assertEqual(completion_exit_code(disabled), 0)
        legacy = normalize_refinement_report(None)
        self.assertEqual(legacy, {"status": "ok"})
        self.assertEqual(completion_exit_code(legacy), 0)

    def test_explicit_failure_cannot_be_upgraded(self):
        report = normalize_refinement_report({"status": "failed", "requested_iterations": 3,
                                               "completed_iterations": 3, "stop_reason": "snapshot_failed"})
        self.assertEqual(report["status"], "failed")
        self.assertEqual(completion_exit_code(report), 3)

    def test_invalid_counts_are_rejected(self):
        for requested, completed in ((-1, 0), (1, -1), (1, 2), (1.5, 0), (True, 0)):
            with self.subTest(requested=requested, completed=completed):
                with self.assertRaises(ValueError):
                    RefinementProgress(requested, completed)


if __name__ == "__main__":
    unittest.main()
