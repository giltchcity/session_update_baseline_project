"""Real-tensor regression tests for GaME's reversible suppression history.

  python -m unittest update_layer.tests.test_game_lifecycle -v
  RUN_GAME_CUDA_TESTS=1 python -m unittest update_layer.tests.test_game_lifecycle -v

The optional integration test uses the actual GaME model and CUDA dependencies.
It is skipped, not simulated, on environments without the original GaME runtime.
"""
from __future__ import annotations

import io
import os
import unittest

import torch

from update_layer.backends.game.lifecycle import (EVENT_BYTES, INT64_MAX, SuppressionHistory,
                                                  apply_surface_weights, recoverable_lifetime)


class TestSuppressionHistory(unittest.TestCase):
    def test_half_open_gap_then_same_uid_returns(self):
        h = SuppressionHistory()
        uids = torch.tensor([10, 20, 30])
        self.assertEqual(h.suppress([20], 100), 1)
        self.assertEqual(h.mask_at(uids, 99).tolist(), [False, False, False])
        self.assertEqual(h.mask_at(uids, 100).tolist(), [False, True, False])
        self.assertEqual(h.reactivate([20], 200), 1)
        self.assertEqual(h.mask_at(uids, 199).tolist(), [False, True, False])
        self.assertEqual(h.mask_at(uids, 200).tolist(), [False, False, False])
        self.assertEqual(h.mask_at(uids, 300).tolist(), [False, False, False])
        self.assertEqual(h.uids.tolist(), [20])
        self.assertEqual(h.open_count, 0)

    def test_repeated_suppression_is_idempotent(self):
        h = SuppressionHistory()
        self.assertEqual(h.suppress([20, 20], 100), 1)
        self.assertEqual(h.suppress([20], 150), 0)
        self.assertEqual(h.starts.tolist(), [100])
        self.assertEqual(h.reactivate([999], 160), 0)
        self.assertEqual(h.reactivate([20], 200), 1)
        self.assertEqual(h.reactivate([20], 220), 0)

    def test_same_stamp_corrections_reclaim_capacity_without_erasing_older_gap(self):
        h = SuppressionHistory(max_bytes=2 * EVENT_BYTES)
        h.suppress([1], 10)
        h.reactivate([1], 20)
        for stamp in range(30, 50):
            h.suppress([1], stamp)
            h.reactivate([1], stamp)
            self.assertEqual(h.nbytes, EVENT_BYTES)
        self.assertEqual(h.mask_at(torch.tensor([1]), 15).tolist(), [True])

    def test_backdated_permanent_end_does_not_backdate_the_operational_gap(self):
        h = SuppressionHistory()
        h.suppress([1], 50)
        h.close_for_permanent_end([1], 80)  # the permanent T1 end itself may be at t=10
        self.assertEqual(h.starts.tolist(), [50])
        self.assertEqual(h.ends.tolist(), [80])
        self.assertEqual(h.open_count, 0)
        # The permanent base end wins at every timestamp >=10, regardless of the stored gap.
        for stamp in (9, 10, 49, 50, 79, 80):
            base_alive = torch.tensor([stamp < 10])
            self.assertEqual((base_alive & ~h.mask_at(torch.tensor([1]), stamp)).tolist(), [stamp < 10])

    def test_multiple_cycles_preserve_both_historical_gaps(self):
        h = SuppressionHistory()
        h.suppress([1], 10)
        h.reactivate([1], 20)
        h.suppress([1], 30)
        h.reactivate([1], 40)
        ids = torch.tensor([1])
        self.assertEqual([bool(h.mask_at(ids, t)[0]) for t in [9, 10, 19, 20, 29, 30, 39, 40]],
                         [False, True, True, False, False, True, True, False])

    def test_uid_lookup_survives_reorder_prune_and_new_rows(self):
        h = SuppressionHistory()
        h.suppress([10, 30], 10)
        rows = torch.tensor([30, 10, 20])
        self.assertEqual(h.current_mask(rows).tolist(), [True, True, False])
        rows[:] = torch.tensor([10, 20, 30])  # in-place UID changes invalidate cached row lookup
        self.assertEqual(h.current_mask(rows).tolist(), [True, False, True])
        self.assertEqual(h.current_mask(torch.tensor([20, 30, 40])).tolist(), [False, True, False])
        self.assertEqual(h.current_mask(torch.empty(0, dtype=torch.int64)).tolist(), [])

    def test_drop_reclaims_only_removed_uid_history(self):
        h = SuppressionHistory()
        h.suppress([1, 2], 10)
        h.reactivate([1], 20)
        h.drop_uids([2, 999])
        self.assertEqual(h.nbytes, EVENT_BYTES)
        self.assertEqual(h.open_count, 0)
        self.assertEqual(h.mask_at(torch.tensor([1]), 15).tolist(), [True])
        self.assertEqual(h.mask_at(torch.tensor([1]), 20).tolist(), [False])

    def test_clone_and_split_children_inherit_gaps_without_changing_parent(self):
        h = SuppressionHistory()
        h.suppress([1, 2], 10)
        h.reactivate([1, 2], 20)
        # Two children of row 1, one of row 2; no history for row 3's child.
        events = h.prepare_inheritance(torch.tensor([1, 1, 2, 3]), torch.tensor([10, 11, 12, 13]))
        self.assertEqual(len(h.uids), 2)  # preparing metadata does not mutate the history or model
        h.append_events(events)
        rows = torch.tensor([1, 2, 3, 10, 11, 12, 13])
        self.assertEqual(h.mask_at(rows, 15).tolist(), [True, True, False, True, True, True, False])
        self.assertFalse(bool(h.mask_at(rows, 20).any()))
        h.suppress([10], 30)
        self.assertEqual(h.current_mask(rows).tolist(), [False, False, False, True, False, False, False])

    def test_capacity_rejection_is_before_any_event_mutation(self):
        h = SuppressionHistory(max_bytes=2 * EVENT_BYTES)
        h.suppress([1, 2], 10)
        before = tuple(t.clone() for t in (h.uids, h.starts, h.ends))
        with self.assertRaisesRegex(RuntimeError, "budget exceeded"):
            h.suppress([3], 20)
        for actual, saved in zip((h.uids, h.starts, h.ends), before):
            self.assertTrue(torch.equal(actual, saved))
        self.assertEqual(h.last_stamp, 10)
        with self.assertRaisesRegex(RuntimeError, "budget exceeded"):
            h.prepare_inheritance(torch.tensor([1]), torch.tensor([100]))
        self.assertEqual(len(h.uids), 2)

    def test_checkpoint_round_trip_preserves_history_and_limit(self):
        h = SuppressionHistory(max_bytes=12 * EVENT_BYTES)
        h.suppress([1], 10)
        h.reactivate([1], 20)
        h.suppress([2], 30)
        buffer = io.BytesIO()
        torch.save(h.state_dict(), buffer)
        buffer.seek(0)
        restored = SuppressionHistory.from_state(torch.load(buffer, weights_only=False))
        self.assertEqual(restored.max_bytes, 12 * EVENT_BYTES)
        self.assertEqual(restored.open_count, 1)
        rows = torch.tensor([1, 2])
        for stamp in (5, 10, 19, 20, 29, 30, 40):
            self.assertTrue(torch.equal(restored.mask_at(rows, stamp), h.mask_at(rows, stamp)))
        restored.reactivate([2], 40)
        self.assertEqual(h.open_count, 1)  # independent restored storage

    def test_relocation_preserves_history_under_original_uids(self):
        memory, present = SuppressionHistory(), SuppressionHistory()
        memory.suppress([1, 2], 10)
        memory.reactivate([1], 20)
        transferred = memory.selected_events([1])
        present.ensure_capacity(len(transferred[0]))
        present.append_events(transferred)
        memory.drop_uids([1])
        self.assertEqual(memory.uids.tolist(), [2])
        self.assertEqual(present.uids.tolist(), [1])
        self.assertEqual(present.mask_at(torch.tensor([1, 100]), 15).tolist(), [True, False])
        self.assertEqual(present.mask_at(torch.tensor([1, 100]), 20).tolist(), [False, False])
        merged = SuppressionHistory()
        for source in (memory, present):
            merged.append_events((source.uids, source.starts, source.ends))
        self.assertEqual(merged.mask_at(torch.tensor([1, 2]), 15).tolist(), [True, True])
        self.assertEqual(merged.mask_at(torch.tensor([1, 2]), 20).tolist(), [False, True])

    def test_closed_identity_or_permanent_death_cannot_be_recovered(self):
        alive = torch.tensor([True, True, True, True, False, True])
        identity = torch.tensor([0, 5, 5, 0, 0, 5])
        state_end = torch.tensor([INT64_MAX, INT64_MAX, 100, INT64_MAX, INT64_MAX, INT64_MAX])
        evidence_end = torch.tensor([INT64_MAX, INT64_MAX, INT64_MAX, 100, INT64_MAX, INT64_MAX])
        prune_end = torch.tensor([INT64_MAX, INT64_MAX, INT64_MAX, INT64_MAX, INT64_MAX, 100])
        self.assertEqual(recoverable_lifetime(alive, identity, state_end, evidence_end, prune_end).tolist(),
                         [True, True, False, False, False, False])

    def test_canonical_weight_updates_use_uid_and_validate_before_writing(self):
        ids, weights = torch.tensor([30, 10, 20]), torch.ones(3)
        changed = apply_surface_weights(ids, weights, torch.tensor([20, 30]), torch.tensor([8.0, -1.0]))
        self.assertEqual(changed, 2)
        self.assertEqual(weights.tolist(), [-1.0, 1.0, 8.0])
        for values in ([float("nan")], [-2.0], [9.0]):
            with self.assertRaisesRegex(ValueError, "finite"):
                apply_surface_weights(ids, weights, [10], values)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            apply_surface_weights(ids, weights, [10, 10], [1.0, 2.0])
        self.assertEqual(weights.tolist(), [-1.0, 1.0, 8.0])

    def test_invalid_history_and_out_of_order_events_fail_explicitly(self):
        h = SuppressionHistory()
        h.suppress([1], 20)
        with self.assertRaisesRegex(ValueError, "timestamp order"):
            h.reactivate([1], 19)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            SuppressionHistory.from_state(dict(uids=[1, 1], starts=[10, 15], ends=[30, 40], last_stamp=40))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            SuppressionHistory.from_state(dict(uids=[1], starts=[10], ends=[30], last_stamp=20))
        with self.assertRaisesRegex(ValueError, "duplicate stable UIDs"):
            h.current_mask(torch.tensor([1, 1]))
        for invalid in ("30", 30.5, True, INT64_MAX):
            with self.assertRaisesRegex(ValueError, "timestamp"):
                SuppressionHistory.from_state(dict(uids=[1], starts=[10], ends=[20], last_stamp=invalid))
        with self.assertRaisesRegex(ValueError, "int64"):
            SuppressionHistory.from_state(dict(uids=[1], starts=[10.5], ends=[20], last_stamp=20))


@unittest.skipUnless(os.environ.get("RUN_GAME_CUDA_TESTS") == "1", "set RUN_GAME_CUDA_TESTS=1 on the GaME machine")
class TestGaMELifecycleCuda(unittest.TestCase):
    def test_actual_backend_parks_restores_and_preserves_history_and_hard_ends(self):
        import numpy as np
        import open3d as o3d
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        from update_layer.backends.game.game import GameBackend, TrackedGaME, gu
        from update_layer.interface import DatasetInfo
        backend = GameBackend(DatasetInfo("real", (), (0.1, 5.0)), own_update=False)
        g = backend.game = TrackedGaME(dict(backend.config))
        g.history_capacity_guard = backend._check_history_budget
        points = np.array([[0.03 * i, 0.02 * (i % 2), 2.0] for i in range(8)])
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points * backend.scale)
        cloud.colors = o3d.utility.Vector3dVector(np.full_like(points, 0.5))
        gu.add_points(g.gaussian_model, cloud)
        g.timed, g.session_start, g.now = True, 0, 0
        g.identity.fill_(0)
        ids = g.uid.clone()
        xyz = g.gaussian_model.get_xyz.detach().clone()
        permanent_frozen = g.frozen.clone()
        self.assertEqual(backend.suppress(ids[:2], 10), 2)
        self.assertEqual(g.surface_weight[:2].tolist(), [-1.0, -1.0])
        self.assertEqual(len(backend.elements()), 6)
        evidence = backend.evidence_elements()
        self.assertEqual(len(evidence), 8)
        self.assertEqual(int(evidence.suppressed.sum()), 2)
        self.assertTrue(bool(g.alive_at(9).all()))
        self.assertFalse(bool(g.alive_at(10)[:2].any()))
        self.assertFalse(bool(g._densify_policy(torch.ones(8, dtype=torch.bool, device="cuda"))[:2].any()))
        self.assertTrue(torch.equal(g.frozen, permanent_frozen))
        self.assertEqual(backend.reactivate(ids[:2], 20), 2)
        self.assertEqual(len(backend.elements()), 8)
        self.assertTrue(torch.equal(g.uid, ids))
        self.assertTrue(torch.equal(g.gaussian_model.get_xyz, xyz))
        self.assertFalse(bool(g.alive_at(19)[:2].any()))
        self.assertTrue(bool(g.alive_at(20).all()))
        self.assertEqual(g.last_update[:2].tolist(), [20, 20])
        self.assertEqual(g.surface_weight[:2].tolist(), [1.0, 1.0])
        self.assertTrue(bool(g._densify_policy(torch.ones(8, dtype=torch.bool, device="cuda")).all()))
        # This is the real positive-only hook called before every keyframe's addition/seeding.
        from update_layer.frames import Frame, Intrinsics
        self.assertEqual(backend.suppress(ids[:2], 30), 2)
        frame = Frame(1, 40, np.full((24, 32), 2.0, dtype=np.float32), np.zeros((24, 32), dtype=np.int64),
                      None, np.eye(4), Intrinsics(32, 24, 40.0, 40.0, 15.5, 11.5))
        backend._recover_before_seeding(frame)
        self.assertTrue(bool(g.alive_at(40).all()))
        self.assertEqual(backend.lifecycle_stats["positive_reactivations"], 2)
        self.assertTrue(torch.equal(g.uid, ids))
        # Confidence, as well as the historical gaps, follows actual clone children.
        backend.update_surface_weights(ids[2:3], torch.tensor([8.0], device="cuda"))
        with torch.no_grad():
            gm = g.gaussian_model
            gm._scaling.fill_(np.log(float(gm.percent_dense) * 0.5))
            gradients = torch.zeros((8, 1), device="cuda")
            gradients[2] = 10
            gm.densify_and_clone(gradients, 1.0, 1.0, limit_num=-1)
        self.assertEqual(float(g.surface_weight[-1]), 8.0)
        # A later permanent end is independent of the reversible interval and cannot be undone.
        self.assertEqual(backend.suppress(ids[:2], 50), 2)
        g.kf_stamp = {0: 0}  # keep the rows for their historical view during a backdated G5 retirement
        backend.retire(ids[:1], 5)
        g.identity[1], g.state_birth[1], g.death_state[1] = 5, 0, 5
        backend._freeze_ended(5)
        self.assertEqual(backend.reactivate(ids[:2], 60), 0)
        self.assertFalse(bool(g.alive_at(60)[:2].any()))
        self.assertTrue(bool(g.frozen[:2].all()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
