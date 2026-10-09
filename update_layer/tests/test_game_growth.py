"""Regression checks for one-pass seeding and protected GaME rows.

CPU (NumPy executes the production mask functions; torch checks run if installed):
  python -m unittest update_layer.tests.test_game_growth -v

On the original machine, also exercise actual GaME appends/clone/split on CUDA:
  RUN_GAME_CUDA_TESTS=1 python -m unittest update_layer.tests.test_game_growth -v

The CUDA checks use the real renderer and Gaussian model, not fabricated render
results. They need the upstream GaME checkout and compiled extensions.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from update_layer.backends.game.growth import (clear_densification_rows, eligible_densification_mask,
                                               merge_seed_masks, restored_session_settings)


class TestGrowthNumpy(unittest.TestCase):
    def test_same_pixel_requested_by_both_paths_is_inserted_once(self):
        game = np.array([[True, True, False, False]])
        front = np.array([[True, False, True, False]])
        explained = np.array([[False, True, False, False]])
        selected = merge_seed_masks(game, front, explained)
        np.testing.assert_array_equal(selected, [[True, False, True, False]])
        # The overlap creates one row request; the genuinely new, front-only surface survives.
        self.assertEqual(np.count_nonzero(selected), 2)

    def test_no_front_request_keeps_existing_game_mask_rule(self):
        game = np.array([True, True, False, False])
        explained = np.array([True, False, False, True])
        np.testing.assert_array_equal(merge_seed_masks(game, None, explained), game & ~explained)

    def test_stale_high_gradient_cannot_select_protected_parents(self):
        # Both protected rows have stronger stale gradients than the eligible row.
        gradients = np.array([100.0, 90.0, 2.0, 0.01])
        frozen = np.array([True, False, False, False])
        carried = np.array([False, True, False, False])
        np.testing.assert_array_equal(
            eligible_densification_mask(gradients >= 1.0, frozen, carried),
            [False, False, True, False])

    def test_freezing_clears_only_protected_density_statistics(self):
        model = SimpleNamespace(xyz_gradient_accum=np.array([[10.0], [20.0], [30.0]]),
                                denom=np.array([[1.0], [2.0], [3.0]]),
                                max_radii2D=np.array([4.0, 5.0, 6.0]))
        clear_densification_rows(model, np.array([True, False, True]))
        np.testing.assert_array_equal(model.xyz_gradient_accum, [[0.0], [20.0], [0.0]])
        np.testing.assert_array_equal(model.denom, [[0.0], [2.0], [0.0]])
        np.testing.assert_array_equal(model.max_radii2D, [0.0, 5.0, 0.0])

    def test_density_statistics_reject_a_misaligned_mask(self):
        model = SimpleNamespace(xyz_gradient_accum=np.zeros((3, 1)),
                                denom=np.zeros((3, 1)), max_radii2D=np.zeros(3))
        with self.assertRaisesRegex(ValueError, "mask has 2"):
            clear_densification_rows(model, np.array([True, False]))

    def test_checkpoint_restores_explicit_session_and_band(self):
        self.assertEqual(restored_session_settings(
            {"session_start": 20, "support_tol": 0.5, "session_starts": [0, 10]}, 0.05), (20, 0.5))

    def test_legacy_checkpoint_uses_recorded_last_boundary(self):
        self.assertEqual(restored_session_settings({"session_starts": [0, 10, 20]}, 0.05), (20, 0.05))

    def test_missing_boundary_is_not_invented_from_row_births(self):
        self.assertEqual(restored_session_settings({"created": [10, 20]}, 0.05), (None, 0.05))


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is not installed")
class TestGrowthTorch(unittest.TestCase):
    def test_boolean_masks_match_numpy_on_real_tensors(self):
        import torch
        game = torch.tensor([[True, True, False, False]])
        front = torch.tensor([[True, False, True, False]])
        explained = torch.tensor([[False, True, False, False]])
        self.assertEqual(merge_seed_masks(game, front, explained).tolist(), [[True, False, True, False]])
        self.assertEqual(eligible_densification_mask(
            torch.tensor([True, True, True, False]), torch.tensor([True, False, False, False]),
            torch.tensor([False, True, False, False])).tolist(), [False, False, True, False])

    def test_restored_tensor_statistics_are_cleared_selectively(self):
        import torch
        model = SimpleNamespace(xyz_gradient_accum=torch.tensor([[10.0], [20.0], [30.0]]),
                                denom=torch.tensor([[1.0], [2.0], [3.0]]),
                                max_radii2D=torch.tensor([4.0, 5.0, 6.0]))
        clear_densification_rows(model, torch.tensor([True, False, True]))
        self.assertEqual(model.xyz_gradient_accum.tolist(), [[0.0], [20.0], [0.0]])
        self.assertEqual(model.denom.tolist(), [[0.0], [2.0], [0.0]])
        self.assertEqual(model.max_radii2D.tolist(), [0.0, 5.0, 0.0])


@unittest.skipUnless(os.environ.get("RUN_GAME_CUDA_TESTS") == "1", "set RUN_GAME_CUDA_TESTS=1 on the GaME machine")
class TestGaMECudaIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        import yaml
        from update_layer.backends.game import game
        cls.torch, cls.game = torch, game
        cls.config = yaml.safe_load((Path(game.__file__).parent / "configs/kinect_real.yaml").read_text())["game"]
        cls.config.setdefault("scale", 1.0)

    def _new_model(self, xyz):
        import open3d as o3d
        g = self.game.TrackedGaME(dict(self.config))
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(xyz)
        cloud.colors = o3d.utility.Vector3dVector(np.full_like(xyz, 0.1))
        self.game.gu.add_points(g.gaussian_model, cloud)
        g.identity.fill_(0)
        g.timed, g.now, g.session_start, g.support_tol = True, 10, 10, 0.05
        return g

    def _density_model(self):
        xyz = np.array([[0.03 * i, 0.02 * (i % 2), 2.0] for i in range(8)])
        g = self._new_model(xyz)
        # Row 0 is frozen, row 1 is carried; all remaining rows belong to this session.
        g.created.fill_(10)
        g.created[1] = 0
        g.frozen[0] = True
        return g

    def test_clone_and_split_never_use_frozen_or_carried_parents(self):
        torch = self.torch
        for split in (False, True):
            with self.subTest(split=split):
                g = self._density_model()
                gm = g.gaussian_model
                original_ids = g.uid.clone()
                cutoff = float(gm.percent_dense)
                with torch.no_grad():
                    gm._scaling.fill_(np.log(cutoff * (2.0 if split else 0.5)))
                    gradients = torch.full((8, 1), 10.0, device="cuda")
                    gradients[:2] = 100.0  # old, high gradients must not override protection
                    if split:
                        gm.densify_and_split(gradients, 1.0, 1.0)
                    else:
                        gm.densify_and_clone(gradients, 1.0, 1.0, limit_num=-1)
                self.assertEqual(len(g.uid), 14)  # six eligible parents clone, or split into two
                self.assertTrue(torch.isin(original_ids[:2], g.uid).all().item())
                self.assertEqual(int((g.created < g.session_start).sum()), 1)
                self.assertEqual(int(g.frozen.sum()), 1)

    def test_same_frame_front_and_game_requests_use_one_real_append(self):
        from unittest.mock import patch
        torch = self.torch
        h, w, focal = 24, 32, 40.0
        v, u = np.mgrid[:h, :w]
        k = np.array([[focal, 0, (w - 1) / 2], [0, focal, (h - 1) / 2], [0, 0, 1]])
        xyz = np.stack([(u.ravel() - k[0, 2]) * 2 / focal,
                        (v.ravel() - k[1, 2]) * 2 / focal, np.full(h * w, 2.0)], 1)
        g = self._new_model(xyz)
        gm = g.gaussian_model
        with torch.no_grad():
            gm._scaling.fill_(np.log(0.03))
            gm._opacity.fill_(np.log(0.95 / 0.05))
            color = torch.full((3, h, w), 0.9, device="cuda")
            depth = torch.full((h, w), 1.5, device="cuda")
            pose = torch.eye(4, device="cuda")
            gm.alive = g.alive_at(g.now)
            front = g._front_of_carried_mask(color, depth, pose, k)
            self.assertGreater(int(front.sum()), 0)
            # Spy only: execute the actual CUDA point insertion, with no fake render or return value.
            with patch.object(self.game.gu, "add_points", wraps=self.game.gu.add_points) as append:
                g._add_gaussians(color, depth, None, pose, k)
                self.assertEqual(append.call_count, 1)
            gm.alive = None
            self.assertGreater(len(g.uid), len(xyz))


if __name__ == "__main__":
    unittest.main(verbosity=2)
