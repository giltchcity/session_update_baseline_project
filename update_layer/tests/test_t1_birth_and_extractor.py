"""Unit tests (CPU) of the T1 lifetimes (backends/game/t1.py) and of the extractor's reconstruction confidence
(core/extractor.py, MeshObjectExtractor computeConfidence + volume gates).

  CUDA_VISIBLE_DEVICES= python -m update_layer.tests.test_t1_birth_and_extractor
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from update_layer.backends.game import t1  # noqa: E402
from update_layer.core.extractor import box_volume, reconstruction_confidence  # noqa: E402
from update_layer.frames import Intrinsics  # noqa: E402

S = 1_000_000_000
MAX = t1.INT64_MAX


def test_background_wall_seeded_late_is_seen_by_early_keyframes():
    # a wall Gaussian (identity 0) created at 100 s: the keyframe of 10 s must render it (the wall stood there)
    ident = torch.tensor([0, 0])
    created = torch.tensor([10 * S, 100 * S])
    sb = torch.full((2,), MAX)
    alive = t1.alive_at(10 * S, ident, created, sb, torch.full((2,), MAX), torch.full((2,), MAX))
    assert alive.tolist() == [True, True]
    # ... until the layer's evidence ends it
    alive = t1.alive_at(50 * S, ident, created, sb, torch.full((2,), MAX), torch.tensor([MAX, 40 * S]))
    assert alive.tolist() == [True, False]


def test_object_gaussian_lives_from_its_state_birth():
    # identity 7: state 1 born 5 s, ended 60 s; state 2 born 60 s, open. Gaussians created at 20 s, 59 s, 70 s, 3 s.
    ident = torch.tensor([7, 7, 7, 7])
    created = torch.tensor([20 * S, 59 * S, 70 * S, 3 * S])
    sb, ds = t1.state_membership(ident, created, {7: [(60 * S, None), (5 * S, 60 * S)]})
    assert sb.tolist() == [5 * S, 5 * S, 60 * S, MAX]           # 3 s: before the first state, no state
    assert ds.tolist() == [60 * S, 60 * S, MAX, MAX]
    de = torch.full((4,), MAX)
    # at 10 s the state-1 Gaussians exist (also the one GaME created at 59 s); the 3 s one exists since 3 s
    assert t1.alive_at(10 * S, ident, created, sb, ds, de).tolist() == [True, True, False, True]
    # at 65 s: state 1 ended, the state-2 Gaussian (created 70 s) exists from its state's birth 60 s
    assert t1.alive_at(65 * S, ident, created, sb, ds, de).tolist() == [False, False, True, True]


def test_unlabelled_and_untracked_keep_creation():
    ident = torch.tensor([-1, 9])                               # not yet labelled; an identity without intervals
    created = torch.tensor([30 * S, 30 * S])
    sb, ds = t1.state_membership(ident, created, {7: [(0, None)]})
    assert t1.birth(ident, created, sb).tolist() == [30 * S, 30 * S]


def _store(frames, H=40, W=40):
    """A pinhole store (fx = fy = 40, centre 20): frames = [(T_world_cam 4x4, range image m, code image)]."""
    K = Intrinsics(W, H, 40.0, 40.0, 20.0, 20.0)
    T = torch.stack([torch.linalg.inv(torch.as_tensor(f[0], dtype=torch.float32)) for f in frames])
    rng = torch.stack([torch.round(torch.as_tensor(f[1], dtype=torch.float32) * 1000).to(torch.int32) for f in frames])
    code = torch.stack([torch.as_tensor(f[2], dtype=torch.int32) for f in frames])
    return SimpleNamespace(K=K, T=T, rng=rng, code=code, n=len(frames))


def _plane_frame(z_obj, label_obj, label_wall, H=40, W=40, z_wall=2.0, obj_cols=(15, 25)):
    """Camera at the origin looking down +z; object patch at depth z_obj in columns obj_cols, wall behind."""
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    x, y = (u - 20) / 40, (v - 20) / 40
    obj = (u >= obj_cols[0]) & (u < obj_cols[1])
    z = torch.where(obj, torch.full_like(u, z_obj), torch.full_like(u, z_wall))
    rng = z * torch.sqrt(x * x + y * y + 1)
    code = torch.where(obj, torch.full_like(u, label_obj), torch.full_like(u, label_wall)).to(torch.int32)
    return torch.eye(4), rng, code


def _points(rng_img, cols, T=None):
    H, W = rng_img.shape
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    m = (u >= cols[0]) & (u < cols[1])
    x, y = (u[m] - 20) / 40, (v[m] - 20) / 40
    d = rng_img[m] / torch.sqrt(x * x + y * y + 1)
    return torch.stack([x * d, y * d, d], 1)


def test_bleed_onto_the_wall_is_erased_and_the_box_shrinks():
    # identity 5 = an object at 1 m (columns 15-25); in frame 0 its mask bleeds onto the wall at 2 m (columns 25-35)
    f0 = _plane_frame(1.0, 5, 0)
    f0[2][:, 25:35] = 5                                          # the bleed: wall pixels labelled 5
    f1 = _plane_frame(1.0, 5, 0)
    f2 = _plane_frame(1.0, 5, 0)
    store = _store([f0, f1, f2])
    obj = _points(f0[1], (15, 25))
    bleed = _points(f0[1], (25, 35))
    pts = torch.cat([obj, bleed])
    conf = reconstruction_confidence(store, None, 5, pts, [0, 1, 2], 0.02)
    assert (conf[:len(obj)] >= 0.5).all()                        # object voxels: 3 of 3 observations labelled 5
    assert (conf[len(obj):] < 0.5).all()                         # bleed voxels: 1 of 3
    keep = conf >= 0.5
    assert box_volume(pts[keep]) < box_volume(pts)


def test_dynamic_pixels_are_skipped_and_unobserved_voxels_have_zero_confidence():
    f0 = _plane_frame(1.0, 5, 0)
    f1 = _plane_frame(1.0, 5, 0)
    f1[2][:, 15:25] = 0                                          # frame 1 labels the object background ...
    store = _store([f0, f1])
    obj = _points(f0[1], (15, 25))
    rej = [None, torch.zeros(40, 40, dtype=torch.bool)]
    rej[1][:, 15:25] = True                                      # ... where it is dynamic: skipped
    assert (reconstruction_confidence(store, rej, 5, obj, [0, 1], 0.02) == 1.0).all()
    assert (reconstruction_confidence(store, None, 5, obj, [0, 1], 0.02) == 0.5).all()
    far = obj + torch.tensor([0.0, 0.0, 0.5])                    # 50 cm behind the surface: never in a band
    assert (reconstruction_confidence(store, None, 5, far, [0, 1], 0.02) == 0.0).all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
