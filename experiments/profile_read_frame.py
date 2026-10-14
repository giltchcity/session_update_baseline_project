"""Where the time of evlight's [L1] _read_frame goes (GPU short, read-only).

The accepted A is loaded as the chain loads it, B's session is started (the snapshot is built), and for N frames of B the parts
of GameBackend._read_frame are timed with CUDA synchronisation:
  _sample (the frame re-read with colour, cropped)
  the GPU keyframe dict
  _StoredKeyframe.of
  GaME's _add_gaussians (render, seeding mask, point cloud, add_points with its global nearest-neighbour scales)
  _seed_identity
  _support (render)
The map is not trained in between, so the seeds accumulate as they would between two pose keyframes.

usage: profile_read_frame.py CODE_DIR DATASET A_CKPT FIRST STEP N
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE, DS, A_CKPT = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
FIRST, STEP, N = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
sys.path.insert(0, str(CODE))
from update_layer.run import dataset_config  # noqa: E402
from update_layer.backends.game import game as G  # noqa: E402
from update_layer.eval.kf_strip import restore  # noqa: E402

cfg, info, specs = dataset_config(DS)
spec_b = next(sp for sp in specs if sp.name.endswith('_b'))
be = G.GameBackend(info, own_update=False)
ck = torch.load(A_CKPT, map_location='cpu', weights_only=False)
g = be.prior_from_state(restore(dict(backend=ck['backend']), DS)['backend'])
be.start_session(spec_b, g)
g.now = be.session_start
parts = {k: [] for k in ('sample', 'gpu dict', 'stored', 'add_gaussians', 'seed_identity', 'support')}
rows = []


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


for i in range(FIRST, FIRST + STEP * N, STEP):
    fr = be.session.load(i)
    pose = np.linalg.inv(fr.T_world_cam)
    pose[:3, 3] *= be.scale
    pose = pose.astype(np.float32)
    t0 = tick()
    sample, instance, dyn = be._sample(i)
    t1 = tick()
    kf = {"color": G.gu.np2torch(sample["color"], device="cuda").permute(2, 0, 1) / 255.0,
          "depth": G.gu.np2torch(sample["depth"], device="cuda"),
          "masks": sample["masks"].cuda(),
          "pose": G.gu.np2torch(pose, device="cuda"),
          "intrinsics": sample["intrinsics"]}
    t2 = tick()
    stored = G._StoredKeyframe.of(kf, sample)
    t3 = tick()
    n0 = len(g.uid)
    g.gaussian_model.alive = g.alive_at(g.now)
    g._add_gaussians(kf["color"], kf["depth"], None, kf["pose"], kf["intrinsics"])
    g.gaussian_model.alive = None
    t4 = tick()
    be._seed_identity(instance, pose, sample["intrinsics"])
    t5 = tick()
    g.gaussian_model.alive = g.alive_at(g.now)
    be._support(g, kf, fr.stamp_ns)
    g.gaussian_model.alive = None
    t6 = tick()
    for k, a, b in (('sample', t0, t1), ('gpu dict', t1, t2), ('stored', t2, t3), ('add_gaussians', t3, t4),
                    ('seed_identity', t4, t5), ('support', t5, t6)):
        parts[k].append(b - a)
    rows.append(len(g.uid) - n0)
    del stored
print(f'{DS}: {N} frames from {FIRST} every {STEP}; map rows {len(g.uid)}; seeded per frame: median {int(np.median(rows))}, '
      f'max {max(rows)}')
tot = sum(np.mean(v) for v in parts.values())
for k, v in parts.items():
    print(f'   {k:14s} mean {1000 * np.mean(v):7.1f} ms  median {1000 * np.median(v):7.1f} ms  ({100 * np.mean(v) / tot:4.1f} %)')
print(f'   total          mean {1000 * tot:7.1f} ms per frame')
print('PROFILE_READ_FRAME_DONE')
