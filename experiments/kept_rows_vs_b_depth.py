"""Where do the A rows that a B map keeps lie relative to what B's frames measure? (CPU, read-only: checkpoints + B's frames.)
For the A rows alive at B's end (not retired by the layer, not pruned by GaME), split by 'confirmed by B' (last_update >= B start),
each row's centre is projected into every K-th B frame (B's pose and intrinsics, the mapping's own reader update_layer.frames) and its
depth z compared with B's measured depth d at that pixel:  e = d - z
  e > 0   B measures beyond the row: B sees through it (the row stands in front of B's surface: a floater / an old surface)
  e < 0   B measures in front of the row: the row is hidden behind B's surface
  e ~ 0   the row is on B's surface
Printed per group: rows seen at least once, and over all (row, frame) observations the quantiles of e and the shares e > 5 cm,
|e| <= 5 cm, e < -5 cm (5 cm = the backend's support tolerance, used here only to report, not to decide).
usage: kept_rows_vs_b_depth.py A_CKPT DATASET SESSION FRAME_STRIDE PIXEL_STEP NAME=B_CKPT [NAME=B_CKPT ...]
"""
import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path('/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/worktrees/chain_clean')
sys.path.insert(0, str(TREE))
from update_layer.frames import FlatSession  # noqa: E402
from update_layer.run import dataset_config  # noqa: E402

a_path, ds, sess, stride, pstep = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
runs = [x.split('=', 1) for x in sys.argv[6:]]
cfg, info, specs = dataset_config(ds)
spec = next(sp for sp in specs if sp.name.endswith('_' + sess))
fs = FlatSession(spec, pixel_step=pstep)
frames = [fs.load(i) for i in range(0, len(fs.ids), stride)]
print(f'{ds} {sess}: {len(frames)} frames (every {stride}th, pixel_step {pstep})', flush=True)
ckA = torch.load(a_path, map_location='cpu', mmap=True, weights_only=False)
uidA = ckA['backend']['uid'].numpy()
for name, p in runs:
    ck = torch.load(p, map_location='cpu', mmap=True, weights_only=False)
    b = ck['backend']
    s0, s1 = int(b['session_start']), int(ck['prev_final'])
    scale = float(ck['backend']['model'][11])
    xyz = ck['backend']['model'][1].detach().numpy() / scale
    created, de, dp, ds_, lu = (b[k].numpy() for k in ('created', 'death_evidence', 'death_prune', 'death_state', 'last_update'))
    old = (created < s0) & np.isin(b['uid'].numpy(), uidA)
    alive_end = (de > s1) & (dp > s1) & (ds_ > s1)
    groups = {'kept, confirmed by B': old & alive_end & (lu >= s0), 'kept, not confirmed by B': old & alive_end & (lu < s0),
              'B-born, alive at B end': (created >= s0) & alive_end}
    print(f'\n== {name}: {p}')
    for gname, m in groups.items():
        P = xyz[m]
        es, seen = [], np.zeros(len(P), bool)
        for fr in frames:
            R, t = fr.T_world_cam[:3, :3], fr.T_world_cam[:3, 3]
            pc = (P - t) @ R                      # world -> camera (T_world_cam is camera-to-world)
            z = pc[:, 2]
            ok = z > 0.1
            K = fr.K
            u = np.floor(pc[ok, 0] / z[ok] * K.fx + K.cx - K.offset + 0.5).astype(np.int64)
            v = np.floor(pc[ok, 1] / z[ok] * K.fy + K.cy - K.offset + 0.5).astype(np.int64)
            h, w = fr.depth.shape
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            idx = np.where(ok)[0][inside]
            d = fr.depth[v[inside], u[inside]]
            val = np.isfinite(d)
            es.append(d[val] - z[idx[val]])
            seen[idx[val]] = True
        e = np.concatenate(es) if es else np.zeros(0)
        if not len(e):
            print(f'   {gname:26s} n={int(m.sum())} never in view'); continue
        q = np.percentile(e, [10, 50, 90])
        print(f'   {gname:26s} n={int(m.sum()):8d}, seen {int(seen.sum()):8d}; observations {len(e):9d}: e p10/p50/p90 '
              f'{100 * q[0]:7.1f} / {100 * q[1]:6.1f} / {100 * q[2]:6.1f} cm;  seen through (e > 5 cm) {100 * (e > 0.05).mean():5.1f} %, '
              f'on surface {100 * (np.abs(e) <= 0.05).mean():5.1f} %, hidden (e < -5 cm) {100 * (e < -0.05).mean():5.1f} %', flush=True)
print('KEPT_ROWS_DONE')
