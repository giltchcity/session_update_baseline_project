"""TODO 1 with the TSDF's own verdict: what does D1's voxel state say at each row, over time? (no new rule, no approximation)

D1 = the layer's port of the Khronos ActiveWindow motion pipeline with the Hydra projective TSDF integrator (core/d1.py). It is run
here exactly as the chain runs it: one D1 per session, D1Config built as layer.py:413 from run.dataset_config, process(frame) on
every backend frame in order (run.py:236; frames = FlatSession(spec, pixel_step=run.INPUT_STEP[dataset]) as run.py:201).
Every QUERY-th frame each row alive at that stamp is looked up in D1's voxels (D1._voxel on the row centre's global voxel):
  OCCUPIED  the voxel's last_occ is this stamp (D1._update_tracking: fused distance < -occupancy_threshold x voxel)
  FREE      D1._is_free: observed (last_obs != 0) and not occupied within temporal_buffer (Khronos' free space)
Time order: a row is ended at the first FREE verdict after its last OCCUPIED verdict (rows never occupied are not ended).
usage: d1_voxels.py CKPT DATASET SESSION [REFERENCE_PLY|-] [QUERY] [MAX_FRAMES]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from update_layer.run import dataset_config, INPUT_STEP                                   # noqa: E402
from update_layer.frames import FlatSession, synthetic_session, real_session             # noqa: E402
from update_layer.core.d1 import D1, D1Config, DEV                                       # noqa: E402

INT64_MAX = 9223372036854775807
SCALE = 10.0                     # flat.yaml game.scale: model coordinates are metres x 10
ckpt, DATASET, SESSION = sys.argv[1], sys.argv[2], sys.argv[3]
REF = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != '-' else None
QUERY = int(sys.argv[5]) if len(sys.argv) > 5 else 10
MAX_FRAMES = int(sys.argv[6]) if len(sys.argv) > 6 else 0
cfg, info, specs = dataset_config(DATASET)
d1 = D1(D1Config(voxel_size=cfg.map_resolution, truncation=cfg.truncation, max_range=cfg.max_range,
                 dynamic_labels=tuple(cfg.dynamic_semantics)))                          # layer.py:413
make = synthetic_session if DATASET == 'synthetic' else real_session
fs = FlatSession(make(SESSION), pixel_step=INPUT_STEP[DATASET])                        # run.py:201

ck = torch.load(ckpt, mmap=True, map_location='cpu', weights_only=False)
b, t_end, m = ck['backend'], int(ck['prev_final']), ck['backend']['model']
end = np.minimum(np.minimum(b['death_state'].numpy(), b['death_evidence'].numpy()), b['death_prune'].numpy())
sb = b['state_birth'].numpy()
born = np.maximum(b['created'].numpy(), np.where(sb < INT64_MAX, sb, -INT64_MAX))
idx = np.flatnonzero((born <= t_end) & (end > t_end))
xyz = (m[1].detach().numpy() / SCALE)[idx].astype(np.float64)
ident, born, end = b['identity'].numpy()[idx], born[idx], end[idx]
n = len(idx)
g_all = torch.as_tensor(np.floor(xyz / cfg.map_resolution).astype(np.int64), device=DEV)
last_occ_row = np.full(n, -1, np.int64)
first_free = np.full(n, INT64_MAX, np.int64)
n_occ = np.zeros(n, np.int32)
n_free = np.zeros(n, np.int32)
indices = list(fs.indices())[:MAX_FRAMES] if MAX_FRAMES else list(fs.indices())
t0 = time.monotonic()
for k, i in enumerate(indices):
    frame = fs.load(i)
    d1.process(frame)
    if k % QUERY:
        continue
    stamp = int(frame.stamp_ns)
    live = np.flatnonzero((born <= stamp) & (end > stamp))
    slot, lin = d1._voxel(g_all[live])
    ok = (slot >= 0).cpu().numpy()
    live, s_, l_ = live[ok], slot[slot >= 0], lin[slot >= 0]
    occ = (d1.last_occ[s_, l_] == stamp).cpu().numpy()
    # D1._is_free (d1.py:698) for the row's own voxel: observed and not occupied within the temporal buffer
    free = ((d1.last_occ[s_, l_].to(torch.float64) * 1e-9 < stamp * 1e-9 - d1.cfg.temporal_buffer) & (d1.last_obs[s_, l_] != 0)).cpu().numpy()
    o, f_ = live[occ], live[free & ~occ]
    last_occ_row[o] = stamp
    first_free[o] = INT64_MAX
    n_occ[o] += 1
    n_free[f_] += 1
    newly = f_[(last_occ_row[f_] >= 0) & (first_free[f_] == INT64_MAX)]
    first_free[newly] = stamp
    if k % (QUERY * 50) == 0:
        print(f'frame {k}/{len(indices)}  {(k + 1) / (time.monotonic() - t0):.2f} frames/s  blocks {len(d1.slots)}', flush=True)
print(f'D1 speed: {len(indices) / (time.monotonic() - t0):.2f} frames/s over {len(indices)} frames on {DEV}', flush=True)
ended = first_free < INT64_MAX


def report(name, sel):
    if not sel.any():
        print(f'  {name:36s} 0 rows'); return
    print(f'  {name:36s} {sel.sum():8d} rows | ended {ended[sel].sum():7d} = {ended[sel].mean() * 100:5.1f} % | '
          f'ever occupied {np.mean(n_occ[sel] > 0) * 100:5.1f} % | ever free {np.mean(n_free[sel] > 0) * 100:5.1f} %', flush=True)


print(f'{DATASET} {SESSION}: rows alive at the end {n}; frames {len(indices)}; query every {QUERY}; D1 voxel {cfg.map_resolution} m, '
      f'truncation {cfg.truncation} m')
report('ALL rows', np.ones(n, bool))
if DATASET == 'synthetic' and SESSION == 'b':
    f3447 = fs.load(3447)
    K, z = f3447.K, f3447.depth
    vv, uu = np.indices(z.shape, dtype=np.float32)
    okp = (f3447.instance == 109) & np.isfinite(z) & (z > 0)
    cam = np.stack(((uu[okp] + K.offset - K.cx) / K.fx * z[okp], (vv[okp] + K.offset - K.cy) / K.fy * z[okp], z[okp], np.ones(okp.sum())), 1)
    plant = cKDTree((cam @ f3447.T_world_cam.T)[:, :3]).query(xyz, k=1)[0]
    report('plant 109 <=3cm of its 3447 surface', (ident == 109) & (plant <= 0.03))
    report('plant 109 3-10cm', (ident == 109) & (plant > 0.03) & (plant <= 0.10))
    report('plant 109 >10cm (trail)', (ident == 109) & (plant > 0.10))
    report('table items 79 80 81 117 170 181', np.isin(ident, [79, 80, 81, 117, 170, 181]))
    for i_, nm in ((69, 'chair_0001'), (70, 'chair_0002'), (72, 'chair_0004'), (164, 'table_0003')):
        report(f'{nm} ({i_})', ident == i_)
    for name, (s_, i_) in {'bathroom a2700': ('a', 2700), 'bathroom a2985': ('a', 2985), 'kitchen wall b2625': ('b', 2625),
                           'chairs b1395': ('b', 1395), 'curtain b1845': ('b', 1845), 'last keyframe b3447': ('b', 3447)}.items():
        f = FlatSession(make(s_), pixel_step=INPUT_STEP[DATASET]).load(i_)
        K, z = f.K, f.depth
        vv, uu = np.indices(z.shape, dtype=np.float32)
        okp = np.isfinite(z) & (z > 0)
        cam = np.stack(((uu[okp] + K.offset - K.cx) / K.fx * z[okp], (vv[okp] + K.offset - K.cy) / K.fy * z[okp], z[okp], np.ones(okp.sum())), 1)
        dd = cKDTree((cam @ f.T_world_cam.T)[:, :3]).query(xyz, k=1, distance_upper_bound=0.03)[0]
        report(f'surface of {name}', np.isfinite(dd))
report('background (id 0)', ident == 0)
if REF:
    from plyfile import PlyData
    v = PlyData.read(REF)['vertex']
    d = cKDTree(np.stack((v['x'], v['y'], v['z']), 1).astype(np.float64)).query(xyz, k=1)[0]
    print(f'  reference: all rows within 5 cm {np.mean(d <= 0.05) * 100:.1f} % | ENDED rows within 5 cm {np.mean(d[ended] <= 0.05) * 100:.1f} %, '
          f'beyond 10 cm {np.mean(d[ended] > 0.10) * 100:.1f} % ({ended.sum()} rows)', flush=True)
np.savez_compressed('d1_rows.npz', idx=idx, ended=ended, first_free=first_free, last_occ=last_occ_row, n_occ=n_occ, n_free=n_free)
print('D1_VOXELS_DONE', flush=True)
