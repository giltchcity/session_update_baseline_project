"""The plant's Gaussians before and after the refinement (CPU, checkpoints + the raw frame 3447 only).

Plant surface at the last keyframe 3447 = the raw depth of its instance-109 pixels lifted to the world. For every checkpoint: the rows
alive at the end stamp, split into rows within 3 cm of that surface (the plant as frame 3447 sees it) and the rest of identity 109;
plus every row of any identity within 3 cm of the plant surface. Reported: counts, largest/smallest axis, axis ratio, opacity.
usage: pot_rows.py LABEL=CHECKPOINT ...
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from update_layer.frames import FlatSession, synthetic_session  # noqa: E402

INT64_MAX = 9223372036854775807
SCALE = 10.0        # flat.yaml game.scale: the model's coordinates are metres x 10
fs = FlatSession(synthetic_session('b'), pixel_step=1)
f = fs.load(3447)
K, z = f.K, f.depth
v, u = np.indices(z.shape, dtype=np.float32)
ok = (f.instance == 109) & np.isfinite(z) & (z > 0)
cam = np.stack(((u[ok] + K.offset - K.cx) / K.fx * z[ok], (v[ok] + K.offset - K.cy) / K.fy * z[ok], z[ok], np.ones(ok.sum())), 1)
plant = (cam @ f.T_world_cam.T)[:, :3]
tree = cKDTree(plant)
print(f'plant surface at 3447: {len(plant)} pixels, extent {np.ptp(plant, 0).round(2)} m', flush=True)
for arg in sys.argv[1:]:
    label, path = arg.split('=', 1)
    ck = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
    b, t = ck['backend'], int(ck['prev_final'])
    m = b['model']
    xyz = m[1].detach().numpy() / SCALE
    end = np.minimum(np.minimum(b['death_state'].numpy(), b['death_evidence'].numpy()), b['death_prune'].numpy())
    born = np.maximum(b['created'].numpy(), np.where(b['state_birth'].numpy() < INT64_MAX, b['state_birth'].numpy(), -INT64_MAX))
    alive = (born <= t) & (end > t)
    ident = b['identity'].numpy()
    d = tree.query(xyz[alive], k=1)[0]
    near = np.zeros(len(xyz), bool); near[np.flatnonzero(alive)[d <= 0.03]] = True
    dist = np.full(len(xyz), np.inf); dist[alive] = d
    s = np.exp(m[4].detach().numpy()) / SCALE
    o = 1 / (1 + np.exp(-m[6].detach().numpy()[:, 0]))
    def stats(sel, name):
        if not sel.any():
            print(f'  {name:34s} 0 rows'); return
        mx, mn = s[sel].max(1), s[sel].min(1)
        print(f'  {name:34s} {sel.sum():7d} rows | largest axis median {np.median(mx)*100:5.2f} cm p90 {np.percentile(mx,90)*100:5.2f} | '
              f'smallest {np.median(mn)*100:5.2f} cm | ratio median {np.median(mx/mn):5.1f} | opacity median {np.median(o[sel]):.2f}')
    p109 = alive & (ident == 109)
    print(f'{label}: rows {len(xyz)}, alive at end {alive.sum()}')
    stats(p109 & near, 'id109 within 3 cm of the plant')
    stats(p109 & ~near & (dist <= 0.10), 'id109 3-10 cm off')
    stats(p109 & (dist > 0.10), 'id109 > 10 cm off (trail)')
    stats(near & (ident != 109), 'other ids within 3 cm of the plant')
