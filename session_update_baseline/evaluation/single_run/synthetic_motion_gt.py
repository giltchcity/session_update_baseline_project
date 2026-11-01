"""Ground truth of the objects moving inside synthetic A/B (user 11-01), from the Isaac temporal ground truth
(datasets/synthetic_ab/session_{a,b}/gt/temporal_gt.npz = Synthetic_ABC's: every frame's object poses): an object moves
at a frame when its pose differs from the previous frame's (translation > 1 mm or rotation > 0.1 deg); its trajectory is
its centre (center_world) at the frames where it moves and is seen (>= 50 visible pixels, as the 09-17 dynamics
protocol). Stamps as the players publish them (A from 1 s, B from 141 s). Writes OUT/moving_objects_{a,b}.csv
(instance, stamp_ns, time_s, frame, x, y, z, visible_px) and OUT/moving_objects.png (top view). Read-only on the data.
usage: synthetic_motion_gt.py OUT"""
import csv, json, sys
from pathlib import Path
import numpy as np
D = Path('/home/jixian/Desktop/FT/datasets/synthetic_ab'); OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ORIGIN = {'a': 1_000_000_000, 'b': 141_000_000_000}
REG = {x['instance_id']: x for x in json.loads((D / 'registry.json').read_text())['objects']}
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
for ax, s in zip(axes, 'ab'):
    g = np.load(D / f'session_{s}/gt/temporal_gt.npz')
    T, t, ids, vis, px, c = g['T_world_object'], g['time_s'], g['instance_ids'], g['render_visible'], g['visible_pixel_count'], g['center_world']
    dt = np.linalg.norm(np.diff(T[:, :, :3, 3], axis=0), axis=2)                       # frames-1 x objects
    R = T[:, :, :3, :3] / np.maximum(np.linalg.norm(T[:, :, :3, :3], axis=2, keepdims=True), 1e-12)   # the poses carry scale: unit columns
    dR = np.degrees(np.arccos(np.clip((np.einsum('fnij,fnij->fn', R[1:], R[:-1]) - 1) / 2, -1, 1)))
    moving = np.vstack([np.zeros((1, T.shape[1]), bool), (dt > 1e-3) | (dR > 0.1)])
    rows = []
    for j, inst in enumerate(ids):
        m = moving[:, j]
        if not m.any(): continue
        seen = m & vis[:, j] & (px[:, j] >= 50)
        name = REG.get(int(inst), {}).get('name', REG.get(int(inst), {}).get('class', '?'))
        path = float(dt[:, j].sum())
        print(f'{s} instance {int(inst):4d} {str(name)[:28]:28s} moving frames {int(m.sum()):4d} ({t[m][0]:.1f}-{t[m][-1]:.1f} s), '
              f'seen while moving {int(seen.sum()):4d}, path {path:.2f} m', flush=True)
        for f in np.nonzero(seen)[0]:
            rows.append([int(inst), ORIGIN[s] + round(float(t[f]) * 1e9), round(float(t[f]), 4), int(f), *np.round(c[f, j], 4), int(px[f, j])])
        if seen.any():
            P = c[seen, j]; ax.plot(P[:, 0], P[:, 1], '-', lw=1.5, label=f'{inst} {str(name)[:18]}: {t[seen][0]:.0f}-{t[seen][-1]:.0f} s, {int(seen.sum())} fr')
            ax.plot(P[0, 0], P[0, 1], 'o', ms=5, color=ax.lines[-1].get_color())
    with (OUT / f'moving_objects_{s}.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['instance', 'stamp_ns', 'time_s', 'frame', 'x', 'y', 'z', 'visible_px']); w.writerows(rows)
    ax.set_title(f'synthetic {s.upper()}: objects moving in view (Isaac GT)'); ax.set_aspect('equal'); ax.grid(alpha=.3)
    ax.legend(fontsize=7); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
fig.tight_layout(); fig.savefig(OUT / 'moving_objects.png', dpi=110)
