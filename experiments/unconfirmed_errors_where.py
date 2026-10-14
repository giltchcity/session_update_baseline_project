"""Where the precision errors of real B's export come from, by what B's own frames say at each error sample (CPU, read-only).
Samples, crop, distances and row groups exactly as g1_by_row_group.py. For every prediction sample that is an error (> 5 cm from
B's reference), attributed to a row group, the sample point is projected into B's frames (every STRIDE-th, full resolution, the
dataset's poses and intrinsics; dynamic pixels excluded) and classified by the L2 rule at the surface match tolerance, per frame:
on (|d - z| <= tol), seen through (d - z > tol), occluded (d - z < -tol), not measured (outside the image or no valid depth).
Per sample: 'seen through' if any frame saw through it and none measured it on its surface; 'on' if any frame measured it on its
surface; 'occluded only' if frames measured only nearer things; 'never measured' otherwise.
usage: unconfirmed_errors_where.py A_CKPT STRIDE NAME=B_CKPT:PRED_PLY [...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, "/home/jixian/Desktop/FT/tools/abc_eval_v2")
sys.path.insert(0, "/home/jixian/Desktop/FT/eval")
import geometry_score as G  # noqa: E402
from geometry_g1 import rotate  # noqa: E402

TREE = Path('/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/worktrees/evsnap')
sys.path.insert(0, str(TREE))
from update_layer.frames import FlatSession, dynamic_mask  # noqa: E402
from update_layer.run import dataset_config  # noqa: E402
from update_layer.core.l2.evidence import EvidenceConfig  # noqa: E402

FT = Path("/home/jixian/Desktop/FT")
a_path, stride = sys.argv[1], int(sys.argv[2])
uidA = torch.load(a_path, map_location='cpu', mmap=True, weights_only=False)['backend']['uid'].numpy()
crop = json.loads((FT / "eval/crops/real_B.json").read_text())
lo, hi, yaw = np.array(crop["lo"]), np.array(crop["hi"]), crop["yaw_deg"]
ref = o3d.io.read_triangle_mesh(str(FT / "results/abc_eval_v2/geometry/B_reference_1cm.ply"))
cfg, info, specs = dataset_config('real')
fs = FlatSession(next(sp for sp in specs if sp.name.endswith('_b')), pixel_step=1)
tol = EvidenceConfig().surface_match_tolerance
frames = list(range(0, len(fs.ids), stride))
names = ['A kept, confirmed', 'A kept, not confirmed', 'B-born', 'other']


def classify(points):
    on = np.zeros(len(points), bool); through = np.zeros(len(points), bool); occl = np.zeros(len(points), bool)
    nearest = np.full(len(points), np.inf)        # per sample the smallest |delta| among the frames that measured nearer (occluded)
    for i in frames:
        fr = fs.load(i)
        d = np.nan_to_num(fr.depth, nan=0.0)
        d[dynamic_mask(fr, info.dynamic_semantics)] = 0.0
        R, t = fr.T_world_cam[:3, :3], fr.T_world_cam[:3, 3]
        pc = (points - t) @ R
        z = pc[:, 2]
        ok = z > 0.05
        K = fr.K
        u = np.floor(pc[:, 0] / np.where(ok, z, 1) * K.fx + K.cx - K.offset + 0.5).astype(np.int64)
        v = np.floor(pc[:, 1] / np.where(ok, z, 1) * K.fy + K.cy - K.offset + 0.5).astype(np.int64)
        h, w = d.shape
        ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
        dd = np.zeros(len(points)); dd[ok] = d[v[ok], u[ok]]
        ok &= dd > 0
        delta = dd - z
        on |= ok & (np.abs(delta) <= tol); through |= ok & (delta > tol); occl |= ok & (delta < -tol)
        nearer = ok & (delta < -tol)
        nearest[nearer] = np.minimum(nearest[nearer], -delta[nearer])
    cls = np.full(len(points), 3)
    cls[occl] = 2
    cls[through & ~on] = 1
    cls[on] = 0
    return cls, nearest


rp, _ = G.surface_sample(ref)
rp = rp[~((rotate(rp, yaw) < lo) | (rotate(rp, yaw) > hi)).any(1)].astype(np.float64)
for arg in sys.argv[3:]:
    name, rest = arg.split('=', 1)
    ckp, pred_path = rest.split(':', 1)
    ck = torch.load(ckp, map_location='cpu', mmap=True, weights_only=False)
    b = ck['backend']
    s0, s1 = int(b['session_start']), int(ck['prev_final'])
    xyz = b['model'][1].detach().numpy() / float(b['model'][11])
    created, de, dp_, ds_, lu = (b[k].numpy() for k in ('created', 'death_evidence', 'death_prune', 'death_state', 'last_update'))
    alive = (de > s1) & (dp_ > s1) & (ds_ > s1)
    old = (created < s0) & np.isin(b['uid'].numpy(), uidA)
    grp = np.full(len(xyz), 3)
    grp[old & (lu >= s0)] = 0
    grp[old & (lu < s0)] = 1
    grp[created >= s0] = 2
    rows_tree = cKDTree(xyz[alive]); rows_grp = grp[alive]
    pred = o3d.io.read_triangle_mesh(pred_path)
    pp, _ = G.surface_sample(pred)
    pp = pp[~((rotate(pp, yaw) < lo) | (rotate(pp, yaw) > hi)).any(1)].astype(np.float64)
    dpred = G.distances(pp, ref)
    _, ip = rows_tree.query(pp, workers=2)
    gp = rows_grp[ip]
    err = dpred > 0.05
    print(f'\n== {name}: {len(pp)} prediction samples, {int(err.sum())} errors ({100 * err.mean():.2f} %); B frames every {stride}th ({len(frames)})', flush=True)
    sel = np.nonzero(err)[0]
    cls, nearest = classify(pp[sel])
    kinds = ['B measured it on its surface', 'B saw through it', 'B saw only nearer things', 'B never measured it']
    print(f'   {"group":24s} {"errors":>8s} ' + ' '.join(f'{k:>30s}' for k in kinds))
    for g in range(4):
        m = gp[sel] == g
        if not m.any():
            continue
        print(f'   {names[g]:24s} {int(m.sum()):8d} ' + ' '.join(f'{100 * ((cls == c) & m).sum() / len(pp):29.2f}%' for c in range(4)))
    print(f'   (shares are of ALL prediction samples, i.e. precision points)')
    trunc = float(cfg.truncation)
    print(f'   "B saw only nearer things": how much nearer B measured at best (the layer\'s TSDF truncation for this dataset: {100 * trunc:.0f} cm)')
    for g in range(4):
        m = (gp[sel] == g) & (cls == 2)
        if not m.any():
            continue
        x = nearest[m]
        bands = [(tol, trunc), (trunc, 2 * trunc), (2 * trunc, np.inf)]
        print(f'   {names[g]:24s} ' + '  '.join(f'{100 * lo_:.0f}-{100 * hi_ if np.isfinite(hi_) else np.inf:.0f} cm: {100 * ((x > lo_) & (x <= hi_)).sum() / len(pp):.2f}%' for lo_, hi_ in bands))
print('UNCONFIRMED_ERRORS_WHERE_DONE')
