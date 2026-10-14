"""G1@5 of a real B export, split by the map rows that make each sample (CPU, read-only).
Samples, crop and distances are those of eval/g1_decompose.py = eval/geometry_g1.py (area-sampled at 2 cm, the oriented room crop,
Open3D distances), so the shares add up to 1 - P and 1 - R of G1@5. Each prediction sample is attributed to the nearest row of the map
alive at B's end, grouped as
  A kept, confirmed      created before B, last_update >= B start (a B keyframe measured it on its surface)
  A kept, not confirmed  created before B, never measured on its surface by B
  B-born                 created in B
Recall misses are attributed the same way to the nearest row (or 'no row within 5 cm').
usage: g1_by_row_group.py A_CKPT NAME=B_CKPT:PRED_PLY [NAME=B_CKPT:PRED_PLY ...]
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

FT = Path("/home/jixian/Desktop/FT")
a_path = sys.argv[1]
uidA = torch.load(a_path, map_location='cpu', mmap=True, weights_only=False)['backend']['uid'].numpy()
crop = json.loads((FT / "eval/crops/real_B.json").read_text())
lo, hi, yaw = np.array(crop["lo"]), np.array(crop["hi"]), crop["yaw_deg"]
ref = o3d.io.read_triangle_mesh(str(FT / "results/abc_eval_v2/geometry/B_reference_1cm.ply"))
rp, _ = G.surface_sample(ref)
rp = rp[~((rotate(rp, yaw) < lo) | (rotate(rp, yaw) > hi)).any(1)].astype(np.float64)
for arg in sys.argv[2:]:
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
    names = ['A kept, confirmed', 'A kept, not confirmed', 'B-born', 'other']
    rows_tree = cKDTree(xyz[alive]); rows_grp = grp[alive]
    pred = o3d.io.read_triangle_mesh(pred_path)
    pp, _ = G.surface_sample(pred)
    pp = pp[~((rotate(pp, yaw) < lo) | (rotate(pp, yaw) > hi)).any(1)].astype(np.float64)
    dpred = G.distances(pp, ref)
    drec = G.distances(rp, pred)
    _, ip = rows_tree.query(pp, workers=2)
    gp = rows_grp[ip]
    dr_rows, ir = rows_tree.query(rp, workers=2)
    gr = np.where(dr_rows <= 0.05, rows_grp[ir], 4)
    P = 1 - (dpred > 0.05).mean(); R = 1 - (drec > 0.05).mean()
    print(f'\n== {name}: {pred_path}\n   G1@5 from these samples: P {100 * P:.2f}  R {100 * R:.2f}  F1 {100 * 2 * P * R / (P + R):.2f}  '
          f'({len(pp)} prediction samples, {len(rp)} reference samples)')
    print(f'   {"group":24s} {"pred samples":>13s} {"precision errors":>17s} {"error share of all":>19s} | {"recall misses":>14s}')
    for g in range(4):
        m = gp == g
        if not m.any() and not (gr == g).any():
            continue
        e = m & (dpred > 0.05)
        mr = (gr == g) & (drec > 0.05)
        print(f'   {names[g]:24s} {int(m.sum()):13d} {100 * e.sum() / max(m.sum(), 1):16.1f}% {100 * e.sum() / len(pp):18.2f}% | {100 * mr.sum() / len(rp):13.2f}%')
    mr = (gr == 4) & (drec > 0.05)
    print(f'   {"no row within 5 cm":24s} {"":13s} {"":17s} {"":19s} | {100 * mr.sum() / len(rp):13.2f}%')
print('G1_BY_ROW_GROUP_DONE')
