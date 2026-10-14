"""A's rows in session B: what the layer's L2 judged, how far each mapping version moved them, and whether the move brought them to
B's surface (CPU, read-only: checkpoints and the official reference; no rendering, no GPU).
Per A row (created before B, not ended before B; matched by uid in A and in each B checkpoint):
  L2      retired by the layer during B (death_evidence in (B start, B end]): the layer judged its state ended
  pruned  ended by GaME's prune during B (death_prune in (B start, B end])
  kept    neither (alive at B's end)
  confirmed  last_update >= B start (a B keyframe measured it on its surface: the backend's support test)
  moved   |xyz_B - xyz_A| (metres; GaME units / scale)
  d_ref   distance to the nearest point of B's official reference (the G1 reference, 1 cm sampling), at A's and at B's position
usage: rows_vs_l2.py A_CKPT B_REFERENCE_PLY SCALE NAME=B_CKPT [NAME=B_CKPT ...]
"""
import sys

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree


def rows(path):
    ck = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    b = ck['backend']
    return dict(uid=b['uid'].numpy().copy(), xyz=ck['backend']['model'][1].detach().numpy().copy(),
                created=b['created'].numpy().copy(), de=b['death_evidence'].numpy().copy(), dp=b['death_prune'].numpy().copy(),
                ds=b['death_state'].numpy().copy(), lu=b['last_update'].numpy().copy(), start=int(b['session_start']),
                now=int(ck['prev_final']))


a_path, ref_path, scale = sys.argv[1], sys.argv[2], float(sys.argv[3])
runs = [x.split('=', 1) for x in sys.argv[4:]]
A = rows(a_path)
ref = np.asarray(o3d.io.read_point_cloud(ref_path).points)
tree = cKDTree(ref)
print(f'A: {len(A["uid"])} rows; B reference {ref_path}: {len(ref)} points', flush=True)
res = {}
for name, p in runs:
    B = rows(p)
    s0, s1 = B['start'], B['now']
    alive_at_start = (A['de'] > s0) & (A['dp'] > s0) & (A['ds'] > s0)          # A rows not ended before B
    ia = np.where(alive_at_start)[0]
    order = np.argsort(B['uid'])
    pos = np.searchsorted(B['uid'], A['uid'][ia], sorter=order)
    pos = np.clip(pos, 0, len(order) - 1)
    ok = B['uid'][order[pos]] == A['uid'][ia]
    ia, ib = ia[ok], order[pos[ok]]
    l2 = (B['de'][ib] > s0) & (B['de'][ib] <= s1)
    pr = (B['dp'][ib] > s0) & (B['dp'][ib] <= s1) & ~l2
    kept = ~l2 & ~pr & (B['ds'][ib] > s1)
    conf = B['lu'][ib] >= s0
    xa, xb = A['xyz'][ia] / scale, B['xyz'][ib] / scale
    mv = np.linalg.norm(xb - xa, axis=1)
    da = tree.query(xa, workers=2)[0]
    db = tree.query(xb, workers=2)[0]
    res[name] = dict(uid=A['uid'][ia], l2=l2, pr=pr, kept=kept, conf=conf, mv=mv, da=da, db=db)
    print(f'\n== {name} ({p}): {len(ia)} A rows alive at B start, matched in B')
    print(f'   L2 retired during B {int(l2.sum())}, GaME-pruned {int(pr.sum())}, kept {int(kept.sum())}; confirmed by B support {int(conf.sum())}')
    for lab, m in (('kept, confirmed', kept & conf), ('kept, not confirmed', kept & ~conf), ('L2 retired', l2), ('GaME-pruned', pr)):
        if not m.any():
            continue
        print(f'   {lab:22s} n={int(m.sum()):8d}  moved median {100 * np.median(mv[m]):6.2f} cm, p90 {100 * np.percentile(mv[m], 90):6.2f} cm, '
              f'> 1 cm {100 * (mv[m] > 0.01).mean():5.1f} %;  d_ref median at A {100 * np.median(da[m]):6.2f} cm -> at B {100 * np.median(db[m]):6.2f} cm;  '
              f'within 5 cm at A {100 * (da[m] <= 0.05).mean():5.1f} % -> at B {100 * (db[m] <= 0.05).mean():5.1f} %', flush=True)
if len(res) == 2:
    (n1, r1), (n2, r2) = res.items()
    common, i1, i2 = np.intersect1d(r1['uid'], r2['uid'], return_indices=True)
    print(f'\n== the same {len(common)} A rows in both runs: L2 decision')
    for a_ in (True, False):
        for b_ in (True, False):
            m = (r1['l2'][i1] == a_) & (r2['l2'][i2] == b_)
            print(f'   L2 retired in {n1}: {a_!s:5s}  in {n2}: {b_!s:5s}  {int(m.sum()):8d}')
    k = r1['kept'][i1] & r2['kept'][i2]
    for lab, m in (('kept in both, confirmed in both', k & r1['conf'][i1] & r2['conf'][i2]), ('kept in both', k)):
        print(f'   {lab}: n={int(m.sum())}; within 5 cm of B ref at B: {n1} {100 * (r1["db"][i1][m] <= 0.05).mean():.1f} %, '
              f'{n2} {100 * (r2["db"][i2][m] <= 0.05).mean():.1f} %; moved median {n1} {100 * np.median(r1["mv"][i1][m]):.2f} cm, '
              f'{n2} {100 * np.median(r2["mv"][i2][m]):.2f} cm')
print('ROWS_VS_L2_DONE')
