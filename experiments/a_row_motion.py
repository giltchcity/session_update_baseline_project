"""How B moved the rows of A, and whether the move brought them to or away from the scorer's reference (CPU, read-only).

For each B run: the rows of A (by uid) that are alive at B's end are compared with A's own state:
  - displacement of the centre
  - opacity change
  - change of the largest scale
  - distance of the centre to the reference before (A's position) and after (B's position)
Reference: real B_reference_1cm.ply; synthetic the exact GT of B's last frame (cvpr_eval_common.worldmesh, render-visible entities).
Rows of A that are not alive at B's end (ended or deleted) are counted with their distance before.
Distances only within the scorer's reach are summarised (5 cm: right <= 5 cm, wrong > 5 cm).

usage: a_row_motion.py DATASET A_CKPT NAME=B_CKPT [NAME=B_CKPT ...]
"""
import sys

import numpy as np
import torch

DS, A_CKPT = sys.argv[1], sys.argv[2]


def rows(ck, alive_at=None):
    b = ck['backend']
    m = b['model']
    scale = float(m[11])
    xyz = m[1].detach().numpy() / scale
    s = np.exp(m[4].detach().numpy()).max(1) / scale
    op = 1 / (1 + np.exp(-m[6].detach().numpy().reshape(-1)))
    uid = b['uid'].numpy()
    if alive_at is None:
        alive = np.ones(len(uid), bool)
    else:
        de, dp, ds = (b[k].numpy() for k in ('death_evidence', 'death_prune', 'death_state'))
        alive = (de > alive_at) & (dp > alive_at) & (ds > alive_at)
    return uid, xyz, s, op, alive


import open3d as o3d  # noqa: E402
if DS == 'real':
    ref = o3d.io.read_triangle_mesh('/home/jixian/Desktop/FT/results/abc_eval_v2/geometry/B_reference_1cm.ply')
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(ref))
else:
    sys.path.insert(0, '/mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts')
    import cvpr_eval_common as C
    frame = len(C.GT['b']['time_s']) - 1
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    for i in C.GT['b']['instance_ids'].tolist():
        i = int(i)
        if i in C.IX['b'] and C.GT['b']['render_visible'][frame, C.IX['b'][i]]:
            v, f = C.worldmesh('b', i, frame)
            if len(f):
                scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(v), o3d.core.Tensor(f)))
dist = lambda p: scene.compute_distance(o3d.core.Tensor(p.astype(np.float32))).numpy()

cka = torch.load(A_CKPT, map_location='cpu', mmap=True, weights_only=False)
ua, xa, sa, oa, aa = rows(cka, int(cka['prev_final']))
xa, sa, oa, ua = xa[aa], sa[aa], oa[aa], ua[aa]
da = dist(xa)
print(f'{DS} A: {len(ua)} rows alive at A\'s end; {100 * (da > 0.05).mean():.1f} % farther than 5 cm from the B reference')
for arg in sys.argv[3:]:
    name, path = arg.split('=', 1)
    ckb = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    ub, xb, sb, ob, ab = rows(ckb, int(ckb['prev_final']))
    pos = {int(u): i for i, u in enumerate(ub)}
    j = np.array([pos.get(int(u), -1) for u in ua])
    present = j >= 0
    live = present.copy()
    live[present] = ab[j[present]]
    i_live = np.nonzero(live)[0]
    jb = j[i_live]
    disp = np.linalg.norm(xb[jb] - xa[i_live], axis=1)
    d_after = dist(xb[jb])
    d_before = da[i_live]
    moved = disp > 0.01
    print(f'\n== {name}: A rows alive at B\'s end {len(i_live)} ({100 * len(i_live) / len(ua):.1f} %), ended or deleted {len(ua) - len(i_live)}')
    gone = ~live
    print(f'   ended/deleted rows: farther than 5 cm before {100 * (da[gone] > 0.05).mean():.1f} %')
    print(f'   displacement of the alive rows: median {100 * np.median(disp):.2f} cm, 90th pct {100 * np.percentile(disp, 90):.2f} cm, '
          f'moved > 1 cm {100 * moved.mean():.1f} %, > 5 cm {100 * (disp > 0.05).mean():.1f} %')
    print(f'   opacity: median change {np.median(ob[jb] - oa[i_live]):+.3f}, rows below 0.5 now {100 * (ob[jb] < 0.5).mean():.1f} % '
          f'(A {100 * (oa[i_live] < 0.5).mean():.1f} %); largest scale median ratio {np.median(sb[jb] / sa[i_live]):.3f}')
    for lab, m in (('all alive', np.ones(len(i_live), bool)), ('moved > 1 cm', moved)):
        if not m.any():
            continue
        closer = (d_after < d_before - 1e-4) & m
        farther = (d_after > d_before + 1e-4) & m
        r2w = (d_before <= 0.05) & (d_after > 0.05) & m
        w2r = (d_before > 0.05) & (d_after <= 0.05) & m
        print(f'   {lab:12s} rows {int(m.sum()):8d}: closer to the reference {100 * closer.sum() / m.sum():5.1f} %, farther '
              f'{100 * farther.sum() / m.sum():5.1f} %; wrong -> right {int(w2r.sum())}, right -> wrong {int(r2w.sum())}; '
              f'wrong before {100 * ((d_before > 0.05) & m).sum() / m.sum():.1f} %, after {100 * ((d_after > 0.05) & m).sum() / m.sum():.1f} %')
    # the present's own rows: created in B, alive at B's end
    cb = ckb['backend']['created'].numpy()
    s0 = int(ckb['backend']['session_start'])
    born = (cb >= s0) & ab
    if born.any():
        db = dist(xb[born])
        print(f'   B-born rows alive at B\'s end {int(born.sum())}: farther than 5 cm from the reference {100 * (db > 0.05).mean():.1f} %, '
              f'median distance {100 * np.median(db):.2f} cm (A rows alive: after {100 * (d_after > 0.05).mean():.1f} %, median {100 * np.median(d_after):.2f} cm)')
print('A_ROW_MOTION_DONE')
