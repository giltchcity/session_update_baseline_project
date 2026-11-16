"""Two checks of the room-facing shell (room_facing_shell.npz) against what the cameras saw, run after making it.

  1. observed points: the structure points of the causal reference bank (instance 0, every observed wall / floor /
     ceiling / door / window / curtain / light point), each assigned to its nearest GT structure entity; the ones on a
     wall, floor or ceiling must lie on the kept shell (README.md: 100.0 / 99.8 / 100.0 % within 5 cm).
  2. camera views: every 15th frame of sessions A and B ray cast against the shell pieces plus the other structure
     (doors, windows, curtains, lights; no furniture, so more of the shell is visible than in the real views); the share
     of visible shell pixels that land on a removed piece (README.md: 0.11 % over 517 views, worst view 1.3 %).
usage: check_shell.py [SHELL.npz]   (default: the working copy, as evaluate_map_vs_gt.py)"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from cvpr_eval_common import GT, IX, REG, worldmesh, sample_surface

D = Path('/home/jixian/Desktop/FT/datasets/synthetic_ab'); classes = json.loads((D / 'registry.json').read_text())['classes']
STRUCTURE = {7, 8, 10, 13, 16, 17, 18, 21, 34, 44, 45}; SHELL = ('wall', 'floor', 'ceiling')
PATH = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/jixian/Desktop/FT/wt_takeover_runs/geometry_4d_20261114/room_facing_shell.npz')
s, f = 'b', len(GT['b']['time_s']) - 1   # the structure is static over both sessions
z = np.load(PATH); kept_v, kept_f = z['vertices'], z['faces']


def subdivide(v, f, max_edge=0.10):   # as room_facing_shell.py
    tri = v[f].astype(np.float64); out = []
    while len(tri):
        e = np.max(np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2), axis=1); small = e <= max_edge
        out.append(tri[small]); t = tri[~small]
        if not len(t): break
        a, b, c = t[:, 0], t[:, 1], t[:, 2]; ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        tri = np.concatenate([np.stack(x, 1) for x in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    return np.concatenate(out)


structure = [i for i in REG if i in IX[s] and GT[s]['render_visible'][f, IX[s][i]] and classes[REG[i]['class']] in STRUCTURE
             and len(worldmesh(s, i, f)[1])]

# 1. observed structure points on the kept shell
bank = Path('/home/jixian/Desktop/FT/datasets/synthetic_ab/evaluation_complete_20260917/causal_reference')
meta = json.loads((bank / 'manifest.json').read_text())
obs = np.load(bank / [r for r in meta if r['key'] == 'structure'][0]['file'])['points'].astype(np.float32)
sc = o3d.t.geometry.RaycastingScene(nthreads=2); gid = {}
for i in structure:
    v, fa = worldmesh(s, i, f); gid[sc.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(fa))] = REG[i]['class']
q = sc.compute_closest_points(o3d.core.Tensor(obs), nthreads=2)
cls = np.array([gid[int(g)] for g in q['geometry_ids'].numpy()]); dist = np.linalg.norm(q['points'].numpy() - obs, axis=1)
d = cKDTree(sample_surface(kept_v, kept_f, 600000, 1)).query(obs, workers=2)[0]
print('1. observed structure points by nearest entity class:', dict(Counter(cls).most_common()))
for k in SHELL:
    m = (cls == k) & (dist <= .05)
    print(f'   {k:8s} {m.sum():7d} points, on the kept shell within 5 cm: {100 * np.mean(d[m] <= .05):5.1f} %')

# 2. visible shell pixels on removed pieces, every 15th frame of A and B
tris = np.concatenate([subdivide(*worldmesh(s, i, f)) for i in structure if REG[i]['class'] in SHELL])
kept = cKDTree(kept_v.reshape(-1, 3, 3).mean(1)).query(tris.mean(1).astype(np.float32))[0] < 1e-4
sc = o3d.t.geometry.RaycastingScene(nthreads=2); t = tris.astype(np.float32)
sid = sc.add_triangles(o3d.core.Tensor(t.reshape(-1, 3)), o3d.core.Tensor(np.arange(3 * len(t), dtype=np.uint32).reshape(-1, 3)))
for i in structure:
    if REG[i]['class'] not in SHELL:
        v, fa = worldmesh(s, i, f); sc.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(fa))
views = []; vis = rem = 0
for x in 'ab':
    K = np.asarray(json.load(open(D / f'session_{x}/VALIDATION.json'))['K'])
    for k in range(0, len(GT[x]['time_s']), 15):
        T = np.loadtxt(D / f'session_{x}/rgbd/{k:06d}_pose.txt')
        h = sc.cast_rays(o3d.t.geometry.RaycastingScene.create_rays_pinhole(o3d.core.Tensor(K), o3d.core.Tensor(np.linalg.inv(T)), 680, 480), nthreads=2)
        on = h['geometry_ids'].numpy() == sid; p = h['primitive_ids'].numpy().astype(np.int64)
        r = int((~kept[p[on]]).sum()); n = int(on.sum()); vis += n; rem += r; views.append((r / max(n, 1), x, k))
views.sort(reverse=True)
print(f'2. {len(views)} views: {vis} shell pixels, {rem} on removed pieces = {100 * rem / vis:.3f} %; worst',
      [(x, k, round(100 * r, 2)) for r, x, k in views[:3]])
