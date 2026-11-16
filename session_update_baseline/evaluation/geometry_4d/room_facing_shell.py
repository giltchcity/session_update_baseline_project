"""The room-facing part of the synthetic scene's shell (walls, floors, ceilings), for the 4D geometry reference.

The Isaac meshes of the shell have thickness: a slab above the suspended ceiling, a slab under the floor, the outer
faces of the walls. No camera inside the flat can see them. A shell surface is kept when a straight line from some
camera position of the sequence (sessions A and B, every 30th frame) reaches it unblocked by the structure (walls,
floors, ceilings, doors, windows, curtains, lights); furniture does not block, so shell behind or under furniture stays.
The shell is subdivided to edges <= 10 cm and each piece is tested at its centroid (2 cm tolerance).
Writes OUT (default the working copy evaluate_map_vs_gt.py reads, room_facing_shell.npz) with vertices and faces, and
prints the kept area per entity; README.md lists the values a correct run prints (about 90 s, 2 GB).
usage: room_facing_shell.py [OUT.npz]"""
import json, sys
from pathlib import Path
import numpy as np
import open3d as o3d
from cvpr_eval_common import GT, IX, REG, worldmesh

D = Path('/home/jixian/Desktop/FT/datasets/synthetic_ab')
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/jixian/Desktop/FT/wt_takeover_runs/geometry_4d_20261114/room_facing_shell.npz')
classes = json.loads((D / 'registry.json').read_text())['classes']
STRUCTURE = {7, 8, 10, 13, 16, 17, 18, 21, 34, 44, 45}; SHELL = {'wall', 'floor', 'ceiling'}
s, f = 'b', len(GT['b']['time_s']) - 1   # the structure is static over both sessions (build_causal_surface_bank.py asserts it)


def subdivide(v, f, max_edge=0.10):
    tri = v[f].astype(np.float64); out = []
    while len(tri):
        e = np.max(np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2), axis=1); small = e <= max_edge
        out.append(tri[small]); t = tri[~small]
        if not len(t): break
        a, b, c = t[:, 0], t[:, 1], t[:, 2]; ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        tri = np.concatenate([np.stack(x, 1) for x in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    return np.concatenate(out)


occ = o3d.t.geometry.RaycastingScene(nthreads=2); shell = []
for i in REG:
    if i not in IX[s] or not GT[s]['render_visible'][f, IX[s][i]]: continue
    if classes[REG[i]['class']] not in STRUCTURE: continue
    v, fa = worldmesh(s, i, f)
    if not len(fa): continue
    occ.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(fa))
    if REG[i]['class'] in SHELL: shell.append((i, REG[i].get('name', str(i)), subdivide(v, fa)))
views = np.concatenate([[np.loadtxt(D / f'session_{x}/rgbd/{k:06d}_pose.txt')[:3, 3] for k in range(0, len(GT[x]['time_s']), 30)] for x in 'ab'])
tris = np.concatenate([t for _, _, t in shell]); owner = np.concatenate([np.full(len(t), k) for k, (_, _, t) in enumerate(shell)])
cen = tris.mean(1).astype(np.float32); area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
seen = np.zeros(len(cen), bool)
for c in views.astype(np.float32):
    todo = np.flatnonzero(~seen)
    if not len(todo): break
    d = cen[todo] - c; dist = np.linalg.norm(d, axis=1); d /= dist[:, None]
    hit = occ.cast_rays(o3d.core.Tensor(np.hstack([np.broadcast_to(c, d.shape), d]).astype(np.float32)), nthreads=2)['t_hit'].numpy()
    seen[todo[hit >= dist - 0.02]] = True
print(f'{len(views)} camera positions, shell {area.sum():.1f} m2 in {len(cen)} pieces, room-facing {area[seen].sum():.1f} m2 '
      f'({100 * area[seen].sum() / area.sum():.1f}%)')
for k, (i, name, t) in enumerate(shell):
    m = owner == k
    if area[m].sum() > 5: print(f'  {i:4d} {name:16s} area {area[m].sum():6.1f} m2  kept {100 * area[m & seen].sum() / area[m].sum():5.1f}%')
kept = tris[seen].astype(np.float32)
np.savez_compressed(OUT, vertices=kept.reshape(-1, 3), faces=np.arange(3 * len(kept), dtype=np.uint32).reshape(-1, 3))
print('wrote', OUT)
