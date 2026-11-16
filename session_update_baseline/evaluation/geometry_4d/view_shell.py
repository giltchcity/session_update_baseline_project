"""Open3D window (WSLg) with the synthetic GT at the end of visit B, every entity under its Isaac name:
walls, floors and ceilings split into the room-facing part kept by scheme C (green, "kept") and the removed part (red,
"removed"); every other entity whole (grey). The list on the right toggles each one.
Reads the working folder's room_facing_shell.npz (GEO4D_SHELL overrides).
usage: view_shell.py"""
import json, os
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from cvpr_eval_common import GT, IX, REG, worldmesh

SHELL = Path(os.environ.get('GEO4D_SHELL', Path('/home/jixian/Desktop/FT/wt_takeover_runs/geometry_4d_20261114') / 'room_facing_shell.npz'))
s, f = 'b', len(GT['b']['time_s']) - 1


def subdivide(v, f, max_edge=0.10):   # as room_facing_shell.py
    tri = v[f].astype(np.float64); out = []
    while len(tri):
        e = np.max(np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2), axis=1); small = e <= max_edge
        out.append(tri[small]); t = tri[~small]
        if not len(t): break
        a, b, c = t[:, 0], t[:, 1], t[:, 2]; ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        tri = np.concatenate([np.stack(x, 1) for x in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    return np.concatenate(out)


def mesh(tri, colour):
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tri.reshape(-1, 3)),
                                  o3d.utility.Vector3iVector(np.arange(3 * len(tri)).reshape(-1, 3)))
    m.remove_duplicated_vertices(); m.compute_vertex_normals(); m.paint_uniform_color(colour)
    return m


z = np.load(SHELL); kept_tree = cKDTree(z['vertices'].reshape(-1, 3, 3).mean(1))
geoms = []
for i in sorted(REG, key=lambda i: (REG[i]['class'], REG[i]['name'])):
    if i not in IX[s] or not GT[s]['render_visible'][f, IX[s][i]]: continue
    v, fa = worldmesh(s, i, f)
    if not len(fa): continue
    name = REG[i]['name']
    if REG[i]['class'] in ('wall', 'floor', 'ceiling'):
        tri = subdivide(v, fa); kept = kept_tree.query(tri.mean(1).astype(np.float32))[0] < 1e-4
        if kept.any(): geoms.append(dict(name=f'{name}  kept', geometry=mesh(tri[kept], (0.55, 0.72, 0.55))))
        if (~kept).any(): geoms.append(dict(name=f'{name}  removed', geometry=mesh(tri[~kept], (0.86, 0.22, 0.20))))
    else:
        geoms.append(dict(name=f'{name}  ({REG[i]["class"]})', geometry=mesh(v[fa], (0.78, 0.78, 0.78))))
print(len(geoms), 'geometries')
o3d.visualization.draw(geoms, title='Home GT at the end of B: scheme C shell (green kept, red removed)', show_ui=True,
                       bg_color=(1, 1, 1, 1), show_skybox=False)
