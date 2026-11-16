"""Pictures of the shell culling (room_facing_shell.py): walls, floors and ceilings split into the same <= 10 cm pieces;
a piece is kept when its centroid is a kept centroid of room_facing_shell.npz (grey-green), else removed (red).
Ray cast with light shading, misses white.
  shell_outside.png   the whole shell from outside, above a corner
  shell_inside.png    the same view without the ceilings (pieces above 2.6 m dropped)
  shell_section.png   a vertical section: pieces within 0.10 m of the plane x = X0, seen along x
Writes the pictures into the working folder; reads its room_facing_shell.npz (GEO4D_SHELL overrides).
usage: show_shell.py [X0]"""
import json, os, sys
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree
from cvpr_eval_common import GT, IX, REG, worldmesh

HERE = Path('/home/jixian/Desktop/FT/wt_takeover_runs/geometry_4d_20261114')
SHELL = Path(os.environ.get('GEO4D_SHELL', HERE / 'room_facing_shell.npz'))
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


tris = np.concatenate([subdivide(*worldmesh(s, i, f)) for i in REG if i in IX[s] and GT[s]['render_visible'][f, IX[s][i]]
                       and REG[i]['class'] in ('wall', 'floor', 'ceiling') and len(worldmesh(s, i, f)[1])])
z = np.load(SHELL); kept_c = z['vertices'].reshape(-1, 3, 3).mean(1)
kept = cKDTree(kept_c).query(tris.mean(1).astype(np.float32))[0] < 1e-4
print(f'{len(tris)} pieces, kept {kept.mean():.3f}')
lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0); centre = (lo + hi) / 2
print('bounds', lo.round(2).tolist(), hi.round(2).tolist())


def render(sel, eye, target, out, W=1100, H=700, fov=40):
    t = tris[sel].astype(np.float32); col = np.where(kept[sel][:, None], [0.62, 0.72, 0.60], [0.86, 0.22, 0.20])
    sc = o3d.t.geometry.RaycastingScene(nthreads=2)
    sc.add_triangles(o3d.core.Tensor(t.reshape(-1, 3)), o3d.core.Tensor(np.arange(3 * len(t), dtype=np.uint32).reshape(-1, 3)))
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(fov_deg=fov, center=target.tolist(), eye=eye.tolist(),
                                                              up=[0, 0, 1], width_px=W, height_px=H)
    h = sc.cast_rays(rays, nthreads=2); pid = h['primitive_ids'].numpy().astype(np.int64)
    ok = pid != o3d.t.geometry.RaycastingScene.INVALID_ID
    n = h['primitive_normals'].numpy()[ok]; d = rays.numpy()[..., 3:][ok]; d /= np.linalg.norm(d, axis=1, keepdims=True)
    img = np.ones((H, W, 3)); img[ok] = col[pid[ok]] * (0.45 + 0.55 * np.abs((n * d).sum(1, keepdims=True)))
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(HERE / out); print('wrote', out)


span = hi - lo
eye = centre + np.array([-0.8 * span[0], -1.1 * span[1], 1.6 * span[2] + 6.0])
render(np.ones(len(tris), bool), eye, centre, 'shell_outside.png')
render(tris.mean(1)[:, 2] < 2.6, eye, centre, 'shell_inside.png')
x0 = float(sys.argv[1]) if len(sys.argv) > 1 else float(centre[0])
sec = np.abs(tris.mean(1)[:, 0] - x0) < 0.10
render(sec, np.array([x0 - 25.0, centre[1], centre[2]]), np.array([x0, centre[1], centre[2]]), 'shell_section.png', fov=22)
