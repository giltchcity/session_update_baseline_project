"""Final maps of a real session from several viewpoints above the room (four upper corners and straight down), each
map raycast and coloured by the session's G1 reference at 5 cm (grey: within 5 cm of the reference; red: farther),
shaded by the surface normal; one row per view, one column per map. The final snapshot of each map is exported in
/dev/shm (eval_fast's geometry_export, the G1 viewer policy). Prints the red share per view and map.
usage: real_views.py OUT.png a|b|c MAP.4dmap.zpk [MAP.4dmap.zpk ...]"""
import os, sys, subprocess, tempfile
from pathlib import Path
import numpy as np, open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree
ROOT = Path('/home/jixian/Desktop/FT')
sys.path.insert(0, str(ROOT / 'eval_fast/vendor/pack'))
import zpk_snapshot as zs, map4d_pack as mp

out, session, maps = sys.argv[1], sys.argv[2], sys.argv[3:]
ref = np.asarray(o3d.io.read_point_cloud(str(ROOT / f'results/abc_eval_v2/geometry/{session.upper()}_reference_1cm.ply')).points)
tree = cKDTree(ref)
lo, hi = np.percentile(ref, 1, axis=0), np.percentile(ref, 99, axis=0)
centre, extent = (lo + hi) / 2, hi - lo
W, H, f = 640, 480, 420.0

def look(eye, target):
    z = (target - eye) / np.linalg.norm(target - eye)
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
    return eye, np.stack([x, y, z], 1)

views = [look(np.array([cx, cy, hi[2] + 0.3]), centre - [0, 0, 0.3 * extent[2]])
         for cx in (lo[0], hi[0]) for cy in (lo[1], hi[1])]
views.append(look(centre + [0, 0, max(extent[:2]) * 1.1], centre))
u, v = np.meshgrid(np.arange(W), np.arange(H))
dirs_cam = np.stack([(u - W / 2) / f, (v - H / 2) / f, np.ones_like(u, float)], -1)
dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

meshes = []
with tempfile.TemporaryDirectory(prefix='real-views-', dir='/dev/shm') as tmp:
    for k, source in enumerate(maps):
        sliced, mesh = Path(tmp) / f'{k}.4dmap', Path(tmp) / f'{k}.ply'
        zs.cut(source, len(mp.read_manifest(source)['snapshots']) - 1, str(sliced))
        subprocess.run(['bash', '-c', 'source "$1" >/dev/null 2>&1 || exit 90; shift; exec "$@"', 'views',
                        str(ROOT / 'eval_fast/runtime_khronos_env.sh'), str(ROOT / 'eval_fast/vendor/geometry/geometry_export'),
                        str(sliced), str(mesh), 'viewer'], check=True, stdout=subprocess.DEVNULL)
        meshes.append(o3d.io.read_triangle_mesh(str(mesh)))
rows = []
for i, (eye, R) in enumerate(views):
    dw = dirs_cam @ R.T
    rays = o3d.core.Tensor(np.concatenate([np.broadcast_to(eye, dw.shape), dw], -1).astype(np.float32).reshape(-1, 6))
    tiles = []
    for k, m in enumerate(meshes):
        sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
        ans = sc.cast_rays(rays)
        t = ans['t_hit'].numpy().reshape(H, W); n = ans['primitive_normals'].numpy().reshape(H, W, 3); hit = np.isfinite(t)
        p = eye + dw[hit] * t[hit, None]
        far = tree.query(p, distance_upper_bound=0.05)[0] > 0.05
        shade = np.clip(0.25 + 0.75 * np.abs((n * dw).sum(-1)), 0, 1)[hit, None]
        col = np.where(far[:, None], [235, 60, 50], [220, 220, 220])
        img = np.full((H, W, 3), 255, np.uint8); img[hit] = (shade * col).astype(np.uint8)
        tiles.append(img)
        print(f'view {i} map {k}: hit {hit.mean():.1%} red {far.mean():.2%} of hits')
    rows.append(np.concatenate(tiles, 1))
Image.fromarray(np.concatenate(rows, 0)).save(out)
print('written', out)
