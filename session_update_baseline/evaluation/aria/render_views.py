"""Our TSDF maps of an Aria room seen from the clip's own cameras, beside the camera's image (user 11-02: look at the map
and compare with the images). For each chosen ADT object: the frame of visit A (if it is there in A) and of visit B that
sees its visit-A place best (its box centre projected, the measured depth not in front of it); per frame: the image, our
map at the end of A and at the end of B, ray-cast from that camera (open3d), shaded by the surface's facing, background
grey, each object its own colour by physical id; the object's ADT boxes drawn (visit A's place red, visit B's place green).
Writes aria/views/ROOM_NAME.png.
usage: render_views.py ROOM RUN_DIR NAME [NAME ...]   (ADT instance names, as gt_check_tsdf.py prints them)"""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/home/jixian/Desktop/FT/eval_fast/vendor/pack'); import zpk_snapshot as zs
room, run, wanted = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
GT_CHECK = Path('/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/analysis/aria_20261101/gt_check_adt.py')
src = GT_CHECK.read_text(); sys.argv = [str(GT_CHECK), room, str(run), 'points']
exec(compile(src[:src.index('if kind == "points":')], str(GT_CHECK), 'exec'))   # PA, PB, BX, names, rot
EXPORT = '/home/jixian/Desktop/FT/wt_takeover_runs/analysis/history_export'
LIBS = ':'.join(f'/home/jixian/ros2_ws/install/{d}/lib' for d in sorted(os.listdir('/home/jixian/ros2_ws/install')))
ENV = dict(os.environ, LD_LIBRARY_PATH=f'{LIBS}:/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu')
D = Path(f'/home/jixian/Desktop/FT/datasets/aria_ab/{room}'); OUT = Path('/home/jixian/Desktop/FT/wt_takeover_runs/aria/views'); OUT.mkdir(exist_ok=True)
F, C = 280.0, 255.5


def final_mesh(s):
    """The final snapshot of visit s: a ray-casting scene, and per triangle its owner (-1 background, else physical id)."""
    path = run / f'session_{s}/final.4dmap.zpk'
    with tempfile.TemporaryDirectory(dir='/dev/shm') as tmp:
        zs.cut_many(str(path), [len(zs.stamps(str(path))) - 1], out=f'{tmp}/s.4dmap', work=tmp)
        subprocess.run([EXPORT, f'{tmp}/s.4dmap', f'{tmp}/s.ply'], env=ENV, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        m = json.load(open(f'{tmp}/s.ply.json')); mesh = o3d.io.read_triangle_mesh(f'{tmp}/s.ply')
    owner = np.full(len(mesh.triangles), -2, np.int64)        # -2: not shown (an object not present at the end)
    for l in m['layers']:
        sl = slice(l['face_start'], l['face_start'] + l['faces'])
        if l['layer'] == 'background': owner[sl] = -1
        elif l['native_present'] and not l.get('trajectory_points'):
            owner[sl] = l['instance_id'] if l.get('instance_id') is not None else int(l['owner']) % 100000
    keep = owner > -2
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[keep]); owner = owner[keep]
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene, owner


def render(scene, owner, T):
    v, u = np.mgrid[0:512, 0:512]
    d = np.stack([(u - C) / F, (v - C) / F, np.ones_like(u, float)], -1).reshape(-1, 3)
    d = d @ T[:3, :3].T; d /= np.linalg.norm(d, axis=1, keepdims=True)
    rays = np.c_[np.repeat(T[None, :3, 3], len(d), 0), d].astype(np.float32)
    r = scene.cast_rays(o3d.core.Tensor(rays))
    hit = np.isfinite(r['t_hit'].numpy()); prim = r['primitive_ids'].numpy(); n = r['primitive_normals'].numpy()
    shade = np.where(hit, 0.35 + 0.65 * np.abs(np.sum(n * d, 1)), 0)
    rgb = np.ones((len(d), 3))
    ow = np.where(hit, owner[np.where(hit, prim, 0)], -3)
    obj = ow >= 0
    rng = np.random.default_rng(0); pal = rng.uniform(0.25, 1.0, (4096, 3))
    rgb[ow == -1] = 0.8
    rgb[obj] = pal[ow[obj] % 4096]
    rgb = rgb * shade[:, None]; rgb[~hit] = 1.0
    return rgb.reshape(512, 512, 3)


def box_lines(uid, pose, T):
    lo, hi = BX[uid]; t, q = pose; R = rot(q)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ R.T + t
    p = (np.linalg.inv(T) @ np.c_[corners, np.ones(8)].T).T[:, :3]
    if np.any(p[:, 2] < 0.05): return []
    uv = np.c_[p[:, 0] / p[:, 2] * F + C, p[:, 1] / p[:, 2] * F + C]
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count('1') == 1]
    return [uv[[a, b]] for a, b in edges]


def best_frame(s, centre):
    rgbd = D / f'session_{s}/rgbd'; best = (-1, None)
    for i in range(0, 500, 5):
        if not (rgbd / f'{i:06d}_pose.txt').exists(): continue
        T = np.loadtxt(rgbd / f'{i:06d}_pose.txt'); p = np.linalg.inv(T) @ np.append(centre, 1)
        if p[2] < 0.3: continue
        u, v = p[0] / p[2] * F + C, p[1] / p[2] * F + C
        if not (40 <= u < 472 and 40 <= v < 472): continue
        dep = np.asarray(Image.open(rgbd / f'{i:06d}_depth.tiff'), np.float64)[int(v), int(u)]
        score = -abs(u - C) - abs(v - C) - 100 * abs(p[2] - 2.0)
        if (dep >= p[2] - 0.15 or not np.isfinite(dep)) and score > best[0] - 1e9 and (best[1] is None or score > best[0]): best = (score, i)
    return best[1]


maps = {s: final_mesh(s) for s in 'ab'}
uid_of = {names.get(str(u), {}).get('instance_name', str(u)): u for u in BX}
for name in wanted:
    uid = uid_of[name]; a, b = PA.get(uid), PB.get(uid)
    rows = [(s, best_frame(s, (a if a is not None else b)[0])) for s in 'ab']
    rows = [(s, i) for s, i in rows if i is not None]
    fig, axes = plt.subplots(2 * len(rows), 3, figsize=(15, 10.2 * len(rows)), squeeze=False)
    for k, (s, i) in enumerate(rows):
        T = np.loadtxt(D / f'session_{s}/rgbd/{i:06d}_pose.txt')
        panels = [(np.asarray(Image.open(D / f'session_{s}/rgbd/{i:06d}_color.png').convert('RGB')), f'{s.upper()} frame {i}: the image'),
                  (render(*maps['a'], T), 'our map at the end of A'), (render(*maps['b'], T), 'our map at the end of B')]
        segs = [(seg, col) for pose, col in ((a, 'red'), (b, 'lime')) if pose is not None
                and not (pose is b and a is not None and np.linalg.norm(a[0] - b[0]) < 0.01) for seg in box_lines(uid, pose, T)]
        # the zoom: around the visit-A box when it is in the image (the place the check reads), else the visit-B box
        ref = [sg for sg, col in segs if col == 'red'] or [sg for sg, col in segs if col == 'lime']
        pts = np.concatenate(ref) if ref else np.array([[C, C]])
        if not np.all((pts > -20) & (pts < 532)) and [sg for sg, col in segs if col == 'lime']:
            pts = np.concatenate([sg for sg, col in segs if col == 'lime'])
        cx, cy = pts.mean(0); half = float(np.clip(1.3 * np.ptp(pts, 0).max(), 40, 200))
        for zoom, row in ((False, axes[2 * k]), (True, axes[2 * k + 1])):
            for ax, (img, title) in zip(row, panels):
                ax.imshow(img, interpolation='nearest'); ax.axis('off')
                ax.set_title(f'{room} {name} | {title}{" (zoom)" if zoom else ""}', fontsize=9)
                for sg, col in segs: ax.plot(sg[:, 0], sg[:, 1], '-', color=col, lw=1.2)
                if zoom: ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy + half, cy - half)
                else: ax.set_xlim(0, 512); ax.set_ylim(512, 0)
    fig.tight_layout(); p = OUT / f'{room}_{name}.png'; fig.savefig(p, dpi=80); plt.close(fig); print('written', p)
