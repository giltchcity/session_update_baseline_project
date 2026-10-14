"""The measurement model the unified rule needs, estimated from the data (CPU, read-only; no GPU, no change to any tree).

Question: when session B measures a surface that A's map already holds, how do the two disagree? Per pixel of a B frame:
    r = d_B - d_A      d_B = B's measured depth, d_A = the depth of A's map surface along the same pixel ray
d_A comes from ray casting A's exported surface (session_a/tsdf_final.ply, the median-depth TSDF export that G1 scores) from B's camera
(Open3D RaycastingScene, CPU). B's frames are read by update_layer.frames.FlatSession, the mapping's own reader (same crop, depth
range, session alignment). Dynamic classes (the dataset config's dynamic_semantics) are left out.
Model per range bin of d_B, fitted by EM, no threshold set by hand:
    p(r) = pi * N(r; mu, sigma^2) + (1 - pi) * U(r; -R, R)        R = the dataset's max_range (config)
  N = the same surface, measured again (mu = the systematic A->B offset, sigma = the measurement noise);
  U = anything else (the surface changed, moved, or a different surface is in front / behind).
Depth edges are fitted separately (a pixel whose 4-neighbours differ in B's depth by more than 3 sigma of that bin's first fit), so the
noise at discontinuities is measured, not assumed.
Printed: per bin n, pi, mu, sigma (cm); a least-squares line mu(z) = a + b z and sigma(z) = c + d z over the bins; the share of A-surface
pixels that B measures as the same surface (posterior > 0.5).
usage: residual_model.py DATASET SESSION A_MESH FRAME_STRIDE PIXEL_STEP [OUT_JSON]
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

TREE = Path('/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/worktrees/chain_clean')
sys.path.insert(0, str(TREE))
from update_layer.frames import FlatSession  # noqa: E402
from update_layer.run import dataset_config  # noqa: E402

ds, sess, mesh_path, stride, pstep = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
out_json = sys.argv[6] if len(sys.argv) > 6 else None
cfg, info, specs = dataset_config(ds)
spec = next(sp for sp in specs if sp.name.endswith('_' + sess))
R = float(cfg.max_range)
dyn = set(int(x) for x in cfg.dynamic_semantics)
fs = FlatSession(spec, pixel_step=pstep)
mesh = o3d.io.read_triangle_mesh(mesh_path)
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
print(f'{ds} {sess}: {len(fs.ids)} frames, every {stride}th, pixel_step {pstep}; A mesh {mesh_path} '
      f'({len(mesh.triangles)} triangles); max_range {R} m; dynamic classes {sorted(dyn)}', flush=True)

edges = np.arange(0.0, R + 0.25, 0.25)
nb = len(edges) - 1
keep = 60000                       # samples kept per bin (uniform reservoir), enough for a 3-parameter fit
res = [[] for _ in range(nb)]
dz = [[] for _ in range(nb)]       # local depth difference of the pixel, for the edge split
seen = np.zeros(nb, dtype=np.int64)
rng = np.random.default_rng(0)
n_frames = n_pix = n_hit = 0
for i in range(0, len(fs.ids), stride):
    fr = fs.load(i)
    d = fr.depth
    h, w = d.shape
    valid = np.isfinite(d)
    if fr.semantic is not None and dyn:
        valid &= ~np.isin(fr.semantic, list(dyn))
    v, u = np.nonzero(valid)
    if not len(v):
        continue
    K = fr.K
    dc = np.stack([(u + K.offset - K.cx) / K.fx, (v + K.offset - K.cy) / K.fy, np.ones(len(u))], 1)     # z = 1: t_hit is depth
    Rw, tw = fr.T_world_cam[:3, :3], fr.T_world_cam[:3, 3]
    rays = np.concatenate([np.broadcast_to(tw, (len(u), 3)), dc @ Rw.T], 1).astype(np.float32)
    t = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
    hit = np.isfinite(t)
    zb = d[v, u]
    # local depth difference in B's own depth: max over the 4-neighbours of |d - d_nb| (NaN neighbours ignored)
    pad = np.pad(d, 1, constant_values=np.nan)
    nbr = np.stack([pad[v, u + 1], pad[v + 2, u + 1], pad[v + 1, u], pad[v + 1, u + 2]], 1)
    loc = np.nanmax(np.abs(nbr - zb[:, None]), axis=1)
    loc[~np.isfinite(loc)] = np.inf
    r = zb - t
    b = np.clip(np.searchsorted(edges, zb, side='right') - 1, 0, nb - 1)
    m = hit & (np.abs(r) <= R)
    n_frames += 1; n_pix += len(v); n_hit += int(m.sum())
    for k in np.unique(b[m]):
        sel = m & (b == k)
        rk, lk = r[sel], loc[sel]
        # reservoir: keep a uniform sample of everything seen in the bin
        seen[k] += len(rk)
        res[k].append(rk); dz[k].append(lk)
        cat = np.concatenate(res[k]); cl = np.concatenate(dz[k])
        if len(cat) > keep:
            idx = rng.choice(len(cat), keep, replace=False)
            cat, cl = cat[idx], cl[idx]
        res[k], dz[k] = [cat], [cl]
print(f'frames {n_frames}, valid static pixels {n_pix}, with an A surface on the ray {n_hit} ({100 * n_hit / max(n_pix, 1):.1f} %)', flush=True)


def em(x, R, it=200):
    """pi N(mu, s^2) + (1 - pi) U(-R, R); robust start (median, MAD)."""
    mu = np.median(x); s = max(1.4826 * np.median(np.abs(x - mu)), 1e-4); pi = 0.8
    u = 1.0 / (2 * R)
    for _ in range(it):
        g = pi * np.exp(-0.5 * ((x - mu) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
        wgt = g / (g + (1 - pi) * u)
        sw = wgt.sum()
        if sw < 10:
            break
        mu_n = (wgt * x).sum() / sw
        s_n = max(np.sqrt((wgt * (x - mu_n) ** 2).sum() / sw), 1e-4)
        pi_n = sw / len(x)
        if abs(mu_n - mu) < 1e-7 and abs(s_n - s) < 1e-7 and abs(pi_n - pi) < 1e-7:
            mu, s, pi = mu_n, s_n, pi_n
            break
        mu, s, pi = mu_n, s_n, pi_n
    return pi, mu, s, wgt


rows = []
print(f'\n{"range m":>10} {"n seen":>10} {"pi":>6} {"mu cm":>8} {"sigma cm":>9} | {"edge share":>10} {"edge pi":>8} {"edge mu":>8} {"edge sigma":>10}')
for k in range(nb):
    if not res[k] or len(res[k][0]) < 500:
        continue
    x, l = res[k][0], dz[k][0]
    pi, mu, s, wgt = em(x, R)
    edge = l > 3 * s
    row = dict(z0=float(edges[k]), z1=float(edges[k + 1]), n_seen=int(seen[k]), pi=float(pi), mu=float(mu), sigma=float(s),
               same_share=float((wgt > 0.5).mean()), edge_share=float(edge.mean()))
    line = f'{edges[k]:4.2f}-{edges[k + 1]:4.2f} {seen[k]:10d} {pi:6.3f} {100 * mu:8.2f} {100 * s:9.2f} | {100 * edge.mean():9.1f}%'
    if (~edge).sum() >= 500 and edge.sum() >= 500:
        pin, mun, sn, _ = em(x[~edge], R)
        pie, mue, se, _ = em(x[edge], R)
        row.update(interior=dict(pi=float(pin), mu=float(mun), sigma=float(sn)), edge=dict(pi=float(pie), mu=float(mue), sigma=float(se)))
        line += f' {pie:8.3f} {100 * mue:8.2f} {100 * se:10.2f}'
    rows.append(row)
    print(line, flush=True)
if len(rows) >= 3:
    z = np.array([(r_['z0'] + r_['z1']) / 2 for r_ in rows]); wts = np.array([r_['n_seen'] for r_ in rows], float)
    A = np.stack([np.ones_like(z), z], 1) * np.sqrt(wts)[:, None]
    a, bb = np.linalg.lstsq(A, np.array([r_['mu'] for r_ in rows]) * np.sqrt(wts), rcond=None)[0]
    c, dd = np.linalg.lstsq(A, np.array([r_['sigma'] for r_ in rows]) * np.sqrt(wts), rcond=None)[0]
    share = float(np.average([r_['same_share'] for r_ in rows], weights=wts))
    print(f'\nmu(z) = {100 * a:.2f} cm + {100 * bb:.2f} % x z;  sigma(z) = {100 * c:.2f} cm + {100 * dd:.2f} % x z  (fits over bins, weighted by pixels)')
    print(f'share of A-surface pixels that B measures as the same surface (posterior > 0.5): {100 * share:.1f} %')
    if out_json:
        Path(out_json).write_text(json.dumps(dict(dataset=ds, session=sess, mesh=mesh_path, stride=stride, pixel_step=pstep,
                                                  frames=n_frames, bins=rows, mu_line=[float(a), float(bb)], sigma_line=[float(c), float(dd)],
                                                  same_share=share), indent=1))
print('RESIDUAL_MODEL_DONE')
