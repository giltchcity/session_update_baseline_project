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


# Joint model over every pixel, range z as covariate (heteroscedastic regression mixture):
#   same surface:  r ~ N(mu(z), sigma(z)^2),  mu(z) = a + b z (the 10-07 real measurement was linear in range),
#                  sigma(z) = c + d z + e z^2 (axial depth noise grows ~ quadratically with range: Nguyen, Izadi, Lovell,
#                  "Modeling Kinect Sensor Noise", 3DIMPVT 2012, sigma_z = 0.0012 + 0.0019 (z - 0.4)^2 m)
#   anything else: r ~ N(m_o, S^2), one broad component of its own, so the narrow one never widens to absorb changed surfaces
#   pi per range bin (how much of each range is the same surface).
# Start: per bin the mode of r (densest 5 mm cell), a robust line through the modes; sigma 1 cm; pi 0.7. EM to convergence.
Z = np.concatenate([np.full(len(res[k][0]), (edges[k] + edges[k + 1]) / 2) for k in range(nb) if res[k]])
Rr = np.concatenate([res[k][0] for k in range(nb) if res[k]])
B = np.concatenate([np.full(len(res[k][0]), k) for k in range(nb) if res[k]])
modes, zs, ns = [], [], []
for k in range(nb):
    if res[k] and len(res[k][0]) >= 500:
        x = res[k][0]
        x = x[np.abs(x) < 1.0]
        if len(x) < 100:
            continue
        hcnt, he = np.histogram(x, bins=np.arange(-1.0, 1.0 + 0.005, 0.005))
        j = int(np.argmax(hcnt)); modes.append((he[j] + he[j + 1]) / 2); zs.append((edges[k] + edges[k + 1]) / 2); ns.append(len(x))
modes, zs, ns = np.array(modes), np.array(zs), np.array(ns, float)
X1 = np.stack([np.ones_like(zs), zs], 1)
coef = np.linalg.lstsq(X1 * np.sqrt(ns)[:, None], modes * np.sqrt(ns), rcond=None)[0]
for _ in range(5):                                         # robust: drop modes far off the line (Tukey-style reweighting)
    e_ = modes - X1 @ coef
    sc = 1.4826 * np.median(np.abs(e_)) + 1e-4
    wr = ns * np.clip(1 - (e_ / (4.685 * sc)) ** 2, 0, None) ** 2
    coef = np.linalg.lstsq(X1 * np.sqrt(wr + 1e-9)[:, None], modes * np.sqrt(wr + 1e-9), rcond=None)[0]
a, b = coef
cs = np.array([0.01, 0.0, 0.0])
m_o, S = float(np.median(Rr)), float(np.std(Rr)) + 1e-3
pi_b = np.full(nb, 0.7)
X3 = lambda z: np.stack([np.ones_like(z), z, z * z], 1)
ll_old = -np.inf
for it in range(300):
    mu = a + b * Z
    sg = np.maximum(X3(Z) @ cs, 1e-3)
    p1 = pi_b[B] * np.exp(-0.5 * ((Rr - mu) / sg) ** 2) / (np.sqrt(2 * np.pi) * sg)
    p2 = (1 - pi_b[B]) * np.exp(-0.5 * ((Rr - m_o) / S) ** 2) / (np.sqrt(2 * np.pi) * S)
    tot = p1 + p2 + 1e-300
    w1 = p1 / tot
    ll = np.log(tot).sum()
    # M-step
    W = w1 / sg ** 2
    a, b = np.linalg.lstsq(X1_ := np.stack([np.ones_like(Z), Z], 1) * np.sqrt(W)[:, None], Rr * np.sqrt(W), rcond=None)[0]
    mu = a + b * Z
    sig_b, wz, zb_ = [], [], []
    for k in np.unique(B):
        sel = B == k
        sw = w1[sel].sum()
        if sw > 50:
            sig_b.append(np.sqrt((w1[sel] * (Rr[sel] - mu[sel]) ** 2).sum() / sw)); wz.append(sw); zb_.append(Z[sel][0])
        pi_b[k] = np.clip(w1[sel].mean(), 1e-3, 1 - 1e-3)
    sig_b, wz, zb_ = np.array(sig_b), np.array(wz), np.array(zb_)
    cs = np.linalg.lstsq(X3(zb_) * np.sqrt(wz)[:, None], sig_b * np.sqrt(wz), rcond=None)[0]
    w2 = 1 - w1
    m_o = float((w2 * Rr).sum() / w2.sum()); S = float(np.sqrt((w2 * (Rr - m_o) ** 2).sum() / w2.sum())) + 1e-4
    if abs(ll - ll_old) < 1e-6 * abs(ll):
        break
    ll_old = ll
print(f'\njoint EM: {it + 1} iterations, {len(Rr)} samples')
print(f'same surface: mu(z) = {100 * a:.2f} cm + {100 * b:.3f} % x z;  sigma(z) = {100 * cs[0]:.2f} cm + {100 * cs[1]:.3f} % x z + {100 * cs[2]:.4f} % x z^2 (z in m)')
print(f'anything else: mean {100 * m_o:.1f} cm, sd {100 * S:.1f} cm')
print(f'{"range m":>10} {"n seen":>10} {"pi":>6} {"mu cm":>8} {"sigma cm":>9} {"same share":>11}')
rows = []
for k in np.unique(B):
    sel = B == k
    z = Z[sel][0]
    sgz = max(float(X3(np.array([z])) @ cs), 1e-3)
    row = dict(z=float(z), n_seen=int(seen[k]), pi=float(pi_b[k]), mu=float(a + b * z), sigma=sgz, same_share=float((w1[sel] > 0.5).mean()))
    rows.append(row)
    print(f'{edges[k]:4.2f}-{edges[k + 1]:4.2f} {seen[k]:10d} {pi_b[k]:6.3f} {100 * row["mu"]:8.2f} {100 * sgz:9.2f} {100 * row["same_share"]:10.1f}%')
wts = np.array([r_['n_seen'] for r_ in rows], float)
share = float(np.average([r_['same_share'] for r_ in rows], weights=wts))
print(f'share of A-surface pixels that B measures as the same surface (posterior > 0.5, weighted by pixels seen): {100 * share:.1f} %')
if out_json:
    Path(out_json).write_text(json.dumps(dict(dataset=ds, session=sess, mesh=mesh_path, stride=stride, pixel_step=pstep, frames=n_frames,
                                              mu_line=[float(a), float(b)], sigma_poly=[float(c_) for c_ in cs], other=[m_o, S],
                                              bins=rows, same_share=share), indent=1))
print('RESIDUAL_MODEL_DONE')
