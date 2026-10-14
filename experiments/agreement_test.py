"""The per-row agreement test with the measurement model the data gives (CPU, read-only), checked on both datasets before any code.
Model of one session (residual_model.py on that session's frames against A's map; the fitted JSON):
    r = d_B - z_row at the row's pixel in a B frame;   same surface: N(mu(z), sigma(z)^2),  anything else: N(m_o, S^2),  pi per range
Per observation the posterior of 'same surface' (the fit's own EM weights; no threshold). Per A row (alive at B's start, A's position):
    n_same = sum of posteriors, mean residual rbar over them, n_through = other-component weight with r beyond the surface (r > mu),
    n_hidden = other-component weight with r nearer than the surface.
Decision per row, from the model only:
    no evidence      n_same + n_through < 1                       -> keep, unmoved (nothing measured it)
    seen through     n_through > n_same                            -> hand over (the present measures past it)
    in place         rbar is explained by 0 better than by mu(z): N(rbar; 0, s2) >= N(rbar; mu(z), s2),  s2 = sigma(z)^2 / n_same
                                                                   -> keep, unmoved
    displaced        otherwise (the session's systematic offset explains it better) -> hand over (the present's measurement defines it)
Read against what happened in a mapping run (an optional B checkpoint: the rows GaME's prune erased there) and, for real, against
B's official reference (distance of the row's A position).
usage: agreement_test.py A_CKPT DATASET SESSION FRAME_STRIDE PIXEL_STEP MODEL_JSON [B_CKPT [REFERENCE_PLY]]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path('/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/worktrees/chain_clean')
sys.path.insert(0, str(TREE))
from update_layer.frames import FlatSession  # noqa: E402
from update_layer.run import dataset_config  # noqa: E402

a_path, ds, sess, stride, pstep, model_path = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
b_path = sys.argv[7] if len(sys.argv) > 7 else None
ref_path = sys.argv[8] if len(sys.argv) > 8 else None
mdl = json.loads(Path(model_path).read_text())
a0, a1 = mdl['mu_line']; c0, c1, c2 = mdl['sigma_poly']; m_o, S = mdl['other']
bins = mdl['bins']
pz = np.array([b['z'] for b in bins]); ppi = np.array([b['pi'] for b in bins])
mu = lambda z: a0 + a1 * z
sig = lambda z: np.maximum(c0 + c1 * z + c2 * z * z, 1e-3)
pi_of = lambda z: np.interp(z, pz, ppi)
cfg, info, specs = dataset_config(ds)
spec = next(sp for sp in specs if sp.name.endswith('_' + sess))
fs = FlatSession(spec, pixel_step=pstep)
ckA = torch.load(a_path, map_location='cpu', mmap=True, weights_only=False)
bA = ckA['backend']
scale = float(bA['model'][11])
xyz = bA['model'][1].detach().numpy() / scale
uid = bA['uid'].numpy()
INT64_MAX = np.iinfo(np.int64).max
end = np.minimum(np.minimum(bA['death_evidence'].numpy(), bA['death_prune'].numpy()), bA['death_state'].numpy())
alive = end == INT64_MAX                   # rows alive at A's end (an ended row is history, not the map B starts from)
P, U = xyz[alive], uid[alive]
N = len(P)
n_same = np.zeros(N); s_r = np.zeros(N); s_z = np.zeros(N); n_thr = np.zeros(N); n_hid = np.zeros(N)
nf = 0
for i in range(0, len(fs.ids), stride):
    fr = fs.load(i)
    R, t = fr.T_world_cam[:3, :3], fr.T_world_cam[:3, 3]
    pc = (P - t) @ R
    z = pc[:, 2]
    ok = z > 0.1
    K = fr.K
    u = np.floor(pc[ok, 0] / z[ok] * K.fx + K.cx - K.offset + 0.5).astype(np.int64)
    v = np.floor(pc[ok, 1] / z[ok] * K.fy + K.cy - K.offset + 0.5).astype(np.int64)
    h, w = fr.depth.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx = np.where(ok)[0][inside]
    d = fr.depth[v[inside], u[inside]]
    val = np.isfinite(d)
    if fr.semantic is not None and cfg.dynamic_semantics:
        val &= ~np.isin(fr.semantic[v[inside], u[inside]], list(cfg.dynamic_semantics))
    idx, d = idx[val], d[val]
    zz = z[idx]
    r = d - zz
    pi = pi_of(d)
    g1 = pi * np.exp(-0.5 * ((r - mu(d)) / sig(d)) ** 2) / (np.sqrt(2 * np.pi) * sig(d))
    g2 = (1 - pi) * np.exp(-0.5 * ((r - m_o) / S) ** 2) / (np.sqrt(2 * np.pi) * S)
    ps = g1 / (g1 + g2 + 1e-300)
    np.add.at(n_same, idx, ps); np.add.at(s_r, idx, ps * r); np.add.at(s_z, idx, ps * d)
    beyond = r > mu(d)
    np.add.at(n_thr, idx, (1 - ps) * beyond); np.add.at(n_hid, idx, (1 - ps) * ~beyond)
    nf += 1
print(f'{ds} {sess}: {nf} B frames (every {stride}th), {N} A rows alive at A end; model mu = {100 * a0:.2f} cm + {100 * a1:.3f} % z, '
      f'sigma(1 m) {100 * sig(1.0):.2f} cm, sigma(3 m) {100 * sig(3.0):.2f} cm', flush=True)
noev = (n_same + n_thr) < 1
thr = ~noev & (n_thr > n_same)
rest = ~noev & ~thr
rbar = np.where(n_same > 0, s_r / np.maximum(n_same, 1e-9), 0.0)
zbar = np.where(n_same > 0, s_z / np.maximum(n_same, 1e-9), 1.0)
s2 = sig(zbar) ** 2 / np.maximum(n_same, 1e-9)
ll0 = -0.5 * rbar ** 2 / s2
ll1 = -0.5 * (rbar - mu(zbar)) ** 2 / s2
inplace = rest & (ll0 >= ll1)
displaced = rest & ~inplace
dec = {'no evidence (keep)': noev, 'in place (keep)': inplace, 'displaced (hand over)': displaced, 'seen through (hand over)': thr}
print(f'\n   {"decision":26s} {"rows":>9s} {"share":>7s}  median |rbar| cm  median n_same')
for k, m in dec.items():
    print(f'   {k:26s} {int(m.sum()):9d} {100 * m.mean():6.1f}%  {100 * np.median(np.abs(rbar[m])) if m.any() else 0:13.2f}  {np.median(n_same[m]) if m.any() else 0:12.1f}')
print('\n   by range of the row (mean measured depth), share handed over (displaced + seen through):')
for z0 in np.arange(0, 6, 1.0):
    m = rest & (zbar >= z0) & (zbar < z0 + 1)
    mm = (~noev) & (zbar >= z0) & (zbar < z0 + 1)
    if mm.sum() > 1000:
        print(f'     {z0:.0f}-{z0 + 1:.0f} m: rows with evidence {int(mm.sum()):8d}; displaced {100 * (displaced & mm).sum() / mm.sum():5.1f} %, seen through {100 * (thr & mm).sum() / mm.sum():5.1f} %, in place {100 * (inplace & mm).sum() / mm.sum():5.1f} %')
if b_path:
    ckB = torch.load(b_path, map_location='cpu', mmap=True, weights_only=False)
    bB = ckB['backend']
    s0, s1 = int(bB['session_start']), int(ckB['prev_final'])
    order = np.argsort(bB['uid'].numpy()); ub = bB['uid'].numpy()
    pos = np.clip(np.searchsorted(ub, U, sorter=order), 0, len(order) - 1)
    hit = ub[order[pos]] == U
    ib = order[pos]
    dp = bB['death_prune'].numpy()[ib]; de = bB['death_evidence'].numpy()[ib]
    erased = hit & (dp > s0) & (dp <= s1) & ~((de > s0) & (de <= s1))
    retired = hit & (de > s0) & (de <= s1)
    print(f'\n   against the mapping run {b_path}: GaME-erased {int(erased.sum())}, layer-retired {int(retired.sum())}')
    for k, m in dec.items():
        print(f'     {k:26s} erased by GaME {100 * (erased & m).sum() / max(m.sum(), 1):5.1f} %   retired by the layer {100 * (retired & m).sum() / max(m.sum(), 1):5.1f} %')
if ref_path:
    import open3d as o3d
    from scipy.spatial import cKDTree
    tree = cKDTree(np.asarray(o3d.io.read_point_cloud(ref_path).points))
    dref = tree.query(P, workers=2)[0]
    print(f'\n   distance of the row (A position) to the reference {ref_path}:')
    for k, m in dec.items():
        if m.any():
            print(f'     {k:26s} within 5 cm {100 * (dref[m] <= 0.05).mean():5.1f} %   median {100 * np.median(dref[m]):6.2f} cm')
print('AGREEMENT_DONE')
