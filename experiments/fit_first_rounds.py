"""Why the E1 fit fails in a session's first rounds (CPU, read-only): the residuals of B's first frames against A's exported surface
(the snapshot's first echo, as test_evidence_fit.py with A_MESH), fitted as the first rounds see them (frames 0..k), with the fit's
intermediate values: per range bin the sample, the mode inside the window, the MAD scale; the biweight line through the modes; the EM's
final line and noise.
usage: fit_first_rounds.py A_MESH DATASET SESSION PIXEL_STEP FRAMES(comma list of 'first:step' sets, e.g. 1:1,60:30,150:10)
"""
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

TREE = Path('/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/worktrees/evsnap')
sys.path.insert(0, str(TREE))
from update_layer.frames import FlatSession  # noqa: E402
from update_layer.run import dataset_config  # noqa: E402
from update_layer.core.l2.evidence import EvidenceConfig  # noqa: E402
import importlib.util  # noqa: E402
spec_ = importlib.util.spec_from_file_location('evidence_model', TREE / 'update_layer/backends/game/evidence_model.py')
em = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(em)

mesh_path, ds, sess, pstep, sets = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(mesh_path)))
cfg, info, specs = dataset_config(ds)
spec = next(sp for sp in specs if sp.name.endswith('_' + sess))
fs = FlatSession(spec, pixel_step=pstep)
window = EvidenceConfig().depth_tolerance
zmax = float(info.depth_range[1])


def residuals(frames):
    zs, rs = [], []
    for i in frames:
        fr = fs.load(i)
        d = fr.depth
        valid = np.isfinite(d) & (d > 0)
        if fr.semantic is not None and cfg.dynamic_semantics:
            valid &= ~np.isin(fr.semantic, list(cfg.dynamic_semantics))
        v, u = np.nonzero(valid)
        K = fr.K
        dc = np.stack([(u + K.offset - K.cx) / K.fx, (v + K.offset - K.cy) / K.fy, np.ones(len(u))], 1)
        Rw, tw = fr.T_world_cam[:3, :3], fr.T_world_cam[:3, 3]
        rays = np.concatenate([np.broadcast_to(tw, (len(u), 3)), dc @ Rw.T], 1).astype(np.float32)
        t = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        hit = np.isfinite(t)
        zs.append(d[v, u][hit]); rs.append(d[v, u][hit] - t[hit])
    return torch.from_numpy(np.concatenate(zs)).float(), torch.from_numpy(np.concatenate(rs)).float()


for st in sets.split(','):
    first, step = (int(x) for x in st.split(':'))
    frames = list(range(0, min(first, len(fs.ids)), step))
    z, r = residuals(frames)
    print(f'\n== {ds} {sess} frames {frames[:6]}{"..." if len(frames) > 6 else ""} ({len(frames)} frames): {len(z)} samples, '
          f'z {float(z.min()):.2f}-{float(z.max()):.2f} m, |r| <= window {100 * float((r.abs() <= window).float().mean()):.1f} %')
    edges = torch.arange(0.0, zmax + em.BIN, em.BIN)
    b = (torch.bucketize(z, edges) - 1).clamp(0, len(edges) - 2)
    zc = 0.5 * (edges[:-1] + edges[1:])
    modes, zs_, ws_ = [], [], []
    for k in torch.unique(b).tolist():
        x = r[(b == k) & (r.abs() <= window)]
        n_all = int((b == k).sum())
        if len(x) < em.MIN_SAMPLE:
            print(f'   bin {float(zc[k]):.3f} m: {n_all} samples, {len(x)} in the window: no mode')
            continue
        m = em._mode(x, window)
        mad = 1.4826 * float((x - m).abs().median())
        modes.append(m); zs_.append(float(zc[k])); ws_.append(float(len(x)))
        print(f'   bin {float(zc[k]):.3f} m: {n_all:7d} samples, {len(x):7d} in the window, mode {100 * m:7.2f} cm, MAD scale {100 * mad:6.2f} cm')
    if len(modes) >= 2:
        a0, a1 = em._tukey_line(torch.tensor(zs_), torch.tensor(modes), torch.tensor(ws_))
        print(f'   biweight line through the modes: {100 * a0:.2f} cm + {100 * a1:.3f} % z')
    m = em.fit(z, r, zmax, window)
    if m is None:
        print('   fit: None')
        continue
    print(f'   EM: mu(z) = {100 * m["a"][0]:.2f} cm + {100 * m["a"][1]:.3f} % z; other mean {100 * m["other"][0]:.1f} cm sd {100 * m["other"][1]:.1f} cm')
    for k in torch.unique(b).tolist():
        print(f'      bin {float(zc[k]):.3f} m: sigma {100 * float(m["sigma"][k]):6.2f} cm  pi {float(m["pi"][k]):.3f}')
    p_on, past = em.posterior(m, z, r)
    print(f'   posterior: on {100 * float((p_on > 0.5).float().mean()):.1f} %, past {100 * float(((p_on <= 0.5) & past).float().mean()):.1f} %, '
          f'in front {100 * float(((p_on <= 0.5) & ~past).float().mean()):.1f} %')
print('FIT_FIRST_ROUNDS_DONE')
