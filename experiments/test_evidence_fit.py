"""Offline test of the E1 measurement-model fit (worktrees/evrow update_layer/backends/game/evidence_model.py) on the residuals the
mapping sees: the A map's row centres projected into B frames (every row in the frustum with a valid depth, occluded ones included),
r = measured depth - row depth. Runs the fit on (a) the first FIRST frames of B every STEP-th frame (the first rounds of a session)
and (b) the whole session every STRIDE-th frame, and prints the fitted line, the noise per range, the weights.
usage: test_evidence_fit.py A_CKPT DATASET SESSION PIXEL_STEP FIRST STEP STRIDE [A_MESH]
  with A_MESH: r = measured depth - depth of A's exported surface along the pixel ray (the map's surface, as the first-echo readout)
"""
import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path('/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/worktrees/evrow')
sys.path.insert(0, str(TREE))
from update_layer.frames import FlatSession  # noqa: E402
from update_layer.run import dataset_config  # noqa: E402
from update_layer.core.l2.evidence import EvidenceConfig  # noqa: E402
import importlib.util  # noqa: E402
spec_ = importlib.util.spec_from_file_location('evidence_model', TREE / 'update_layer/backends/game/evidence_model.py')
em = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(em)

a_path, ds, sess, pstep, first, step, stride = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
mesh_path = sys.argv[8] if len(sys.argv) > 8 else None
if mesh_path:
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(mesh_path)))
cfg, info, specs = dataset_config(ds)
spec = next(sp for sp in specs if sp.name.endswith('_' + sess))
fs = FlatSession(spec, pixel_step=pstep)
b = torch.load(a_path, map_location='cpu', mmap=True, weights_only=False)['backend']
P = b['model'][1].detach().numpy() / float(b['model'][11])
INT64_MAX = np.iinfo(np.int64).max
end = np.minimum(np.minimum(b['death_evidence'].numpy(), b['death_prune'].numpy()), b['death_state'].numpy())
P = P[end == INT64_MAX]


def sample_mesh(frames):
    zs, rs = [], []
    for i in frames:
        fr = fs.load(i)
        d = fr.depth
        valid = np.isfinite(d)
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


def sample(frames):
    if mesh_path:
        return sample_mesh(frames)
    zs, rs = [], []
    for i in frames:
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
        d = fr.depth[v[inside], u[inside]]
        zz = z[ok][inside]
        val = np.isfinite(d)
        zs.append(d[val]); rs.append(d[val] - zz[val])
    return torch.from_numpy(np.concatenate(zs)).float(), torch.from_numpy(np.concatenate(rs)).float()


cfgE = EvidenceConfig()
for lab, frames in (('first frames', range(0, min(first, len(fs.ids)), step)), ('whole session', range(0, len(fs.ids), stride))):
    z, r = sample(frames)
    keep = torch.randperm(len(z))[:2_000_000]
    z, r = z[keep], r[keep]
    m = em.fit(z, r, float(info.depth_range[1]), cfgE.depth_tolerance)
    print(f'\n== {ds} {sess}, {lab}: {len(list(frames))} frames, {len(z)} samples')
    if m is None:
        print('   no fit (fewer than two range bins with a robust mode)')
        continue
    zc = 0.5 * (m['edges'][:-1] + m['edges'][1:])
    print(f'   same surface: mu(z) = {100 * m["a"][0]:.2f} cm + {100 * m["a"][1]:.3f} % z;  other: mean {100 * m["other"][0]:.1f} cm, sd {100 * m["other"][1]:.1f} cm')
    bb = (torch.bucketize(z, m['edges']) - 1).clamp(0, len(zc) - 1)
    for zq in (1.0, 2.0, 3.0, 4.0, 5.0, 7.0):
        k = int(torch.clamp(torch.bucketize(torch.tensor([zq]), m['edges']) - 1, 0, len(zc) - 1))
        n = int((bb == k).sum())
        if n:
            print(f'     z {zq:.1f} m: mu {100 * (m["a"][0] + m["a"][1] * zq):6.2f} cm  sigma {100 * float(m["sigma"][k]):5.2f} cm  pi {float(m["pi"][k]):.3f}  samples {n}')
    p_on, past = em.posterior(m, z, r)
    print(f'   posterior: share on the row\'s surface {100 * float((p_on > 0.5).float().mean()):.1f} %, past it {100 * float(((p_on <= 0.5) & past).float().mean()):.1f} %')
print('TEST_FIT_DONE')
