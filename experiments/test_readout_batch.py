"""The snapshot readout with batched voxel unions (evlight2) against the readout of 93d4b7e: identical output, and the time (GPU, short).

A B checkpoint is restored and loaded as a resumed chain loads it (prior_from_state). The readout of the map at the checkpoint's last
stamp from every stored keyframe view is computed twice:
  - by the backend's _readout (the batched unions of this tree)
  - by 93d4b7e's _readout, copied verbatim below
The background points, normals and every object's points and normals must be equal bitwise; the wall time of each is printed.

usage: test_readout_batch.py CODE_DIR DATASET B_CKPT [REPEATS]
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE, DS, B_CKPT = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
REPEATS = int(sys.argv[4]) if len(sys.argv) > 4 else 1
sys.path.insert(0, str(CODE))
from update_layer.run import dataset_config  # noqa: E402
from update_layer.backends.game import game as G  # noqa: E402
from update_layer.eval.kf_strip import restore  # noqa: E402


def old_readout(self, gm, identity, kfs, t_ns):
    """93d4b7e's GameBackend._readout, verbatim (module names through G)."""
    empty = G.EvaluationScene(t_ns, np.zeros((0, 3), np.float32), np.zeros(0, np.uint32), [],
                              np.zeros((0, 3), np.float32))
    if gm.get_xyz.shape[0] == 0 or not kfs:
        return empty
    empty3 = torch.zeros((0, 3), device="cuda")
    acc_bg = [empty3, torch.zeros(0, dtype=torch.int64, device="cuda"), empty3]
    acc_ob = [empty3, torch.zeros(0, dtype=torch.int64, device="cuda"), empty3]

    def union(run, q, ids, nn, voxel, with_id):
        q, ids, nn = torch.cat([run[0], q]), torch.cat([run[1], ids]), torch.cat([run[2], nn])
        if len(q) == 0:
            return [q, ids, nn]
        cols = torch.floor(q / voxel).long()
        if with_id:
            cols = torch.cat([cols, ids[:, None]], dim=1)
        _, inv = torch.unique(cols, dim=0, return_inverse=True)
        first = torch.full((int(inv.max()) + 1,), len(q), dtype=torch.int64, device="cuda")
        first.scatter_reduce_(0, inv, torch.arange(len(q), device="cuda"), reduce="amin")
        return [q[first], ids[first], nn[first]]

    for kf in kfs:
        K = kf["intrinsics"]
        h, w, valid = G._view_valid(kf)
        pose = kf["pose"].cpu()
        view = G.gu.flashsplat_cam(torch.zeros((3, h, w), device="cuda"), torch.zeros((h, w), device="cuda"),
                                   None, K, pose, None)
        fe = G.probe_render_fe(view, gm)
        mi = fe["median_index"].squeeze().long()
        ok = (mi >= 0) & valid.to(mi.device).reshape(mi.shape)
        depth = torch.where(ok, fe["median"].squeeze(), torch.zeros_like(fe["median"].squeeze())) / self.scale
        ok &= depth > 0
        v, u = torch.nonzero(ok, as_tuple=True)
        ident = identity[mi[v, u]].clamp(min=0)
        z = depth[v, u]
        Tw = torch.linalg.inv(pose.to(torch.float64))
        Tw[:3, 3] /= self.scale
        Tw = Tw.to("cuda", torch.float32)
        p = torch.stack([(u.float() - K[0, 2]) / K[0, 0] * z, (v.float() - K[1, 2]) / K[1, 1] * z, z], 1) \
            @ Tw[:3, :3].T + Tw[:3, 3]
        uu = (torch.arange(w, device="cuda", dtype=torch.float32) - K[0, 2]) / K[0, 0]
        vv = (torch.arange(h, device="cuda", dtype=torch.float32) - K[1, 2]) / K[1, 1]
        dd = torch.where(ok, depth, torch.full_like(depth, float("nan")))
        Pc = torch.stack([uu[None, :] * dd, vv[:, None] * dd, dd], dim=-1)
        dx, dy = torch.full_like(Pc, float("nan")), torch.full_like(Pc, float("nan"))
        dx[:, 1:-1] = Pc[:, 2:] - Pc[:, :-2]
        dy[1:-1, :] = Pc[2:, :] - Pc[:-2, :]
        nc = torch.cross(dx, dy, dim=-1)
        nc = nc / torch.linalg.norm(nc, dim=-1, keepdim=True)
        nc = torch.where(((nc * Pc).sum(-1) > 0)[..., None], -nc, nc)
        n_w = nc[v, u] @ Tw[:3, :3].T
        b, o = ident == 0, ident > 0
        acc_bg = union(acc_bg, p[b], ident[b], n_w[b], self.bg_voxel, False)
        acc_ob = union(acc_ob, p[o], ident[o], n_w[o], self.obj_voxel, True)
    if not len(acc_bg[0]) and not len(acc_ob[0]):
        return empty
    bp, bi, bn = acc_bg
    op, oi, on = acc_ob
    objects = []
    oi_np, op_np, on_np = oi.cpu().numpy(), op.cpu().numpy(), on.cpu().numpy().astype(np.float32)
    for i in np.unique(oi_np):
        m = oi_np == i
        objects.append(G.SceneObject(id=str(int(i)), instance_id=int(i),
                                     semantic=int(self.semantic_of.get(int(i), -1)),
                                     points=op_np[m].astype(np.float32), present=True,
                                     first_observed_ns=[0], last_observed_ns=[G.UINT64_MAX],
                                     normals=on_np[m]))
    return G.EvaluationScene(t_ns, bp.cpu().numpy().astype(np.float32), np.zeros(len(bp), np.uint32),
                             objects, bn.cpu().numpy().astype(np.float32))


cfg, info, specs = dataset_config(DS)
be = G.GameBackend(info, own_update=False)
ck = torch.load(B_CKPT, map_location='cpu', weights_only=False)
g = be.prior_from_state(restore(dict(backend=ck['backend']), DS)['backend'])
be.game = g
t_ns = int(ck['prev_final'])
kfs = list(g.keyframes.values())
print(f'{DS}: {len(g.uid)} rows, {len(kfs)} stored views, readout at {t_ns}', flush=True)


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g.gaussian_model.alive = g.alive_at(t_ns)
    try:
        out = fn(g.gaussian_model, g.identity, kfs, t_ns)
    finally:
        g.gaussian_model.alive = None
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


for rep in range(REPEATS):
    new, t_new = timed(be._readout)
    old, t_old = timed(lambda *a: old_readout(be, *a))
    same_bg = np.array_equal(new.points, old.points) and np.array_equal(new.normals, old.normals, equal_nan=True)
    same_ob = len(new.objects) == len(old.objects) and all(
        a.instance_id == b.instance_id and np.array_equal(a.points, b.points) and np.array_equal(a.normals, b.normals, equal_nan=True)
        for a, b in zip(new.objects, old.objects))
    print(f'repeat {rep}: background {len(new.points)} points identical {same_bg}; objects {len(new.objects)} identical {same_ob}; '
          f'time old {t_old:.2f} s, batched {t_new:.2f} s', flush=True)
print('TEST_READOUT_BATCH', 'PASS' if same_bg and same_ob else 'FAIL')
