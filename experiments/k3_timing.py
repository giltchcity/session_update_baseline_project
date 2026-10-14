"""Per-part time of the K3 check on one unmoved frame (GPU, nothing trained): where do the 0.18 s/frame of speed150 go?
原理思想：速度门测的是整段建图，和 CPU 上的离线任务重叠时不可靠；这里把 K3 检查拆开逐段计时（取帧 _sample、上传 GPU、
参照点投影、按像素取最小 scatter_reduce、掩码求和、颜色差+池化），每段前后 torch.cuda.synchronize，30 帧取平均，
真实与合成各一组。各段与 static_kf3 的 scene_change.changed_since 逐行相同（这里复制是为了分段计时；总时间另外直接调用
原函数测一次，核对复制没有走样）。来源：static_kf3 7d2db97 的 GameBackend.integrate（K3 块）与 scene_change.py。
usage: k3_timing.py CODE_DIR
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CODE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(CODE))
from update_layer.run import dataset_config                                               # noqa: E402
from update_layer.frames import FlatSession                                               # noqa: E402
from update_layer.backends.game.game import GameBackend, Crop, gu                        # noqa: E402
from update_layer.backends.game.scene_change import changed_since                        # noqa: E402


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


for ds, ref_i, frames in (('synthetic', 200, range(201, 231)), ('real', 200, range(201, 231))):
    cfg, info, specs = dataset_config(ds)
    spec = next(sp for sp in specs if sp.name == f'{ds}_b')
    be = GameBackend(info, own_update=False)
    be.session = FlatSession(spec, pixel_step=1)
    be.crop = Crop(be.session.K, be.step)
    c = be.config
    eps, ce, cov = c['depth_change_threshold'], c['color_error_threshold'], c['addition_coverage_threshold']

    def kf_of(i):
        fr = be.session.load(i)
        pose = np.linalg.inv(fr.T_world_cam)
        pose[:3, 3] *= be.scale
        sample, _, _ = be._sample(i)
        return {"color": gu.np2torch(sample["color"], device="cuda").permute(2, 0, 1) / 255.0,
                "depth": gu.np2torch(sample["depth"], device="cuda"), "masks": sample["masks"].cuda(),
                "pose": gu.np2torch(pose.astype(np.float32), device="cuda"), "intrinsics": sample["intrinsics"]}

    ref = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in kf_of(ref_i).items()}      # stored keyframes live on the CPU
    T = dict(sample=0.0, upload=0.0, ref_upload=0.0, project=0.0, scatter=0.0, masks=0.0, through_pool=0.0, function=0.0)
    for i in frames:
        fr = be.session.load(i)
        pose = np.linalg.inv(fr.T_world_cam)
        pose[:3, 3] *= be.scale
        t0 = sync()
        sample, _, _ = be._sample(i)
        t1 = sync()
        cur = {"color": gu.np2torch(sample["color"], device="cuda").permute(2, 0, 1) / 255.0,
               "depth": gu.np2torch(sample["depth"], device="cuda"), "masks": sample["masks"].cuda(),
               "pose": gu.np2torch(pose.astype(np.float32), device="cuda"), "intrinsics": sample["intrinsics"]}
        t2 = sync()
        d_cur, c_cur = cur["depth"], cur["color"]
        dev = d_cur.device
        H, W = d_cur.shape
        kc = torch.as_tensor(cur["intrinsics"], dtype=torch.float32, device=dev)
        kr = torch.as_tensor(ref["intrinsics"], dtype=torch.float32, device=dev)
        d_ref, c_ref = ref["depth"].to(dev), ref["color"].to(dev)
        t3 = sync()
        v, u = torch.nonzero(d_ref > 0, as_tuple=True)
        z = d_ref[v, u]
        p = torch.stack(((u.float() - kr[0, 2]) / kr[0, 0] * z, (v.float() - kr[1, 2]) / kr[1, 1] * z, z), 1)
        Tm = cur["pose"].float() @ torch.linalg.inv(ref["pose"].to(dev).float())
        q = p @ Tm[:3, :3].T + Tm[:3, 3]
        zq = q[:, 2]
        zs = zq.clamp(min=1e-6)
        uq = torch.round(q[:, 0] / zs * kc[0, 0] + kc[0, 2])
        vq = torch.round(q[:, 1] / zs * kc[1, 1] + kc[1, 2])
        ok = (zq > 0) & (uq >= 0) & (uq < W) & (vq >= 0) & (vq < H)
        pix, zq, col = (vq[ok] * W + uq[ok]).long(), zq[ok], c_ref[:, v[ok], u[ok]]
        t4 = sync()
        render = torch.full((H * W,), float("inf"), device=dev).scatter_reduce(0, pix, zq, "amin")
        front = zq <= render[pix]
        render_col = torch.zeros((3, H * W), device=dev)
        render_col[:, pix[front]] = col[:, front]
        have = torch.isfinite(render).view(H, W)
        t5 = sync()
        valid = have & (d_cur > 0)
        diff = d_cur - torch.where(have, render.view(H, W), torch.zeros_like(d_cur))
        ma = cur["masks"].to(dev) & valid
        den = (ma & (diff < eps)).sum((1, 2))
        num = (ma & (diff < -eps)).sum((1, 2))
        added = int(((den > 0) & (num.float() > cov * den.float())).sum())
        t6 = sync()
        dm = d_cur.reshape(-1)[pix]
        through = (dm > 0) & (dm - zq > eps) & ((c_cur.reshape(3, -1)[:, pix] - render_col[:, pix]).abs().mean(0) > ce)
        hit = torch.zeros(H * W, device=dev)
        hit[pix[through]] = 1.0
        region = int((-F.max_pool2d(-hit.view(1, 1, H, W), 5, stride=1, padding=2)).sum())
        t7 = sync()
        res = changed_since(ref, cur, eps, ce, cov)
        t8 = sync()
        assert res == (added, region), (res, added, region)
        for k, a, b in (('sample', t0, t1), ('upload', t1, t2), ('ref_upload', t2, t3), ('project', t3, t4), ('scatter', t4, t5),
                        ('masks', t5, t6), ('through_pool', t6, t7), ('function', t7, t8)):
            T[k] += b - a
    n = len(frames)
    print(f'{ds} b, {H}x{W}, reference {ref_i}, frames {frames.start}-{frames.stop - 1}: ' +
          ', '.join(f'{k} {v / n * 1000:.2f} ms' for k, v in T.items()), flush=True)
print('K3_TIMING_DONE')
