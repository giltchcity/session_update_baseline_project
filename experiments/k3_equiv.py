"""Same decisions, less time? K3's efficiency version (GameBackend._scene_change) against the full test of 7d2db97 (GPU).
原理思想：效率版只改了“怎么算”（读链里已读好的帧、颜色与掩码按需、参照每个关键帧上传一次），没有改“算什么”：同一帧、同一个
参照，两版给出的（新增掩码数，看穿区域像素数）必须相同。真实数据上效率版用预读帧的像素网格（输入步长的子采样网格），全量版用
GaME 的居中裁剪网格，两者差不到一个像素，所以真实帧上比较的是“是否判为变化”。同时逐帧计时效率版（含 GPU 同步）。
帧：合成 B 参照 3447 对 3448-3554（花盆停下）、参照 200 对 201-230（相机在动、场景不变）；真实 B 参照 200 对 201-230。
来源：static_kf3 7d2db97（全量版 scene_change.changed_since）与 adf3e74（效率版）。
usage: k3_equiv.py CODE_DIR(efficiency version) OLD_SCENE_CHANGE_PY(7d2db97's scene_change.py)
"""
import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

CODE, OLD = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(CODE))
from update_layer.run import dataset_config, INPUT_STEP                                   # noqa: E402
from update_layer.frames import FlatSession                                               # noqa: E402
from update_layer.backends.game.game import GameBackend, Crop, gu                        # noqa: E402
spec_ = importlib.util.spec_from_file_location('old_scene_change', OLD)
old = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(old)

for ds, ref_i, frames in (('synthetic', 3447, range(3448, 3555)), ('synthetic', 200, range(201, 231)), ('real', 200, range(201, 231))):
    cfg, info, specs = dataset_config(ds)
    spec = next(sp for sp in specs if sp.name == f'{ds}_b')
    be = GameBackend(info, own_update=False)
    be.session = FlatSession(spec, pixel_step=1)
    be.crop = Crop(be.session.K, be.step)
    front = FlatSession(spec, pixel_step=INPUT_STEP[ds])
    c = be.config

    def pose_of(i):
        p = np.linalg.inv(front.load(i).T_world_cam)
        p[:3, 3] *= be.scale
        return p.astype(np.float32)

    def kf_of(i):
        sample, _, _ = be._sample(i)
        return {"color": gu.np2torch(sample["color"], device="cuda").permute(2, 0, 1) / 255.0,
                "depth": gu.np2torch(sample["depth"], device="cuda"), "masks": sample["masks"].cuda(),
                "pose": gu.np2torch(pose_of(i), device="cuda"), "intrinsics": sample["intrinsics"]}

    ref_gpu = kf_of(ref_i)
    stored = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in ref_gpu.items()}
    be.game = SimpleNamespace(keyframes={0: stored}, _last_keyframe_id=0, config=c)
    same = differ = 0
    t_new = []
    lines = []
    for i in frames:
        fr = front.load(i)
        p = pose_of(i)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        new = be._scene_change(fr, p)
        torch.cuda.synchronize()
        t_new.append(time.perf_counter() - t0)
        ref_full = old.changed_since(stored, kf_of(i), c['depth_change_threshold'], c['color_error_threshold'],
                                     c['addition_coverage_threshold'])
        if ds == 'synthetic':
            ok = new == ref_full
        else:
            ok = bool(new[0] or new[1]) == bool(ref_full[0] or ref_full[1])
        same += ok
        differ += not ok
        if not ok or new != (0, 0) or ref_full != (0, 0):
            lines.append(f'{i}:{new}/{ref_full}')
    t = np.array(t_new[1:]) * 1000
    print(f'{ds} b ref {ref_i}, frames {frames.start}-{frames.stop - 1}: agree {same}, differ {differ}; efficiency version '
          f'{t.mean():.2f} ms/frame (median {np.median(t):.2f}, max {t.max():.2f}); non-zero or differing (new/full): {" ".join(lines[:40])}',
          flush=True)
print('K3_EQUIV_DONE')
