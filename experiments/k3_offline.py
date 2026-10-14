"""Offline count with the chain's own K3 code (CPU, nothing trained): which frames would K3 make keyframes?

原理思想：K3（static_kf3 014294c 起：scene_change.changed_since，输入由 GameBackend.k3_inputs / k3_reference 准备）让“相机没动够、但上一关键帧之后场景变了”的帧成为关键帧，参照是
上一关键帧自己的测量。进链之前离线数三件事：多出几个关键帧（成本，决定能否过 1.10x 速度门）；花盆停下的 3448-3465 有没有
被取到（收益）；真实 B/C 上会不会被传感器噪声触发（真实中性，应约为 0）。这里逐帧走链里的同一套代码：D1 前端（run.py:
layer.motion -> frames.MOTION）、GameBackend._sample（关键帧的取帧、裁剪、动态像素深度置零、实例掩码）、GaME 的关键帧判据
（TrackedGaME.is_keyframe，C1/C2 版）、K3 的 k3_inputs + k3_reference + changed_since（链里同一段代码，device=cpu）；只是不建图、不训练（K3 的参照只用关键帧自己的测量，与地图无关，
所以离线结果与链里一致）。与链里的唯一差别：会话开头在链里和上一会话的最后一个关键帧比；这里从本会话第一帧起算。
来源：本仓库 static_kf3 的 update_layer/backends/game/{game.py,scene_change.py}；run.py:221（G8 留出帧）、run.py:258-271（D1 每帧）。
输出：每个相机没动够的帧一行 JSON（帧号、新增掩码数、看穿区域像素数、是否成为关键帧），最后一行汇总。
usage: k3_offline.py CODE_DIR DATASET SESSION OUT_JSONL
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

CODE, DATASET, SESSION, OUT = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, str(CODE))
from update_layer.run import dataset_config, INPUT_STEP                                   # noqa: E402
from update_layer.frames import FlatSession, MOTION                                       # noqa: E402
from update_layer.core.d1 import D1, D1Config                                             # noqa: E402
from update_layer.backends.game.game import GameBackend, Crop, TrackedGaME, gu            # noqa: E402
from update_layer.backends.game.scene_change import changed_since                         # noqa: E402

torch.set_num_threads(3)
cfg, info, specs = dataset_config(DATASET)
spec = next(sp for sp in specs if sp.name == f'{DATASET}_{SESSION}')
final, first_session = spec is specs[-1], spec is specs[0]
be = GameBackend(info, own_update=False)
be.session = FlatSession(spec, pixel_step=1)                                              # GameBackend.start_session
be.crop = Crop(be.session.K, be.step)
c = be.config
d1 = D1(D1Config(voxel_size=cfg.map_resolution, truncation=cfg.truncation, max_range=cfg.max_range,
                 dynamic_labels=tuple(cfg.dynamic_semantics)))                            # layer.py:413
front = FlatSession(spec, pixel_step=INPUT_STEP[DATASET])                                # run.py: what D1 and the backend receive
indices = list(range(len(front.ids)))
held = {n for n in range(len(indices)) if n % 10 == 0 and n != 0} if final else set()   # run.py:221 g8_holdout
state = SimpleNamespace(keyframes={}, _last_keyframe_id=None, config=c)                  # what TrackedGaME.is_keyframe reads


def keyframe_dict(i, pose):
    sample, _, _ = be._sample(i)
    return {"color": gu.np2torch(sample["color"], device="cpu").permute(2, 0, 1) / 255.0,
            "depth": gu.np2torch(sample["depth"], device="cpu"), "masks": sample["masks"],
            "pose": gu.np2torch(pose, device="cpu"), "intrinsics": sample["intrinsics"]}


n_motion = n_k3 = 0
prior = 0 if first_session else 3          # a carried model already holds the earlier sessions' keyframes (> 2)
t0 = time.monotonic()
with open(OUT, 'w') as out:
    for k, i in enumerate(indices):
        if k in held:
            continue
        fr = front.load(i)
        motion = d1.process(fr)                                                     # layer.motion
        if motion is not None:
            MOTION[fr.stamp_ns] = motion
            while len(MOTION) > 64:
                MOTION.pop(next(iter(MOTION)))
        pose = np.linalg.inv(fr.T_world_cam)
        pose[:3, 3] *= be.scale
        pose = pose.astype(np.float32)
        if TrackedGaME.is_keyframe(state, pose):
            state.keyframes = {k: keyframe_dict(i, pose)}
            state._last_keyframe_id = k
            n_motion += 1
            continue
        if prior + n_motion + n_k3 <= 2:
            continue
        added, region = changed_since(GameBackend.k3_reference(state.keyframes[state._last_keyframe_id], 'cpu'),
                                      *be.k3_inputs(fr, pose, 'cpu'), c["depth_change_threshold"],
                                      c["color_error_threshold"], c["addition_coverage_threshold"])
        promote = bool(added or region)
        out.write(json.dumps(dict(frame=int(i), added_masks=added, region_px=region, keyframe=promote)) + '\n')
        out.flush()
        if promote:
            state.keyframes = {k: keyframe_dict(i, pose)}
            state._last_keyframe_id = k
            n_k3 += 1
        if k % 500 == 0:
            print(f'frame {i}: motion keyframes {n_motion}, K3 keyframes {n_k3} ({(k + 1) / (time.monotonic() - t0):.1f} frames/s)', flush=True)
    out.write(json.dumps(dict(summary=True, dataset=DATASET, session=SESSION, motion_keyframes=n_motion, k3_keyframes=n_k3,
                              frames=len(indices) - len(held))) + '\n')
print(f'K3_OFFLINE {DATASET} {SESSION}: motion keyframes {n_motion}, K3 keyframes {n_k3}, frames {len(indices) - len(held)}', flush=True)
