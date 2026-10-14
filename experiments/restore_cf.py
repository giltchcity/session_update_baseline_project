"""Counterfactual render: put the A-time parameters back into the rows a later map shares with A (by uid), render, compare.
原理思想：若卫生间的损坏只来自 B 改写了它测到的 A 行，那么把这些行的参数换回 A 时的值，渲染应当回到 A 的画面；只换尺度+
旋转与全部换回分开做，能分出是形状还是位置/颜色/不透明度造成的。只渲染，不训练。来源：10-13 同类反事实（veil_source）；
渲染流程同 experiments/render_views.py（t1.alive_at、flashsplat_render）。
usage: restore_cf.py CODE_DIR OUT_DIR A_CKPT B_CKPT VIEWS(comma)
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CODE, OUT, A_CKPT, B_CKPT, VIEWS = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5].split(',')
sys.path.insert(0, str(CODE))
from update_layer.run import dataset_config                                                      # noqa: E402
from update_layer.backends.game.game import GameBackend, GaussianModel, Crop, gu, flashsplat_render, t1   # noqa: E402
from update_layer.frames import FlatSession, synthetic_session                                   # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
_, info, _ = dataset_config('synthetic')
scale = GameBackend(info, own_update=False).scale
A = torch.load(A_CKPT, map_location='cpu', weights_only=False, mmap=True)['backend']
ckB = torch.load(B_CKPT, map_location='cpu', weights_only=False, mmap=True)
B, stamp = ckB['backend'], int(ckB['prev_final'])
alive = t1.alive_at(stamp, B['identity'], B['created'], B['state_birth'], B['death_state'], B['death_evidence'],
                    B.get('bg_birth'), B.get('death_prune'))
uidA, uidB = A['uid'].numpy(), B['uid'].numpy()
_, ia, ib = np.intersect1d(uidA, uidB, return_indices=True)
start = int(B['session_start']) if 'session_start' in B else int(B['session_starts'][-1])
touched = B['last_update'].numpy()[ib] >= start
attrs = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')
arms = {'B_as_is': (), 'B_touched_shape_from_A': ('_scaling', '_rotation'), 'B_touched_all_from_A': attrs}
cams = {}
for key in VIEWS:
    frame = FlatSession(synthetic_session(key[0]), pixel_step=1).load(int(key[1:]), color=True)
    crop = Crop(frame.K, 1)
    photo = crop(frame.color)
    pose = np.linalg.inv(frame.T_world_cam)
    pose[:3, 3] *= scale
    h, w = photo.shape[:2]
    cams[key] = (gu.flashsplat_cam(torch.zeros((3, h, w), device='cuda'), torch.zeros((h, w), device='cuda'), None, crop.K,
                                   torch.as_tensor(pose.astype(np.float32)), None), photo)
pipe, background = gu.flashsplat_pipe(), torch.ones(3, device='cuda')
print(f'shared rows {len(ib)}, of them touched by B measurements {touched.sum()}', flush=True)
with torch.inference_mode():
    for arm, restored in arms.items():
        gm = GaussianModel(3, scale=scale)
        gm.active_sh_degree = int(B['model'][0])
        for j, attr in enumerate(attrs, 1):
            values = B['model'][j].detach().clone()
            if attr in restored:
                values[ib[touched]] = A['model'][j].detach()[ia[touched]]
            setattr(gm, attr, values[alive].cuda())
        for key, (view, photo) in cams.items():
            rgb = flashsplat_render(view, gm, pipe, background, obj_num=1)['render'].clamp(0, 1)
            pixels = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(pixels).save(OUT / f'{arm}__{key}.png')
            print(f'{arm:26s} {key}: MAE vs photo {np.abs(pixels.astype(float) - photo).mean():.2f}', flush=True)
        del gm
        torch.cuda.empty_cache()
print('RESTORE_CF_DONE', flush=True)
