"""restore_cf2 with the arm the supervisor asked for at 13:44: EVERY row of A (also the 1.23 M rows B never measured on their surface but
whose parameters B's optimisation changed) back to its A values, with and without B's own rows (GPU renders only, nothing trained).
The rest of this docstring is restore_cf2's (same rendering, same classification).
Where does the bathroom's error after B come from? Counterfactual renders of the B map with A's rows put back (GPU, nothing trained).
原理思想：A 建好的远处卫生间在 B 之后变差（10-14 测试 1：a2985 照片 MAE A 4.27 -> B 34.02；隐藏 B 新建行后 17.58；把 B 量到过的 A 行
的参数换回 A 时的值又降一些；约 9.7 没拆开）。剩下的候选是“A 结束时还活着、B 末已经不活”的 A 行：被 GaME 的不透明度剪枝结束
（P1：只结束、仍留在检查点）或已从检查点删除（_prune_unrenderable），以及被物体状态/层的证据结束。这里在 B 末的地图上逐项把
它们放回、各自单独和叠加地渲染，与 A 末地图本身（锚点）比，看每一项各占多少，最后一档应回到锚点附近；回不到就说明还有别的
成分，不靠猜。只渲染，不训练。
arms（都用 B 末的存活掩码为底，参数默认取 B 的值）：
  A_at_A_end                     A 末的地图本身（锚点）
  B_as_is                        B 末的地图
  hide_B_born                    去掉 B 里新建的行
  hide_B_born+touched_from_A     再把 B 量到过（last_update >= B 起点）的 A 行参数换回 A 时的值
  +revive_ended_A                再把“A 末活着、B 里还在、B 末不活”的 A 行设为活着，参数取 A 时的值
  +revive_deleted_A              再把“A 末活着、B 检查点里已没有”的 A 行从 A 加回来
  B_as_is+revive_ended_A / B_as_is+revive_deleted_A   单独一项的作用
来源：experiments/restore_cf.py（同一渲染流程：t1.alive_at、GaussianModel 载入、flashsplat_render）；行的分类按 checkpoint 里的
uid / created / last_update / 寿命字段。
usage: restore_cf2.py CODE_DIR OUT_DIR A_CKPT B_CKPT VIEWS(comma)
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
ATTRS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')


def alive(b, stamp):
    a = t1.alive_at(stamp, b['identity'], b['created'], b['state_birth'], b['death_state'], b['death_evidence'],
                    b.get('bg_birth'), b.get('death_prune'))
    return (a if torch.is_tensor(a) else torch.as_tensor(a)).cpu().numpy().astype(bool)


ckA = torch.load(A_CKPT, map_location='cpu', weights_only=False, mmap=True)
ckB = torch.load(B_CKPT, map_location='cpu', weights_only=False, mmap=True)
A, B = ckA['backend'], ckB['backend']
aliveA, aliveB = alive(A, int(ckA['prev_final'])), alive(B, int(ckB['prev_final']))
uidA, uidB = A['uid'].numpy(), B['uid'].numpy()
_, ia, ib = np.intersect1d(uidA, uidB, return_indices=True)            # rows of A still in B's checkpoint
start = int(B['session_start']) if B.get('session_start') is not None else int(B['session_starts'][-1])
b_born = B['created'].numpy() >= start
touched = np.zeros(len(uidB), bool)
touched[ib] = B['last_update'].numpy()[ib] >= start
ended = np.zeros(len(uidB), bool)                                       # A alive at A's end, still in B, not alive at B's end
ended[ib] = aliveA[ia] & ~aliveB[ib]
deleted = np.ones(len(uidA), bool)
deleted[ia] = False
deleted &= aliveA                                                       # A alive at A's end, gone from B's checkpoint
pB = {k: B['model'][j].detach() for j, k in enumerate(ATTRS, 1)}
pA = {k: A['model'][j].detach() for j, k in enumerate(ATTRS, 1)}
fromA = {k: v.clone() for k, v in pB.items()}                           # B's rows with A's values where A has the row
for k in ATTRS:
    fromA[k][ib] = pA[k][ia]
print(f'A rows alive at A end {aliveA.sum()}; B rows {len(uidB)}, alive at B end {aliveB.sum()}; B-born {b_born.sum()}; '
      f'A rows in B touched by B {touched.sum()}; ended in B (still stored) {ended.sum()}; deleted from B {deleted.sum()}', flush=True)


def model(rows_b, params_b, extra_a=None):
    gm = GaussianModel(3, scale=scale)
    gm.active_sh_degree = int(B['model'][0])
    for k in ATTRS:
        v = params_b[k][torch.as_tensor(rows_b)]
        if extra_a is not None and extra_a.any():
            v = torch.cat([v, pA[k][torch.as_tensor(extra_a)]])
        setattr(gm, k, v.cuda())
    return gm


def mixed(base, rows):
    out = {k: v.clone() for k, v in base.items()}
    for k in ATTRS:
        out[k][torch.as_tensor(rows)] = fromA[k][torch.as_tensor(rows)]
    return out


tB = mixed(pB, touched)
arms = {
    'A_at_A_end': None,
    'B_as_is': (aliveB, pB, None),
    'hide_B_born': (aliveB & ~b_born, pB, None),
    'hide_B_born+touched_from_A': (aliveB & ~b_born, tB, None),
    'hide_B_born+all_A_from_A': (aliveB & ~b_born, fromA, None),
    'hide_B_born+all_A_from_A+revive_ended_A': ((aliveB & ~b_born) | ended, fromA, None),
    'B_as_is+all_A_from_A': (aliveB, fromA, None),
}
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
    Image.fromarray(photo).save(OUT / f'photo__{key}.png')
pipe, background = gu.flashsplat_pipe(), torch.ones(3, device='cuda')
table = {}
with torch.inference_mode():
    for arm, spec in arms.items():
        if spec is None:                                                # the anchor: A's own map at A's end
            gm = GaussianModel(3, scale=scale)
            gm.active_sh_degree = int(A['model'][0])
            for k in ATTRS:
                setattr(gm, k, pA[k][torch.as_tensor(aliveA)].cuda())
        else:
            gm = model(*spec)
        for key, (view, photo) in cams.items():
            rgb = flashsplat_render(view, gm, pipe, background, obj_num=1)['render'].clamp(0, 1)
            pixels = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(pixels).save(OUT / f'{arm}__{key}.png')
            table.setdefault(arm, {})[key] = float(np.abs(pixels.astype(float) - photo).mean())
        del gm
        torch.cuda.empty_cache()
print('photo MAE (0-255) per view:')
print('%-30s' % 'arm' + ''.join('%10s' % k for k in VIEWS))
for arm, row in table.items():
    print('%-30s' % arm + ''.join('%10.2f' % row[k] for k in VIEWS), flush=True)
print('RESTORE_CF3_DONE', flush=True)
