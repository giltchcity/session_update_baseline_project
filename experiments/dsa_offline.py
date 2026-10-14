"""Offline proxy: which rows would GaME's own removal test take out at the chain's keyframes? (GPU renders, nothing trained)
原理思想：GaME 在每个关键帧上判“这个高斯所在的位置已经空了”时，要求测量同时否定它的深度（测得的表面在它后面超过
depth_change_threshold）和颜色（颜色误差超过 color_error_threshold），并且这些高斯渲染出来连成一块可见区域（5x5 腐蚀后
仍非空）才删。物体被拿走/挪走后，旧位置后面露出别的东西：深度、颜色都对不上，会删；深色物体在远处被深度读穿时，像素颜色
仍是这个物体：颜色不矛盾，不删。这里用会话末的存档地图（按 T1 取每个关键帧时刻还活着的高斯）在链里的关键帧上跑同一个
测试，数出会被删掉的行，对照真实参考表面（误删）与合成 GT（拖影、桌上物品、旧位置）。代理，不等于链内（存档地图是会话末
的几何；链内删除后还会影响之后的帧）。来源：GaME（CVPR 2026）变化检测 baselines/GaME/src/entities/game.py:300-370
（_detect_conflicting_gaussians，阈值 flat.yaml 原值）；链内入口 GameBackend.integrate 的 detect_removals（own_update）。
usage: dsa_offline.py CODE_DIR CKPT DATASET SESSION [REFERENCE_PLY|-]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

CODE, CKPT, DATASET, SESSION = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3], sys.argv[4]
REF = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != '-' else None
sys.path.insert(0, str(CODE))
from update_layer.run import dataset_config                                                    # noqa: E402
from update_layer.backends.game.game import GameBackend, Crop, gu                              # noqa: E402
from update_layer.eval.kf_strip import restore                                                 # noqa: E402
from update_layer.frames import FlatSession, synthetic_session                                 # noqa: E402

INT64_MAX = 9223372036854775807
torch.set_num_threads(2)
_, info, specs = dataset_config(DATASET)
spec = next(sp for sp in specs if sp.name == f'{DATASET}_{SESSION}')
ck = torch.load(CKPT, mmap=True, map_location='cpu', weights_only=False)
be = GameBackend(info, own_update=False)
be.game = be.prior_from_state(restore(ck, DATASET)['backend'])
g = be.game
be.session = FlatSession(spec, pixel_step=1)
be.crop = Crop(be.session.K, be.step)
sess_index = len(specs) - 1 if SESSION == specs[-1].name.split('_')[-1] else [sp.name for sp in specs].index(spec.name)
kfs = sorted(((int(g.kf_stamp[k]), k) for k in g.keyframes if g.kf_session.get(k, -1) == sess_index))
t_end = int(ck['prev_final'])
n = len(g.uid)
first_flag = torch.full((n,), INT64_MAX, dtype=torch.int64, device='cuda')
t0 = time.monotonic()
for j, (stamp, k) in enumerate(kfs):
    index = int(ck['backend']['keyframes'][k]['index'])
    sample, instance, dyn = be._sample(index)
    kf = {"color": gu.np2torch(sample["color"], device="cuda").permute(2, 0, 1) / 255.0,
          "depth": gu.np2torch(sample["depth"], device="cuda"), "masks": sample["masks"].cuda(),
          "pose": gu.np2torch(sample["pose"], device="cuda"), "intrinsics": sample["intrinsics"]}
    with torch.no_grad():
        g.gaussian_model.alive = g.alive_at(stamp)
        conflict = g._detect_conflicting_gaussians(kf)
        g.gaussian_model.alive = None
    if conflict is not None:
        first_flag[conflict & (first_flag == INT64_MAX)] = stamp
    if j % 100 == 0:
        print(f'keyframe {j}/{len(kfs)} flagged so far {int((first_flag < INT64_MAX).sum())} ({(j + 1) / (time.monotonic() - t0):.1f} kf/s)', flush=True)
b = ck['backend']
alive_end = g.alive_at(t_end).cpu().numpy()
flag = (first_flag < INT64_MAX).cpu().numpy()
ident = b['identity'].numpy()
xyz = b['model'][1].detach().numpy() / float(be.scale)
print(f'DSA_OFFLINE {DATASET} {SESSION}: keyframes {len(kfs)}; rows flagged {flag.sum()} of {n}; of the rows alive at the end flagged '
      f'{(flag & alive_end).sum()} of {alive_end.sum()}', flush=True)
if REF:
    from plyfile import PlyData
    v = PlyData.read(REF)['vertex']
    d = cKDTree(np.stack((v['x'], v['y'], v['z']), 1).astype(np.float64)).query(xyz[flag & alive_end], k=1)[0]
    print(f'  flagged rows alive at the end: within 5 cm of the reference {np.mean(d <= 0.05) * 100:.1f} % (would be false removals), '
          f'beyond 10 cm {np.mean(d > 0.10) * 100:.1f} %', flush=True)
    ids, cnt = np.unique(ident[flag & alive_end], return_counts=True)
    print('  by identity:', ', '.join(f'id{a} {c}' for a, c in sorted(zip(ids, cnt), key=lambda t: -t[1])[:10]), flush=True)
if DATASET == 'synthetic' and SESSION == 'b':
    fs = FlatSession(synthetic_session('b'), pixel_step=1)
    f = fs.load(3447)
    K, z = f.K, f.depth
    vv, uu = np.indices(z.shape, dtype=np.float32)
    ok = (f.instance == 109) & np.isfinite(z) & (z > 0)
    cam = np.stack(((uu[ok] + K.offset - K.cx) / K.fx * z[ok], (vv[ok] + K.offset - K.cy) / K.fy * z[ok], z[ok], np.ones(ok.sum())), 1)
    plant = cKDTree((cam @ f.T_world_cam.T)[:, :3]).query(xyz, k=1)[0]
    groups = {'plant 109 <=3cm of its 3447 surface': (ident == 109) & (plant <= 0.03), 'plant 109 3-10cm': (ident == 109) & (plant > 0.03) & (plant <= 0.10),
              'plant 109 >10cm (trail)': (ident == 109) & (plant > 0.10), 'table items 79 80 81 117 170 181': np.isin(ident, [79, 80, 81, 117, 170, 181]),
              'chair_0004 (72)': ident == 72, 'table_0003 (164)': ident == 164, 'chair_0001 (69)': ident == 69, 'background (id 0)': ident == 0}
    for name, sel in groups.items():
        sel = sel & alive_end
        print(f'  {name:36s} {sel.sum():8d} rows alive at the end | flagged {(flag & sel).sum():7d} = {(flag & sel).sum() / max(sel.sum(), 1) * 100:5.1f} %', flush=True)
    for name, (s_, i_) in {'bathroom a2700': ('a', 2700), 'bathroom a2985': ('a', 2985), 'kitchen wall b2625': ('b', 2625), 'chairs b1395': ('b', 1395),
                           'curtain b1845': ('b', 1845), 'last keyframe b3447': ('b', 3447)}.items():
        f = FlatSession(synthetic_session(s_), pixel_step=1).load(i_)
        K, z = f.K, f.depth
        okp = np.isfinite(z) & (z > 0)
        cam = np.stack(((uu[okp] + K.offset - K.cx) / K.fx * z[okp], (vv[okp] + K.offset - K.cy) / K.fy * z[okp], z[okp], np.ones(okp.sum())), 1)
        dd = cKDTree((cam @ f.T_world_cam.T)[:, :3]).query(xyz, k=1, distance_upper_bound=0.03)[0]
        sel = np.isfinite(dd) & alive_end
        print(f'  surface of {name:24s} {sel.sum():8d} rows | flagged {(flag & sel).sum():7d} = {(flag & sel).sum() / max(sel.sum(), 1) * 100:5.1f} %', flush=True)
print('DSA_OFFLINE_DONE', flush=True)
