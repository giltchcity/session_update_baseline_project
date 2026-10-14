"""Offline count (CPU, nothing trained): which unmoved frames show a change since the last keyframe's own measurement?
原理思想：GaME 的地图只由关键帧训练，一帧对地图有用的信息 = 关键帧集合里还没有的信息。GaME 只在相机移动够了时取关键帧
（平移 > keyframe_translation_diff 或任一欧拉角 > 50°，game.py:487-507）；相机没动够时，这一帧的视角和上一关键帧几乎相同，
它唯一可能带来的新信息是“上一关键帧之后场景变了”（合成 B：3447 之后相机停住，花盆又挪了 13.7 cm，从没进图）。
K2（e37823e）拿地图渲染和这一帧比（GaME 的新增/冲突测试）；链里删除归更新层（own_update 关），层还没判掉的旧行每一帧都与
测量冲突，于是几乎每个静止帧都成了关键帧（speed150：150 帧里多出约 80 个关键帧）。这里把参照从“地图渲染”换成“上一关键帧
自己的测量”，测试与阈值不变：
  新增（GaME _find_added_geometry_masks，C5 版）：每个实例掩码里，参照有表面且本帧有深度的像素中，本帧比参照近
    depth_change_threshold 以上的占比 > addition_coverage_threshold；
  冲突（GaME _detect_conflicting_gaussians）：参照的表面点投到本帧，本帧在它后面超过 depth_change_threshold（看穿了）且颜色
    L1 > color_error_threshold，这些点在本帧连成 5x5 腐蚀后仍非空的区域。
有变化的帧成为关键帧，也成为新的参照，所以同一个变化只取一次。人等动态像素（语义动态类 + D1 运动像素）深度置零，与链里
_sample 一致。阈值全是 flat.yaml 原值，不新增参数；投影用已知位姿（不配准）。
来源：GaME（CVPR 2026）baselines/GaME/src/entities/game.py:198-232（新增）、299-352（冲突）、487-507（关键帧，C1/C2 版）；
configs/flat/flat.yaml；本仓库 GameBackend._sample / _masks（输入与掩码），d1_trace.py（D1 按链里的方式运行）。
输出：每个“多出的关键帧”一行 JSON（帧号、新增掩码数、冲突区域像素数），最后一行汇总（原关键帧数、多出的数）。
usage: k3_offline.py DATASET SESSION OUT_JSONL
"""
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from update_layer.run import dataset_config, INPUT_STEP                                   # noqa: E402
from update_layer.frames import FlatSession, synthetic_session, real_session, dynamic_mask, MOTION   # noqa: E402
from update_layer.core.d1 import D1, D1Config                                            # noqa: E402

DATASET, SESSION, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
GAME_CFG = yaml.safe_load(open('/home/jixian/Desktop/FT/baselines/GaME/configs/flat/flat.yaml'))


def flat(key):
    """a flat.yaml value wherever it sits in the file"""
    stack = [GAME_CFG]
    while stack:
        d = stack.pop()
        if key in d:
            return d[key]
        stack += [v for v in d.values() if isinstance(v, dict)]
    raise KeyError(key)


SCALE, EPS = float(flat('scale')), float(flat('depth_change_threshold'))          # scene units = metres x scale
EPS_M = EPS / SCALE                                                               # 0.30 m
KF_T_M = float(flat('keyframe_translation_diff')) / SCALE                         # 0.125 m
COLOR, COVER, MIN_OP = float(flat('color_error_threshold')), float(flat('addition_coverage_threshold')), float(flat('min_opacity'))
STEP = {'synthetic': 1, 'real': 2, 'flat': 1}[DATASET]                           # game.py STEP (= run.INPUT_STEP)
cfg, info, specs = dataset_config(DATASET)
spec = next(sp for sp in specs if sp.name == f'{DATASET}_{SESSION}')
final = spec is specs[-1]
d1 = D1(D1Config(voxel_size=cfg.map_resolution, truncation=cfg.truncation, max_range=cfg.max_range,
                 dynamic_labels=tuple(cfg.dynamic_semantics)))                   # layer.py:413, as d1_trace.py
front = FlatSession(spec, pixel_step=INPUT_STEP[DATASET])                       # run.py: what D1 and the backend receive
full = FlatSession(spec, pixel_step=1)                                           # GameBackend._sample reads full frames
K0 = full.K


def grid(n_full, c, offset):
    """GameBackend Crop.axis: the stride-STEP grid centred on the principal point (game.py:205-232)"""
    best = None
    for first in range(STEP):
        cs = (c - offset - first) / STEP
        n = int(round(2 * cs + 1))
        if n < 2 or first + STEP * (n - 1) > n_full - 1:
            continue
        err = abs((n - 1) / 2 - cs)
        if best is None or err < best[0] - 1e-9 or (abs(err - best[0]) < 1e-9 and n > best[2]):
            best = (err, first, n)
    return best[1], best[2]


top, rows = grid(K0.height, K0.cy, K0.offset)
left, cols = grid(K0.width, K0.cx, K0.offset)
fx, fy, cx, cy = K0.fx / STEP, K0.fy / STEP, (cols - 1) / 2.0, (rows - 1) / 2.0


def crop(img):
    return img[top:top + STEP * (rows - 1) + 1:STEP, left:left + STEP * (cols - 1) + 1:STEP]


min_mask_px = max(1, 50 // (STEP * STEP))                                       # GameBackend: min_mask_px_full 50
vv, uu = np.indices((rows, cols), dtype=np.float32)
ray = np.stack(((uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)), -1)         # pixel -> camera ray (z = 1)


def masks(instance, semantic, dyn):
    """GameBackend._masks: every physical instance + every 4-connected region of one class among the pixels without one"""
    out = [instance == i for i, n in zip(*np.unique(instance[instance > 0], return_counts=True)) if n >= min_mask_px]
    if semantic is not None:
        rest = (instance <= 0) & ~dyn
        for lab in np.unique(semantic[rest]).tolist():
            n, comp = cv.connectedComponents((rest & (semantic == lab)).astype(np.uint8), connectivity=4)
            out += [m for m in (comp == k for k in range(1, n)) if m.sum() >= min_mask_px]
    return out


def sample(i):
    """GameBackend._sample: cropped depth (metres, dynamic pixels 0), colour 0..1, instance cover, world-from-camera pose"""
    f = full.load(i, color=True)
    depth = np.nan_to_num(crop(f.depth), nan=0.0).astype(np.float32)
    instance = crop(f.instance).astype(np.int64)
    semantic = crop(f.semantic) if f.semantic is not None else None
    small = type(f)(f.index, f.stamp_ns, depth, instance, semantic, f.T_world_cam, f.K)
    dyn = dynamic_mask(small, info.dynamic_semantics)
    m = MOTION.get(int(f.stamp_ns))
    if m is not None:                                       # D1 motion pixels, nearest to this grid (frames.motion_mask)
        mh, mw = m.shape
        dyn = dyn | m[(np.arange(rows) * mh) // rows][:, (np.arange(cols) * mw) // cols]
    depth[dyn] = 0.0
    return dict(depth=depth, color=crop(f.color).astype(np.float32) / 255.0, masks=masks(instance, semantic, dyn),
                T=f.T_world_cam.astype(np.float64), stamp=int(f.stamp_ns))


def moved(T_ref, T):
    """GaME is_keyframe (C1 relative pose, C2 degrees): translation > keyframe_translation_diff or any Euler angle > 50"""
    d = np.linalg.inv(T_ref) @ T            # = last_pose @ inv(pose) of GameBackend.is_keyframe (C1): ref camera <- current camera
    return np.linalg.norm(d[:3, 3]) > KF_T_M or np.any(np.abs(Rotation.from_matrix(d[:3, :3]).as_euler('xyz', degrees=True)) > 50)


def change(ref, cur):
    """the two GaME tests with the reference keyframe's measurement in place of the map render"""
    ok = ref['depth'] > 0
    p = ray[ok] * ref['depth'][ok, None]                                    # reference surface points, reference camera
    rc = np.linalg.inv(cur['T']) @ ref['T']                                 # current camera <- reference camera
    q = p @ rc[:3, :3].T + rc[:3, 3]
    z = q[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.rint(q[:, 0] / z * fx + cx)
        v = np.rint(q[:, 1] / z * fy + cy)
    inside = (z > 0) & (u >= 0) & (u < cols) & (v >= 0) & (v < rows)
    u, v, z, col = u[inside].astype(np.int64), v[inside].astype(np.int64), z[inside], ref['color'][ok][inside]
    flat_ix = v * cols + u
    zbuf = np.full(rows * cols, np.inf, dtype=np.float64)                   # the reference's surface seen from here
    np.minimum.at(zbuf, flat_ix, z)
    zbuf = zbuf.reshape(rows, cols)
    have = np.isfinite(zbuf)
    d_cur = cur['depth']
    # new geometry: measured closer than the reference by more than EPS, per instance mask (C5: measured pixels only)
    valid = have & (d_cur > 0)
    diff = d_cur - np.where(have, zbuf, 0.0)
    added = 0
    for m in cur['masks']:
        ma = m & valid
        den = (ma & (diff < EPS_M)).sum()
        if den and (ma & (diff < -EPS_M)).sum() / den > COVER:
            added += 1
    # conflict: a reference surface point the current measurement sees through (behind it by EPS) with a colour change
    front_pt = np.abs(z - zbuf[v, u]) < 1e-9                               # only the visible (front) reference points
    dc = d_cur[v, u]
    through = front_pt & (dc > 0) & (dc - z > EPS_M)
    cdiff = np.abs(cur['color'][v, u] - col).mean(-1) > COLOR
    hit = np.zeros((rows, cols), np.uint8)
    hit[v[through & cdiff], u[through & cdiff]] = 1
    region = int(cv.erode(hit, np.ones((5, 5), np.uint8)).sum())
    return added, region


ref = None
n_kf_motion = n_kf_k3 = 0
out = open(OUT, 'w')
indices = list(range(len(front.ids)))                                     # run.py: every frame of the session
held = {n for n in range(len(indices)) if n % 10 == 0 and n != 0} if final else set()   # run.py:221 g8_holdout
for k, i in enumerate(indices):
    if k in held:
        continue
    fr = front.load(i)
    motion = d1.process(fr)                                 # layer.motion: D1 front end on every frame
    MOTION[int(fr.stamp_ns)] = motion
    while len(MOTION) > 64:
        MOTION.pop(next(iter(MOTION)))
    T = fr.T_world_cam.astype(np.float64)
    if ref is None or moved(ref['T'], T):
        ref = sample(i)
        n_kf_motion += 1
        continue
    if n_kf_motion + n_kf_k3 <= 2:                          # GaME's change detection starts after two keyframes
        continue
    cur = sample(i)
    added, region = change(ref, cur)
    if added or region:
        n_kf_k3 += 1
        out.write(json.dumps(dict(frame=int(i), added_masks=added, conflict_region_px=region)) + '\n')
        out.flush()
        ref = cur
out.write(json.dumps(dict(summary=True, dataset=DATASET, session=SESSION, motion_keyframes=n_kf_motion, extra_keyframes=n_kf_k3,
                          frames=len(indices) - len(held))) + '\n')
out.close()
print(f'K3_OFFLINE {DATASET} {SESSION}: motion keyframes {n_kf_motion}, extra keyframes {n_kf_k3}, frames {len(indices) - len(held)}', flush=True)
