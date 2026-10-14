"""How much of what B observed is visible in B's keyframes at all (CPU, read-only), on both datasets.

GaME seeds and trains only at keyframes, and the official export renders only the stored keyframes. A surface B observed only
between keyframes is in no map row and in no exported view.

Reference points
  real       the G1 samples of B_reference_1cm.ply inside the oriented room crop (eval/geometry_g1.py)
  synthetic  area samples of the exact GT of B's last frame (cvpr_eval_common.worldmesh, render-visible entities)
A point is observed by a frame when it projects inside the image with a valid measured depth within 5 cm of its own depth
(dynamic pixels excluded); the frame list is every STRIDE-th frame of B (all frames) or the keyframes of a B run (by stamp).
Reported: the share of reference points observed by any frame, by any keyframe, and the keyframes' share of the frames'.

usage: keyframe_coverage.py DATASET B_CKPT STRIDE [EVIDENCE_HZ]
  EVIDENCE_HZ: also the coverage of the keyframes plus every frame the layer reads at that rate (30 Hz input / EVIDENCE_HZ)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path('/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/worktrees/evsnap')
sys.path.insert(0, str(TREE))
from update_layer.frames import FlatSession, dynamic_mask  # noqa: E402
from update_layer.run import dataset_config  # noqa: E402

DS, B_CKPT, STRIDE = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg, info, specs = dataset_config(DS)
fs = FlatSession(next(sp for sp in specs if sp.name.endswith('_b')), pixel_step=1)
FT = Path('/home/jixian/Desktop/FT')
if DS == 'real':
    sys.path.insert(0, str(FT / 'tools/abc_eval_v2'))
    sys.path.insert(0, str(FT / 'eval'))
    import open3d as o3d
    import geometry_score as GS
    from geometry_g1 import rotate
    crop = json.loads((FT / 'eval/crops/real_B.json').read_text())
    lo, hi, yaw = np.array(crop['lo']), np.array(crop['hi']), crop['yaw_deg']
    rp, _ = GS.surface_sample(o3d.io.read_triangle_mesh(str(FT / 'results/abc_eval_v2/geometry/B_reference_1cm.ply')))
    pts = rp[~((rotate(rp, yaw) < lo) | (rotate(rp, yaw) > hi)).any(1)].astype(np.float64)
else:
    sys.path.insert(0, '/mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts')
    import cvpr_eval_common as C
    frame = len(C.GT['b']['time_s']) - 1
    parts = []
    for i in C.GT['b']['instance_ids'].tolist():
        i = int(i)
        if i in C.IX['b'] and C.GT['b']['render_visible'][frame, C.IX['b'][i]]:
            v, f = C.worldmesh('b', i, frame)
            if len(f):
                parts.append((v, f))
    areas = np.array([np.linalg.norm(np.cross(v[f][:, 1] - v[f][:, 0], v[f][:, 2] - v[f][:, 0]), axis=1).sum() for v, f in parts])
    n_total = 300000
    pts = np.concatenate([C.sample_surface(v, f, max(1, int(round(n_total * a / areas.sum()))), 20261014 + k)
                          for k, ((v, f), a) in enumerate(zip(parts, areas))]).astype(np.float64)

bb = torch.load(B_CKPT, map_location='cpu', mmap=True, weights_only=False)['backend']
b_session = len(bb['session_starts']) - 1
kf_stamps = sorted(int(bb['kf_stamp'][k]) for k in bb['keyframes'] if int(bb['kf_session'].get(k, -1)) == b_session)
frame_stamps = np.array([fs.stamp_ns(i) for i in range(len(fs.ids))], dtype=np.int64)
kf_index = set(np.searchsorted(frame_stamps, kf_stamps).tolist())


def observed_by(indices, why=False):
    seen = np.zeros(len(pts), bool)
    # why: for every point, over the keyframes, whether it was in a keyframe's image at all, and whether the keyframe's pixel
    # there was invalid (no depth), dynamic (a person) or measured something else (farther than 5 cm from the point)
    in_img = np.zeros(len(pts), bool); invalid = np.zeros(len(pts), bool); dyn_px = np.zeros(len(pts), bool)
    other = np.zeros(len(pts), bool)
    for i in indices:
        fr = fs.load(i)
        d = np.nan_to_num(fr.depth, nan=0.0)
        dm = dynamic_mask(fr, info.dynamic_semantics)
        d_raw = d.copy()
        d[dm] = 0.0
        R, t = fr.T_world_cam[:3, :3], fr.T_world_cam[:3, 3]
        pc = (pts - t) @ R
        z = pc[:, 2]
        ok = z > 0.05
        K = fr.K
        u = np.floor(pc[:, 0] / np.where(ok, z, 1) * K.fx + K.cx - K.offset + 0.5).astype(np.int64)
        v = np.floor(pc[:, 1] / np.where(ok, z, 1) * K.fy + K.cy - K.offset + 0.5).astype(np.int64)
        h, w = d.shape
        ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
        dd = np.zeros(len(pts)); dd[ok] = d[v[ok], u[ok]]
        seen |= ok & (dd > 0) & (np.abs(dd - z) <= 0.05)
        if why:
            raw = np.zeros(len(pts)); raw[ok] = d_raw[v[ok], u[ok]]
            dmk = np.zeros(len(pts), bool); dmk[ok] = dm[v[ok], u[ok]]
            in_img |= ok
            invalid |= ok & (raw <= 0)
            dyn_px |= ok & dmk & (raw > 0)
            other |= ok & (dd > 0) & (np.abs(dd - z) > 0.05)
    if why:
        return seen, in_img, invalid, dyn_px, other
    return seen


all_frames = sorted(set(range(0, len(fs.ids), STRIDE)) | kf_index)
seen_all = observed_by(all_frames)
seen_kf, in_img, invalid, dyn_px, other = observed_by(sorted(kf_index), why=True)
print(f'== {DS} B: {len(pts)} reference points, {len(fs.ids)} frames, {len(kf_index)} keyframes of {B_CKPT}')
print(f'   observed by any frame (every {STRIDE}th + keyframes, {len(all_frames)}): {100 * seen_all.mean():.2f} %')
print(f'   observed by any keyframe: {100 * seen_kf.mean():.2f} %')
print(f'   of the points the frames observed, the keyframes observed {100 * (seen_kf & seen_all).sum() / max(seen_all.sum(), 1):.2f} %;  '
      f'observed only between keyframes: {100 * (seen_all & ~seen_kf).sum() / len(pts):.2f} % of all reference points')
if len(sys.argv) > 4:
    hz = float(sys.argv[4])
    step = max(1, int(round(30.0 / hz)))
    light = sorted(set(range(0, len(fs.ids), step)) | kf_index)
    seen_l = observed_by(light)
    print(f'   observed by the keyframes plus every frame read at {hz:g} Hz (every {step}th, {len(light)} frames): {100 * seen_l.mean():.2f} %;  '
          f'of the points the frames observed {100 * (seen_l & seen_all).sum() / max(seen_all.sum(), 1):.2f} %')
gap = seen_all & ~seen_kf
n = max(int(gap.sum()), 1)
print(f'   the points observed only between keyframes ({int(gap.sum())}), over the keyframes: never inside a keyframe\'s image '
      f'{100 * (gap & ~in_img).sum() / n:.1f} %; inside but every such pixel '
      f'invalid {100 * (gap & in_img & invalid & ~dyn_px & ~other).sum() / n:.1f} %, '
      f'a person {100 * (gap & in_img & dyn_px & ~other).sum() / n:.1f} %, '
      f'something else measured there (occluded or misregistered) {100 * (gap & in_img & other).sum() / n:.1f} %')
print('KEYFRAME_COVERAGE_DONE')
