"""Which tracked objects does D1 see move, by their own observed centroids? (CPU, reads d1_trace.py output)
原理思想：物体层的“移动”判据。Khronos 的 D1 用 computeDynamicDisplacement（运动簇质心离第一次的最大距离）与
min_dynamic_displacement（1.0 m）判定一次移动；静态类别物体的运动簇被 promoteDynamicSemanticClusters 删掉，所以合成 B 里
花盆、椅子、桌子被搬动时从未触发。这里对同一批按实例跟踪的物理 track，用它每帧的观测质心（像素均值 computeCentroid，
即 dynamic_centroids 用的同一个量，trace 里的 native_centroid）算同一个位移，看 1.0 m 的门能否认出 GT 里被搬动的物体，同时不把真实的静止物体判成移动。稳健版本：最早 10 次与最后 10 次观测的
质心中位数之差（局部遮挡、只看见一部分时质心会跳）。来源：d1.py displacement()/_episode()（Khronos MeshObjectExtractor 的
computeDynamicDisplacement、min_dynamic_displacement 原值 1.0）；GT 只用于核对（synthetic_ab/GT_ANIMATION_INTERVALS.csv）。
usage: trace_moves.py TRACE_JSONL [synthetic|real] [SESSION]
"""
import csv
import json
import sys
from collections import defaultdict

import numpy as np

path, dataset = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'synthetic'
session = sys.argv[3] if len(sys.argv) > 3 else 'b'
FIRST_DYNAMIC = 1 << 16                        # d1.py D1.FIRST_DYNAMIC_TRACK_ID: ids below are physical (instance) tracks
MIN_DISP = 1.0                                 # d1.py D1Config.min_dynamic_displacement (Khronos L2_FINAL2 config)
obs = defaultdict(list)
for line in open(path):
    rec = json.loads(line)
    for t in rec['tracks']:
        if t['id'] < FIRST_DYNAMIC and not t['dynamic']:
            obs[t['id']].append((rec['frame'], np.asarray(t['native_centroid']), t['points']))   # d1.py:283, the pixel mean
moved_gt = {}
if dataset == 'synthetic':
    for r in csv.DictReader(open('/home/jixian/Desktop/FT/datasets/synthetic_ab/GT_ANIMATION_INTERVALS.csv')):
        if r['session'] == session:
            moved_gt[int(r['instance_id'])] = (r['name'], r['kind'], float(r['displacement_m']), int(r['start_frame']), int(r['end_frame']))
rows = []
for tid, o in obs.items():
    if len(o) < 20:
        continue
    c = np.stack([x[1] for x in o])
    disp_max = float(np.max(np.linalg.norm(c - c[0], axis=1)))
    robust = float(np.linalg.norm(np.median(c[-10:], axis=0) - np.median(c[:10], axis=0)))
    rows.append((tid, len(o), o[0][0], o[-1][0], disp_max, robust, int(np.median([x[2] for x in o]))))
rows.sort(key=lambda r: -r[5])
print(f'{dataset} {session}: physical tracks observed >= 20 frames: {len(rows)}; gate = displacement >= {MIN_DISP} m')
print(f'{"id":>5s} {"obs":>5s} {"first":>6s} {"last":>6s} {"max-from-first":>14s} {"robust":>7s} {"pts":>6s}  GT')
for tid, n, f0, f1, dm, rb, pts in rows:
    gt = moved_gt.get(tid)
    flag = ('MOVE' if rb >= MIN_DISP else '    ') + (' max' if dm >= MIN_DISP else '    ')
    if rb >= 0.3 or dm >= MIN_DISP or gt:
        print(f'{tid:5d} {n:5d} {f0:6d} {f1:6d} {dm:14.2f} {rb:7.2f} {pts:6d}  {flag}  {gt if gt else ""}')
robust_moves = {r[0] for r in rows if r[5] >= MIN_DISP}
max_moves = {r[0] for r in rows if r[4] >= MIN_DISP}
print(f'tracks over the gate: robust {len(robust_moves)}, max-from-first {len(max_moves)}')
if dataset == 'synthetic':
    real_moves = {k for k, v in moved_gt.items() if v[1] == 'motion' and v[2] >= MIN_DISP}
    seen = {r[0] for r in rows}
    print(f'GT motions >= {MIN_DISP} m: {sorted(real_moves)}; observed as tracks: {sorted(real_moves & seen)}')
    for name, s in (('robust', robust_moves), ('max-from-first', max_moves)):
        print(f'  {name}: found {sorted(s & real_moves)} | missed {sorted((real_moves & seen) - s)} | false (no GT motion >= 1 m) {sorted(s - real_moves)}')
print('TRACE_MOVES_DONE')
