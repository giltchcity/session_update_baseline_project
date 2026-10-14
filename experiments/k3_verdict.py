"""Read the K3 offline counts and show the raw frames of every K3 trigger (CPU).
原理思想：K3 进链前的三条（监工 09:21）：多出的关键帧少到能过速度门；合成 B 的花盆停下（3456-3465）被取到；真实 B/C 上不被
传感器噪声触发。前两条直接数；第三条要先看原图再下结论（用户 10-06：先看原始 RGB 和深度，不加叠加图）：每个触发帧和它的
参照关键帧并排（参照 RGB | 本帧 RGB | 参照深度 | 本帧深度），人工判断是场景真的变了（人搬了东西、行李箱）还是噪声。参照帧按
k3_offline.py 的同一逻辑重放得到（GaME 的关键帧判据 + K3 触发帧都会成为新参照）。来源：k3_offline.py（c13a917 起）的输出。
usage: k3_verdict.py RUN_DIR DATASET SESSION OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from update_layer.run import dataset_config, INPUT_STEP                                   # noqa: E402
from update_layer.frames import FlatSession                                               # noqa: E402

RUN, DATASET, SESSION, OUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in open(RUN / 'k3.jsonl')]
summary = next((r for r in rows if r.get('summary')), None)
cmp_ = [r for r in rows if not r.get('summary')]
k3 = [r for r in cmp_ if r['keyframe']]
print(f'{DATASET} {SESSION}: {summary}')
print(f'  unmoved frames compared {len(cmp_)}, K3 keyframes {len(k3)}')
print('  K3 frames (added masks / region px):', ' '.join(f"{r['frame']}:{r['added_masks']}/{r['region_px']}" for r in k3))
if DATASET == 'synthetic' and SESSION == 'b':
    win = [r for r in cmp_ if 3440 <= r['frame'] <= 3554]
    print('  3440-3554:', ' '.join(f"{r['frame']}:{r['added_masks']}/{r['region_px']}{'*' if r['keyframe'] else ''}" for r in win))
# reference of each K3 frame: replay of k3_offline.py's keyframe logic on the poses (GaME is_keyframe C1/C2, flat.yaml values)
cfg, info, specs = dataset_config(DATASET)
spec = next(sp for sp in specs if sp.name == f'{DATASET}_{SESSION}')
front = FlatSession(spec, pixel_step=INPUT_STEP[DATASET])
held = {n for n in range(len(front.ids)) if n % 10 == 0 and n != 0} if spec is specs[-1] else set()
k3_frames = {r['frame'] for r in k3}
KF_T_M = 0.125                                         # flat.yaml keyframe_translation_diff 1.25 / scale 10
ref_of, ref, ref_T = {}, None, None
for k in range(len(front.ids)):
    if k in held:
        continue
    T = front.pose(k)
    if ref_T is None:
        ref, ref_T = k, T
        continue
    d = np.linalg.inv(ref_T) @ T
    moved = np.linalg.norm(d[:3, 3]) > KF_T_M or np.any(np.abs(Rotation.from_matrix(d[:3, :3]).as_euler('xyz', degrees=True)) > 50)
    if moved:
        ref, ref_T = k, T
    elif k in k3_frames:
        ref_of[k] = ref
        ref, ref_T = k, T
full = FlatSession(spec, pixel_step=1)


def depth_img(d, hi):
    v = np.nan_to_num(d, nan=0.0)
    return Image.fromarray((np.clip(v / hi, 0, 1) * 255).astype(np.uint8)).convert('RGB')


if DATASET != 'synthetic' or SESSION != 'b':
    for r in k3[:40]:
        i, j = ref_of.get(r['frame']), r['frame']
        if i is None:
            continue
        a, b = full.load(i, color=True), full.load(j, color=True)
        hi = float(np.nanpercentile(np.concatenate([a.depth.ravel(), b.depth.ravel()]), 99))
        pics = [Image.fromarray(a.color), Image.fromarray(b.color), depth_img(a.depth, hi), depth_img(b.depth, hi)]
        w, h = pics[0].size
        s = 480 / w
        pics = [p.resize((int(w * s), int(h * s))) for p in pics]
        sheet = Image.new('RGB', (pics[0].width * 4, pics[0].height + 22), 'white')
        draw = ImageDraw.Draw(sheet)
        for c, (lab, p) in enumerate(zip((f'reference {i} RGB', f'frame {j} RGB', f'reference {i} depth', f'frame {j} depth'), pics)):
            sheet.paste(p, (c * pics[0].width, 22))
            draw.text((c * pics[0].width + 4, 4), lab, fill=(0, 0, 0))
        sheet.save(OUT / f'k3_{DATASET}_{SESSION}_{j:05d}_ref{i:05d}.jpg', quality=85)
    print(f'  sheets: {OUT}')
print('K3_VERDICT_DONE')
