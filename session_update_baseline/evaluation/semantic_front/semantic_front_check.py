"""A semantic object front (user 11-03: an instance is semantics plus a number): per frame, every object class's semantic
region (the real data's ADE ids, as its segmentation network gave them) split into connected pieces, tracked into
physical ids the way Khronos does it (the GaME session's update_layer/front.py at e4a02cf, steps 1-3: InstanceForwarding,
MaxIoUTracker over 0.2 m voxels in a 3 s window, re-identification by the voxels an id was ever seen in; its SAM step 4
off), real A -> B -> C through one front (B and C in A's frame). The reviewed physical ids are only read to check: per
reviewed object, how its pixels spread over front ids (in all and per visit), and per front id how many reviewed objects
it holds. Every STEP-th frame, every 2nd pixel (the runner's real input). No map is built.
Writes OUT/FRONT_CHECK.txt, OUT/contingency.json.
usage: semantic_front_check.py OUT [STEP]"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import yaml
from scipy import ndimage
sys.path.insert(0, '/home/jixian/Desktop/FT/wt_takeover_runs/aria/front_e4a02cf')
from update_layer.frames import FlatSession, real_session
from update_layer.front import FrontConfig, KhronosObjectFront
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True); STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 3
M = Path('/home/jixian/Desktop/FT/datasets/local_ab/mapping_configs')
OBJECTS = yaml.safe_load((M / 'nss_ade20k_room_label_space.yaml').read_text())['object_labels']
CATALOG = {e['physical_instance_id']: e for e in json.loads((M / 'room18_physical_catalog.json').read_text())['entities']}
PIXEL_STEP = 2
front = KhronosObjectFront(FrontConfig(OBJECTS, max_range=5.0, min_cluster_size=50 // PIXEL_STEP ** 2))


def segments(sem):
    """The frame's segments: each object class's region split into 8-connected pieces, numbered from 1."""
    out = np.zeros(sem.shape, np.int32); n = 0
    for c in OBJECTS:
        lab, k = ndimage.label(sem == c, structure=np.ones((3, 3), bool))
        if k:
            out[lab > 0] = lab[lab > 0] + n; n += k
    return out


pairs = defaultdict(int)            # (visit, reviewed id, front id) -> pixels of an object class
t0 = time.time()
for s in 'abc':
    session = FlatSession(real_session(s), pixel_step=PIXEL_STEP)
    front.begin_session()
    for i in range(0, len(session.ids), STEP):
        frame = session.load(i)
        reviewed = frame.instance.copy()
        frame.instance = segments(frame.semantic)
        ids = front.process(frame)
        sel = (reviewed > 0) | (ids > 0)
        r, f = reviewed[sel], ids[sel]
        key, cnt = np.unique(np.stack([r, f], 1), axis=0, return_counts=True)
        for (a, b), c in zip(key.tolist(), cnt.tolist()):
            pairs[(s, a, b)] += c
        if i % 300 == 0:
            print(f'{s} {i}/{len(session.ids)}: {len(front.tracks)} front ids, {time.time() - t0:.0f} s', flush=True)
(OUT / 'contingency.json').write_text(json.dumps([[s, a, b, c] for (s, a, b), c in sorted(pairs.items())]))

lines = [f'front: {front.log}; {len(front.tracks)} front ids over A, B, C; every {STEP}th frame, every {PIXEL_STEP}nd pixel', '']
reviewed_ids = sorted({a for (_, a, _) in pairs if a > 0})
lines.append('per reviewed object (pixels where it is, over front ids; 0 = no front id): main front id share, front ids with >= 5 %, '
             'main front id per visit')
for a in reviewed_ids:
    tot = defaultdict(int); per = defaultdict(lambda: defaultdict(int))
    for (s, x, b), c in pairs.items():
        if x == a: tot[b] += c; per[s][b] += c
    n = sum(tot.values()); main = max(tot, key=tot.get)
    big = [b for b, c in sorted(tot.items(), key=lambda x: -x[1]) if c >= 0.05 * n]
    visits = {s: max(d, key=d.get) for s, d in per.items()}
    name = CATALOG.get(a, {}).get('name', '?'); cls = CATALOG.get(a, {}).get('semantic_id', '?')
    lines.append(f'  I{a:<3d} {name[:22]:22s} class {cls:>3}: {n:8d} px, main front id {main} {100 * tot[main] / n:5.1f} %, '
                 f'ids >= 5 %: {big}, per visit {visits}')
lines.append('')
lines.append('per front id holding >= 5 % of 2+ reviewed objects (merges):')
for b in sorted({b for (_, _, b) in pairs if b > 0}):
    tot = defaultdict(int)
    for (s, a, x), c in pairs.items():
        if x == b: tot[a] += c
    n = sum(tot.values()); held = [a for a, c in tot.items() if a > 0 and c >= 0.05 * n]
    if len(held) >= 2:
        lines.append(f'  front {b}: {n} px, reviewed objects {sorted(held)}, background share {100 * tot.get(0, 0) / n:.0f} %')
(OUT / 'FRONT_CHECK.txt').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))
