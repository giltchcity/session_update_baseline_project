"""Flat change F1 of a 4D map (ours, Khronos) by runs/update_layer_game_T1_20261006/analysis/wm_flat_eval_20261028/
flat_change_eval.py, the script the GaME and wavemap rows are scored with: its changes, its ground-truth states (read from
that directory's gt_states_{a,b}.npz, never rewritten here), its represented() and its event loop, unchanged. Only the
input differs: session B's first and last snapshot come from the map's eval/chain_fill/timeline_export lines (every
surface point, world frame) instead of a timeline.pkl. A listed move counts only beyond the 3RScan gate, as
eval/khronos4d_gt.py's displaced() (no poses: the ground-truth states' centroids more than 0.10 m apart; the user,
10-29): SM_Chair_b (0.08 m) and SM_Picture_c (0.00 m) are unchanged controls.
  usage: flat_change_maps.py OUT.json NAME=B_TIMELINE.jsonl ...
"""
import csv, json, sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

F = Path('/home/jixian/Desktop/FT/datasets/panoptic_mapping/flat_dataset')
GT = Path('/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/analysis/wm_flat_eval_20261028')
out_path, ARGS = Path(sys.argv[1]), sys.argv[2:]
lab = {int(r['InstanceID']): r['Name'] for r in csv.DictReader(open(F / 'groundtruth_labels.csv'))}
ids = {v: k for k, v in lab.items()}
changes, sec = {}, None
for l in open(F / 'changes.txt'):
    l = l.strip()
    if l.startswith('#'):
        sec = l[2:].split()[0].lower(); continue
    if l and l in ids:
        changes[ids[l]] = sec


def gt_states(stage):
    return np.load(GT / f'gt_states_{stage}.npz', allow_pickle=True)['states'].item()


def scene(line):
    return SimpleNamespace(objects=[SimpleNamespace(semantic=o['semantic'], present=o['present'],
                                                    points=np.asarray(o['surface'], dtype=np.float32).reshape(-1, 3))
                                    for o in line['objects']])


def represented(scene, state):
    if scene is None:
        return False, 0.0
    best = 0.0
    for o in scene.objects:
        if int(o.semantic) != int(state['semantic']) or not o.present or not len(o.points):
            continue
        f = float((cKDTree(o.points).query(state['points'], distance_upper_bound=0.10)[0] <= 0.10).mean())
        best = max(best, f)
        if best >= 0.5:
            return True, best
    return best >= 0.5, best


A, B = gt_states('a'), gt_states('b')
for k in [k for k, c in changes.items() if c == 'moved' and k in A and k in B]:
    if np.linalg.norm(A[k]['points'].astype(float).mean(0) - B[k]['points'].astype(float).mean(0)) <= 0.10:
        del changes[k]                                 # listed, but moved no more than the gate
print(f'GT states: A {len(A)}, B {len(B)}; changes evaluated {len(changes)} ({dict(Counter(changes.values()))})', flush=True)
summary = {}
for name, path in (a.split('=') for a in ARGS):
    lines = [json.loads(l) for l in open(path)]
    before, after = scene(lines[0]), scene(lines[-1])
    rows, tp, fp, fn = [], 0, 0, 0
    for k in sorted(set(A) | set(B)):
        kind = changes.get(k, 'unchanged' if k in A and k in B else None)
        if kind is None:
            continue
        comps = []
        if kind in ('moved', 'removed') and k in A:
            comps.append(('disappeared', A[k], True))
        if kind in ('moved', 'new') and k in B:
            comps.append(('appeared', B[k], True))
        if kind == 'unchanged':
            comps.append(('disappeared', A[k], False))
        for comp, state, expected in comps:
            rb, fb = represented(before, state); ra, fa = represented(after, state)
            pred = (rb and not ra) if comp == 'disappeared' else (not rb and ra)
            if expected:
                tp += pred; fn += not pred
            else:
                fp += pred
            if expected or pred:
                rows.append(dict(id=k, name=lab[k], kind=kind, component=comp, expected=expected, predicted=pred,
                                 before=round(fb, 3), after=round(fa, 3)))
    f1 = 200 * tp / (2 * tp + fp + fn) if tp + fp + fn else float('nan')
    summary[name] = dict(TP=tp, FP=fp, FN=fn, F1=round(f1, 2), before_stamp=lines[0]['stamp_ns'],
                         after_stamp=lines[-1]['stamp_ns'], events=rows)
    print(f'{name}: TP/FP/FN {tp}/{fp}/{fn}  change F1 {f1:.1f}', flush=True)
    for r in rows:
        if r['expected'] != r['predicted']:
            print(f"   {'FN' if r['expected'] else 'FP'} {r['name']} ({r['kind']}, {r['component']}): before {r['before']}, after {r['after']}")
out_path.write_text(json.dumps(summary, indent=1))
