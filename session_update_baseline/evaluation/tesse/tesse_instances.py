"""Ground-truth instance labels for the extracted TESSE frames (office, the default, or apartment), from Khronos's ground-truth scene graph.

The bag has no instance image (its segmentation holds classes). The ground-truth graph holds every object with its mesh
(box-local points), box and observed interval. A pixel of an object class gets the instance of the ground-truth object
that is present at the frame's time, has the pixel's class and whose box, grown by one map voxel, holds the pixel's back
projected point; if several do, the one with the nearest mesh point. Humans (dynamic class) get none. Instance id =
object index + 1. Also writes the physical catalog of those objects.
usage: tesse_instances.py DATA_DIR CATALOG.json [SCENE]
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path('/home/jixian/Desktop/FT')
SCENE = sys.argv[3] if len(sys.argv) > 3 else 'office'
GT = ROOT / f'datasets/khronos/tesse_cd/{SCENE}/ground_truth/gt_dsg_consolidated.json'
LABELS = {'office': 19, 'apartment': 21}[SCENE]  # total_semantic_labels of Khronos's tesse_cd_SCENE label space
VOXEL = 0.05
FOREVER = 18446744073709551615
data, catalog_path = Path(sys.argv[1]), Path(sys.argv[2])
rgbd, sem = data / 'rgbd', data / 'semantics'
inst = data / 'instance_labels'; inst.mkdir(exist_ok=False)
session = json.loads((data / 'session.json').read_text()); t0 = session['start_ns']


def quat(q):
    x, y, z, w = q['x'], q['y'], q['z'], q['w']
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


graph = json.load(open(GT))
objects = []
for n in (n for n in graph['nodes'] if n.get('layer') == 2):
    a = n['attributes']; label = a['semantic_label']
    if label >= 2**31: continue                      # unknown class (the dynamic humans)
    bb = a['bounding_box']; R = quat(bb['world_R_center']); c = np.array(bb['world_P_center']); half = np.array(bb['dimensions']) / 2
    pts = np.array(a['mesh'].get('points', []), dtype=np.float64).reshape(-1, 3)
    objects.append(dict(name=a['name'], label=int(label), R=R, c=c, half=half, world=pts @ R.T + c,
                        first=a['first_observed_ns'][0], last=a['last_observed_ns'][0]))
classes = sorted({o['label'] for o in objects})
print(len(objects), 'objects with a class; classes', classes)
trees = [cKDTree(o['world']) if len(o['world']) else None for o in objects]
K = dict(l.split(':') for l in (rgbd / 'Intrinsics.txt').read_text().splitlines())
fx, fy, cx, cy = (float(K[k]) for k in ('f_x', 'f_y', 'u', 'v'))
stats = dict(frames=0, object_pixels=0, labelled=0)
rows = list(csv.DictReader(open(rgbd / 'timestamps.csv')))
for row in rows:
    i, t = row['ImageID'], t0 + int(row['TimeStamp'])
    D = cv2.imread(str(rgbd / f'{i}_depth.tiff'), cv2.IMREAD_UNCHANGED).astype(np.float64)
    S = cv2.imread(str(sem / f'{i}_segmentation.png'), cv2.IMREAD_UNCHANGED)
    T = np.loadtxt(rgbd / f'{i}_pose.txt')
    out = np.zeros(S.shape, np.uint16)
    v, u = np.nonzero(np.isin(S, classes) & (D > 0) & np.isfinite(D))
    if len(u):
        z = D[v, u]; P = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], 1) @ T[:3, :3].T + T[:3, 3]
        best = np.full(len(P), np.inf); who = np.zeros(len(P), np.int64)
        for k, o in enumerate(objects):
            if not (o['first'] <= t <= (o['last'] if o['last'] != FOREVER else t)): continue
            m = S[v, u] == o['label']
            if not m.any(): continue
            local = (P[m] - o['c']) @ o['R']
            inside = np.all(np.abs(local) <= o['half'] + VOXEL, axis=1)
            if not inside.any(): continue
            idx = np.nonzero(m)[0][inside]
            d = trees[k].query(P[idx])[0] if trees[k] is not None else np.zeros(len(idx))
            closer = d < best[idx]; best[idx[closer]] = d[closer]; who[idx[closer]] = k + 1
        out[v, u] = who
    cv2.imwrite(str(inst / f'{i}_segmentation.png'), out)
    stats['frames'] += 1; stats['object_pixels'] += int(len(u)); stats['labelled'] += int((out > 0).sum())
ents = [dict(physical_instance_id=k + 1, semantic_id=o['label'], name=o['name']) for k, o in enumerate(objects)]
catalog_path.write_text(json.dumps(dict(schema='session_update_physical_catalog/v1', semantic_id_range=[0, LABELS - 1], entities=ents), indent=1) + '\n')
stats['labelled_share'] = round(stats['labelled'] / max(1, stats['object_pixels']), 4)
(data / 'instances.json').write_text(json.dumps(stats, indent=1) + '\n')
print(json.dumps(stats))
