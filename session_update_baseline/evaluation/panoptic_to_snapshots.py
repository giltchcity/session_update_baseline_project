#!/usr/bin/env python3
"""Turn a panoptic_export.py output dir (objects.json + submaps/*.ply) into one snapshots.jsonl record so threerscan_eval.py
can score a Panoptic Multi-TSDF map with the same protocol. Physical id = the '3RScan objectId N' in the submap's stored
label name; present = change state not Absent; surface = up to 1500 mesh vertices.
usage: panoptic_to_snapshots.py EXPORT_DIR OUT_DIR"""
import json, re, sys, numpy as np, open3d as o3d
from pathlib import Path
E, O = Path(sys.argv[1]), Path(sys.argv[2]); O.mkdir(parents=True, exist_ok=True)
m = json.load((E / 'objects.json').open()); objs = []
for r in m['submaps']:
    mm = re.search(r'objectId (\d+)', r.get('name', '') or ''); pid = int(mm.group(1)) if mm else None
    ply = next(iter(E.glob('submaps/*_%s.ply' % r['frame_name'])), None)
    if not ply: continue
    v = np.asarray(o3d.io.read_triangle_mesh(str(ply)).vertices)
    if len(v) == 0: continue
    if len(v) > 1500: v = v[np.random.default_rng(0).choice(len(v), 1500, replace=False)]
    objs.append({'id': str(r['id']), 'semantic': r.get('class_id', -1), 'present': r.get('change_state') != 'Absent', 'mesh_vertices': int(len(v)),
                 'surface': v.round(4).tolist(), 'details': {'instance_id': [pid] if pid else []}, 'panoptic_state': r.get('change_state'), 'name': r.get('name')})
(O / 'snapshots.jsonl').write_text(json.dumps({'index': 0, 'stamp_ns': 200000000000, 'objects': objs}) + '\n')
print('wrote', len(objs), 'submaps ->', O / 'snapshots.jsonl'); print([(o['details']['instance_id'], o['panoptic_state'], o['name'][:22]) for o in objs])
