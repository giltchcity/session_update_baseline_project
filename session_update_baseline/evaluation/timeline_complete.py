#!/usr/bin/env python3
"""One object timeline of a 3RScan visit chain for the official evaluation (eval/chain_fill/timeline_to_4dmap), from
the visits' timelines in the timeline_export format, completed by one set of rules for every method, from what the
earlier snapshots hold (for a 4D map's timeline, what eval/chain_fill does with the map itself):
  surface: an object listed without surface points (taken away; its node kept) gets the last surface, with its box,
    of the same physical object (details instance_id, else its id) in an earlier snapshot, so the evaluator can
    associate it;
  moves: a method that counts its objects' relocations (details session_mobility_changes, 0 when absent) declares
    them: when the count rises, the object's last surface is held, and at its next surface, if that surface no longer
    represents the held one (under 50 % of the held points within 0.10 m of it, the criterion of the ground truth's
    states) the object moved, at its first observation at the new place (details observation_first_stamp_ns, else
    the snapshot's stamp); otherwise it was re-observed in place. A method that counts none has every surface
    compared with the last earlier one: it moved, at this snapshot, when neither represents the other (a surface
    that grows or is re-observed in part at the same place represents, or is represented by, the earlier one).
    A move keeps the earlier record, closed then (last observed = the move), in every later snapshot (node 'h' n,
    as chain_fill), and the object is first observed then (it appeared at its new place);
  appearance: an object first present in a later visit than the first gets, while its first observed time is 0,
    that snapshot's stamp (objects of the first visit keep 0, present from the start as the ground truth's);
  removal: an object listed before and no longer listed while its last listing still had it present (a timeline
    that drops what its map no longer holds) is kept, closed at the first snapshot without it (last observed = its
    stamp), in every later snapshot; listed again later, it is first observed then (it appeared again). An object
    the method closed itself before it stopped listing it, and a method that keeps a removed object's node, absent,
    are left as they are.
Mesh: the official evaluator times an object first observed at 0 by the global mesh around it (when it was first
seen); a snapshot line that names its mesh (timeline_export: its map snapshot) keeps it, else it names a mesh file
(uint64 count, per vertex float32 x y z and uint64 first-seen ns): VISIT_K.jsonl's directory/mesh.bin when the method
exported one, else one made of the surface points of every object listed so far, each first seen when its object was
first listed (OUT.jsonl's directory/mesh_K.bin).
usage: timeline_complete.py OUT.jsonl VISIT_0.jsonl [VISIT_1.jsonl ...]   (prints the counts)
"""
import copy, json, struct, sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

GATE, FRAC, MAX = 0.10, 0.5, 2000
BOX = ('bbox_center', 'bbox_dimensions', 'bbox_min', 'bbox_max')


def sample(points):
    p = np.asarray(points, float)
    return p[:: max(1, (len(p) + MAX - 1) // MAX)]


def represented(a, b):
    return float((cKDTree(b).query(a)[0] <= GATE).mean()) if len(a) and len(b) else 0.0


def write_mesh(path, points, stamps):
    order = np.argsort(stamps, kind='stable')
    rows = np.zeros(len(order), dtype=[('p', '<f4', 3), ('t', '<u8')])
    rows['p'] = np.asarray(points, np.float32)[order]; rows['t'] = np.asarray(stamps, np.uint64)[order]
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(rows))); f.write(rows.tobytes())


def key(o):
    iid = (o.get('details') or {}).get('instance_id') or []
    return int(iid[0]) if iid else o['id']


def detail(o, name, fallback):
    v = (o.get('details') or {}).get(name) or []
    return v[0] if v else fallback


def last_before(times, t):              # khronos lastAppearedBefore / lastDisappearedBefore
    found = None
    for x in times:
        if x > t:
            break
        found = x
    return found


def is_present(o, t):                   # khronos::isPresent
    a = last_before(o['first_observed_ns'], t)
    if a is None:
        return False
    d = last_before(o['last_observed_ns'], t)
    return d is None or a > d


out_path, visits = sys.argv[1], sys.argv[2:]
last, first_present, moved_at, places, listed, gone, listed_now = {}, {}, {}, [], {}, {}, set()
counts, held, listing = {}, {}, {}                     # listing: the last listed record of each object
filled = appeared = 0
snaps = [[json.loads(line) for line in open(path)] for path in visits]
declared = any('session_mobility_changes' in (o.get('details') or {}) for s in snaps for x in s for o in x['objects'])
meshes = []
for v, path in enumerate(visits):                     # the global mesh of each visit's snapshots without their own
    own = Path(path).parent / 'mesh.bin'
    if all('mesh' in snap for snap in snaps[v]) or own.exists():
        meshes.append(str(own)); continue
    for snap in snaps[v]:
        for o in snap['objects']:
            if o['surface'] and key(o) not in listed: listed[key(o)] = (sample(o['surface']), snap['stamp_ns'])
    made = Path(out_path).parent / f'mesh_{v}.bin'
    write_mesh(made, np.concatenate([p for p, _ in listed.values()]) if listed else np.zeros((0, 3)),
               np.concatenate([[t] * len(p) for p, t in listed.values()]) if listed else np.zeros(0))
    meshes.append(str(made))


def close(record, at, k):
    place = copy.deepcopy(record); place['last_observed_ns'] = [at]
    place['id'] = str((ord('h') << 56) | len(places))       # spark_dsg NodeSymbol('h', n)
    places.append(place); moved_at[k] = at


with open(out_path, 'w') as out:
    for v, path in enumerate(visits):
        for snap in snaps[v]:
            t = snap['stamp_ns']; snap.setdefault('mesh', meshes[v])
            here = {key(o) for o in snap['objects']}
            for k in [k for k in listed_now if k not in here and k not in gone and k in last]:
                if not is_present(listing[k], t):        # closed by the method itself
                    continue
                place = copy.deepcopy(last[k]); place['last_observed_ns'] = [t]
                place['id'] = str((ord('h') << 56) | len(places))
                places.append(place); gone[k] = t
            listed_now = here
            for o in snap['objects']:
                k = key(o)
                if k in gone:                             # listed again after it was dropped: it appeared again
                    gone.pop(k); moved_at[k] = t
                if declared:
                    count = detail(o, 'session_mobility_changes', 0)
                    if k in counts and count > counts[k] and k in last and k not in held:
                        held[k] = copy.deepcopy(last[k])
                    counts[k] = count
                    if k in held and o['surface']:
                        if represented(sample(held[k]['surface']), sample(o['surface'])) < FRAC:
                            close(held[k], detail(o, 'observation_first_stamp_ns', t), k)
                        del held[k]
                elif o['surface'] and k in last:
                    a, b = sample(last[k]['surface']), sample(o['surface'])
                    if represented(a, b) < FRAC and represented(b, a) < FRAC:
                        close(last[k], t, k)
                if k in moved_at and o['first_observed_ns'] and o['first_observed_ns'][0] < moved_at[k]:
                    o['first_observed_ns'][0] = moved_at[k]
                if o['surface']:
                    last[k] = copy.deepcopy(o)
                elif k in last:
                    o['surface'] = last[k]['surface']; filled += 1
                    for b in BOX:
                        if b in last[k]: o[b] = last[k][b]
                if is_present(o, t) and k not in first_present:
                    first_present[k] = (v, t)
                if k in first_present and first_present[k][0] > 0 and o['first_observed_ns'] and o['first_observed_ns'][0] == 0:
                    o['first_observed_ns'][0] = first_present[k][1]; appeared += 1
                listing[k] = o
            snap['objects'] += places
            out.write(json.dumps(snap) + '\n')
print(f'TIMELINE_COMPLETE visits={len(visits)} moves={len(places)} filled={filled} appeared={appeared} objects={len(last)}'
      f' relocations={"declared" if declared else "from surfaces"}')
