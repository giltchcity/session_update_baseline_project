#!/usr/bin/env python3
"""One object timeline of a visit chain for the official evaluation (eval/chain_fill/timeline_to_4dmap), from the
visits' timelines in the timeline_export format. It writes what each method's own snapshots say, and judges no change
for a method (the user, 10-29: official evaluator unchanged; a non-native method's output converted as it is):
  removal: an object listed before and no longer listed while its last listing still had it present (a timeline
    that drops what its map no longer holds) is kept, closed at the first snapshot without it (last observed = its
    stamp), in every later snapshot; listed again later, it is first observed then. An object the method closed
    itself before it stopped listing it, and a method that keeps a removed object's node, absent, are left as they are;
  moves: the method's own count of its objects' relocations (details session_mobility_changes): when it rises, the
    object's earlier record is closed at the method's time of the move (details observation_first_stamp_ns, else the
    snapshot's stamp) and kept in every later snapshot (node 'h' n, as chain_fill), and the object is first observed
    then (at its new place). A method that counts none has no moves;
  appearance: the method's own first_observed_ns, as it gives it (0: present from the start);
  surface: an object listed without surface points (taken away; its node kept) gets the last surface, with its box,
    of the same physical object (details instance_id, else its id) in an earlier snapshot, so the evaluator can
    associate it.
Until 10-29 it also judged moves from the surfaces (10 cm / 50 %, also re-checking a declared move) and gave an
object first listed in a later visit an appearance at that snapshot: both left out (regression_20261029/).
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

MAX = 2000
BOX = ('bbox_center', 'bbox_dimensions', 'bbox_min', 'bbox_max')


def sample(points):
    p = np.asarray(points, float)
    return p[:: max(1, (len(p) + MAX - 1) // MAX)]


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
last, moved_at, places, listed, gone, listed_now = {}, {}, [], {}, {}, set()
counts, listing = {}, {}                               # listing: the last listed record of each object
filled = 0
snaps = [[json.loads(line) for line in open(path)] for path in visits]
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
                again = k in gone
                if again:                                 # listed again after it was dropped: first observed now
                    gone.pop(k); moved_at[k] = t
                count = detail(o, 'session_mobility_changes', 0)       # the method's own relocations
                if k in counts and count > counts[k] and k in last and not again:
                    close(last[k], detail(o, 'observation_first_stamp_ns', t), k)
                counts[k] = count
                if k in moved_at and o['first_observed_ns'] and o['first_observed_ns'][0] < moved_at[k]:
                    o['first_observed_ns'][0] = moved_at[k]
                if o['surface']:
                    last[k] = copy.deepcopy(o)
                elif k in last:
                    o['surface'] = last[k]['surface']; filled += 1
                    for b in BOX:
                        if b in last[k]: o[b] = last[k][b]
                listing[k] = o
            snap['objects'] += places
            out.write(json.dumps(snap) + '\n')
print(f'TIMELINE_COMPLETE visits={len(visits)} closed={len(places)} filled={filled} objects={len(last)}')
