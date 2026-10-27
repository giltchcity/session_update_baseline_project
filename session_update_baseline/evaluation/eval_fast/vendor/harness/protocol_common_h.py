"""Harness copy of tools/abc_eval_protocol_v1/protocol_common.py, one session at a time.

Origin (unchanged, read-only):
  tools/abc_eval_protocol_v1/protocol_common.py            (2026-09-09 20:35)
  tools/abc_eval_final_baseline/temporal_core.py:load_versions (2026-09-09 09:06)

Differences from the originals are path/scope only:
  * The original created results/abc_eval_protocol_v1 at import; nothing is created here.
  * Snapshots come from an explicit snapshots.jsonl of ONE session (written by
    tools/abc_eval_final_baseline/export_timeline for the map under test) instead of
    results/abc_eval_final_{method}/predictions/{a,b,c}/snapshots.jsonl for all sessions.
    Loading one session is equivalent for every consumer in this harness: the only
    cross-session state in load_versions is the centroid cache, which only fills
    centroids of zero-vertex nodes, and those are never `_current`.
  * GT, events, evidence and the session clock still come from
    results/abc_eval_final_baseline (dataset-level, method independent).
Presence flags, association gates, surface representation (0.10 m / 50 %) and the
matching code are copied verbatim.
"""
from pathlib import Path
import csv
import json

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path('/home/jixian/Desktop/FT')
RESULTS = ROOT / 'results'
BASELINE = RESULTS / 'abc_eval_final_baseline'


def method_dir(method):
    return RESULTS / f'abc_eval_final_{method}'


class SessionClock:
    """Session constants exactly as in temporal_core.py (LENGTH, BASE, OFFSET)."""

    def __init__(self):
        c = json.loads((BASELINE / 'input_manifest.json').read_text())
        self.LENGTH = {s: c['sessions'][s]['last_timestamp'] / 1e9 for s in 'ABC'}
        self.BASE = {s: c['sessions'][s]['timestamp_provenance']['session_start_ns'] for s in 'ABC'}
        self.OFFSET = {'A': 0., 'B': self.LENGTH['A'], 'C': self.LENGTH['A'] + self.LENGTH['B']}
        self.TOTAL = sum(self.LENGTH.values())


def load_versions_session(snapshots_path, session):
    """temporal_core.load_versions() restricted to one session and one explicit file."""
    mod = SessionClock()
    s = session
    result = []
    cache = {}
    raw = []
    for line in Path(snapshots_path).open():
        r = json.loads(line)
        relative = (r['stamp_ns'] - mod.BASE[s]) / 1e9
        for p in r['objects']:
            p.pop('surface', None)
            p.pop('details', None)
            if p['centroid'] is not None:
                cache[p['id']] = (p['centroid'], r['stamp_ns'])
                p['geometry_source_ns'] = r['stamp_ns']
                p['historical_geometry'] = False
            elif p['id'] in cache:
                p['centroid'], p['geometry_source_ns'] = cache[p['id']]
                p['historical_geometry'] = True
            else:
                p['geometry_source_ns'] = None
                p['historical_geometry'] = False
        r.update(session=s, time_s=relative, active_time_s=mod.OFFSET[s] + max(0., relative),
                 version=f'{s}_{r["index"]}')
        raw.append(r)
    if s == 'A':
        result.append(dict(session=s, time_s=0., active_time_s=0., stamp_ns=mod.BASE[s], index=-1,
                           version='A_initial_empty', objects=[]))
    elif not any(r['time_s'] <= 0 for r in raw):
        raise ValueError('Missing inherited baseline ' + s +
                         ' (no snapshot at or before session start: the map has no seed/prior snapshot)')
    result.extend(raw)
    for j, r in enumerate(result):
        r['robot_interval_end_s'] = result[j + 1]['active_time_s'] if j + 1 < len(result) else mod.TOTAL
    return mod, result


def load_gt():
    base = method_dir('baseline')
    states = json.loads((base / 'gt_objects/states.json').read_text())
    finals = {s: json.loads((base / f'gt_objects/GT_{s}_final.json').read_text()) for s in 'ABC'}
    events = json.loads((base / 'events.json').read_text())
    evidence = json.loads((base / 'evidence.json').read_text())
    manifest = json.loads((base / 'input_manifest.json').read_text())
    return dict(states=states, finals=finals, events=events, evidence=evidence, manifest=manifest,
                by_id={x['id']: x for x in states})


def load_snapshots_session(snapshots_path, session):
    mod, versions = load_versions_session(snapshots_path, session)
    # attach a clean boolean presence based on the exported native field, not the buggy flags()
    for v in versions:
        for p in v['objects']:
            p['_present'] = bool(p.get('present'))
            p['_current'] = p['_present'] and p.get('mesh_vertices', 0) > 0 and p.get('centroid') is not None
    return mod, versions


def version_at(versions, session, time_s):
    cand = [v for v in versions if v['session'] == session and v['time_s'] <= time_s + 1e-9]
    if not cand:
        return None
    return max(cand, key=lambda v: v['time_s'])


def object_centroid(p):
    c = p.get('centroid')
    return None if c is None else np.asarray(c, float)


def associate_state(version, state, gate=0.5, require_current=True):
    """Return current map object nodes matching a GT state by semantic+centroid."""
    if version is None:
        return []
    out = []
    sc = np.asarray(state['centroid'], float)
    for p in version['objects']:
        if p.get('semantic') != state['semantic']:
            continue
        if require_current and not p.get('_current'):
            continue
        c = object_centroid(p)
        if c is None:
            continue
        d = float(np.linalg.norm(c - sc))
        if d <= gate:
            out.append((p, d))
    return sorted(out, key=lambda x: x[1])


def state_represented(version, state, gate=0.5, require_current=True):
    return bool(associate_state(version, state, gate=gate, require_current=require_current))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def metric(tp, fp, fn):
    return dict(TP=int(tp), FP=int(fp), FN=int(fn),
                Precision=tp / (tp + fp) if tp + fp else None,
                Recall=tp / (tp + fn) if tp + fn else None,
                F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)


def load_raw_surfaces_path(snapshots_path):
    """Return {stamp_ns: {object_id: surface sample}} from one raw snapshots.jsonl."""
    out = {}
    for line in Path(snapshots_path).open():
        r = json.loads(line)
        objs = {}
        for p in r.get('objects', []):
            surf = p.get('surface')
            if surf:
                objs[p['id']] = dict(surface=np.asarray(surf, float), semantic=p.get('semantic'),
                                     present=bool(p.get('present')), mesh_vertices=int(p.get('mesh_vertices', 0)))
        out[int(r['stamp_ns'])] = objs
    return out


def surface_represented(mod, raw_by_stamp, version, state, gate=0.10, frac=0.5):
    """Whether a GT state surface is explained by the current object surfaces."""
    if version is None:
        return False
    raw = raw_by_stamp.get(int(version.get('stamp_ns', -1)), {})
    if not raw:
        # empty initial map
        return False
    gt_pts = np.load(state['surface'])
    best = 0.0
    for oid, info in raw.items():
        if info.get('semantic') != state.get('semantic'):
            continue
        if not info.get('present') or info.get('mesh_vertices', 0) <= 0:
            continue
        surf = info.get('surface')
        if surf is None or len(surf) == 0:
            continue
        tree = cKDTree(surf)
        d = tree.query(gt_pts)[0]
        f = float((d <= gate).mean())
        if f > best:
            best = f
        if best >= frac:
            return True
    return best >= frac
