"""Harness copy of tools/abc_eval_protocol_v1/cleanup_eval.py for ONE session (B or C).

Corrected old-state cleanup evaluation from full current-map probes (object layer +
global mesh).  classify_event() and GTGeometry are copied verbatim. Changes are
path/scope only:
  * the probe distances come from an explicit export_cleanup_binary output (.bin)
    instead of results/abc_eval_protocol_v1/{method}/cleanup/{b,c}.bin;
  * snapshot node tables come from an explicit snapshots.jsonl (one session);
  * output goes to --out instead of results/abc_eval_protocol_v1/{method}/.
The `method` column is kept as 'baseline' (the label the original used for Ours) so
the CSV is directly comparable with the published CLEANUP_PROBES_{B,C}.csv.

usage: cleanup_eval_h.py --session B --snapshots S.jsonl --bin probes.bin --out DIR
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol_common_h import (ROOT, method_dir, load_snapshots_session, object_centroid,  # noqa: E402
                               write_csv)

sys.path.append(str(Path(__file__).resolve().parents[1] / 'cleanup'))
from cleanup_binary import read_binary  # noqa: E402  (original reader, no paths inside)

STATIC_LABELS = {0, 3, 5, 8, 14}
DYNAMIC_LABEL = 12
DIST_THRESHOLDS = (0.05, 0.10, 0.15, 0.20)
GHOST_RETAINED_THRESHOLD = 0.05
GHOST_CLEAR_THRESHOLD = 0.01


class GTGeometry:
    def __init__(self, root):
        self.states = json.loads((root / 'gt_objects/states.json').read_text())
        self.by_id = {s['id']: s for s in self.states}
        self.trees = {}
        self.current_cache = {}

    def tree(self, state_id):
        if state_id not in self.trees:
            self.trees[state_id] = cKDTree(np.load(self.by_id[state_id]['surface']))
        return self.trees[state_id]

    def current_states(self, session, t):
        key = (session, round(float(t), 6))
        if key in self.current_cache:
            return self.current_cache[key]
        out = []
        for s in self.states:
            if s['session'] != session:
                continue
            lo, hi = s.get('aggregation_window_s', [0, 1000])
            if lo - 1e-9 <= t <= hi + 1e-9:
                out.append(s)
        self.current_cache[key] = out
        return out


def owner_state_maps(vby, start_ns, session, snaps, gt):
    """For every binary snapshot, map owner node id -> associated current GT state id.

    Original signature took (method, session, ...) and re-loaded the snapshots; the
    body is unchanged."""
    out = {}
    for snap in snaps:
        v = vby.get(int(snap['stamp_ns']))
        t = (snap['stamp_ns'] - start_ns) / 1e9
        cur = gt.current_states(session, t)
        node_by_id = {}
        if v is not None:
            for p in v['objects']:
                node_by_id[p['id']] = p
        mapping = {}
        for o in snap['owners']:
            node = node_by_id.get(o['id'])
            if node is None or not node.get('_current'):
                mapping[o['id']] = None
                continue
            sem = node.get('semantic')
            cen = object_centroid(node)
            best = None
            best_d = np.inf
            if cen is not None:
                for s in cur:
                    if s['semantic'] != sem:
                        continue
                    d = float(np.linalg.norm(cen - np.asarray(s['centroid'], float)))
                    if d < best_d:
                        best_d = d
                        best = s['id']
            mapping[o['id']] = best if best_d <= 0.5 else None
        out[int(snap['index'])] = mapping
    return out


def classify_event(method, session, snaps, probes, events, gt, dist_threshold=0.10,
                   start_ns=None, owner_maps=None, evidence_times=None):
    # verbatim from tools/abc_eval_protocol_v1/cleanup_eval.py (start_ns is always passed here)
    tree_cache = {}
    new_tree_cache = {}
    out = []
    relevant = [e for e in events if e['session'] == session and e['event_id'] in probes]
    for snap in snaps:
        t = (snap['stamp_ns'] - start_ns) / 1e9
        mapping = owner_maps.get(int(snap['index']), {}) if owner_maps else {}
        for ev in relevant:
            eid = ev['event_id']
            if t < ev['start_s'] - 1e-9:
                continue
            pts = np.asarray(probes[eid]['points'], float)
            old_id = ev['old']['id'] if ev.get('old') else None
            new_id = ev['new']['id'] if ev.get('new') else None
            cur = gt.current_states(session, t)
            excl = old_id if old_id in {s['id'] for s in cur} else None
            key = (snap['index'], excl)
            if key not in tree_cache:
                use = [s for s in cur if s['id'] != excl]
                if use:
                    pts_list = [np.load(s['surface']) for s in use]
                    owners = np.concatenate([np.full(len(p), i) for i, p in enumerate(pts_list)])
                    allpts = np.concatenate(pts_list, axis=0)
                    tree_cache[key] = (cKDTree(allpts), owners, use)
                else:
                    tree_cache[key] = (None, None, use)
            tree, owners, use = tree_cache[key]
            if tree is not None:
                d_other, idx = tree.query(pts)
                other_owner = owners[idx]  # noqa: F841 (kept from original)
            else:
                d_other = np.full(len(pts), np.inf)
                other_owner = np.full(len(pts), -1)  # noqa: F841
            if new_id is not None:
                if new_id not in new_tree_cache:
                    new_tree_cache[new_id] = gt.tree(new_id)
                d_new = new_tree_cache[new_id].query(pts)[0]
            else:
                d_new = np.full(len(pts), np.inf)
            evbin = snap['events'][eid]
            empty_t = None if evidence_times is None else evidence_times.get((eid, 'old_site_empty'))
            d_full = np.minimum(evbin['global_d'], evbin['obj_d'])
            global_near = evbin['global_d'] <= dist_threshold
            obj_near = evbin['obj_d'] <= dist_threshold
            owner_sem = np.full(len(pts), 0xFFFFFFFF, dtype=np.uint32)
            owner_state = np.array([None] * len(pts), dtype=object)
            owner_current = np.zeros(len(pts), dtype=bool)
            owner_dynamic = np.zeros(len(pts), dtype=bool)
            meta = snap['owners']
            for i, oi in enumerate(evbin['obj_owner']):
                if oi == 0xFFFFFFFF or int(oi) >= len(meta):
                    continue
                owner_sem[i] = meta[int(oi)]['semantic']
                oid = meta[int(oi)]['id']
                st = mapping.get(oid)
                if st is not None:
                    owner_state[i] = st
                    owner_current[i] = True
                if meta[int(oi)]['semantic'] == DYNAMIC_LABEL:
                    owner_dynamic[i] = True
            labels = evbin['global_label']
            static_bg = np.array([lab in STATIC_LABELS for lab in labels])
            current_sems = {s['semantic'] for s in cur}
            new_sem = gt.by_id[new_id]['semantic'] if (new_id and new_id in gt.by_id) else None
            global_label_sem = np.array([(int(lab) in current_sems) if lab != 0xFFFFFFFF else False for lab in labels])
            global_label_new = np.array([(int(lab) == new_sem) if (new_sem is not None and lab != 0xFFFFFFFF) else False for lab in labels])
            explained_new = (d_new <= dist_threshold) | (obj_near & (owner_state == new_id)) | (global_near & global_label_new & (d_new <= 0.20))
            explained_other = (d_other <= dist_threshold) | (obj_near & owner_current & ~explained_new) | (global_near & global_label_sem & (d_other <= 0.20))
            explained_bg = global_near & static_bg
            explained_dynamic = obj_near & owner_dynamic
            explained = explained_new | explained_other | explained_bg | explained_dynamic
            map_support = (d_full <= dist_threshold) & (~explained)
            ghost_obj = obj_near & (~explained)
            ghost_global = global_near & (~explained)
            after_evidence = empty_t is not None and t >= empty_t - 1e-9
            ghost = map_support & after_evidence
            ghost_obj_cert = ghost_obj & after_evidence
            ghost_global_cert = ghost_global & after_evidence
            unresolved = map_support & (~after_evidence)
            unsupported = d_full > dist_threshold
            n = len(pts)
            out.append(dict(
                method=method, session=session, event_id=eid, index=snap['index'],
                stamp_ns=snap['stamp_ns'], time_s=t, n_probes=n,
                explained_new=int(explained_new.sum()),
                explained_other=int((explained_other & ~explained_new).sum()),
                explained_owner=int(((obj_near & owner_current) & ~explained_new & ~(d_other <= dist_threshold)).sum()),
                explained_background=int((explained_bg & ~explained_new & ~explained_other).sum()),
                explained_dynamic=int((explained_dynamic & ~explained_new & ~explained_other & ~explained_bg).sum()),
                ghost=int(ghost.sum()), unresolved=int(unresolved.sum()),
                unsupported=int(unsupported.sum()),
                ghost_obj=int(ghost_obj_cert.sum()), ghost_global=int(ghost_global_cert.sum()),
                ghost_obj_fraction=float(ghost_obj_cert.mean()) if n else None,
                ghost_global_fraction=float(ghost_global_cert.mean()) if n else None,
                ghost_fraction=float(ghost.mean()) if n else None,
                unresolved_fraction=float(unresolved.mean()) if n else None,
                unsupported_fraction=float(unsupported.mean()) if n else None,
                explained_fraction=float(explained.mean()) if n else None,
                evidence_time_s=empty_t,
                after_evidence=bool(after_evidence),
                status=('evidence_unavailable' if empty_t is None else
                        'not_yet_evidenced' if not after_evidence else
                        'retained' if n and ghost.mean() >= GHOST_RETAINED_THRESHOLD else
                        'cleared' if n and ghost.mean() <= GHOST_CLEAR_THRESHOLD else
                        'ambiguous' if n else 'no_probes')
            ))
    return out


def run(session, snapshots, binpath, out_dir, method='baseline'):
    base = method_dir('baseline')  # GT / probes / events / evidence are dataset-level
    gt = GTGeometry(base)
    probes = {g['event_id']: g for g in json.loads((base / 'temporal/RESIDUAL_PROBES.json').read_text())}
    events = json.loads((base / 'events.json').read_text())
    ev_json = json.loads((base / 'evidence.json').read_text())
    evidence_times = {}
    for r in ev_json:
        eid = r['id']
        for suffix in ('_old_site_empty', '_new_state'):
            if eid.endswith(suffix):
                eid = eid[:-len(suffix)]
        evidence_times[(eid, r['kind'])] = r.get('evidence_time_s')
    mod, versions = load_snapshots_session(snapshots, session)
    vby = {int(v['stamp_ns']): v for v in versions if v['session'] == session and v.get('index', -1) >= 0}
    start_ns = mod.BASE[session]
    snaps = read_binary(binpath)
    owner_maps = owner_state_maps(vby, start_ns, session, snaps, gt)
    rows = classify_event(method, session, snaps, probes, events, gt, owner_maps=owner_maps,
                          evidence_times=evidence_times, start_ns=start_ns)
    write_csv(Path(out_dir) / f'CLEANUP_PROBES_{session}.csv', rows)
    last = {}
    for r in rows:
        last[r['event_id']] = r
    for eid, r in sorted(last.items()):
        print(eid, round(r['time_s'], 3), r['status'], 'ghost', round(r['ghost_fraction'] or 0, 4),
              'ghost_obj', round(r['ghost_obj_fraction'] or 0, 4), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--session', choices=['B', 'C'], required=True)
    ap.add_argument('--snapshots', type=Path, required=True)
    ap.add_argument('--bin', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run(args.session, args.snapshots, args.bin, args.out)


if __name__ == '__main__':
    main()
