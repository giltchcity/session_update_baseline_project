"""Harness copy of tools/abc_eval_protocol_v1/eval_state_change.py for ONE session,
plus the two published post-hoc steps that tools/abc_final/build_final_tables.py
applied by hand (its D2/D3 tuples are hard-coded there).

Protocol (verbatim from the original):
  appeared    := new state not represented before AND represented after
  disappeared := old state represented before AND NOT represented after
  represented := >=50 % of the GT state's surface points within 0.10 m of a
                 same-semantic, native-present, non-empty object surface
                 (1500-point subsample per node from snapshots.jsonl).
  old state after a changed-object event: protocol_v1 cleanup probes
  (CLEANUP_PROBES_{S}.csv, object_layer: ghost_obj_fraction >= 0.05 => still there).
  Anchors ("before" times) are the original constants: D2 A 90.0 s, B 34.1 s,
  C 100.499978 s; D3 before = t <= 0 (the inherited seed snapshot inside the
  current map); after = session end.

Published post-hoc steps (build_final_tables.py comments), applied on top and
reported separately from the raw protocol_v1 counts:
  1. non-empty-mesh policy: a false disappearance of an UNCHANGED object is voided
     when the end-of-session version still holds a same-semantic node with a
     non-empty mesh whose centroid is within 0.5 m of the GT state (regardless of
     native present). This is exactly the published C_I4 correction ("node keeps a
     991,611-vertex mesh, so it is present"; centroid error 0.0888 m).
  2. Table 1 Change = D2 object_layer combined (appeared + disappeared) TP/FP/FN.
     Table 2 Change = D3 object_layer combined over the common cohort excluding I2.

usage: eval_state_change_h.py --session B --snapshots S.jsonl [--cleanup CLEANUP_PROBES_B.csv] --out DIR
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol_common_h import (load_gt, load_raw_surfaces_path, load_snapshots_session, metric,  # noqa: E402
                               surface_represented, version_at, write_csv)

CLEAN_GHOST_THRESHOLD = 0.05
EXCLUDED_FROM_D3_COHORT = {2}   # I2 (cabinet) is Table 4, not the common D3 cohort
NONEMPTY_POLICY_GATE_M = 0.5


def load_cleanup_csv(path):
    """Original load_cleanup(method) reading an explicit CLEANUP_PROBES_{S}.csv."""
    if path is None or not Path(path).exists():
        return {}
    rows = list(csv.DictReader(Path(path).open()))
    out = defaultdict(list)
    for r in rows:
        r['time_s'] = float(r['time_s'])
        for key in ('ghost_fraction', 'ghost_obj_fraction'):
            r[key] = float(r[key]) if r.get(key) not in ('', 'None', None) else None
        out[r['event_id']].append(r)
    for k in out:
        out[k].sort(key=lambda x: x['time_s'])
    return out


def inventory(finals, session):
    return {s['object_id']: s for s in finals[session]}


def build_events(events, finals):
    # verbatim
    out = []
    for tr in ['AB', 'BC']:
        old_s, new_s = tr[0], tr[1]
        old_inv = inventory(finals, old_s)
        new_inv = inventory(finals, new_s)
        changed = {e['object_id']: e for e in events if e['scope'] == 'D3' and e['transition'] == tr}
        for oid in sorted(set(old_inv) | set(new_inv) | set(changed)):
            e = changed.get(oid)
            if e:
                out.append(dict(scope='D3', transition=tr, session=new_s, object_id=oid,
                                old=e['old'], new=e['new'], start_s=e['start_s'], end_s=e['end_s'],
                                expected_appeared=e['expected_appeared'], expected_disappeared=e['expected_disappeared'],
                                change_type=e['change_type']))
            else:
                out.append(dict(scope='D3', transition=tr, session=new_s, object_id=oid,
                                old=old_inv.get(oid), new=new_inv.get(oid), start_s=0.0,
                                end_s=None, expected_appeared=False, expected_disappeared=False,
                                change_type='unchanged'))
    for s, before in [('A', 90.0), ('B', 34.1), ('C', 100.499978)]:
        inv = inventory(finals, s)
        changed = {e['object_id']: e for e in events if e['scope'] == 'D2' and e['transition'] == s}
        for oid in sorted(inv):
            e = changed.get(oid)
            if e:
                out.append(dict(scope='D2', transition=s, session=s, object_id=oid,
                                old=e['old'], new=e['new'], start_s=before, end_s=None,
                                expected_appeared=e['expected_appeared'],
                                expected_disappeared=e['expected_disappeared'],
                                change_type=e['change_type']))
            else:
                st = inv[oid]
                out.append(dict(scope='D2', transition=s, session=s, object_id=oid,
                                old=st, new=st, start_s=before, end_s=None,
                                expected_appeared=False, expected_disappeared=False,
                                change_type='unchanged'))
    return out


def cleanup_at(cleanup, eid, deadline):
    rows = cleanup.get(eid)
    if not rows:
        return None
    use = [r for r in rows if r['time_s'] <= deadline + 1e-9]
    return use[-1] if use else None


def evaluate(session, snapshots, cleanup_csv, method='baseline'):
    mod, versions = load_snapshots_session(snapshots, session)
    raw = {session: load_raw_surfaces_path(snapshots)}
    gt = load_gt()
    finals = gt['finals']
    events = gt['events']
    evs = [e for e in build_events(events, finals) if e['session'] == session]
    cleanup = load_cleanup_csv(cleanup_csv)
    rows = []
    for e in evs:   # loop body verbatim
        session = e['session']
        end_old = mod.LENGTH[session]
        end_new = e['end_s'] if e['end_s'] is not None else end_old
        v_before = version_at(versions, session, e['start_s'])
        v_after_new = version_at(versions, session, end_new)
        v_after_old = version_at(versions, session, end_old)
        old = e['old']
        new = e['new']
        old_before = surface_represented(mod, raw[session], v_before, old) if old else False
        new_before = surface_represented(mod, raw[session], v_before, new) if new else False
        new_after = surface_represented(mod, raw[session], v_after_new, new) if new else False
        old_after_obj = old_after_full = None
        old_source = ''
        if old is not None:
            eid = f'{e["scope"]}_{e["transition"]}_I{e["object_id"]}'
            cr = cleanup_at(cleanup, eid, end_old)
            if cr is not None and cr['status'] != 'evidence_unavailable':
                old_after_full = (cr['ghost_fraction'] is not None and cr['ghost_fraction'] >= CLEAN_GHOST_THRESHOLD)
                old_after_obj = (cr['ghost_obj_fraction'] is not None and cr['ghost_obj_fraction'] >= CLEAN_GHOST_THRESHOLD)
                old_source = 'cleanup'
            elif e['change_type'] == 'unchanged':
                old_after_obj = old_after_full = surface_represented(mod, raw[session], v_after_old, old)
                old_source = 'object_surface'
            else:
                old_after_obj = old_after_full = None
                old_source = 'unavailable'
        pred_appeared = bool((not new_before) and new_after) if new is not None else False

        def pred_dis(oa):
            return bool(old_before and (not oa)) if (old is not None and oa is not None) else None
        rows.append(dict(
            method=method, scope=e['scope'], transition=e['transition'], session=session,
            object_id=e['object_id'], change_type=e['change_type'],
            expected_appeared=e['expected_appeared'], expected_disappeared=e['expected_disappeared'],
            old_before=old_before, new_before=new_before, new_after=new_after,
            old_after_obj=old_after_obj, old_after_full=old_after_full, old_source=old_source,
            pred_appeared=pred_appeared,
            pred_disappeared_obj=pred_dis(old_after_obj),
            pred_disappeared_full=pred_dis(old_after_full),
            new_state_id=new['id'] if new else '', old_state_id=old['id'] if old else '',
        ))
    return mod, versions, gt, rows


def protocol_summaries(rows, session, method='baseline'):
    """Original summary block, restricted to this session's groups."""
    groups = ['D2_' + session] + {'A': [], 'B': ['D3_AB'], 'C': ['D3_BC']}[session]
    summaries = []
    for protocol, old_key in [('object_layer', 'pred_disappeared_obj'), ('full_map', 'pred_disappeared_full')]:
        for group in groups:
            if group.startswith('D2_'):
                sel = [r for r in rows if r['scope'] == 'D2' and (group == 'D2_ABC' or r['transition'] == group[-1])]
            else:
                sel = [r for r in rows if r['scope'] == 'D3' and (group == 'D3_ABC' or r['transition'] == group[3:])]
            app_tp = app_fp = app_fn = 0
            dis_tp = dis_fp = dis_fn = 0
            old_applicable = not (method == 'khronos' and group.startswith('D3'))
            for r in sel:
                if r['expected_appeared']:
                    if r['pred_appeared']:
                        app_tp += 1
                    else:
                        app_fn += 1
                if old_applicable:
                    pred = r[old_key]
                    if pred is not None:
                        if r['expected_disappeared'] and pred:
                            dis_tp += 1
                        elif r['expected_disappeared'] and not pred:
                            dis_fn += 1
                        elif (not r['expected_disappeared']) and pred:
                            dis_fp += 1
            summaries.append(dict(method=method, protocol=protocol, group=group, component='appeared',
                                  **metric(app_tp, app_fp, app_fn), n_components=app_tp + app_fp + app_fn))
            summaries.append(dict(method=method, protocol=protocol, group=group, component='disappeared',
                                  **metric(dis_tp, dis_fp, dis_fn), n_components=dis_tp + dis_fp + dis_fn,
                                  applicable=old_applicable))
            summaries.append(dict(method=method, protocol=protocol, group=group, component='combined',
                                  **metric(app_tp + dis_tp, app_fp + dis_fp, app_fn + dis_fn),
                                  n_components=app_tp + dis_tp + app_fp + dis_fp + app_fn + dis_fn,
                                  old_applicable=old_applicable))
    return summaries


def apply_nonempty_mesh_policy(rows, versions, mod, gt, session):
    """Published step 1 (see module docstring). Adds pred_disappeared_obj_published."""
    v_end = version_at(versions, session, mod.LENGTH[session])
    overrides = []
    for r in rows:
        r['pred_disappeared_obj_published'] = r['pred_disappeared_obj']
        r['published_override'] = ''
        if r['change_type'] != 'unchanged' or r['pred_disappeared_obj'] is not True:
            continue
        st = gt['by_id'][r['old_state_id']]
        sc = np.asarray(st['centroid'], float)
        hits = []
        for p in (v_end or {}).get('objects', []):
            if p.get('semantic') != st['semantic'] or p.get('mesh_vertices', 0) <= 0 or p.get('centroid') is None:
                continue
            d = float(np.linalg.norm(np.asarray(p['centroid'], float) - sc))
            if d <= NONEMPTY_POLICY_GATE_M:
                hits.append((d, p))
        if hits:
            d, p = min(hits, key=lambda x: x[0])
            r['pred_disappeared_obj_published'] = False
            r['published_override'] = ('nonempty_mesh_policy node=%s mesh_vertices=%d native_present=%s centroid_err_m=%.4f'
                                       % (p['id'], p['mesh_vertices'], p.get('present'), d))
            overrides.append(dict(scope=r['scope'], transition=r['transition'], object_id=r['object_id'],
                                  state=r['old_state_id'], node=p['id'], mesh_vertices=p['mesh_vertices'],
                                  native_present=p.get('present'), centroid_err_m=round(d, 4)))
    return overrides


def combined_counts(sel, key):
    tp = fp = fn = 0
    for r in sel:
        if r['expected_appeared']:
            if r['pred_appeared']:
                tp += 1
            else:
                fn += 1
        pred = r[key]
        if pred is not None:
            if r['expected_disappeared'] and pred:
                tp += 1
            elif r['expected_disappeared'] and not pred:
                fn += 1
            elif (not r['expected_disappeared']) and pred:
                fp += 1
    return tp, fp, fn


def prf(counts):
    """build_final_tables.prf (percent)."""
    tp, fp, fn = counts
    p = 100.0 * tp / (tp + fp) if tp + fp else float('nan')
    r = 100.0 * tp / (tp + fn) if tp + fn else float('nan')
    f = 2 * p * r / (p + r) if (p == p and r == r and p + r > 0) else float('nan')
    return dict(TP=tp, FP=fp, FN=fn, P=round(p, 2) if p == p else None,
                R=round(r, 2) if r == r else None, F1=round(f, 2) if f == f else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--session', choices=['A', 'B', 'C'], required=True)
    ap.add_argument('--snapshots', type=Path, required=True)
    ap.add_argument('--cleanup', type=Path, default=None, help='CLEANUP_PROBES_{S}.csv (required for B/C)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    s = args.session
    if s in 'BC' and (args.cleanup is None or not args.cleanup.exists()):
        raise SystemExit('session %s needs --cleanup CLEANUP_PROBES_%s.csv (old-state probes)' % (s, s))
    args.out.mkdir(parents=True, exist_ok=True)
    mod, versions, gt, rows = evaluate(s, args.snapshots, args.cleanup)
    write_csv(args.out / 'STATE_CHANGE_DETAILS.csv', rows)
    write_csv(args.out / 'STATE_CHANGE_SUMMARY.csv', protocol_summaries(rows, s))
    overrides = apply_nonempty_mesh_policy(rows, versions, mod, gt, s)
    write_csv(args.out / 'STATE_CHANGE_DETAILS_PUBLISHED_POLICY.csv', rows)
    d2 = [r for r in rows if r['scope'] == 'D2']
    result = dict(
        session=s, snapshots=str(args.snapshots), cleanup_csv=str(args.cleanup) if args.cleanup else None,
        anchors_s=dict(D2={'A': 90.0, 'B': 34.1, 'C': 100.499978}[s], D3=0.0, after='session end %.6f s' % mod.LENGTH[s]),
        anchor_note='Published text quotes the D2 A anchor as 94.63 s (evidence time of D2_A_I2); '
                    'the protocol_v1 code that produced the published counts uses 90.0 s.',
        D2=dict(table='Table 1 Change (D2, object_layer combined)',
                raw_protocol_v1=prf(combined_counts(d2, 'pred_disappeared_obj')),
                published_policy=prf(combined_counts(d2, 'pred_disappeared_obj_published'))),
        overrides=overrides,
    )
    if s in 'BC':
        tr = {'B': 'AB', 'C': 'BC'}[s]
        d3_all = [r for r in rows if r['scope'] == 'D3']
        d3 = [r for r in d3_all if r['object_id'] not in EXCLUDED_FROM_D3_COHORT]
        result['D3'] = dict(table='Table 2 Change (D3 %s, object_layer combined, common cohort excluding I2)' % tr,
                            transition=tr,
                            raw_protocol_v1_incl_I2=prf(combined_counts(d3_all, 'pred_disappeared_obj')),
                            raw_protocol_v1_excl_I2=prf(combined_counts(d3, 'pred_disappeared_obj')),
                            published_policy=prf(combined_counts(d3, 'pred_disappeared_obj_published')))
    (args.out / 'CHANGE_ROWS.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
