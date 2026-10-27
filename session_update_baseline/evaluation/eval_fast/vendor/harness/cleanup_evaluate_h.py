"""Harness copy of tools/abc_eval_v2/cleanup_evaluate.py for ONE method and ONE session.

Causal RGBD certificates for full-current-map cleanup at the common old-surface
probes (results/abc_eval_v2/cleanup/probes_{b,c}.json). The evidence projection,
cohort, per-nearest-point certificates and summaries are copied verbatim. Changes
are path/scope only:
  * the per-snapshot nearest-surface table comes from an explicit cleanup_mesh
    output (--mesh-jsonl) for one method, instead of the fixed list
    results/abc_eval_v2/cleanup/{baseline,khronos,panoptic,...}_{b,c}_mesh.jsonl;
  * the snapshot-count assertion uses --expected-snapshots (count of the map under
    test) instead of results/abc_eval_final_{method}/input_manifest.json;
  * outputs go to --out instead of results/abc_eval_v2/cleanup/.
Running one method alone gives the same final_* values as the published multi-method
run: evidence is per point (start time + position), frames are processed exactly at
their own sample times, and the final row is always the query at the session end.

PROVENANCE CAVEAT: results/abc_eval_v2/cleanup/PROVENANCE.json records
cleanup_evaluate.py sha256 0b787f84...; the file on disk (copied here) is 865e9f13...,
modified 8 min after event_summary_{b,c}.csv were written. Validation mode compares the
final_full_* columns with the published event_summary rows to detect any difference.

usage: cleanup_evaluate_h.py --session B --mesh-jsonl M.jsonl --expected-snapshots 55 --out DIR
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('/home/jixian/Desktop/FT')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ghost'))
from depth_surface_check import project_evidence  # noqa: E402

P_ORIG = ROOT / 'results/abc_eval_v2/cleanup'      # read-only: probes_{b,c}.json
REF = ROOT / 'results/abc_eval_final_baseline'     # read-only: input manifest


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def latency_summary(times, status, evidence, deadline):
    # A map query uses latest <= query; recompute the verdict at evidence.
    use = [(t, s) for t, s in zip(times, status) if evidence <= t <= deadline]
    first = next((t for t, s in use if s == 'clear'), None)
    sustained = next((t for j, (t, s) in enumerate(use) if s == 'clear' and all(x[1] == 'clear' for x in use[j:])), None)
    return dict(first_clear_s=first, first_latency_s=None if first is None else first - evidence,
                sustained_clear_s=sustained, sustained_latency_s=None if sustained is None else sustained - evidence,
                correct_at_evidence=bool(use and use[0][0] == evidence and use[0][1] == 'clear'),
                deadline_status=use[-1][1] if use else 'not_evaluable',
                clear_then_nonclear=sum(a[1] == 'clear' and b[1] != 'clear' for a, b in zip(use, use[1:])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--session', choices=['B', 'C'], required=True)
    ap.add_argument('--period', type=float, default=1.0)
    ap.add_argument('--mesh-jsonl', type=Path, required=True)
    ap.add_argument('--expected-snapshots', type=int, required=True)
    ap.add_argument('--probes', type=Path, default=None, help='default: results/abc_eval_v2/cleanup/probes_{s}.json')
    ap.add_argument('--method', default='baseline', help="row label; keep 'baseline' (Ours label of the original)")
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.method.startswith('panoptic') or args.method.endswith('_viewer'):
        raise SystemExit('method label must not trigger the panoptic/viewer branches of the original')
    OUTD = args.out
    OUTD.mkdir(parents=True, exist_ok=True)
    s = args.session
    cfg = json.load(open(REF / 'input_manifest.json'))['sessions'][s]
    base = cfg['timestamp_provenance']['session_start_ns']
    duration = cfg['last_timestamp'] / 1e9
    groups = json.load(open(args.probes or (P_ORIG / f'probes_{s.lower()}.json')))
    maps = {}
    for method in [args.method]:
        rows = [json.loads(l) for l in open(args.mesh_jsonl)]
        expected = args.expected_snapshots
        assert len(rows) == expected, (method, len(rows), expected)
        maps[method] = rows
        for r in rows:
            r['time_s'] = max(0, (r['stamp_ns'] - base) / 1e9)
    # ---- verbatim from here (P -> OUTD for outputs) ----
    event_data = {}
    allpoints = []
    allstarts = []
    offset = 0
    for g in groups:
        eid = g['event_id']
        probes = np.array(g['points'])
        hit_arrays = []
        bindings = []
        for method, rows in maps.items():
            for r in rows:
                hits = r['events'].get(eid)
                if hits is None:
                    continue
                present = np.array([h is not None for h in hits])
                xyz = np.array([h[1:4] for h in hits if h is not None]).reshape(-1, 3)
                hit_arrays.append(xyz)
                bindings.append((method, r['index'], present, len(xyz)))
        flat = np.concatenate(hit_arrays) if hit_arrays else np.empty((0, 3))
        candidates, inv = np.unique(np.round(flat, 5), axis=0, return_inverse=True)
        d = dict(group=g, probe_offset=offset, probe_count=len(probes), candidate_offset=offset + len(probes),
                 candidate_count=len(candidates), bindings={})
        pos = 0
        for method, ix, present, n in bindings:
            ids = np.full(len(probes), -1, dtype=int)
            ids[present] = offset + len(probes) + inv[pos:pos + n]
            pos += n
            d['bindings'][(method, ix)] = ids
        event_data[eid] = d
        allpoints.extend([probes, candidates])
        allstarts.extend([np.full(len(probes) + len(candidates), g['start_s'])])
        offset += len(probes) + len(candidates)
    points = np.concatenate(allpoints)
    starts = np.concatenate(allstarts)
    support = np.zeros(len(points), np.uint16)
    free = np.zeros_like(support)
    strictfree = np.zeros_like(support)
    occluded = np.zeros_like(support)
    rgbd = Path(cfg['rgbd'])
    transform = np.array(cfg['transform'])
    selected = []
    last = -np.inf
    for f in csv.DictReader(open(rgbd / 'timestamps.csv')):
        t = int(f['TimeStamp']) / 1e9
        if t - last >= args.period - 1e-8:
            selected.append((t, f['ImageID']))
            last = t
    query_times = sorted(set([r['time_s'] for rows in maps.values() for r in rows] + [t for t, stem in selected] + [duration]))
    timeline = []
    frame_idx = 0
    cohorts = {}
    method_info = {}
    traces = []
    snap_times = {m: np.array([r['time_s'] for r in rows]) for m, rows in maps.items()}
    for eid, d in event_data.items():
        g = d['group']
        initial_time = g.get('initial_snapshot_anchor_s', g['start_s'])
        for method, rows in maps.items():
            if (method, 0) not in d['bindings']:
                continue
            initial_ix = int(np.searchsorted(snap_times[method], initial_time, side='right') - 1)
            if initial_ix < 0:
                initial = np.zeros(d['probe_count'], bool)
            else:
                initial = d['bindings'][(method, initial_ix)] >= 0
            method_info[(method, eid)] = dict(initial=initial, initial_ix=initial_ix,
                                              initial_time_s=None if initial_ix < 0 else float(snap_times[method][initial_ix]),
                                              eligible=False)
    print(s, 'all test points', len(points), 'frames', len(selected), flush=True)
    for qt in query_times:
        while frame_idx < len(selected) and selected[frame_idx][0] <= qt + 1e-8:
            t, stem = selected[frame_idx]
            depth = np.array(Image.open(rgbd / (stem + '_depth.tiff')), dtype=np.float32)
            semantic = np.array(Image.open(Path(cfg['semantic']) / (stem + '_segmentation.png')))
            pose = transform @ np.loadtxt(rgbd / (stem + '_pose.txt'))
            active = np.flatnonzero(starts <= t + 1e-8)
            for begin in range(0, len(active), 100000):
                ix = active[begin:begin + 100000]
                v, _ = project_evidence(points[ix], depth, pose, cfg['intrinsics'], .10, semantic)
                vstrict, _ = project_evidence(points[ix], depth, pose, cfg['intrinsics'], .20, semantic)
                support[ix] += v == 1
                free[ix] += v == 2
                strictfree[ix] += vstrict == 2
                occluded[ix] += v == 3
            traces.append(dict(time_s=t, frame=stem))
            frame_idx += 1
            if frame_idx % 20 == 0:
                print(s, 'evidence', frame_idx, '/', len(selected), flush=True)
        for eid, d in event_data.items():
            g = d['group']
            n = d['probe_count']
            idx = np.arange(d['probe_offset'], d['probe_offset'] + n)
            known = (strictfree[idx] >= 3) & (support[idx] < 3)
            threshold = max(30, int(np.ceil(.05 * n)))
            if eid not in cohorts and known.sum() >= threshold:
                cohorts[eid] = dict(time_s=qt, mask=known.copy(), count=int(known.sum()))
            if eid not in cohorts:
                continue
            cohort = cohorts[eid]
            cohort_mask = cohort['mask']
            for method, rows in maps.items():
                if method.startswith('panoptic') and qt != duration:
                    continue
                info = method_info.get((method, eid))
                if info is None:
                    continue
                ix = int(np.searchsorted(snap_times[method], qt + 1e-8, side='right') - 1)
                if ix < 0:
                    continue
                initial = info['initial']
                fixed = cohort_mask & initial
                active = fixed & known
                hits = d['bindings'][(method, ix)]
                near = active & (hits >= 0)
                candidate_ids = hits[near]
                hit_free = (free[candidate_ids] >= 3) & (support[candidate_ids] < 3)
                hit_supported = (support[candidate_ids] >= 3) & (free[candidate_ids] < 3)
                confirmed = int(hit_free.sum())
                legal = int(hit_supported.sum())
                unresolved = int(len(candidate_ids) - confirmed - legal)
                full_active = known & initial
                full_ids = hits[full_active & (hits >= 0)]
                full_wrong = int(((free[full_ids] >= 3) & (support[full_ids] < 3)).sum())
                full_legal = int(((support[full_ids] >= 3) & (free[full_ids] < 3)).sum())
                full_unknown = len(full_ids) - full_wrong - full_legal
                common_ids = hits[known & (hits >= 0)]
                common_wrong = int(((free[common_ids] >= 3) & (support[common_ids] < 3)).sum())
                common_legal = int(((support[common_ids] >= 3) & (free[common_ids] < 3)).sum())
                common_unknown = len(common_ids) - common_wrong - common_legal
                denom = int(active.sum())
                fixedn = int(fixed.sum())
                minimum = max(30, int(np.ceil(.05 * n)))
                status = 'not_evaluable'
                if fixedn >= minimum and denom >= .95 * fixedn:
                    if (confirmed + unresolved) / max(1, denom) <= .01:
                        status = 'clear'
                    elif confirmed / denom > .01:
                        status = 'residual'
                    else:
                        status = 'uncertain_surface'
                elif fixedn >= minimum:
                    status = 'cohort_reoccupied_or_conflicting'
                row = dict(method=method, event_id=eid, scope=g['scope'], transition=g['transition'], time_s=qt,
                           snapshot_index=ix, snapshot_time_s=float(snap_times[method][ix]),
                           evidence_time_s=cohort['time_s'], status=status,
                           old_probe_count=n, cohort_probe_count=cohort['count'], initial_supported_cohort=fixedn,
                           currently_empty_initial_cohort=denom,
                           nearest_current_surface=int(near.sum()), confirmed_wrong_surface=confirmed,
                           legal_current_surface=legal, unresolved_current_surface=unresolved,
                           residual_fraction=None if not denom else confirmed / denom,
                           conservative_residual_fraction=None if not denom else (confirmed + unresolved) / denom,
                           full_known_empty_probe_count=int(known.sum()),
                           full_initial_supported_known_empty=int((known & initial).sum()),
                           full_confirmed_wrong_surface=full_wrong, full_legal_current_surface=full_legal,
                           full_unresolved_current_surface=full_unknown,
                           full_conservative_residual_fraction=None if not full_active.any() else (full_wrong + full_unknown) / int(full_active.sum()),
                           common_confirmed_wrong_surface=common_wrong, common_legal_current_surface=common_legal,
                           common_unresolved_current_surface=common_unknown,
                           common_confirmed_residual_fraction=None if not known.any() else common_wrong / int(known.sum()),
                           common_conservative_residual_fraction=None if not known.any() else (common_wrong + common_unknown) / int(known.sum()),
                           background_nearest=0, object_nearest=0)
                raw = rows[ix]['events'][eid]
                row['background_nearest'] = sum(raw[j][4] == 0 for j in np.flatnonzero(near))
                row['object_nearest'] = sum(raw[j][4] == 1 for j in np.flatnonzero(near))
                timeline.append(row)
    summaries = []
    for (method, eid), info in method_info.items():
        d = event_data[eid]
        g = d['group']
        rr = [r for r in timeline if r['method'] == method and r['event_id'] == eid]
        cohort = cohorts.get(eid)
        row = dict(method=method, event_id=eid, scope=g['scope'], transition=g['transition'], old_probe_count=d['probe_count'],
                   initial_supported_probes=int(info['initial'].sum()),
                   initial_support_fraction=float(info['initial'].mean()), initial_snapshot_time_s=info['initial_time_s'],
                   deadline_s=duration,
                   evidence_time_s=None if cohort is None else cohort['time_s'],
                   evidence_cohort_probes=0 if cohort is None else cohort['count'],
                   evidence_cohort_fraction=0 if cohort is None else cohort['count'] / d['probe_count'],
                   status='no_sufficient_empty_evidence' if not rr else rr[-1]['status'])
        if rr:
            if not method.startswith('panoptic'):
                row.update(latency_summary([r['time_s'] for r in rr], [r['status'] for r in rr], cohort['time_s'], duration))
            else:
                row.update(latency_status='N/A_final_maps_only', deadline_status=rr[-1]['status'])
            row.update({'final_' + k: rr[-1][k] for k in [
                'initial_supported_cohort', 'currently_empty_initial_cohort', 'nearest_current_surface',
                'confirmed_wrong_surface', 'legal_current_surface', 'unresolved_current_surface', 'residual_fraction',
                'conservative_residual_fraction', 'full_known_empty_probe_count', 'full_initial_supported_known_empty',
                'full_confirmed_wrong_surface', 'full_legal_current_surface', 'full_unresolved_current_surface',
                'full_conservative_residual_fraction', 'common_confirmed_wrong_surface', 'common_legal_current_surface',
                'common_unresolved_current_surface', 'common_confirmed_residual_fraction',
                'common_conservative_residual_fraction', 'background_nearest', 'object_nearest']})
        summaries.append(row)
    # (the original also appended 'khronos' N/A rows for D3 here; omitted, they carry no values)
    write_csv(OUTD / f'timeline_{s.lower()}.csv', timeline)
    write_csv(OUTD / f'event_summary_{s.lower()}.csv', summaries)
    (OUTD / f'event_summary_{s.lower()}.json').write_text(json.dumps(summaries, indent=2, default=int))
    (OUTD / f'evidence_frames_{s.lower()}.json').write_text(json.dumps(traces, indent=2))
    np.savez_compressed(OUTD / f'evidence_final_{s.lower()}.npz', points=points, support=support, free=free,
                        strictfree=strictfree, occluded=occluded)
    (OUTD / f'evidence_layout_{s.lower()}.json').write_text(json.dumps(
        {eid: {k: v for k, v in d.items() if k not in ['bindings', 'group']} for eid, d in event_data.items()}, indent=2))
    print(json.dumps(summaries, indent=2, default=int), flush=True)


if __name__ == '__main__':
    main()
