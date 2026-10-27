"""Final per-metric summary for one session of evaluate_real_session.sh, with an optional
validation against the published 2026-09-07 "Ours A->B->C" values.

usage: summarize_session.py --session b --name NAME --real-abc DIR [--table2] [--validate]
                            [--map M] [--prior P] [--geom-policies "viewer native"]
Exit status: 0, or 1 in --validate mode when any check FAILs (KNOWN_GAP / INFO do not fail).
"""
import argparse
import csv
import filecmp
import json
import math
from pathlib import Path

FT = Path('/home/jixian/Desktop/FT')
PROV = FT / 'results/provenance/official_native_eval_matched'           # read-only
BASE_PRED = FT / 'results/abc_eval_final_baseline/predictions'          # read-only
PV1 = FT / 'results/abc_eval_protocol_v1/baseline'                      # read-only
V2C = FT / 'results/abc_eval_v2/cleanup'                                # read-only
CV_GEOM = FT / 'results/current_version_20260921/real_abc/geometry'     # read-only (1 cm, native policy)

# Published values (results/abc_final_cvpr/TABLE1_MAIN.csv, TABLE2_CROSS_SESSION.csv,
# OBJECT_ONLINE_FINAL.csv, DIAGNOSTIC_OFFICIAL_NATIVE_CHANGE.csv; build_final_tables.py tuples).
PUB = {
    'obj_weighted': {'a': (100.0, 89.88, 93.75), 'b': (99.43, 97.62, 98.48), 'c': (96.85, 95.43, 96.04)},
    'obj_unweighted': {'a': (100.0, 82.31, 85.95), 'b': (99.35, 97.42, 98.34), 'c': (94.91, 97.34, 96.03)},
    'queries_nr': {'a': (61, 88), 'b': (55, 255), 'c': (50, 264)},
    'native_change_diag': {'a': 11.76, 'b': 11.76, 'c': 12.95},
    'd2': {'a': (1, 0, 0), 'b': (2, 0, 0), 'c': (2, 0, 0)},
    'd2_raw_protocol_v1': {'a': (1, 0, 0), 'b': (2, 0, 0), 'c': (2, 1, 0)},   # STATE_CHANGE_SUMMARY.csv
    'd3': {'b': (15, 0, 1), 'c': (10, 0, 0)},
    'd3_code_expected': {'b': (16, 0, 0), 'c': (10, 0, 0)},   # what protocol_v1 + policy gives on the published inputs
    'ghost': {'b': 21.70, 'c': 24.73},
    'geom_2cm_viewer': {'a': (93.28, 98.87, 99.80), 'b': (92.51, 99.05, 99.84), 'c': (91.71, 99.16, 99.94)},
}


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def fmt3(t):
    return ' / '.join('--' if v is None else '%.2f' % v for v in t)


class Checks:
    def __init__(self):
        self.items = []

    def add(self, status, name, got, expected, note=''):
        self.items.append(dict(status=status, check=name, got=got, expected=expected, note=note))
        print('VALIDATE %-9s %-44s got=%s expected=%s %s' % (status, name, got, expected, note), flush=True)

    def eq(self, name, got, expected, tol=None, note='', gap=False):
        if got is None:
            return self.add('MISSING', name, got, expected, note)
        if tol is None:
            ok = got == expected
        else:
            ok = all(abs(a - b) <= tol for a, b in zip(got if isinstance(got, (list, tuple)) else [got],
                                                     expected if isinstance(expected, (list, tuple)) else [expected]))
        self.add('PASS' if ok else ('KNOWN_GAP' if gap else 'FAIL'), name, got, expected, note)

    @property
    def failed(self):
        return [c for c in self.items if c['status'] in ('FAIL', 'MISSING')]


def csv_rows(path, keep=None):
    rows = list(csv.DictReader(Path(path).open()))
    return [r for r in rows if keep is None or keep(r)]


def num_equal(a, b):
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return (math.isnan(fa) and math.isnan(fb)) or abs(fa - fb) <= 1e-9 * max(1.0, abs(fb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', choices=list('abc'), required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--real-abc', type=Path, required=True)
    ap.add_argument('--table2', action='store_true')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--map', default='')
    ap.add_argument('--prior', default='')
    ap.add_argument('--geom-policies', default='viewer')
    args = ap.parse_args()
    s, S, n, RA = args.session, args.session.upper(), args.name, args.real_abc
    tag = '%s_%s' % (n, s)
    OBJ, EXP, CH, GEO, GH = RA / 'objects' / tag, RA / 'exports' / tag, RA / 'changes' / tag, RA / 'geometry' / tag, RA / 'ghost' / tag
    out = dict(session=S, name=n, map=args.map, prior_map=args.prior or None, table2=args.table2)
    lines = []

    obj = load(OBJ / 'weighted_nr/OBJECT_WEIGHTED_NR.json')
    nat = load(OBJ / 'explanation/NATIVE_ONLINE_SUMMARY.json')
    raw_native = None
    if nat:
        raw_native = tuple(100 * nat['summary'][k]['mean'] for k in ('ObjectPrecision', 'ObjectRecall', 'ObjectF1'))
    if obj:
        w, u = obj['duration_weighted_nr'], obj['unweighted_nr']
        lines.append('OBJECT  weighted+NR (Table 1)   P/R/F1 = %s   [queries %d, NR %d]'
                     % (fmt3((w['P'], w['R'], w['F1'])), obj['unique_queries'], obj['non_reviewed_nodes_total']))
        lines.append('OBJECT  unweighted+NR           P/R/F1 = %s' % fmt3((u['P'], u['R'], u['F1'])))
    else:
        lines.append('OBJECT  weighted+NR             MISSING (%s)' % (OBJ / 'weighted_nr/OBJECT_WEIGHTED_NR.json'))
    if raw_native:
        lines.append('OBJECT  raw native (unweighted) P/R/F1 = %s' % fmt3(raw_native))
    if obj:
        lines.append('OBJECT  native ChangeF1 (diag)  = %s' % obj['native_change_diagnostic']['ChangeF1_unweighted'])
    out.update(object=obj and {k: obj[k] for k in ('duration_weighted_nr', 'unweighted_nr', 'unique_queries',
                                                    'non_reviewed_nodes_total', 'native_change_diagnostic')},
               object_raw_native=raw_native)

    ch = load(CH / 'CHANGE_ROWS.json')
    if ch:
        d2p, d2r = ch['D2']['published_policy'], ch['D2']['raw_protocol_v1']
        lines.append('CHANGE  D2 (Table 1)            P/R/F1 = %s   TP/FP/FN %d/%d/%d   [raw protocol_v1 %d/%d/%d]'
                     % (fmt3((d2p['P'], d2p['R'], d2p['F1'])), d2p['TP'], d2p['FP'], d2p['FN'], d2r['TP'], d2r['FP'], d2r['FN']))
        for o in ch.get('overrides', []):
            lines.append('        non-empty-mesh policy override: %s' % json.dumps(o))
    else:
        lines.append('CHANGE  D2                      MISSING (%s)' % (CH / 'CHANGE_ROWS.json'))
    out['change'] = ch

    # G1 (eval/geometry_g1.py, room crop, 1 cm reference) is the primary geometry metric; the
    # 1 cm policy lines below are diagnostics and are labelled as such.
    g1 = load(GEO / 'geometry_viewer_G1.json')
    if g1:
        lines.append('GEOM    G1 (primary, room crop) F1@5/10/20 = %s   P/R@5 = %.2f / %.2f   area %.1f m2'
                     % (fmt3(tuple(100 * g1['F1_%scm' % t] for t in ('05', '10', '20'))),
                        100 * g1['P_05cm'], 100 * g1['R_05cm'], g1['pred_area_m2']))
    else:
        lines.append('GEOM    G1                      MISSING (%s)' % (GEO / 'geometry_viewer_G1.json'))
    out['geometry_G1'] = g1
    geo = {}
    for pol in args.geom_policies.split():
        g = load(GEO / ('geometry_%s_1cm.json' % pol))
        geo[pol] = g
        if g:
            lines.append('GEOM    %-6s policy, 1 cm ref   F1@5/10/20 = %s   P/R@5 = %.2f / %.2f   area %.1f m2'
                         '   [diagnostic, not G1]'
                         % (pol, fmt3(tuple(100 * g['F1_%scm' % t] for t in ('05', '10', '20'))),
                            100 * g['P_05cm'], 100 * g['R_05cm'], g['area_m2']))
        else:
            lines.append('GEOM    %-6s policy             MISSING' % pol)
    lines.append('GEOM    (published definition: viewer policy, 2 cm reference, deleted -> not reproducible; '
                 'published Ours %s F1@5/10/20 = %s)' % (S, fmt3(PUB['geom_2cm_viewer'][s])))
    out['geometry_1cm'] = geo

    gh = None
    if args.table2:
        d3 = ch and ch.get('D3')
        if d3:
            p = d3['published_policy']
            lines.append('CHANGE  D3 %s (Table 2)         P/R/F1 = %s   TP/FP/FN %d/%d/%d   [raw excl I2 %d/%d/%d, incl I2 %d/%d/%d]'
                         % (d3['transition'], fmt3((p['P'], p['R'], p['F1'])), p['TP'], p['FP'], p['FN'],
                            d3['raw_protocol_v1_excl_I2']['TP'], d3['raw_protocol_v1_excl_I2']['FP'], d3['raw_protocol_v1_excl_I2']['FN'],
                            d3['raw_protocol_v1_incl_I2']['TP'], d3['raw_protocol_v1_incl_I2']['FP'], d3['raw_protocol_v1_incl_I2']['FN']))
        else:
            lines.append('CHANGE  D3                      MISSING')
        gh = load(GH / 'GHOST.json')
        if gh:
            lines.append('GHOST   old-site full-map %s (Table 2) = %.2f %%   (%d / %d probes, %d D3 events)'
                         % (gh['transition'], gh['old_site_full_map_ghost_pct'], gh['numerator'], gh['denominator'],
                            gh['events_included']))
        else:
            lines.append('GHOST   old-site full-map        MISSING (%s)' % (GH / 'GHOST.json'))
        out['ghost'] = gh and {k: v for k, v in gh.items() if k != 'events'}

    # seed / prior consistency (cheap: first line of this export, last line of the prior export if present)
    if args.table2 and (EXP / 'snapshots.jsonl').exists():
        prev = {'b': 'a', 'c': 'b'}[s]
        with (EXP / 'snapshots.jsonl').open() as f:
            seed = json.loads(f.readline())['stamp_ns']
        pexp = RA / 'exports' / ('%s_%s' % (n, prev)) / 'snapshots.jsonl'
        if pexp.exists():
            last = None
            with pexp.open() as f:
                for line in f:
                    last = line
            pstamp = json.loads(last)['stamp_ns']
            lines.append('SEED    first snapshot stamp %d vs prior-session final stamp %d (%s): %s'
                         % (seed, pstamp, pexp, 'MATCH' if seed == pstamp else 'DIFFERENT'))
        else:
            lines.append('SEED    first snapshot stamp %d (no %s export to compare against)' % (seed, pexp.parent.name))

    print('=' * 100)
    print('SUMMARY %s session %s   map=%s' % (n, S, args.map))
    for l in lines:
        print('SUMMARY ' + l)
    print('=' * 100, flush=True)

    checks = Checks()
    if args.validate:
        print('VALIDATION against published 2026-09-07 Ours A->B->C (%s)' % S)
        # intermediate exports
        pub_snap = BASE_PRED / s / 'snapshots.jsonl'
        mine = EXP / 'snapshots.jsonl'
        checks.eq('snapshots.jsonl identical to published export', mine.exists() and filecmp.cmp(mine, pub_snap, shallow=False), True)
        pub_rows = PROV / ('ours_chain_%s_latest' % s) / 'explanation/native_unique_online_rows.csv'
        my_rows = OBJ / 'explanation/native_unique_online_rows.csv'
        checks.eq('native_unique_online_rows.csv identical', my_rows.exists() and filecmp.cmp(my_rows, pub_rows, shallow=False), True)
        pub_nat = load(PROV / ('ours_chain_%s_latest' % s) / 'explanation/NATIVE_ONLINE_SUMMARY.json')
        if pub_nat and raw_native:
            exp = tuple(100 * pub_nat['summary'][k]['mean'] for k in ('ObjectPrecision', 'ObjectRecall', 'ObjectF1'))
            checks.eq('raw native Object P/R/F1', tuple(round(v, 6) for v in raw_native), tuple(round(v, 6) for v in exp), tol=1e-6)
        if obj:
            w, u = obj['duration_weighted_nr'], obj['unweighted_nr']
            checks.eq('Object weighted+NR P/R/F1 (Table 1)', (w['P'], w['R'], w['F1']), PUB['obj_weighted'][s], tol=0.006)
            checks.eq('Object unweighted+NR P/R/F1', (u['P'], u['R'], u['F1']), PUB['obj_unweighted'][s], tol=0.006)
            checks.eq('unique queries / NR total', (obj['unique_queries'], obj['non_reviewed_nodes_total']), PUB['queries_nr'][s])
            checks.eq('native ChangeF1 diagnostic', obj['native_change_diagnostic']['ChangeF1_unweighted'], PUB['native_change_diag'][s], tol=0.006)
        if ch:
            d2r, d2p = ch['D2']['raw_protocol_v1'], ch['D2']['published_policy']
            checks.eq('D2 raw protocol_v1 TP/FP/FN', (d2r['TP'], d2r['FP'], d2r['FN']), PUB['d2_raw_protocol_v1'][s])
            checks.eq('D2 Table 1 TP/FP/FN (with non-empty-mesh policy)', (d2p['TP'], d2p['FP'], d2p['FN']), PUB['d2'][s])
            # row-level: STATE_CHANGE_DETAILS vs published rows of this session
            mine_rows = csv_rows(CH / 'STATE_CHANGE_DETAILS.csv')
            pub_rows_sc = csv_rows(PV1 / 'STATE_CHANGE_DETAILS.csv', lambda r: r['session'] == S)
            same = len(mine_rows) == len(pub_rows_sc) and all(
                all(num_equal(a.get(k), b.get(k)) for k in b) for a, b in zip(mine_rows, pub_rows_sc))
            checks.eq('STATE_CHANGE_DETAILS rows identical (%d)' % len(pub_rows_sc), same, True)
        if s in 'bc':
            mine_c = CH / ('CLEANUP_PROBES_%s.csv' % S)
            if mine_c.exists():
                a, b = csv_rows(mine_c), csv_rows(PV1 / ('CLEANUP_PROBES_%s.csv' % S))
                same = len(a) == len(b) and all(all(num_equal(x.get(k), y.get(k)) for k in y) for x, y in zip(a, b))
                checks.eq('CLEANUP_PROBES_%s rows identical (%d)' % (S, len(b)), same, True)
        if args.table2 and s in 'bc':
            d3 = ch and ch.get('D3')
            if d3:
                p = d3['published_policy']
                got = (p['TP'], p['FP'], p['FN'])
                checks.eq('D3 Table 2 TP/FP/FN', got, PUB['d3'][s], gap=(s == 'b' and got == PUB['d3_code_expected']['b']),
                          note=('published AB 15/0/1 has no surviving source; protocol_v1 + policy gives 16/0/0'
                                if s == 'b' else ''))
            if gh:
                checks.eq('old-site full-map ghost %', gh['old_site_full_map_ghost_pct'], PUB['ghost'][s], tol=0.006)
                mine_e = csv_rows(GH / ('event_summary_%s.csv' % s))
                pub_e = {r['event_id']: r for r in csv_rows(V2C / ('event_summary_%s.csv' % s), lambda r: r['method'] == 'baseline')}
                keys = [k for k in (mine_e[0] if mine_e else {}) if k.startswith('final_')] + ['evidence_time_s', 'initial_supported_probes']
                same = bool(mine_e) and all(all(num_equal(r.get(k), pub_e.get(r['event_id'], {}).get(k)) for k in keys) for r in mine_e)
                checks.eq('cleanup event_summary final_* identical', same, True,
                          note='detects the cleanup_evaluate.py source drift (0b787f84 -> 865e9f13)')
        for pol, g in geo.items():
            ref = load(CV_GEOM / ('published_chain_%s.json' % S))
            if not g:
                continue
            if pol == 'native' and ref:
                keys = [k for k in ref if k[:2] in ('P_', 'R_', 'F1')]
                checks.eq('geometry native 1cm == current_version published_chain_%s' % S,
                          tuple(g[k] for k in keys), tuple(ref[k] for k in keys), tol=1e-9)
            else:
                checks.add('INFO', 'geometry %s 1cm (no published 1 cm value)' % pol,
                           tuple(round(100 * g['F1_%scm' % t], 2) for t in ('05', '10', '20')),
                           PUB['geom_2cm_viewer'][s], note='expected = published 2 cm viewer values, not comparable')
        print('VALIDATION RESULT: %s (%d checks, %d failed)'
              % ('PASS' if not checks.failed else 'FAIL', len(checks.items), len(checks.failed)), flush=True)
        out['validation'] = checks.items

    sd = RA / 'summary' / tag
    sd.mkdir(parents=True, exist_ok=True)
    (sd / 'SUMMARY.json').write_text(json.dumps(out, indent=2, default=str))
    (sd / 'SUMMARY.txt').write_text('\n'.join(['SUMMARY ' + l for l in lines] +
                                              ['VALIDATE %(status)s %(check)s got=%(got)s expected=%(expected)s %(note)s' % c
                                               for c in checks.items]) + '\n')
    return 1 if (args.validate and checks.failed) else 0


if __name__ == '__main__':
    raise SystemExit(main())
