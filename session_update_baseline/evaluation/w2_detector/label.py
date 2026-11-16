"""Correct or false: every absence commitment of a replayed variant (replay.py) against the ground truth.

A commitment on state record r (object i) at time t is judged against r's reference moment t0: the start of the
latest round of this visit in which the object stood in place on r (identified=1) before t (the previous round of the
run: a round judges the frames since it); for a state inherited from the previous visit, always the end of that visit
(its surface is the earlier visit's: a later partial overlap with the moved object does not move it); a state of this
visit with no scored round before t has no known place (uncertain unless its object left the visit).
  correct    the object is no longer where it was at t0 (moved or removed by t), or absent from the visit
  false      the object is still there at t
  uncertain  an inherited state of an object that did not change between the visits but that this visit never finds in
             place on r (r may be a stale state of the earlier map), or (real) t or t0 inside an unobserved change
             window of the ground truth (cabinet I2)
Synthetic: exact Isaac poses per frame (temporal_gt.npz). Real: results/abc_eval_final_baseline/events.json (D3 between
visits) and the cabinet's in-visit state intervals (gt_{b,c}_latest; every other object is static within a visit).
Per record only its first commitment counts (the replay is exact up to it). A record has a change to detect when some
round of it was scored while its object was away; recall = records detected / records with a change to detect;
delay = commitment time minus the first such round.
usage: label.py   (all variants, both datasets)  -> table on stdout and results.json"""
import json, math, statistics, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import replay as R

FT = Path('/home/jixian/Desktop/FT'); LOGS = FT / 'wt_takeover_runs/motion_diag_20261112_summary/diag'
RUNS = dict(synthetic=['synthetic_a', 'synthetic_b'], real=['real_a', 'real_b', 'real_c'])
VARIANTS = [('method (eta=99)', {}), ('eta=3', dict(eta=3)), ('eta=9', dict(eta=9)), ('eta=19', dict(eta=19)),
            ('eta=49', dict(eta=49)), ('eta=999', dict(eta=999)), ('eta=9999', dict(eta=9999)),
            ('eta=99999', dict(eta=99999)), ('weights w=1', dict(weights='one')), ('single round', dict(single=True)),
            ('fixed Beta', dict(beta='fixed')), ('no cross-visit prior', dict(prior=False))]

# ---- synthetic truth: Isaac poses ----
SYN = None
def syn():
    global SYN
    if SYN is None:
        D = FT / 'datasets/synthetic_ab'
        G = {s: dict(np.load(D / f'session_{s}/gt/temporal_gt.npz')) for s in 'ab'}
        SYN = dict(G=G, IX={s: {int(i): k for k, i in enumerate(G[s]['instance_ids'])} for s in 'ab'},
                   O={'a': 1_000_000_000, 'b': 141_000_000_000})
    return SYN

def syn_pose(s, i, stamp):
    S = syn(); g = S['G'][s]
    if i not in S['IX'][s]: return None
    f = int(np.clip(round((stamp - S['O'][s]) / 1e9 * 30), 0, len(g['time_s']) - 1)); k = S['IX'][s][i]
    return g['T_world_object'][f, k] if g['render_visible'][f, k] else None

def syn_truth(s, i, t0_stamp, t0_session, t, seen_in_place):
    ref = syn_pose(t0_session, i, t0_stamp) if t0_stamp is not None else syn_pose('a', i, 10**20)
    now = syn_pose(s, i, t)
    if now is None: return 'correct'
    if ref is None: return 'uncertain'
    if np.max(np.abs(now - ref)) > 1e-3: return 'correct'
    return 'false' if t0_session == s or seen_in_place else 'uncertain'

# ---- real truth ----
START = {'a': 1786279210201000000, 'b': 1786302302620000000, 'c': 1787394423387000000}
EVENTS = json.loads((FT / 'results/abc_eval_final_baseline/events.json').read_text())
D3 = {(e['session'].lower(), e['object_id']) for e in EVENTS if e['scope'] == 'D3' and e['expected_disappeared']}
# cabinet I2 within a visit: (state, certain from s, certain through s); gaps between are unobserved change windows
I2 = {'a': [('A', 94.6333, 1e9)], 'b': [('start', 0, 34.1), ('final', 57.466656, 1e9)],
      'c': [('start', 0, 35.0), ('moving', 35.0 + 1e-3, 39.0 - 1e-3), ('mid', 39.0, 100.499978), ('final', 125.866645, 1e9)]}

def i2_state(s, sec):
    for name, a, b in I2[s]:
        if a - 1e-6 <= sec <= b + 1e-6: return name
    return None

PRESENT = {s: {o['physical_object_id'] for o in json.loads((FT / f'results/current_version_20260921/real_abc/objects/gt_{s}_latest/gt_payload.json').read_text())['objects']} for s in 'abc'}

def real_truth(s, i, t0_stamp, t0_session, t, seen_in_place):
    if i not in PRESENT[s]: return 'correct'
    if t0_session != s:   # inherited, not yet found in place: the previous visit's final state
        return 'correct' if (s, i) in D3 else 'false' if seen_in_place else 'uncertain'
    if i != 2: return 'false'   # every other object is static within a visit
    a, b = i2_state(s, (t0_stamp - START[s]) / 1e9), i2_state(s, (t - START[s]) / 1e9)
    if a is None or b is None: return 'uncertain'
    return 'false' if a == b else 'correct'

PREV = {'b': 'a', 'c': 'b'}

def judge(kind, run, rs):
    s = run[-1]; truth = syn_truth if kind == 'synthetic' else real_truth
    by = defaultdict(list)
    for r in rs: by[r['record']].append(r)
    stamps = sorted({r['stamp'] for r in rs})
    def start(st):   # the previous round of the run: a round judges the frames since it
        k = np.searchsorted(stamps, st) - 1
        return stamps[k] if k >= 0 else st
    out = []
    for rec, rr in by.items():
        rr.sort(key=lambda r: r['stamp'])
        seen = any(r['identified'] for r in rr)
        def reference(t):
            if rr[0]['inherited'] and s in PREV: return (None if kind == 'synthetic' else 0), PREV[s]
            ins = [r['stamp'] for r in rr if r['identified'] and r['stamp'] < t]
            if ins: return start(ins[-1]), s
            before = [r['stamp'] for r in rr if r['stamp'] < t]
            return (start(before[0]), s) if before else (None, None)
        def lab(r):
            t0, s0 = reference(r['stamp'])
            if s0 is None:   # no known place: correct only when the object is gone from the visit
                gone = syn_pose(s, r['inst'], r['stamp']) is None if kind == 'synthetic' else r['inst'] not in PRESENT[s]
                return 'correct' if gone else 'uncertain'
            return truth(s, r['inst'], t0, s0, r['stamp'], seen)
        labels = [lab(r) for r in rr]
        away = [r['stamp'] for r, l in zip(rr, labels) if l == 'correct']
        first = next(((r, l) for r, l in zip(rr, labels) if r['commit']), None)
        out.append(dict(run=run, record=rec, inst=rr[0]['inst'], inherited=rr[0]['inherited'], change=bool(away),
                        commit=first[1] if first else None,
                        delay_s=(first[0]['stamp'] - away[0]) / 1e9 if first and first[1] == 'correct' and away else None))
    return out

if __name__ == '__main__':
    raw = {run: R.rounds(LOGS / run / 'control/logs/khronos.log') for runs in RUNS.values() for run in runs}
    results = {}
    print(f"{'variant':22s} {'dataset':9s} {'commits':>7s} {'correct':>7s} {'false':>5s} {'uncert':>6s} {'changes':>7s} {'recall':>6s} {'med delay s':>11s}")
    for name, opts in VARIANTS:
        for kind, runs in RUNS.items():
            recs = []
            for run in runs:
                rs = R.replay([dict(r) for r in raw[run]], **opts)
                recs += judge(kind, run, rs)
            c = [x for x in recs if x['commit']]
            correct = [x for x in c if x['commit'] == 'correct']; false = [x for x in c if x['commit'] == 'false']
            unc = [x for x in c if x['commit'] == 'uncertain']; changes = [x for x in recs if x['change']]
            det = [x for x in changes if x['commit'] == 'correct']
            d = [x['delay_s'] for x in correct if x['delay_s'] is not None]
            row = dict(commits=len(c), correct=len(correct), false=len(false), uncertain=len(unc), changes=len(changes),
                       recall=len(det) / len(changes) if changes else None, median_delay_s=statistics.median(d) if d else None,
                       false_records=[(x['run'], x['inst'], x['record']) for x in false])
            results[f'{name} | {kind}'] = row
            print(f"{name:22s} {kind:9s} {len(c):7d} {len(correct):7d} {len(false):5d} {len(unc):6d} {len(changes):7d} "
                  f"{(row['recall'] or 0) * 100:5.1f}% {row['median_delay_s'] if row['median_delay_s'] is not None else float('nan'):11.1f}")
    (Path('/home/jixian/Desktop/FT/wt_takeover_runs/w2_detector_replay_20261116') / 'results.json').write_text(json.dumps(results, indent=1) + '\n')
