"""Key ablations of the absence test (Eq. 2) at the level of the map's decision: a state closes on the absence path
when the test commits and the same round's rays confirm (C > S); its other paths (seen elsewhere) do not use Eq. 2 and
are the same in every variant. Exact replay of the logged evidence (replay.py), every first close per state judged
against the ground truth (label.py). Writes key_ablations.json and prints the table.
usage: key_ablations.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import replay as R, label as Lb

VARIANTS = [('Ours (eta=99)', {}), ('eta=3', dict(eta=3)), ('eta=9', dict(eta=9)), ('eta=999', dict(eta=999)),
            ('eta=9999', dict(eta=9999)), ('No weighting (w=1)', dict(weights='one')),
            ('Fixed Beta (no online model)', dict(beta='fixed')), ('Eq. 2 off', dict(off=True))]
raw = {run: R.rounds(Lb.LOGS / run / 'control/logs/khronos.log') for runs in Lb.RUNS.values() for run in runs}
out = {}
for name, opts in VARIANTS:
    for kind, runs in Lb.RUNS.items():
        recs = []
        for run in runs:
            rs = R.replay([dict(r) for r in raw[run]], **opts)
            for r in rs: r['commit'] = r['close']   # judge the first close instead of the first commitment
            recs += Lb.judge(kind, run, rs)
        changes = [x for x in recs if x['change']]
        row = dict(correct=sum(x['commit'] == 'correct' for x in recs), false=sum(x['commit'] == 'false' for x in recs),
                   uncertain=sum(x['commit'] == 'uncertain' for x in recs), changes=len(changes),
                   detected=sum(x['commit'] == 'correct' for x in changes))
        row['recall'] = row['detected'] / row['changes']
        out[f'{name} | {kind}'] = row
print(f"{'variant':30s} | {'synthetic: correct / false deletions, recall':44s} | real: correct / false deletions, recall")
for name, _ in VARIANTS:
    s, r = out[f'{name} | synthetic'], out[f'{name} | real']
    print(f"{name:30s} | {s['correct']:3d} / {s['false']:2d}, {100 * s['recall']:5.1f}% ({s['detected']}/{s['changes']}){'':16s} | "
          f"{r['correct']:3d} / {r['false']:2d}, {100 * r['recall']:5.1f}% ({r['detected']}/{r['changes']})  uncertain {r['uncertain']}")
(Path('/home/jixian/Desktop/FT/wt_takeover_runs/w2_detector_replay_20261116') / 'key_ablations.json').write_text(json.dumps(out, indent=1) + '\n')
