#!/usr/bin/env python3
"""The paper's metrics (Khronos's official evaluator and table definitions) on a 3RScan visit chain, with the ground
truth of threerscan_khronos_gt.py (configs/khronos/eval_3rscan_ROOM_local.yaml). Every snapshot of the method's maps is
cut out and evaluated alone at its own time by the official pipeline (official_native_eval_run.py), as
wt_takeover_runs/analysis/tesse_online_eval.py scores a TESSE map's snapshots; our chained maps are one per visit, so
a map cannot be queried at an earlier visit's time and every snapshot is taken at its own time, for both methods.
Maps: ours = METHOD_DIR/visit_K/final.4dmap.zpk for each visit, joined by eval/chain_fill (one map of the chain's
snapshots; an object node without a mesh takes its last earlier mesh, an object first present in a later visit its
first present stamp as first observed: what the snapshots already hold, nothing rerun; in /dev/shm, removed after);
khronos = METHOD_DIR/state/final.4dmap.zpk; timeline = METHOD_DIR/visit_K/snapshots.jsonl for each visit
(any method's object timeline in the eval/chain_fill/timeline_export format), completed by eval/timeline_complete.py
(one set of rules for every method) and written as a map by eval/chain_fill/timeline_to_4dmap (the layer layout of
this room's ours_290e110/visit_0 map; objects only).
Output METHOD_DIR/khronos_metrics/: results/{static_objects,background_mesh,dynamic_objects}.csv and
map_timestamps.txt (the snapshots numbered 0..n-1 over the chain), METRICS.json: per metric (Object P/R/F1,
Change P/R/F1, the official utils' definitions; no background metric) the mean over the snapshots, the mean weighted
by how long each snapshot is the current map (to the next one in its visit; a visit's last one, the gap before it),
and each visit's last snapshot.
usage: threerscan_khronos_eval.py ROOM ours|khronos|timeline METHOD_DIR
"""
import json, os, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

FT = Path('/home/jixian/Desktop/FT')
sys.path.insert(0, str(FT / 'eval_fast/vendor/pack'))
import zpk_snapshot as zs  # noqa: E402
PLOTTING = FT / 'build_artifacts/official_khronos_63faadde_abc/src/khronos_eval/plotting'
RUNNER = FT / 'eval_fast/vendor/object/official_native_eval_run.py'
TABLES = ('static_objects.csv', 'background_mesh.csv', 'dynamic_objects.csv')
METRICS = ['ObjectPrecision', 'ObjectRecall', 'ObjectF1', 'ChangePrecision', 'ChangeRecall', 'ChangeF1']  # no background (user)

room, method, mdir = sys.argv[1], sys.argv[2], Path(sys.argv[3]).resolve()
config = FT / f'configs/khronos/eval_3rscan_{room}_local.yaml'
work = tempfile.TemporaryDirectory(prefix='3rscan-chain-', dir='/dev/shm')
if os.environ.get('THREERSCAN_OBJECTS_ONLY') == '1':
    # METRICS read only the object table (objects and changes); the background-mesh and dynamic-object
    # evaluations, whose tables nothing here reads, are switched off.
    text = config.read_text().replace('evaluate_mesh: true', 'evaluate_mesh: false')
    text = text.replace('evaluate_dynamic_objects: true', 'evaluate_dynamic_objects: false')
    config = Path(work.name) / config.name; config.write_text(text)
    TABLES = ('static_objects.csv',)
visits = json.load((FT / 'datasets/3rscan/mapping_configs' / room / 'chain.json').open())['visits']
if method == 'ours':
    joined = Path(work.name) / 'chain.4dmap'
    fill = subprocess.run(['bash', '-c', 'set +u; source /opt/ros/jazzy/setup.bash; source /home/jixian/ros2_ws/install/setup.bash; '
                           f'export LD_LIBRARY_PATH={FT}/wt_290e110_build/canonical/install/lib:$LD_LIBRARY_PATH; exec "$@"', 'chain-fill',
                           str(FT / 'eval/chain_fill/chain_fill'), str(joined),
                           *[str(mdir / f'visit_{k}/final.4dmap.zpk') for k in range(len(visits))]],
                          check=True, capture_output=True, text=True)
    print(fill.stdout.strip().splitlines()[-1], flush=True)
    sources = [(None, joined)]
elif method == 'timeline':
    env = ('set +u; source /opt/ros/jazzy/setup.bash; source /home/jixian/ros2_ws/install/setup.bash; '
           f'export LD_LIBRARY_PATH={FT}/wt_290e110_build/canonical/install/lib:$LD_LIBRARY_PATH; exec "$@"')
    combined = Path(work.name) / 'timeline.jsonl'; joined = Path(work.name) / 'timeline.4dmap'
    done = subprocess.run([sys.executable, str(FT / 'eval/timeline_complete.py'), str(combined),
                           *[str(mdir / f'visit_{k}/snapshots.jsonl') for k in range(len(visits))]],
                          check=True, capture_output=True, text=True)
    print(done.stdout.strip().splitlines()[-1], flush=True)
    template = FT / 'wt_takeover_runs/3rscan' / room / 'ours_290e110/visit_0/final.4dmap.zpk'
    conv = subprocess.run(['bash', '-c', env, 'timeline-to-4dmap', str(FT / 'eval/chain_fill/timeline_to_4dmap'),
                           str(template), str(combined), str(joined)], check=True, capture_output=True, text=True)
    print(conv.stdout.strip().splitlines()[-1], flush=True)
    sources = [(None, joined)]
else:
    sources = [(None, mdir / 'state/final.4dmap.zpk')]
out = mdir / 'khronos_metrics'; (out / 'results').mkdir(parents=True, exist_ok=False)
headers, rows, stamps, visit_of = {}, {name: [] for name in TABLES}, [], []
n = 0
def score_one(job):
    """The official tables of one snapshot, scored alone in /dev/shm."""
    src, i = job
    with tempfile.TemporaryDirectory(prefix='3rscan-khronos-eval-', dir='/dev/shm') as tmp:
        sliced = Path(tmp) / 'one.4dmap'; zs.cut_many(str(src), [i], out=str(sliced), work=tmp)
        subprocess.run(['/usr/bin/python3', str(RUNNER), '--map', str(sliced), '--config', str(config), '--output',
                        f'{tmp}/official', '--only-final-argument', 'false'], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {name: (Path(tmp) / 'official/results' / name).read_text().splitlines() for name in TABLES}


jobs = [(k, src, i, stamp) for k, src in sources for i, stamp in enumerate(zs.stamps(str(src)))]
with ProcessPoolExecutor(int(os.environ.get('THREERSCAN_JOBS', '1'))) as pool:   # snapshots are independent
    tables = list(pool.map(score_one, [(src, i) for _, src, i, _ in jobs]))
for (k, src, i, stamp), result in zip(jobs, tables):
    v = k if k is not None else (stamp - 1000000000) // 200000000000
    for name in TABLES:
        lines = result[name]
        assert headers.setdefault(name, lines[0]) == lines[0] and all(l.startswith('0,') for l in lines[1:]), (src, i, name)
        rows[name] += [f'{n},' + l[2:] for l in lines[1:]]
    stamps.append(stamp); visit_of.append(int(v)); n += 1
    print(f'{room} {method} snapshot {n} (visit {v}, {stamp / 1e9:.1f} s)', flush=True)
for name in TABLES:
    (out / 'results' / name).write_text('\n'.join([headers[name]] + rows[name]) + '\n')
(out / 'results/map_timestamps.txt').write_text(''.join(f'{s}\n' for s in stamps))

sys.path.insert(0, str(PLOTTING))
import utils  # noqa: E402  the official tables.py readers and metric definitions
data = utils.parse_table_data_simple(str(mdir), [out.name])[0]
utils.augment_object_metrics(data)
weights = []
for j in range(n):                                   # time this snapshot is the current map within its visit
    same = [t for t, v in zip(stamps, visit_of) if v == visit_of[j]]
    pos = same.index(stamps[j])
    weights.append(same[pos + 1] - same[pos] if pos + 1 < len(same) else (same[pos] - same[pos - 1] if pos else 1))
def value(metric, j):
    per_query = data[metric].get(j, data[metric].get(str(j), {}))
    v = per_query.get(str(stamps[j]), per_query.get(stamps[j], next(iter(per_query.values()), float('nan'))))
    return float(v)
result = {}
for metric in METRICS:
    vals = [value(metric, j) for j in range(n)]
    live = [(x, w) for x, w in zip(vals, weights) if x == x]
    ends = {v: vals[max(j for j in range(n) if visit_of[j] == v)] for v in sorted(set(visit_of))}
    result[metric] = dict(mean=round(100 * sum(x for x, _ in live) / len(live), 2) if live else None,
                          weighted=round(100 * sum(x * w for x, w in live) / sum(w for _, w in live), 2) if live else None,
                          visit_end={str(v): (round(100 * x, 2) if x == x else None) for v, x in ends.items()})
(out / 'METRICS.json').write_text(json.dumps(dict(room=room, method=method, snapshots=n, metrics=result), indent=1) + '\n')
work.cleanup()
print(room, method, {m: result[m]['mean'] for m in ('ObjectF1', 'ChangeF1')}, 'weighted',
      {m: result[m]['weighted'] for m in ('ObjectF1', 'ChangeF1')})
