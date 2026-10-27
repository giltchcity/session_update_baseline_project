"""Evaluation of a TESSE office map with the official Khronos pipeline (official_native_eval_run.py) and the TESSE
config of tesse_eval.py (configs/khronos/eval_office_local.yaml; background thresholds include 0.2 m):
  results/{static_objects,background_mesh,dynamic_objects}.csv and map_timestamps.txt, the official result tables;
  TABLES_4D.txt, the official tables.py (unchanged) 4D averages of the paper's metrics;
  WEIGHTED.json, the same metrics, each map at its own time, weighted by how long it is the current map (the time to
    the next map; the last map, the interval before it);
  weighted_nr/, the online object P/R/F1 duration-weighted with the NR correction (eval_fast's object_weighted_nr.py;
    the TESSE person class is 17, Humans).
ours MAP OUT [NEAR_OUT]: each snapshot of a map (packed .4dmap.zpk, or plain .4dmap as eval/chain_fill writes) scored alone, as eval_fast scores a real session's
  snapshots (the official pipeline evaluates each map at its own time), the rows numbered as one map; with NEAR_OUT,
  only the snapshots closest in time to that evaluation's maps.
khronos MAP.4dmap OUT: the official pipeline on an official raw map, all its maps; NR from that run's own object export
  (eval_visualization/objects.csv: present, label; no mesh sizes there, so empty-mesh nodes are not counted).
ours4d MAP OUT [NEAR_OUT]: as ours, but each map is queried at every map time up to its own, the paper's 4D
  definition (eq. 20), with the evaluator copy eval_tools/khronos_eval_query_earlier (the official loop queried only a
  map's own time); every query is a row.
tables OUT: only the tables (TABLES_4D.txt, WEIGHTED.json) of an evaluation made by one of the above.
--scene apartment (first argument): the apartment scene instead (configs/khronos/eval_apartment_local.yaml, the same
  config on its ground truth; person class 20, Humans).
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path('/home/jixian/Desktop/FT')
sys.path.insert(0, str(ROOT / 'eval_fast/vendor/pack'))
import zpk_snapshot as zs  # noqa: E402

PY = '/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python'
SCENE = 'office'
if sys.argv[1:2] == ['--scene']:
    SCENE = sys.argv[2]
    del sys.argv[1:3]
CONFIG = ROOT / f'configs/khronos/eval_{SCENE}_local.yaml'
CONFIG = Path(os.environ.get('TESSE_EVAL_CONFIG', CONFIG))   # e.g. eval_SCENE_objects_only_local.yaml
PERSON = {'office': '17', 'apartment': '20'}[SCENE]  # Humans in Khronos's tesse_cd_SCENE label space
VENDOR = ROOT / 'eval_fast/vendor'
PLOTTING = ROOT / 'build_artifacts/official_khronos_63faadde_abc/src/khronos_eval/plotting'
METRICS = ['Accuracy@0.2', 'Completeness@0.2', 'F1@0.2', 'ObjectPrecision', 'ObjectRecall', 'ObjectF1',
           'DynamicPrecision', 'DynamicRecall', 'DynamicF1', 'ChangePrecision', 'ChangeRecall', 'ChangeF1']
CANONICAL = ['bash', '-c', 'source "$1" >/dev/null 2>&1 || exit 90; shift; exec "$@"', 'tesse-eval',
             str(ROOT / 'eval_fast/runtime_khronos_env.sh')]
TABLES = ('static_objects.csv', 'background_mesh.csv', 'dynamic_objects.csv')
TABLES = tuple(os.environ.get('TESSE_EVAL_TABLES', ','.join(TABLES)).split(','))   # the tables the config produces
QUERY_EARLIER = ROOT / 'wt_takeover_runs/eval_tools/khronos_eval_query_earlier/lib'


def official(sliced, out, query_times=None):
    """The official pipeline on a map; with query_times, the evaluator copy querying those times up to each map's."""
    if query_times is None:
        subprocess.run(['/usr/bin/python3', str(VENDOR / 'object/official_native_eval_run.py'), '--map', str(sliced),
                        '--config', str(CONFIG), '--output', str(out), '--only-final-argument', 'false'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    out.mkdir()
    (out / 'final.4dmap').symlink_to(sliced)
    (out / 'experiment_log.txt').write_text('[INFO] Evaluation input directory only.\n')
    (out / 'query_times.txt').write_text(''.join(f'{t}\n' for t in query_times))
    subprocess.run(['bash', '-c', 'source "$1" >/dev/null 2>&1; export LD_LIBRARY_PATH="$2:$LD_LIBRARY_PATH"; shift 2; exec "$@"',
                    'tesse-eval', str(ROOT / 'eval_fast/official/environment.sh'), str(QUERY_EARLIER),
                    str(ROOT / 'eval_fast/official/install/lib/khronos_eval/exp_pipeline'), str(CONFIG), str(out),
                    'false', 'true', 'false'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   env=dict(os.environ, KHRONOS_EVAL_QUERY_TIMES=str(out / 'query_times.txt')))


def score_one(job):
    """One snapshot scored alone (official tables, timeline line); run in a worker process."""
    n, i, source, query_times = job
    with tempfile.TemporaryDirectory(prefix='tesse-eval-', dir='/dev/shm') as tmp:
        sliced = Path(tmp) / 'one.4dmap'
        zs.cut_many(source, [i], out=str(sliced), work=tmp)
        official(sliced, Path(tmp) / 'official', query_times)
        tables_ = {name: (Path(tmp) / 'official/results' / name).read_text().splitlines() for name in TABLES}
        subprocess.run([*CANONICAL, str(VENDOR / 'timeline/export_timeline'), str(sliced), f'{tmp}/timeline'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return n, tables_, json.loads(Path(f'{tmp}/timeline/snapshots.jsonl').read_text())


def score_snapshots(source, out, near, earlier=False):
    """The official tables of the chosen snapshots, each scored alone, numbered as maps 0..n-1; snapshots.jsonl.
    earlier: each map also queried at the chosen snapshots' earlier times."""
    stamps = zs.stamps(source)
    if near:
        targets = [int(x) for x in (near / 'results/map_timestamps.txt').read_text().split()]
        chosen = sorted({min(range(len(stamps)), key=lambda i: abs(stamps[i] - t)) for t in targets})
    else:
        chosen = list(range(len(stamps)))
    (out / 'results').mkdir(parents=True)
    jobs = [(n, i, source, [stamps[k] for k in chosen[:n + 1]] if earlier else None) for n, i in enumerate(chosen)]
    done = {}
    with ProcessPoolExecutor(int(os.environ.get('TESSE_EVAL_JOBS', '8'))) as pool:   # snapshots are independent
        for n, tables_, line in pool.map(score_one, jobs):
            done[n] = (tables_, line)
            print(f'{len(done)}/{len(chosen)} snapshot {chosen[n]}', flush=True)
    headers, rows = {}, {name: [] for name in TABLES}
    with (TIMELINE).open('w') as timeline:
        for n, i in enumerate(chosen):
            tables_, line = done[n]
            for name in TABLES:
                lines = tables_[name]
                assert headers.setdefault(name, lines[0]) == lines[0] and all(l.startswith('0,') for l in lines[1:]), (i, name)
                renamed = [f'{n},' + l[2:] for l in lines[1:]]
                rows[name] += renamed * (n + 1) if name == 'static_objects.csv' and not earlier else renamed
            assert rows['static_objects.csv'][-1].split(',')[1] == str(stamps[i]), i
            timeline.write(json.dumps(dict(line, index=n)) + '\n')
    for name in TABLES:
        (out / 'results' / name).write_text('\n'.join([headers[name]] + rows[name]) + '\n')
    (out / 'results/map_timestamps.txt').write_text(''.join(f'{stamps[i]}\n' for i in chosen))
    (out / 'SNAPSHOTS.json').write_text(json.dumps(dict(map=str(source), snapshots=chosen), indent=1) + '\n')


def khronos_snapshots(out):
    """snapshots.jsonl for NR from the official run's own object export: per map, its present nodes and labels."""
    maps = {}
    with (out / 'eval_visualization/objects.csv').open() as f:
        for row in csv.DictReader(f):
            if row['IsGT'] == '0':
                maps.setdefault(int(row['MapName']), dict(stamp_ns=int(row['QueryTime']), objects={}))
                maps[int(row['MapName'])]['objects'][row['ID']] = dict(present=row['Present'] == '1',
                                                                      semantic=int(row['Label']), mesh_vertices=1)
    stamps = [int(x) for x in (out / 'results/map_timestamps.txt').read_text().split()]
    with (TIMELINE).open('w') as timeline:
        for i, stamp in enumerate(stamps):
            snap = maps.get(i, dict(stamp_ns=stamp, objects={}))
            assert snap['stamp_ns'] == stamp, (i, snap['stamp_ns'], stamp)
            timeline.write(json.dumps(dict(index=i, stamp_ns=stamp, objects=list(snap['objects'].values()))) + '\n')


def weighted(out):
    """The paper metrics, each map at its own time, weighted by how long it is the current map."""
    sys.path.insert(0, str(PLOTTING))
    import utils  # the official tables.py readers and metric definitions
    data = utils.parse_table_data_simple(str(out.parent), [out.name])[0]
    utils.augment_object_metrics(data)
    utils.augment_dynamic_metrics(data)
    utils.augment_bg_metrics(data)
    stamps = [int(x) for x in (out / 'results/map_timestamps.txt').read_text().split()]
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    weights = gaps + gaps[-1:] if gaps else [1]
    result = {}
    for metric in METRICS:
        values = []
        for i, stamp in enumerate(stamps):
            per_query = data[metric].get(i, data[metric].get(str(i), {}))
            value = per_query.get(str(stamp), per_query.get(stamp, next(iter(per_query.values()), float('nan'))))
            values.append(float(value))
        live = [(v, w) for v, w in zip(values, weights) if v == v]
        result[metric] = round(100 * sum(v * w for v, w in live) / sum(w for _, w in live), 2) if live else None
    (out / 'WEIGHTED.json').write_text(json.dumps(dict(maps=len(stamps), weighted=result), indent=1) + '\n')
    return result


def tables(out):
    """The official tables.py 4D averages and the duration-weighted metrics of an evaluated directory."""
    table = subprocess.run([PY, str(PLOTTING / 'tables.py'), str(out.parent), '-m', f'{out.name}:{out.name}'],
                           check=True, capture_output=True, text=True, cwd=PLOTTING).stdout
    (out / 'TABLES_4D.txt').write_text(table)
    print(table)
    print('duration-weighted', weighted(out))
    print('online object (weighted+NR)', json.loads((out / 'weighted_nr/OBJECT_WEIGHTED_NR.json').read_text())['duration_weighted_nr'])


method, out = sys.argv[1], Path(sys.argv[-1] if sys.argv[1] == 'tables' else sys.argv[3]).resolve()
# The per-snapshot export only feeds the NR count below: it lives in memory (/dev/shm) and is dropped at the end.
TIMELINE = Path(tempfile.mkdtemp(prefix='tesse-eval-timeline-', dir='/dev/shm')) / 'snapshots.jsonl'
if method != 'tables':
    source = Path(sys.argv[2]).resolve()
    if method in ('ours', 'ours4d'):
        out.mkdir(parents=True, exist_ok=False)
        score_snapshots(str(source), out, Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else None, method == 'ours4d')
    else:
        official(source, out)                   # the runner creates OUT
        (out / 'final.4dmap').unlink()          # the runner's link to the input map
        khronos_snapshots(out)
    subprocess.run([PY, str(VENDOR / 'object/official_native_eval_postprocess.py'), str(out / 'results/static_objects.csv'),
                    '--output', str(out / 'explanation')], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([PY, str(VENDOR / 'harness/object_weighted_nr.py'), '--rows', str(out / 'explanation/native_unique_online_rows.csv'),
                    '--native-summary', str(out / 'explanation/NATIVE_ONLINE_SUMMARY.json'), '--snapshots', str(TIMELINE),
                    '--person-label', PERSON, '--out', str(out / 'weighted_nr')], check=True, stdout=subprocess.DEVNULL)
tables(out)
shutil.rmtree(TIMELINE.parent)
