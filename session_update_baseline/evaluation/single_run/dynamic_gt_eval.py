"""Khronos's official dynamic-object metric on the real and synthetic single runs (user 11-01).
gt  KIND: the official ground truth of eval/khronos4d_gt/KIND plus one dynamic object per motion track (KhronosObjectAttributes
     with trajectory_timestamps / trajectory_positions, as the TESSE ground truth's dynamic objects): real = the people
     (eval/real_motion_gt/person_tracks_*.csv), synthetic = the objects moved in view (eval/synthetic_motion_gt/
     moving_objects_*.csv, Isaac). Writes eval/khronos4d_gt/KIND_dynamic/gt_dsg_consolidated.json.
map KIND MAP.4dmap.zpk OUT: the official evaluator (eval_fast/official exp_pipeline, the 4D copy querying every earlier
     map time, as the TESSE table) with the background evaluation off (the objects stay on: only they load the
     ground truth), every snapshot scored alone; the dynamic rows
     (Name = map, Query) into OUT/dynamic_objects.csv, and the official 4D average (tables.py: per map x query cell
     P / R / F1 from NumObjDetected / Hallucinated / Missed, then the mean over cells) into OUT/DYNAMIC_4D.json.
One ground truth per visit (11-01: with all visits in one, a B map was charged A's people as missed):
usage: dynamic_gt_eval.py gt real|synthetic SESSION | dynamic_gt_eval.py map real|synthetic SESSION MAP OUT"""
import copy, csv, json, os, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
FT = Path('/home/jixian/Desktop/FT'); R = FT / 'wt_takeover_runs'
sys.path.insert(0, str(FT / 'eval_fast/vendor/pack'))
import zpk_snapshot as zs
QUERY_EARLIER = R / 'eval_tools/khronos_eval_query_earlier/lib'


def tracks(kind, sessions):
    if kind == 'real':
        for s in sessions:
            rows = list(csv.DictReader((R / f'eval/real_motion_gt/person_tracks_{s}.csv').open()))
            for k in sorted({r['track'] for r in rows}, key=int):
                tr = [r for r in rows if r['track'] == k]
                yield f'person_{s}_{k}', 12, [int(r['stamp_ns']) for r in tr], [[float(r[a]) for a in 'xyz'] for r in tr]
    else:
        for s in sessions:
            rows = list(csv.DictReader((R / f'eval/synthetic_motion_gt/moving_objects_{s}.csv').open()))
            for inst in sorted({int(r['instance']) for r in rows}):
                tr = [r for r in rows if int(r['instance']) == inst]
                yield f'moving_{s}_{inst}', 4294967295, [int(r['stamp_ns']) for r in tr], [[float(r[a]) for a in 'xyz'] for r in tr]


def build_gt(kind, session):
    src = FT / f'eval/khronos4d_gt/{kind}/gt_dsg_consolidated.json'; d = json.loads(src.read_text())
    template = next(n for n in d['nodes'] if n['layer'] == 2)
    next_id = max(n['id'] for n in d['nodes']) + 1
    for name, semantic, stamps, positions in tracks(kind, session):
        n = copy.deepcopy(template); a = n['attributes']; n['id'] = next_id; next_id += 1
        P = np.array(positions); c = P[0]
        a.update(name=name, semantic_label=semantic, position=list(c), trajectory_timestamps=stamps, trajectory_positions=positions,
                 first_observed_ns=[stamps[0]], last_observed_ns=[stamps[-1]], dynamic_object_points=[], is_active=False)
        a['bounding_box'] = dict(a['bounding_box'], world_P_center=list(c), dimensions=[0.5, 0.5, 1.7 if kind == 'real' else 0.8])
        a['mesh'] = dict(a['mesh'], points=[], **({'colors': []} if 'colors' in a['mesh'] else {}),
                         **({'labels': []} if 'labels' in a['mesh'] else {}), **({'stamps': []} if 'stamps' in a['mesh'] else {}),
                         **({'first_seen_stamps': []} if 'first_seen_stamps' in a['mesh'] else {}), **({'faces': []} if 'faces' in a['mesh'] else {}))
        d['nodes'].append(n)
        print(f'{name}: {len(stamps)} trajectory points, {(stamps[-1] - stamps[0]) / 1e9:.1f} s', flush=True)
    out = FT / f'eval/khronos4d_gt/{kind}_dynamic_{session}'; out.mkdir(parents=True, exist_ok=True)
    (out / 'gt_dsg_consolidated.json').write_text(json.dumps(d))
    for f in ('gt_changes.csv', 'gt_background.ply'):
        if (src.parent / f).exists() and not (out / f).exists(): (out / f).symlink_to(src.parent / f)
    print('written', out / 'gt_dsg_consolidated.json')


def config(kind, session):
    base = (FT / f'configs/khronos/eval_chain_{kind}_local.yaml').read_text()
    text = base.replace(f'eval/khronos4d_gt/{kind}/gt_dsg_consolidated.json', f'eval/khronos4d_gt/{kind}_dynamic_{session}/gt_dsg_consolidated.json')
    assert text != base
    for key in ('evaluate_mesh',):   # the objects stay on: the pipeline loads the ground truth DSG only there
        assert text.count(f'{key}: true') == 1, key
        text = text.replace(f'{key}: true', f'{key}: false')
    assert text.count('evaluate_dynamic_objects: true') == 1
    path = R / f'eval/khronos4d_gt_{kind}_dynamic_{session}.yaml'; path.write_text(text); return path


def score_one(job):
    i, n, source, queries, cfg = job
    with tempfile.TemporaryDirectory(prefix='dyn-eval-', dir='/dev/shm') as tmp:
        sliced = Path(tmp) / 'one.4dmap'; zs.cut_many(source, [i], out=str(sliced), work=tmp)
        out = Path(tmp) / 'official'; out.mkdir(); (out / 'final.4dmap').symlink_to(sliced)
        (out / 'experiment_log.txt').write_text('[INFO] Evaluation input directory only.\n')
        (out / 'query_times.txt').write_text(''.join(f'{t}\n' for t in queries))
        subprocess.run(['bash', '-c', 'source "$1" >/dev/null 2>&1; export LD_LIBRARY_PATH="$2:$LD_LIBRARY_PATH"; shift 2; exec "$@"',
                        'dyn-eval', str(FT / 'eval_fast/official/environment.sh'), str(QUERY_EARLIER),
                        str(FT / 'eval_fast/official/install/lib/khronos_eval/exp_pipeline'), str(cfg), str(out), 'false', 'true', 'false'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(os.environ, KHRONOS_EVAL_QUERY_TIMES=str(out / 'query_times.txt')))
        lines = (out / 'results/dynamic_objects.csv').read_text().splitlines()
    return n, lines[0], [f'{n},' + l.split(',', 1)[1] for l in lines[1:]]


def metrics(tp, fp, fn):
    p = np.nan if tp + fp == 0 else tp / (tp + fp); r = np.nan if tp + fn == 0 else tp / (tp + fn)
    f = np.nan if not (p + r) or np.isnan(p + r) else 2 * p * r / (p + r)   # the official compute_metrics_single
    return p, r, f


def score_map(kind, session, source, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); cfg = config(kind, session)
    stamps = zs.stamps(source); jobs = [(i, i, source, stamps[:i + 1], cfg) for i in range(len(stamps))]
    done = {}
    with ProcessPoolExecutor(int(os.environ.get('DYN_EVAL_JOBS', '8'))) as pool:
        for n, header, rows in pool.map(score_one, jobs):
            done[n] = (header, rows)
    header = done[0][0]; rows = [r for n in sorted(done) for r in done[n][1]]
    (out / 'dynamic_objects.csv').write_text('\n'.join([header] + rows) + '\n')
    (out / 'map_timestamps.txt').write_text(''.join(f'{t}\n' for t in stamps))
    cols = header.split(','); cells = []
    for r in rows:
        v = dict(zip(cols, r.split(',')))
        cells.append(metrics(float(v['NumObjDetected']), float(v['NumObjHallucinated']), float(v['NumObjMissed'])))
    C = np.array(cells); tot = [sum(float(dict(zip(cols, r.split(',')))[k]) for r in rows) for k in ('NumObjDetected', 'NumObjHallucinated', 'NumObjMissed')]
    res = dict(map=str(source), maps=len(stamps), cells=len(rows), DynamicPrecision=float(np.nanmean(C[:, 0]) * 100) if np.any(~np.isnan(C[:, 0])) else None,
               DynamicRecall=float(np.nanmean(C[:, 1]) * 100) if np.any(~np.isnan(C[:, 1])) else None,
               DynamicF1=float(np.nanmean(C[:, 2]) * 100) if np.any(~np.isnan(C[:, 2])) else None, totals_TP_FP_FN=tot)
    (out / 'DYNAMIC_4D.json').write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res))


if __name__ == '__main__':
    if sys.argv[1] == 'gt': build_gt(sys.argv[2], sys.argv[3])
    else: score_map(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
