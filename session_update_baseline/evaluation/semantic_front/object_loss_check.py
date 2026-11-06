"""Where a TESSE map's static-object score goes (user 11-03: the semantic front's maps have the instance run's meshes in
every snapshot; is it the objects' states?): the TESSE table's evaluation of the map's last snapshot (tesse_online_eval.py
ours4d: the official evaluator copy that queries earlier times, the scene's config, objects only) at every snapshot time,
keeping the evaluator's own per-object output (eval_visualization: objects.csv, each object's presence at each query;
associations.csv, map object -> GT object and GT object -> map object, the most surface points within 0.5 m of the same
class). Per query, as the evaluator counts them (static objects; its ObjectEvaluator never reads the config's
compensate_missegmentation, so it always compensates: per GT object only one of its map objects is looked at, per map
object only one of its GT objects):
  detected, missed, hallucinated: the evaluator's own counts (static_objects.csv);
  missed, no map object: a present GT object with no map object of its class within 0.5 m (how the geometry is cut);
  missed, people: the GT's dynamic objects (no mesh, never matched, the same for every map; presence from the GT graph,
    the evaluator's isPresent: its per-object file leaves out objects without a surface centroid);
  missed, map object absent (a state): the rest of the missed, a present GT object whose map object is not present then;
  hallucinated, no GT object: a present map object with no GT object of its class within 0.5 m (a fragment, a false one;
    map objects that are only points are not in the per-object file and fall in the next line);
  hallucinated, GT absent (a state) or point-only map objects: the rest of the hallucinated;
and present GT objects sharing their map object with another present GT object (merged: neither detected nor missed).
In associations.csv the people's raw node ids print as the static GT objects O(66)... do; people are never matched, so a
matched row there is the static object's.
The map: the run's final.4dmap.zpk through eval/chain_fill, as the runner does.
usage: object_loss_check.py office|apartment RUN_DIR [RUN_DIR ...]"""
import csv, json, os, subprocess, sys, tempfile
from collections import Counter, defaultdict
from pathlib import Path
FT = Path('/home/jixian/Desktop/FT'); R = FT / 'wt_takeover_runs'
scene, runs = sys.argv[1], sys.argv[2:]
os.environ['TESSE_EVAL_CONFIG'] = str(FT / f'configs/khronos/eval_{scene}_objects_only_local.yaml')
src = (R / 'analysis/tesse_online_eval.py').read_text()
sys.argv = ['tesse_online_eval.py'] + (['--scene', scene] if scene == 'apartment' else [])
exec(compile(src[:src.index('\nmethod, out = sys.argv')], 'tesse_online_eval.py', 'exec'))   # official(), zs
ENV = (f'set +u; source /opt/ros/jazzy/setup.bash; source /home/jixian/ros2_ws/install/setup.bash; '
       f'export LD_LIBRARY_PATH={FT}/wt_290e110_build/canonical/install/lib:$LD_LIBRARY_PATH; exec "$@"')
GT = json.loads((FT / f'datasets/khronos/tesse_cd/{scene}/ground_truth/gt_dsg_consolidated.json').read_text())
PEOPLE = [n['attributes'] for n in GT['nodes'] if n['attributes']['semantic_label'] == 4294967295]


def is_present(first, last, q):
    """The evaluator's isPresent (khronos_attribute_utils.cpp)."""
    appeared = [t for t in first if t <= q]; gone = [t for t in last if t <= q]
    return bool(appeared) and (not gone or appeared[-1] > gone[-1])


KEYS = ('detected', 'missed, no map object', 'missed, people', 'missed, map object absent', 'hallucinated, no GT object',
        'hallucinated, GT absent or point-only', 'merged GT objects')
for run in runs:
    with tempfile.TemporaryDirectory(prefix='object-loss-', dir='/dev/shm') as tmp:
        chain = f'{tmp}/chain.4dmap.zpk'
        subprocess.run(['bash', '-c', ENV, 'chain-fill', str(FT / 'eval/chain_fill/chain_fill'), chain, f'{run}/final.4dmap.zpk'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        stamps = zs.stamps(chain)
        zs.cut_many(chain, [len(stamps) - 1], out=f'{tmp}/one.4dmap', work=tmp)
        official(Path(tmp) / 'one.4dmap', Path(tmp) / 'official', stamps)
        objects = list(csv.DictReader(open(f'{tmp}/official/eval_visualization/objects.csv')))
        links = list(csv.DictReader(open(f'{tmp}/official/eval_visualization/associations.csv')))
        table = {r['Query']: r for r in csv.DictReader(open(f'{tmp}/official/results/static_objects.csv'))}
    present = defaultdict(dict)                       # query -> (is GT, id) -> present
    for o in objects:
        present[o['QueryTime']][(o['IsGT'] == '1', o['ID'])] = o['Present'] == '1'
    to_gt, to_map = {}, {}
    for a in links:
        d = to_map if a['FromGT'] == '1' else to_gt
        if d.get(a['FromID'], 'Failed') == 'Failed': d[a['FromID']] = a['ToID']
    c = Counter()
    for q, p in present.items():
        live_gt = {i for (g, i), v in p.items() if g and v}
        live_map = {i for (g, i), v in p.items() if not g and v}
        r = table[q]; e_det, e_hal, e_mis = (int(r[k]) for k in ('NumObjDetected', 'NumObjHallucinated', 'NumObjMissed'))
        mis_free = sum(to_map.get(g, 'Failed') == 'Failed' for g in live_gt)
        people = sum(is_present(a['first_observed_ns'], a['last_observed_ns'], int(q)) for a in PEOPLE)
        hal_free = sum(to_gt.get(m, 'Failed') == 'Failed' for m in live_map)
        assert e_mis >= mis_free + people and e_hal >= hal_free, (run, q, e_mis, mis_free, people, e_hal, hal_free)
        c.update({'detected': e_det, 'missed, no map object': mis_free, 'missed, people': people,
                  'missed, map object absent': e_mis - mis_free - people, 'hallucinated, no GT object': hal_free,
                  'hallucinated, GT absent or point-only': e_hal - hal_free})
        per_map = Counter(to_map[g] for g in live_gt if to_map.get(g) in live_map)
        c['merged GT objects'] += sum(n - 1 for n in per_map.values() if n > 1)
    det = c['detected']; mis = sum(c[k] for k in KEYS[1:4]); hal = sum(c[k] for k in KEYS[4:6])
    print(f'{scene} {Path(run).name}: last map at {len(present)} query times; P {100 * det / (det + hal):.1f} R {100 * det / (det + mis):.1f}; '
          + ', '.join(f'{k} {c[k]}' for k in KEYS), flush=True)
