"""TESSE background Pre / Rec / F1 of our 4D map with its history (user 11-01): each map is scored with everything the 4D map
holds up to its time, the background, the objects present then and the historical objects. A closed object's node keeps
no mesh in the later slices (the 11-01 finding: the closed nodes are empty), so a historical object enters with its mesh
from the latest earlier slice where it was present. Otherwise the official mesh evaluation unchanged: DSG vertices
against the GT background points at each threshold of the evaluation's config (0.2 m the paper's; accuracy: vertices
with a GT point in reach; completeness: GT points with a vertex in reach), each map at its own time; the official 4D average (tables.py) over the evaluation's map x query rows.
Writes OUT (a copy of the main-table evaluation EVAL with background_mesh.csv replaced) and its TABLES_4D.txt.
usage: tesse_bg_history.py MAP.4dmap.zpk EVAL OUT GT_BACKGROUND.ply
"""
import csv, json, os, shutil, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np, open3d as o3d
from scipy.spatial import cKDTree
FT = '/home/jixian/Desktop/FT'
sys.path.insert(0, f'{FT}/eval_fast/vendor/pack')
import zpk_snapshot as zs
SRC, EVAL, OUT, GT = [os.path.abspath(a) for a in sys.argv[1:5]]
EXPORT = f'{FT}/wt_takeover_runs/analysis/history_export'
LIBS = ':'.join(f'/home/jixian/ros2_ws/install/{d}/lib' for d in sorted(os.listdir('/home/jixian/ros2_ws/install')))
ENV = dict(os.environ, LD_LIBRARY_PATH=f'{LIBS}:/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu')
PLOTTING = f'{FT}/build_artifacts/official_khronos_63faadde_abc/src/khronos_eval/plotting'


def export(i):
    """Slice i: the exported vertices (background and present objects) and the present objects' vertices by owner."""
    with tempfile.TemporaryDirectory(prefix='bg-history-', dir='/dev/shm') as tmp:
        zs.cut_many(SRC, [i], out=f'{tmp}/s.4dmap', work=tmp)
        subprocess.run([EXPORT, f'{tmp}/s.4dmap', f'{tmp}/s.ply'], env=ENV, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        meta = json.load(open(f'{tmp}/s.ply.json'))
        raw = open(f'{tmp}/s.ply', 'rb').read()
    off = raw.index(b'end_header\n') + 11
    V = np.frombuffer(raw, np.float32, meta['vertices'] * 3, off).reshape(-1, 3).copy()
    present = {l['owner']: V[l['vertex_start']:l['vertex_start'] + l['vertices']]
               for l in meta['layers'] if l['layer'] == 'object' and l['native_present'] and l['vertices'] > 0}
    absent = [o['owner'] for o in meta['native_nonpresent_objects'] if o['vertices'] == 0]   # those without their mesh
    return V, present, absent


gt = np.asarray(o3d.io.read_point_cloud(GT).points)
gt_tree = cKDTree(gt)
n = len(zs.stamps(SRC))
# Every threshold the evaluation's config scores (its background_mesh.csv columns; 11-01: the 5 cm one as well)
THRESHOLDS = [c.split('@')[1] for c in next(csv.reader(open(f'{EVAL}/results/background_mesh.csv'))) if c.startswith('Accuracy@')]
last = {}   # owner -> its mesh in the latest slice where it was present
rows, by_threshold = [], []
with ProcessPoolExecutor(4) as pool:
    for i, (V, present, absent) in enumerate(pool.map(export, range(n))):
        last.update(present)
        history = [last[o] for o in absent if o in last]
        P = np.concatenate([V] + history) if history else V
        # Bounded at the largest threshold (exact below it, all that is counted); 4 threads: office has 3 M GT points.
        bound = max(map(float, THRESHOLDS))
        to_gt = gt_tree.query(P, distance_upper_bound=bound, workers=4)[0]
        from_gt = cKDTree(P).query(gt, distance_upper_bound=bound, workers=4)[0]
        by_threshold.append({t: (float((to_gt <= float(t)).mean()), float((from_gt <= float(t)).mean())) for t in THRESHOLDS})
        acc, comp = by_threshold[-1]['0.2']
        rows.append((i, acc, comp, len(V), sum(len(h) for h in history), len(history)))
        print(f'map {i}: accuracy {acc:.6f} completeness {comp:.6f} vertices {len(V)} + history {rows[-1][4]} '
              f'({len(history)} closed objects)', flush=True)
if os.path.exists(OUT):
    shutil.rmtree(OUT)
shutil.copytree(EVAL, OUT)
p = f'{OUT}/results/background_mesh.csv'
table = list(csv.DictReader(open(p)))
assert len(table) == n, (len(table), n)
for r, (i, *_), scores in zip(table, rows, by_threshold):
    assert int(r['Name']) == i
    for t, (acc, comp) in scores.items():
        r[f'Accuracy@{t}'], r[f'Completeness@{t}'] = f'{acc:.6f}', f'{comp:.6f}'
with open(p, 'w') as f:
    w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
with open(f'{OUT}/BACKGROUND_HISTORY.csv', 'w') as f:
    w = csv.writer(f); w.writerow(['map', 'accuracy', 'completeness', 'vertices', 'history_vertices', 'closed_objects']); w.writerows(rows)
name = os.path.basename(OUT)
table4d = subprocess.run([sys.executable, f'{PLOTTING}/tables.py', os.path.dirname(OUT), '-m', f'{name}:{name}'],
                         check=True, capture_output=True, text=True, cwd=PLOTTING).stdout
open(f'{OUT}/TABLES_4D.txt', 'w').write(table4d)
print(table4d)
