"""G1 of a Flat session's final map: the last snapshot exported (in /dev/shm) with the real G1 "viewer" policy
(eval_fast's geometry_export: background plus the objects present at the final stamp), scored by the update_layer
project's Flat definition
(flat_geometry.py, copied here unchanged in method: point reference flat_{1,2}_gt_10000.ply, no crop, 5/10/20 cm).
A PLY in place of the map is scored as the final map itself (e.g. Panoptic's exported final mesh); SNAPSHOT picks
the snapshot of a packed map that is the session's final map (a session of a continuous map), default the last.
usage: flat_eval.py a|b MAP.4dmap.zpk|FINAL_MESH.ply OUT_DIR [SNAPSHOT]
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ROOT = Path('/home/jixian/Desktop/FT')
sys.path.insert(0, str(ROOT / 'eval_fast/vendor/pack'))
sys.path.insert(0, str(ROOT / 'tools/abc_eval_v2'))
import geometry_score as G  # noqa: E402
import map4d_pack as mp  # noqa: E402
import zpk_snapshot as zs  # noqa: E402

SPACING = 0.02
REF = {'a': ROOT / 'datasets/panoptic_mapping/flat_dataset/flat_1_gt_10000.ply',
       'b': ROOT / 'datasets/panoptic_mapping/flat_dataset/flat_2_gt_10000.ply'}


def voxel_union(points, spacing=SPACING):
    keys = np.floor(points / spacing).astype(np.int32)
    error = np.sum((points - (keys + .5) * spacing) ** 2, axis=1)
    order = np.lexsort((error, keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    first = np.r_[True, np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)]
    return points[order[first]]


def export_final(source, mesh, snapshot=None):
    """The final map of a 4D map as a mesh (the G1 viewer export); returns the map's snapshot count if known."""
    with tempfile.TemporaryDirectory(prefix='flat-eval-', dir='/dev/shm') as tmp:
        if source.endswith('.zpk'):   # a packed map: the final snapshot only
            count = len(mp.read_manifest(source)['snapshots'])
            sliced = Path(tmp) / 'one.4dmap'
            zs.cut(source, count - 1 if snapshot is None else snapshot, str(sliced))
        else:                         # a raw map (official Khronos): the exporter takes its final stamp
            count, sliced = None, Path(source)
        subprocess.run(['bash', '-c', 'source "$1" >/dev/null 2>&1 || exit 90; shift; exec "$@"', 'flat-eval',
                        str(ROOT / 'eval_fast/runtime_khronos_env.sh'), str(ROOT / 'eval_fast/vendor/geometry/geometry_export'),
                        str(sliced), str(mesh), 'viewer'], check=True, stdout=subprocess.DEVNULL)
    return count


session, source, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
work = tempfile.TemporaryDirectory(prefix='flat-eval-mesh-', dir='/dev/shm')   # the exported mesh, in RAM only
if source.endswith('.ply'):
    mesh, count = Path(source), None
else:
    mesh = Path(work.name) / 'geometry_viewer.ply'
    count = export_final(source, mesh, int(sys.argv[4]) if len(sys.argv) > 4 else None)
ref = np.asarray(o3d.io.read_point_cloud(str(REF[session])).points, dtype=np.float64)
pred = o3d.io.read_triangle_mesh(str(mesh))
pp, meta = G.surface_sample(pred)
rp = voxel_union(ref)
dp = cKDTree(ref).query(pp, k=1, workers=4)[0]
dr = G.distances(rp.astype(np.float32), pred)
row = dict(protocol='G1_flat_point_reference', session=session, prediction=source, reference=str(REF[session]),
           snapshots=count, pred_area_m2=round(float(meta['area_m2']), 1), pred_samples=int(len(pp)), ref_samples=int(len(rp)))
for t, tag in ((.05, '05'), (.10, '10'), (.20, '20')):
    p, r = float((dp <= t).mean()), float((dr <= t).mean())
    row.update({f'P_{tag}cm': round(p, 4), f'R_{tag}cm': round(r, 4), f'F1_{tag}cm': round(2 * p * r / (p + r) if p + r else 0., 4)})
(out / 'G1_flat.json').write_text(json.dumps(row, indent=2) + '\n')
work.cleanup()
print(json.dumps(row))
