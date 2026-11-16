"""The Home column of the main table from the per-slice results in OUT_DIR (run_4d.sh, run_wavemap.sh, run_panoptic.sh,
and the two 3DGS folders written by slice_geometry.py, README.md):
  Mesh     F1 of the slice at session B's last frame (the continued map at the end)
  Mesh_4D  mean F1 over the method's own slices (a row is a slice when its map was emitted at that query moment; B's
           inherited slice is queried at B's frame 0); Panoptic over its 25 given query frames
usage: summarize.py [OUT_DIR]   (default: the working folder's room/)"""
import csv, statistics, sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/jixian/Desktop/FT/wt_takeover_runs/geometry_4d_20261114/room')
ORIGIN = {'a': 1_000_000_000, 'b': 141_000_000_000}; LAST = {'a': 4185, 'b': 3554}
ROWS = [('TSDF', 'Khronos', 'map_vs_gt_khronos_cont', None), ('TSDF', 'Panoptic Mapping', 'map_vs_gt_panoptic', 'map_vs_gt_panoptic_final'),
        ('TSDF', 'Ours', 'map_vs_gt_ours', None), ('3DGS', 'GaME', 'map_vs_gt_game', None), ('3DGS', 'Ours', 'map_vs_gt_3dgs_ours', None),
        ('Occupancy', 'Wavemap', 'map_vs_gt_wavemap_row3', None), ('Occupancy', 'Ours', 'map_vs_gt_wavemap_row4', None)]


def slices(rows, s, given):
    sel = [r for r in rows if r['session'] == s]
    if given or not sel[0].get('prediction_stamp_ns'): return sel
    return [r for r in sel if r['prediction_stamp_ns'] and (int(float(r['prediction_stamp_ns'])) == ORIGIN[s] + round(float(r['time_s']) * 1e9)
                                                           or (s == 'b' and int(r['frame']) == 0))]


print('| Map | Method | Mesh | Mesh_4D | Mesh_4D A | Mesh_4D B | slices |')
print('|---|---|---|---|---|---|---|')
for kind, method, folder, final in ROWS:
    path = OUT / folder / 'MAP_VS_GT.csv'
    if not path.exists():
        print(f'| {kind} | {method} | missing {path} |'); continue
    rows = list(csv.DictReader(open(path))); given = final is not None
    per = {s: [100 * float(r['F1']) for r in slices(rows, s, given)] for s in 'ab'}
    last = [r for r in csv.DictReader(open(OUT / (final or folder) / 'MAP_VS_GT.csv')) if r['session'] == 'b' and int(r['frame']) == LAST['b']]
    allf = per['a'] + per['b']
    print(f"| {kind} | {method} | {100 * float(last[-1]['F1']):.1f} | {statistics.mean(allf):.1f} | {statistics.mean(per['a']):.1f} | "
          f"{statistics.mean(per['b']):.1f} | {len(per['a'])} + {len(per['b'])} |")
