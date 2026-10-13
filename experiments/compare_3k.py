"""Per-view photo MAE (0-255) of the 3K arms: whole image, and on the pixels the colour floor covers (whole-session window).

usage: compare_3k.py  (reads runs_quick3k/*/core_check and runs_quick3k/colour_floor_*/)
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / 'runs_quick3k'
OLD = Path('/home/jixian/Desktop/FT/runs/focused_update_20261012/quick_r11p_3k/core_check')
VIEWS = ['a2700', 'a2850', 'a2985', 'b345', 'b1395', 'b1830', 'b1845', 'b1860', 'b2190', 'b2625', 'b2955', 'b3135',
         'b3447', 'b3480', 'b3525', 'b3540']
mine = next(ROOT.glob('mine_*'), None)
rt = next(ROOT.glob('rt_*'), None)
arms = [('pre-ref', mine, 'input'), ('rt 3K', rt, 'rt'), ('mine 3K', mine, 'mine'),
        ('old r15u 3K', OLD.parent, 'r15u'), ('old r15v 3K', OLD.parent, 'r15v')]
floor_dir = next(ROOT.glob('colour_floor_*'), None)
floor = json.loads((floor_dir / 'colour_floor.json').read_text()) if floor_dir and (floor_dir / 'colour_floor.json').exists() else {}


def load(path):
    return np.asarray(Image.open(path).convert('RGB')).astype(np.float64)


def picture(run, label, view):
    if run is None:
        return None
    path = run / 'core_check' / f'{label}__{view}.png'
    return load(path) if path.exists() else None


print(f'{"view":6s}' + ''.join(f'{a[0]:>13s}' for a in arms) + '   whole image, mean |render - photo|')
for view in VIEWS:
    photo_path = mine / 'core_check' / f'photo__{view}.png'
    if not photo_path.exists():
        continue
    photo = load(photo_path)
    cells = []
    for _, run, label in arms:
        img = picture(run, label, view)
        cells.append(f'{np.abs(img - photo).mean():13.2f}' if img is not None else f'{"-":>13s}')
    print(f'{view:6s}' + ''.join(cells))
if floor:
    print()
    print(f'{"view":6s}{"covered":>9s}' + ''.join(f'{a[0]:>13s}' for a in arms[:3])
          + f'{"floor 0.5s":>12s}{"floor 2s":>10s}{"floor 20s":>11s}{"floor all":>11s}   (covered pixels only)')
    for view, rec in floor.items():
        mask = np.load(floor_dir / f'covered_{view}_None.npy')
        photo = load(mine / 'core_check' / f'photo__{view}.png')
        cells = []
        for _, run, label in arms[:3]:
            img = picture(run, label, view)
            cells.append(f'{np.abs(img[mask] - photo[mask]).mean():13.2f}' if img is not None else f'{"-":>13s}')
        floors = [rec[f'window_{w}s']['floor_MAE_255'] for w in ('0.5', '2.0', '20.0', 'None')]
        print(f'{view:6s}{rec["window_Nones"]["covered"]:9.2f}' + ''.join(cells)
              + ''.join(f'{f:12.2f}' if f is not None else f'{"-":>12s}' for f in floors))
for name, run, label in arms[1:3]:
    if run is not None and (run / f'short_{label}.json').exists():
        j = json.loads((run / f'short_{label}.json').read_text())
        b, a = j['before'].get('b3525', {}), j['after'].get('b3525', {})
        print(f'{name}: pot ROI before {b.get("pot_photo_MAE_255", float("nan")):.2f} after {a.get("pot_photo_MAE_255", float("nan")):.2f}'
              f' | alive object rows {j["after"]["alive_object_rows"]} | {j["seconds"]:.0f} s | peak reserved {j["peak_reserved_GB"]:.1f} GB'
              f' | rows after {j["report"].get("n_gaussians", j["report"].get("rows", "?"))}')
