"""The 3RScan P3 rooms (datasets/3rscan/CHAINS_P3.txt) in Panoptic Mapping's flat-dataset format, for its rerun
(analysis/panoptic_rerun_3rscan.sh), as the synthetic input was made (Synthetic_ABC/scripts/prepare_panoptic_synthetic.py):
per visit (mapping_configs/ROOM/chain.json order) NNNNNN_{color.png, depth.tiff, pose.txt, segmentation.png},
timestamps.csv and the visit's camera (camera.txt: fx fy cx cy width height); per room groundtruth_labels.csv.
Frames: every 2nd pixel of the 960x540 session frames (datasets/3rscan/sessions/SCAN/rgbd), the 480x270 input every method
gets on 3RScan; intrinsics halved accordingly. ID image (8 bit, the player reads one channel): a thing pixel (the session's
instance label, a 3RScan objectId of the room's physical catalog) carries the object's place in the catalog (1..N), any
other pixel N + 1 + its ADE20K class (the session's semantics); groundtruth_labels.csv names each id with its ADE20K
class (things: the catalog's semantic_id and name, which keeps the 3RScan objectId), PanopticID 1 for things.
  usage: panoptic_3rscan_input.py OUT_DIR [ROOM ...]
"""
import csv, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

D = Path('/home/jixian/Desktop/FT/datasets/3rscan')


def frame(job):
    src, dst, fid, lut = job
    color = np.asarray(Image.open(src / 'rgbd' / f'{fid}_color.png'))[::2, ::2]
    Image.fromarray(color).save(dst / f'{fid}_color.png')
    depth = np.asarray(Image.open(src / 'rgbd' / f'{fid}_depth.tiff'), dtype=np.float32)[::2, ::2]
    Image.fromarray(depth).save(dst / f'{fid}_depth.tiff')
    inst = np.asarray(Image.open(src / 'instance_labels' / f'{fid}_segmentation.png')).astype(np.int64)[::2, ::2]
    sem = np.asarray(Image.open(src / 'semantics' / f'{fid}_segmentation.png')).astype(np.int64)[::2, ::2]
    thing = inst != 0
    assert np.all(lut[inst[thing]] > 0), (str(src), fid, 'instance not in the catalog')
    seg = np.where(thing, lut[np.where(thing, inst, 0)], lut[0] + sem)
    assert seg.max() <= 255
    Image.fromarray(seg.astype(np.uint8)).save(dst / f'{fid}_segmentation.png')
    os.link(src / 'rgbd' / f'{fid}_pose.txt', dst / f'{fid}_pose.txt')


def room(out, name):
    cfg = D / 'mapping_configs' / f'p3_{name}'
    catalog = sorted(json.loads((cfg / 'physical_catalog.json').read_text())['entities'], key=lambda e: e['physical_instance_id'])
    n = len(catalog)
    lut = np.zeros(max(e['physical_instance_id'] for e in catalog) + 1, np.int64)
    for k, e in enumerate(catalog):
        lut[e['physical_instance_id']] = k + 1
    lut[0] = n + 1                                         # stuff: lut[0] + ADE class
    room_out = out / f'p3_{name}'; room_out.mkdir(parents=True, exist_ok=True)
    with (room_out / 'groundtruth_labels.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['InstanceID', 'ClassID', 'PanopticID', 'R', 'G', 'B', 'Name', 'Size'])
        for k, e in enumerate(catalog):
            i = k + 1
            w.writerow([i, e['semantic_id'], 1, (i * 37) % 256, (i * 71) % 256, (i * 113) % 256, e['name'], 'L'])
        for ade in range(150):
            w.writerow([n + 1 + ade, ade, 0, (ade * 37) % 256, (ade * 71) % 256, (ade * 113) % 256, f'ade_{ade}', 'L'])
    visits = json.loads((cfg / 'chain.json').read_text())['visits']
    jobs = []
    for scan in visits:
        src, dst = D / 'sessions' / scan, room_out / scan
        dst.mkdir(exist_ok=True)
        rows = list(csv.DictReader((src / 'rgbd' / 'timestamps.csv').open()))
        (dst / 'timestamps.csv').write_text((src / 'rgbd' / 'timestamps.csv').read_text())
        k = dict(line.split(':') for line in (src / 'rgbd' / 'Intrinsics.txt').read_text().split('\n') if ':' in line)
        (dst / 'camera.txt').write_text(' '.join(str(v) for v in (float(k['f_x']) / 2, float(k['f_y']) / 2, float(k['u']) / 2,
                                                                 float(k['v']) / 2, int(k['Res_x']) // 2, int(k['Res_y']) // 2)) + '\n')
        jobs += [(src, dst, r['ImageID'], lut) for r in rows if not (dst / f"{r['ImageID']}_pose.txt").exists()]
    with ProcessPoolExecutor(8) as pool:
        list(pool.map(frame, jobs, chunksize=16))
    print(name, 'visits', len(visits), 'frames written', len(jobs), 'things', n, flush=True)


if __name__ == '__main__':
    out = Path(sys.argv[1])
    rooms = sys.argv[2:] or [line.split()[0][:8] for line in (D / 'CHAINS_P3.txt').read_text().splitlines() if line and not line.startswith('#')]
    for r in rooms:
        room(out, r)
