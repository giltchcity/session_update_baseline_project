"""Real A/B/C object positions over time from the data itself (user 11-01: a motion ground truth for the real visits):
every STEP-th frame, each physical instance's pixels (instance_labels_validdepth, the instance image restricted to valid
depth) back-projected with the frame's depth, intrinsics and camera pose (camera-to-world), B and C put in A's frame
(alignment/session_{b,c}_to_session_a.txt); per instance the pixel count and the median, 10th and 90th percentile of
its points. Writes OUT/tracks_{a,b,c}.csv. Read-only on the data.
usage: real_motion_tracks.py OUT [STEP]"""
import csv, sys, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np, cv2
L = Path('/home/jixian/Desktop/FT/datasets/local_ab')
OUT = Path(sys.argv[1]); STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 5
OUT.mkdir(parents=True, exist_ok=True)
RUNS = {s: next((L / 'rgbd').glob(f'session_{s}_*')) for s in 'abc'}
WORLD = {'a': np.eye(4), **{s: np.loadtxt(L / f'alignment/session_{s}_to_session_a.txt') for s in 'bc'}}


def intrinsics(run):
    k = dict(l.split(': ') for l in (run / 'Intrinsics.txt').read_text().splitlines() if ': ' in l)
    return float(k['f_x']), float(k['f_y']), float(k['u']), float(k['v'])


def frame(job):
    s, i, t = job
    run = RUNS[s]; fx, fy, cx, cy = intrinsics(run)
    seg = cv2.imread(str(L / f'instance_labels_validdepth/session_{s}/{i:06d}_segmentation.png'), cv2.IMREAD_UNCHANGED)
    if seg is None: return []
    depth = cv2.imread(str(run / f'{i:06d}_depth.tiff'), cv2.IMREAD_UNCHANGED)
    T = WORLD[s] @ np.loadtxt(run / f'{i:06d}_pose.txt')
    rows = []
    for inst in np.unique(seg):
        if inst == 0: continue
        v, u = np.nonzero(seg == inst)
        z = depth[v, u].astype(np.float64); ok = np.isfinite(z) & (z > 0.1) & (z < 8.0)
        if ok.sum() < 200: continue
        u, v, z = u[ok], v[ok], z[ok]
        P = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], 1) @ T[:3, :3].T + T[:3, 3]
        q = np.percentile(P, [10, 50, 90], axis=0)
        rows.append([s, i, t, int(inst), int(ok.sum()), *np.round(q[1], 4), *np.round(q[0], 4), *np.round(q[2], 4), round(float(np.median(z)), 3)])
    return rows


if __name__ == '__main__':
    for s in 'abc':
        ts = [int(r['TimeStamp']) for r in csv.DictReader((RUNS[s] / 'timestamps.csv').open())]
        jobs = [(s, i, ts[i] / 1e9) for i in range(0, len(ts), STEP)]
        with Pool(8) as pool:
            rows = [r for part in pool.imap(frame, jobs, chunksize=4) for r in part]
        with (OUT / f'tracks_{s}.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['session', 'frame', 'time_s', 'instance', 'pixels', 'x', 'y', 'z', 'x10', 'y10', 'z10', 'x90', 'y90', 'z90', 'range_m'])
            w.writerows(rows)
        print(s, len(jobs), 'frames,', len(rows), 'instance observations', flush=True)
