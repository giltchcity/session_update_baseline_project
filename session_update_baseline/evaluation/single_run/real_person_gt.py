"""Ground truth of the people in real A/B/C (user 11-01), for Khronos's official dynamic-object metric: every frame,
the person pixels (semantic label 12, the dynamic label of the label space) split into connected blobs (>= MIN_PX),
each blob back-projected with the frame's depth, intrinsics and camera pose (B, C put in A's frame); a blob's position is
the median of its points. Blobs are linked frame to frame into tracks (nearest, within LINK_M, gap <= GAP_S). Stamps are
the frames' stamps as the players publish them (session start + timestamps.csv). Writes OUT/person_tracks_{a,b,c}.csv
(track, stamp_ns, time_s, x, y, z, pixels) and OUT/person_tracks.png (top view: x-z, the data's y points down).
Read-only on the data.
usage: real_person_gt.py OUT"""
import csv, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, cv2
L = Path('/home/jixian/Desktop/FT/datasets/local_ab'); OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
RUNS = {s: next((L / 'rgbd').glob(f'session_{s}_*')) for s in 'abc'}
START = {'a': 1786279210201000000, 'b': 1786302302620000000, 'c': 1787394423387000000}   # the players' session starts
WORLD = {'a': np.eye(4), **{s: np.loadtxt(L / f'alignment/session_{s}_to_session_a.txt') for s in 'bc'}}
PERSON, MIN_PX, LINK_M, GAP_S = 12, 3000, 0.75, 1.0


def intrinsics(run):
    k = dict(l.split(': ') for l in (run / 'Intrinsics.txt').read_text().splitlines() if ': ' in l)
    return float(k['f_x']), float(k['f_y']), float(k['u']), float(k['v'])


def frame(job):
    s, i, stamp = job
    sem = cv2.imread(str(L / f'semantics/session_{s}/{i:06d}_segmentation.png'), cv2.IMREAD_UNCHANGED)
    if sem is None: return []
    sem = sem.reshape(sem.shape[0], sem.shape[1], -1)[:, :, 0]
    mask = (sem == PERSON).astype(np.uint8)
    if mask.sum() < MIN_PX: return []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    run = RUNS[s]; fx, fy, cx, cy = intrinsics(run)
    depth = cv2.imread(str(run / f'{i:06d}_depth.tiff'), cv2.IMREAD_UNCHANGED)
    T = WORLD[s] @ np.loadtxt(run / f'{i:06d}_pose.txt')
    out = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < MIN_PX: continue
        v, u = np.nonzero(lab == k); z = depth[v, u].astype(np.float64)
        ok = np.isfinite(z) & (z > 0.1) & (z < 8.0)
        if ok.sum() < MIN_PX // 2: continue
        u, v, z = u[ok], v[ok], z[ok]
        P = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], 1) @ T[:3, :3].T + T[:3, 3]
        out.append((s, i, stamp, *np.median(P, 0), int(ok.sum())))
    return out


def link(obs):
    """Frame-to-frame nearest-neighbour tracks."""
    tracks = []   # each: list of observations
    for s, i, stamp, x, y, z, px in sorted(obs, key=lambda o: o[2]):
        p = np.array([x, y, z]); best = None
        for t in tracks:
            q = t[-1]
            if 0 < stamp - q[2] <= GAP_S * 1e9:
                d = np.linalg.norm(p - np.array(q[3:6]))
                if d <= LINK_M and (best is None or d < best[0]): best = (d, t)
        (best[1] if best else tracks.append([]) or tracks[-1]).append((s, i, stamp, x, y, z, px))
    return [t for t in tracks if len(t) >= 10]   # at least 10 frames (1/3 s)


if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, s in zip(axes, 'abc'):
        ts = [int(r['TimeStamp']) for r in csv.DictReader((RUNS[s] / 'timestamps.csv').open())]
        with Pool(8) as pool:
            obs = [o for part in pool.imap(frame, [(s, i, START[s] + t) for i, t in enumerate(ts)], chunksize=16) for o in part]
        tracks = link(obs)
        with (OUT / f'person_tracks_{s}.csv').open('w', newline='') as f:
            w = csv.writer(f); w.writerow(['track', 'stamp_ns', 'time_s', 'frame', 'x', 'y', 'z', 'pixels'])
            for k, t in enumerate(tracks):
                for (_, i, stamp, x, y, z, px) in t:
                    w.writerow([k, stamp, round((stamp - START[s]) / 1e9, 4), i, round(x, 4), round(y, 4), round(z, 4), px])
        for k, t in enumerate(tracks):
            P = np.array([o[3:6] for o in t]); tt = np.array([(o[2] - START[s]) / 1e9 for o in t])
            ax.plot(P[:, 0], P[:, 2], '-', lw=1.5, label=f'track {k}: {tt[0]:.0f}-{tt[-1]:.0f} s, {len(t)} frames')
            ax.plot(P[0, 0], P[0, 2], 'o', ms=5, color=ax.lines[-1].get_color())
        ax.set_title(f'real {s.upper()}: {len(tracks)} person tracks, {len(obs)} person blobs in {len(ts)} frames')
        ax.set_aspect('equal'); ax.grid(alpha=.3); ax.legend(fontsize=7); ax.set_xlabel('x [m] (A frame)'); ax.set_ylabel('z [m]')
        print(s, len(ts), 'frames,', len(obs), 'person blobs,', len(tracks), 'tracks:',
              [(len(t), round((t[0][2] - START[s]) / 1e9, 1), round((t[-1][2] - START[s]) / 1e9, 1)) for t in tracks], flush=True)
    fig.tight_layout(); fig.savefig(OUT / 'person_tracks.png', dpi=110)
