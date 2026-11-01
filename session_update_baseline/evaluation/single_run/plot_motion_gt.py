"""Pictures to check the motion ground truth (user 11-01): (1) the trajectories drawn on a top view of our final map of
each visit (eval/motion_gt_maps/*.ply, the history_export of the results table's maps; ceiling removed, coloured by
height); (2) for each trajectory, three frames (start, middle, end): the RGB image with the person's / object's label
outline and the ground-truth position projected back into the image (red dot). Real: person tracks
(real_person_gt.py), data frame y down, top view x-z. Synthetic: moving objects (synthetic_motion_gt.py, Isaac), z up,
top view x-y. Writes PNGs next to the ground truth. Read-only on the data."""
import csv, json
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = Path('/home/jixian/Desktop/FT/wt_takeover_runs/eval'); MAPS = R / 'motion_gt_maps'
L = Path('/home/jixian/Desktop/FT/datasets/local_ab'); S = Path('/home/jixian/Desktop/FT/datasets/synthetic_ab')


def vertices(ply):
    raw = ply.read_bytes(); head = raw[:raw.index(b'end_header\n') + 11].decode(); n = int(head.split('element vertex ')[1].split()[0])
    return np.frombuffer(raw, np.float32, n * 3, len(head)).reshape(-1, 3)


def background(ax, V, horiz, up_sign, up_axis):
    h = up_sign * V[:, up_axis]; floor = np.percentile(h, 1)
    keep = (h > floor + 0.03) & (h < floor + 2.0)      # no floor plane, no ceiling
    V = V[keep][::7]; hh = (up_sign * V[:, up_axis]) - floor
    ax.scatter(V[:, horiz[0]], V[:, horiz[1]], c=hh, s=0.15, cmap='Greys', vmin=-0.5, vmax=2.2, rasterized=True)


def draw_tracks(ax, tracks, horiz):
    for (name, t, P), col in zip(tracks, plt.cm.tab10.colors):
        ax.plot(P[:, horiz[0]], P[:, horiz[1]], '-', color=col, lw=2.2, label=f'{name}: {t[0]:.0f}-{t[-1]:.0f} s')
        ax.plot(P[0, horiz[0]], P[0, horiz[1]], 'o', color=col, ms=8, mec='k')
        ax.plot(P[-1, horiz[0]], P[-1, horiz[1]], 's', color=col, ms=8, mec='k')
        ax.annotate(f'{t[0]:.0f}s', (P[0, horiz[0]], P[0, horiz[1]]), fontsize=8, color=col, xytext=(4, 4), textcoords='offset points')
        ax.annotate(f'{t[-1]:.0f}s', (P[-1, horiz[0]], P[-1, horiz[1]]), fontsize=8, color=col, xytext=(4, -10), textcoords='offset points')
    ax.legend(fontsize=8, loc='best'); ax.set_aspect('equal'); ax.grid(alpha=.2)


def frames_figure(path, panels):
    fig, axes = plt.subplots(len(panels), 3, figsize=(15, 3.6 * len(panels)), squeeze=False)
    for row, (title, items) in zip(axes, panels):
        for ax, (img, contour, uv, label) in zip(row, items):
            ax.imshow(img); ax.axis('off'); ax.set_title(f'{title} | {label}', fontsize=8)
            for c in contour: ax.plot(c[:, 0, 0], c[:, 0, 1], '-', color='cyan', lw=1)
            if uv is not None: ax.plot(uv[0], uv[1], 'o', color='red', ms=9, mec='white', mew=1.5)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def project(P, T, K, W, H):
    q = np.linalg.inv(T) @ np.r_[P, 1.0]
    if q[2] <= 0.05: return None
    u, v = K[0, 0] * q[0] / q[2] + K[0, 2], K[1, 1] * q[1] / q[2] + K[1, 2]
    return (u, v) if 0 <= u < W and 0 <= v < H else None


# ------------------------------------------------------------------------------------------------- real: people
RUNS = {s: next((L / 'rgbd').glob(f'session_{s}_*')) for s in 'abc'}
WORLD = {'a': np.eye(4), **{s: np.loadtxt(L / f'alignment/session_{s}_to_session_a.txt') for s in 'bc'}}
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
for ax, s in zip(axes, 'abc'):
    rows = list(csv.DictReader((R / f'real_motion_gt/person_tracks_{s}.csv').open()))
    tracks = []
    for k in sorted({r['track'] for r in rows}, key=int):
        tr = [r for r in rows if r['track'] == k]
        tracks.append((f'person {int(k) + 1}', np.array([float(r['time_s']) for r in tr]), np.array([[float(r['x']), float(r['y']), float(r['z'])] for r in tr]), tr))
    background(ax, vertices(MAPS / f'real_{s}.ply'), (0, 2), -1, 1)
    draw_tracks(ax, [(n, t, P) for n, t, P, _ in tracks], (0, 2))
    ax.set_title(f'real {s.upper()}: people walking (ground truth) on our map, top view'); ax.set_xlabel('x [m]'); ax.set_ylabel('z [m]')
    k = dict(l.split(': ') for l in (RUNS[s] / 'Intrinsics.txt').read_text().splitlines() if ': ' in l)
    K = np.array([[float(k['f_x']), 0, float(k['u'])], [0, float(k['f_y']), float(k['v'])], [0, 0, 1]])
    panels = []
    for name, t, P, tr in tracks:
        items = []
        for j in (int(len(tr) * .1), len(tr) // 2, int(len(tr) * .9)):
            f = int(tr[j]['frame']); img = cv2.cvtColor(cv2.imread(str(RUNS[s] / f'{f:06d}_color.png')), cv2.COLOR_BGR2RGB)
            sem = cv2.imread(str(L / f'semantics/session_{s}/{f:06d}_segmentation.png'), cv2.IMREAD_UNCHANGED)
            sem = sem.reshape(sem.shape[0], sem.shape[1], -1)[:, :, 0]
            contour, _ = cv2.findContours((sem == 12).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            Ps = (np.linalg.inv(WORLD[s]) @ np.r_[P[j], 1.0])[:3]      # back into the visit's frame
            uv = project(Ps, np.loadtxt(RUNS[s] / f'{f:06d}_pose.txt'), K, img.shape[1], img.shape[0])
            items.append((img, contour, uv, f'frame {f}, {t[j]:.1f} s'))
        panels.append((f'real {s.upper()} {name}', items))
    frames_figure(R / f'real_motion_gt/check_frames_{s}.png', panels)
fig.tight_layout(); fig.savefig(R / 'real_motion_gt/people_on_map.png', dpi=100); plt.close(fig)

# ------------------------------------------------------------------------------------------- synthetic: objects
REG = {x['instance_id']: x for x in json.loads((S / 'registry.json').read_text())['objects']}
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
for ax, s in zip(axes, 'ab'):
    rows = list(csv.DictReader((R / f'synthetic_motion_gt/moving_objects_{s}.csv').open()))
    tracks = []
    for inst in sorted({int(r['instance']) for r in rows}):
        tr = [r for r in rows if int(r['instance']) == inst]
        tracks.append((REG[inst]['name'], np.array([float(r['time_s']) for r in tr]), np.array([[float(r['x']), float(r['y']), float(r['z'])] for r in tr]), tr, inst))
    background(ax, vertices(MAPS / f'syn_{s}.ply'), (0, 1), 1, 2)
    draw_tracks(ax, [(n, t, P) for n, t, P, _, _ in tracks], (0, 1))
    ax.set_title(f'synthetic {s.upper()}: objects moved in view (Isaac ground truth) on our map, top view'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    K = np.array(json.loads((S / f'session_{s}/VALIDATION.json').read_text())['K'])
    panels = []
    for name, t, P, tr, inst in tracks:
        items = []
        for j in (int(len(tr) * .1), len(tr) // 2, int(len(tr) * .9)):
            f = int(tr[j]['frame']); img = cv2.cvtColor(cv2.imread(str(S / f'session_{s}/rgbd/{f:06d}_color.png')), cv2.COLOR_BGR2RGB)
            lab = cv2.imread(str(S / f'session_{s}/instance_labels/{f:06d}_segmentation.png'), cv2.IMREAD_UNCHANGED)
            contour, _ = cv2.findContours((lab == inst).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            uv = project(P[j], np.loadtxt(S / f'session_{s}/rgbd/{f:06d}_pose.txt'), K, img.shape[1], img.shape[0])
            items.append((img, contour, uv, f'frame {f}, {t[j]:.1f} s'))
        panels.append((f'synthetic {s.upper()} {name}', items))
    frames_figure(R / f'synthetic_motion_gt/check_frames_{s}.png', panels)
fig.tight_layout(); fig.savefig(R / 'synthetic_motion_gt/objects_on_map.png', dpi=100); plt.close(fig)
print('written')
