"""Our own motion trajectories of a run, from what the run already recorded (user 11-01: an object is dynamic while it
moves, once it stops the map builds it as usual): the end of a move is our static-surface boundary (the run's log,
STATIC_SURFACE_BOUNDARY: the earliest frame of the object's current static state; frames before it were rejected as not
that surface); its positions are our tracker's per-frame centroids (the observer's live_tracks.jsonl). The move starts
at the last frame before the end where the smoothed centroid (running median, 9 frames) is still farther than 80% of
its largest distance from the settled position (the median centroid of the 2 s after the end) within the 10 s
before the end. No ground truth is read. Writes OUT.csv (instance, stamp_ns, x, y, z).
usage: ours_motion_from_run.py RUN_DIR LIVE_TRACKS.jsonl OUT.csv"""
import csv, json, re, sys
import numpy as np
run, live, out = sys.argv[1:4]
ends = {}
for line in open(f'{run}/control/logs/khronos.log', errors='replace'):
    m = re.search(r'STATIC_SURFACE_BOUNDARY inst=(\d+) rejected_stamp=(\d+)', line)
    if m: ends.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
tracks = {}
for l in open(live):
    r = json.loads(l)
    for t in r['tracks']:
        if t['id'] in ends: tracks.setdefault(t['id'], []).append((r['stamp_ns'], t['centroid']))
rows = []
for inst, stamps in sorted(ends.items()):
    v = sorted(tracks.get(inst, [])); T = np.array([a for a, _ in v], dtype=np.int64); P = np.array([b for _, b in v])
    for end in sorted(stamps):
        after = (T >= end) & (T <= end + 2e9); before = (T < end) & (T >= end - 10e9)
        if after.sum() < 5 or before.sum() < 10: print(f'inst {inst} end {end/1e9:.1f}: too few tracked frames'); continue
        settled = np.median(P[after], 0); idx = np.nonzero(before)[0]
        S = np.array([np.median(P[max(idx[0], i - 4):min(idx[-1], i + 4) + 1], 0) for i in idx])
        d = np.linalg.norm(S - settled, axis=1); far = np.nonzero(d >= 0.8 * d.max())[0]
        start = idx[far[-1]]; seg = range(start, idx[-1] + 1)
        print(f'inst {inst}: move {T[start]/1e9:.2f}-{end/1e9:.2f} s (stamps), {len(seg)} tracked frames, {d.max():.2f} m', flush=True)
        rows += [[inst, int(T[i]), *np.round(P[i], 4)] for i in seg]
with open(out, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['instance', 'stamp_ns', 'x', 'y', 'z']); w.writerows(rows)
