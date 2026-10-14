"""Is the real B mapping slower because the map holds more elements? Per layer round, from the runs' own progress lines (CPU, reads logs).
Each progress line 'real_b t=..s frames=F/N elements=E ... gpu=..GB  W s' gives, per round, the frames done, the live elements and the
cumulative wall time W since the session start. All runs map the same frames (same input, same keyframes: the keyframe rule is pose-only),
so a round's work differs between runs only by the map and the code. Model, least squares over every round of every run:
    seconds per frame of round r in run k = a_r (the round: its frames and keyframes) + b * elements (k, r) + c_k (the run's code)
b is the cost of one live element per frame; c_k what the code costs beyond its elements. The decomposition of the time difference
between two runs is then sum_r frames_r * (b * delta elements_r) + sum_r frames_r * delta c.
usage: round_time_vs_elements.py SESSION NAME=LOG [NAME=LOG ...]     (the last complete pass of SESSION in each log is used)
"""
import re
import sys

import numpy as np

S = sys.argv[1]
PAT = re.compile(r'real_' + S + r' t=\s*([\d.]+)s frames=(\d+)/(\d+) elements=(\d+) .*?gpu=[\d.]+/[\d.]+GB\s+([\d.]+)s')
runs = {}
for arg in sys.argv[2:]:
    name, path = arg.split('=', 1)
    rows = [tuple(map(float, m.groups())) for m in PAT.finditer(open(path, errors='ignore').read())]
    passes, cur = [], []
    for r in rows:                                  # a restart shows as frames going down: keep the last pass
        if cur and r[1] <= cur[-1][1]:
            passes.append(cur)
            cur = []
        cur.append(r)
    passes.append(cur)
    last = passes[-1]
    runs[name] = np.array(last)                     # columns: t, frames, total, elements, wall
names = list(runs)
n = min(len(v) for v in runs.values())
for k in names:
    print(f'{k:16s} rounds {len(runs[k]):3d}  frames {int(runs[k][-1, 1])}/{int(runs[k][-1, 2])}  wall at the last progress line {runs[k][-1, 4]:8.1f} s  '
          f'elements first {int(runs[k][0, 3])} last {int(runs[k][-1, 3])}')
# per round (rounds 1..n-1; round 0 holds the session start: checkpoint load etc.)
dfr = {k: np.diff(v[:n, 1]) for k, v in runs.items()}
dwall = {k: np.diff(v[:n, 4]) for k, v in runs.items()}
el = {k: 0.5 * (v[1:n, 3] + v[:n - 1, 3]) for k, v in runs.items()}             # mean live elements over the round
spf = {k: dwall[k] / dfr[k] for k in names}
R = n - 1
X, y, w = [], [], []
for j, k in enumerate(names):
    for r in range(R):
        row = np.zeros(R + 1 + len(names) - 1)
        row[r] = 1.0                                  # a_r
        row[R] = el[k][r] / 1e6                       # b (per million elements)
        if j:
            row[R + j] = 1.0                          # c_k relative to the first run
        X.append(row); y.append(spf[k][r]); w.append(dfr[k][r])
X, y, w = np.array(X), np.array(y), np.sqrt(np.array(w))
coef, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
b = coef[R]
c = {names[0]: 0.0, **{k: coef[R + j] for j, k in enumerate(names) if j}}
res = y - X @ coef
print(f'\nfit over {len(y)} rounds: b = {b:.4f} s per frame per million live elements; code terms c (s per frame, vs {names[0]}): '
      + ', '.join(f'{k} {v:+.4f}' for k, v in c.items()) + f'; residual rms {np.sqrt(np.mean(res ** 2)):.4f} s per frame')
base = names[0]
for k in names[1:]:
    tot = float(np.sum(dwall[k] - dwall[base]))
    by_el = float(np.sum(dfr[k] * b * (el[k] - el[base]) / 1e6))
    by_code = float(np.sum(dfr[k]) * (c[k] - c[base]))
    print(f'{k} vs {base}: wall over rounds 1..{R} {np.sum(dwall[k]):.1f} vs {np.sum(dwall[base]):.1f} s (diff {tot:+.1f} s); '
          f'explained by elements {by_el:+.1f} s, by the code term {by_code:+.1f} s, rest {tot - by_el - by_code:+.1f} s')
print('\nper round (every 8th): round, frames, s/frame and mean elements (M) per run')
for r in range(0, R, 8):
    print(f'  {r + 1:3d} {int(dfr[base][r]):4d} ' + '  '.join(f'{k}: {spf[k][r]:.3f} s/f {el[k][r] / 1e6:.3f} M' for k in names))
print('ROUND_TIME_DONE')
