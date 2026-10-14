"""From the 150-frame immune() cache measurement to the full real B (CPU; supervisor 13:20).
Inputs: the measurement's run.log (IMMUNE_CACHE lines: calls, hits, recomputes, fresh_s, fresh_s_on_hits, cumulative per round) and the
full chain's run.log (F3E_PRUNE lines every 10th call: rows, contradicted; F3E_PRESSURE lines: released_total = the pressured set).
The saving is the uncached work on calls whose inputs had not changed (fresh_s_on_hits). Per call it is modelled as proportional to the
work of immune(): one comparison pass over N rows plus torch.isin of N uids against each set (sort-based, ~ (N + M) log2(N + M));
the constant comes from the measurement, N and M of every full-B call from the full chain's own log, the number of calls from the
measurement's calls per frame times the full B's frames, and the hit rate from the measurement. These are stated assumptions, not
measurements of the full B; the result is an estimate with them.
usage: immune_extrapolate.py MEASUREMENT_RUN_LOG FULL_CHAIN_RUN_LOG FULL_B_SECONDS LINE_SECONDS
"""
import ast
import math
import re
import sys

meas, full, b_sec, line = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
txt = open(meas, errors='ignore').read()
stats = [ast.literal_eval(m.group(1)) for m in re.finditer(r'IMMUNE_CACHE (\{.*\})', txt)]
if not stats:
    sys.exit('no IMMUNE_CACHE line in ' + meas)
st = stats[-1]
frames = max(int(m.group(1)) for m in re.finditer(r'real_b t=\s*[\d.]+s frames=(\d+)/', txt))
prune = [(int(a), int(b), int(c)) for a, b, c in re.findall(r'F3E_PRUNE call (\d+) rows=(\d+) immune=\d+ contradicted=(\d+)', txt)]
print(f'measurement: {frames} frames; calls {st["calls"]}, hits {st["hits"]} ({100 * st["hits"] / st["calls"]:.1f} %), recomputes '
      f'{st["recomputes"]}; uncached time total {st["fresh_s"]:.2f} s, on hits {st["fresh_s_on_hits"]:.2f} s '
      f'= {st["fresh_s_on_hits"] / frames:.4f} s per frame saved by the cache (over {frames} frames)')


def work(n, m):
    return n + sum((n + x) * math.log2(max(n + x, 2)) for x in m if x)


# the measurement's mean work per call: N from its F3E_PRUNE lines (every 10th call), sets from the same lines (contradicted)
meas_work = [work(r, (c,)) for _, r, c in prune] or [1.0]
k = st['fresh_s'] / st['calls'] / (sum(meas_work) / len(meas_work))           # seconds per unit of work
ftxt = open(full, errors='ignore').read()
fb = ftxt.split('real_c t=')[0]                                                 # session B only
fprune = [(int(r), int(c)) for r, c in re.findall(r'F3E_PRUNE call \d+ rows=(\d+) immune=\d+ contradicted=(\d+)', fb)]
fpress = [int(x) for x in re.findall(r'F3E_PRESSURE call \d+ rows=\d+ immune_before=\d+ released_now=\d+ released_total=(\d+)', fb)]
ffr = max(int(m.group(1)) for m in re.finditer(r'real_b t=\s*[\d.]+s frames=(\d+)/', fb))
calls_full = st['calls'] / frames * ffr
hit = st['hits'] / st['calls']
mean_work_full = sum(work(r, (c, p)) for (r, c), p in zip(fprune, fpress + [fpress[-1] if fpress else 0] * len(fprune))) / max(len(fprune), 1)
saving = calls_full * hit * k * mean_work_full
print(f'full B: {ffr} frames, ~{calls_full:.0f} calls (measurement rate), hit rate {100 * hit:.1f} %, mean work per call x '
      f'{mean_work_full / (sum(meas_work) / len(meas_work)):.2f} of the measurement\'s')
print(f'estimated saving on the full B: {saving:.1f} s -> {b_sec - saving:.1f} s vs the line {line:.1f} s: '
      f'{"WITHIN" if b_sec - saving <= line else "STILL ABOVE"} ({b_sec - saving - line:+.1f} s)')
print('IMMUNE_EXTRAPOLATE_DONE')
