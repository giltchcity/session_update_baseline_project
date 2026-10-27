"""Cut a continuous Khronos map (khronos_continuous.sh) into one map per session, as our session maps are laid out:
session s = the last snapshot before s starts (the map Khronos holds when s begins; none for the first session) plus
the snapshots inside s. Each slice is written to disk, packed to .zpk (source removed after the bit-exact check).
usage: khronos_continuous_slices.py MAP.4dmap.zpk OUT_DIR real|flat|synthetic
       khronos_continuous_slices.py MAP.4dmap.zpk --moments real|flat|synthetic   (print "session first last" per session:
       the same snapshots, as indices into MAP, for evaluating each session in memory; nothing is written)"""
import os, subprocess, sys, tempfile
sys.path.insert(0, '/home/jixian/Desktop/FT/eval_fast/vendor/pack')
import map4d_pack as mp
import zpk_snapshot as zs
src, out, dataset = sys.argv[1], sys.argv[2], sys.argv[3]
# session start [ns] and length [s]
SESSIONS = {'real': [('a', 1786279210201000000, 133.4), ('b', 1786302302620000000, 134.7), ('c', 1787394423387000000, 240.5)],
            'flat': [('a', 1627288677766243541, 197.3), ('b', 1628857488229302372, 140.4)],
            'synthetic': [('a', 1000000000, 139.5), ('b', 141000000000, 118.5)]}[dataset]
stamps = mp.read_manifest(src)['source']['stamps']
moments_only = out == '--moments'
if not moments_only:
    os.makedirs(out, exist_ok=True)
prev_last = None
for k, (name, start, length) in enumerate(SESSIONS):
    # the terminal save lands shortly after the last frame; never past the next session's start
    end = start + int((length + 5.0) * 1e9)
    if k + 1 < len(SESSIONS):
        end = min(end, SESSIONS[k + 1][1] - 1)
    inside = [i for i, t in enumerate(stamps) if start <= t <= end]
    if not inside:
        break  # the run stopped before this session
    idx = ([prev_last] if prev_last is not None else []) + inside
    if moments_only:
        print(name, idx[0], idx[-1], flush=True)
        prev_last = inside[-1]
        continue
    print(f'session {name}: {len(inside)} snapshots inside [{stamps[inside[0]]}, {stamps[inside[-1]]}]'
          f'{", seed " + str(stamps[prev_last]) if prev_last is not None else ""}', flush=True)
    path = os.path.join(out, f'session_{name}.4dmap')
    with tempfile.TemporaryDirectory(dir=out) as work:
        zs.cut_many(src, idx, out=path, work=work)
    subprocess.run([sys.executable, mp.__file__, 'pack', path, '--remove-source'], check=True)
    prev_last = inside[-1]
