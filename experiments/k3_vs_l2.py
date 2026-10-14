"""K3's triggers (unmoved-camera frames GaME's own change tests promoted to keyframes) against the layer's L2 decisions at the same time
(CPU, read-only: k3.jsonl of the K3 offline runs, layer_log.txt of the chain's mapping run).
L2 state events read from layer_log.txt (the L2 core = the TSDF session_core port):
  NEW    a state record seen for the first time for an instance (ABSENCE_LOOK record=N, first appearance), at that look's stamp
  CLOSE  INHERITED_CLOSE / SESSION_CLOSE / TOP_CLOSE lines, at the stamp of the round they are written in
The stamp of a line is the latest stamp written before it (ABSENCE_LOOK stamp=, MOBILITY_PRIOR event_stamp=, a leading stamp).
For each K3 trigger at frame i (stamp t_i), the L2 events in [t_i, t_i + 2 rounds] (the layer decides at round ends) are listed.
usage: k3_vs_l2.py DATASET SESSION K3_JSONL LAYER_LOG
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

TREE = Path('/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/worktrees/chain_clean')
sys.path.insert(0, str(TREE))
from update_layer.frames import FlatSession  # noqa: E402
from update_layer.run import dataset_config  # noqa: E402

ds, sess, k3p, logp = sys.argv[1:5]
cfg, info, specs = dataset_config(ds)
spec = next(sp for sp in specs if sp.name.endswith('_' + sess))
fs = FlatSession(spec)
round_ns = int(cfg.round_s * 1e9)
trig = [json.loads(l) for l in open(k3p)]
judged = len(trig)
trig = [r for r in trig if r['keyframe']]

cur = None
seen_rec = set()
events = []                                  # (stamp, kind, inst, detail)
for line in open(logp, errors='ignore'):
    m = re.match(r'^(\d{9,})\s', line)
    if m:
        cur = int(m.group(1))
    for key in ('stamp', 'event_stamp'):
        m = re.search(r'\b' + key + r'=(\d{9,})', line)
        if m:
            cur = int(m.group(1))
    if line.startswith('ABSENCE_LOOK'):
        inst = int(re.search(r'inst=(\d+)', line).group(1)); rec = int(re.search(r'record=(\d+)', line).group(1))
        if (inst, rec) not in seen_rec:
            seen_rec.add((inst, rec))
            events.append((cur, 'NEW', inst, f'record {rec}'))
    elif line.startswith(('INHERITED_CLOSE', 'SESSION_CLOSE', 'TOP_CLOSE')):
        inst = int(re.search(r'inst=(\d+)', line).group(1))
        events.append((cur, line.split()[0], inst, ' '.join(line.split()[2:5])))
first_stamp = fs.stamp_ns(0)
in_session = [e for e in events if e[0] is not None and e[0] >= first_stamp]
print(f'{ds} {sess}: {judged} unmoved frames judged by K3, {len(trig)} triggers; round {cfg.round_s} s; '
      f'L2 events in the session: {sum(1 for e in in_session if e[1] == "NEW")} new states, '
      f'{sum(1 for e in in_session if e[1] != "NEW")} closes, at {len(set(e[0] for e in in_session))} distinct stamps')
hit = 0
for r in trig:
    t = fs.stamp_ns(r['frame'])
    win = [e for e in in_session if t <= e[0] <= t + 2 * round_ns]
    if win:
        hit += 1
    by = defaultdict(list)
    for e in win:
        by[e[2]].append(e[1])
    print(f'  frame {r["frame"]:5d}  t={(t - first_stamp) / 1e9:7.2f} s  region_px={r["region_px"]:4d} added_masks={r["added_masks"]}  '
          f'L2 within 2 rounds: ' + (', '.join(f'inst {k}: {"+".join(sorted(set(v)))}' for k, v in sorted(by.items())) or 'none'))
print(f'K3 triggers with an L2 state event within 2 rounds: {hit} of {len(trig)}')
print('K3_VS_L2_DONE')
