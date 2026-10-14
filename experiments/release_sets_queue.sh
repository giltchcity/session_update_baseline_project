#!/usr/bin/env bash
# Diagnostic (branch release_sets, diagnostic only; supervisor 13:50): the first 1300 frames of synthetic B from the accepted A with
# every F3e5 release recorded (uid, set, stamp); then, on CPU, the A rows B never measured on their surface but moved > 5 cm (and
# > 1 cm), split by the set that released them first and by identity. After every acceptance item; checkpoints deleted afterwards.
set -u
X=$(cd "$(dirname "$0")" && pwd); RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; CODE=$RR/worktrees/release_sets
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=$RR/code_mine_20261014/runs_quick3k; L=$Q/release_sets_queue.log
LOCK=/home/jixian/Desktop/FT/runs/GPU.lock; K=$(git -C $CODE rev-parse --short HEAD); D=$Q/release1300_syn_b_$K
git -C $CODE status --porcelain | grep -q . && { echo "^refused dirty $CODE" >> $L; exit 3; }
until grep -q '^\^equiv queue end' $Q/equiv_queue.log 2>/dev/null; do sleep 60; done          # after every acceptance item
mkdir -p $D; cd $D; git -C $CODE rev-parse HEAD > code_commit.txt
echo "^start release1300 $K $(date '+%T')" >> $L
flock $LOCK $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -W ignore $X/quick_map.py $CODE $D/map 1300 synthetic b
echo "^done release1300 $K $(date '+%T') $(cat status.txt 2>/dev/null)" >> $L
( cd $RR && nice -n 10 $PY -W ignore - $D <<'PY' ) >> $L 2>&1
import sys, glob
from collections import Counter
import numpy as np, torch
D = sys.argv[1]
ev = np.load(glob.glob(f'{D}/release_events_*.npz')[0])
first = {}
for u, c, t in sorted(zip(ev['uid'].tolist(), ev['code'].tolist(), ev['stamp'].tolist()), key=lambda x: x[2]):
    first.setdefault(u, c)
A = torch.load('syn_row4s0/checkpoint_a.pt', map_location='cpu', weights_only=False, mmap=True)['backend']
B = torch.load(f'{D}/map/checkpoint_b.pt', map_location='cpu', weights_only=False, mmap=True)['backend']
start = int(B['session_start'])
uA, uB = A['uid'].numpy(), B['uid'].numpy()
_, ia, ib = np.intersect1d(uA, uB, return_indices=True)
moved = np.linalg.norm(B['model'][1].detach().numpy()[ib] - A['model'][1].detach().numpy()[ia], axis=1) / 10.0
untouched = B['last_update'].numpy()[ib] < start
ident = B['identity'].numpy()[ib]
names = {1: 'contradicted', 2: 'overwritten', 4: 'pressured', 0: 'never released'}
for thr in (0.01, 0.05):
    sel = untouched & (moved > thr)
    codes = np.array([first.get(int(u), 0) for u in uB[ib][sel]])
    print(f'^untouched A rows moved > {thr * 100:.0f} cm: {int(sel.sum())}; first release: '
          + ', '.join(f'{names[c]} {int((codes == c).sum())}' for c in (1, 2, 4, 0)))
    for c in (1, 2, 4, 0):
        top = Counter(ident[sel][codes == c].tolist()).most_common(6)
        print(f'   {names[c]:15s} by identity: {top}')
PY
for f in $(ls $D/map/checkpoint_*.pt $D/map/session_*/timeline*.pkl 2>/dev/null); do
  [ -L "$f" ] || { echo "$(date '+%F %H:%M') release-set diagnostic (synthetic B 1300 frames; run.log and release_events kept): $(stat -c %s $f) $f" >> $RR/analysis/DISK_RECLAIM_20261013.txt; rm -- "$f"; }; done
echo "^release sets end $(date '+%T')" >> $L
