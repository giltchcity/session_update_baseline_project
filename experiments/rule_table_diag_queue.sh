#!/usr/bin/env bash
# Diagnostic of the rule-table mapping (branch rule_table_mapping, diagnostic only; supervisor 13:50 / 13:57): the first 1300 frames
# of synthetic B from the accepted A with the rule in the mapping, then the counterfactual probe restore_cf3.py on its B (A = syn_row4s0)
# and the time per frame against the chain's own first 1300 frames. After every acceptance item (the equivalence queue's end).
set -u
X=$(cd "$(dirname "$0")" && pwd); RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; CODE=$RR/worktrees/rule_table_mapping
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=$RR/code_mine_20261014/runs_quick3k; L=$Q/rule_table_diag_queue.log
LOCK=/home/jixian/Desktop/FT/runs/GPU.lock; K=$(git -C $CODE rev-parse --short HEAD); D=$Q/rule1300_syn_b_$K
git -C $CODE status --porcelain | grep -q . && { echo "^refused dirty $CODE" >> $L; exit 3; }
until grep -q '^\^equiv queue end' $Q/equiv_queue.log 2>/dev/null; do sleep 60; done          # after every acceptance item
mkdir -p $D; cd $D; git -C $CODE rev-parse HEAD > code_commit.txt
echo "^start rule1300 $K $(date '+%T')" >> $L
flock $LOCK $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -W ignore $X/quick_map.py $CODE $D/map 1300 synthetic b
echo "^done rule1300 $K $(date '+%T') $(cat status.txt 2>/dev/null)" >> $L
if [ -s $D/map/checkpoint_b.pt ]; then
  mkdir -p $D/cf3
  ( cd $D/cf3 && flock $LOCK $PY -W ignore $X/restore_cf3.py $RR/worktrees/r21c_base $D/cf3/renders $RR/syn_row4s0/checkpoint_a.pt $D/map/checkpoint_b.pt \
      a2700,a2850,a2985,b2625,b3447 ) > $D/cf3/run.log 2>&1
  echo "^done rule1300 cf3 $(date '+%T')" >> $L
fi
python3 - $D/run.log $RR/syn_chain_4ca6976/run.log >> $L <<'PY'
import re, sys
def wall(p, frames):
    rows = [(int(f), float(w)) for f, w in re.findall(r'synthetic_b t=\s*[\d.]+s frames=(\d+)/\d+ .*?gpu=[\d.]+/[\d.]+GB\s+([\d.]+)s', open(p, errors='ignore').read())]
    rows = [r for r in rows if r[0] <= frames]
    return rows[-1] if rows else None
a, b = wall(sys.argv[1], 1300), wall(sys.argv[2], 1300)
if a and b:
    print(f'^time rule branch {a[1]:.1f} s for {a[0]} frames = {a[1] / a[0]:.3f} s/frame; chain {b[1]:.1f} s for {b[0]} frames = {b[1] / b[0]:.3f} s/frame (whole chain B 1.533)')
PY
for f in $(ls $D/map/checkpoint_*.pt $D/map/session_*/timeline*.pkl 2>/dev/null); do
  [ -L "$f" ] || { echo "$(date '+%F %H:%M') rule-table mapping diagnostic (synthetic B 1300 frames; run.log, cf3 kept): $(stat -c %s $f) $f" >> $RR/analysis/DISK_RECLAIM_20261013.txt; rm -- "$f"; }; done
echo "^rule table diag end $(date '+%T')" >> $L
