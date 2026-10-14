#!/usr/bin/env bash
# TODO 6, speed gate only (supervisor 09:21): the SH arm of the 4K refinement speed test (r21c 14f3855) on the R11p B. The r21 arm
# (r21c_base 4ca6976) already ran 08:59:42-09:05:11 on the same input with the same harness (runs_quick3k/speed3k_r21c_base_4ca6976)
# and is not rerun. Gate: time per step over steps 3000-4000 <= 1.10 x r21, and SH degree 3 at the end of the SH arm.
# The 30K refinement is not started here: it waits for the K3 verdict, the K3 full-B trial and the real chain (supervisor 09:21).
set -u
W=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/worktrees; H=$W/harness_34bcf12/experiments
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/code_mine_20261014/runs_quick3k
L=$Q/sh_queue.log; LOCK=/home/jixian/Desktop/FT/runs/GPU.lock
SH=$(git -C $W/r21c rev-parse --short HEAD); git -C $W/r21c status --porcelain | grep -q . && { echo "^refused dirty r21c" >> $L; exit 3; }
D=$Q/speed3k_sh_$SH; mkdir -p $D
echo "^start speed3k sh $(date '+%T')" >> $L
flock $LOCK env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -W ignore $H/quick_screen.py short4k sh synthetic $D $W/r21c > $D/run.log 2>&1
rc=$?; echo "^done speed3k sh $(date '+%T') rc=$rc" >> $L
step34() { python3 - "$1/run.log" <<'PY'
import re, sys
txt = open(sys.argv[1], errors='ignore').read().replace('\r', '\n')
el = {}
for m in re.finditer(r'(\d+)/4000 \[(\d+):(\d+)(?::(\d+))?<', txt):
    n = int(m.group(1)); a, b, c = m.group(2), m.group(3), m.group(4)
    el[n] = (int(a) * 3600 + int(b) * 60 + int(c)) if c else (int(a) * 60 + int(b))
k3 = min((n for n in el if n >= 3000), default=None); k4 = max(el, default=None)
print((el[k4] - el[k3]) / max(k4 - k3, 1) if k3 is not None and k4 and k4 > k3 else 0)
PY
}
tb=$(step34 $Q/speed3k_r21c_base_4ca6976); ts=$(step34 $D)
deg=$(python3 -c "import json,glob,sys; print(json.load(open(glob.glob(sys.argv[1]+'/short4k_*.json')[0])).get('active_sh_degree_at_end'))" $D 2>/dev/null)
echo "^speed per step 3000-4000: sh=$ts r21=$tb; SH degree at the end of the SH arm: $deg" >> $L
if [ "$deg" = "3" ] && python3 -c "import sys; tb, ts = float('$tb'), float('$ts'); sys.exit(0 if tb > 0 and 0 < ts <= 1.10 * tb else 1)"; then
  echo "^gate speed PASS sh=$ts r21=$tb (30K waits for the K3 verdict, the K3 full-B trial and the real chain)" >> $L
else
  echo "^gate speed FAIL sh=$ts r21=$tb degree=$deg" >> $L
fi
echo "^sh speed queue end $(date '+%T')" >> $L
