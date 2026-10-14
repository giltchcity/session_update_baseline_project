#!/usr/bin/env bash
# Real B again with chain_clean 7f12297 (immune cache + no zeroing of rows the compact step does not touch; supervisor 14:26):
#  1) check run, branch check_7f12297 a2bed18: the first 150 frames of real B with the three assertions (immune cache == uncached;
#     (immune | frozen) & _step_rows empty; untouched rows' parameters and Adam state unchanged by the step); a failure stops here.
#  2) the full real B (all 4041 frames) from the accepted A with 7f12297, refinement off: its seconds against the 1.3x line
#     (1532.6 s) and its retirements / live elements against this chain's B. Each step under GPU.lock; checkpoints deleted after.
set -u
X=$(cd "$(dirname "$0")" && pwd); RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; W=$RR/worktrees
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=$RR/code_mine_20261014/runs_quick3k; L=$Q/realb_rerun_queue.log; LOCK=/home/jixian/Desktop/FT/runs/GPU.lock
M=$RR/analysis/DISK_RECLAIM_20261013.txt
CHK=$W/check_7f12297; CC=$W/chain_clean
for c in $CHK $CC; do git -C $c status --porcelain | grep -q . && { echo "^refused dirty $c" >> $L; exit 3; }; done
KC=$(git -C $CHK rev-parse --short HEAD); K=$(git -C $CC rev-parse --short HEAD)
D1=$Q/check150_real_b_$KC; mkdir -p $D1; cd $D1; git -C $CHK rev-parse HEAD > code_commit.txt
echo "^start check150 $KC $(date '+%T')" >> $L
flock $LOCK $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -W ignore $X/quick_map.py $CHK $D1/map 150 real b
echo "^done check150 $KC $(date '+%T') $(cat status.txt 2>/dev/null)" >> $L
grep "CHECK_7F12297" $D1/run.log | tail -1 >> $L
for f in $(ls $D1/map/checkpoint_*.pt $D1/map/session_*/timeline*.pkl 2>/dev/null); do [ -L "$f" ] || { echo "$(date '+%F %H:%M') check run of 7f12297 (real B 150 frames; run.log kept): $(stat -c %s $f) $f" >> $M; rm -- "$f"; }; done
if grep -q "CHECK .*differs\|CHECK .*not empty\|CHECK untouched" $D1/run.log || ! grep -q "QUICK_MAP_DONE" $D1/run.log; then
  echo "^check FAILED or incomplete: no rerun" >> $L; exit 6; fi
D2=$RR/real_b_rerun_$K; mkdir -p $D2; cd $D2; git -C $CC rev-parse HEAD > code_commit.txt
echo "^start real_b_rerun $K $(date '+%T') disk $(df --output=used -BG /home/jixian | tail -1 | tr -d ' ')" >> $L
flock $LOCK $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -W ignore $X/quick_map.py $CC $D2/map 0 real b
echo "^done real_b_rerun $K $(date '+%T') $(cat status.txt 2>/dev/null)" >> $L
python3 - $D2/map/session_b/run.json $RR/real_chain_4ca6976/map/session_b/run.json $D2/run.log $RR/real_chain_4ca6976/run.log >> $L <<'PY'
import json, re, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
def last_el(p):
    m = re.findall(r'real_b t=\s*[\d.]+s frames=\d+/\d+ elements=(\d+)', open(p, errors='ignore').read())
    return int(m[-1]) if m else None
print(f'^rerun B seconds {a["seconds"]} (line 1532.6 s, native 1178.9 s -> {a["seconds"] / 1178.9:.3f}x) '
      f'{"PASS" if a["seconds"] <= 1532.6 else "FAIL"}; chain B {b["seconds"]}; peak {a.get("peak_gpu_gb")} GB')
print(f'^retired rerun {a["retired_by_layer"]} chain {b["retired_by_layer"]}; live elements at B end rerun {last_el(sys.argv[3])} chain {last_el(sys.argv[4])}')
PY
for f in $(ls $D2/map/checkpoint_*.pt $D2/map/session_*/timeline*.pkl 2>/dev/null); do [ -L "$f" ] || { echo "$(date '+%F %H:%M') real B rerun with 7f12297 (time measurement; run.json / run.log kept): $(stat -c %s $f) $f" >> $M; rm -- "$f"; }; done
echo "^realb rerun queue end $(date '+%T')" >> $L
