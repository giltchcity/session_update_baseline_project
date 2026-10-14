#!/usr/bin/env bash
# GPU queue behind quick mapping test 1 (each job takes GPU.lock on its own), ordered for the earliest synthetic B plant result:
# [K2] offline proxy on synthetic B -> 150-frame speed test baseline / K2 -> speed gate (K2 wall time <= 1.10 x baseline, the
# user's 10 % rule) -> full synthetic B mapping with K2 (refinement off) -> [K2] offline proxies on real C, A, B (the real gate
# before any real chain).
set -u
X=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$X"); R=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006
S=/tmp/claude-1000/-home-jixian-Desktop-FT/633eaaed-1a6e-4213-936b-f03c710b0681/scratchpad; PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python
Q=$REPO/runs_quick3k; L=$Q/k2_trial_queue.log; H=$(git -C $REPO rev-parse --short HEAD); K=$(git -C $S/wt_static rev-parse --short HEAD)
LOCK=/home/jixian/Desktop/FT/runs/GPU.lock
run() { local name=$1; shift; local D=$Q/${name}; mkdir -p $D; cd $D; echo "^start $name $(date '+%T')" >> $L; local t0=$(date +%s)
  flock $LOCK $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@"
  echo "^done $name $(date '+%T') seconds=$(( $(date +%s) - t0 )) $(cat status.txt)" >> $L; }
run k2off_synB_$K $PY -W ignore $X/k2_offline.py $S/wt_static $R/syn_row4f3e5_r11p/checkpoint_b.pt synthetic b k2.jsonl
run speed150_final0638_b9400c9 $PY -W ignore $X/quick_map.py $S/wt_final0638 $Q/speed150_final0638_b9400c9/map 150
run speed150_static_kf_$K $PY -W ignore $X/quick_map.py $S/wt_static $Q/speed150_static_kf_$K/map 150
tb=$(grep -o "^\^done speed150_final0638_b9400c9 .* seconds=[0-9]*" $L | tail -1 | sed 's/.*seconds=//'); tk=$(grep -o "^\^done speed150_static_kf_$K .* seconds=[0-9]*" $L | tail -1 | sed 's/.*seconds=//')
if python3 -c "import sys; sys.exit(0 if int('$tk') <= 1.10 * int('$tb') else 1)"; then
  echo "^gate speed PASS k2=${tk}s base=${tb}s" >> $L
  run mapB_static_kf_$K $PY -W ignore $X/quick_map.py $S/wt_static $Q/mapB_static_kf_$K/map 0
else
  echo "^gate speed FAIL k2=${tk}s base=${tb}s: full synthetic B trial not started" >> $L
fi
for s in c a b; do run k2off_real${s}_$K $PY -W ignore $X/k2_offline.py $S/wt_static $R/real_row4f3e5/checkpoint_$s.pt real $s k2.jsonl; done
echo "^queue end $(date '+%T')" >> $L
