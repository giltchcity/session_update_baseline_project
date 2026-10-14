#!/usr/bin/env bash
# GPU queue behind quick mapping test 1 (each job takes GPU.lock on its own): [K2] offline proxy on synthetic B and real C/A/B,
# then the 150-frame speed test of the baseline (final0638 b9400c9) and of K2 (static_kf ae5f24f).
set -u
X=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$X"); R=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006
S=/tmp/claude-1000/-home-jixian-Desktop-FT/633eaaed-1a6e-4213-936b-f03c710b0681/scratchpad; PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python
Q=$REPO/runs_quick3k; L=$Q/k2_gpu_queue.log; H=$(git -C $REPO rev-parse --short HEAD); LOCK=/home/jixian/Desktop/FT/runs/GPU.lock
run() { local name=$1; shift; local D=$Q/${name}_$H; mkdir -p $D; cd $D; echo "^start $name $(date '+%T')" >> $L
  flock $LOCK $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@"
  echo "^done $name $(date '+%T') $(cat status.txt)" >> $L; }
run k2off_synB $PY -W ignore $X/k2_offline.py $S/wt_static $R/syn_row4f3e5_r11p/checkpoint_b.pt synthetic b k2.jsonl
for s in c a b; do run k2off_real$s $PY -W ignore $X/k2_offline.py $S/wt_static $R/real_row4f3e5/checkpoint_$s.pt real $s k2.jsonl; done
for arm in final0638:$S/wt_final0638 static_kf:$S/wt_static; do n=${arm%%:*}; c=${arm#*:}
  run speed150_$n bash -c "t0=\$(date +%s.%N); $PY -W ignore $X/quick_map.py $c \$PWD/map 150; echo SPEED150 wall_s=\$(echo \"\$(date +%s.%N) - \$t0\" | bc)"; done
