#!/usr/bin/env bash
# GPU queue behind quick mapping test 1 (each job takes GPU.lock on its own), ordered for the earliest synthetic B plant result.
# Code under test = persistent detached worktrees under runs/update_layer_game_T1_20261006/worktrees (survive a reboot); every run dir
# gets code_commit.txt. Speed gate = in-chain mapping time per frame (session_b/run.json seconds / frames), not process wall time:
# [K2] offline proxy on synthetic B -> 150-frame speed test baseline / K2 -> speed gate (K2 wall time <= 1.10 x baseline, the
# user's 10 % rule) -> full synthetic B mapping with K2 (refinement off) -> [K2] offline proxies on real C, A, B (the real gate
# before any real chain).
set -u
X=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$X"); R=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006
W=$R/worktrees; BASE=$W/final0638_b9400c9; K2=$W/static_kf_e37823e; PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python
Q=$REPO/runs_quick3k; L=$Q/k2_trial_queue.log; K=$(git -C $K2 rev-parse --short HEAD)
LOCK=/home/jixian/Desktop/FT/runs/GPU.lock
run() { local name=$1; local code=$2; shift 2; local D=$Q/${name}; mkdir -p $D; cd $D; git -C $code rev-parse HEAD > code_commit.txt
  echo "^start $name $(date '+%T')" >> $L; local t0=$(date +%s)
  flock $LOCK $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@"
  echo "^done $name $(date '+%T') seconds=$(( $(date +%s) - t0 )) $(cat status.txt)" >> $L; }
run k2off_synB_$K $K2 $PY -W ignore $X/k2_offline.py $K2 $R/syn_row4f3e5_r11p/checkpoint_b.pt synthetic b k2.jsonl
# GaME's own removal test at the chain's keyframes (dsa_offline.py): the evidence for turning own_update on, read before the full-B trial
run dsa_synB_$K $K2 $PY -W ignore $X/dsa_offline.py $K2 $R/syn_row4f3e5_r11p/checkpoint_b.pt synthetic b -
run dsa_realC_$K $K2 $PY -W ignore $X/dsa_offline.py $K2 $R/real_row4f3e5/checkpoint_c.pt real c /home/jixian/Desktop/FT/results/abc_eval_v2/geometry/C_reference_1cm.ply
run speed150_final0638_b9400c9 $BASE $PY -W ignore $X/quick_map.py $BASE $Q/speed150_final0638_b9400c9/map 150
run speed150_static_kf_$K $K2 $PY -W ignore $X/quick_map.py $K2 $Q/speed150_static_kf_$K/map 150
perframe() { python3 -c "import json,sys; j=json.load(open(sys.argv[1])); print(j['seconds']/max(j['frames'],1))" $1/map/session_b/run.json 2>/dev/null || echo 0; }
tb=$(perframe $Q/speed150_final0638_b9400c9); tk=$(perframe $Q/speed150_static_kf_$K)
if python3 -c "import sys; tb, tk = float('$tb'), float('$tk'); sys.exit(0 if tb > 0 and 0 < tk <= 1.10 * tb else 1)"; then
  echo "^gate speed PASS in-chain s/frame k2=$tk base=$tb" >> $L
  run mapB_static_kf_$K $K2 $PY -W ignore $X/quick_map.py $K2 $Q/mapB_static_kf_$K/map 0
else
  echo "^gate speed FAIL in-chain s/frame k2=$tk base=$tb: full synthetic B trial not started" >> $L
fi
for s in c a b; do run k2off_real${s}_$K $K2 $PY -W ignore $X/k2_offline.py $K2 $R/real_row4f3e5/checkpoint_$s.pt real $s k2.jsonl; done
echo "^queue end $(date '+%T')" >> $L
