#!/usr/bin/env bash
# K2 queue, part 3 (pinned harness): wait for the running speed150 baseline, then the K2 speed test, a 30-frame REAL smoke of
# quick_map's real entry (bugs show now, not at 11:30), the in-chain per-frame speed gate, and the full synthetic B K2 trial.
set -u
X=$(cd "$(dirname "$0")" && pwd); W=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/worktrees; K2=$W/static_kf_e37823e; BASE=$W/final0638_b9400c9
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/code_mine_20261014/runs_quick3k; L=$Q/k2_trial_queue.log; LOCK=/home/jixian/Desktop/FT/runs/GPU.lock; K=e37823e
run() { local name=$1; local code=$2; shift 2; local D=$Q/${name}; mkdir -p $D; cd $D; git -C $code rev-parse HEAD > code_commit.txt
  echo "^start $name $(date '+%T')" >> $L; local t0=$(date +%s)
  flock $LOCK $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@"
  echo "^done $name $(date '+%T') seconds=$(( $(date +%s) - t0 )) $(cat status.txt 2>/dev/null)" >> $L; }
until [ -s $Q/speed150_final0638_b9400c9/status.txt ]; do sleep 20; done
echo "^done speed150_final0638_b9400c9 (orphaned job of queue2) $(date '+%T') $(cat $Q/speed150_final0638_b9400c9/status.txt)" >> $L
run speed150_static_kf_$K $K2 $PY -W ignore $X/quick_map.py $K2 $Q/speed150_static_kf_$K/map 150
run smoke30_real_static_kf_$K $K2 $PY -W ignore $X/quick_map.py $K2 $Q/smoke30_real_static_kf_$K/map 30 real
perframe() { python3 -c "import json,sys; j=json.load(open(sys.argv[1])); print(j['seconds']/max(j['frames'],1))" $1/map/session_b/run.json 2>/dev/null || echo 0; }
tb=$(perframe $Q/speed150_final0638_b9400c9); tk=$(perframe $Q/speed150_static_kf_$K)
if python3 -c "import sys; tb, tk = float('$tb'), float('$tk'); sys.exit(0 if tb > 0 and 0 < tk <= 1.10 * tb else 1)"; then
  echo "^gate speed PASS in-chain s/frame k2=$tk base=$tb" >> $L
  run mapB_static_kf_$K $K2 $PY -W ignore $X/quick_map.py $K2 $Q/mapB_static_kf_$K/map 0
else
  echo "^gate speed FAIL in-chain s/frame k2=$tk base=$tb: full synthetic B trial not started" >> $L
fi
echo "^queue3 end $(date '+%T')" >> $L
