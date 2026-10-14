#!/usr/bin/env bash
# K3 on the GPU (supervisor 09:21): after the SH speed queue, the 150-frame speed test of K3 (static_kf3) on synthetic B and a
# 30-frame real smoke; the in-chain per-frame speed gate (<= 1.10 x the baseline speed150_final0638_b9400c9, 1.531 s/frame); then
# the full synthetic B mapping trial, only when the gate passed AND the offline verdict (written by Claude into k3_queue.log as
# '^k3 verdict PASS' after reading the three offline counts) passed. Markers in runs_quick3k/k3_trial_queue.log.
set -u
X=$(cd "$(dirname "$0")" && pwd); W=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/worktrees; K3=$W/static_kf3
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/code_mine_20261014/runs_quick3k
L=$Q/k3_trial_queue.log; LOCK=/home/jixian/Desktop/FT/runs/GPU.lock; K=$(git -C $K3 rev-parse --short HEAD)
git -C $K3 status --porcelain | grep -q . && { echo "^refused dirty static_kf3" >> $L; exit 3; }
run() { local name=$1; local code=$2; shift 2; local D=$Q/${name}; mkdir -p $D; cd $D; git -C $code rev-parse HEAD > code_commit.txt
  echo "^start $name $(date '+%T')" >> $L; local t0=$(date +%s)
  flock $LOCK $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@"
  echo "^done $name $(date '+%T') seconds=$(( $(date +%s) - t0 )) $(cat status.txt 2>/dev/null)" >> $L; }
until grep -q '^\^sh speed queue end' $Q/sh_queue.log 2>/dev/null; do sleep 20; done
run speed150_static_kf3_$K $K3 $PY -W ignore $X/quick_map.py $K3 $Q/speed150_static_kf3_$K/map 150
run smoke30_real_static_kf3_$K $K3 $PY -W ignore $X/quick_map.py $K3 $Q/smoke30_real_static_kf3_$K/map 30 real
perframe() { python3 -c "import json,sys; j=json.load(open(sys.argv[1])); print(j['seconds']/max(j['frames'],1))" $1/map/session_b/run.json 2>/dev/null || echo 0; }
tb=$(perframe $Q/speed150_final0638_b9400c9); tk=$(perframe $Q/speed150_static_kf3_$K)
n3=$(grep -c "K3 scene change" $Q/speed150_static_kf3_$K/run.log 2>/dev/null)
if python3 -c "import sys; tb, tk = float('$tb'), float('$tk'); sys.exit(0 if tb > 0 and 0 < tk <= 1.10 * tb else 1)"; then
  echo "^gate speed PASS in-chain s/frame k3=$tk base=$tb (K3 keyframes in 150 frames: $n3)" >> $L
else
  echo "^gate speed FAIL in-chain s/frame k3=$tk base=$tb (K3 keyframes in 150 frames: $n3): full synthetic B trial not started" >> $L
  echo "^k3 trial queue end $(date '+%T')" >> $L; exit 0
fi
until grep -q '^\^k3 verdict' $Q/k3_queue.log 2>/dev/null; do sleep 30; done
if grep -q '^\^k3 verdict PASS' $Q/k3_queue.log; then
  run mapB_static_kf3_$K $K3 $PY -W ignore $X/quick_map.py $K3 $Q/mapB_static_kf3_$K/map 0
else
  echo "^offline verdict not PASS: full synthetic B trial not started" >> $L
fi
echo "^k3 trial queue end $(date '+%T')" >> $L
