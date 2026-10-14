#!/usr/bin/env bash
# K3 efficiency version (static_kf3 adf3e74) on the GPU, slotted between the SH 30K steps (supervisor 09:43): the equivalence check
# against 7d2db97's full test with per-frame timing (k3_equiv.py), the 150-frame speed test (load average recorded), the in-chain
# speed gate (<= 1.10 x baseline 1.531 s/frame), then the full synthetic B trial once the offline verdict '^k3 verdict PASS' is in
# k3_queue.log. Markers in runs_quick3k/k3_trial_queue.log.
set -u
X=$(cd "$(dirname "$0")" && pwd); W=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/worktrees; K3=$W/static_kf3
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/code_mine_20261014/runs_quick3k
L=$Q/k3_trial_queue.log; LOCK=/home/jixian/Desktop/FT/runs/GPU.lock; K=$(git -C $K3 rev-parse --short HEAD)
git -C $K3 status --porcelain | grep -q . && { echo "^refused dirty static_kf3" >> $L; exit 3; }
run() { local name=$1; local code=$2; shift 2; local D=$Q/${name}; mkdir -p $D; cd $D; git -C $code rev-parse HEAD > code_commit.txt
  echo "^start $name $(date '+%T') load $(cut -d' ' -f1-3 /proc/loadavg)" >> $L; local t0=$(date +%s)
  flock $LOCK $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@"
  echo "^done $name $(date '+%T') seconds=$(( $(date +%s) - t0 )) $(cat status.txt 2>/dev/null) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $L; }
OLD=$Q/k3_equiv_$K/old_scene_change_7d2db97.py; mkdir -p $Q/k3_equiv_$K; git -C $K3 show 7d2db97:update_layer/backends/game/scene_change.py > $OLD
run k3_equiv_$K $K3 $PY -W ignore $X/k3_equiv.py $K3 $OLD
run speed150_static_kf3_$K $K3 $PY -W ignore $X/quick_map.py $K3 $Q/speed150_static_kf3_$K/map 150
perframe() { python3 -c "import json,sys; j=json.load(open(sys.argv[1])); print(j['seconds']/max(j['frames'],1))" $1/map/session_b/run.json 2>/dev/null || echo 0; }
tb=$(perframe $Q/speed150_final0638_b9400c9); tk=$(perframe $Q/speed150_static_kf3_$K)
n3=$(grep -c "K3 scene change" $Q/speed150_static_kf3_$K/run.log 2>/dev/null)
if python3 -c "import sys; tb, tk = float('$tb'), float('$tk'); sys.exit(0 if tb > 0 and 0 < tk <= 1.10 * tb else 1)"; then
  echo "^gate speed PASS in-chain s/frame k3=$tk base=$tb (K3 keyframes in 150 frames: $n3)" >> $L
else
  echo "^gate speed FAIL in-chain s/frame k3=$tk base=$tb (K3 keyframes in 150 frames: $n3): full synthetic B trial not started" >> $L
  echo "^k3v2 queue end $(date '+%T')" >> $L; exit 0
fi
until grep -q '^\^k3 verdict' $Q/k3_queue.log 2>/dev/null; do sleep 30; done
if grep -q '^\^k3 verdict PASS' $Q/k3_queue.log; then
  run mapB_static_kf3_$K $K3 $PY -W ignore $X/quick_map.py $K3 $Q/mapB_static_kf3_$K/map 0
else
  echo "^offline verdict not PASS: full synthetic B trial not started" >> $L
fi
echo "^k3v2 queue end $(date '+%T')" >> $L
