#!/usr/bin/env bash
# K3 efficiency version (static_kf3) on the GPU with a PAIRED speed gate (supervisor 09:55): the GPU lock is taken once and held
# for: the equivalence check against 7d2db97 (k3_equiv.py); a wait until the K3 offline counts have ended (no CPU job beside the
# pair); the baseline arm (final0638 b9400c9) and the K3 arm, 150 frames each, back to back, load average recorded; the gate
# (K3 <= 1.10 x baseline of the same pair); then, after '^k3 verdict PASS' in k3_queue.log, the full synthetic B trial.
# The speed arms' checkpoints are deleted right after each arm (ledger). Markers in runs_quick3k/k3_trial_queue.log.
set -u
X=$(cd "$(dirname "$0")" && pwd); W=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/worktrees; K3=$W/static_kf3; BASE=$W/final0638_b9400c9
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/code_mine_20261014/runs_quick3k
L=$Q/k3_trial_queue.log; LOCK=/home/jixian/Desktop/FT/runs/GPU.lock; K=$(git -C $K3 rev-parse --short HEAD); B=$(git -C $BASE rev-parse --short HEAD)
for c in $K3 $BASE; do git -C $c status --porcelain | grep -q . && { echo "^refused dirty $c" >> $L; exit 3; }; done
run() { local name=$1; local code=$2; shift 2; local D=$Q/${name}; mkdir -p $D; cd $D; git -C $code rev-parse HEAD > code_commit.txt
  echo "^start $name $(date '+%T') load $(cut -d' ' -f1-3 /proc/loadavg)" >> $L; local t0=$(date +%s)
  $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@"
  echo "^done $name $(date '+%T') seconds=$(( $(date +%s) - t0 )) $(cat status.txt 2>/dev/null) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $L; }
clean() { $X/clean_speed_ckpts.sh "$@"; }
exec 9>$LOCK
echo "^waiting for GPU.lock $(date '+%T')" >> $L
flock 9
echo "^GPU.lock held from $(date '+%T') until this queue ends" >> $L
OLD=$Q/k3_equiv_$K/old_scene_change_7d2db97.py; mkdir -p $Q/k3_equiv_$K; git -C $K3 show 7d2db97:update_layer/backends/game/scene_change.py > $OLD
run k3_equiv_$K $K3 $PY -W ignore $X/k3_equiv.py $K3 $OLD
until grep -q '^\^k3 queue end' $Q/k3_queue.log 2>/dev/null; do sleep 15; done
echo "^CPU offline jobs ended $(date '+%T') load $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
run speed150_pair_base_$B $BASE $PY -W ignore $X/quick_map.py $BASE $Q/speed150_pair_base_$B/map 150; clean speed150_pair_base_$B
run speed150_pair_k3_$K $K3 $PY -W ignore $X/quick_map.py $K3 $Q/speed150_pair_k3_$K/map 150; clean speed150_pair_k3_$K
perframe() { python3 -c "import json,sys; j=json.load(open(sys.argv[1])); print(j['seconds']/max(j['frames'],1))" $1/map/session_b/run.json 2>/dev/null || echo 0; }
tb=$(perframe $Q/speed150_pair_base_$B); tk=$(perframe $Q/speed150_pair_k3_$K)
n3=$(grep -c "K3 scene change" $Q/speed150_pair_k3_$K/run.log 2>/dev/null)
if python3 -c "import sys; tb, tk = float('$tb'), float('$tk'); sys.exit(0 if tb > 0 and 0 < tk <= 1.10 * tb else 1)"; then
  echo "^gate speed PASS (paired) in-chain s/frame k3=$tk base=$tb ratio=$(python3 -c "print(round($tk/$tb,4))") (K3 keyframes in 150 frames: $n3)" >> $L
else
  echo "^gate speed FAIL (paired) in-chain s/frame k3=$tk base=$tb (K3 keyframes in 150 frames: $n3): full synthetic B trial not started" >> $L
  echo "^k3v3 queue end $(date '+%T')" >> $L; exit 0
fi
until grep -q '^\^k3 verdict' $Q/k3_queue.log 2>/dev/null; do sleep 15; done
if grep -q '^\^k3 verdict PASS' $Q/k3_queue.log; then
  run mapB_static_kf3_$K $K3 $PY -W ignore $X/quick_map.py $K3 $Q/mapB_static_kf3_$K/map 0
else
  echo "^offline verdict not PASS: full synthetic B trial not started" >> $L
fi
echo "^k3v3 queue end $(date '+%T')" >> $L
