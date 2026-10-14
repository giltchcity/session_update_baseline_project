#!/usr/bin/env bash
# Rest of the K2 queue after its collapse at 08:58:56 (rule-1 gate refused every job while the harness repo was dirty). Runs from a
# PINNED harness worktree (worktrees/harness_<commit>) so that edits in code_mine cannot refuse its jobs; results still go to /home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/code_mine_20261014/runs_quick3k.
# Real K2 proxies dropped: on the stored end-of-session map the synthetic proxy promoted 47 % of the frames from frame 1 (the plant's
# B-start rows are not in that map) -- a proxy bias, not a K2 property; the real gate is the count of K2 lines in a real chain's log.
set -u
X=$(cd "$(dirname "$0")" && pwd); W=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/worktrees; K2=$W/static_kf_e37823e; BASE=$W/final0638_b9400c9
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/code_mine_20261014/runs_quick3k; L=$Q/k2_trial_queue.log; LOCK=/home/jixian/Desktop/FT/runs/GPU.lock; K=e37823e
run() { local name=$1; local code=$2; shift 2; local D=$Q/${name}; mkdir -p $D; cd $D; git -C $code rev-parse HEAD > code_commit.txt
  echo "^start $name $(date '+%T')" >> $L; local t0=$(date +%s)
  flock $LOCK $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@"
  echo "^done $name $(date '+%T') seconds=$(( $(date +%s) - t0 )) $(cat status.txt 2>/dev/null)" >> $L; }
run dsa_realC_$K $K2 $PY -W ignore $X/dsa_offline.py $K2 /home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/real_row4f3e5/checkpoint_c.pt real c /home/jixian/Desktop/FT/results/abc_eval_v2/geometry/C_reference_1cm.ply
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
echo "^queue2 end $(date '+%T')" >> $L
