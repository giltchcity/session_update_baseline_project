#!/usr/bin/env bash
# K3 (static_kf3 014294c: corrected new-geometry test) on the GPU with a paired speed gate (supervisor 09:55 / 10:00). GPU.lock is taken once (after the
# SH refinement step) and released after the pair: the SH run's restored pre-refinement checkpoint is deleted (its later steps read the
# post_ref files only; the queue's own last step deletes it too) to keep the disk well below 450 GB; the K3 offline processes are paused (SIGSTOP) so the pair runs on a quiet CPU with no GPU waiting, and resumed
# (SIGCONT) right after; baseline arm (final0638 b9400c9) and K3 arm, 150 frames each, back to back, load average recorded, each arm's
# checkpoints deleted right after it; gate = K3 <= 1.10 x the baseline of the same pair. The full synthetic B trial is started separately
# after the offline verdict. Markers in runs_quick3k/k3_trial_queue.log.
set -u
X=$(cd "$(dirname "$0")" && pwd); RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; W=$RR/worktrees; K3=$W/static_kf3; BASE=$W/final0638_b9400c9
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=$RR/code_mine_20261014/runs_quick3k; M=$RR/analysis/DISK_RECLAIM_20261013.txt
L=$Q/k3_trial_queue.log; LOCK=/home/jixian/Desktop/FT/runs/GPU.lock; K=$(git -C $K3 rev-parse --short HEAD); B=$(git -C $BASE rev-parse --short HEAD)
for c in $K3 $BASE; do git -C $c status --porcelain | grep -q . && { echo "^refused dirty $c" >> $L; exit 3; }; done
run() { local name=$1; local code=$2; shift 2; local D=$Q/${name}; mkdir -p $D; cd $D; git -C $code rev-parse HEAD > code_commit.txt
  echo "^start $name $(date '+%T') load $(cut -d' ' -f1-3 /proc/loadavg)" >> $L; local t0=$(date +%s)
  $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@"
  echo "^done $name $(date '+%T') seconds=$(( $(date +%s) - t0 )) $(cat status.txt 2>/dev/null) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $L; }
exec 9>$LOCK
echo "^waiting for GPU.lock $(date '+%T')" >> $L
flock 9
echo "^GPU.lock held from $(date '+%T')" >> $L
E=$RR/exp_sh_r11p_14f3855
if grep -q '^\^done sh30k refine' $Q/sh_queue.log && [ -s $E/checkpoint_b_post_ref.pt ] && [ -f $E/checkpoint_b.pt ] && [ ! -L $E/checkpoint_b.pt ]; then
  echo "$(date '+%F %H:%M') SH 30K run: restored pre-refinement copy of syn_row4f3e5_r11p/checkpoint_b.pt (source kept), not read after the refinement: $(stat -c %s $E/checkpoint_b.pt) $E/checkpoint_b.pt" >> $M
  rm -- $E/checkpoint_b.pt; echo "^deleted $E/checkpoint_b.pt (ledger) $(date '+%T')" >> $L
fi
P=$(pgrep -f 'python -W ignore .*k3_offline.py' | tr '\n' ' ')
[ -n "$P" ] && { kill -STOP $P; echo "^paused K3 offline processes $P $(date '+%T')" >> $L; }
run speed150_pair_base_$B $BASE $PY -W ignore $X/quick_map.py $BASE $Q/speed150_pair_base_$B/map 150; $X/clean_speed_ckpts.sh speed150_pair_base_$B
run speed150_pair_k3_$K $K3 $PY -W ignore $X/quick_map.py $K3 $Q/speed150_pair_k3_$K/map 150; $X/clean_speed_ckpts.sh speed150_pair_k3_$K
[ -n "$P" ] && { kill -CONT $P; echo "^resumed K3 offline processes $P $(date '+%T')" >> $L; }
perframe() { python3 -c "import json,sys; j=json.load(open(sys.argv[1])); print(j['seconds']/max(j['frames'],1))" $1/map/session_b/run.json 2>/dev/null || echo 0; }
tb=$(perframe $Q/speed150_pair_base_$B); tk=$(perframe $Q/speed150_pair_k3_$K)
n3=$(grep -c "K3 scene change" $Q/speed150_pair_k3_$K/run.log 2>/dev/null)
if python3 -c "import sys; tb, tk = float('$tb'), float('$tk'); sys.exit(0 if tb > 0 and 0 < tk <= 1.10 * tb else 1)"; then
  echo "^gate speed PASS (paired) in-chain s/frame k3=$tk base=$tb (K3 keyframes in 150 frames: $n3)" >> $L
else
  echo "^gate speed FAIL (paired) in-chain s/frame k3=$tk base=$tb (K3 keyframes in 150 frames: $n3)" >> $L
fi
exec 9>&-
echo "^GPU.lock released; k3v5 queue end $(date '+%T')" >> $L
