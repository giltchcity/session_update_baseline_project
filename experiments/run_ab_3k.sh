#!/usr/bin/env bash
# A/B 3K on the same input, one GPU job at a time: arm "mine" = this repository's HEAD; arm "rt" = the checkout given as $1
# (the peer's final code with only the dataset-keyed export_e1 flag removed). Run dirs: runs_quick3k/<arm>_<commit>.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd); RT=$1
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python
OUT=$REPO/runs_quick3k; mkdir -p $OUT
run() {
  local arm=$1 code=$2; local dir=$OUT/${arm}_$(git -C $code rev-parse --short HEAD)
  mkdir -p $dir
  echo "^start $arm $dir $(date '+%T')" >> $OUT/queue.log
  flock /home/jixian/Desktop/FT/runs/GPU.lock env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -W ignore $REPO/experiments/quick_screen.py short $arm synthetic $dir $code > $dir/run.log 2>&1
  echo "^done $arm rc=$? $(date '+%T')" >> $OUT/queue.log
}
run mine $REPO
run rt $RT
