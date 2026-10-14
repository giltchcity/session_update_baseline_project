#!/usr/bin/env bash
# TODO 5, mapping part (supervisor 10:07): the full chains with the final mapping and the r21 refinement tree (r21c_base 4ca6976 =
# final0638 mapping + r21 refinement rules), refinement OFF here (quick_map.py: refinement_iters 0); the refinement runs afterwards on
# the saved pre-refinement maps (refine_only), r21 or r21+SH as the SH 30K scores decide.
#   synthetic: B from the accepted A (syn_row4s0), all frames;  real: B and C from the accepted A (real_row4k), all frames.
# After each chain: the 0-step post_ref copies are deleted (ledger) and the checkpoints are stripped (kf_strip, as RUN_ALL). Every GPU
# step takes GPU.lock on its own (other short jobs can run between steps). Markers in runs_quick3k/chain_queue.log.
set -u
X=$(cd "$(dirname "$0")" && pwd); RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; W=$RR/worktrees; CODE=$W/r21c_base
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=$RR/code_mine_20261014/runs_quick3k; L=$Q/chain_queue.log
LOCK=/home/jixian/Desktop/FT/runs/GPU.lock; M=$RR/analysis/DISK_RECLAIM_20261013.txt; K=$(git -C $CODE rev-parse --short HEAD)
git -C $CODE status --porcelain | grep -q . && { echo "^refused dirty $CODE" >> $L; exit 3; }
used() { df --output=used -BG /home/jixian | tail -1 | tr -d ' G'; }
run() { local name=$1; shift; local D=$RR/$name; mkdir -p $D; cd $D; git -C $CODE rev-parse HEAD > code_commit.txt
  echo "^start $name $(date '+%T') disk $(used) GB" >> $L; local t0=$(date +%s)
  flock $LOCK $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@"
  echo "^done $name $(date '+%T') seconds=$(( $(date +%s) - t0 )) $(cat status.txt 2>/dev/null) disk $(used) GB" >> $L; }
tidy() { local MAP=$1 DS=$2; shift 2
  for f in $(ls $MAP/checkpoint_*_post_ref.pt $MAP/session_*/timeline_post_ref.pkl 2>/dev/null); do
    [ -L "$f" ] && continue
    echo "$(date '+%F %H:%M') 0-step post_ref copy of a refinement-off mapping chain (refinement runs separately on the pre_ref checkpoint): $(stat -c %s $f) $f" >> $M; rm -- "$f"; done
  for s in "$@"; do [ -s $MAP/checkpoint_$s.pt ] && [ ! -L $MAP/checkpoint_$s.pt ] && ( cd $CODE && nice -n 5 flock $LOCK $PY -m update_layer.eval.kf_strip strip $MAP/checkpoint_$s.pt $MAP/checkpoint_$s.pt $DS ) > $MAP/strip_$s.log 2>&1; done
  echo "^tidied $MAP $(date '+%T') disk $(used) GB" >> $L; }
[ "$(used)" -lt 420 ] || { echo "^refused: disk $(used) GB before the synthetic chain" >> $L; exit 4; }
run syn_chain_$K $PY -W ignore $X/quick_map.py $CODE $RR/syn_chain_$K/map 0 synthetic b
tidy $RR/syn_chain_$K/map synthetic b
[ "$(used)" -lt 420 ] || { echo "^refused: disk $(used) GB before the real chain" >> $L; exit 4; }
run real_chain_$K $PY -W ignore $X/quick_map.py $CODE $RR/real_chain_$K/map 0 real bc
tidy $RR/real_chain_$K/map real b c
echo "^chain queue end $(date '+%T')" >> $L
