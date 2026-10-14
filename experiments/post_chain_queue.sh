#!/usr/bin/env bash
# TODO 5, second part: the end-of-run refinement (F2) on the chains' pre_ref maps, the scoring, the view tools, then Flat.
#   post_chain_queue.sh REFINE_TREE        (worktree: r21c_base = r21, or r21c = r21 + GaME's SH ramp, by the SH 30K scores)
# Synthetic B and real C (each chain's last session) are refined with refine_only (the checkpoint is restored first: the chain
# queue stripped it); pre_ref and post_ref are scored as RUN_ALL scores them (synthetic: eval_synthetic_scene.sh, CPU only, but
# under GPU.lock so that it never runs beside a timed GPU job -- the mapping chains, the refinements, Flat -- supervisor 11:13;
# about 3 min each; real: eval_row.sh, GPU parts, under the lock; acceptance), the fixed view tools render the refined synthetic map.
# Flat: our chain (mapping + refinement in one process, as RUN_ALL's flat_ours) and GaME's official run on Flat.
# Every GPU step takes GPU.lock on its own. Markers in runs_quick3k/post_chain_queue.log.
set -u
RT=$(realpath "$1"); X=$(cd "$(dirname "$0")" && pwd); RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006
F=$RR/FINAL_20261014; P=/home/jixian/Desktop/FT/session_update_baseline_project; A=$RR/analysis
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=$RR/code_mine_20261014/runs_quick3k; L=$Q/post_chain_queue.log
LOCK=/home/jixian/Desktop/FT/runs/GPU.lock; M=$A/DISK_RECLAIM_20261013.txt; RK=$(git -C $RT rev-parse --short HEAD)
S=$RR/syn_chain_4ca6976/map; RL=$RR/real_chain_4ca6976/map; CQ=$Q/chain_queue.log
git -C $RT status --porcelain | grep -q . && { echo "^refused dirty $RT" >> $L; exit 3; }
used() { df --output=used -BG /home/jixian | tail -1 | tr -d ' G'; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
refine() {   # refine RUNMAP DATASET LAST OUT: OUT gets links to RUNMAP's sessions and the restored last checkpoint, then refine_only
  local SRC=$1 DS=$2 LS=$3 E=$4
  mkdir -p $E/session_$LS
  for d in $SRC/session_*; do s=${d##*_}; [ "$s" = "$LS" ] && continue; ln -sfn $d $E/session_$s
    [ -e $SRC/checkpoint_$s.pt ] && ln -sfn "$(readlink -f $SRC/checkpoint_$s.pt)" $E/checkpoint_$s.pt; done
  for f in $SRC/session_$LS/*; do b=$(basename $f); case $b in timeline_post_ref.pkl|tsdf_final.ply|run.json) continue;; esac; ln -sfn $f $E/session_$LS/$b; done
  cp -f $SRC/session_$LS/run.json $E/session_$LS/run.json; git -C $RT rev-parse HEAD > $E/code_commit.txt
  echo "^start refine $DS $LS $RK $(date '+%T') disk $(used) GB" >> $L
  ( cd $RT && flock $LOCK $PY -m update_layer.eval.kf_strip restore $SRC/checkpoint_$LS.pt $E/checkpoint_$LS.pt $DS ) > $E/restore_$LS.log 2>&1
  ( cd $RT && flock $LOCK $PY -W ignore -m update_layer.eval.refine_only $E $DS $LS ) > $E/refine.log 2>&1
  local rc=$?; echo "^done refine $DS $LS $(date '+%T') rc=$rc" >> $L
  if [ -s $E/checkpoint_${LS}_post_ref.pt ] && [ -f $E/checkpoint_$LS.pt ] && [ ! -L $E/checkpoint_$LS.pt ]; then
    echo "$(date '+%F %H:%M') restored pre-refinement copy of $SRC/checkpoint_$LS.pt (source kept, stripped), not read after the refinement: $(stat -c %s $E/checkpoint_$LS.pt) $E/checkpoint_$LS.pt" >> $M
    rm -- $E/checkpoint_$LS.pt; fi
  bash $RR/post_ref_view.sh $E $LS > $E/post_ref_view.log 2>&1; }

# ---- synthetic B
until grep -q "^\^tidied $S " $CQ 2>/dev/null; do sleep 30; done
ES=$RR/syn_chain_4ca6976/refine_$RK
refine $S synthetic b $ES
( echo "^queued synthetic scoring (CPU, under GPU.lock) $(date '+%T')" >> $L
  ( cd $RT && flock $LOCK bash update_layer/eval/eval_synthetic_scene.sh $S $S/eval_syn ) > $S/eval_b.log 2>&1
  ( cd $RT && flock $LOCK bash update_layer/eval/eval_synthetic_scene.sh $ES/post_ref $ES/post_ref/eval_syn ) > $ES/eval_post_b.log 2>&1
  echo "^done synthetic scoring $(date '+%T')" >> $L ) &
for s in views15_final bathroom_struct_final compensated_end_final heldout_last_final last_kf_final; do
  ( cd $RR && FINAL_CKPT=$ES/checkpoint_b_post_ref.pt FINAL_CODE=worktrees/$(basename $RT) FINAL_TAG=syn_chain_$RK flock $LOCK $PY -W ignore $F/tools/$s.py ) > $ES/$s.out 2>&1
done
echo "^done synthetic views $(date '+%T')" >> $L

# ---- real B -> C
until grep -q "^\^tidied $RL " $CQ 2>/dev/null; do sleep 30; done
# run_chain links checkpoint.pt to checkpoint_b.pt (a hard link, for resuming C); after the strip it alone keeps B's unstripped data
if [ -f $RL/checkpoint.pt ] && [ ! -L $RL/checkpoint.pt ]; then
  echo "$(date '+%F %H:%M') real chain: checkpoint.pt, the resume hard link to the unstripped checkpoint_b.pt (the chain is complete; checkpoint_b.pt is stripped): $(stat -c %s $RL/checkpoint.pt) $RL/checkpoint.pt" >> $M
  rm -- $RL/checkpoint.pt; fi
ER=$RR/real_chain_4ca6976/refine_$RK
refine $RL real c $ER
echo "^start real scoring $(date '+%T')" >> $L
( cd $RT && SES="a b c" UNDER_SIDE_GATE=1 timeout 4h flock $LOCK bash $RR/eval_row.sh $RL real ) > $RL/eval_abc.out 2>&1
( cd $RT && SES="c" UNDER_SIDE_GATE=1 timeout 4h flock $LOCK bash $RR/eval_row.sh $ER/post_ref real ) > $ER/eval_post_c.out 2>&1
( cd $P && PYTHONPATH=. $PY -m update_layer.eval.acceptance real real_chain_4ca6976/map --ref real_row4k --out $A/acceptance_real_chain_4ca6976.json > $A/acceptance_real_chain_4ca6976.txt 2>&1 )
echo "^done real scoring $(date '+%T')" >> $L
wait

# ---- Flat: ours (mapping + refinement, one process, as RUN_ALL flat_ours) and GaME's official run
FO=$RR/flat_chain_$RK; mkdir -p $FO; git -C $RT rev-parse HEAD > $FO/code_commit.txt
echo "^start flat ours $(date '+%T') disk $(used) GB" >> $L
( cd $RT && flock $LOCK $PY -m update_layer.run --backend game --row 4 --core l2 --d1 --g8-holdout --dataset flat --out $FO ) > $FO/chain.log 2>&1
echo "^done flat ours mapping $(date '+%T') rc=$?" >> $L
[ -s $FO/session_b/timeline_post_ref.pkl ] && bash $RR/post_ref_view.sh $FO b > $FO/post_ref_view.log 2>&1
for v in pre_ref post_ref; do src=$FO; [ $v = post_ref ] && src=$FO/post_ref
  [ -d $src ] && ( cd $RT && flock $LOCK $PY -W ignore -m update_layer.eval.game_render_metrics $src b flat $FO/render_metrics_$v.json ) > $FO/render_metrics_$v.log 2>&1; done
echo "^done flat ours $(date '+%T')" >> $L
D=/home/jixian/Desktop/FT/runs/game_flat_official_20261007
echo "^start flat native $(date '+%T')" >> $L
( cd $D && [ -f run.log ] && mv -f run.log run_before_$(date +%H%M).log; rm -rf $D/out; mkdir -p $D/out; flock $LOCK bash $D/run_flat.sh ) > $Q/flat_native_stdout.log 2>&1
echo "^done flat native $(date '+%T') $(grep -m1 '^exit' $D/run.log)" >> $L
echo "^post chain queue end $(date '+%T')" >> $L
