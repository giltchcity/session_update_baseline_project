#!/usr/bin/env bash
# Flat smoke before the full Flat runs (supervisor 14:44): does each pipeline run through and give its metrics? Numbers are not results.
#  native: GaME's run.py, published config with refinement_iters 0, on a 40-frame copy of each run (symlinks to game_flat_hf: GaME's
#          Flat loader has no frame cap); run2's test split of that copy (frames 10, 20, 30) is scored by GaME itself.
#  ours:   the post-chain queue's Flat tree (worktrees/r21c, the refinement tree of the decision) through FINAL's map_smoke.py, 40
#          frames per run, 30 refinement steps, then post_ref_view and game_render_metrics pre / post (as RUN_ALL's flat_ours).
# Each GPU step under GPU.lock.
set -u
RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; FT=/home/jixian/Desktop/FT; PY=$FT/envs/gs-cu128/bin/python
LOCK=$FT/runs/GPU.lock; Q=$RR/code_mine_20261014/runs_quick3k; L=$Q/flat_smoke_queue.log; S=$RR/flat_smoke_20261014
mkdir -p $S/data/run1 $S/data/run2
ln -sfn $FT/datasets/game_flat_hf/Intrinsics.txt $S/data/Intrinsics.txt
for r in run1 run2; do
  for i in $(seq -f '%06g' 0 39); do for k in color.png depth.tiff pose.txt; do ln -sfn $FT/datasets/game_flat_hf/$r/${i}_$k $S/data/$r/${i}_$k; done; done
  ln -sfn $FT/datasets/game_flat_hf/$r/sam_masks.h5 $S/data/$r/sam_masks.h5
done
sed 's/^  refinement_iters: .*/  refinement_iters: 0/' $FT/runs/game_flat_official_20261007/flat_official.yaml > $S/flat_smoke_native.yaml
diff $FT/runs/game_flat_official_20261007/flat_official.yaml $S/flat_smoke_native.yaml > $S/flat_smoke_native.diff
echo "^start flat smoke native $(date '+%T')" >> $L
( cd $FT/baselines/GaME && WANDB_MODE=disabled flock $LOCK $PY run.py --config_path $S/flat_smoke_native.yaml --data_path $S/data \
    --output_path $S/native_out/ ) > $S/native.log 2>&1
echo "^done flat smoke native $(date '+%T') rc=$?" >> $L
grep -iE "psnr|ssim|lpips|depth" $S/native.log | tail -6 | sed 's/^/  native: /' >> $L
O=$S/ours; rm -rf $O; mkdir -p $O; git -C $RR/worktrees/r21c rev-parse HEAD > $O/code_commit.txt
echo "^start flat smoke ours $(date '+%T')" >> $L
( cd $RR/worktrees/r21c && flock $LOCK env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -W ignore \
    $RR/FINAL_20261014/tools/smoke/map_smoke.py worktrees/r21c flat $O 40 30 ) > $O/chain.log 2>&1
echo "^done flat smoke ours chain $(date '+%T') rc=$?" >> $L
bash $RR/post_ref_view.sh $O b >> $O/post_ref_view.log 2>&1
for v in pre_ref post_ref; do src=$O; [ $v = post_ref ] && src=$O/post_ref
  ( cd $RR/worktrees/r21c && flock $LOCK env OMP_NUM_THREADS=2 $PY -W ignore -m update_layer.eval.game_render_metrics $src b flat \
      $O/render_metrics_$v.json ) > $O/render_metrics_$v.log 2>&1
  echo "^done flat smoke ours metrics $v $(date '+%T') rc=$? $(head -c 300 $O/render_metrics_$v.json 2>/dev/null)" >> $L
done
for f in $(ls $O/checkpoint_*.pt $O/session_*/timeline*.pkl 2>/dev/null); do [ -L "$f" ] || { echo "$(date '+%F %H:%M') Flat smoke (ours, 40 frames per run; logs and metrics kept): $(stat -c %s $f) $f" >> $RR/analysis/DISK_RECLAIM_20261013.txt; rm -- "$f"; }; done
echo "^flat smoke end $(date '+%T')" >> $L
