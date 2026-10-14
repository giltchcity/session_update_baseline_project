#!/usr/bin/env bash
# TODO 6 (supervisor 09:43: start now, the GPU is idle): the 30K end-of-run refinement of the R11p synthetic B pre-refinement map
# with r21 + SH ramp (r21c 14f3855; its 4K speed gate passed 09:35: 0.0661 vs 0.065 s/step, degree 3), scored as RUN_ALL's control
# stage: post_ref view, official synthetic evaluation, views15 / end / last-kf tools. Every step takes GPU.lock on its own, so
# other short GPU jobs (the K3 speed test) can run between steps. Markers in runs_quick3k/sh_queue.log.
set -u
R=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; F=$R/FINAL_20261014; W=$R/worktrees
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=$R/code_mine_20261014/runs_quick3k; L=$Q/sh_queue.log; LOCK=/home/jixian/Desktop/FT/runs/GPU.lock
SH=$(git -C $W/r21c rev-parse --short HEAD); git -C $W/r21c status --porcelain | grep -q . && { echo "^refused dirty r21c" >> $L; exit 3; }
S=$R/syn_row4f3e5_r11p; E=$R/exp_sh_r11p_$SH; mkdir -p $E/session_b; git -C $W/r21c rev-parse HEAD > $E/code_commit.txt
ln -sfn "$S/session_a" "$E/session_a"; ln -sfn "$(readlink -f "$S/checkpoint_a.pt")" "$E/checkpoint_a.pt"
for f in "$S/session_b/"*; do b=$(basename "$f"); case $b in timeline_post_ref.pkl|tsdf_final.ply|run.json) continue;; esac; ln -sfn "$f" "$E/session_b/$b"; done
cp -f "$S/session_b/run.json" "$E/session_b/run.json"
echo "^start sh30k $SH $(date '+%T')" >> $L
( cd $W/r21c && flock $LOCK $PY -m update_layer.eval.kf_strip restore "$S/checkpoint_b.pt" "$E/checkpoint_b.pt" synthetic ) > $E/restore_b.log 2>&1
echo "^done sh30k restore $(date '+%T') rc=$?" >> $L
( cd $W/r21c && flock $LOCK env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -W ignore -m update_layer.eval.refine_only "$E" synthetic b ) > $E/refine.log 2>&1
rc=$?; echo "^done sh30k refine $(date '+%T') rc=$rc" >> $L
bash "$R/post_ref_view.sh" "$E" b > "$E/post_ref_view.log" 2>&1
( cd $W/r21c && timeout 4h flock $LOCK bash update_layer/eval/eval_synthetic_scene.sh "$E/post_ref" "$E/post_ref/eval_syn" ) > "$E/eval_post_b.log" 2>&1
echo "^done sh30k eval $(date '+%T') rc=$?" >> $L
for s in views15_final compensated_end_final last_kf_final; do
  ( cd "$R" && FINAL_CKPT="$E/checkpoint_b_post_ref.pt" FINAL_CODE="worktrees/r21c" FINAL_TAG=syn_sh flock $LOCK $PY -W ignore "$F/tools/$s.py" ) > "$E/$s.out" 2>&1; done
rm -f "$E/checkpoint_b.pt"
echo "^done sh30k all $(date '+%T')" >> $L; echo "^sh30k queue end $(date '+%T')" >> $L
