#!/usr/bin/env bash
# The rest of sh30k_queue.sh after its refinement step, without holding the GPU lock where none is needed (supervisor 10:08):
# the restored pre-refinement copy is deleted (ledger; only the post_ref files are read from here on), the post_ref view is
# built, the official synthetic scoring runs on the CPU now (3d_vsg python: native export + geometry/state scorers, no GPU),
# then the three view tools (GPU renders) take GPU.lock between the chain's steps. Markers in runs_quick3k/sh_queue.log.
set -u
R=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; F=$R/FINAL_20261014; W=$R/worktrees; E=$R/exp_sh_r11p_14f3855
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=$R/code_mine_20261014/runs_quick3k; L=$Q/sh_queue.log; LOCK=/home/jixian/Desktop/FT/runs/GPU.lock
M=$R/analysis/DISK_RECLAIM_20261013.txt
[ -s $E/checkpoint_b_post_ref.pt ] && [ -s $E/session_b/timeline_post_ref.pkl ] || { echo "^sh30k rest: no post_ref output" >> $L; exit 3; }
if [ -f $E/checkpoint_b.pt ] && [ ! -L $E/checkpoint_b.pt ]; then
  echo "$(date '+%F %H:%M') SH 30K run: restored pre-refinement copy of syn_row4f3e5_r11p/checkpoint_b.pt (source kept), not read after the refinement: $(stat -c %s $E/checkpoint_b.pt) $E/checkpoint_b.pt" >> $M
  rm -- $E/checkpoint_b.pt; fi
bash "$R/post_ref_view.sh" "$E" b > "$E/post_ref_view.log" 2>&1
echo "^start sh30k eval (CPU, no lock) $(date '+%T')" >> $L
( cd $W/r21c && bash update_layer/eval/eval_synthetic_scene.sh "$E/post_ref" "$E/post_ref/eval_syn" ) > "$E/eval_post_b.log" 2>&1
echo "^done sh30k eval $(date '+%T') rc=$?" >> $L
for s in views15_final compensated_end_final last_kf_final; do
  ( cd "$R" && FINAL_CKPT="$E/checkpoint_b_post_ref.pt" FINAL_CODE="worktrees/r21c" FINAL_TAG=syn_sh flock $LOCK $PY -W ignore "$F/tools/$s.py" ) > "$E/$s.out" 2>&1; done
echo "^done sh30k all $(date '+%T')" >> $L; echo "^sh30k rest end $(date '+%T')" >> $L
