#!/usr/bin/env bash
# The bathroom structure tool (FINAL_20261014/tools/bathroom_struct_final.py) on the SH 30K run's post_ref, which the SH view set
# (views15 / compensated_end / last_kf) does not include; needed for the seven-item 'not worse than pre-ref' table (supervisor 12:04).
# Runs after the SH decision, under GPU.lock (~30 s), only if the SH post_ref still exists (it is deleted when SH loses).
set -u
RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; Q=$RR/code_mine_20261014/runs_quick3k; L=$Q/sh_queue.log
E=$RR/exp_sh_r11p_14f3855; PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; LOCK=/home/jixian/Desktop/FT/runs/GPU.lock
until grep -q '^\^sh decision' $L 2>/dev/null; do sleep 30; done
[ -s $E/checkpoint_b_post_ref.pt ] || { echo "^sh bathroom_struct skipped: no SH post_ref $(date '+%T')" >> $L; exit 0; }
echo "^start sh bathroom_struct $(date '+%T')" >> $L
( cd $RR && FINAL_CKPT="$E/checkpoint_b_post_ref.pt" FINAL_CODE="worktrees/r21c" FINAL_TAG=syn_sh flock $LOCK $PY -W ignore \
    $RR/FINAL_20261014/tools/bathroom_struct_final.py ) > $E/bathroom_struct_final.out 2>&1
echo "^done sh bathroom_struct $(date '+%T') rc=$?" >> $L
