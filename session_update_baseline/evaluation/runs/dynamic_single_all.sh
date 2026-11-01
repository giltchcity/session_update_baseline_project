#!/usr/bin/env bash
# The official dynamic-object metric (analysis/dynamic_gt_eval.py map, one ground truth per visit) on every single-run map,
# one after another; our real C is scored by its own call once mapped.
cd /home/jixian/Desktop/FT/wt_takeover_runs
run() { rm -rf eval/dynamic_single/$3; DYN_EVAL_JOBS=8 nice -n 10 /home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python analysis/dynamic_gt_eval.py map $1 $2 $4 eval/dynamic_single/$3 > eval/dynamic_single/$3.log 2>&1; echo "$(date +%H:%M) $3: $(tail -1 eval/dynamic_single/$3.log | cut -c1-320)"; }
mkdir -p eval/dynamic_single
run real a ours_real_a $PWD/real_a_states/final.4dmap.zpk
run real a khronos_real_a /home/jixian/Desktop/FT/runs/official_khronos_63faadde_A_full_20260909/state/final.4dmap.zpk
run real b ours_real_b $PWD/real_b_single/final.4dmap.zpk
run real b khronos_real_b /home/jixian/Desktop/FT/runs/official_khronos_63faadde_B_full_20260909/state/final.4dmap.zpk
run real c khronos_real_c /home/jixian/Desktop/FT/runs/official_khronos_63faadde_C_full_20260909/state/final.4dmap.zpk
run synthetic a ours_syn_a $PWD/synthetic_a_once/final.4dmap.zpk
run synthetic a khronos_syn_a $PWD/khronos_continuous/synthetic_5cm/sessions/session_a.4dmap.zpk
run synthetic b ours_syn_b $PWD/synthetic_b_single/final.4dmap.zpk
run synthetic b khronos_syn_b $PWD/khronos_continuous/synthetic_5cm_clean_b/state/final.4dmap.zpk
echo "$(date +%H:%M) DYNAMIC SINGLE DONE"
