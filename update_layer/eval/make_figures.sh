#!/usr/bin/env bash
# The report figures (old style): top-down maps (render_topdown) and perspective views of change sites
# (render_perspective: photo | backends, rows 2 / 3 / 4). CPU only; output D:/3Study/ETH/FT/101.
#
#   make_figures.sh synthetic|real|all
#   make_figures.sh game_rows [RUN_ROOT]   GaME rows 1 / 3 / 4 per real session, top-down from the exported final maps
#                                         (ROWSPEC="label:dir[:dir...]|..." and TAG=<file-name tag> override the rows;
#                                         red-sample counts per panel -> analysis/figure_counts/<TAG>_<S>.json)
#   make_figures.sh game_rows_closeups [RUN_ROOT]   the same rows, session C close-ups: I2's sites, I10's sites, where people
#                                         walked (red there: map surface farther than 5 cm from the present surface)
#                                         (render_rows_topdown: median-depth TSDF export where it exists, else the surfel
#                                         export), ghost on seen-empty old sites in red
set -uo pipefail
cd /home/jixian/Desktop/FT/session_update_baseline_project
PY="nice -n 10 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 /home/jixian/Desktop/FT/envs/gs-cu128/bin/python"
fig() { echo "[$(date +%T)] $*"; $PY -m "$@" 2>&1 | tail -1; }
if [ "$1" = synthetic ] || [ "$1" = all ]; then
  fig update_layer.eval.render_topdown synthetic b rgb
  fig update_layer.eval.render_topdown synthetic b identity
  fig update_layer.eval.render_topdown synthetic a rgb
  for site in "b bed_0000" "b chair_0000" "b flower_0003" "a chair_0001" "a sofa_0000"; do
    fig update_layer.eval.render_perspective synthetic $site
  done
fi
if [ "$1" = real ] || [ "$1" = all ]; then
  fig update_layer.eval.render_topdown real c rgb
  fig update_layer.eval.render_topdown real b rgb
  fig update_layer.eval.render_perspective 7071
  for site in "c D3_BC_I2" "c D3_BC_I10" "b D3_AB_I18" "b D2_B_I2" "b D3_AB_I11"; do
    fig update_layer.eval.render_perspective real $site
  done
fi
if [ "$1" = game_rows ]; then
  RR=${2:-/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006}
  for s in a b c; do
    panels=()
    IFS='|' read -r -a SPECS <<< "${ROWSPEC:-row 1 (from scratch):real_row1v2:real_row1b:real_row1|row 3 (GaME own update):real_row3v2:real_row3|row 4 (+ update layer):real_row4v3:real_row4v2:real_row4}"
    for spec in "${SPECS[@]}"; do
      label=${spec%%:*}; dirs=${spec#*:}
      for d in ${dirs//:/ }; do
        E=$RR/$d/eval_real
        if [ -s $RR/$d/session_$s/tsdf_final.ply ]; then panels+=("$label, median TSDF=$RR/$d/session_$s/tsdf_final.ply:$E/geometry/$s/G1_tsdf.json:$E/ghost/$s/GHOST.json"); break; fi
        if [ -s $E/final_$s.ply ]; then panels+=("$label, surfel export=$E/final_$s.ply:$E/geometry/$s/G1.json:$E/ghost/$s/GHOST.json"); break; fi
      done
    done
    G=$(ls $RR/real_row*/eval_real/ghost/$s/evidence_final_$s.npz 2>/dev/null | head -1); G=${G%/*}
    REF=/home/jixian/Desktop/FT/results/abc_eval_v2/geometry/${s^^}_reference_1cm.ply
    fig update_layer.eval.render_rows_topdown real $s /mnt/d/3Study/ETH/FT/101/game_rows_real_${s^^}_topdown${TAG:+_$TAG}.png \
        ${G:+--probes $G} --reference $REF --counts $RR/analysis/figure_counts/${TAG:-rows}_${s^^}.json "${panels[@]}"
  done
fi
if [ "$1" = game_rows_closeups ]; then
  RR=${2:-/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006}; s=c
  IFS='|' read -r -a SPECS <<< "${ROWSPEC:-row 1 (from scratch):real_row1v3|row 3 (GaME own update):real_row3v3|row 4 (+ update layer):real_row4v4}"
  panels=()
  for spec in "${SPECS[@]}"; do
    label=${spec%%:*}; d=${spec#*:}; d=${d%%:*}
    panels+=("$label=$RR/$d/session_$s/tsdf_final.ply")
  done
  G=$(ls $RR/real_row4v4/eval_real/ghost/$s/evidence_final_$s.npz 2>/dev/null | head -1); G=${G%/*}
  REF=/home/jixian/Desktop/FT/results/abc_eval_v2/geometry/C_reference_1cm.ply
  # boxes (x0,x1,z0,z1, metres, real world frame): GT medians (abc_eval_final/gt_objects) and person positions (observations/C)
  for item in "I2:-1.0,3.0,-0.5,2.8:" "I10:-1.4,2.9,-1.2,0.9:" "people:-1.3,2.9,-0.1,2.7:--offref"; do
    IFS=: read -r name bx extra <<< "$item"
    fig update_layer.eval.render_rows_topdown real $s /mnt/d/3Study/ETH/FT/101/game_rows_real_C_closeup_${name}${TAG:+_$TAG}.png \
        ${G:+--probes $G} --reference $REF --box $bx $extra --counts $RR/analysis/figure_counts/${TAG:-rows}_C_closeup_${name}.json "${panels[@]}"
  done
fi
echo "[$(date +%T)] figures done"
