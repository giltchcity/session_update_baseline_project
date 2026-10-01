#!/usr/bin/env bash
# The report figures (old style): top-down maps (render_topdown) and perspective views of change sites
# (render_perspective: photo | backends, rows 2 / 3 / 4). CPU only; output D:/3Study/ETH/FT/101.
#
#   make_figures.sh synthetic|real|all
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
echo "[$(date +%T)] figures done"
