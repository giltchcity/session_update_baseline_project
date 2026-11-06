#!/bin/bash
# Waits for the GaME session's export of the e4a02cf front ids (chain.log ends "exit 0", 500 images per visit), then our
# TSDF core on room0 and room1 (run_tsdf.sh, one at a time) and the ADT check (gt_check_tsdf.py).
A=/home/jixian/Desktop/FT/wt_takeover_runs/aria; X=/tmp/claude-1000/-home-jixian-Desktop-FT/701d920c-ea2f-488c-8a14-a275a810f5e4/scratchpad
step() { echo "$(date +%H:%M) $*"; }
for r in room0 room1; do
  until [[ "$(tail -1 $X/hexport_$r/chain.log 2>/dev/null)" == "exit 0" ]]; do sleep 30; done
  step "$r: front ids exported"
done
for r in room0 room1; do
  bash $A/run_tsdf.sh $r || { step "$r: TSDF failed"; continue; }
  /home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python $A/gt_check_tsdf.py $r $A/tsdf/$r > $A/tsdf/$r/GT_CHECK.txt 2>&1 \
    && step "$r: ADT check written (tsdf/$r/GT_CHECK.txt)" || step "$r: ADT check failed (tsdf/$r/GT_CHECK.txt)"
done
step "ARIA TSDF ALL DONE"
