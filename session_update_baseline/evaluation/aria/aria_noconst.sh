#!/bin/bash
# The two-constant ablation on Aria: run_tsdf.sh with configs/aria_instance_5cm_noconst.yaml (TAG noconst), then the ADT check.
A=/home/jixian/Desktop/FT/wt_takeover_runs/aria
for r in room0 room1; do
  MAPPER_CONFIG=$A/configs/aria_instance_5cm_noconst.yaml TAG=noconst bash $A/run_tsdf.sh $r || { echo "$(date +%H:%M) $r noconst: TSDF failed"; continue; }
  /home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python $A/gt_check_tsdf.py $r $A/tsdf/${r}_noconst > $A/tsdf/${r}_noconst/GT_CHECK.txt 2>&1 \
    && echo "$(date +%H:%M) $r noconst: ADT check written" || echo "$(date +%H:%M) $r noconst: ADT check failed"
done
echo "$(date +%H:%M) ARIA NOCONST DONE"
