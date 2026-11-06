#!/bin/bash
# Panoptic synthetic A -> B at 5 cm, objects and background (user 10-31: every method at the same resolution; the
# table's run until then had a 10 cm background), with panoptic_rerun_synthetic.sh (a visit whose map is saved is
# skipped: A was run 11-01 08:29-08:43), then scored by panoptic_score.sh into eval/panoptic_rerun_synthetic_5cm.
# usage: panoptic_syn5cm.sh
set -eo pipefail
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; RUN=$FT/runs/panoptic_mapping/rerun_syn5cm_20261031
bash $R/analysis/panoptic_rerun_synthetic.sh $RUN >> $RUN/queue_synthetic.log 2>&1
echo "[$(date +%F' '%T)] scoring" >> $RUN/queue_synthetic.log
bash $R/analysis/panoptic_score.sh synthetic $RUN $R/eval/panoptic_rerun_synthetic_5cm > $R/logs/panoptic_score_synthetic_5cm.log 2>&1 \
  && echo "[$(date +%F' '%T)] scored into eval/panoptic_rerun_synthetic_5cm" >> $RUN/queue_synthetic.log \
  || echo "[$(date +%F' '%T)] scoring failed (see logs/panoptic_score_synthetic_5cm.log)" >> $RUN/queue_synthetic.log
