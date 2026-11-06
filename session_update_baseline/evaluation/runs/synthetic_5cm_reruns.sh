#!/usr/bin/env bash
# Synthetic baselines at 5 cm (user 10-31: every method at the same resolution; both ran their background at 10 cm),
# after the TESSE Khronos 5 cm runs (logs/tesse_khronos_inst_5cm.log ends with ALL DONE). Low priority (nice 19, idle IO)
# beside the GPU jobs, each mapping under runs/HEAVY.lock.
#   Khronos: analysis/khronos_continuous.sh synthetic into khronos_continuous/synthetic_5cm (zsnap, the observer rebuilt
#     against it), sliced per session (khronos_continuous_slices.py), the table's evaluation (as synthetic_obj_unified.sh:
#     run_synthetic_eval.py, the one object setting) into eval/obj100_khronos_cont_synthetic_5cm against the 10 cm one;
#   Panoptic: analysis/panoptic_rerun_synthetic.sh into runs/panoptic_mapping/rerun_syn5cm_20261031 (the 10-29 input and
#     launcher), scored by analysis/panoptic_score.sh synthetic into eval/panoptic_rerun_synthetic_5cm.
set -Eeuo pipefail
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; LOCK=$FT/runs/HEAVY.lock
CAP=(systemd-run --user --scope --quiet -p OOMPolicy=stop)   # no fixed limits (user 11-01): the run is watched; the scope stops as a whole if the kernel kills anything in it
step() { echo "$(date +%H:%M) $*"; }

K=$R/khronos_continuous/synthetic_5cm
if [[ ! -e $K ]]; then
  flock $LOCK "${CAP[@]}" nice -n 19 ionice -c3 bash $R/analysis/khronos_continuous.sh synthetic $K > $R/logs/khronos_continuous_synthetic_5cm.log 2>&1 \
    || { step "Khronos synthetic mapping failed: $(tail -2 $R/logs/khronos_continuous_synthetic_5cm.log)"; exit 1; }
fi
step "Khronos synthetic mapped: $(du -h $K/state/final.4dmap.zpk | cut -f1)"
[[ -d $K/sessions ]] || nice -n 19 ionice -c3 python3 $R/analysis/khronos_continuous_slices.py $K/state/final.4dmap.zpk $K/sessions synthetic \
  > $R/logs/khronos_continuous_slices_synthetic_5cm.log 2>&1
step "sliced: $(grep -h '^session' $R/logs/khronos_continuous_slices_synthetic_5cm.log | tr '\n' ';')"
SYN_MAPS=$K/sessions SYN_OUT=$R/eval/obj100_khronos_cont_synthetic_5cm SYNTHETIC_TOOLS=khronos \
  SYNTHETIC_TOOLS_BIN=$FT/eval/synthetic_object_eval_official SYN_REF=$R/eval/obj100_khronos_cont_synthetic SYN_REF_LABEL=khronos_10cm \
  "${CAP[@]}" ionice -c3 nice -n 19 taskset -c 0-9 python3 $R/analysis/run_synthetic_eval.py > $R/logs/obj100_khronos_cont_synthetic_5cm.log 2>&1 \
  || { step "Khronos synthetic evaluation failed: $(tail -2 $R/logs/obj100_khronos_cont_synthetic_5cm.log)"; exit 1; }
step "Khronos synthetic evaluated: $(tail -1 $R/logs/obj100_khronos_cont_synthetic_5cm.log)"

P=$FT/runs/panoptic_mapping/rerun_syn5cm_20261031
nice -n 19 ionice -c3 bash $R/analysis/panoptic_rerun_synthetic.sh $P > $P/queue_synthetic.log 2>&1 \
  || { step "Panoptic synthetic mapping failed: $(tail -2 $P/queue_synthetic.log)"; exit 1; }
step "Panoptic synthetic mapped: $(tail -1 $P/queue_synthetic.log)"
mkdir -p $R/eval/panoptic_rerun_synthetic_5cm
nice -n 19 ionice -c3 bash $R/analysis/panoptic_score.sh synthetic $P $R/eval/panoptic_rerun_synthetic_5cm \
  > $R/eval/panoptic_rerun_synthetic_5cm/score.log 2>&1 || { step "Panoptic synthetic scoring failed"; exit 1; }
step "Panoptic synthetic scored"
step "SYNTHETIC 5CM DONE"
