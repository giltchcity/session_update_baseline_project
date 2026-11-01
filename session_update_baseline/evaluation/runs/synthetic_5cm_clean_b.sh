#!/usr/bin/env bash
# Synthetic clean build of official Khronos at 5 cm, for the clean build table (user 10-31: every method at the same
# resolution; until then the 09-17 per-visit runs, 10 cm: datasets/synthetic_ab/evaluation_complete_20260917).
# A from empty is the A slice of the continuous 5 cm run (khronos_continuous/synthetic_5cm, same config and frames);
# B from empty is analysis/khronos_continuous.sh synthetic with only session b, into khronos_continuous/synthetic_5cm_clean_b.
# Both through the table's evaluation (run_synthetic_eval.py) into eval/obj100_khronos_clean_synthetic_5cm.
# Low priority, the mapping under runs/HEAVY.lock (after the continuous run's).
set -Eeuo pipefail
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; LOCK=$FT/runs/HEAVY.lock
CAP=(systemd-run --user --scope --quiet -p OOMPolicy=stop)
step() { echo "$(date +%H:%M) $*"; }

K=$R/khronos_continuous/synthetic_5cm_clean_b
if [[ ! -e $K ]]; then
  flock $LOCK "${CAP[@]}" nice -n 19 ionice -c3 env KHRONOS_SESSIONS=b bash $R/analysis/khronos_continuous.sh synthetic $K \
    > $R/logs/khronos_continuous_synthetic_5cm_clean_b.log 2>&1 \
    || { step "Khronos synthetic clean B failed: $(tail -2 $R/logs/khronos_continuous_synthetic_5cm_clean_b.log)"; exit 1; }
fi
step "Khronos synthetic clean B mapped: $(du -h $K/state/final.4dmap.zpk | cut -f1)"
A=$R/khronos_continuous/synthetic_5cm/sessions/session_a.4dmap.zpk
while [[ ! -s $A ]]; do sleep 60; done   # the continuous run's slices (synthetic_5cm_reruns.sh)
mkdir -p $K/sessions
ln -sfn $A $K/sessions/session_a.4dmap.zpk
ln -sfn $K/state/final.4dmap.zpk $K/sessions/session_b.4dmap.zpk
SYN_MAPS=$K/sessions SYN_OUT=$R/eval/obj100_khronos_clean_synthetic_5cm SYNTHETIC_TOOLS=khronos \
  SYNTHETIC_TOOLS_BIN=$FT/eval/synthetic_object_eval_official SYN_REF=$R/eval/obj100_khronos_cont_synthetic_5cm SYN_REF_LABEL=khronos_cont_5cm \
  "${CAP[@]}" ionice -c3 nice -n 19 taskset -c 0-9 python3 $R/analysis/run_synthetic_eval.py > $R/logs/obj100_khronos_clean_synthetic_5cm.log 2>&1 \
  || { step "Khronos synthetic clean evaluation failed: $(tail -2 $R/logs/obj100_khronos_clean_synthetic_5cm.log)"; exit 1; }
step "Khronos synthetic clean evaluated: $(tail -1 $R/logs/obj100_khronos_clean_synthetic_5cm.log)"
step "SYNTHETIC CLEAN 5CM DONE"
