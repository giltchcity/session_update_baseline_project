#!/bin/bash
# P3 instance control with Khronos's own ExternalTracker (tracks by the given instance id, as our runs): official
# Khronos on the five 3RScan rooms every backend ran, KHRONOS_INSTANCES=1 KHRONOS_TRACKER=external into
# ROOM/khronos_inst_ext, then the official evaluation at 0.5 m (eval/threerscan_khronos_eval.py khronos) and the
# geometry (eval/threerscan_chain_eval.py khronos). The MaxIoUTracker control (ROOM/khronos_inst) stays as it was.
# usage: p3_khronos_inst_ext.sh [ROOM_REF ...]   (default the five rooms)
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python
LOW=(ionice -c3 nice -n 19 taskset -c 0-9)
step() { echo "$(date +%H:%M) $*"; }
for ref in ${@:-ad408c8f 0cac766c 6bde6070 bf9a3de7 0958220d}; do
  room=p3_$ref; out=$R/3rscan/$room; d=$out/khronos_inst_ext
  if [[ ! -s $d/state/final.4dmap.zpk ]]; then
    rm -rf $d.incomplete.*
    KHRONOS_INSTANCES=1 KHRONOS_TRACKER=external THREERSCAN_ROOM=$room flock $FT/runs/HEAVY.lock taskset -c 0-9 \
      bash $R/analysis/khronos_continuous.sh 3rscan $d > $out/khronos_inst_ext.log 2>&1 \
      || { step "$room FAILED: Khronos ($(tail -1 $out/khronos_inst_ext.log))"; continue; }
    grep -q 'type: *ExternalTracker\|ExternalTracker' $d/control/logs/khronos.log || step "$room WARNING: ExternalTracker not in the config dump"
    step "$room mapped: $(du -h $d/state/final.4dmap.zpk | cut -f1)"
  fi
  rm -rf $d/khronos_metrics
  THREERSCAN_OBJECTS_ONLY=1 "${LOW[@]}" $PY $FT/eval/threerscan_khronos_eval.py $room khronos $d > $out/khronos_metrics_khronos_inst_ext_m050.log 2>&1 \
    && step "$(tail -1 $out/khronos_metrics_khronos_inst_ext_m050.log | cut -c1-160)" || step "$room FAILED: metrics ($(tail -1 $out/khronos_metrics_khronos_inst_ext_m050.log))"
  "${LOW[@]}" $PY $FT/eval/threerscan_chain_eval.py $room khronos $d > $out/eval_khronos_inst_ext_geom.log 2>&1 \
    && step "$room geometry: $(grep -c 'geometry @5cm' $out/eval_khronos_inst_ext_geom.log) visits" || step "$room FAILED: geometry ($(tail -1 $out/eval_khronos_inst_ext_geom.log))"
done
step "P3 Khronos instance control (ExternalTracker) done"
