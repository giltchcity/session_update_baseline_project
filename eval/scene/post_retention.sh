#!/usr/bin/env bash
# Unobserved retention / absent residue (eval/scene/retention.py) for every mode of a runs root.
#   post_retention.sh synthetic RUNS_ROOT      -> RUNS_ROOT/eval_<mode>/retention_b.json
#   post_retention.sh real RUNS_ROOT           -> RUNS_ROOT/eval_<mode>/retention_{b,c}.json
set -u
DS=$1; ROOT=$(realpath "$2")
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python
cd /home/jixian/Desktop/FT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
if [[ $DS == synthetic ]]; then pairs="a:b"; else pairs="a:b b:c"; fi
for run in "$ROOT"/*/; do
  run=${run%/}; mode=$(basename "$run"); [[ $mode == eval_* ]] && continue
  case $mode in syn_*) [[ $DS == synthetic ]] || continue ;; real_*) [[ $DS == real ]] || continue ;; esac
  [[ -d $ROOT/eval_$mode ]] || continue
  for pair in $pairs; do
    p=${pair%:*}; c=${pair#*:}; out=$ROOT/eval_$mode/retention_$c.json
    [[ -s $out || ! -s $run/session_$c/timeline.pkl ]] && continue
    echo "[$(date +%T)] $mode $p->$c"
    nice -n 10 "$PY" -m eval.scene.retention "$run/session_$p/timeline.pkl" "$run/session_$c/timeline.pkl" "$DS" "$c" "$out" > /dev/null
  done
done
