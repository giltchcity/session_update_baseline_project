#!/bin/bash
# Flat object F1 and change F1 of ours (flat_a_tk, flat_b_resupport) and Khronos continuous (khronos_continuous/flat,
# cut per session by khronos_continuous_slices.py), scored as the GaME and wavemap rows are
# (runs/update_layer_game_T1_20261006/analysis/wm_flat_eval_20261028):
#   object F1: the unchanged official exp_pipeline (official_khronos_63faadde_abc) on each session's map with that
#     directory's official_objects_{a,b}.yaml (its per-session ground truth; Surface, 0.5 m, every 100th point,
#     semantics required, no compensation), then tools/abc_eval_v2/official_native_eval_postprocess.py (the online mean
#     over the map's snapshots), as score_flat_objects_identity.sh does with a timeline's built map;
#   change F1: analysis/flat_change_maps.py (flat_change_eval.py's scoring on session B's first and last snapshot).
# Evaluation only, nothing remapped; intermediates in /dev/shm, removed at the end. Output eval/flat_objects_changes/.
set -eo pipefail
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; OUT=$R/eval/flat_objects_changes
OFF=$FT/build_artifacts/official_khronos_63faadde_abc
CFG=$FT/runs/update_layer_game_T1_20261006/analysis/wm_flat_eval_20261028/official_objects
PY=$FT/envs/gs-cu128/bin/python
step() { echo "$(date +%H:%M) $*"; }
mkdir -p $OUT
W=$(mktemp -d /dev/shm/flat-eval-XXXXXX); trap 'rm -rf $W' EXIT
python3 $R/analysis/khronos_continuous_slices.py $R/khronos_continuous/flat/state/final.4dmap.zpk $W/khronos flat > $OUT/khronos_slices.log 2>&1
declare -A MAP=([ours_a]=$R/flat_a_tk/final.4dmap.zpk [ours_b]=$R/flat_b_resupport/final.4dmap.zpk
                [khronos_a]=$W/khronos/session_a.4dmap.zpk [khronos_b]=$W/khronos/session_b.4dmap.zpk)
for m in ours khronos; do
  for s in a b; do
    o=$OUT/objects_${m}_$s; rm -rf $o; mkdir -p $o/official
    mkfifo $o/official/final.4dmap
    echo '[INFO] Evaluation input directory only. Source mapper completion must be checked separately.' > $o/official/experiment_log.txt
    python3 $FT/eval_fast/vendor/pack/map4d_pack.py cat ${MAP[${m}_$s]} > $o/official/final.4dmap &
    cat_pid=$!
    bash -c 'source "$1" >/dev/null 2>&1; shift; exec "$@"' x $OFF/environment.sh \
      $OFF/install/lib/khronos_eval/exp_pipeline ${CFG}_$s.yaml $o/official false true false > $o/official/native_pipeline.log 2>&1
    wait $cat_pid
    rm -f $o/official/final.4dmap
    $PY $FT/tools/abc_eval_v2/official_native_eval_postprocess.py $o/official/results/static_objects.csv --output $o/post > /dev/null
    step "$m $s objects: $($PY -c "import json; s = json.load(open('$o/post/NATIVE_ONLINE_SUMMARY.json'))['summary']; print(' '.join(f'{k}={s[k][\"mean\"]:.4f}' if s[k]['mean'] is not None else f'{k}=NaN' for k in ('ObjectPrecision', 'ObjectRecall', 'ObjectF1')))")"
  done
done
(set +u; source /opt/ros/jazzy/setup.bash; source /home/jixian/ros2_ws/install/setup.bash
 export LD_LIBRARY_PATH=$FT/wt_290e110_build/canonical/install/lib:$LD_LIBRARY_PATH
 $FT/eval/chain_fill/timeline_export ${MAP[ours_b]} $W/ours_b.jsonl 0
 $FT/eval/chain_fill/timeline_export ${MAP[khronos_b]} $W/khronos_b.jsonl 0)
$PY $R/analysis/flat_change_maps.py $OUT/FLAT_CHANGE.json ours=$W/ours_b.jsonl khronos=$W/khronos_b.jsonl > $OUT/flat_change.log 2>&1
step "changes: $(grep 'change F1' $OUT/flat_change.log | tr '\n' ';')"
step "Flat objects and changes done"
