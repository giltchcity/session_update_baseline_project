#!/usr/bin/env bash
# The results table's Panoptic cells from the rerun (analysis/panoptic_rerun.sh, panoptic_rerun_synthetic.sh): each
# saved map read by its own submap states (analysis/panoptic_maps.py: New / Matched / Persistent shown, Absent and
# Unobserved not; Panoptic Multi-TSDFs Sec. III-E), then the scripts the other methods are scored with, unchanged:
#   G1: the visit's final map, shown mesh (real: eval/geometry_g1.py with eval/crops/real_<S>.json and the 1 cm
#     reference; Flat: analysis/flat_eval.py; synthetic: the harness's evaluate_cross_session_geometry.py);
#   the maps scored: Panoptic has no map schedule of its own; of the maps the rerun saved every 2 s, those current at
#     the times the Khronos run of the dataset saved its maps (khronos_times: the latest Panoptic map at or before each
#     Khronos map time), so the two baselines are sampled alike (ours and Khronos save every ~2 s on real data and
#     every ~10.8 s on synthetic data; the online object F1 counts the time before a visit's first map as an empty
#     map); plus each visit's final map;
#   object F1: every map of a visit at its own time by the official evaluator at the one object setting (Surface,
#     0.5 m, every 100th point, semantics required, no compensation; real eval_fast/configs/<s>.yaml and Flat
#     wm_flat_eval_20261028/official_objects_<s>.yaml through official_khronos_63faadde_abc exp_pipeline, then
#     tools/abc_eval_v2/official_native_eval_postprocess.py and object_weighted_nr.py's weights; synthetic: the
#     harness's query manifest, eval/synthetic_object_eval_official/khronos_historical_object_eval, the harness's
#     verbatim aggregation), the maps' shown objects written as a 4D map by eval/chain_fill/timeline_to_4dmap;
#   change F1 (panoptic_maps.py's changes reading: a submap is gone only when judged Absent): real D2 + D3 by the
#     harness (cleanup_eval_h.py, eval_state_change_h.py); Flat analysis/flat_change_maps.py (B's first record =
#     A's final map, the map B inherited, and B's final map); synthetic the harness's score_complete_states.py
#     (D2_hidden, D3 State) on the native_changes export.
# 3RScan (panoptic_rerun_3rscan.sh), per room as the table's other methods: object F1 by eval/threerscan_khronos_eval.py
#   on the visits' objects.jsonl (visit k at 1 + 200 k s) written as one 4D map as real and Flat (timeline_to_4dmap: one
#   snapshot per map, nothing added between maps), packed as METHOD_DIR/state/final.4dmap.zpk and read by its khronos
#   mode (each snapshot cut out and scored at its own time; the online slice, METRICS.json weighted). Not its timeline
#   mode: eval/timeline_complete.py keeps an object a method stops listing as a closed record in every later map, and
#   Panoptic deletes the new submaps it does not confirm (79 such records by the end of p3_e48c48f4's first visit, which
#   the evaluator associated in place of the present submaps; scored so until 10-30: object F1 35.7). Changes and G1 by
#   eval/threerscan_chain_eval.py's Panoptic reader (each visit's final map), into wt_takeover_runs/3rscan/p3_ROOM/panoptic_rerun.
# usage: panoptic_score.sh flat|real|synthetic|3rscan RERUN_DIR OUT_DIR
set -Eeuo pipefail
KIND=$1; RUN=$(realpath "$2"); OUT=$(realpath -m "$3")
FT=/home/jixian/Desktop/FT; A=$FT/wt_takeover_runs/analysis
PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python; GPY=$FT/envs/gs-cu128/bin/python
OFF=$FT/build_artifacts/official_khronos_63faadde_abc
H=$FT/results/current_version_20260921/tools/harness_real
ENV='set +u; source /opt/ros/jazzy/setup.bash; source /home/jixian/ros2_ws/install/setup.bash; export LD_LIBRARY_PATH=/home/jixian/Desktop/FT/wt_290e110_build/canonical/install/lib:$LD_LIBRARY_PATH; exec "$@"'
step() { echo "[$(date +%T)] $*"; }
mkdir -p "$OUT"
maps() {  # maps KIND NAME SNAPSHOTS FINAL END_NS [ARGS...]: panoptic_maps.py into OUT/NAME
  [[ -s $OUT/$2/COMPLETE.json ]] || CUDA_VISIBLE_DEVICES= nice -n 10 $PY $A/panoptic_maps.py "$1" "$OUT/$2" "$3" "$4" "$5" "${@:6}" \
    > "$OUT/$2.maps.log" 2>&1 || { tail -5 "$OUT/$2.maps.log"; exit 1; }
  step "$2: $(tail -1 "$OUT/$2.maps.log")"
}
to4d() {  # to4d TEMPLATE TIMELINE OUT.4dmap
  bash -c "$ENV" t2m $FT/eval/chain_fill/timeline_to_4dmap "$1" "$2" "$3" | tail -1
}
objects() {  # objects NAME TEMPLATE CONFIG: the official online object rows of OUT/NAME/objects.jsonl
  local d=$OUT/$1/objects
  [[ -s $d/post/native_unique_online_rows.csv ]] && return
  rm -rf "$d"; mkdir -p "$d/official"
  to4d "$2" "$OUT/$1/objects.jsonl" "$d/official/final.4dmap" > "$d/build.log" 2>&1
  echo '[INFO] Evaluation input directory only. Source mapper completion must be checked separately.' > "$d/official/experiment_log.txt"
  bash -c 'source "$1" >/dev/null 2>&1; shift; exec "$@"' x $OFF/environment.sh \
    $OFF/install/lib/khronos_eval/exp_pipeline "$3" "$d/official" false true false > "$d/official/native_pipeline.log" 2>&1
  rm -f "$d/official/final.4dmap"
  $GPY $FT/tools/abc_eval_v2/official_native_eval_postprocess.py "$d/official/results/static_objects.csv" --output "$d/post" > /dev/null
}
template() {  # template MAP.zpk OUT.4dmap: its first snapshot (the layer layout timeline_to_4dmap writes into)
  $PY -c "import sys; sys.path.insert(0, '$FT/eval_fast/vendor/pack'); import zpk_snapshot as zs; zs.cut_many(sys.argv[1], [0], out=sys.argv[2], work='/dev/shm')" "$1" "$2"
}
stamp() { python3 -c "import csv,sys; f=$1; print(f(int(r[1]) for r in list(csv.reader(open(sys.argv[1])))[1:]))" "$2"; }   # stamp min|max TIMESTAMPS.csv
khronos_times() {  # khronos_times KIND SESSION > times: the map times of the Khronos run the table scores (ns)
  $PY - "$1" "$2" <<'PY'
import csv, sys
sys.path.insert(0, '/home/jixian/Desktop/FT/eval_fast/vendor/pack')
kind, s = sys.argv[1:]
E = '/home/jixian/Desktop/FT/wt_takeover_runs/eval'
if kind == 'synthetic':
    import map4d_pack as mp
    times = mp.read_manifest(f'/home/jixian/Desktop/FT/wt_takeover_runs/khronos_continuous/synthetic_5cm/sessions/session_{s}.4dmap.zpk')['source']['stamps']   # the table's Khronos run (5 cm since 10-31)
elif kind == '3rscan':      # s = 'ROOM K': the continuous map's snapshots in visit K's window (1 + 200 K s)
    import map4d_pack as mp
    room, k = s.split(); lo = 1000000000 + 200000000000 * int(k)
    times = [t for t in mp.read_manifest(f'/home/jixian/Desktop/FT/wt_takeover_runs/3rscan/p3_{room}/khronos_inst_ext/state/final.4dmap.zpk')['source']['stamps'] if lo <= t < lo + 200000000000]
elif kind == 'flat':
    times = [int(r['Query']) for r in csv.DictReader(open(f'{E}/flat_objects_changes/objects_khronos_{s}/post/native_unique_online_rows.csv'))]
else:
    last = {}
    for r in csv.DictReader(open(f'{E}/khronos_cont_real_inst_external_{s}/real_abc/objects/khronos_cont_{s}/results/static_objects.csv')):
        last[r['Name']] = max(last.get(r['Name'], 0), int(r['Query']))
    times = sorted(last.values())
print('\n'.join(map(str, times)))
PY
}
at_times() {  # at_times SNAPDIR STAMP_BASE TIMES_FILE OUT_DIR: links to the latest map at or before each time
  python3 - "$@" <<'PY'
import sys
from pathlib import Path
snap, base, times, out = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
maps = sorted((base + int(p.stem.split('_')[1]), p) for p in snap.glob('*.panmap'))
chosen = {next((p for t2, p in reversed(maps) if t2 <= t), None) for t in map(int, open(times).read().split())} - {None}
out.mkdir(parents=True, exist_ok=True)
for p in chosen:
    (out / p.name).symlink_to(p.resolve())
print(f'{len(chosen)} of {len(maps)} maps, at the Khronos map times', file=sys.stderr)
PY
}
if [[ $KIND == flat ]]; then
  F=$FT/datasets/panoptic_mapping/flat_dataset; W=$FT/runs/update_layer_game_T1_20261006/analysis/wm_flat_eval_20261028
  a0=$(stamp min $F/run1/timestamps.csv); b0=$(stamp min $F/run2/timestamps.csv)
  for s in a b; do khronos_times flat $s > "$OUT/khronos_times_$s.txt"; done
  at_times "$RUN/flat/run1_snapshots" 0 "$OUT/khronos_times_a.txt" "$OUT/maps_a"
  at_times "$RUN/flat/run2_snapshots" 0 "$OUT/khronos_times_b.txt" "$OUT/maps_b"
  maps flat a "$OUT/maps_a" "$RUN/flat/run1.panmap" "$(stamp max $F/run1/timestamps.csv)" --start-ns "$a0"
  maps flat b "$OUT/maps_b" "$RUN/flat/run2.panmap" "$(stamp max $F/run2/timestamps.csv)" --start-ns "$b0" \
    --seed "$RUN/flat/run1.panmap" --seed-ns "$b0"
  for s in a b; do
    objects $s $FT/wt_takeover_runs/flat_a_tk/final.4dmap.zpk $W/official_objects_$s.yaml
    $PY $A/flat_eval.py $s "$OUT/$s/final_present.ply" "$OUT/g1_$s" > "$OUT/g1_$s.log" 2>&1 || tail -3 "$OUT/g1_$s.log"
  done
  $GPY $A/flat_change_maps.py "$OUT/FLAT_CHANGE.json" panoptic="$OUT/b/snapshots.jsonl" > "$OUT/flat_change.log" 2>&1
  step "Flat: $(grep 'change F1' "$OUT/flat_change.log")"
elif [[ $KIND == real ]]; then
  BASE=$FT/results/abc_eval_final_baseline/input_manifest.json
  start() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['sessions'][sys.argv[2]]['timestamp_provenance']['session_start_ns'])" $BASE $1; }
  DS=$FT/datasets/panoptic_mapping/flat_dataset/local_ab_official_contract_640x480_v1
  prev=""
  for s in a b c; do
    S=${s^^}; t0=$(start $S); end=$((t0 + $(stamp max $DS/session_$s/timestamps.csv)))
    more=(--stamp-base "$t0" --start-ns "$t0")
    [[ -n $prev ]] && more+=(--seed "$prev" --seed-ns "$t0")
    [[ $s != a ]] && more+=(--probes $FT/results/abc_eval_final_baseline/temporal/RESIDUAL_PROBES.json $FT/results/abc_eval_v2/cleanup/probes_$s.json)
    khronos_times real $s > "$OUT/khronos_times_$s.txt"
    at_times "$RUN/real/session_${s}_snapshots" "$t0" "$OUT/khronos_times_$s.txt" "$OUT/maps_$s"
    maps real $s "$OUT/maps_$s" "$RUN/real/session_$s.panmap" "$end" "${more[@]}"
    prev=$RUN/real/session_$s.panmap
  done
  T=/dev/shm/panoptic-real-template.4dmap; template $FT/wt_takeover_runs/real_a_states/final.4dmap.zpk $T
  for s in a b c; do
    S=${s^^}; X=$OUT/$s; CH=$OUT/changes/$s; mkdir -p "$CH" "$OUT/geometry/$s"
    objects $s $T $FT/eval_fast/configs/$s.yaml
    $PY $FT/eval/geometry_g1.py score real $S $FT/results/abc_eval_v2/geometry/${S}_reference_1cm.ply "$X/final_present.ply" \
      $FT/eval/crops/real_$S.json "$OUT/geometry/$s/G1.json" > "$OUT/geometry/$s/G1.log" 2>&1 || tail -3 "$OUT/geometry/$s/G1.log"
    if [[ $s != a ]]; then
      $PY $H/cleanup_eval_h.py --session $S --snapshots "$X/snapshots.jsonl" --bin "$X/cleanup_probes.bin" --out "$CH" > "$CH/cleanup.log" 2>&1 || tail -3 "$CH/cleanup.log"
      $PY $H/eval_state_change_h.py --session $S --snapshots "$X/snapshots.jsonl" --cleanup "$CH/CLEANUP_PROBES_$S.csv" --out "$CH" > "$CH/state.log" 2>&1 || tail -3 "$CH/state.log"
    else
      $PY $H/eval_state_change_h.py --session $S --snapshots "$X/snapshots.jsonl" --out "$CH" > "$CH/state.log" 2>&1 || tail -3 "$CH/state.log"
    fi
    step "real $S scored"
  done
  rm -f $T
elif [[ $KIND == 3rscan ]]; then
  for room in $(grep -v '^#' $FT/datasets/3rscan/CHAINS_P3.txt | cut -c1-8); do
    R3=$RUN/3rscan/p3_$room; M=$FT/wt_takeover_runs/3rscan/p3_$room/panoptic_rerun; J=/dev/shm/panoptic-3rscan-$room
    mapfile -t visits < <(python3 -c "import json,sys; print('\n'.join(json.load(open(sys.argv[1]))['visits']))" $FT/datasets/3rscan/mapping_configs/p3_$room/chain.json)
    for k in "${!visits[@]}"; do
      scan=${visits[$k]}; t0=$((1000000000 + 200000000000 * k))
      khronos_times 3rscan "$room $k" > "$OUT/khronos_times_${room}_$k.txt"
      [[ -d $OUT/p3_$room/maps_$k ]] || at_times "$R3/${scan}_snapshots" "$t0" "$OUT/khronos_times_${room}_$k.txt" "$OUT/p3_$room/maps_$k"
      maps 3rscan "p3_$room/v$k" "$OUT/p3_$room/maps_$k" "$R3/$scan.panmap" "$((t0 + $(stamp max $RUN/3rscan_input/p3_$room/$scan/timestamps.csv)))" \
        --stamp-base "$t0" --start-ns "$t0"
      mkdir -p "$M/visit_$k"; ln -sfn "$R3/$scan.panmap" "$M/visit_$k/session_v$k.panmap"
    done
    for k in "${!visits[@]}"; do cat "$OUT/p3_$room/v$k/objects.jsonl"; done > $J.jsonl
    rm -rf "$M/khronos_metrics" "$M/state"; mkdir -p "$M/state"
    to4d $FT/wt_takeover_runs/3rscan/p3_$room/ours_290e110/visit_0/final.4dmap.zpk $J.jsonl $J.4dmap > "$OUT/p3_$room/build.log" 2>&1
    $PY $FT/eval_fast/vendor/pack/map4d_pack.py pack $J.4dmap --out "$M/state/final.4dmap.zpk" --remove-source > "$OUT/p3_$room/pack.log" 2>&1
    rm -f $J.jsonl
    THREERSCAN_OBJECTS_ONLY=1 THREERSCAN_JOBS=2 $PY $FT/eval/threerscan_khronos_eval.py p3_$room khronos "$M" > "$OUT/p3_$room/objects.log" 2>&1 || tail -3 "$OUT/p3_$room/objects.log"
    $PY $FT/eval/threerscan_chain_eval.py p3_$room panoptic "$M" > "$OUT/p3_$room/chain.log" 2>&1 || tail -3 "$OUT/p3_$room/chain.log"
    step "3RScan $room: $(tail -1 "$OUT/p3_$room/objects.log" | cut -c1-120) | $(grep 'chain:' "$OUT/p3_$room/chain.log" | cut -c1-100)"
  done
else
  SYN=/mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts; HS=$FT/eval/synthetic_base_v38/harness
  ORIG=$FT/datasets/synthetic_ab/evaluation_complete_20260917; I=$RUN/synthetic_input
  declare -A ORIGIN=([a]=1000000000 [b]=141000000000)            # Synthetic_ABC/scripts/cvpr_eval_common.py ORIGIN
  base=$OUT/harness; cbase=$OUT/harness_changes     # the scene reading (G1, objects); the changes reading (states)
  mkdir -p "$base/ours/native" "$base/online/ours" "$cbase/ours/native" "$cbase/eval_logs"
  for s in a b; do
    t0=${ORIGIN[$s]}; more=()
    [[ $s == b ]] && more=(--seed "$RUN/synthetic/session_a.panmap" --seed-ns "$t0")
    khronos_times synthetic $s > "$OUT/khronos_times_$s.txt"
    at_times "$RUN/synthetic/session_${s}_snapshots" "$t0" "$OUT/khronos_times_$s.txt" "$OUT/maps_$s"
    maps synthetic $s "$OUT/maps_$s" "$RUN/synthetic/session_$s.panmap" \
      "$((t0 + $(stamp max $I/session_$s/timestamps.csv)))" --stamp-base "$t0" --start-ns "$t0" --native "${more[@]}"
    ln -sfn "$OUT/$s/native" "$base/ours/native/$s"; ln -sfn "$OUT/$s/native_changes" "$cbase/ours/native/$s"
  done
  harness() { local root=$1; shift; env PYTHONPATH="$SYN" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
      HARNESS_IN_ROOT="$root" HARNESS_OUT_ROOT="$root" HARNESS_CAUSAL_BANK="$ORIG/causal_reference" \
      HARNESS_SYN_SCRIPTS="$SYN" HARNESS_SKIP_RUN_CHECKS=1 nice -n 10 "$PY" "$@"; }
  mkdir -p "$base/eval_logs"
  harness $base $HS/evaluate_cross_session_geometry.py --methods ours > "$base/eval_logs/geometry.log" 2>&1 || { tail -5 "$base/eval_logs/geometry.log"; exit 1; }
  harness $base $HS/prepare_historical_object_eval.py --native-only --ours-only > "$base/eval_logs/online_preparation.log" 2>&1 || { tail -5 "$base/eval_logs/online_preparation.log"; exit 1; }
  T=/dev/shm/panoptic-synthetic-template.4dmap; template $FT/wt_takeover_runs/synthetic_a_once/final.4dmap.zpk $T
  for s in a b; do
    M=/dev/shm/panoptic-synthetic-$s.4dmap; rm -rf "$base/online/ours/$s"
    to4d $T "$OUT/$s/objects.jsonl" $M > "$base/eval_logs/build_$s.log" 2>&1
    bash -c 'source "$1" >/dev/null 2>&1; shift; exec "$@"' x $OFF/environment.sh \
      $FT/eval/synthetic_object_eval_official/khronos_historical_object_eval $M "$base/history/ours_${s}_online_manifest.json" \
      "$base/online/ours/$s" > "$base/eval_logs/online_object_$s.log" 2>&1 || { tail -5 "$base/eval_logs/online_object_$s.log"; exit 1; }
    rm -f $M
  done
  rm -f $T
  $PY $A/run_synthetic_eval.py --aggregate-online "$base" "$base/online" > "$base/eval_logs/online_summary.log" 2>&1
  harness $cbase $HS/score_complete_states.py --ours-only > "$cbase/eval_logs/state.log" 2>&1 || { tail -5 "$cbase/eval_logs/state.log"; exit 1; }
  step "synthetic: $(cat "$base/online/ONLINE_OBJECT_SUMMARY.csv" | cut -c1-200)"
fi
step "$KIND done: $OUT"
