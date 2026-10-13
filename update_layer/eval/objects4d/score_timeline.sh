#!/usr/bin/env bash
# Official Khronos Object P/R/F1 (online, all saved snapshots) of any backend's saved timeline:
#   timeline.pkl --export_obj4d--> pipe --objects4d build--> pipe (official/final.4dmap)
#   --unchanged official exp_pipeline--> official/results/static_objects.csv
#   --official postprocess--> post/NATIVE_ONLINE_SUMMARY.json
# Nothing intermediate touches the disk: the OBJ4D and the built map only pass through named pipes
# (the official loader reads final.4dmap once, sequentially, as with map4d_pack.py cat).
# Only ObjectPrecision/Recall/F1 are meaningful here (change metrics: see objects4d.cpp).
#
#   score_timeline.sh SESSION(a|b|c) TIMELINE.pkl OUT_DIR
set -euo pipefail
s=$1; TL=$(realpath "$2"); OUT=$(realpath -m "$3")
FT=/home/jixian/Desktop/FT
PY=$FT/envs/gs-cu128/bin/python
TOOL=$FT/build_artifacts/update_layer_objects4d/objects4d
TEMPLATE=$FT/build_artifacts/update_layer_objects4d/template.4dmap   # layer layout only (make_template.sh)
OFFICIAL=$FT/build_artifacts/official_khronos_63faadde_abc
CFG=$FT/results/current_version_20260921/real_abc/objects/gt_${s}_latest/official_objects.yaml
HERE=$(dirname "$(realpath "$0")")
mkdir -p "$OUT/official"
[ ! -e "$OUT/official/results/static_objects.csv" ] || { echo "already scored: $OUT"; exit 0; }
rm -f "$OUT/objects.pipe" "$OUT/official/final.4dmap"
mkfifo "$OUT/objects.pipe" "$OUT/official/final.4dmap"
echo '[INFO] Evaluation input directory only. Source mapper completion must be checked separately.' \
  > "$OUT/official/experiment_log.txt"
cmd=("$OFFICIAL/install/lib/khronos_eval/exp_pipeline" "$CFG" "$OUT/official" false true false)
$PY - "$OUT/official/PROVENANCE.json" "$TL" "$CFG" "${cmd[@]}" <<'PY'
import hashlib, json, sys
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
out, tl, cfg, *cmd = sys.argv[1:]
json.dump(dict(schema="update_layer-objects4d-official-v1", timeline=tl, timeline_sha256=sha(tl), config=cfg,
               config_sha256=sha(cfg), command=cmd, executable_sha256=sha(cmd[0]),
               delivered="timeline -> export_obj4d.py -> objects4d build -> named pipe final.4dmap"),
          open(out, "w"), indent=2)
PY
OMP_NUM_THREADS=4 $PY "$HERE/export_obj4d.py" "$TL" "$OUT/objects.pipe" > "$OUT/export.log" 2>&1 &
exp_pid=$!
bash -c "source $OFFICIAL/environment.sh >/dev/null 2>&1; exec $TOOL build $TEMPLATE $OUT/objects.pipe $OUT/official/final.4dmap" \
  > "$OUT/build.log" 2>&1 &
build_pid=$!
rc=0
bash -c 'source "$1" >/dev/null 2>&1; shift; exec "$@"' x "$OFFICIAL/environment.sh" "${cmd[@]}" \
  > "$OUT/official/native_pipeline.log" 2>&1 || rc=$?
wait $exp_pid || rc=$?
wait $build_pid || rc=$?
rm -f "$OUT/objects.pipe" "$OUT/official/final.4dmap"
[ $rc = 0 ] && [ -f "$OUT/official/results/static_objects.csv" ] || {
  echo "FAILED ($rc): see $OUT/{export,build}.log, official/native_pipeline.log"; exit 1; }
$PY $FT/tools/abc_eval_v2/official_native_eval_postprocess.py "$OUT/official/results/static_objects.csv" \
  --output "$OUT/post" > /dev/null
$PY -c "
import json; s = json.load(open('$OUT/post/NATIVE_ONLINE_SUMMARY.json'))['summary']
print(' '.join(f'{k}={s[k][\"mean\"]:.4f}' if s[k]['mean'] is not None else f'{k}=NaN' for k in ('ObjectPrecision', 'ObjectRecall', 'ObjectF1')))"
