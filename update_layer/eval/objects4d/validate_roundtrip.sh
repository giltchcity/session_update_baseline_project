#!/usr/bin/env bash
# Round trip through objects4d must not change the official Object F1:
#   MAP (.4dmap.zpk) --official ObjectEvaluator--> A/results/static_objects.csv
#   MAP --objects4d dump--> OBJ4D --objects4d build--> REBUILT.4dmap --official--> B/results/static_objects.csv
# Maps are streamed through named pipes (tools/map4d_pack.py cat); nothing unpacked is written.
#
#   validate_roundtrip.sh SESSION(a|b|c) MAP.4dmap.zpk OUT_DIR
set -euo pipefail
s=$1; MAP=$(realpath "$2"); OUT=$(realpath -m "$3")
FT=/home/jixian/Desktop/FT
TOOL=$FT/build_artifacts/update_layer_objects4d/objects4d
ENVSH=$FT/build_artifacts/official_khronos_63faadde_abc/environment.sh
CFG=$FT/results/current_version_20260921/real_abc/objects/gt_${s}_latest/official_objects.yaml
mkdir -p "$OUT"
stream() {  # stream() ZPK PIPE: named pipe fed with the unpacked map
  rm -f "$2"; mkfifo "$2"; python3 $FT/tools/map4d_pack.py cat "$1" > "$2" &
}
if [ -f "$OUT/original/results/static_objects.csv" ]; then
  echo "[$(date +%T)] official on the original map: already scored, kept"
else
  echo "[$(date +%T)] official on the original map"
  python3 $FT/tools/abc_eval_v2/official_native_eval_run_zpk.py --map "$MAP" --config "$CFG" --output "$OUT/original" \
    > "$OUT/original.log" 2>&1
fi
echo "[$(date +%T)] dump"
stream "$MAP" "$OUT/pipe_dump.4dmap"
bash -c "source $ENVSH >/dev/null 2>&1; $TOOL dump $OUT/pipe_dump.4dmap $OUT/objects.obj4d"
wait; rm -f "$OUT/pipe_dump.4dmap"
echo "[$(date +%T)] build"
stream "$MAP" "$OUT/pipe_tmpl.4dmap"
bash -c "source $ENVSH >/dev/null 2>&1; $TOOL build $OUT/pipe_tmpl.4dmap $OUT/objects.obj4d $OUT/rebuilt.4dmap"
wait; rm -f "$OUT/pipe_tmpl.4dmap"
echo "[$(date +%T)] official on the rebuilt map"
python3 $FT/tools/abc_eval_v2/official_native_eval_run.py --map "$OUT/rebuilt.4dmap" --config "$CFG" --output "$OUT/rebuilt" \
  > "$OUT/rebuilt.log" 2>&1
echo "[$(date +%T)] compare"
# Object detection columns must be identical. The Appeared*/Disappeared* change columns are not
# preserved: they read first_observed_ns, which dump replaces for static objects (see objects4d.cpp).
python3 - "$OUT/original/results/static_objects.csv" "$OUT/rebuilt/results/static_objects.csv" <<'PY'
import csv, sys
a, b = (list(csv.DictReader(open(f))) for f in sys.argv[1:3])
cols = [k for k in a[0] if not k.startswith(("Appeared", "Disappeared"))]
bad = [k for k in cols if len(a) != len(b) or any(x[k] != y[k] for x, y in zip(a, b))]
chg = sorted({k for k in a[0] if k not in cols for x, y in zip(a, b) if x[k] != y[k]})
print("IDENTICAL object columns" if not bad else f"DIFFERENT object columns: {bad}")
print(f"change columns (not preserved) that differ: {chg}")
PY
