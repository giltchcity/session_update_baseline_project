#!/usr/bin/env bash
# Synthetic Online Object F1 of any backend's timelines, the metric of MAIN_RESULTS.csv
# (Online_Object_full_duration_F1_pct / Online_Object_compat_nonempty_F1_pct), with the unchanged tools:
#   1 native export of every snapshot (emit_synthetic.write_native, as eval_synthetic_scene.sh)
#   2 query manifest: harness prepare_historical_object_eval.py --native-only --ours-only (unchanged)
#   3 the timeline as a Khronos 4D map: export_obj4d.py --plain-ids (node id = identity, the id the
#     manifest's ignore_prediction_nodes name) -> objects4d build
#   4 ObjectEvaluator: the prebuilt ours_historical_object_eval of evaluation_complete_20260917 (unchanged,
#     its own V37 environment) -> online/ours/<s>/object.csv
#   5 aggregation: the block marked "verbatim from the original" in harness summarize_ours_online.py
#     (duration-weighted over the native-snapshot queries), copied below without change. The rest of
#     that script checks Khronos mapper runs (live tracks, playback manifests) and does not apply.
# Intermediates (native export, OBJ4D, 4D map) are deleted unless KEEP_INTERMEDIATES=1.
#
#   synthetic_object_f1.sh RUN_DIR OUT_DIR        (RUN_DIR/session_{a,b}/timeline.pkl)
set -Eeuo pipefail
RUN=$(realpath "$1"); OUT=$(realpath -m "$2")
FT=/home/jixian/Desktop/FT
PROJECT=$FT/session_update_baseline_project
PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python      # harness python (as eval_synthetic_scene.sh)
GPY=$FT/envs/gs-cu128/bin/python
H=$FT/eval/synthetic_base_v38/harness
SYN=/mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts
ORIG=$FT/datasets/synthetic_ab/evaluation_complete_20260917
KENV=$PROJECT/session_update_baseline/scripts/khronos_env.sh   # evaluate_ours.sh KHRONOS_ENV default
TOOL=$FT/build_artifacts/update_layer_objects4d/objects4d
TEMPLATE=$FT/build_artifacts/update_layer_objects4d/template.4dmap
OENV=$FT/build_artifacts/official_khronos_63faadde_abc/environment.sh
HERE=$(dirname "$(realpath "$0")")
mkdir -p "$OUT/eval_logs"
[ ! -e "$OUT/online/ours" ] || { echo "already scored: $OUT/online/ours (ObjectEvaluator appends; remove it to redo)"; exit 1; }
for s in a b; do
  echo "[$(date +%T)] $s native export"
  OMP_NUM_THREADS=4 nice -n 10 "$GPY" - "$RUN/session_$s/timeline.pkl" "$OUT/ours/native/$s" 0.02 0.01 "$PROJECT" <<'EOF'
import sys
sys.path.insert(0, sys.argv[5])
from update_layer.eval.scenelist import load_timeline
from update_layer.eval.emit_synthetic import write_native
write_native(load_timeline(sys.argv[1]), sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
EOF
done
echo "[$(date +%T)] manifests"
env PYTHONPATH="$SYN" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    HARNESS_IN_ROOT="$OUT" HARNESS_OUT_ROOT="$OUT" HARNESS_CAUSAL_BANK="$ORIG/causal_reference" \
    HARNESS_SYN_SCRIPTS="$SYN" HARNESS_SKIP_RUN_CHECKS=1 nice -n 10 \
    "$PY" "$H/prepare_historical_object_eval.py" --native-only --ours-only > "$OUT/eval_logs/online_preparation.log" 2>&1
for s in a b; do
  echo "[$(date +%T)] $s 4D map + ObjectEvaluator"
  OMP_NUM_THREADS=4 nice -n 10 "$GPY" "$HERE/export_obj4d.py" "$RUN/session_$s/timeline.pkl" "$OUT/objects_$s.obj4d" --plain-ids \
    > "$OUT/eval_logs/export_$s.log" 2>&1
  bash -c "source $OENV >/dev/null 2>&1; exec $TOOL build $TEMPLATE $OUT/objects_$s.obj4d $OUT/map_$s.4dmap" > "$OUT/eval_logs/build_$s.log" 2>&1
  ( set +u; unset SESSION_UPDATE_CANONICAL_PREFIX SESSION_UPDATE_CANONICAL_BUILD SESSION_UPDATE_CANONICAL_ROOT
    source "$KENV" >/dev/null 2>&1; set -u
    exec nice -n 10 "$ORIG/ours_historical_object_eval" "$OUT/map_$s.4dmap" "$OUT/history/ours_${s}_online_manifest.json" \
         "$OUT/online/ours/$s" ) > "$OUT/eval_logs/ours_${s}_online_object.log" 2>&1
done
echo "[$(date +%T)] aggregation"
"$PY" - "$OUT" <<'EOF'
import csv, json, sys
from pathlib import Path
C = Path(sys.argv[1])
def read(path): return list(csv.DictReader(path.open()))
def write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f: w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)
online = []; online_queries = []
for method in ['ours']:
    for s in 'ab':
        # ---- verbatim from the original (Online Object aggregation) ----
        om=json.loads((C/f'history/{method}_{s}_online_manifest.json').read_text());oo=read(C/f'online/{method}/{s}/object.csv')
        assert len(oo)==len(om['queries']) and len({(r['Name'],r['Query']) for r in oo})==len(oo)
        orows=[]
        for j,(r,q) in enumerate(zip(oo,om['queries'])):
            assert int(r['Name'])==j and int(r['Query'])==q['query_stamp_ns']
            tp=int(r['NumObjDetected']);fp=int(r['NumObjHallucinated']);fn=int(r['NumObjMissed'])
            t=q['query_time_s'];dt=om['queries'][j+1]['query_time_s']-t if j+1<len(oo) else 0
            assert dt>=0
            orows.append(dict(method=method,session=s,time_s=t,duration_s=dt,TP=tp,FP=fp,FN=fn,
                Precision=tp/(tp+fp) if tp+fp else 0,Recall=tp/(tp+fn) if tp+fn else 0,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0))
        duration=sum(r['duration_s'] for r in orows);assert abs(duration-om['queries'][-1]['query_time_s'])<1e-7
        nonempty=[r for r in orows if r['TP']+r['FP']>0];ne_duration=sum(r['duration_s'] for r in nonempty)
        online.append(dict(method=method,session=s,duration_s=duration,queries=len(orows),
            **{k:sum(r[k]*r['duration_s'] for r in orows)/duration for k in ['Precision','Recall','F1']},
            nonempty_output_duration_s=ne_duration,compatibility_nonempty_output_F1=sum(r['F1']*r['duration_s'] for r in nonempty)/ne_duration,
            **{'final_'+k:orows[-1][k] for k in ['TP','FP','FN','Precision','Recall','F1']}))
        online_queries.extend(orows)
        # ---- end verbatim ----
write(C/'online/ONLINE_OBJECT_SUMMARY.csv', online); write(C/'online/ONLINE_OBJECT_QUERIES.csv', online_queries)
for r in online:
    print(f"{r['session']}: full-duration P {100*r['Precision']:.2f} R {100*r['Recall']:.2f} F1 {100*r['F1']:.2f} | "
          f"non-empty F1 {100*r['compatibility_nonempty_output_F1']:.2f} | final TP/FP/FN {r['final_TP']}/{r['final_FP']}/{r['final_FN']}")
EOF
[ "${KEEP_INTERMEDIATES:-0}" = 1 ] || rm -rf "$OUT/ours/native" "$OUT"/objects_?.obj4d "$OUT"/map_?.4dmap
