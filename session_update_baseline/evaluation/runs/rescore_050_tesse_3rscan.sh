#!/usr/bin/env bash
# TESSE and 3RScan rescored with the static-object matching distance at 0.5 m (configs/khronos/*.yaml now 0.5 m; the
# 2 m configs are in archive/eval_configs_2m_20261024). Evaluation only, nothing remapped.
#   TESSE: tesse_online_eval.py ours4d (8 workers) per map into eval/tesse_{paper,apt}_qj/NAME_m050, then the shared
#     grid of ours against Khronos;
#   3RScan: eval/threerscan_khronos_eval.py for ours (ours_290e110), ours uniform (timeline, ours_290e110_timeline),
#     Khronos semantic (khronos_scene) and Khronos instances (khronos_inst), room after room; the 2 m metrics are kept
#     as METHOD_DIR/khronos_metrics_2m.
# Then the mean tables.
# usage: rescore_050_tesse_3rscan.sh
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python
LOW=(ionice -c3 nice -n 19 taskset -c 0-9)
step() { echo "$(date +%H:%M) $*"; }
tesse() {  # tesse EVAL_DIR NAME MAP [--scene apartment]
  local out=$1/$2_m050
  [[ -s $out/TABLES_4D.txt ]] || { rm -rf $out; TESSE_EVAL_JOBS=8 "${LOW[@]}" $PY $R/analysis/tesse_online_eval.py ${4:+$4 $5} ours4d $3 $out > $out.log 2>&1; }
  [[ -s $out/snapshots.jsonl ]] && zstd -T4 -q --rm $out/snapshots.jsonl
  step "TESSE $(basename $1) $2: $(grep -A1 '^Data' $out/TABLES_4D.txt | tail -1 | tr -s ' ')"
}
O=$R/eval/tesse_paper_qj; A=$R/eval/tesse_apt_qj
tesse $O ours_290e110_5m_all $R/tesse_a_5m_290e110/final.4dmap.zpk
tesse $O khronos_5m_all $R/tesse/khronos_5m_cd5/state/final.4dmap.zpk
tesse $O khronos_inst_5m_all $R/tesse/khronos_inst_office/state/final.4dmap.zpk
tesse $A ours_290e110_5m_all $R/tesse_apt_5m_290e110/final.4dmap.zpk --scene apartment
tesse $A khronos_5m_all $R/tesse/khronos_apt_5m_cd5/state/final.4dmap.zpk --scene apartment
$PY $R/analysis/tesse_shared_grid.py $O/ours_290e110_5m_all_m050 $O/khronos_5m_all_m050 > $O/SHARED_GRID_m050.txt 2>&1
$PY $R/analysis/tesse_shared_grid.py $A/ours_290e110_5m_all_m050 $A/khronos_5m_all_m050 > $A/SHARED_GRID_m050.txt 2>&1
step "TESSE shared grids: $O/SHARED_GRID_m050.txt, $A/SHARED_GRID_m050.txt"

for ref in $(grep -v '^#' $FT/datasets/3rscan/CHAINS_P3.txt | cut -c1-8); do
  room=p3_$ref
  for job in "ours ours_290e110" "timeline ours_290e110_timeline" "khronos khronos_scene" "khronos khronos_inst"; do
   (set -- $job; d=$R/3rscan/$room/$2
    [[ -d $d ]] || exit 0
    [[ -e $d/khronos_metrics_2m && -s $d/khronos_metrics/METRICS.json ]] && exit 0   # already scored at 0.5 m
    [[ -d $d/khronos_metrics && ! -e $d/khronos_metrics_2m ]] && mv $d/khronos_metrics $d/khronos_metrics_2m
    rm -rf $d/khronos_metrics
    THREERSCAN_OBJECTS_ONLY=1 THREERSCAN_JOBS=2 "${LOW[@]}" $PY $FT/eval/threerscan_khronos_eval.py $room $1 $d > $R/3rscan/$room/khronos_metrics_$2_m050.log 2>&1 \
      && step "3RScan $room $2: $(tail -1 $R/3rscan/$room/khronos_metrics_$2_m050.log | cut -c1-160)" \
      || step "3RScan $room $2 FAILED ($(tail -1 $R/3rscan/$room/khronos_metrics_$2_m050.log | cut -c1-160))")
  done   # one method at a time, 2 snapshots at a time (the machine is shared)
done
python3 - <<'EOF'
import json, glob, statistics as st
R = '/home/jixian/Desktop/FT/wt_takeover_runs/3rscan'
print('3RScan, 0.5 m, mean over rooms (rooms with a value): ObjectF1 / ChangeF1 / ChangeF1Incl')
for d in ('ours_290e110', 'ours_290e110_timeline', 'khronos_scene', 'khronos_inst'):
    vals = {k: [] for k in ('ObjectF1', 'ChangeF1', 'ChangeF1Incl')}
    for f in sorted(glob.glob(f'{R}/p3_*/{d}/khronos_metrics/METRICS.json')):
        m = json.load(open(f))['metrics']
        for k in vals:
            v = (m.get(k) or {}).get('mean')
            if v is not None:
                vals[k].append(v)
    print(f'{d:24s}', ' / '.join(f'{st.mean(v):.1f} ({len(v)})' if v else '-' for v in vals.values()))
EOF
step "rescore 0.5 m TESSE + 3RScan done"
