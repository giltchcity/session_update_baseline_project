#!/usr/bin/env bash
# Real room, every method rescored by eval_fast with the static-object matching distance at 0.5 m
# (EVAL_FAST_CONFIGS=configs_match050; everything else as before), one visit at a time (the exports live in /dev/shm),
# into eval/m050_METHOD_{a,b,c}; then one table. Evaluation only.
#   ours            real_{a,b,c}_tk (B against A, C against B as the prior)
#   khronos_pv      official per-visit semantic Khronos (runs/official_khronos_63faadde_{A,B,C}_full_20260909)
#   khronos_cont    official semantic Khronos continuous A->B->C (khronos_continuous/real)
#   khronos_inst    official Khronos fed instances, MaxIoUTracker, continuous (khronos_continuous/real_inst_maxiou)
#   khronos_inst_ext  the same with ExternalTracker (khronos_continuous/real_inst_external), when its map exists
# usage: real_rescore_050.sh
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; K=$R/khronos_continuous
step() { echo "$(date +%H:%M) $*"; }
score() {  # score METHOD SESSION MAP PRIOR [MOMENTS]
  local out=$R/eval/m050_$1_$2 prior=() moments=()
  if [[ -e $out ]]; then
    ls $out/real_abc/objects/*/weighted_nr/OBJECT_WEIGHTED_NR.json >/dev/null 2>&1 && { step "$1 $2: already scored"; return; }
    mv $out $out.partial.$(date +%H%M%S)   # an interrupted scoring, kept aside
  fi
  [[ -n $4 ]] && prior=(--prior $4)
  [[ -n ${5:-} ]] && moments=(--moments $5)
  (cd $FT && EVAL_FAST_CONFIGS=configs_match050 EVAL_FAST_JOBS=4 SESSION_UPDATE_CANONICAL_PREFIX=$FT/wt_290e110_build/canonical/install \
    ionice -c3 nice -n 19 taskset -c 0-9 /home/jixian/Desktop/miniconda3/bin/python eval_fast/run.py --map $3 --session $2 \
    --name m050_$1 --mode full --out $out "${prior[@]}" "${moments[@]}") > $out.log 2>&1
  step "$1 $2: scored (exit $?)"
}
continuous() {  # continuous METHOD MAP: the three visits of one continuous map
  local m; m=$(python3 $R/analysis/khronos_continuous_slices.py $2 --moments real)
  local prev=""
  while read -r s first last; do score $1 $s $2 "${prev:+$2}" $first:$last; prev=$s; done <<< "$m"   # one visit at a time
}
score ours a $R/real_a_tk/final.4dmap.zpk "" & score ours b $R/real_b_tk/final.4dmap.zpk $R/real_a_tk/final.4dmap.zpk &
[[ -s $R/real_c_tk/final.4dmap.zpk ]] && score ours c $R/real_c_tk/final.4dmap.zpk $R/real_b_tk/final.4dmap.zpk &   # not on disk now
wait
P=$FT/runs/official_khronos_63faadde; A=${P}_A_full_20260909/state/final.4dmap.zpk
score khronos_pv a $A "" & score khronos_pv b ${P}_B_full_20260909/state/final.4dmap.zpk $A &
score khronos_pv c ${P}_C_full_20260909/state/final.4dmap.zpk $A & wait
continuous khronos_cont $K/real/state/final.4dmap.zpk
continuous khronos_inst $K/real_inst_maxiou/state/final.4dmap.zpk
[[ -s $K/real_inst_external/state/final.4dmap.zpk ]] && continuous khronos_inst_ext $K/real_inst_external/state/final.4dmap.zpk
python3 - <<'EOF'
import json, glob
R = '/home/jixian/Desktop/FT/wt_takeover_runs/eval'
print('method       visit  objF1(0.5m)  P      R      G1@5   D2F1   D3F1')
for m in ('ours', 'khronos_pv', 'khronos_cont', 'khronos_inst', 'khronos_inst_ext'):
    for s in 'abc':
        o = glob.glob(f'{R}/m050_{m}_{s}/**/OBJECT_WEIGHTED_NR.json', recursive=True)
        if not o:
            continue
        w = json.load(open(o[0]))['duration_weighted_nr']
        g = glob.glob(f'{R}/m050_{m}_{s}/**/geometry_viewer_G1.json', recursive=True)
        g1 = json.load(open(g[0])).get('F1_05cm') if g else None
        S = glob.glob(f'{R}/m050_{m}_{s}/**/SUMMARY.json', recursive=True)
        d2 = d3 = None
        if S:
            ch = json.load(open(S[0])).get('change') or {}
            d2 = (ch.get('D2') or {}).get('published_policy', {}).get('F1')
            d3 = (ch.get('D3') or {}).get('published_policy', {}).get('F1')
        print(f'{m:14s} {s}  {w["F1"]!s:>8}  {w["P"]!s:>6} {w["R"]!s:>6}  {g1 if g1 is None else round(100 * g1, 2)!s:>6} {d2!s:>6} {d3!s:>6}')
EOF
step "real rescore 0.5 m done"
