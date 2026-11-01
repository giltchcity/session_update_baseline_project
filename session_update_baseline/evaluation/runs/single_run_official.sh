#!/bin/bash
# The single runs scored as the TESSE table (user 11-01): each map through the TESSE table's 4D evaluation
# (tesse_online_eval.py ours4d: each snapshot alone, queried at every earlier map time; the official tables.py 4D
# average) with the visit's own ground truth (single_run_official.py: eval/khronos4d_gt_KIND_single_S.yaml). Ours as in
# the TESSE table, the run's map through eval/chain_fill (synthetic: final_4d.4dmap.zpk, chain_fill and the run's own
# motion, motion_fill); Khronos its raw map. Into eval/single_official/{ours,khronos}_{real,syn}_S, one map at a time.
# Lane 2 (cores 10-15): G1 of our synthetic maps scored here (the observer runs; the 11-01 table had the same
# configuration's earlier runs), run_synthetic_eval.py as single_run_compare.sh, into eval/obj100_single_synthetic_obs.
# usage: single_run_official.sh
set -eo pipefail
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python
E=$R/eval/single_official; mkdir -p $E
ENV="set +u; source /opt/ros/jazzy/setup.bash; source /home/jixian/ros2_ws/install/setup.bash; export LD_LIBRARY_PATH=$FT/wt_290e110_build/canonical/install/lib:\$LD_LIBRARY_PATH; exec \"\$@\""
step() { echo "$(date +%H:%M) $*"; }

score() {  # NAME KIND SESSION MAP
  [[ -s $E/$1/TABLES_4D.txt ]] && { step "$1: there"; return; }
  [[ -e $E/$1 ]] && mv $E/$1 $E/$1.before.$(date +%m%d%H%M%S)
  TESSE_EVAL_CONFIG=$R/eval/khronos4d_gt_$2_single_$3.yaml TESSE_EVAL_JOBS=8 ionice -c3 nice -n 19 taskset -c 0-9 \
    $PY $R/analysis/tesse_online_eval.py ours4d $4 $E/$1 > $E/$1.log 2>&1 || true
  if [[ -s $E/$1/TABLES_4D.txt ]]; then step "$1: $(grep -A1 '^Data' $E/$1/TABLES_4D.txt | tail -1 | tr -s ' ')"
  else step "$1: scoring failed (see $E/$1.log)"; fi
}

g1_lane() {
  maps=$(mktemp -d /dev/shm/synthetic-single-obs-XXXXXX)
  ln -s $R/synthetic_a_single_obs/final.4dmap.zpk $maps/session_a.4dmap.zpk
  ln -s $R/synthetic_b_single_obs/final.4dmap.zpk $maps/session_b.4dmap.zpk
  SYN_MAPS=$maps SYN_OUT=$R/eval/obj100_single_synthetic_obs SYN_REF=$R/eval/obj100_single_synthetic SYN_REF_LABEL=single_once \
    SYNTHETIC_TOOLS_BIN=$R/eval_tools_official ionice -c3 nice -n 19 taskset -c 10-15 $PY $R/analysis/run_synthetic_eval.py \
    > $R/logs/obj100_single_synthetic_obs.log 2>&1 && step "G1 synthetic observer runs: $(grep 'Mesh F1@5cm' $R/eval/obj100_single_synthetic_obs/FULL_TABLE.md | tr -s ' ' | tr '\n' ' ')" \
    || step "G1 synthetic observer runs: failed (see logs/obj100_single_synthetic_obs.log)"
  rm -r $maps
}
[[ -e $R/eval/obj100_single_synthetic_obs ]] || g1_lane &

w=/dev/shm/single-official; mkdir -p $w
for s in a b c; do
  src=$R/real_${s}_single/final.4dmap.zpk; [[ $s == a ]] && src=$R/real_a_states/final.4dmap.zpk
  [[ -s $E/ours_real_$s/TABLES_4D.txt ]] || step "ours real $s: $(bash -c "$ENV" chain-fill $FT/eval/chain_fill/chain_fill $w/real_$s.4dmap.zpk $src | tail -1)"
  score ours_real_$s real $s $w/real_$s.4dmap.zpk
  rm -f /dev/shm/single-official/real_$s.4dmap.zpk
  score khronos_real_$s real $s $FT/runs/official_khronos_63faadde_${s^^}_full_20260909/state/final.4dmap.zpk
done
score ours_syn_a synthetic a $R/synthetic_a_single_obs/final_4d.4dmap.zpk
score khronos_syn_a synthetic a $R/khronos_continuous/synthetic_5cm/sessions/session_a.4dmap.zpk
score ours_syn_b synthetic b $R/synthetic_b_single_obs/final_4d.4dmap.zpk
score khronos_syn_b synthetic b $R/khronos_continuous/synthetic_5cm_clean_b/state/final.4dmap.zpk
rmdir /dev/shm/single-official
wait
step "SINGLE OFFICIAL DONE"
