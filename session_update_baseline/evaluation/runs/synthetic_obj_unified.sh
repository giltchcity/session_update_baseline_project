#!/bin/bash
# Synthetic object F1 with the one object setting of every dataset (0.5 m, every 100th surface point, semantics required,
# no missegmentation compensation; eval/synthetic_object_eval_official, 2026-10-29), for ours (synthetic_a_once +
# synthetic_b_states, the maps of the results table; a full evaluation, the earlier one kept no exports) and for
# Khronos continuous (objects only, from its earlier evaluation). Evaluation only, nothing remapped.
R=/home/jixian/Desktop/FT/wt_takeover_runs; PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python
step() { echo "$(date +%H:%M) $*"; }
T=$R/eval_tools_official            # as eval_tools_m050, with the object evaluator of the one setting
mkdir -p $T
ln -sfn $R/eval_tools_m050/ours_export $T/ours_export
ln -sfn $R/eval_tools/khronos_eval_query_earlier $T/khronos_eval_query_earlier
ln -sfn /home/jixian/Desktop/FT/eval/synthetic_object_eval_official/ours_historical_object_eval $T/ours_historical_object_eval
maps=$(mktemp -d /dev/shm/synthetic-maps-XXXXXX)
ln -s $R/synthetic_a_once/final.4dmap.zpk $maps/session_a.4dmap.zpk
ln -s $R/synthetic_b_states/final.4dmap.zpk $maps/session_b.4dmap.zpk
SYN_MAPS=$maps SYN_OUT=$R/eval/obj100_states_synthetic SYN_REF=$R/eval/states_synthetic SYN_REF_LABEL=published_object_setting \
  SYNTHETIC_TOOLS_BIN=$T ionice -c3 nice -n 19 taskset -c 0-9 $PY $R/analysis/run_synthetic_eval.py > $R/logs/obj100_states_synthetic.log 2>&1
step "ours synthetic: exit $? ($(tail -1 $R/logs/obj100_states_synthetic.log))"
rm -r $maps
SYN_MAPS=$R/khronos_continuous/synthetic/sessions SYN_OUT=$R/eval/obj100_khronos_cont_synthetic SYNTHETIC_TOOLS=khronos \
  SYNTHETIC_TOOLS_BIN=/home/jixian/Desktop/FT/eval/synthetic_object_eval_official SYN_OBJECTS_FROM=$R/eval/khronos_cont_synthetic \
  SYN_REF=$R/eval/m050_khronos_cont_synthetic SYN_REF_LABEL=published_object_setting \
  ionice -c3 nice -n 19 taskset -c 0-9 python3 $R/analysis/run_synthetic_eval.py > $R/logs/obj100_khronos_cont_synthetic.log 2>&1
step "khronos synthetic: exit $? ($(tail -1 $R/logs/obj100_khronos_cont_synthetic.log))"
step "synthetic object rescoring done"
