#!/usr/bin/env bash
# Point-map rows of the final consistent set, on the CPU: an external process holds the GPU at
# ~99 % (layer on the contended GPU ~270 s for synthetic A vs 31 s on 2 CPU threads). GaME keeps
# the GPU (final_game_gpu.sh). Same code for every mode: t2 decision-core port + element rule +
# closed-state background rule; inherited elements alive in the seed snapshot; scratch = empty seed.
#  1. candidate: t2's static-surface frame selection (--static-frames), synthetic + real
#  2. decision from $R/STATIC_DECISION ("on"/"off"; "off" if absent after 90 min)
#  3. layer, naive, scratch, carve, decision single / none (synthetic + real), retention
set -u
R=/home/jixian/Desktop/FT/runs
P=$R/layer_final_20260925
mkdir -p $P
GPY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python
cd /home/jixian/Desktop/FT/wt_layer_43c
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
keep() { local from=$1 to=$2; [[ -d $P/$from && ! -d $P/$to ]] || return 0
  mv $P/$from $P/$to; mv $P/eval_$from $P/eval_$to 2>/dev/null; mv $P/$from.log $P/$to.log 2>/dev/null; mv $P/eval_$from.log $P/eval_$to.log 2>/dev/null; }
score() { local ds=$1 run=$2 out=$3
  if [[ $ds == synthetic ]]; then bash /home/jixian/Desktop/FT/eval/scene/eval_synthetic_scene.sh $run $out 0.02 0.01 > $out.log 2>&1
  else nice -n 10 bash /home/jixian/Desktop/FT/eval/scene/eval_real_scene.sh $run $out 0.02 0.01 > $out.log 2>&1; fi; }
run() { local ds=$1 name=$2; shift 2
  if nice -n 5 $GPY -m update_layer.run_chain $ds $P/$name "$@" > $P/$name.log 2>&1; then score $ds $P/$name $P/eval_$name; echo "$(date +%T) done $name"
  else echo "$(date +%T) FAILED $name"; fi; }
echo "$(date +%T) points queue (CPU) started"
run synthetic syn_layer --mode layer
run synthetic syn_layer_static --mode layer --static-frames
run real real_layer --mode layer
run real real_layer_static --mode layer --static-frames
echo "$(date +%T) STATIC_TESTED"
for k in $(seq 1 90); do [[ -s $R/STATIC_DECISION ]] && break; sleep 60; done
d=$(cat $R/STATIC_DECISION 2>/dev/null || echo off); flag=""; [[ $d == on ]] && flag="--static-frames"
echo "$(date +%T) static decision: $d"
if [[ $d == on ]]; then
  keep syn_layer syn_layer_nostatic; keep syn_layer_static syn_layer
  keep real_layer real_layer_nostatic; keep real_layer_static real_layer
fi
run synthetic syn_naive --mode naive
run synthetic syn_scratch --mode scratch $flag
run synthetic syn_carve --mode carve
run synthetic syn_decision_single --mode layer --decision single $flag
run synthetic syn_decision_none --mode layer --decision none $flag
run real real_naive --mode naive
run real real_scratch --mode scratch $flag
run real real_carve --mode carve
run real real_decision_single --mode layer --decision single $flag
run real real_decision_none --mode layer --decision none $flag
bash /home/jixian/Desktop/FT/eval/scene/post_retention.sh synthetic $P > /dev/null 2>&1
bash /home/jixian/Desktop/FT/eval/scene/post_retention.sh real $P > /dev/null 2>&1
echo "$(date +%T) POINTS_DONE"
