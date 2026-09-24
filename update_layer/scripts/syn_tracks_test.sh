#!/usr/bin/env bash
# Synthetic check (CPU): Khronos-cadence track segments with and without t2's static frame selection.
set -u
P=/home/jixian/Desktop/FT/runs/layer_final_20260925
GPY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python
cd /home/jixian/Desktop/FT/wt_layer_43c
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for spec in "syn_tracks --segments tracks" "syn_tracks_static --segments tracks --static-frames"; do
  set -- $spec; name=$1; shift
  nice -n 10 $GPY -m update_layer.run_chain synthetic $P/$name --mode layer "$@" > $P/$name.log 2>&1 && \
    OMP_NUM_THREADS=2 nice -n 10 bash /home/jixian/Desktop/FT/eval/scene/eval_synthetic_scene.sh $P/$name $P/eval_$name 0.02 0.01 > $P/eval_$name.log 2>&1
  echo "$(date +%T) done $name"
done
