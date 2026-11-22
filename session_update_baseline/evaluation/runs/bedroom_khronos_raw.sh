#!/usr/bin/env bash
# Export raw semantic IDs, then use the same Khronos mapping and scoring pipeline.
set -eo pipefail
output=$(realpath -m "${1:?usage: bedroom_khronos_raw.sh OUTPUT_DIR}")
here=$(cd "$(dirname "$0")" && pwd)
ft=/home/jixian/Desktop/FT
mkdir -p "$output"
if [[ ! -s "$output/map/state/final.4dmap.zpk" ]]; then
  HF_HUB_OFFLINE=1 "$ft/envs/clsplats-cu128/bin/python" -I -B "$here/bedroom_oneformer_raw.py" \
    "$ft/datasets/local_ab/rgbd" "$output/raw_semantics" > "$output/predict.log" 2>&1
fi
exec bash "$here/bedroom_khronos_sem.sh" "$output/raw_semantics" "$output"
