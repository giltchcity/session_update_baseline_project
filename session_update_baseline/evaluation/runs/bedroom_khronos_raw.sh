#!/usr/bin/env bash
# One pretrained semantic prediction per RGB frame, native Khronos A/B/C,
# and the same evaluation as the unfinetuned BedRoom comparison.
set -eo pipefail
output=$(realpath -m "${1:?usage: bedroom_khronos_raw.sh OUTPUT_DIR}")
here=$(cd "$(dirname "$0")" && pwd)
ft=/home/jixian/Desktop/FT
experiments=$ft/wt_takeover_runs
bedroom=$experiments/bedroom_finetune_20261121
map=$output/map/state/final.4dmap.zpk
mkdir -p "$output"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export KHRONOS_LABEL_DIR="$output/raw_semantics"
export KHRONOS_LABELSPACE="$bedroom/labelspace_oneformer_things.yaml"
export KHRONOS_PLAYER="$ft/wt_tesse_fix/session_update_baseline/scripts/nss_flat_ros2_player.py"
export KHRONOS_INSTANCES=0 KHRONOS_SESSIONS="a b c"
unset REAL_INSTANCES KHRONOS_TRACKER

stage() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"; }
if [[ ! -s "$map" ]]; then
  stage "Predicting raw OneFormer semantics"
  "$ft/envs/clsplats-cu128/bin/python" -I -B "$here/bedroom_oneformer_raw.py" \
    "$ft/datasets/local_ab/rgbd" "$KHRONOS_LABEL_DIR" > "$output/predict.log" 2>&1
  stage "Mapping A/B/C with native Khronos"
  bash "$here/khronos_continuous.sh" real "$output/map" > "$output/map.log" 2>&1
fi

python3 "$here/khronos_continuous_slices.py" "$map" --moments real > "$output/moments.txt"
[[ "$(cut -d' ' -f1 "$output/moments.txt" | paste -sd ' ' -)" == "a b c" ]] || {
  echo "Missing one or more visits in the final map" >&2
  exit 1
}
export EVAL_FAST_CONFIGS="$bedroom/gt_renamed4/configs"
export EVAL_GT_BASELINE="$bedroom/gt_renamed4/baseline"
export EVAL_REVIEWED_CLASSES=7,10,24,33,35,41,74,75,92,112,115,139
export SESSION_UPDATE_CANONICAL_PREFIX="$ft/wt_takeover_build/canonical/install"
while read -r visit first last; do
  prior=()
  [[ "$visit" == a ]] || prior=(--prior "$map")
  stage "Scoring visit $visit, snapshots $first through $last"
  (cd "$ft" && ionice -c3 nice -n 19 taskset -c 0-9 \
    /home/jixian/Desktop/miniconda3/bin/python eval_fast/run.py \
    --map "$map" --moments "$first:$last" --session "$visit" --name khronos_raw_sem \
    --mode full --out "$output/real_${visit}_eval" "${prior[@]}") \
    > "$output/score_$visit.log" 2>&1
done < "$output/moments.txt"
python3 "$experiments/ablation_object_element_20261120/s6_row.py" "$output" | tee "$output/scores.txt"
