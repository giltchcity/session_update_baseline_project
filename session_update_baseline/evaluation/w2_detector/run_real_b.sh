#!/bin/bash
# Key ablations of the absence test (reviewer W2) on real B: one real A with the method (the common start), then real B
# from it once per arm, the latest version (wt_tesse_fix) with one switch (projected_physical_evidence.cpp, unset = the
# method); mapped as the main-table rows (ablation_valid_domain.sh real()), scored by eval_fast full mode with A as prior.
#   method     no switch            w1        SESSION_CORE_ABSENCE_WEIGHTS=one
#   fixedbeta  SESSION_CORE_ABSENCE_BETA=fixed    eta9 / eta999   SESSION_CORE_ABSENCE_ETA=9 / 999
#   off        SESSION_CORE_ABSENCE_OFF=1
# One run at a time on cores 8-15, nice; each waits for 12 GB of free memory. Output: OUT (default real_b/ here).
# usage: run_real_b.sh [ARM ...]   (default: all, in the order above)
set -eo pipefail
FT=/home/jixian/Desktop/FT; HERE=$FT/wt_takeover_runs/w2_detector_replay_20261116; O=${OUT:-$HERE/real_b}
WORK=$FT/wt_tesse_fix/session_update_baseline; BUILD=$FT/wt_tesse_fix_build; L=$FT/datasets/local_ab; CORES=8-15
step() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $O/progress.log; }
mkdir -p $O/logs
wait_memory() { while (( $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) < 12 )); do sleep 60; done; }
common=(--image-scale 0.5 --sensor-max-range 5.0 --change-detection-every-n-backend-updates 5
  --mapper-config $WORK/configs/room18_instance_5cm.yaml --input-config $L/mapping_configs/nss_flat_input.yaml
  --labelspace-config $L/mapping_configs/nss_ade20k_room_label_space.yaml --physical-catalog $L/mapping_configs/room18_physical_catalog.json)
switch() { case $1 in method) ;; w1) echo SESSION_CORE_ABSENCE_WEIGHTS=one ;; fixedbeta) echo SESSION_CORE_ABSENCE_BETA=fixed ;;
  eta9) echo SESSION_CORE_ABSENCE_ETA=9 ;; eta999) echo SESSION_CORE_ABSENCE_ETA=999 ;; off) echo SESSION_CORE_ABSENCE_OFF=1 ;;
  *) echo "unknown arm $1" >&2; exit 1 ;; esac; }

map() {  # map NAME ARM run_session.sh arguments...
  local name=$1 arm=$2; shift 2; local out=$O/$name
  [[ -e $out ]] && { step "$name: exists, kept"; return 0; }
  wait_memory
  (cd $WORK && env $(switch $arm) SESSION_UPDATE_CANONICAL_ROOT=$BUILD/canonical SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install \
    SESSION_UPDATE_CANONICAL_BUILD=$BUILD/baseline taskset -c $CORES nice -n 10 /usr/bin/time -v bash scripts/run_session.sh \
    --output-state $out --session-id w2_$name --tf-settle-s .02 --play-rate 100 --finalization-timeout-s 86400 \
    --ack-timeout-s 86400 --discovery-timeout-s 86400 "$@") > $O/logs/$name.log 2>&1
  local seen; seen=$( (grep -rh "ABLATION: absence test" $out/control/logs 2>/dev/null || true) | head -1)
  if [[ $arm == method && -n $seen ]] || [[ $arm != method && -z $seen ]]; then step "$name: STOP, switch seen by the mapper: '$seen'"; exit 1; fi
  step "$name: mapped in $(grep -o 'Elapsed (wall clock) time.*' $O/logs/$name.log | cut -d' ' -f8)${seen:+, $(echo $seen | grep -o 'ABLATION.*')}"
}

summary() {  # the table's numbers from eval_fast's SUMMARY.json and G1
  python3 - "$1" <<'E'
import json, sys, glob
d = sys.argv[1]
s = json.load(open(glob.glob(f'{d}/real_abc/summary/*/SUMMARY.json')[0])); g = glob.glob(f'{d}/real_abc/geometry/*/geometry_viewer_G1.json')
c = s['change']; f = lambda x: '{F1} ({TP}/{FP}/{FN})'.format(**x['published_policy'])
g1 = json.load(open(g[0])) if g else {}
print(f"object F1 {s['object']['duration_weighted_nr']['F1']} | D2 F1 (TP/FP/FN) {f(c['D2'])} | D3 F1 {f(c['D3'])} | G1 F1@5cm {g1.get('F1_05cm')}")
E
}

map real_a method --run-dir $L/rgbd/session_a_20260809_204010_201_flat_rgbd_30hz_1080p --semantic-dir $L/semantics/session_a \
  --instance-dir $L/instance_labels/session_a --session-start-ns 1786279210201000000 "${common[@]}"
arms=("$@"); [[ ${#arms[@]} == 0 ]] && arms=(method w1 fixedbeta eta9 eta999 off)
for arm in "${arms[@]}"; do
  map real_b_$arm $arm --run-dir $L/rgbd/session_b_20260810_030502_620_flat_rgbd_30hz_1080p --semantic-dir $L/semantics/session_b \
    --instance-dir $L/instance_labels/session_b --session-start-ns 1786302302620000000 \
    --world-transform $L/alignment/session_b_to_session_a.txt --input-state $O/real_a "${common[@]}"
  E=$O/eval_$arm; rm -rf $E
  (cd $FT && SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install ionice -c3 nice -n 19 taskset -c $CORES \
    /home/jixian/Desktop/miniconda3/bin/python eval_fast/run.py --map $O/real_b_$arm/final.4dmap.zpk --session b \
    --name w2_$arm --mode full --out $E --prior $O/real_a/final.4dmap.zpk) > $O/logs/eval_$arm.log 2>&1
  step "$arm scored: $(summary $E)"
done
step "done"
