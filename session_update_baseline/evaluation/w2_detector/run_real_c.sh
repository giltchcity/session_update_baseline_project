#!/bin/bash
# Real C of the W2 ablation arms (after run_real_b.sh): each arm continues from its own real B map with the same switch,
# mapped as the main-table rows (ablation_valid_domain.sh real()), scored by eval_fast full mode with that B as prior.
# One run at a time on cores 8-15, nice; each waits for 12 GB of free memory. Output: OUT (default real_b/ here).
# usage: run_real_c.sh [ARM ...]   (default: method w1 fixedbeta eta9 eta999 off)
set -eo pipefail
FT=/home/jixian/Desktop/FT; HERE=$FT/wt_takeover_runs/w2_detector_replay_20261116; O=${OUT:-$HERE/real_b}
WORK=$FT/wt_tesse_fix/session_update_baseline; BUILD=$FT/wt_tesse_fix_build; L=$FT/datasets/local_ab; CORES=8-15
step() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $O/progress.log; }
wait_memory() { while (( $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) < 12 )); do sleep 60; done; }
common=(--image-scale 0.5 --sensor-max-range 5.0 --change-detection-every-n-backend-updates 5
  --mapper-config $WORK/configs/room18_instance_5cm.yaml --input-config $L/mapping_configs/nss_flat_input.yaml
  --labelspace-config $L/mapping_configs/nss_ade20k_room_label_space.yaml --physical-catalog $L/mapping_configs/room18_physical_catalog.json)
switch() { case $1 in method) ;; w1) echo SESSION_CORE_ABSENCE_WEIGHTS=one ;; fixedbeta) echo SESSION_CORE_ABSENCE_BETA=fixed ;;
  eta9) echo SESSION_CORE_ABSENCE_ETA=9 ;; eta999) echo SESSION_CORE_ABSENCE_ETA=999 ;; off) echo SESSION_CORE_ABSENCE_OFF=1 ;;
  *) echo "unknown arm $1" >&2; exit 1 ;; esac; }

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

arms=("$@"); [[ ${#arms[@]} == 0 ]] && arms=(method w1 fixedbeta eta9 eta999 off)
for arm in "${arms[@]}"; do
  B=$O/real_b_$arm; out=$O/real_c_$arm
  [[ -s $B/final.4dmap.zpk ]] || { step "real_c_$arm: no real B map of the arm, skipped"; continue; }
  if [[ -e $out ]]; then step "real_c_$arm: exists, kept"; else
    wait_memory
    (cd $WORK && env $(switch $arm) SESSION_UPDATE_CANONICAL_ROOT=$BUILD/canonical SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install \
      SESSION_UPDATE_CANONICAL_BUILD=$BUILD/baseline taskset -c $CORES nice -n 10 /usr/bin/time -v bash scripts/run_session.sh \
      --output-state $out --session-id w2_real_c_$arm --tf-settle-s .02 --play-rate 100 --finalization-timeout-s 86400 \
      --ack-timeout-s 86400 --discovery-timeout-s 86400 \
      --run-dir $L/rgbd/session_c_20260822_182703_387_flat_rgbd_30hz_1080p --semantic-dir $L/semantics/session_c \
      --instance-dir $L/instance_labels/session_c --session-start-ns 1787394423387000000 \
      --world-transform $L/alignment/session_c_to_session_a.txt --input-state $B "${common[@]}") > $O/logs/real_c_$arm.log 2>&1
    seen=$( (grep -rh "ABLATION: absence test" $out/control/logs 2>/dev/null || true) | head -1)
    if [[ $arm == method && -n $seen ]] || [[ $arm != method && -z $seen ]]; then step "real_c_$arm: STOP, switch seen by the mapper: '$seen'"; exit 1; fi
    step "real_c_$arm: mapped in $(grep -o 'Elapsed (wall clock) time.*' $O/logs/real_c_$arm.log | cut -d' ' -f8)"
  fi
  E=$O/eval_c_$arm; rm -rf $E
  (cd $FT && SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install ionice -c3 nice -n 19 taskset -c $CORES \
    /home/jixian/Desktop/miniconda3/bin/python eval_fast/run.py --map $out/final.4dmap.zpk --session c \
    --name w2_c_$arm --mode full --out $E --prior $B/final.4dmap.zpk) > $O/logs/eval_c_$arm.log 2>&1
  step "C $arm scored: $(summary $E)"
done
step "C done"
