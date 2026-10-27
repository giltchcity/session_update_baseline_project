#!/usr/bin/env bash
# Real A and real C with a state limiting the frames before it once, when it starts (wt_resupport, built in
# wt_resupport_build): A without prior into real_a_states, C from real_b_once into real_c_states, each scored by
# eval_fast (0.5 m) into eval/m050_states_{a,c} and compared with the read-once rows (m050_once_{a,c}, the same priors).
# Nothing in here stops a run. usage: real_states_check.sh
set -eo pipefail
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; L=$FT/datasets/local_ab
WORK=$FT/wt_resupport/session_update_baseline; BUILD=$FT/wt_resupport_build
step() { echo "$(date +%H:%M) $*"; }
map() {  # map NAME run_session.sh arguments...
  local name=$1; shift; [[ -e $R/$name ]] && mv $R/$name $R/$name.before.$(date +%m%d%H%M%S)
  (cd $WORK && SESSION_UPDATE_CANONICAL_ROOT=$BUILD/canonical SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install \
    SESSION_UPDATE_CANONICAL_BUILD=$BUILD/baseline taskset -c 0-9 nice -n 10 bash scripts/run_session.sh --output-state $R/$name \
    --session-id $name --tf-settle-s .02 --play-rate 100 --finalization-timeout-s 86400 --ack-timeout-s 86400 \
    --discovery-timeout-s 86400 --image-scale 0.5 --sensor-max-range 5.0 --change-detection-every-n-backend-updates 5 \
    --mapper-config $WORK/configs/room18_instance_5cm.yaml --input-config $L/mapping_configs/nss_flat_input.yaml \
    --labelspace-config $L/mapping_configs/nss_ade20k_room_label_space.yaml \
    --physical-catalog $L/mapping_configs/room18_physical_catalog.json "$@") > $R/logs/${name}_mapping.log 2>&1
  step "$name: mapped, $(head -1 $R/$name/control/logs/khronos.log | cut -c10-17) -> $(tail -1 $R/$name/control/logs/khronos.log | cut -c10-17), $(grep -ho 'publish_to_ack_ms=[0-9.]*' $R/$name/control/logs/player.log | tail -1)"
}
score() {  # score SESSION NAME [PRIOR]
  [[ -e $R/eval/m050_states_$1 ]] && mv $R/eval/m050_states_$1 $R/eval/m050_states_$1.before.$(date +%m%d%H%M%S)
  local prior=(); [[ -n ${3:-} ]] && prior=(--prior $3/final.4dmap.zpk)
  (cd $FT && SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install ionice -c3 nice -n 19 taskset -c 0-9 \
    /home/jixian/Desktop/miniconda3/bin/python eval_fast/run.py --map $R/$2/final.4dmap.zpk --session $1 --name m050_states \
    --mode full --out $R/eval/m050_states_$1 "${prior[@]}") > $R/eval/m050_states_$1.log 2>&1
  step "real $1 states: scored into eval/m050_states_$1"
}
map real_a_states --run-dir $L/rgbd/session_a_20260809_204010_201_flat_rgbd_30hz_1080p --semantic-dir $L/semantics/session_a \
  --instance-dir $L/instance_labels/session_a --session-start-ns 1786279210201000000
score a real_a_states
map real_c_states --run-dir $L/rgbd/session_c_20260822_182703_387_flat_rgbd_30hz_1080p --semantic-dir $L/semantics/session_c \
  --instance-dir $L/instance_labels/session_c --session-start-ns 1787394423387000000 \
  --world-transform $L/alignment/session_c_to_session_a.txt --input-state $R/real_b_once
score c real_c_states $R/real_b_once
python3 - <<'PY'
import json, glob
R = '/home/jixian/Desktop/FT/wt_takeover_runs/eval'
def row(d):
    o = glob.glob(f'{R}/{d}/**/OBJECT_WEIGHTED_NR.json', recursive=True)
    w = json.load(open(o[0]))['duration_weighted_nr'] if o else {}
    g = glob.glob(f'{R}/{d}/**/geometry_viewer_G1.json', recursive=True)
    g1 = json.load(open(g[0])).get('F1_05cm') if g else None
    S = glob.glob(f'{R}/{d}/**/SUMMARY.json', recursive=True)
    ch = (json.load(open(S[0])).get('change') or {}) if S else {}
    return [w.get('F1'), w.get('P'), w.get('R'), None if g1 is None else round(100 * g1, 2),
            (ch.get('D2') or {}).get('published_policy', {}).get('F1'), (ch.get('D3') or {}).get('published_policy', {}).get('F1')]
print('visit  method          objF1   P      R      G1@5   D2     D3')
for s in 'ac':
    for name, d in (('read-once', f'm050_once_{s}'), ('states once', f'm050_states_{s}')):
        print(f'{s}      {name:14s}', '  '.join(f'{x!s:>6}' for x in row(d)))
PY
step "states check done"
