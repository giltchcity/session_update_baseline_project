#!/usr/bin/env bash
# Acceptance of the read-once surface (wt_resupport, built in wt_resupport_build): every session remapped with it, each B
# (and C) from the new map before it, with the commands of the accepted rows; each scored as its accepted row was.
#   real A -> B -> C   real_{a,b,c}_once, eval_fast (0.5 m) into eval/m050_once_{a,b,c}
#   synthetic A -> B   synthetic_{a,b}_once, run_synthetic_eval.py (0.5 m object tools) into eval/once_synthetic
#   Flat A -> B        flat_{a,b}_once, flat_eval.py into eval/g1final_flat_{a,b}_once
#   TESSE office, apartment   tesse_{a,apt}_5m_once, chain_fill 4D, tesse_online_eval.py ours4d (0.5 m)
#     into eval/tesse_paper_qj/ours_once_chain4d_5m_all_m050, eval/tesse_apt_qj/ours_once_chain4d_5m_all_m050
# Nothing in here stops a run. usage: once_chain.sh [real] [synthetic] [flat] [tesse]   (default all, in order)
set -eo pipefail
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; BUILD=$FT/wt_resupport_build; WORK=$FT/wt_resupport/session_update_baseline
PY=/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python
ENV="set +u; source /opt/ros/jazzy/setup.bash; source /home/jixian/ros2_ws/install/setup.bash; export LD_LIBRARY_PATH=$FT/wt_290e110_build/canonical/install/lib:\$LD_LIBRARY_PATH; exec \"\$@\""
step() { echo "$(date +%H:%M) $*"; }
fresh() { for p in "$@"; do [[ -e $p ]] && mv $p $p.before.$(date +%m%d%H%M%S); done; return 0; }
map() {  # map NAME LOG run_session.sh arguments...
  local name=$1 log=$2; shift 2; fresh $R/$name
  (cd $WORK && SESSION_UPDATE_CANONICAL_ROOT=$BUILD/canonical SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install \
    SESSION_UPDATE_CANONICAL_BUILD=$BUILD/baseline taskset -c 0-9 nice -n 10 bash scripts/run_session.sh --output-state $R/$name \
    --session-id $name --tf-settle-s .02 --play-rate 100 --finalization-timeout-s 86400 --ack-timeout-s 86400 \
    --discovery-timeout-s 86400 "$@") > $R/logs/$log 2>&1
  step "$name: mapped, $(head -1 $R/$name/control/logs/khronos.log | cut -c10-17) -> $(tail -1 $R/$name/control/logs/khronos.log | cut -c10-17), $(grep -ho 'publish_to_ack_ms=[0-9.]*' $R/$name/control/logs/player.log | tail -1)"
}
real() {
  local L=$FT/datasets/local_ab
  local common=(--image-scale 0.5 --sensor-max-range 5.0 --change-detection-every-n-backend-updates 5
    --mapper-config $WORK/configs/room18_instance_5cm.yaml --input-config $L/mapping_configs/nss_flat_input.yaml
    --labelspace-config $L/mapping_configs/nss_ade20k_room_label_space.yaml --physical-catalog $L/mapping_configs/room18_physical_catalog.json)
  map real_a_once real_a_once_mapping.log --run-dir $L/rgbd/session_a_20260809_204010_201_flat_rgbd_30hz_1080p \
    --semantic-dir $L/semantics/session_a --instance-dir $L/instance_labels/session_a --session-start-ns 1786279210201000000 "${common[@]}"
  map real_b_once real_b_once_mapping.log --run-dir $L/rgbd/session_b_20260810_030502_620_flat_rgbd_30hz_1080p \
    --semantic-dir $L/semantics/session_b --instance-dir $L/instance_labels/session_b --session-start-ns 1786302302620000000 \
    --world-transform $L/alignment/session_b_to_session_a.txt --input-state $R/real_a_once "${common[@]}"
  map real_c_once real_c_once_mapping.log --run-dir $L/rgbd/session_c_20260822_182703_387_flat_rgbd_30hz_1080p \
    --semantic-dir $L/semantics/session_c --instance-dir $L/instance_labels/session_c --session-start-ns 1787394423387000000 \
    --world-transform $L/alignment/session_c_to_session_a.txt --input-state $R/real_b_once "${common[@]}"
  for s in a b c; do
    fresh $R/eval/m050_once_$s
    local prior=(); [[ $s == b ]] && prior=(--prior $R/real_a_once/final.4dmap.zpk); [[ $s == c ]] && prior=(--prior $R/real_b_once/final.4dmap.zpk)
    (cd $FT && SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install ionice -c3 nice -n 19 taskset -c 0-9 \
      /home/jixian/Desktop/miniconda3/bin/python eval_fast/run.py --map $R/real_${s}_once/final.4dmap.zpk --session $s --name m050_once \
      --mode full --out $R/eval/m050_once_$s "${prior[@]}") > $R/eval/m050_once_$s.log 2>&1
    step "real $s once: scored into eval/m050_once_$s"
  done
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
for s, accepted in (('a', 'm050_ours_a'), ('b', 'm050_resupport_b'), ('c', 'm050_resupport_c')):
    for name, d in (('accepted', accepted), ('read-once', f'm050_once_{s}')):
        print(f'{s}      {name:14s}', '  '.join(f'{x!s:>6}' for x in row(d)))
PY
}
synthetic() {
  local S=$FT/datasets/synthetic_ab
  local common=(--image-scale 1 --sensor-max-range 15 --change-detection-every-n-backend-updates 25 --mapper-config $WORK/configs/synthetic_5cm.yaml
    --input-config $S/mapping_configs/input.yaml --labelspace-config $S/mapping_configs/label_space.yaml --physical-catalog $S/mapping_configs/physical_catalog.json)
  map synthetic_a_once synthetic_a_once_mapping.log --run-dir $S/session_a/rgbd --semantic-dir $S/session_a/semantics \
    --instance-dir $S/session_a/instance_labels --session-start-ns 1000000000 "${common[@]}"
  map synthetic_b_once synthetic_b_once_mapping.log --run-dir $S/session_b/rgbd --semantic-dir $S/session_b/semantics \
    --instance-dir $S/session_b/instance_labels --session-start-ns 141000000000 --input-state $R/synthetic_a_once "${common[@]}"
  local maps=$(mktemp -d /dev/shm/synthetic-maps-XXXXXX)
  ln -s $R/synthetic_a_once/final.4dmap.zpk $maps/session_a.4dmap.zpk
  ln -s $R/synthetic_b_once/final.4dmap.zpk $maps/session_b.4dmap.zpk
  fresh $R/eval/once_synthetic
  SYN_MAPS=$maps SYN_OUT=$R/eval/once_synthetic SYN_REF=$R/eval/resupport_synthetic SYN_REF_LABEL=accepted \
    SYNTHETIC_TOOLS_BIN=$R/eval_tools_m050 ionice -c3 nice -n 19 taskset -c 0-9 $PY $R/analysis/run_synthetic_eval.py > $R/logs/once_synthetic_eval.log 2>&1
  rm -r $maps
  step "synthetic once: evaluated into eval/once_synthetic (FULL_TABLE.md)"
}
flat() {
  local D=$FT/datasets/flat_ab
  for s in a b; do
    local prior=(); [[ $s == b ]] && prior=(--input-state $R/flat_a_once)
    map flat_${s}_once flat_${s}_once_mapping.log --run-dir $D/session_$s/rgbd --semantic-dir $D/session_$s/semantics \
      --instance-dir $D/session_$s/instance_labels --session-start-ns $(python3 -c "import json; print(json.load(open('$D/session.json'))['$s']['start_ns'])") \
      --image-scale 1 --sensor-max-range 15 --change-detection-every-n-backend-updates 25 --mapper-config $R/flat/configs/mapper_flat_5cm.yaml \
      --input-config $FT/datasets/synthetic_ab/mapping_configs/input.yaml --labelspace-config $D/mapping_configs/label_space.yaml \
      --physical-catalog $R/flat/configs/physical_catalog.json "${prior[@]}"
    fresh $R/eval/g1final_flat_${s}_once
    step "Flat $s once: $(ionice -c3 nice -n 19 taskset -c 0-9 $PY $R/analysis/flat_eval.py $s $R/flat_${s}_once/final.4dmap.zpk $R/eval/g1final_flat_${s}_once | tail -1)"
  done
}
tesse() {
  for scene in office apartment; do
    local name data cfg eval flag=()
    case $scene in
      office) name=tesse_a_5m_once; data=$R/tesse/data/session_a; cfg=$R/tesse/configs; eval=$R/eval/tesse_paper_qj;;
      apartment) name=tesse_apt_5m_once; data=$R/tesse/data/session_apt; cfg=$R/tesse/configs/apartment; eval=$R/eval/tesse_apt_qj; flag=(--scene apartment);;
    esac
    map $name ${name}_mapping.log --run-dir $data/rgbd --semantic-dir $data/semantics --instance-dir $data/instance_labels \
      --session-start-ns $(python3 -c "import json; print(json.load(open('$data/session.json'))['start_ns'])") --image-scale 1 \
      --sensor-max-range 5 --change-detection-every-n-backend-updates 5 --mapper-config $cfg/mapper_tesse_5cm.yaml \
      --input-config $FT/datasets/synthetic_ab/mapping_configs/input.yaml --labelspace-config $cfg/label_space.yaml \
      --physical-catalog $cfg/physical_catalog.json
    local w=/dev/shm/once-chain4d-$scene out=$eval/ours_once_chain4d_5m_all_m050
    rm -rf $w; mkdir -p $w
    step "$scene once: $(bash -c "$ENV" chain-fill $FT/eval/chain_fill/chain_fill $w/chain.4dmap.zpk $R/$name/final.4dmap.zpk | tail -1)"
    fresh $out
    TESSE_EVAL_JOBS=4 ionice -c3 nice -n 19 taskset -c 0-9 $PY $R/analysis/tesse_online_eval.py "${flag[@]}" ours4d $w/chain.4dmap.zpk $out > $out.log 2>&1
    rm -rf $w
    step "TESSE $scene once 4D (0.5 m): $(grep -A1 '^Data' $out/TABLES_4D.txt | tail -1 | tr -s ' ')"
  done
}
for part in ${@:-real synthetic flat tesse}; do $part; done
step "read-once acceptance chain done"
