#!/usr/bin/env bash
# Real B -> C on the object-memory re-support branch (wt_resupport, object-memory-resupport on 4898f5b; built in
# wt_resupport_build): the commands of our real B (analysis/tk_real_b_run.json, prior real_a_tk) and real C
# (analysis/real_c_4898f5b.sh, prior: this B) into real_{b,c}_resupport, each then scored by eval_fast (0.5 m object
# matching) into eval/m050_resupport_{b,c}; then the comparison with eval/m050_ours_{b,c}. Nothing in here stops the run.
# usage: real_bc_resupport.sh [b] [c]   (default both, in order)
set -eo pipefail
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; L=$FT/datasets/local_ab
WORK=$FT/wt_resupport/session_update_baseline; BUILD=$FT/wt_resupport_build
step() { echo "$(date +%H:%M) $*"; }
session() {  # session b|c PRIOR_STATE RUN_DIR START_NS
  local s=$1 out=$R/real_${1}_resupport
  [[ -e $out ]] && mv $out $out.before.$(date +%m%d%H%M%S)
  step "real $s re-support: mapping"
  (cd $WORK && SESSION_UPDATE_CANONICAL_ROOT=$BUILD/canonical SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install \
    SESSION_UPDATE_CANONICAL_BUILD=$BUILD/baseline taskset -c 0-9 nice -n 10 bash scripts/run_session.sh --output-state $out \
    --session-id real_${s}_resupport --run-dir $L/rgbd/$3 --semantic-dir $L/semantics/session_$s \
    --instance-dir $L/instance_labels/session_$s --session-start-ns $4 --tf-settle-s 0.02 --play-rate 100 --image-scale 0.5 \
    --finalization-timeout-s 86400 --sensor-max-range 5.0 --ack-timeout-s 86400 --discovery-timeout-s 86400 \
    --change-detection-every-n-backend-updates 5 --mapper-config $WORK/configs/room18_instance_5cm.yaml \
    --input-config $L/mapping_configs/nss_flat_input.yaml --labelspace-config $L/mapping_configs/nss_ade20k_room_label_space.yaml \
    --physical-catalog $L/mapping_configs/room18_physical_catalog.json --world-transform $L/alignment/session_${s}_to_session_a.txt \
    --input-state $2) > $R/logs/real_${s}_resupport_mapping.log 2>&1
  step "real $s re-support: mapped, $(du -h $out/final.4dmap.zpk | cut -f1)"
  (cd $FT && SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install ionice -c3 nice -n 19 taskset -c 0-9 \
    /home/jixian/Desktop/miniconda3/bin/python eval_fast/run.py --map $out/final.4dmap.zpk --session $s --name m050_resupport \
    --mode full --out $R/eval/m050_resupport_$s --prior $2/final.4dmap.zpk) > $R/eval/m050_resupport_$s.log 2>&1
  step "real $s re-support: scored into eval/m050_resupport_$s"
}
SESSIONS=${@:-b c}
for s in $SESSIONS; do [[ -e $R/eval/m050_resupport_$s ]] && mv $R/eval/m050_resupport_$s $R/eval/m050_resupport_$s.before.$(date +%m%d%H%M%S); done
for s in $SESSIONS; do
  case $s in
    b) session b $R/real_a_tk session_b_20260810_030502_620_flat_rgbd_30hz_1080p 1786302302620000000;;
    c) session c $R/real_b_resupport session_c_20260822_182703_387_flat_rgbd_30hz_1080p 1787394423387000000;;
  esac
done
python3 - <<'EOF'
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
for s in 'bc':
    for name, d in (('4898f5b', f'm050_ours_{s}'), ('re-support', f'm050_resupport_{s}')):
        print(f'{s}      {name:14s}', '  '.join(f'{x!s:>6}' for x in row(d)))
EOF
step "real B -> C re-support: done"
