#!/bin/bash
# Aria on the TSDF side (11-02, the GaME session's request): our core (wt_tesse_fix, cc262a3) on room ROOM, visit A from an
# empty map, then B loading A's map, with the object ids of the GaME session's SAM front (e4a02cf, exported every frame:
# datasets/aria_ab/ROOM/front_e4a02cf_labels) as the physical instance input, one object class (configs/aria_label_space.yaml;
# semantics: 1 where the front gave an id, 0 elsewhere), configs/aria_instance_5cm.yaml (room18_instance_5cm.yaml without
# the class moveability prior), the clips' depth range 0.1-10 m as the points backend. Into aria/tsdf/ROOM/session_{a,b}.
# Extra overrides for an ablation: MAPPER_CONFIG=..., TAG=... (into aria/tsdf/ROOM_TAG).
# usage: [MAPPER_CONFIG=... TAG=...] run_tsdf.sh ROOM
set -eo pipefail
ROOM=$1; FT=/home/jixian/Desktop/FT; A=$FT/wt_takeover_runs/aria; D=$FT/datasets/aria_ab/$ROOM; F=$D/front_e4a02cf_labels
WORK=$FT/wt_tesse_fix/session_update_baseline; BUILD=$FT/wt_tesse_fix_build
CANON=(SESSION_UPDATE_CANONICAL_ROOT=$BUILD/canonical SESSION_UPDATE_CANONICAL_PREFIX=$BUILD/canonical/install SESSION_UPDATE_CANONICAL_BUILD=$BUILD/baseline)
CFG=${MAPPER_CONFIG:-$A/configs/aria_instance_5cm.yaml}; OUT=$A/tsdf/$ROOM${TAG:+_$TAG}; IN=$A/inputs/$ROOM
step() { echo "$(date +%H:%M) $*"; }
# The frames the front gave ids: all of A; B without its held-out frames (every 10th but the first, --g8-holdout: the
# points backend does not map them either). Each visit is played from IN/session_S/rgbd: those frames' colour, depth and
# pose linked, the camera file linked, timestamps.csv cut to them.
if [[ ! -s $IN/catalog.json ]]; then
  /home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python - $F $IN $D <<'EOF'
import csv, json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image
F, IN, D = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); ids = set()
for s in 'ab':
    out = IN / f'session_{s}/semantics'; out.mkdir(parents=True, exist_ok=True)
    rgbd, src = IN / f'session_{s}/rgbd', D / f'session_{s}/rgbd'; rgbd.mkdir(parents=True, exist_ok=True)
    have = sorted(p.name[:6] for p in (F / f'session_{s}/instance_labels').glob('*_segmentation.png'))
    for fid in have:
        for suffix in ('_color.png', '_depth.tiff', '_pose.txt'):
            os.symlink(src / f'{fid}{suffix}', rgbd / f'{fid}{suffix}')
    os.symlink(src / 'Intrinsics.txt', rgbd / 'Intrinsics.txt')
    rows = [r for r in csv.DictReader(open(src / 'timestamps.csv')) if r['ImageID'] in set(have)]
    with open(rgbd / 'timestamps.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['ImageID', 'TimeStamp']); w.writeheader(); w.writerows(rows)
    print(f'session {s}: {len(have)} of {len(list(src.glob("*_color.png")))} frames with front ids')
    for p in sorted((F / f'session_{s}/instance_labels').glob('*_segmentation.png')):
        inst = np.asarray(Image.open(p)); ids.update(np.unique(inst).tolist())
        Image.fromarray((inst > 0).astype(np.uint8)).save(out / p.name)
ids.discard(0)
cat = dict(schema='session_update_physical_catalog/v1', policy='A nonzero physical-instance pixel receives the catalog semantic '
           'class (1, object); the ids are the GaME session front e4a02cf tracks + 1', semantic_id_range=[0, 1],
           entities=[dict(physical_instance_id=int(i), semantic_id=1, name=f'front_{int(i)}') for i in sorted(ids)])
(IN / 'catalog.json').write_text(json.dumps(cat, indent=1)); print(f'catalog: {len(ids)} ids')
EOF
fi
for s in a b; do
  out=$OUT/session_$s; start=$(python3 -c "import json; print(json.load(open('$D/session.json'))['$s']['start_ns'])")
  prior=(); [[ $s == b ]] && prior=(--input-state $OUT/session_a)
  [[ -s $out/final.4dmap.zpk ]] && { step "$ROOM $s: map there"; continue; }
  [[ -e $out ]] && mv $out $out.before.$(date +%m%d%H%M%S)
  step "$ROOM $s: mapping"
  (cd $WORK && export ROS_DOMAIN_ID=76 && env "${CANON[@]}" taskset -c 10-15 nice -n 10 bash scripts/run_session.sh --output-state $out \
    --session-id aria_${ROOM}_${s}${TAG:+_$TAG} --run-dir $IN/session_$s/rgbd --semantic-dir $IN/session_$s/semantics \
    --instance-dir $F/session_$s/instance_labels --session-start-ns $start --tf-settle-s 0.02 --play-rate 100 --image-scale 1 \
    --finalization-timeout-s 86400 --sensor-max-range 10.0 --ack-timeout-s 86400 --discovery-timeout-s 86400 \
    --change-detection-every-n-backend-updates 5 --mapper-config $CFG \
    --input-config $FT/datasets/local_ab/mapping_configs/nss_flat_input.yaml --labelspace-config $A/configs/aria_label_space.yaml \
    --physical-catalog $IN/catalog.json "${prior[@]}") > $A/logs/tsdf_${ROOM}${TAG:+_$TAG}_$s.log 2>&1 || true
  [[ -s $out/final.4dmap.zpk ]] || { step "$ROOM $s: no map (see logs/tsdf_${ROOM}${TAG:+_$TAG}_$s.log)"; exit 3; }
  step "$ROOM $s: mapped, $(du -h $out/final.4dmap.zpk | cut -f1)"
done
step "$ROOM TSDF DONE"
