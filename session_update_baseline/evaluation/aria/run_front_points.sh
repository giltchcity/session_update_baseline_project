#!/bin/bash
# Aria on the TSDF side (11-02, the GaME session's request): the GaME session's SAM object front (e4a02cf, a copy with every
# frame's tracked ids written as instance images, FRONT_IDS_OUT) on its points backend (row 4, D1, no hold-out: every frame
# gets ids), room ROOM A -> B; the ids then feed our TSDF core (aria/run_tsdf.sh). CPU only.
# usage: run_front_points.sh ROOM
set -eo pipefail
ROOM=$1; FT=/home/jixian/Desktop/FT; A=$FT/wt_takeover_runs/aria; C=$A/front_e4a02cf; OUT=$A/points_e4a02cf/$ROOM
mkdir -p $OUT
cd $C && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=$C FRONT_IDS_OUT=$A/front_ids/$ROOM \
  nice -n 15 $FT/envs/gs-cu128/bin/python -W ignore -m update_layer.run --backend points --row 4 --d1 \
  --labels $FT/datasets/aria_ab/$ROOM/sam_labels --dataset aria/$ROOM --out $OUT > $A/logs/front_points_$ROOM.log 2>&1
echo "$(date +%H:%M) front + points $ROOM done: $(ls $A/front_ids/$ROOM/session_a | wc -l) / $(ls $A/front_ids/$ROOM/session_b | wc -l) id images"
