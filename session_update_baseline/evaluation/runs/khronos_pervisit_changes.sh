#!/usr/bin/env bash
# The detail page's row 1 (Khronos, each real visit from an empty map: runs/official_khronos_63faadde_{A,B,C}_full_20260909)
# had no change scoring for B and C; here eval_fast full mode (0.5 m, as the table) on those maps, B then C, into
# eval/khronos_pv_full_{b,c} (the prior is the previous visit's per-visit map: eval_fast needs one to label D3; the
# objects, D2 and G1 never read it). Evaluation only.
FT=/home/jixian/Desktop/FT; R=$FT/wt_takeover_runs; K=$FT/runs/official_khronos_63faadde
for s in b c; do
  case $s in b) P=A; M=B;; c) P=B; M=C;; esac
  rm -rf $R/eval/khronos_pv_full_$s
  (cd $FT && SESSION_UPDATE_CANONICAL_PREFIX=$FT/wt_takeover_build/canonical/install ionice -c3 nice -n 19 \
    /home/jixian/Desktop/miniconda3/bin/python eval_fast/run.py --map ${K}_${M}_full_20260909/state/final.4dmap.zpk --session $s \
    --name khronos_pv --mode full --out $R/eval/khronos_pv_full_$s --prior ${K}_${P}_full_20260909/state/final.4dmap.zpk) \
    > $R/eval/khronos_pv_full_$s.log 2>&1 && echo "$(date +%H:%M) Khronos per-visit $s: scored" || echo "$(date +%H:%M) Khronos per-visit $s: scoring failed"
done
