#!/usr/bin/env bash
# TODO 6 on the GPU after the K2 queue ends: 4K refinement speed test r21 (r21c 4ca6976) vs r21 + SH ramp (r21c 14f3855) on the
# same R11p B; gate = time per step over steps 3000-4000 (SH at degree 3 in the SH arm; tqdm's elapsed times) <= 1.10 x r21; then the 30K refinement of the same pre-refinement map with
# r21 + SH and the same scoring as RUN_ALL's control stage (post_ref view, official synthetic evaluation, views15/end/last-kf tools).
set -u
X=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$X"); R=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; F=$R/FINAL_20261014
W=$R/worktrees; PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=$REPO/runs_quick3k; L=$Q/sh_queue.log; LOCK=/home/jixian/Desktop/FT/runs/GPU.lock
until grep -q "^\^queue end" $Q/k2_trial_queue.log 2>/dev/null; do sleep 60; done
SH=$(git -C $W/r21c rev-parse --short HEAD); git -C $W/r21c status --porcelain | grep -q . && { echo "^refused dirty r21c" >> $L; exit 3; }
rm -rf $W/r21c_base; git -C $W/r21c worktree add -q --detach $W/r21c_base 4ca6976
for arm in r21c_base:$W/r21c_base sh:$W/r21c; do n=${arm%%:*}; c=${arm#*:}; D=$Q/speed3k_${n}_$(git -C $c rev-parse --short HEAD); mkdir -p $D
  echo "^start speed3k $n $(date '+%T')" >> $L
  flock $LOCK env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -W ignore $X/quick_screen.py short4k $n synthetic $D $c > $D/run.log 2>&1
  echo "^done speed3k $n $(date '+%T')" >> $L; done
step34() { python3 - "$1/run.log" <<'PY'
import re, sys
txt = open(sys.argv[1], errors='ignore').read().replace('\r', '\n')
el = {}
for m in re.finditer(r'(\d+)/4000 \[(\d+):(\d+)(?::(\d+))?<', txt):
    n = int(m.group(1)); a, b, c = m.group(2), m.group(3), m.group(4)
    el[n] = (int(a) * 3600 + int(b) * 60 + int(c)) if c else (int(a) * 60 + int(b))
k3 = min((n for n in el if n >= 3000), default=None); k4 = max(el, default=None)
print((el[k4] - el[k3]) / max(k4 - k3, 1) if k3 is not None and k4 and k4 > k3 else 0)
PY
}
tb=$(step34 $Q/speed3k_r21c_base_4ca6976); ts=$(step34 $Q/speed3k_sh_$SH)
deg=$(python3 -c "import json,glob,sys; print(json.load(open(glob.glob(sys.argv[1]+'/short4k_*.json')[0])).get('active_sh_degree_at_end'))" $Q/speed3k_sh_$SH 2>/dev/null)
echo "^speed per step 3000-4000: sh=$ts r21=$tb; SH degree at the end of the SH arm: $deg" >> $L
if [ "$deg" != "3" ] || ! python3 -c "import sys; tb, ts = float('$tb'), float('$ts'); sys.exit(0 if tb > 0 and 0 < ts <= 1.10 * tb else 1)"; then
  echo "^gate speed FAIL sh=$ts r21=$tb: 30K not started" >> $L; echo "^queue end $(date '+%T')" >> $L; exit 0; fi
echo "^gate speed PASS sh=$ts r21=$tb" >> $L
S=$R/syn_row4f3e5_r11p; E=$R/exp_sh_r11p_$SH; mkdir -p $E/session_b; git -C $W/r21c rev-parse HEAD > $E/code_commit.txt
ln -sfn "$S/session_a" "$E/session_a"; ln -sfn "$(readlink -f "$S/checkpoint_a.pt")" "$E/checkpoint_a.pt"
for f in "$S/session_b/"*; do b=$(basename "$f"); case $b in timeline_post_ref.pkl|tsdf_final.ply|run.json) continue;; esac; ln -sfn "$f" "$E/session_b/$b"; done
cp -f "$S/session_b/run.json" "$E/session_b/run.json"
echo "^start sh30k $(date '+%T')" >> $L
( cd $W/r21c && flock $LOCK $PY -m update_layer.eval.kf_strip restore "$S/checkpoint_b.pt" "$E/checkpoint_b.pt" synthetic ) > $E/restore_b.log 2>&1
( cd $W/r21c && flock $LOCK env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -W ignore -m update_layer.eval.refine_only "$E" synthetic b ) > $E/refine.log 2>&1
echo "^done sh30k refine $(date '+%T') rc=$?" >> $L
bash "$R/post_ref_view.sh" "$E" b > "$E/post_ref_view.log" 2>&1
( cd $W/r21c && timeout 4h flock $LOCK bash update_layer/eval/eval_synthetic_scene.sh "$E/post_ref" "$E/post_ref/eval_syn" ) > "$E/eval_post_b.log" 2>&1
for s in views15_final compensated_end_final last_kf_final; do
  ( cd "$R" && FINAL_CKPT="$E/checkpoint_b_post_ref.pt" FINAL_CODE="worktrees/r21c" FINAL_TAG=syn_sh flock $LOCK $PY -W ignore "$F/tools/$s.py" ) > "$E/$s.out" 2>&1; done
rm -f "$E/checkpoint_b.pt"
echo "^done sh30k all $(date '+%T')" >> $L; echo "^queue end $(date '+%T')" >> $L
