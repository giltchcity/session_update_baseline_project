#!/usr/bin/env bash
# What follows the decisive experiment, by the rule written before its numbers (supervisor 15:30; user 16:2x: one 3DGS code that keeps
# both best results). Every step's commands are those of the reviewed queues (chain_queue.sh run/tidy, post_chain_queue.sh scoring,
# realb_rerun_queue.sh check); only the tree changes.
#   verdict PASS = post_ref F1 >= 96.43 and P >= 96.6 and D1/D2/D3 (ab) TP/FP/FN identical to every reference (4/0/2, 34/0/0, 29/0/3)
#   FAIL -> '^verdict FAIL', nothing runs (the two named versions go to the user with their numbers)
#   PASS ->
#     1. assertion check, real B 150 frames, branch check_lazyreset (f5ecf79 = 878d222 + assertions). If it holds (no CHECK error,
#        QUICK_MAP_DONE, checked_steps > 0), f3e5r9h_clean is fast-forwarded to 878d222 (moments of immune rows zeroed only when the
#        immune set is recomputed; bitwise the same). Otherwise f3e5r9h_clean stays at dfb98cd and the time line is reported as f3e5's.
#     2. real B -> C on f3e5r9h_clean from the accepted A (real_row4k), all frames, refinement off; tidy (0-step post_ref copies deleted,
#        kf_strip) as chain_queue.sh.
#     3. real scoring as post_chain_queue.sh: eval_row.sh a b c under the lock, acceptance.py against real_row4k.
#     4. synthetic B on f3e5r9h_clean from the accepted A (syn_row4s0), refinement with the same tree (refine_only), official scoring
#        pre_ref and post_ref.
# Stop between steps: touch runs_quick3k/after_decisive.stop (checked before every step; a running step is never signalled).
set -u
X=$(cd "$(dirname "$0")" && pwd); RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; W=$RR/worktrees
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=$RR/code_mine_20261014/runs_quick3k; L=$Q/after_decisive.log
LOCK=/home/jixian/Desktop/FT/runs/GPU.lock; M=$RR/analysis/DISK_RECLAIM_20261013.txt; STOP=$Q/after_decisive.stop
EV=$RR/analysis/evaluator_snapshot_20261014; P=/home/jixian/Desktop/FT/session_update_baseline_project; A=$RR/analysis
T=$W/f3e5r9h_clean; CHK=$W/check_lazyreset; E=$RR/syn_row4f3e5/refine_14f3855
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
used() { df --output=used -BG /home/jixian | tail -1 | tr -d ' G'; }
halt() { [ -f $STOP ] && { echo "^stopped by the stop file before $1 $(date '+%T')" >> $L; exit 0; }; return 0; }
pin_eval() { ( cd /home/jixian/Desktop/FT && xargs -a $EV/FILES.txt sha256sum ) > $1/evaluator_sha256.txt 2>&1
  if cmp -s $1/evaluator_sha256.txt $EV/SHA256SUMS.txt; then echo "^evaluator same as the snapshot for $1 $(date '+%T')" >> $L
  else echo "^evaluator DIFFERS from the snapshot for $1 $(date '+%T')" >> $L; fi; }
run() { local D=$1; shift; mkdir -p $D; cd $D; git -C $T rev-parse HEAD > code_commit.txt
  echo "^start $D $(date '+%T') disk $(used) GB" >> $L; local t0=$(date +%s)
  flock $LOCK $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$@"
  echo "^done $D $(date '+%T') seconds=$(( $(date +%s) - t0 )) $(cat status.txt 2>/dev/null) disk $(used) GB" >> $L; }
tidy() { local MAP=$1 DS=$2; shift 2
  for f in $(ls $MAP/checkpoint_*_post_ref.pt $MAP/session_*/timeline_post_ref.pkl 2>/dev/null); do [ -L "$f" ] && continue
    echo "$(date '+%F %H:%M') 0-step post_ref copy of a refinement-off mapping chain: $(stat -c %s $f) $f" >> $M; rm -- "$f"; done
  for s in "$@"; do [ -s $MAP/checkpoint_$s.pt ] && [ ! -L $MAP/checkpoint_$s.pt ] && ( cd $T && nice -n 5 flock $LOCK $PY -m update_layer.eval.kf_strip strip $MAP/checkpoint_$s.pt $MAP/checkpoint_$s.pt $DS ) > $MAP/strip_$s.log 2>&1; done
  echo "^tidied $MAP $(date '+%T') disk $(used) GB" >> $L; }
for c in $T $CHK; do git -C $c status --porcelain | grep -q . && { echo "^refused dirty $c" >> $L; exit 3; }; done
echo "^waiting for the decisive experiment $(date '+%T')" >> $L
until grep -q '^\^decisive experiment end' $Q/decisive_f3e5_refine.log 2>/dev/null; do sleep 30; done
V=$(python3 - $E/post_ref/eval_syn <<'PY'
import csv, sys
d = sys.argv[1]
g = None
for r in csv.DictReader(open(d + '/geometry/FINAL_GEOMETRY.csv')):
    if r['method'] == 'ours' and r['session'] == 'b' and r['scope'] == 'all' and r['threshold_m'] == '0.05':
        g = (100 * float(r['Precision']), 100 * float(r['Recall']), 100 * float(r['F1']))
st = {r['kind']: (int(r['TP']), int(r['FP']), int(r['FN'])) for r in csv.DictReader(open(d + '/ours/state_eval/group_scores/GROUP_STATE_SUMMARY.csv')) if r['session'] == 'ab'}
ref = {'D1_visible': (4, 0, 2), 'D2_hidden': (34, 0, 0), 'D3': (29, 0, 3)}
ok = g is not None and g[2] >= 96.43 and g[0] >= 96.6 and all(st.get(k) == v for k, v in ref.items())
print(('PASS' if ok else 'FAIL') + f' P {g[0]:.2f} R {g[1]:.2f} F1 {g[2]:.2f} D {[st.get(k) for k in ref]}')
PY
)
echo "^verdict $V $(date '+%T')" >> $L
case "$V" in PASS*) ;; *) exit 0;; esac
# 1) assertion check of the lazy reset, then fast-forward
halt check150
KC=$(git -C $CHK rev-parse --short HEAD); D1=$Q/check150_lazyreset_real_b_$KC; mkdir -p $D1; cd $D1; git -C $CHK rev-parse HEAD > code_commit.txt
echo "^start check150 $KC $(date '+%T')" >> $L
flock $LOCK $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -W ignore $X/quick_map.py $CHK $D1/map 150 real b
LAST=$(grep "CHECK_LAZYRESET" $D1/run.log | tail -1); echo "^done check150 $(date '+%T') $(cat status.txt 2>/dev/null) $LAST" >> $L
for f in $(ls $D1/map/checkpoint_*.pt $D1/map/session_*/timeline*.pkl 2>/dev/null); do [ -L "$f" ] || { echo "$(date '+%F %H:%M') check run (real B 150, lazy reset; run.log kept): $(stat -c %s $f) $f" >> $M; rm -- "$f"; }; done
if ! grep -q "RuntimeError: CHECK" $D1/run.log && grep -q "QUICK_MAP_DONE" $D1/run.log && echo "$LAST" | grep -q "'checked_steps': [1-9]"; then
  git -C $T merge -q --ff-only 878d222 && echo "^f3e5r9h_clean fast-forwarded to $(git -C $T rev-parse --short HEAD) $(date '+%T')" >> $L
else echo "^check did not hold: f3e5r9h_clean stays at $(git -C $T rev-parse --short HEAD) (time as f3e5's)" >> $L; fi
K=$(git -C $T rev-parse --short HEAD)
# 2) real B -> C
halt real_chain
[ "$(used)" -lt 420 ] || { echo "^refused: disk $(used) GB before the real chain" >> $L; exit 4; }
RL=$RR/real_chain_$K/map
run $RR/real_chain_$K $PY -W ignore $X/quick_map.py $T $RL 0 real bc
tidy $RL real b c
# 3) real scoring
halt real_scoring
echo "^start real scoring $(date '+%T')" >> $L
pin_eval $RL
( cd $T && SES="a b c" UNDER_SIDE_GATE=1 timeout 4h flock $LOCK bash $RR/eval_row.sh $RL real ) > $RL/eval_abc.out 2>&1
( cd $P && PYTHONPATH=. $PY -m update_layer.eval.acceptance real real_chain_$K/map --ref real_row4k --out $A/acceptance_real_chain_$K.json > $A/acceptance_real_chain_$K.txt 2>&1 )
python3 - $RL >> $L <<'PY'
import json, sys
for s in 'abc':
    try:
        d = json.load(open(f'{sys.argv[1]}/eval_real/geometry/{s}/G1_tsdf.json'))
        print(f'^real {s.upper()} G1 P {100 * d["P_05cm"]:.2f} R {100 * d["R_05cm"]:.2f} F1 {100 * d["F1_05cm"]:.2f}')
    except Exception as e:
        print(f'^real {s.upper()} G1 missing ({e})')
for s, n in (('b', 1178.9), ('c', 3795.4)):
    try:
        j = json.load(open(f'{sys.argv[1]}/session_{s}/run.json'))
        print(f'^real {s.upper()} mapping {j["seconds"]} s = {j["seconds"] / n:.3f} x native GaME row 3 ({n} s); peak {j.get("peak_gpu_gb")} GB')
    except Exception as e:
        print(f'^real {s.upper()} run.json missing ({e})')
PY
echo "^done real scoring $(date '+%T')" >> $L
# 4) synthetic B, refinement with the same tree, scoring
halt synthetic_chain
[ "$(used)" -lt 420 ] || { echo "^refused: disk $(used) GB before the synthetic chain" >> $L; exit 4; }
S=$RR/syn_chain_$K/map
run $RR/syn_chain_$K $PY -W ignore $X/quick_map.py $T $S 0 synthetic b
tidy $S synthetic b
halt synthetic_refine
ES=$RR/syn_chain_$K/refine_$K; mkdir -p $ES/session_b
for d in $S/session_*; do s=${d##*_}; [ "$s" = b ] && continue; ln -sfn $d $ES/session_$s
  [ -e $S/checkpoint_$s.pt ] && ln -sfn "$(readlink -f $S/checkpoint_$s.pt)" $ES/checkpoint_$s.pt; done
for f in $S/session_b/*; do b=$(basename $f); case $b in timeline_post_ref.pkl|tsdf_final.ply|run.json) continue;; esac; ln -sfn $f $ES/session_b/$b; done
cp -f $S/session_b/run.json $ES/session_b/run.json; git -C $T rev-parse HEAD > $ES/code_commit.txt
echo "^start refine synthetic b $K $(date '+%T') disk $(used) GB" >> $L
( cd $T && flock $LOCK $PY -m update_layer.eval.kf_strip restore $S/checkpoint_b.pt $ES/checkpoint_b.pt synthetic ) > $ES/restore_b.log 2>&1
( cd $T && flock $LOCK $PY -W ignore -m update_layer.eval.refine_only $ES synthetic b ) > $ES/refine.log 2>&1
echo "^done refine synthetic b $(date '+%T') rc=$?" >> $L
if [ -s $ES/checkpoint_b_post_ref.pt ] && [ -f $ES/checkpoint_b.pt ] && [ ! -L $ES/checkpoint_b.pt ]; then
  echo "$(date '+%F %H:%M') restored pre-refinement copy of $S/checkpoint_b.pt (source kept, stripped), not read after the refinement: $(stat -c %s $ES/checkpoint_b.pt) $ES/checkpoint_b.pt" >> $M; rm -- $ES/checkpoint_b.pt; fi
bash $RR/post_ref_view.sh $ES b > $ES/post_ref_view.log 2>&1
halt synthetic_scoring
for d in $S $ES/post_ref; do git -C $T rev-parse HEAD > $d/eval_syn_code_commit.txt; pin_eval $d; done
( cd $T && flock $LOCK bash update_layer/eval/eval_synthetic_scene.sh $S $S/eval_syn ) > $S/eval_b.log 2>&1
( cd $T && flock $LOCK bash update_layer/eval/eval_synthetic_scene.sh $ES/post_ref $ES/post_ref/eval_syn ) > $ES/eval_post_b.log 2>&1
python3 - $S/eval_syn $ES/post_ref/eval_syn >> $L <<'PY'
import csv, sys
for lab, d in (('pre_ref', sys.argv[1]), ('post_ref', sys.argv[2])):
    try:
        for r in csv.DictReader(open(d + '/geometry/FINAL_GEOMETRY.csv')):
            if r['method'] == 'ours' and r['session'] == 'b' and r['scope'] == 'all' and r['threshold_m'] == '0.05':
                print(f'^synthetic B {lab}: P {100 * float(r["Precision"]):.2f} R {100 * float(r["Recall"]):.2f} F1 {100 * float(r["F1"]):.2f}')
        st = {r['kind']: (r['TP'], r['FP'], r['FN']) for r in csv.DictReader(open(d + '/ours/state_eval/group_scores/GROUP_STATE_SUMMARY.csv')) if r['session'] == 'ab'}
        print(f'^synthetic B {lab} states (ab): {st}')
    except Exception as e:
        print(f'^synthetic B {lab} missing ({e})')
PY
echo "^after decisive end $(date '+%T')" >> $L
