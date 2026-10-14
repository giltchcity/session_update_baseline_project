#!/usr/bin/env bash
# The E1 chain toward every line of the 10-14 goal, on ONE tree: worktrees/evrow = chain_clean 32938fa + SH (14f3855) + K4/K4b
# (a state change the layer decides whose evidence began after the latest keyframe enters as a keyframe) + E1 c098a0a (whether an
# earlier-session row is still there and how it changes is decided by this session's measurements of it, with the session's own
# fitted measurement model and in-place learned rates; no fixed boundary). Commands are those of the reviewed queues
# (chain_queue.sh run/tidy, post_chain_queue.sh scoring, views and Flat, after_decisive.sh refinement).
#   1. speed150 real B: evrow vs evstep_k4b, recorded; it is also the smoke: without E1_ROUND lines in the evrow run nothing goes on.
#      The time gate is the user's line on the full sessions: B <= 1532.6 s, C <= 4934.0 s
#   2. real B, tidy, G1 of B: below 92.98 the chain stops
#   3. synthetic B, tidy, refinement with the same tree, official scoring, the five view tools of FINAL_20261014 (visual items)
#   4. real C resumed from the B of step 2 (checkpoint.pt), tidy, real scoring a b c + acceptance
#   5. Flat native (GaME's official run), then Flat ours on this tree (mapping + refinement, post_ref view, render metrics)
# Stop between steps: touch runs_quick3k/final_chain.stop (checked before every step; a running step is never signalled).
set -u
X=$(cd "$(dirname "$0")" && pwd); RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; W=$RR/worktrees
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=$RR/code_mine_20261014/runs_quick3k; L=$Q/e1_chain.log
LOCK=/home/jixian/Desktop/FT/runs/GPU.lock; M=$RR/analysis/DISK_RECLAIM_20261013.txt; STOP=$Q/e1_chain.stop
EV=$RR/analysis/evaluator_snapshot_20261014; P=/home/jixian/Desktop/FT/session_update_baseline_project; A=$RR/analysis
F=$RR/FINAL_20261014; T=$W/evrow; BASE=$W/evstep_k4b
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
g1() { python3 -c "import json;d=json.load(open('$1'));print('%.2f %.2f %.2f'%(100*d['P_05cm'],100*d['R_05cm'],100*d['F1_05cm']))" 2>/dev/null; }
secs() { python3 -c "import json;print(json.load(open('$1'))['seconds'])" 2>/dev/null; }
for c in $T $BASE; do git -C $c status --porcelain | grep -q . && { echo "^refused dirty $c" >> $L; exit 3; }; done
K=$(git -C $T rev-parse --short HEAD)
# 1) speed150, recorded
halt speed150
for tree in $BASE $T; do kk=$(git -C $tree rev-parse --short HEAD); D=$Q/speed150_e1_$kk; mkdir -p $D; cd $D; git -C $tree rev-parse HEAD > code_commit.txt
  flock $LOCK $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -W ignore $X/quick_map.py $tree $D/map 150 real b
  echo "^speed150 $kk $(date '+%T') $(cat status.txt 2>/dev/null) seconds $(secs $D/map/session_b/run.json) K4 keyframes $(grep -c K4_KEYFRAME $D/run.log) E1 rounds $(grep -c E1_ROUND $D/run.log) (recorded, not a gate)" >> $L
  grep E1_ROUND $D/run.log | tail -2 >> $L
  for f in $(ls $D/map/checkpoint*.pt $D/map/session_*/timeline*.pkl 2>/dev/null); do [ -L "$f" ] || { echo "$(date '+%F %H:%M') speed150 (recorded; run.json kept): $(stat -c %s $f) $f" >> $M; rm -- "$f"; }; done
done
grep -q E1_ROUND $Q/speed150_e1_$(git -C $T rev-parse --short HEAD)/run.log || { echo "^E1 did not run in the smoke: chain stopped" >> $L; exit 9; }
# 2) real B
halt real_b
[ "$(used)" -lt 420 ] || { echo "^refused: disk $(used) GB before real B" >> $L; exit 4; }
RB=$RR/real_e1_$K/map
run $RR/real_e1_$K $PY -W ignore $X/quick_map.py $T $RB 0 real b
tidy $RB real b
pin_eval $RB
( cd $T && SES="b" UNDER_SIDE_GATE=1 timeout 2h flock $LOCK bash $RR/eval_row.sh $RB real ) > $RB/eval_b.out 2>&1
GB=$(g1 $RB/eval_real/geometry/b/G1_tsdf.json); SB=$(secs $RB/session_b/run.json)
echo "^real B G1 P/R/F1 $GB (line 92.98); mapping $SB s (line 1532.6 s = 1.3 x native); K4 keyframes $(grep -c K4_KEYFRAME $RR/real_e1_$K/run.log); E1 ended $(grep E1_ROUND $RR/real_e1_$K/run.log | tail -1 | grep -o "ended_session=[0-9]*")" >> $L
grep E1_ROUND $RR/real_e1_$K/run.log | tail -1 >> $L
python3 -c "import sys; sys.exit(0 if float('$(echo $GB | awk '{print $3}')' or 0) >= 92.98 else 1)" 2>/dev/null || { echo "^real B below the line: chain stopped $(date '+%T')" >> $L; exit 7; }
# 3) synthetic B, refinement, scoring, views
halt synthetic_chain
[ "$(used)" -lt 420 ] || { echo "^refused: disk $(used) GB before the synthetic chain" >> $L; exit 4; }
S=$RR/syn_e1_$K/map
run $RR/syn_e1_$K $PY -W ignore $X/quick_map.py $T $S 0 synthetic b
tidy $S synthetic b
echo "^K4 keyframes in synthetic B: $(grep -c K4_KEYFRAME $RR/syn_e1_$K/run.log); $(grep E1_ROUND $RR/syn_e1_$K/run.log | tail -1); mapping $(secs $S/session_b/run.json) s (native row 3 4496.8 s)" >> $L
halt synthetic_refine
ES=$RR/syn_e1_$K/refine_$K; mkdir -p $ES/session_b
for d in $S/session_*; do s=${d##*_}; [ "$s" = b ] && continue; ln -sfn $d $ES/session_$s
  [ -e $S/checkpoint_$s.pt ] && ln -sfn "$(readlink -f $S/checkpoint_$s.pt)" $ES/checkpoint_$s.pt; done
for f in $S/session_b/*; do b=$(basename $f); case $b in timeline_post_ref.pkl|tsdf_final.ply|run.json) continue;; esac; ln -sfn $f $ES/session_b/$b; done
cp -f $S/session_b/run.json $ES/session_b/run.json; git -C $T rev-parse HEAD > $ES/code_commit.txt
echo "^start refine synthetic b $K $(date '+%T') disk $(used) GB" >> $L
( cd $T && flock $LOCK $PY -m update_layer.eval.kf_strip restore $S/checkpoint_b.pt $ES/checkpoint_b.pt synthetic ) > $ES/restore_b.log 2>&1
# refine_only reads checkpoint_b.pt once at its start; once the loop has begun the restored copy (~22 GB) is deleted (disk peak)
( cd $T && flock $LOCK $PY -W ignore -m update_layer.eval.refine_only $ES synthetic b ) > $ES/refine.log 2>&1 &
RP=$!
until grep -q "REFINEMENT_MEM it=0" $ES/refine.log 2>/dev/null || ! kill -0 $RP 2>/dev/null; do sleep 5; done
if grep -q "REFINEMENT_MEM it=0" $ES/refine.log && [ -f $ES/checkpoint_b.pt ] && [ ! -L $ES/checkpoint_b.pt ]; then
  echo "$(date '+%F %H:%M') restored pre-refinement copy of $S/checkpoint_b.pt (source kept, stripped), loaded by refine_only and not read again: $(stat -c %s $ES/checkpoint_b.pt) $ES/checkpoint_b.pt" >> $M; rm -- $ES/checkpoint_b.pt; fi
wait $RP; rc=$?; echo "^done refine synthetic b $(date '+%T') rc=$rc $(python3 -c "import json;p=json.load(open('$ES/session_b/run.json')).get('post_ref') or {};print(p.get('completed_iterations'),'/',p.get('requested_iterations'),p.get('seconds'),'s peak',p.get('peak_gpu_gb'),'GB')" 2>/dev/null)" >> $L
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
                print(f'^synthetic B {lab}: P {100 * float(r["Precision"]):.2f} R {100 * float(r["Recall"]):.2f} F1 {100 * float(r["F1"]):.2f}  (lines post_ref F1 >= 96.43, P >= 96.6)')
        st = {r['kind']: (r['TP'], r['FP'], r['FN']) for r in csv.DictReader(open(d + '/ours/state_eval/group_scores/GROUP_STATE_SUMMARY.csv')) if r['session'] == 'ab' and r['kind'] in ('D1_visible', 'D2_hidden', 'D3')}
        print(f'^synthetic B {lab} D1/D2/D3 (ab) {st}  (unchanged = 4/0/2, 34/0/0, 29/0/3)')
    except Exception as e:
        print(f'^synthetic B {lab} missing ({e})')
PY
halt synthetic_views
for s in views15_final bathroom_struct_final compensated_end_final heldout_last_final last_kf_final; do
  ( cd $RR && FINAL_CKPT=$ES/checkpoint_b_post_ref.pt FINAL_CODE=worktrees/$(basename $T) FINAL_TAG=syn_e1_$K flock $LOCK $PY -W ignore $F/tools/$s.py ) > $ES/$s.out 2>&1
done
echo "^done synthetic views $(date '+%T')" >> $L
# 4) real C from the B of step 2, real scoring
halt real_c
[ "$(used)" -lt 425 ] || { echo "^refused: disk $(used) GB before real C" >> $L; exit 4; }
run $RR/real_e1_$K $PY -W ignore $X/quick_map.py $T $RB 0 real bc
tidy $RB real c
echo "^K4 keyframes in real B+C: $(grep -c K4_KEYFRAME $RR/real_e1_$K/run.log); real C mapping $(secs $RB/session_c/run.json) s (line 4934.0 s = 1.3 x native 3795.4 s)" >> $L
halt real_scoring
( cd $T && SES="a c" UNDER_SIDE_GATE=1 timeout 4h flock $LOCK bash $RR/eval_row.sh $RB real ) > $RB/eval_ac.out 2>&1
( cd $P && PYTHONPATH=. $PY -m update_layer.eval.acceptance real real_e1_$K/map --ref real_row4k --out $A/acceptance_real_e1_$K.json > $A/acceptance_real_e1_$K.txt 2>&1 )
echo "^real G1 P/R/F1  A $(g1 $RB/eval_real/geometry/a/G1_tsdf.json)  B $(g1 $RB/eval_real/geometry/b/G1_tsdf.json)  C $(g1 $RB/eval_real/geometry/c/G1_tsdf.json)  (lines 93.66 / 92.98 / 94.82)" >> $L
echo "^done real scoring $(date '+%T')" >> $L
# 5) Flat native, then Flat ours on this tree
halt flat_native
D=/home/jixian/Desktop/FT/runs/game_flat_official_20261007
echo "^start flat native $(date '+%T')" >> $L
( cd $D && [ -f run.log ] && mv -f run.log run_before_$(date +%H%M).log; rm -rf $D/out; mkdir -p $D/out; bash $D/run_flat.sh ) > $Q/flat_native_stdout.log 2>&1
echo "^done flat native $(date '+%T') $(grep -m1 '^exit' $D/run.log)" >> $L
halt flat_ours
FO=$RR/flat_e1_$K; mkdir -p $FO; git -C $T rev-parse HEAD > $FO/code_commit.txt
echo "^start flat ours $(date '+%T') disk $(used) GB" >> $L
( cd $T && flock $LOCK $PY -m update_layer.run --backend game --row 4 --d1 --g8-holdout --dataset flat --out $FO ) > $FO/chain.log 2>&1
echo "^done flat ours mapping $(date '+%T') rc=$?" >> $L
[ -s $FO/session_b/timeline_post_ref.pkl ] && bash $RR/post_ref_view.sh $FO b > $FO/post_ref_view.log 2>&1
for v in pre_ref post_ref; do src=$FO; [ $v = post_ref ] && src=$FO/post_ref
  [ -d $src ] && ( cd $T && flock $LOCK $PY -W ignore -m update_layer.eval.game_render_metrics $src b flat $FO/render_metrics_$v.json ) > $FO/render_metrics_$v.log 2>&1; done
echo "^done flat ours $(date '+%T')" >> $L
echo "^e1 chain end $(date '+%T')" >> $L
