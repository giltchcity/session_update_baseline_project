#!/usr/bin/env bash
# [R11e + K4b] final chain. Tree worktrees/evstep_k4b = chain_clean 32938fa + SH (14f3855) + R11e 98e930b + K4 85eca92 + K4b 13050eb.
# The 150-frame speed pair is run and recorded, not used as the gate: K4b adds a keyframe exactly where real B starts with a parked
# camera and a change (round 4.2 s), which is all a 150-frame window holds. The gate is the user's time line on the full real B:
# <= 1.3 x native (1532.6 s).
#   1. speed150 real B: evstep_k4b vs evstep (recorded); gate = real B mapping seconds <= 1532.6
#   2. real B (all frames from the accepted A, refinement off), tidy, G1 of B: the real gate is B >= 92.98 (the user's line); below it the
#      chain stops and the number is reported
#   3. synthetic B, tidy, refinement with the same tree, official scoring pre_ref / post_ref
#   4. real B -> C (all frames), tidy, real scoring a b c + acceptance
# Stop between steps: touch runs_quick3k/evstep_chain.stop.
set -u
X=$(cd "$(dirname "$0")" && pwd); RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; W=$RR/worktrees
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=$RR/code_mine_20261014/runs_quick3k; L=$Q/evstep_k4b_chain.log
LOCK=/home/jixian/Desktop/FT/runs/GPU.lock; M=$RR/analysis/DISK_RECLAIM_20261013.txt; STOP=$Q/evstep_k4b_chain.stop
EV=$RR/analysis/evaluator_snapshot_20261014; P=/home/jixian/Desktop/FT/session_update_baseline_project; A=$RR/analysis
T=$W/evstep_k4b; BASE=$W/evstep
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
# R11e alone on real B first (evstep_chain.sh): the final chain starts only if it holds the real line
until grep -q '^\^real B G1\|^\^real B below\|^\^refused' $Q/evstep_chain.log 2>/dev/null; do sleep 30; done
grep -q '^\^real B below\|^\^refused' $Q/evstep_chain.log && { echo "^R11e alone did not hold real B: final chain not started $(date '+%T')" >> $L; exit 8; }
echo "^R11e alone on real B: $(grep '^\^real B G1' $Q/evstep_chain.log | tail -1 | cut -c1-120); final chain starts $(date '+%T')" >> $L
for c in $T $BASE; do git -C $c status --porcelain | grep -q . && { echo "^refused dirty $c" >> $L; exit 3; }; done
# 1) speed150 pair
halt speed150
for tree in $BASE $T; do kk=$(git -C $tree rev-parse --short HEAD); D=$Q/speed150_k4b_$kk; mkdir -p $D; cd $D; git -C $tree rev-parse HEAD > code_commit.txt
  echo "^start speed150 $kk $(date '+%T')" >> $L
  flock $LOCK $X/run_logged.sh . env OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -W ignore $X/quick_map.py $tree $D/map 150 real b
  echo "^done speed150 $kk $(date '+%T') $(cat status.txt 2>/dev/null) seconds $(python3 -c "import json;print(json.load(open('$D/map/session_b/run.json'))['seconds'])" 2>/dev/null) " >> $L
  for f in $(ls $D/map/checkpoint_*.pt $D/map/session_*/timeline*.pkl 2>/dev/null); do [ -L "$f" ] || { echo "$(date '+%F %H:%M') speed150 (R11e gate; run.json kept): $(stat -c %s $f) $f" >> $M; rm -- "$f"; }; done
done
SB=$(python3 -c "import json;print(json.load(open('$Q/speed150_k4b_$(git -C $BASE rev-parse --short HEAD)/map/session_b/run.json'))['seconds'])" 2>/dev/null)
SK=$(python3 -c "import json;print(json.load(open('$Q/speed150_k4b_$(git -C $T rev-parse --short HEAD)/map/session_b/run.json'))['seconds'])" 2>/dev/null)
echo "^speed150 recorded (not a gate): base $SB s, R11e+K4b $SK s" >> $L
K=$(git -C $T rev-parse --short HEAD)
K=$(git -C $T rev-parse --short HEAD)
g1() { python3 -c "import json;d=json.load(open('$1'));print('%.2f %.2f %.2f'%(100*d['P_05cm'],100*d['R_05cm'],100*d['F1_05cm']))" 2>/dev/null; }
# 2) real B and its G1
halt real_b
[ "$(used)" -lt 420 ] || { echo "^refused: disk $(used) GB before real B" >> $L; exit 4; }
RB=$RR/real_b_k4b_$K/map
run $RR/real_b_k4b_$K $PY -W ignore $X/quick_map.py $T $RB 0 real b
tidy $RB real b
pin_eval $RB
( cd $T && SES="b" UNDER_SIDE_GATE=1 timeout 2h flock $LOCK bash $RR/eval_row.sh $RB real ) > $RB/eval_b.out 2>&1
GB=$(g1 $RB/eval_real/geometry/b/G1_tsdf.json)
echo "^real B G1 P/R/F1 $GB (line 92.98; f3e5 92.32/93.64/92.98, chain 91.16/92.85/92.00); B mapping $(python3 -c "import json;j=json.load(open('$RB/session_b/run.json'));print(j['seconds'],'s =',round(j['seconds']/1178.9,3),'x native, peak',j.get('peak_gpu_gb'),'GB')")" >> $L
echo "^K4 keyframes in real B: $(grep -c K4_KEYFRAME $RR/real_b_k4b_$K/run.log)" >> $L
SEC=$(python3 -c "import json;print(json.load(open('$RB/session_b/run.json'))['seconds'])")
python3 -c "import sys; sys.exit(0 if $SEC <= 1532.6 else 1)" || echo "^real B time $SEC s above the 1.3 x line (1532.6 s): reported, the chain goes on for the quality lines" >> $L
F=$(echo $GB | awk '{print $3}')
python3 -c "import sys; sys.exit(0 if float('${F:-0}') >= 92.98 else 1)" || { echo "^real B below the line: chain stopped" >> $L; exit 7; }
# 3) synthetic B, refinement with the same tree, scoring
halt synthetic_chain
[ "$(used)" -lt 420 ] || { echo "^refused: disk $(used) GB before the synthetic chain" >> $L; exit 4; }
S=$RR/syn_chain_$K/map
run $RR/syn_chain_$K $PY -W ignore $X/quick_map.py $T $S 0 synthetic b
tidy $S synthetic b
echo "^K4 keyframes in the synthetic chain: $(grep -c K4_KEYFRAME $RR/syn_chain_$K/run.log)" >> $L
halt synthetic_refine
ES=$RR/syn_chain_$K/refine_$K; mkdir -p $ES/session_b
for d in $S/session_*; do s=${d##*_}; [ "$s" = b ] && continue; ln -sfn $d $ES/session_$s
  [ -e $S/checkpoint_$s.pt ] && ln -sfn "$(readlink -f $S/checkpoint_$s.pt)" $ES/checkpoint_$s.pt; done
for f in $S/session_b/*; do b=$(basename $f); case $b in timeline_post_ref.pkl|tsdf_final.ply|run.json) continue;; esac; ln -sfn $f $ES/session_b/$b; done
cp -f $S/session_b/run.json $ES/session_b/run.json; git -C $T rev-parse HEAD > $ES/code_commit.txt
echo "^start refine synthetic b $K $(date '+%T') disk $(used) GB" >> $L
( cd $T && flock $LOCK $PY -m update_layer.eval.kf_strip restore $S/checkpoint_b.pt $ES/checkpoint_b.pt synthetic ) > $ES/restore_b.log 2>&1
# refine_only reads checkpoint_b.pt once at its start; once the loop has begun (REFINEMENT_MEM it=0) the restored copy (~22 GB) is
# deleted, as in decisive_f3e5_refine.sh, so the disk peak stays one copy lower
( cd $T && flock $LOCK $PY -W ignore -m update_layer.eval.refine_only $ES synthetic b ) > $ES/refine.log 2>&1 &
RP=$!
until grep -q "REFINEMENT_MEM it=0" $ES/refine.log 2>/dev/null || ! kill -0 $RP 2>/dev/null; do sleep 5; done
if grep -q "REFINEMENT_MEM it=0" $ES/refine.log && [ -f $ES/checkpoint_b.pt ] && [ ! -L $ES/checkpoint_b.pt ]; then
  echo "$(date '+%F %H:%M') restored pre-refinement copy of $S/checkpoint_b.pt (source kept, stripped), loaded by refine_only and not read again: $(stat -c %s $ES/checkpoint_b.pt) $ES/checkpoint_b.pt" >> $M; rm -- $ES/checkpoint_b.pt; fi
wait $RP; rc=$?; echo "^done refine synthetic b $(date '+%T') rc=$rc" >> $L
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
# 4) real B -> C
halt real_chain
[ "$(used)" -lt 420 ] || { echo "^refused: disk $(used) GB before the real chain" >> $L; exit 4; }
RL=$RR/real_chain_$K/map
run $RR/real_chain_$K $PY -W ignore $X/quick_map.py $T $RL 0 real bc
tidy $RL real b c
echo "^K4 keyframes in the real chain: $(grep -c K4_KEYFRAME $RR/real_chain_$K/run.log)" >> $L
# 4b) real scoring
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
echo "^evstep_k4b chain end $(date '+%T')" >> $L
