#!/usr/bin/env bash
# One ordered GPU sequence after the real scoring. It follows the supervisor's 15:18 order and replaces the separate queues that raced
# for GPU.lock. The step commands are those of the reviewed queues: realb_rerun_queue.sh (harness_b696e0e), equiv_queue.sh
# (harness_1155a0e) and the two diagnostic queues. The Flat smoke (ours) is already mapped (flat_smoke_queue.log 15:23:26); its render
# metrics wait on the lock in that queue.
#   phase 1, one hold of GPU.lock, requested at launch:
#            equivalence synthetic B 150 frames on chain_clean (the 15:23 run was killed; it is kept as *_killed1525)
#            -> assertion check real B 150 frames on the check branch.
#            The real scoring's last GPU step holds the lock now. The post-chain queue asks for the lock again only for Flat ours,
#            after its acceptance step, so this hold comes before Flat ours.
#   phase 2, after '^post chain queue end' (Flat ours and Flat native done):
#            real B rerun with chain_clean (only if the check passed) -> equivalence real B->C 30 frames -> equivalence 4K refinement
#            -> rule-table diagnostic -> release-set diagnostic, each under GPU.lock.
# If 20:00 comes first, phase 2 runs late; the Flat runs are never pushed back by it.
# Stop between steps: touch runs_quick3k/after_real_sequence.stop. The file is checked before every step; a running step is never
# signalled, because signalling the queue lets the task harness kill its running job.
set -u
X=$(cd "$(dirname "$0")" && pwd); RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; W=$RR/worktrees
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=$RR/code_mine_20261014/runs_quick3k; L=$Q/after_real_sequence.log
LOCK=/home/jixian/Desktop/FT/runs/GPU.lock; M=$RR/analysis/DISK_RECLAIM_20261013.txt; STOP=$Q/after_real_sequence.stop
CC=$W/chain_clean; CHK=$W/check_7f12297
for c in $CC $CHK; do git -C $c status --porcelain | grep -q . && { echo "^refused dirty $c" >> $L; exit 3; }; done
K=$(git -C $CC rev-parse --short HEAD); KC=$(git -C $CHK rev-parse --short HEAD)
ENVV="OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
halt() { [ -f $STOP ] && { echo "^stopped by the stop file before $1 $(date '+%T')" >> $L; exit 0; }; return 0; }
clean() { for f in $(ls $1/checkpoint_*.pt $1/session_*/timeline*.pkl 2>/dev/null); do
  [ -L "$f" ] || { echo "$(date '+%F %H:%M') $2: $(stat -c %s $f) $f" >> $M; rm -- "$f"; }; done; }
# one equivalence step as equiv_queue.sh's run(): its own run directory, the tree's commit, markers in equiv_queue.log
equiv() { local name=$1; shift; local D=$Q/$name; mkdir -p $D; cd $D; git -C $CC rev-parse HEAD > code_commit.txt
  echo "^start $name $(date '+%T')" >> $Q/equiv_queue.log; local t0=$(date +%s)
  $X/run_logged.sh . env $ENVV "$@"
  echo "^done $name $(date '+%T') seconds=$(( $(date +%s) - t0 )) $(cat status.txt 2>/dev/null)" >> $Q/equiv_queue.log; }
E150=$Q/equiv150_clean_$K
[ -d $E150 ] && ! grep -q QUICK_MAP_DONE $E150/run.log 2>/dev/null && mv $E150 ${E150}_killed1525
echo "^queued phase 1 $(date '+%T') (chain_clean $K, check $KC)" >> $L
exec 9>>$LOCK
flock 9
echo "^start phase 1 (lock held) $(date '+%T')" >> $L
halt equiv150
equiv equiv150_clean_$K $PY -W ignore $X/quick_map.py $CC $E150/map 150
for f in $E150/map/checkpoint_b.pt $E150/map/checkpoint_b_post_ref.pt; do
  [ -f "$f" ] && [ ! -L "$f" ] && { echo "$(date '+%F %H:%M') equivalence 150 frames (only run.json / run.log are compared): $(stat -c %s $f) $f" >> $M; rm -- "$f"; }; done
halt check150
D1=$Q/check150_real_b_$KC; mkdir -p $D1; cd $D1; git -C $CHK rev-parse HEAD > code_commit.txt
echo "^start check150 $KC $(date '+%T')" >> $L
$X/run_logged.sh . env $ENVV $PY -W ignore $X/quick_map.py $CHK $D1/map 150 real b
echo "^done check150 $KC $(date '+%T') $(cat status.txt 2>/dev/null)" >> $L
grep "CHECK_7F12297" $D1/run.log | tail -1 >> $L
clean $D1/map "check run of 7f12297 (real B 150 frames; run.log kept)"
flock -u 9; exec 9>&-
echo "^end phase 1 (lock released) $(date '+%T')" >> $L
CHECK_OK=1
if grep -q "CHECK .*differs\|CHECK .*not empty\|CHECK untouched" $D1/run.log || ! grep -q "QUICK_MAP_DONE" $D1/run.log; then
  CHECK_OK=0; echo "^check FAILED or incomplete: no real B rerun" >> $L; fi
until grep -q '^\^post chain queue end' $Q/post_chain_queue.log 2>/dev/null; do sleep 60; done
echo "^start phase 2 $(date '+%T')" >> $L
halt real_b_rerun
if [ $CHECK_OK = 1 ]; then
  D2=$RR/real_b_rerun_$K; mkdir -p $D2; cd $D2; git -C $CC rev-parse HEAD > code_commit.txt
  echo "^start real_b_rerun $K $(date '+%T') disk $(df --output=used -BG /home/jixian | tail -1 | tr -d ' ')" >> $L
  flock $LOCK $X/run_logged.sh . env $ENVV $PY -W ignore $X/quick_map.py $CC $D2/map 0 real b
  echo "^done real_b_rerun $K $(date '+%T') $(cat status.txt 2>/dev/null)" >> $L
  python3 - $D2/map/session_b/run.json $RR/real_chain_4ca6976/map/session_b/run.json $D2/run.log $RR/real_chain_4ca6976/run.log >> $L <<'PY'
import json, re, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
def last_el(p):
    m = re.findall(r'real_b t=\s*[\d.]+s frames=\d+/\d+ elements=(\d+)', open(p, errors='ignore').read())
    return int(m[-1]) if m else None
print(f'^rerun B seconds {a["seconds"]} (line 1532.6 s, native 1178.9 s -> {a["seconds"] / 1178.9:.3f}x) '
      f'{"PASS" if a["seconds"] <= 1532.6 else "FAIL"}; chain B {b["seconds"]}; peak {a.get("peak_gpu_gb")} GB')
print(f'^retired rerun {a["retired_by_layer"]} chain {b["retired_by_layer"]}; live elements at B end rerun {last_el(sys.argv[3])} chain {last_el(sys.argv[4])}')
PY
  clean $D2/map "real B rerun with chain_clean $K (time measurement; run.json / run.log kept)"
fi
halt equiv30_real_bc
equiv equiv30_real_bc_clean_$K flock $LOCK $PY -W ignore $X/quick_map.py $CC $Q/equiv30_real_bc_clean_$K/map 30 real bc
clean $Q/equiv30_real_bc_clean_$K/map "equivalence smoke (real B->C 30 frames, chain_clean)"
halt equiv4k
equiv equiv4k_clean_$K flock $LOCK $PY -W ignore $X/quick_screen.py short4k clean synthetic $Q/equiv4k_clean_$K $CC
echo "^equiv queue end $(date '+%T')" >> $Q/equiv_queue.log
halt rule_table_diag
$W/harness_f3c7f07/experiments/rule_table_diag_queue.sh
halt release_sets
$W/harness_b67723f/experiments/release_sets_queue.sh
echo "^after real sequence end $(date '+%T')" >> $L
