#!/usr/bin/env bash
# The synthetic chain's pre_ref score as soon as the chain is mapped, for the rule written before the run
# (FINAL_20261014/RULES.txt, STAGE syn: "pre_ref G1 F1 vs R11p 93.85: R9h removes reopened history; nothing else may fall.
# D1/D2/D3 as R11p."; supervisor 11:43: judge it as soon as the pre_ref score exists).
# CPU only (eval_synthetic_scene.sh reads the timelines, not the checkpoint), no GPU lock. It starts after the chain queue's tidy
# deleted the 0-step post_ref copy (the scorer's native export needs tens of GB of disk for a while), i.e. while the tidy's
# kf_strip runs; the real mapping chain can only start after that tidy. If the real chain's mapping process runs before the score
# is done (its python, not its flock waiter: the queue's '^start' marker is written before the lock is taken), the scoring process
# group is stopped (SIGSTOP) until that process has ended (its time is measured), then continued.
# The post-chain queue scores the same pre_ref again later (same scorer, same input).
set -u
RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; Q=$RR/code_mine_20261014/runs_quick3k; CQ=$Q/chain_queue.log
S=$RR/syn_chain_4ca6976/map; RT=$RR/worktrees/r21c_base; EV=$RR/analysis/evaluator_snapshot_20261014; L=$Q/syn_preref_early.log
until grep -q '^\^done syn_chain_4ca6976' $CQ 2>/dev/null && [ ! -e $S/checkpoint_b_post_ref.pt ]; do sleep 20; done
echo "^start pre_ref scoring $(date '+%T') disk $(df --output=used -BG /home/jixian | tail -1 | tr -d ' ')" >> $L
git -C $RT rev-parse HEAD > $S/eval_syn_code_commit.txt
( cd /home/jixian/Desktop/FT && xargs -a $EV/FILES.txt sha256sum ) > $S/evaluator_sha256.txt 2>&1
cmp -s $S/evaluator_sha256.txt $EV/SHA256SUMS.txt && echo "^evaluator same as the snapshot $(date '+%T')" >> $L || echo "^evaluator DIFFERS from the snapshot $(date '+%T')" >> $L
set -m                                   # job control: the scoring runs in its own process group (PGID = its PID)
bash -c "cd $RT && bash update_layer/eval/eval_synthetic_scene.sh $S $S/eval_syn" > $S/eval_b_early.log 2>&1 &
PID=$!
set +m
stopped=0
while kill -0 $PID 2>/dev/null; do
  running=0; pgrep -f '^/home/jixian/Desktop/FT/envs/gs-cu128/bin/python -W ignore .*quick_map\.py .*/real_chain_4ca6976/map ' > /dev/null && running=1
  if [ $running = 1 ] && [ $stopped = 0 ]; then kill -STOP -- -$PID; stopped=1; echo "^paused (real chain started) $(date '+%T')" >> $L; fi
  if [ $running = 0 ] && [ $stopped = 1 ]; then kill -CONT -- -$PID; stopped=0; echo "^continued (real chain done) $(date '+%T')" >> $L; fi
  sleep 10
done
wait $PID; rc=$?
echo "^done pre_ref scoring $(date '+%T') rc=$rc" >> $L
python3 - $S/eval_syn $RR/syn_row4f3e5_r11p/eval_syn >> $L <<'EOF'
import csv, sys
from pathlib import Path
def geo(d):
    for r in csv.DictReader(open(Path(d) / 'geometry/FINAL_GEOMETRY.csv')):
        if r['method'] == 'ours' and r['session'] == 'b' and r['scope'] == 'all' and r['threshold_m'] == '0.05':
            return {k: round(100 * float(r[k]), 2) for k in ('Precision', 'Recall', 'F1')}
def states(d):
    return {r['kind']: (int(r['TP']), int(r['FP']), int(r['FN'])) for r in csv.DictReader(open(Path(d) / 'ours/state_eval/group_scores/GROUP_STATE_SUMMARY.csv')) if r['session'] == 'ab'}
new, ref = geo(sys.argv[1]), geo(sys.argv[2])
print(f'pre_ref geometry (official, all scope, 5 cm): chain {new}  R11p {ref}')
print(f'rule "pre_ref F1 vs R11p 93.85": {"HOLDS" if new["F1"] >= ref["F1"] else "BELOW"} ({new["F1"] - ref["F1"]:+.2f})')
sn, sr = states(sys.argv[1]), states(sys.argv[2])
for k in sorted(set(sn) | set(sr)):
    print(f'  {k:18s} chain TP/FP/FN {sn.get(k)}  R11p {sr.get(k)}  {"same" if sn.get(k) == sr.get(k) else "DIFFERENT"}')
EOF
echo "^syn preref early end $(date '+%T')" >> $L
