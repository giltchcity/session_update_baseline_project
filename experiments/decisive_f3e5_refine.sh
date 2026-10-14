#!/usr/bin/env bash
# Decisive experiment (supervisor 15:30, for the user: "先守住真实，再让合成好"). Question: does the synthetic line need R11p?
# Input: the f3e5 mapping's synthetic B (syn_row4f3e5, pre_ref F1 91.42, no R11p). It is refined with the chain's refinement tree
# r21c 14f3855 (SH) and scored by the same evaluator, with the same steps as post_chain_queue.sh's refine() and synthetic scoring.
# Rule, written before the numbers (supervisor 15:30):
#   post_ref F1 >= 96.43 and P >= 96.6 and D1/D2/D3 unchanged
#     -> R11p is not needed for the synthetic line; the delivered mapping becomes f3e5 + R9h (real measured: B 92.98 / C 94.86).
#   otherwise
#     -> "no version on record holds both lines"; 4ca6976 and f3e5 + R9h go to the user as two named versions with their numbers.
# One hold of GPU.lock covers restore -> refine_only -> post_ref view -> synthetic scoring. The scoring is CPU work; it runs inside the
# hold so nothing else runs beside it.
# Disk: the restored pre-refinement copy is deleted after the refinement, as refine() does; the source checkpoint is kept.
set -u
RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; RT=$RR/worktrees/r21c
PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; Q=$RR/code_mine_20261014/runs_quick3k; L=$Q/decisive_f3e5_refine.log
LOCK=/home/jixian/Desktop/FT/runs/GPU.lock; M=$RR/analysis/DISK_RECLAIM_20261013.txt; EV=$RR/analysis/evaluator_snapshot_20261014
SRC=$RR/syn_row4f3e5; RK=$(git -C $RT rev-parse --short HEAD); E=$SRC/refine_$RK
git -C $RT status --porcelain | grep -q . && { echo "^refused dirty $RT" >> $L; exit 3; }
[ "$RK" = 14f3855 ] || { echo "^refused: r21c is at $RK, not 14f3855" >> $L; exit 4; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
mkdir -p $E/session_b
ln -sfn $SRC/session_a $E/session_a
ln -sfn "$(readlink -f $SRC/checkpoint_a.pt)" $E/checkpoint_a.pt
for f in $SRC/session_b/*; do b=$(basename $f); case $b in timeline_post_ref.pkl|tsdf_final.ply|run.json) continue;; esac; ln -sfn $f $E/session_b/$b; done
cp -f $SRC/session_b/run.json $E/session_b/run.json; git -C $RT rev-parse HEAD > $E/code_commit.txt
echo "^queued $(date '+%T'): refine $(readlink -f $SRC/checkpoint_b.pt) with r21c $RK -> $E" >> $L
exec 9>>$LOCK
flock 9
echo "^start (lock held) $(date '+%T') disk $(df --output=used -BG /home/jixian | tail -1 | tr -d ' ')" >> $L
( cd $RT && $PY -m update_layer.eval.kf_strip restore $SRC/checkpoint_b.pt $E/checkpoint_b.pt synthetic ) > $E/restore_b.log 2>&1
rc=$?; echo "^done restore $(date '+%T') rc=$rc" >> $L
# refine_only reads checkpoint_b.pt once, at its start (refine_only.py: torch.load before the backend is built). Once the refinement
# loop has begun (REFINEMENT_MEM it=0), the restored copy is no longer read and is deleted. Disk: 396 GB + 22.5 GB restored copy
# + ~22 GB post_ref would pass the 440 GB gate.
( cd $RT && $PY -W ignore -m update_layer.eval.refine_only $E synthetic b ) > $E/refine.log 2>&1 &
RP=$!
until grep -q "REFINEMENT_MEM it=0" $E/refine.log 2>/dev/null || ! kill -0 $RP 2>/dev/null; do sleep 5; done
if grep -q "REFINEMENT_MEM it=0" $E/refine.log && [ -f $E/checkpoint_b.pt ] && [ ! -L $E/checkpoint_b.pt ]; then
  echo "$(date '+%F %H:%M') restored pre-refinement copy of $SRC/checkpoint_b.pt (source kept on D:), loaded by refine_only and not read again: $(stat -c %s $E/checkpoint_b.pt) $E/checkpoint_b.pt" >> $M
  rm -- $E/checkpoint_b.pt; echo "^deleted the restored copy after the load $(date '+%T')" >> $L; fi
wait $RP; rc=$?; echo "^done refine $(date '+%T') rc=$rc" >> $L
bash $RR/post_ref_view.sh $E b > $E/post_ref_view.log 2>&1
rc=$?; echo "^done post_ref view $(date '+%T') rc=$rc" >> $L
( cd /home/jixian/Desktop/FT && xargs -a $EV/FILES.txt sha256sum ) > $E/post_ref/evaluator_sha256.txt 2>&1
if cmp -s $E/post_ref/evaluator_sha256.txt $EV/SHA256SUMS.txt; then echo "^evaluator same as the snapshot $(date '+%T')" >> $L
else echo "^evaluator DIFFERS from the snapshot $(date '+%T')" >> $L; fi
git -C $RT rev-parse HEAD > $E/post_ref/eval_syn_code_commit.txt
( cd $RT && bash update_layer/eval/eval_synthetic_scene.sh $E/post_ref $E/post_ref/eval_syn ) > $E/eval_post_b.log 2>&1
rc=$?; echo "^done synthetic scoring $(date '+%T') rc=$rc" >> $L
flock -u 9; exec 9>&-
echo "^lock released $(date '+%T')" >> $L
python3 - $E $SRC $RR >> $L 2>&1 <<'PY'
import csv, json, sys
E, SRC, RR = sys.argv[1:4]
def geo(p):
    try:
        for r in csv.DictReader(open(p)):
            if r['method'] == 'ours' and r['session'] == 'b' and r['scope'] == 'all' and r['threshold_m'] == '0.05':
                return {k: round(100 * float(r[k]), 2) for k in ('Precision', 'Recall', 'F1')}
    except Exception as e:
        return f'missing ({e})'
def states(d):
    try:
        return {r['kind']: f"TP {r['TP']} FP {r['FP']} FN {r['FN']} F1 {float(r['F1']):.3f}"
                for r in csv.DictReader(open(d + '/ours/state_eval/group_scores/GROUP_STATE_SUMMARY.csv'))
                if r['session'] == 'ab' and r['kind'] in ('D1_visible', 'D2_hidden', 'D3')}
    except Exception as e:
        return f'missing ({e})'
rows = [('this experiment post_ref (f3e5 B + r21c 14f3855)', E + '/post_ref/eval_syn'),
        ('its input pre_ref (syn_row4f3e5)', SRC + '/eval_syn'),
        ('chain post_ref (4ca6976 B + 14f3855)', RR + '/syn_chain_4ca6976/refine_14f3855/post_ref/eval_syn'),
        ('R11p pre_ref (RULES reference for D1/D2/D3)', RR + '/syn_row4f3e5_r11p/eval_syn')]
for name, d in rows:
    print(f'^{name}: G1 {geo(d + "/geometry/FINAL_GEOMETRY.csv")}; states (ab) {states(d)}')
pr = (json.load(open(E + '/session_b/run.json')).get('post_ref') or {})
print(f'^refinement {pr.get("completed_iterations")}/{pr.get("requested_iterations")} steps, {pr.get("seconds")} s, peak {pr.get("peak_gpu_gb")} GB')
print('^lines: post_ref F1 >= 96.43 and P >= 96.6 and D1/D2/D3 unchanged (rule written 15:30, before the numbers)')
PY
echo "^decisive experiment end $(date '+%T')" >> $L
