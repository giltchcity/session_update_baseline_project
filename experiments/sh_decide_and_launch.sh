#!/usr/bin/env bash
# Applies the decision rule written at 10:22:27 (sh_queue.log) as soon as the SH run's views15 table exists, then starts the TODO 5
# post-chain queue with the chosen refinement tree (no waiting on a person). Rule: r21c (r21 + SH ramp, 14f3855) iff the SH run's
# synthetic B post_ref F1 >= 96.43 and P >= 96.6 (all scope, 5 cm; official FINAL_GEOMETRY.csv) AND the mean MAE over the views15
# tool's views (the rows a2700 ... b3540 of the same table) is lower in the SH column ('final') than in the r21 column; else r21c_base.
# The losing refinement's post_ref checkpoint is deleted when it is the SH run (r21's is the tool's fixed reference).
set -u
RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; W=$RR/worktrees; Q=$RR/code_mine_20261014/runs_quick3k; L=$Q/sh_queue.log
E=$RR/exp_sh_r11p_14f3855; X=$(cd "$(dirname "$0")" && pwd); M=$RR/analysis/DISK_RECLAIM_20261013.txt
until grep -q '^\^sh30k rest end' $L 2>/dev/null; do sleep 30; done
verdict=$(python3 - "$E/post_ref/eval_syn/geometry/FINAL_GEOMETRY.csv" "$E/views15_final.out" <<'PY'
import csv, re, sys
geo = [r for r in csv.DictReader(open(sys.argv[1])) if r['session'] == 'b' and r['scope'] == 'all' and r['threshold_m'] == '0.05'][0]
f1, p = float(geo['F1']) * 100, float(geo['Precision']) * 100
lines = open(sys.argv[2], errors='ignore').read().splitlines()
head = next(i for i, l in enumerate(lines) if l.startswith('view'))
cols = lines[head].split()[1:]
rows = {}
for l in lines[head + 1:]:
    parts = l.split()
    if len(parts) == len(cols) + 1 and re.fullmatch(r'[ab]\d+', parts[0]):
        rows[parts[0]] = dict(zip(cols, map(float, parts[1:])))
mean = {c: sum(r[c] for r in rows.values()) / len(rows) for c in cols}
ok = f1 >= 96.43 and p >= 96.6 and mean['final'] < mean['r21']
print(('SH' if ok else 'R21') + f' F1={f1:.2f} P={p:.2f} views={len(rows)} meanMAE preref={mean["preref"]:.3f} r21={mean["r21"]:.3f} sh={mean["final"]:.3f}')
PY
)
echo "^sh decision $(date '+%T'): $verdict" >> $L
if [ "${verdict%% *}" = "SH" ]; then TREE=$W/r21c; else TREE=$W/r21c_base
  for f in $E/checkpoint_b_post_ref.pt; do [ -f $f ] && { echo "$(date '+%F %H:%M') SH 30K run: post_ref of the refinement that lost the decision ($verdict): $(stat -c %s $f) $f" >> $M; rm -- $f; }; done; fi
echo "^post chain queue launched with $(basename $TREE) $(git -C $TREE rev-parse --short HEAD) $(date '+%T')" >> $L
exec $X/post_chain_queue.sh $TREE
