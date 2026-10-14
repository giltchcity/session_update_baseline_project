#!/usr/bin/env bash
# Disk reclaim of superseded maps whose conclusions are recorded, each only after the event that makes it unneeded (supervisor 12:44:
# the GPU queue ahead writes refinement copies, scorer exports, Flat maps and the equivalence runs; the user's limit is 450 GB).
# Rules kept: a line's newest map stays until its successor is evaluated and read; every deletion is written to the ledger first.
set -u
RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; Q=$RR/code_mine_20261014/runs_quick3k; M=$RR/analysis/DISK_RECLAIM_20261013.txt
CQ=$Q/chain_queue.log; PQ=$Q/post_chain_queue.log; L=$Q/disk_reclaim_after.log
used() { df --output=used -BG /home/jixian | tail -1 | tr -d ' G'; }
del() { local f=$1 why=$2
  if [ ! -f "$f" ] || [ -L "$f" ]; then echo "^skip $f (absent or a link) $(date '+%T')" >> $L; return; fi
  echo "$(date '+%F %H:%M') $why: $(stat -c %s "$f") $f" >> $M; rm -- "$f"; echo "^deleted $f $(date '+%T') disk $(used) GB" >> $L; }
# 1) after the real chain's tidy: maps of lines whose successors are evaluated
until grep -q "^\^tidied $RR/real_chain_4ca6976/map " $CQ 2>/dev/null; do sleep 60; done
del $RR/smoke_rt_syn_20261014/checkpoint_b.pt "RT (row-level truncation-band rule) smoke; the RT tree is the control stage, not shipped (FINAL_20261014/RULES.txt status line)"
del $RR/syn_row4f3e5_r10m/checkpoint_b.pt "R10m synthetic B, superseded by the R11p B (evaluated; R10m numbers in the 10-13/10-14 records)"
del $RR/syn_row4f3e5_r15p/checkpoint_b_post_ref.pt "r15p refinement of the R10m/R11p B, superseded by r21 and the SH run (both evaluated)"
del $RR/real_row4f3e5/session_c/timeline.pkl "real f3e5 C timeline, superseded by real_row4f3e5_r9h (evaluated) and this chain"
# 2) the SH 30K post_ref after this chain's refined B is read by the view tools and the SH-only measurements are done
until grep -q '^\^done synthetic views' $PQ 2>/dev/null && grep -q '^\^done sh bathroom_struct\|^\^sh bathroom_struct skipped' $Q/sh_queue.log 2>/dev/null \
      && grep -q TABLE_ITEMS_DONE $Q/table_items_count_1.txt 2>/dev/null; do sleep 60; done
del $RR/exp_sh_r11p_14f3855/checkpoint_b_post_ref.pt "SH 30K refinement of the R11p B: decision made (sh_queue.log), view tables / bathroom / table count recorded; this chain's refined B is read"
# 3) the r9h real chain after this chain's real scoring
until grep -q '^\^done real scoring' $PQ 2>/dev/null; do sleep 60; done
for f in session_c/timeline.pkl checkpoint_c.pt checkpoint_c_post_ref.pt; do
  del $RR/real_row4f3e5_r9h/$f "real r9h chain, superseded by this chain (real scoring done; acceptance against real_row4k)"; done
echo "^disk reclaim after end $(date '+%T') disk $(used) GB" >> $L
