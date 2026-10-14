#!/usr/bin/env bash
# The five fixed view tools (FINAL_20261014/tools) on this chain's own pre_ref (syn_chain_4ca6976/map/checkpoint_b.pt), so that its
# refined map can be judged against the pre_ref it was refined from (seven_items.py; supervisor 12:04 / 12:14). GPU renders only,
# each under GPU.lock (~30 s), after the real mapping chain's process has ended (its time is measured).
set -u
RR=/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006; Q=$RR/code_mine_20261014/runs_quick3k; L=$Q/chain_preref_tools.log
S=$RR/syn_chain_4ca6976/map; PY=/home/jixian/Desktop/FT/envs/gs-cu128/bin/python; LOCK=/home/jixian/Desktop/FT/runs/GPU.lock
until grep -q '^\^done real_chain_4ca6976' $Q/chain_queue.log 2>/dev/null; do sleep 60; done
for s in views15_final bathroom_struct_final compensated_end_final heldout_last_final last_kf_final; do
  echo "^start $s $(date '+%T')" >> $L
  ( cd $RR && FINAL_CKPT=$S/checkpoint_b.pt FINAL_CODE=worktrees/r21c_base FINAL_TAG=syn_chain_preref flock $LOCK $PY -W ignore \
      $RR/FINAL_20261014/tools/$s.py ) > $S/$s.out 2>&1
  echo "^done $s $(date '+%T') rc=$?" >> $L
done
echo "^chain preref tools end $(date '+%T')" >> $L
