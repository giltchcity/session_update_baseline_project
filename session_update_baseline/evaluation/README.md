# Evaluation behind the results table

The evaluation code and the run scripts that produced the current results table, kept here as a versioned copy.
They run from their working locations under `/home/jixian/Desktop/FT` (`eval/`, `eval_fast/`, `configs/khronos/`,
`wt_takeover_runs/analysis/`), so their paths are absolute. Object matching is 0.5 m everywhere. Every evaluation
keeps its intermediate files in `/dev/shm` and writes only tables and scores.

## The official evaluator and its fix

Khronos's evaluator (MIT-SPARK/Khronos `khronos_eval` at 63faadde, under `eval_fast/official`) is used unchanged,
except `khronos_eval_query_earlier/pipeline_evaluator.cpp`: the official loop over a map's queries used the map's own
time for every query (`map_timestamps_[i]` for each `j <= i`); the fixed copy queries each map at every map time up to
its own, the paper's 4D definition (eq. 20). The query times come from `KHRONOS_EVAL_QUERY_TIMES` when maps are
evaluated one at a time.

## Per dataset

| Dataset | Code | Notes |
|---|---|---|
| Real room A/B/C | `eval_fast/` (`run.py`, `object_only.py`, `configs/`) | online object P/R/F1 duration-weighted with the NR correction, G1, D2, D3; one snapshot at a time in `/dev/shm` |
| TESSE office/apartment | `tesse/tesse_online_eval.py`, `khronos_configs/eval_{office,apartment}*.yaml` | `ours4d`: each map queried at every map time up to its own (the fixed evaluator); ours as a 4D map derived by `chain_fill` |
| 3RScan | `threerscan_khronos_eval.py`, `chain_fill/`, `timeline_complete.py`, `khronos_configs/eval_3rscan_*` | the visit chain joined by `chain_fill`; `timeline` completes object timelines of methods without a 4D map (the Gaussian rows) with the same rules, and gives our maps the same scores as `chain_fill` |
| Synthetic A/B | `synthetic/` (`run_synthetic_eval.py`, `synthetic_ram.py`, `historical_object_eval.cpp`) | the published synthetic metrics; the historical object evaluator at 0.5 m |
| Flat A/B | `flat/flat_eval.py` | G1 of a session's final map (objects present at the final stamp) against the Flat point references |

`chain_fill/` derives a 4D map from a run's own snapshots without remapping: an object node without a mesh keeps its
last mesh, a relocation the map counted keeps the old place as a closed node, an object first present in a later
visit is first observed then. `timeline_export` / `timeline_to_4dmap` carry an object timeline (and, for a 4D map, its
snapshots' layers and global meshes) to and from the evaluator's map format.

## Runs

`runs/` holds the scripts of the current rows: ours on the object-memory-resupport branch (`real_bc_resupport.sh`,
`tesse_resupport.sh`, `synthetic_b_resupport.sh`, `flat_b_resupport.sh`), the 4D derivation of our TESSE maps
(`tesse_chain4d.sh`), the official Khronos baselines (`official_khronos_flat.sh`, `tesse_khronos_inst.sh`,
`real_khronos_inst.sh`), the 0.5 m rescoring of earlier maps (`real_rescore_050.sh`, `rescore_050_tesse_3rscan.sh`,
`tesse_objects_050.sh`, `timeline_fix_3rscan.sh`), and the player that re-sends an unacknowledged frame
(`nss_flat_ros2_player_resend.py`, a copy of the branch's player).

The acceptance of the 2026-10-28 surface speed-up (every frame read once, a state limiting earlier frames once):
`once_chain.sh` (real A -> B -> C and synthetic A remapped and scored), `real_states_check.sh` (real A and C with the
state change), `synthetic_b_once.sh` and `synthetic_b_states.sh` (synthetic B and the full synthetic table),
`real_mapping_only.sh` and `tesse_mapping_only.sh` (mapping only, for timing and round-by-round comparison of the
round logs), `real_views.py` (a real session's final maps from five viewpoints, coloured by its G1 reference).

`GOALS.md` lists the goal, the acceptance rule for every change (no metric below the current one by more than
0.1 pp) and the status.
