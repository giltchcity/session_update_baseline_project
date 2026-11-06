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
Its ObjectEvaluator does not read `compensate_missegmentation` (its `declare_config` has no such field), so static
objects are always counted with the compensation, whatever the config says: per GT object only one of its map objects
is looked at, per map object only one of its GT objects.

## Per dataset

| Dataset | Code | Notes |
|---|---|---|
| Real room A/B/C | `eval_fast/` (`run.py`, `object_only.py`, `configs/`) | online object P/R/F1 duration-weighted with the NR correction, G1, D2, D3; one snapshot at a time in `/dev/shm` |
| TESSE office/apartment | `tesse/tesse_online_eval.py`, `khronos_configs/eval_{office,apartment}*.yaml` | `ours4d`: each map queried at every map time up to its own (the fixed evaluator); ours as a 4D map derived by `chain_fill`; the background at the config's 5 cm threshold (`single_run/official_tables_at.py`), ours with its objects and historical objects (`tesse/tesse_bg_history.py`); G1 the final map at 5 cm |
| TESSE, semantic input | `semantic_front/` (`tesse_semantic_front.py --3d`: the front; `tesse_semfront_run.sh`: our run on its ids; `object_loss_check.py`: where a map's static-object score goes, from the evaluator's own per-object output), `runs/tesse_khronos_inst.sh` with `KHRONOS_DETECTOR=semantic` | object ids from the semantics only, for both: Khronos's own ConnectedSemantics at 5 cm; ours the semantic front (Khronos's 3-D regions and MaxIoU tracking, an id written once its surface has been measured again more often than seen through, no ids for people, as the annotation); the rest as the TESSE row. `semantic_front/front_e4a02cf/` is the GaME session's `update_layer/front.py` and `frames.py` at e4a02cf, which the front builds on (not in this branch's history) |
| Single runs (each visit mapped alone: real A/B/C, synthetic A/B) | `single_run/` (`single_run_official.py`: the visit's ground truth; `dynamic_gt_eval.py`: the moving things, real people and synthetic objects moved in view; `single_official_table.py`), `tesse/` | the TESSE table's metrics with the visit's own ground truth; our 4D map with the run's own records (`chain_fill/change_fill`, `chain_fill/motion_fill`); runs `runs/single_run_compare.sh`, `runs/real_single_fix1101.sh`, `runs/synthetic_live_obs.sh`, scoring `runs/single_run_official{,_mv}.sh`, `runs/single_bg_all.sh` |
| 3RScan | `threerscan_khronos_eval.py`, `chain_fill/`, `timeline_complete.py`, `khronos_configs/eval_3rscan_*` | the visit chain joined by `chain_fill`; `timeline` completes object timelines of methods without a 4D map (the Gaussian rows) with the same rules, and gives our maps the same scores as `chain_fill` |
| Synthetic A/B | `synthetic/` (`run_synthetic_eval.py`, `synthetic_ram.py`, `historical_object_eval.cpp`) | the published synthetic metrics; the historical object evaluator at 0.5 m |
| Flat A/B | `flat/flat_eval.py` | G1 of a session's final map (objects present at the final stamp) against the Flat point references |

`chain_fill/` derives a 4D map from a run's own snapshots without remapping: an object node without a mesh keeps its
last mesh, an object first present in a later visit is first observed then; no move is declared (since 10-30, as the
GaME and wavemap exports: the run's closure count also counts removals and states reopened in place). For a single
run, `change_fill` (before `chain_fill`) writes the run's own changes as presence times: a state the run closed ends
between its snapshots, an appearance follows the official reconciler on the run's change evidence, and a move needs
the closure count to rise and the surface to move beyond the ground truth's own state gate (0.10 m); `motion_fill`
writes the run's moving-object trajectories (`single_run/ours_motion_from_run.py`). `timeline_export` / `timeline_to_4dmap` carry an object timeline (and, for a 4D map, its
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
