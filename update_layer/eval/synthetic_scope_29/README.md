# Synthetic evaluation units = synthetic mapping identities (29)

The synthetic scene's paper scope (datasets/synthetic_ab/results/cvpr_surface_final_20260916/paper_aligned, 09-16:
"12 authored furniture GROUPs + 16 independent major objects = 28 units") scores a table together with what stands
on it, but the mapping input kept Isaac's per-member labels (152 identities), so the update layer saw a table's cups,
ornaments and books as separate few-millimetre segments, dropped them, and left their old copies to GaME's own
pruning when a group moved (10-21). User 10-21: input and evaluation must match.

This folder is the one definition both read:

- `EVAL_SCOPE_UNITS.csv`: the paper's 28 units with GROUP_large_sofa_CTRL split in two, the sofa (155) and its three
  throw pillows (173, 174, 175), because session B parts them (the sofa moves 3.78 m at 60.0 s, the pillows 0.40 m
  at 61.1 s). 29 units. Every other row is the paper's.
- `EVAL_SCOPE_MEMBERS.csv`: the paper's member list with the pillows' unit and role changed accordingly.
- Mapping: `frames.synthetic_instance_groups()` reads every member's label as its unit's first core member
  (114 labels regrouped); the harness reads the units from `$HARNESS_SCOPE_UNITS` (eval_synthetic_scene.sh).
