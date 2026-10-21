# Synthetic evaluation units = synthetic mapping identities (29): backup copy

The definition lives in the dataset, once (user 10-21): the paper scope's
`datasets/synthetic_ab/results/cvpr_surface_final_20260916/paper_aligned/EVAL_SCOPE_{UNITS,MEMBERS}.csv` (the 28-unit
files kept beside them as `*_28units.csv`) and the label images `datasets/synthetic_ab/session_{a,b}/instance_labels`,
where every member of a unit reads as the unit's first core member (Isaac's per-member labels kept in
`instance_labels_per_member`; analysis/group_input_20261021/dataset_units.py wrote both from this folder and read them
back). The mapping reads the label images as they are; eval_synthetic_scene.sh and synthetic_object_f1.sh pass the
dataset's unit list to the harness. The files here are byte-identical copies, kept as a backup.

Why 29: the paper's scope (09-16: "12 authored furniture GROUPs + 16 independent major objects = 28 units") scores a
table together with what stands on it, but the mapping input had kept Isaac's per-member labels (152 identities), so
the update layer saw a table's cups, ornaments and books as separate few-millimetre segments, dropped them, and left
their old copies to GaME's own pruning when a group moved (10-21). User 10-21: input and evaluation must match. The
large sofa and its three throw pillows (173, 174, 175) are two units because session B parts them (the sofa moves
3.78 m at 60.0 s, the pillows 0.40 m at 61.1 s); every other row is the paper's.
