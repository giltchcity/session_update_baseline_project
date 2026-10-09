# Gap analysis 2026-10-09 (offline, existing checkpoints and logs; no GPU)

Code 0c9c487 (c066077 split rules + sparse I1 label table + P1 + GPU guard + resume device fix). Lines: A 93.21, B 93.61 (relaxed 93.11), C 95.33 (relaxed 94.83); synthetic rows: row 1 84.91, row 3 90.25.

## 1. Real B 92.66 vs 93.61 and real C 94.86 vs 95.33 — G1@5 decomposition (update_layer/eval/g1_decompose, same samples as the scorer)

| map | P | errors: change / behind / in_front / lateral (% of samples) | R | misses: change / unobserved / observed |
|---|---|---|---|---|
| B final 0c9c487 | 91.47 | 2.64 / 1.85 / 0.99 / **3.05** (lateral median offset 13.4 cm) | 93.89 | 2.19 / 1.18 / 2.74 |
| B decided 29d7f6e | 91.02 | 2.63 / 2.31 / 0.99 / 3.05 | 93.85 | 2.23 / 1.24 / 2.68 |
| B row 1 (GaME native) | 95.86 | 2.96 / 0.60 / 0.24 / 0.34 | 91.47 | 3.37 / 2.16 / 3.00 |
| B row 3 | 87.98 | 3.85 / 3.51 / 1.50 / 3.18 | 89.31 | 3.27 / 1.01 / 6.41 |
| C final 0c9c487 | 95.12 | 2.03 / 1.46 / 0.58 / 0.82 | 94.61 | 2.47 / 1.22 / 1.70 |
| C decided | 94.63 | 1.97 / 1.78 / 0.61 / 1.01 | 94.51 | 2.44 / 1.15 / 1.90 |
| C row 1 | 97.00 | 1.70 / 0.67 / 0.31 / 0.31 | 93.72 | 2.29 / 1.91 / 2.08 |
| C row 3 | 90.12 | 3.65 / 3.75 / 0.95 / 1.53 | 91.38 | 3.25 / 1.17 / 4.21 |

Reading. Our change handling is at least as good as row 1 (change errors 2.64 vs 2.96 in B, 2.03 vs 1.70 in C; change misses 2.19 vs 3.37, 2.47 vs 2.29) and recall is higher than row 1 in both sessions. The whole B gap to the line is precision of the *carried map seen from the new session*: lateral 3.05 % (surfaces 13 cm off any reference surface sideways: duplicated / over-extended carried A geometry) + behind 1.85 % + in_front 0.99 % against row 1's 0.34 + 0.60 + 0.24. Reaching 93.61 at B's recall needs P >= 93.4: 1.9 pp of the 5.9 pp lateral+behind+in_front excess. In C the same three classes are 2.86 % vs row 1's 1.29 %; the line 95.33 at C's recall needs P >= 96.1, i.e. 1.0 pp of the 1.6 pp excess. Mechanism (measured 10-08: B-views-only export 94.42 vs official 92.42): the stored A keyframes render the B-trained map as novel views (S3 trains only this session's keyframes) and the fused export carries their lateral/behind errors. The layer's decisions (D2/D3 all GT-right, history PASS) are not where B and C lose; the carried-map alignment/extent is. No derived fix yet; J1 (joint pose refinement of the carried keyframes) was tried on 10-08 and did not close it (92.72).

## 2. Synthetic A 96.02 vs decided 96.82, synthetic B 88.86 vs row 3 90.25 — the observed-absence gate

Per-scope B (5 cm): structure 87.86 vs row 3 88.79; major_body 94.52 vs 94.59; group_attachments 94.62 vs 98.08 (P 90.1 vs 96.9); other_detail 76.31 vs 99.50 (P 64.1 vs 99.3). Time course (CURRENT_GEOMETRY_CURVES): at 55-65 s (the GT event cluster at 60-61 s) the new map's precision falls to 60.6 (all) / 66.8 (major_body) vs decided 74.8 / 88.7, recovers by 70 s and ends above the decided (84.2 vs 82.5). Both versions close the moved objects at the same looks (14 closures, identical).

Root cause of the missed closures (synthetic A inst 89, 156; and the per-look evidence everywhere): the present model. The look's evidence is -log density of the seen-through share f under the Beta projected from the in-place population's moments (projected_physical_evidence.cpp:625-640, ported to l2/evidence.py present_beta) with the variance floor kShareVarianceFloor = 1e-4 ("the standard deviation of a share is at least one percentage point", :183-191). Under the strict footprint rule (the TSDF's own rule) the in-place share is exactly 0 in more than half of the looks on both datasets (MAD = 0 everywhere), so the scatter is the floor everywhere, and the concentration of the projected Beta is c = m(1-m)/v - 1 ~ m / 1e-4: proportional to the mean share. Real sessions (m = 0.35-0.80 %): b = 33-78, a = 0.12-0.63; synthetic A (m = 0.033 %): b = 9, a = 0.010. With a < 1 the projected density is singular at 0 and flat elsewhere: a half-seen-through look is worth 9.6 nats on synthetic A, 24-52 on real, 56 in the decided version (m = 0.87 % under the lenient rule). inst 89 at 109 s: contradiction 1877 > support 598 in both versions (the dominance vote would close it) but the gate (page statistic > log 99 = 4.595) read 1.78 (then 3.80) instead of 13.24: the closure was lost by 0.8 nats. The evidence scale shrinks as the sensor gets cleaner: the opposite of robustness.

Derived fix (no new number). The floor is the regulariser against a degenerate projection (HALCON Regularize / Bishop 9.2.1 per the source comment). A moment projection with v >= floor is non-degenerate (a >= 1, no singularity at 0) iff m * c >= 1, i.e. m >= sd for small m: the floor on the standard deviation implies the same bound on the mean, otherwise the regulariser produces the singular density it exists to prevent. So present_beta bounds the mean by the floor's own noise level: m = max(mean, sqrt(K_SHARE_VARIANCE_FLOOR)) (replacing the arbitrary 0.001 guard, which the source says never acted on its data). Whenever the learned share is >= 1 pp (the TSDF's and the decided version's regime) nothing changes. Below it the present model is the exponential-like Beta(0.98, 97) of a share with 1 pp noise: a half-seen-through look is 65 nats on every sensor, a 5 % look 0.4 nats.

Offline check (scratch gate_replay.py: the gate replayed from every ABSENCE_LOOK k/n with the page weights approximated by the judged fraction, closure = gate open at a window with contradiction > support; the current scale reproduces the logged llr within 0.6-0.9 nats median; GT used only to label the outcome):

| session | current | mean bounded by the floor's noise level |
|---|---|---|
| real A | {6, 7, 10} (run: 7) | identical |
| real B | 9 (7 GT-change) | identical |
| real C | 9 (6 GT-change) | identical |
| synthetic A | 13 (all GT-change) | 15: + 89, + 156 (both GT-change), no static closure |
| synthetic B | 30 | identical |

The alternatives were not good on both: the learned full variance loses real C's inst 7 and collapses on synthetic B (one contaminated object); the mean absolute deviation adds static closures in real A (1, 17, 19, 20). The chosen form is the only one of the tested that is unchanged on every real session and on synthetic B and recovers the synthetic A closures.

Not yet done: the full points replay on real A (replay_points/base_a, CPU ~40 min) and the GPU chains (synthetic A->B, real A->B->C) with this one change; the synthetic has no recorded replay inputs.

## 3. Synthetic B: final-model depth L1 17.1 cm (decided 11.0), retention 73.9 % (75.4), absent_residue 21.0 % (18.0)

Localised but not resolved. The rendered timelines (new vs decided) differ at the old sites of sofa 156 and its pillows 147/148/149 at the end of B: 1,597-3,662 rendered background points within 0.4 m of each old centroid in the new readout vs 0-541 in the decided. The checkpoint rows, however, hold 0 alive rows within 0.4 m of the pillows' old centroids and only 104 (156) / 511 (155) background rows, 98 % of them horizontal floor rows 0.37 m below the centroid (the floor revealed under the moved sofa: correct geometry). So the extra points at the pillow sites come from the readout (rendering of the time-indexed map from the stored keyframes), not from rows the layer kept. This needs the render check on the GPU (minutes), not done today.
