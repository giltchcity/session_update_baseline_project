# Absence test (Eq. 2) ablations for reviewer W2 (11-16 / 11-17)

The absence test of `session_core/src/evidence/projected_physical_evidence.cpp` (Page's statistic over candidate
change times with first-judged coverage weights, learned in-place Beta, cross-visit prior, commit at ln 99) and the
close rule of `persistent_object_state.cpp` (a state closes on the absence path when the test commits and the round's
rays confirm, contradiction > support; it also closes when this visit sees the identity elsewhere, which does not use
Eq. 2).

## Switches (environment, read once by the mapper; unset = the method)

| variable | effect |
|---|---|
| `SESSION_CORE_ABSENCE_ETA=X` | commit at ln X instead of ln 99 |
| `SESSION_CORE_ABSENCE_WEIGHTS=one` | every candidate's weight 1 (plain Page statistic) |
| `SESSION_CORE_ABSENCE_BETA=fixed` | the present model fixed at the cold start Beta(0.1, 1.9) instead of the learned one |
| `SESSION_CORE_ABSENCE_OFF=1` | the test never commits |

A set switch logs `ABLATION: absence test eta=... unit_weights=... fixed_beta=... off=...` once; the run scripts stop
when the line is missing (or present in the method's run).

## 1. Map-level ablation (real data)

`run_real_b.sh` maps one real A with the method, then real B once per arm from it; `run_real_c.sh` maps real C per arm
from that arm's B. Mapping as the main-table rows; scored by `eval_fast/run.py --mode full` (previous map as prior).
Output: `wt_takeover_runs/w2_detector_replay_20261116/real_b/` (progress.log has every score). About 8 min per B,
17-21 min per C; one run at a time on cores 8-15.

| arm | B object F1 | B D3 (TP/FP/FN) | C object F1 | C D2 | C D3 |
|---|---|---|---|---|---|
| method (eta = 99) | 98.39 | 15/0/1 | 97.46 | 2/0/0 | 10/0/0 |
| no weighting | 97.47 | 14/0/2 | 96.96 | 2/0/0 | 9/0/1 |
| fixed Beta | 98.28 | 15/0/1 | 92.95 | 0/0/2 | 8/0/2 |
| eta = 9 | 97.43 | 15/0/1 | 96.95 | 2/0/0 | 10/0/0 |
| eta = 999 | 98.27 | 15/0/1 | 95.64 | 1/0/1 | 10/0/0 |
| Eq. 2 off | 98.31 | 15/0/1 | 92.95 | 0/0/2 | 8/0/2 |

B D2 is 2/0/0 and G1 F1@5cm 0.954-0.960 in every arm. In B every object moved between the visits is also seen at its
new site, so it closes through `by_session_elsewhere` without Eq. 2 (khronos.log: method 4 absence + 4 elsewhere
closes, off 0 + 8). In C the removed I12 and I14 can only close through Eq. 2: off and fixed Beta (which never commits
on real data) miss them and both in-visit cabinet changes; no weighting closes the static objects 5 and 6.

## 2. Exact replay of the recorded evidence (state level)

Every scored round of a mapping run is one `ABSENCE_LOOK` line (k, n, reliable samples, llr, cumulative first-judged
counts per candidate) followed by `ABSENCE_COMMIT` (prior) and `STATE_EVIDENCE_WINDOW` (support, contradiction) of the
same thread. `replay.py` recomputes Eq. 2 from them; `replay.py --check LOG` must report 0 statistic and 0 decision
mismatches (the five diag runs: 2043 rounds, 707 paired decisions). A variant is exact until it commits later than
the method; past the method's commitment its older candidates restart (a conservative bound: it can only delay the
variant's decisions). `label.py` judges every first close of a state against the ground truth (synthetic: Isaac poses;
real: `results/abc_eval_final_baseline/events.json` and the cabinet's state intervals); `key_ablations.py` prints the
table. Logs: `wt_takeover_runs/motion_diag_20261112_summary/diag/{synthetic_a,synthetic_b,real_a,real_b,real_c}`.

| variant | synthetic: correct / false deletions, recall | real: correct / false deletions, recall |
|---|---|---|
| method (eta = 99) | 44 / 0, 95.7% | 12 / 2, 75.0% |
| eta = 3 | 46 / 0, 100% | 16 / 8, 100% |
| eta = 9 | 46 / 0, 100% | 15 / 3, 93.8% |
| eta = 999 | 44 / 0, 95.7% | 6 / 0, 37.5% |
| eta = 9999 | 42 / 0, 91.3% | 4 / 0, 25.0% |
| no weighting | 46 / 0, 100% | 14 / 11, 87.5% |
| fixed Beta | 14 / 0, 30.4% | 0 / 0, 0% |
| Eq. 2 off | 0 / 0, 0% | 0 / 0, 0% |

The method's two real false deletions: real A object 10 (127.3 s) and real B object 13 (95.3 s, where objects 11, 12
and 13 are seen through at once).
