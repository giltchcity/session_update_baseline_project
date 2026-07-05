# Base1.1 Round13 Method Search: Graph-Cut Plane Hole Fill

Date: 2026-07-05

Scope: Base1.1 / Mode A, single-session Khronos office final-current map.

## Goal Restatement

The target is still the one in `base1.1_mode_a_single_session.md`: improve the
raw Khronos final background/global mesh, report Background Precision and Recall
separately, and do not claim success from F1 alone. Per user request, 5cm is
also mandatory.

Round12 already exceeded `+0.02` F1@10 over raw, but it did not improve 5cm F1:

```text
Round12 vs raw:
F1@5:  -0.002777
F1@10: +0.022416
F1@20: +0.005863
```

So the next question is not "tune a threshold"; it is whether a better region
selection method can keep the 10/20cm gain while repairing the 5cm loss.

## Failure Symptom

Round12 footprint fill selected cells with a hand-written row/column interval
rule. It correctly avoided full rectangle overfill, but the 5cm result says the
surface boundary and/or height is too coarse:

```text
P@5 raw:      0.864665
P@5 Round12:  0.809938
R@5 raw:      0.633729
R@5 Round12:  0.662148
```

Recall moved, precision fell. That is the signature of plausible surfaces being
placed a little too broadly or at a nearby but wrong height for the 5cm metric.

Axis-bin evidence also explains why 10cm can look good while 5cm still suffers:

```text
GT z bins: 4.00 has 1119048 points, 3.92 has 68964 points
GT z bins: 1.04 has 596899 points, 0.96 has 500369 points
DSG z bins: 1.04 has 174602 points, 0.96 has only 4213 points
```

The 8cm layer gap is mostly tolerated at 10cm, but not at 5cm.

## Source Search

Search terms:

```text
surface reconstruction minimum s-t cut visibility energy
indoor scene reconstruction graph-cut walls floors ceilings
graph cuts binary energy data smoothness term
volumetric graph cuts scene reconstruction max-flow
```

### What Energy Functions Can Be Minimized via Graph Cuts?

Paper: `What Energy Functions Can Be Minimized via Graph Cuts?`

URL: https://www.cs.cornell.edu/~rdz/Papers/KZ-PAMI04.pdf

Short original phrases:

```text
"binary-valued variables"
"minimum cut"
"max flow algorithms"
```

Method takeaway:

```text
Use a binary label per grid cell: FILL or EMPTY. Keep the energy in unary +
pairwise Potts form so the s-t cut is exact for this subproblem.
```

### Multi-view Stereo via Volumetric Graph-cuts

Paper: `Multi-view Stereo via Volumetric Graph-cuts`

URL: https://www.robots.ox.ac.uk/~phst/Papers/CVPR05/George_cvpr2005.pdf

Short original phrases:

```text
"weighted graph"
"minimum cut solution"
"existing max-flow algorithms"
```

Method takeaway:

```text
The surface decision can be made as a weighted graph partition instead of a
greedy local rule.
```

### Robust and Efficient Surface Reconstruction from Range Data

Paper: `Robust and efficient surface reconstruction from range data`

URL: https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Labatut09.pdf

Short original phrases:

```text
"energy minimisation problem"
"soft visibility constraints"
"minimum s-t cuts"
```

Method takeaway:

```text
Even when data is incomplete/noisy, the reconstruction decision should balance
surface quality against evidence, not blindly fill every geometric hole.
```

### Indoor Scene Reconstruction using Primitive-driven Space Partitioning and Graph-cut

Paper: `Indoor Scene Reconstruction using Primitive-driven Space Partitioning and Graph-cut`

URL: https://www-sop.inria.fr/members/Sven.Oesau/

Short original phrases:

```text
"walls, floors and ceilings"
"inside/outside labeling"
"data consistency"
```

Method takeaway:

```text
For indoor scenes, walls/floors/ceilings are exactly the structural primitives
where graph-cut region labeling is a natural fit. Later semantic wall/floor
labels can enter as unary costs.
```

## Selected Method

Method name:

```text
Graph-cut horizontal structural plane fill.
```

Variables:

```text
x_i = 1 means grid cell i should be filled.
x_i = 0 means grid cell i should stay empty.
```

Energy:

```text
E(x) = sum_i D_i(x_i) + lambda * sum_(i,j in N) [x_i != x_j]
```

Data term:

```text
Observed plane-support cells are hard FILL seeds.
Cells outside the footprint support are hard EMPTY by exclusion.
Unknown cells become cheaper to fill when they are near observed support and
more expensive near the footprint boundary.
```

Pairwise term:

```text
4-neighbor Potts smoothness on the plane grid.
```

Optimization:

```text
Solve the binary energy with an s-t min-cut/max-flow.
```

5cm-specific repair:

```text
For each strong structural horizontal plane, also consider the immediately
lower adjacent z-bin if it has real support. This targets the observed
0.96/1.04 and 3.92/4.00 layer split instead of hoping 10cm tolerance hides it.
```

## Expected Metric Movement

Desired:

```text
5cm: recover precision/F1 by reducing broad fill and adding the missed adjacent
     structural layer.
10cm: stay above raw by >2 percentage points in F1.
20cm: stay improved.
```

Failure condition:

```text
If F1@10 drops below the Round12 >2% result or 5cm still fails, this round is
not a better final answer; it is evidence that graph-cut region selection alone
is insufficient without visibility/free-space rays or deletion.
```
