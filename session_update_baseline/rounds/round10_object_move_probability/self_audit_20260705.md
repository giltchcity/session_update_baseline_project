# Self Audit: Round10 Process Failure

Date: 2026-07-05

## Direct Answer

Yes, there was negative improvement.

The previous object-injection line improves 10cm but hurts 5cm. That means the
method is not a clean geometric improvement. It mainly adds approximate coverage
that falls inside the 10cm tolerance while damaging stricter surface accuracy.

## Metric Facts

Official evaluator:

| method | F1@5cm | delta vs raw | F1@10cm | delta vs raw | delta vs Round09 at 10cm |
|---|---:|---:|---:|---:|---:|
| raw Khronos | 0.731401 | 0.000000 | 0.939408 | 0.000000 | -0.016614 |
| old full append | 0.718690 | -0.012710 | 0.954065 | +0.014657 | -0.001957 |
| Round09 hard gate | 0.721297 | -0.010104 | 0.956022 | +0.016614 | 0.000000 |
| Round10 probability best | 0.721887 | -0.009514 | 0.956135 | +0.016727 | +0.000113 |
| Round10 expected utility | 0.721580 | -0.009821 | 0.956003 | +0.016595 | -0.000019 |

The important failure:

```text
Round09 vs raw:
  F1@10cm: +0.016614
  F1@5cm:  -0.010104
```

So the apparent 10cm improvement is not enough to claim a robust method.

## What I Did Wrong

1. I optimized around the visible 10cm gain instead of first asking whether the
   geometry improved at stricter thresholds.

2. I treated a tiny `+0.000113` change as if it was a meaningful result. It is
   not. It is too small to justify a new method.

3. I kept modifying gates and thresholds before doing error attribution.

4. I did not stop when the evidence showed a contradiction:

```text
10cm improves.
5cm gets worse.
```

That contradiction should have forced a method rethink immediately.

5. I let object-level logic dominate the search even though the metric failure
   is surface-level.

## Why This Was Methodologically Weak

The current object injection method answers:

```text
Should this object mesh be appended or skipped?
```

But the actual metric asks:

```text
Are the final mesh surfaces within 5cm / 10cm of the ground-truth background?
```

Those are not the same question.

Appending a whole object mesh can help 10cm recall/precision while still adding
surfaces that are too misaligned for 5cm. That is exactly what the numbers show.

## Correct Process From Now On

Before any new method:

```text
1. Report P/R/F1 at 5cm, 10cm, and 20cm.
2. Compare against raw Khronos and the current best.
3. Explicitly mark negative deltas.
4. Run error attribution before changing code:
   - FP@5 / FP@10
   - FN@5 / FN@10
   - bucket by object bbox, semantic label, and spatial region
5. Only then choose a method.
```

## What Counts As A Real Improvement

Tiny metric changes do not count.

A method must satisfy one of:

```text
F1@10cm improves by >= 0.003 without lowering F1@5cm.
F1@5cm improves by >= 0.005 while keeping F1@10cm within 0.001 of current best.
Precision@5cm improves materially without collapsing Recall@10cm.
```

## What To Stop Doing

Stop:

```text
threshold sweeps as a substitute for method
object skip probability tweaks without surface evidence
reporting only the best threshold
calling tiny deltas success
adding modules before error attribution
```

## Next Required Work

The next work item is not another updater. It is an error audit:

```text
Build a 5cm/10cm FP/FN attribution table.
Find whether the biggest loss is:
  - object surface misalignment
  - ghost surfaces
  - missing background planes
  - global/local pose drift
  - duplicate/over-thick surfaces
```

Only after that can the method be chosen honestly.
