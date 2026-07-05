# Round10 Method Search Notes: Object Move Probability

Date: 2026-07-05

Scope: Base1.1 / Mode A. Round09 improved the best 10 cm result by using
Khronos `object_changes.csv` as a hard object-level gate. The next idea is to
replace the hard gate with a probability:

```text
P_moved(o) or P_static(o)
```

The goal is not to add a big module. The goal is one object-level probability
that controls injection and, later, conservative deletion.

## Current Best Baseline

Round09:

```text
raw Khronos F1@10cm:      0.939408
old full append F1@10cm:  0.954065
change-gated F1@10cm:    0.956022
```

Mechanism:

```text
if last_absent > last_persistent:
  skip object injection
else:
  inject object private mesh
```

This is effective, but binary. The probability version should preserve the same
intuition while becoming less brittle.

## Papers Read For Probability Inspiration

### 1. POCD: Probabilistic Object-Level Change Detection and Volumetric Mapping in Semi-Static Scenes

Source:

```text
https://arxiv.org/abs/2205.01202
https://www.roboticsproceedings.org/rss18/p013.pdf
```

Original sentence:

> "The representation jointly models a stationarity score and a TSDF change measure for each object."

Inspiration for us:

POCD says the state should live at the object level, not vertex level. Our direct
translation:

```text
P_static(o) = f(stationarity evidence, geometric change evidence)
P_moved(o)  = 1 - P_static(o)
```

For Base1.1, Khronos already gives rough stationarity/change evidence:

```text
last_persistent(o)
last_absent(o)
first_persistent(o)
first_absent(o)
```

### 2. Probabilistic Object Maps for Long-Term Robot Localization

Source:

```text
https://www.joydeepb.com/Publications/iros2022_pom.pdf
```

Original sentence:

> "considering a distribution of objects rather than a discrete set of landmarks."

Inspiration for us:

Do not treat an object as simply present/absent. Treat the object as a
probability mass over possible states:

```text
static here
moved away
new/moved elsewhere
unknown/unobserved
```

For current Base1.1, we only need the first two:

```text
P_inject(o) = P_static_here(o)
P_skip(o)   = P_moved_away(o)
```

### 3. Online Probabilistic Change Detection in Feature-Based Maps

Source:

```text
https://arpg.github.io/papers/08461111.pdf
```

Original sentence:

> "Bayesian filter to model feature persistence in a time-varying feature-based environmental model."

Inspiration for us:

Use persistence as a posterior, not a one-shot decision. The practical version:

```text
L_static(o) = logit(P_static(o))
L_static(o) += evidence_persistent(o)
L_static(o) -= evidence_absent(o)
```

Then:

```text
P_static(o) = sigmoid(L_static(o))
```

This is more graceful than:

```text
last_absent > last_persistent
```

because near-ties can become soft weights instead of hard flips.

### 4. MoPe: Motion Permanence for Robust Monocular Gaussian Mapping in Dynamic Environments

Source:

```text
https://arxiv.org/html/2606.29237v1
```

Original sentence:

> "Dynamic-ness is not an instantaneous appearance property, but a temporal property defined by motion history."

Inspiration for us:

This supports exactly the move from threshold gates to temporal posterior. For
Base1.1, object dynamic/moved state should remember the sequence of
absent/persistent observations instead of re-deciding from the final snapshot.

Minimal translation:

```text
history(o) = object_changes.csv row
P_moved(o) = posterior(history(o))
```

## Candidate Probability Model

Start with a tiny log-odds model:

```text
L_move(o) =
  b0
  + w_absent * recent_absent(o)
  - w_persistent * recent_persistent(o)
  + w_conflict * object_conflict(o)
```

Where the first loop can ignore `object_conflict` and use only Khronos object
change evidence:

```text
recent_absent(o) =
  1 if last_absent(o) > 0 else 0

recent_persistent(o) =
  1 if last_persistent(o) > 0 else 0

recency_margin(o) =
  (last_absent(o) - last_persistent(o)) / T
```

Then:

```text
P_moved(o) = sigmoid(L_move(o))
P_static(o) = 1 - P_moved(o)
```

Suggested first simple form:

```text
L_move(o) =
  alpha * clamp((last_absent - last_persistent) / 30s, -1, 1)
  + beta * I(last_absent > 0)
  - gamma * I(last_persistent > 0)
```

No learning yet. Hand-set weights, then sweep:

```text
alpha in {1, 2, 3}
beta  in {0, 0.5, 1}
gamma in {0, 0.5, 1}
```

## How To Use It In The Next Code Loop

Replace the binary Round09 gate:

```text
if last_absent > last_persistent:
  skip injection
```

with:

```text
if P_moved(o) >= tau_skip:
  skip injection
else:
  inject object mesh
```

Sweep:

```text
tau_skip in {0.5, 0.6, 0.7, 0.8}
```

Expected effect:

```text
Precision @10cm should remain high or improve.
Recall @10cm should recover on borderline objects that hard-gate skipped too aggressively.
```

## Future Deletion Use

Deletion should require a stricter threshold:

```text
delete ghost vertices only if P_moved(o) >= tau_delete
```

with:

```text
tau_delete > tau_skip
```

Suggested:

```text
tau_skip   = 0.5 - 0.7
tau_delete = 0.8 - 0.9
```

This matches the previous negative result: deletion needs stronger evidence than
injection skipping.

## First Round Acceptance

A probability round is useful only if it beats Round09:

```text
Round09 F1@10cm = 0.956022
Round09 P@10cm  = 0.976665
Round09 R@10cm  = 0.936233
```

Do not claim success unless:

```text
F1@10cm > 0.956022
```

or:

```text
P@10cm stays >= 0.976665
and R@10cm improves enough to justify the tradeoff.
```

If it does not beat Round09, record the failed weights and keep Round09 as the
current best.
