"""Snapshot readout by the first echo (planned replacement of the [E1] readout of game.py _render_scene_alive; the
CUDA output follows after 2026-10-06 18:20, this is its reference).

Today's snapshot labels a pixel by the alpha-weighted mean of the class ids of the Gaussians it blends and drops it
when their variance is >= 0.25 (a variance of class ids: a 5/6 half-half blend gives 0.25, a 0/20 blend 100; the 0.25
has no source), and its depth is D / alpha with a 5% jump test. The sourced readout (README M1a, the first echo; the
median depth of 2DGS, ul_rasterizer forward.cu:406-408 'T first drops to 0.5'): along each pixel's ray, front to back,
the Gaussian at which the transmittance T first falls to <= 0.5 gives both the depth (the median depth, as the
median-depth TSDF export) and the identity of the pixel. A pixel whose T never falls to 0.5 has no surface.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch


def first_echo(alphas: Sequence[float], depths: Sequence[float], ids: Sequence[int]) -> Tuple[Optional[float], int]:
    """One ray, Gaussians sorted front to back: (depth, id) of the Gaussian at which T first drops to <= 0.5, else
    (None, -1). The rule of forward.cu: T > 0.5 before it and T * (1 - alpha) <= 0.5 after it."""
    T = 1.0
    for a, d, i in zip(alphas, depths, ids):
        test_T = T * (1.0 - a)
        if T > 0.5 and test_T <= 0.5:
            return float(d), int(i)
        T = test_T
    return None, -1


def pixel_identity(median_index: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
    """Per pixel the identity of its first-echo Gaussian (median_index: the Gaussian index the rasterizer records
    at the median crossing, -1 where T never fell to 0.5); -1 where there is none."""
    out = torch.full_like(median_index, -1, dtype=torch.int64)
    ok = median_index >= 0
    out[ok] = identity[median_index[ok]].clamp(min=0)
    return out
