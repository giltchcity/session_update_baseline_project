"""Session-end memory test of the TSDF version (session_refusion.cpp step 5 @192c1cf, lines 1045-1059 and
1129-1208, decisions 1254-1294), on any backend's elements.

A memory element -- one that existed before this session's first frame and that the object reasoning does
not own (background, or an object whose physical identity is current and whose state did not begin in this
session) -- is tested against every stored frame of the session (the layer's evidence store: the ranges the
present measured, with the rejected dynamic/motion pixels removed):
  hit       some pixel of its footprint (radius f tau sqrt(xn^2 + yn^2 + 1) / q) reads within tau and the
            reading's point lies inside the element's ball of radius tau;
  through   every pixel of the footprint is in the image, valid, and reads beyond it by more than tau;
  blocked   its own pixel reads nearer than -tau; blocked_band: of those, within its truncation.
It gives way to the present when seen_through = through > hit, or hidden = no hit, no through and
2 blocked_band > blocked. displaced (needs the sessions' depth scales, P41) is not ported yet: off.

Representation terms (CSV 2026-10-06 08:55): tau = max(half, sigma(q)); half = the element's extent
(interface.Elements.extent: a Gaussian's 3 sigma, a surfel's radius, half a cell's diagonal, 0 for points);
the truncation = the distance within which the representation cannot hold a second surface behind the
observed one = the element's extent (for a Gaussian its own support; assumption: two surfaces closer than
it are not resolved by it), so for tau >= extent the band is empty and `hidden` cannot fire. sigma(q): the
P37 range table is not ported; the layer's surface tolerance stands in (temporary). Frames: the evidence
store's (5 Hz) instead of every frame (30 Hz); hit and through are counted over the same frames. The
`stale` pixel removal of the archive (step 1b) and the removal of object pixels before t_L (step 1) are not
ported.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from .evidence import DEV


def memory_test(store, centroid: torch.Tensor, half: torch.Tensor, trunc: torch.Tensor, sigma: float,
                rejected=None, chunk: int = 65536) -> Dict[str, torch.Tensor]:
    """Evidence of every tested element over the store's frames: hit, through, blocked, blocked_band."""
    n = len(centroid)
    hit = torch.zeros(n, dtype=torch.int32, device=DEV)
    through = torch.zeros_like(hit)
    blocked = torch.zeros_like(hit)
    band = torch.zeros_like(hit)
    if not n or not store.n:
        return dict(hit=hit, through=through, blocked=blocked, blocked_band=band)
    K = store.K
    H, W = store.rng.shape[1:]
    for f in range(store.n):
        rng = store.rng[f].to(torch.float32) * 1e-3                    # metres, 0 = invalid
        if rejected is not None and rejected[f] is not None:
            rng = torch.where(rejected[f], torch.zeros_like(rng), rng)
        T = store.T[f]
        for a in range(0, n, chunk):
            p = centroid[a:a + chunk]
            cam = p @ T[:3, :3].T + T[:3, 3]
            x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
            ok = z > 0.1
            zs = torch.where(ok, z, torch.ones_like(z))
            u = torch.floor(x / zs * K.fx + K.cx - K.offset + 0.5).to(torch.int64)
            v = torch.floor(y / zs * K.fy + K.cy - K.offset + 0.5).to(torch.int64)
            ok &= (u >= 0) & (v >= 0) & (u < W) & (v < H)
            idx = torch.nonzero(ok).squeeze(1)
            if not len(idx):
                continue
            u, v, cam = u[idx], v[idx], cam[idx]
            q = torch.linalg.norm(cam, dim=1)
            tau = torch.clamp(half[a:a + chunk][idx], min=sigma)
            tr = trunc[a:a + chunk][idx]
            gi = idx + a
            # blocked / blocked_band at the element's own pixel
            d0 = rng[v, u]
            r0 = d0 - q
            bl = (d0 > 0) & (r0 < -tau)
            blocked[gi] += bl.to(torch.int32)
            band[gi] += (bl & (r0 >= -tr)).to(torch.int32)
            # footprint
            xn = (u.to(torch.float32) + K.offset - K.cx) / K.fx
            yn = (v.to(torch.float32) + K.offset - K.cy) / K.fy
            rp = K.fx * tau * torch.sqrt(xn * xn + yn * yn + 1.0) / q
            R = torch.floor(rp).to(torch.int64)
            h = torch.zeros(len(idx), dtype=torch.bool, device=DEV)
            all_ok = torch.ones(len(idx), dtype=torch.bool, device=DEV)
            for r_val in torch.unique(R).tolist():
                m = torch.nonzero(R == r_val).squeeze(1)
                dv, du = torch.meshgrid(torch.arange(-r_val, r_val + 1, device=DEV),
                                        torch.arange(-r_val, r_val + 1, device=DEV), indexing="ij")
                du, dv = du.reshape(-1), dv.reshape(-1)
                d2 = (du * du + dv * dv).to(torch.float32)
                for s in range(0, len(m), max(1, (1 << 22) // len(du))):
                    mm = m[s:s + max(1, (1 << 22) // len(du))]
                    inside_disk = (d2[None, :] <= rp[mm, None] ** 2) | ((du == 0) & (dv == 0))[None, :]
                    xx, yy = u[mm, None] + du[None, :], v[mm, None] + dv[None, :]
                    in_img = (xx >= 0) & (yy >= 0) & (xx < W) & (yy < H)
                    d = rng[yy.clamp(0, H - 1), xx.clamp(0, W - 1)]
                    valid = in_img & (d > 0)
                    r = d - q[mm, None]
                    t = tau[mm, None]
                    ray = torch.stack([(xx.to(torch.float32) + K.offset - K.cx) / K.fx,
                                       (yy.to(torch.float32) + K.offset - K.cy) / K.fy,
                                       torch.ones_like(d)], dim=-1)
                    point = ray / torch.linalg.norm(ray, dim=-1, keepdim=True) * d[..., None]
                    ball = ((point - cam[mm, None, :]) ** 2).sum(-1) <= t * t
                    hh = (inside_disk & valid & (r.abs() <= t) & ball).any(dim=1)
                    # every footprint pixel valid and beyond by more than tau
                    beyond = valid & (r > t)
                    ao = (~inside_disk | beyond).all(dim=1)
                    h[mm] = hh
                    all_ok[mm] = ao
            hit[gi] += h.to(torch.int32)
            through[gi] += (~h & all_ok).to(torch.int32)
    return dict(hit=hit, through=through, blocked=blocked, blocked_band=band)


def decide(ev: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """(seen_through, hidden) of step 5."""
    seen_through = ev["through"] > ev["hit"]
    hidden = (ev["hit"] == 0) & (ev["through"] == 0) & (2 * ev["blocked_band"] > ev["blocked"])
    return seen_through, hidden & ~seen_through
