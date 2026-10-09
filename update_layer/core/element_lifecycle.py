"""Bounded surface evidence, sampled in sensor time rather than reconciliation rounds.

This is a candidate observation model, not a calibrated sensor-noise estimator. It keeps
positive evidence instead of treating every depth miss as an independent Bernoulli trial.
Uncertain rays add nothing. A negative decision requests reversible suppression; physical
object-state closure and explicit retirement remain separate, permanent decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..frames import Frame, dynamic_mask, torch_pixel_normals
from ..interface import Elements


@dataclass(frozen=True)
class SurfaceWeightConfig:
    # An integration horizon, in effective sampled observations, common to both datasets.
    # These are explicit candidate design constants; no claimed calibration to held-out F1.
    max_weight: float = 8.0
    initial_weight: float = 1.0
    suppress_at: float = -1.0
    max_state_bytes: int = 256 * 1024 * 1024

    def __post_init__(self):
        if not (np.isfinite(self.max_weight) and np.isfinite(self.initial_weight) and
                np.isfinite(self.suppress_at) and self.suppress_at < self.initial_weight <= self.max_weight) or \
                self.max_state_bytes <= 0:
            raise ValueError("invalid bounded surface-weight configuration")


def _pool_min(depth: torch.Tensor, radius: int) -> torch.Tensor:
    """Separable min filter: invalid pixels are +inf and do not assert free space."""
    k = 2 * radius + 1
    image = -depth[None, None]
    image = F.max_pool2d(image, (1, k), stride=1, padding=(0, radius))
    return -F.max_pool2d(image, (k, 1), stride=1, padding=(radius, 0))[0, 0]


@torch.no_grad()
def classify_elements(frame: Frame, el: Elements, tolerance: float, max_range: float,
                      dynamic_semantics: Sequence[int] = (), chunk_size: int = 131072,
                      positive_only: bool = False) -> dict:
    """Current-frame support and reliable free-space weight for current/dormant elements.

    ``hit`` requires geometry AND the same physical label (including background). It is the
    only recovery signal. ``support`` also admits a geometrically matching unlabeled pixel
    as weak evidence for a live row, without allowing that pixel to resurrect an object.
    ``through_weight`` is zero for occlusion, dynamic/invalid pixels, grazing observations,
    or nearby measured geometry inside the positional-uncertainty footprint. Missing
    neighbors reduce reliability rather than making every hole a universal veto.

    Distances use ray range, including the element's extent. The footprint uses positional
    tolerance only, so a large Gaussian does not acquire an arbitrarily large image veto.
    Radii beyond 32 pixels produce unknown negative evidence, bounding temporary work.
    """
    if not np.isfinite(tolerance) or tolerance <= 0 or not np.isfinite(max_range) or max_range <= 0:
        raise ValueError("surface tolerance and max_range must be finite and positive")
    dev, n = el.ids.device, len(el)
    hit = torch.zeros(n, dtype=torch.bool, device=dev)
    support = torch.zeros(n, dtype=torch.float32, device=dev)
    through_weight = torch.zeros_like(support)
    if not n:
        return dict(hit=hit, support=support, through_weight=through_weight)
    depth = torch.as_tensor(frame.depth, dtype=torch.float32, device=dev)
    inst = torch.as_tensor(frame.instance, dtype=torch.int64, device=dev).clamp(min=0)
    rejected = torch.as_tensor(dynamic_mask(frame, dynamic_semantics), dtype=torch.bool, device=dev)
    H, W = depth.shape
    K = frame.K
    if inst.shape != depth.shape or (H, W) != (K.height, K.width):
        raise ValueError("surface evidence frame/instance/intrinsics shape mismatch")
    T = torch.as_tensor(frame.T_world_cam, dtype=torch.float32, device=dev)
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    ray_scale = torch.sqrt(1 + ((xx + K.offset - K.cx) / K.fx) ** 2 +
                           ((yy + K.offset - K.cy) / K.fy) ** 2)
    valid = torch.isfinite(depth) & (depth > 0) & (depth * ray_scale <= max_range) & ~rejected
    clean = torch.where(valid, depth, torch.full_like(depth, float("inf")))
    near = _pool_min(clean, 1)
    far = F.max_pool2d(torch.where(valid, depth, torch.full_like(depth, -float("inf")))[None, None],
                       3, stride=1, padding=1)[0, 0]
    coverage = F.avg_pool2d(valid.to(torch.float32)[None, None], 3, stride=1, padding=1)[0, 0]
    spread = (far - near) * ray_scale
    coherent = (coverage >= 1.0 / 3.0) & (spread <= 2 * tolerance)
    local_weight = coverage * (tolerance / spread.clamp(min=tolerance)).clamp(max=1)
    normals = None if positive_only else torch_pixel_normals(
        torch.where(valid, depth, torch.full_like(depth, float("nan"))), K, T[:3, :3])
    # Only a few shared depth-image filters, rather than N x patch_size samples for millions of rows.
    clean_range = None if positive_only else torch.where(
        valid, depth * ray_scale, torch.full_like(depth, float("inf")))
    minimum = {} if positive_only else {1: _pool_min(clean_range, 1)}
    radii = (1, 2, 4, 8, 16, 32)
    for begin in range(0, n, chunk_size):
        end = min(n, begin + chunk_size)
        pts = el.xyz[begin:end]
        cam = (pts - T[:3, 3]) @ T[:3, :3]
        z = cam[:, 2]
        qrange = torch.linalg.vector_norm(cam, dim=1)
        safe_z = z.clamp(min=1e-8)
        uf = torch.floor(K.fx * cam[:, 0] / safe_z + K.cx - K.offset + 0.5)
        vf = torch.floor(K.fy * cam[:, 1] / safe_z + K.cy - K.offset + 0.5)
        in_view = torch.isfinite(cam).all(dim=1) & (z > 0) & (qrange <= max_range) & \
            (uf >= 0) & (uf < W) & (vf >= 0) & (vf < H)
        u = torch.nan_to_num(uf).long().clamp(0, W - 1)
        v = torch.nan_to_num(vf).long().clamp(0, H - 1)
        usable = in_view & valid[v, u]
        # Use the selected pixel's actual ray after rounding, not the point's continuous ray.
        delta = depth[v, u] * ray_scale[v, u] - qrange
        ext = el.extent[begin:end].clamp(min=0)
        band = tolerance + ext
        on = usable & torch.isfinite(ext) & (delta.abs() <= band)
        identity = el.identity[begin:end].clamp(min=0)
        own = inst[v, u] == identity
        # A physical instance mask is independent confirmation for thin surfaces whose depth
        # neighborhood is mixed. Background recovery instead needs a coherent surface patch.
        strict = on & own & (coherent[v, u] | (identity > 0))
        hit[begin:end] = strict
        weak = on & (own | (inst[v, u] == 0))
        support[begin:end] = torch.where(strict, torch.ones_like(delta), weak.float() * 0.5)
        if positive_only:
            continue
        # A nearby foreground sample vetoes a free-space claim even if this row's exact pixel
        # reads the wall behind it. This protects thin/edge geometry without globally disabling carving.
        radius = torch.ceil(max(K.fx, K.fy) * tolerance / safe_z).clamp(min=1)
        through = usable & (delta > band) & coherent[v, u] & (radius <= radii[-1])
        minimum_range = torch.full_like(z, float("inf"))
        previous = 0
        for r in radii:
            selected = through & (radius > previous) & (radius <= r)
            selected &= (u >= r) & (u + r < W) & (v >= r) & (v + r < H)
            if bool(selected.any()):
                if r not in minimum:
                    minimum[r] = _pool_min(clean_range, r)
                minimum_range[selected] = minimum[r][v[selected], u[selected]]
            previous = r
        through &= torch.isfinite(minimum_range) & (minimum_range - qrange > band)
        row_normal = el.normal[begin:end]
        known = torch.isfinite(row_normal).all(dim=1)
        test_normal = torch.where(known[:, None], row_normal, normals[v, u])
        test_normal = F.normalize(torch.nan_to_num(test_normal), dim=1)
        view = F.normalize(T[:3, 3] - pts, dim=1)
        facing = (test_normal * view).sum(dim=1).abs() >= 0.5  # same 60-degree incidence limit
        through &= facing & ~on
        through_weight[begin:end] = torch.where(through, local_weight[v, u], torch.zeros_like(z))
    return dict(hit=hit, support=support, through_weight=through_weight)


class SurfaceWeightEvidence:
    """One bounded signed weight per candidate uid; no saved frames/geometry/optimizer copies."""

    BYTES_PER_ROW = 8 + 4 + 8 + 1  # uid, weight, physical identity, last observed suppression flag

    def __init__(self, hz: float = 1.0, config: Optional[SurfaceWeightConfig] = None, state: Optional[dict] = None):
        if not np.isfinite(hz) or hz <= 0:
            raise ValueError("element observation rate must be positive")
        self.period_ns = max(1, int(round(1e9 / hz)))
        self.config = config or SurfaceWeightConfig()
        self.last_stamp: Optional[int] = None
        self.ids = self.weights = self.identities = self.suppressed = None
        if state is not None:
            self.load_state(state)

    def due(self, stamp: int) -> bool:
        return self.last_stamp is None or int(stamp) - self.last_stamp >= self.period_ns

    def _guard(self, n: int) -> None:
        if n * self.BYTES_PER_ROW > self.config.max_state_bytes:
            raise RuntimeError(f"element evidence needs {n * self.BYTES_PER_ROW} metadata bytes; "
                               f"budget is {self.config.max_state_bytes}; stop without discarding recovery state")

    def state(self) -> dict:
        return dict(version=1, period_ns=self.period_ns, last_stamp=self.last_stamp,
                    weight_parameters=(self.config.max_weight, self.config.initial_weight, self.config.suppress_at),
                    **{k: None if getattr(self, k) is None else getattr(self, k).detach().cpu()
                       for k in ("ids", "weights", "identities", "suppressed")})

    def load_state(self, state: dict) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported element evidence checkpoint version")
        if int(state.get("period_ns", 0)) != self.period_ns:
            raise ValueError("element evidence rate differs from checkpoint")
        parameters = (self.config.max_weight, self.config.initial_weight, self.config.suppress_at)
        if tuple(state.get("weight_parameters", ())) != parameters:
            raise ValueError("element evidence weight parameters differ from checkpoint")
        fields = [state.get(k) for k in ("ids", "weights", "identities", "suppressed")]
        if any(x is None for x in fields):
            if not all(x is None for x in fields):
                raise ValueError("incomplete element evidence checkpoint")
        else:
            n = len(fields[0])
            self._guard(n)
            if any(x.ndim != 1 or len(x) != n for x in fields) or \
                    fields[0].dtype != torch.int64 or fields[1].dtype != torch.float32 or \
                    fields[2].dtype != torch.int64 or fields[3].dtype != torch.bool or \
                    not bool(torch.isfinite(fields[1]).all()) or \
                    (n > 1 and not bool((fields[0][1:] > fields[0][:-1]).all())) or \
                    not bool(((fields[1] >= self.config.suppress_at) &
                              (fields[1] <= self.config.max_weight)).all()):
                raise ValueError("invalid element evidence checkpoint")
        last_stamp = state.get("last_stamp")
        if last_stamp is not None and (type(last_stamp) is not int or
                                       not 0 <= last_stamp < torch.iinfo(torch.int64).max):
            raise ValueError("invalid element evidence timestamp")
        if fields[0] is not None and last_stamp is None:
            raise ValueError("element evidence rows require an observation timestamp")
        self.last_stamp = last_stamp
        self.ids, self.weights, self.identities, self.suppressed = fields

    @torch.no_grad()
    def update(self, stamp: int, el: Elements, observations: dict) -> dict:
        empty = el.ids[:0]
        if not self.due(stamp):
            return dict(suppress=empty, reactivate=empty)
        n, dev = len(el), el.ids.device
        self._guard(n)
        hit, support, negative = (observations[k] for k in ("hit", "support", "through_weight"))
        if any(x.shape != (n,) for x in (hit, support, negative)) or \
                not bool(torch.isfinite(support).all() & torch.isfinite(negative).all()) or \
                bool(((support < 0) | (support > 1) | (negative < 0) | (negative > 1)).any()):
            raise ValueError("invalid surface evidence weights")
        cfg = self.config
        weights = torch.full((n,), cfg.initial_weight, dtype=torch.float32, device=dev)
        dormant = getattr(el, "suppressed", None)
        dormant = torch.zeros(n, dtype=torch.bool, device=dev) if dormant is None else dormant
        # New-to-this-checkpoint dormant rows have no guessed positive history.
        weights[dormant] = cfg.suppress_at
        if self.ids is not None and len(self.ids) and n:
            old_ids = self.ids.to(dev)
            pos = torch.searchsorted(old_ids, el.ids).clamp(max=len(old_ids) - 1)
            found = (old_ids[pos] == el.ids) & (self.identities.to(dev)[pos] == el.identity)
            old_weights = self.weights.to(dev)
            weights[found] = old_weights[pos[found]]
            # A keyframe can restore between sampled events. Keep its validated correction;
            # the old negative accumulator must not suppress it again on the next sample.
            restored = found & self.suppressed.to(dev)[pos] & ~dormant
            weights[restored] = cfg.initial_weight
        canonical_weights = getattr(el, "surface_weight", None)
        if canonical_weights is not None:
            if canonical_weights.shape != (n,) or not bool(torch.isfinite(canonical_weights).all()) or \
                    bool(((canonical_weights < cfg.suppress_at) | (canonical_weights > cfg.max_weight)).any()):
                raise ValueError("invalid backend surface weights")
            # The backend propagates these through clone/split and independent refinement.
            # New uid does not mean new evidence if it represents a densified parent surface.
            weights = canonical_weights.to(device=dev, dtype=torch.float32).clone()
        weights = (weights + torch.where(dormant, 0.0, support) - negative).clamp(cfg.suppress_at, cfg.max_weight)
        restore = dormant & hit
        weights[restore] = cfg.initial_weight
        suppress = ~dormant & ~hit & (negative > 0) & (weights <= cfg.suppress_at)
        final_dormant = (dormant | suppress) & ~restore
        order = torch.argsort(el.ids)
        self.ids = el.ids[order].clone()
        self.weights = weights[order]
        self.identities = el.identity[order].clone()
        self.suppressed = final_dormant[order]
        self.last_stamp = int(stamp)
        return dict(suppress=el.ids[suppress], reactivate=el.ids[restore], weights=weights)
