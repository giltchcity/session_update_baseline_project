"""Projected physical evidence and the observed-absence test (torch; runs on the GPU).

Representation-free port of the t2 decision code (commit 43c663d):
  khronos/src/backend/change_detection/physical_evidence_store.cpp   -> EvidenceStore
  khronos/src/backend/change_detection/projected_physical_evidence.cpp
      classifyMeasurement             -> classify()
      countProjectedPhysicalSurface   -> count_projected_surface()
      applyObservedAbsence            -> ObservedAbsence.apply()
      countCurrentPhysicalSurface     -> ObservedAbsence.measure_state()
      save/loadAbsenceSensorStatistics-> SensorStatistics.save()/load()

The only input about a stored state is a set of surface samples with optional normals, in
world coordinates. How a backend produces them is the adapter's business. Rules, tolerances
and the single decision constant ln(99) are unchanged; points are float32 as in t2 (Eigen
Vector3f). t2 walks the frames one by one; here each per-sample "latest verdict" is the
maximum stamp over the round's frames with that verdict, which is the same value.
One deliberate difference: t2 first consults Khronos' mesh-ray index
(RayVerificator::countPhysicalSurface) and uses projected pixels only when that proxy proposes
absence or has no rays; with no mesh-ray index, the projected pixels always decide.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .frames import Frame, Intrinsics

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INVALID_CODE = int(np.iinfo(np.int32).min)
UNIDENTIFIED_CODE = -1
BACKGROUND_CODE = 0

# Endpoint classes (EndpointClass in physical_evidence_store.h).
UNAVAILABLE, INVALID, BACKGROUND, UNIDENTIFIED, PHYSICAL = range(5)


class Vote(IntEnum):
    UNAVAILABLE = 0
    INVALID = 1
    OCCLUDED = 2
    SUPPORTED = 3
    FREE = 4
    BACKGROUND = 5
    OTHER = 6
    UNIDENTIFIED = 7


def to_dev(x, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(DEV, dtype)
    return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=DEV)


def voxel_keys(points: torch.Tensor, cell: float) -> torch.Tensor:
    """One int64 per voxel (21 bits per axis, +-1M cells)."""
    k = torch.floor(points / cell).to(torch.int64) + (1 << 20)
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def first_per_key(keys: torch.Tensor) -> torch.Tensor:
    """Indices of the first occurrence of each distinct key, in input order."""
    if keys.numel() == 0:
        return torch.zeros(0, dtype=torch.int64, device=keys.device)
    uniq, inv = torch.unique(keys, return_inverse=True)
    first = torch.full((len(uniq),), keys.numel(), dtype=torch.int64, device=keys.device)
    first.scatter_reduce_(0, inv, torch.arange(keys.numel(), device=keys.device), reduce="amin")
    return torch.sort(first).values


def dedup_cells(points: torch.Tensor, cell: float) -> torch.Tensor:
    """First finite point per grid cell, in input order (the C++ std::set insertion rule)."""
    finite = torch.isfinite(points).all(dim=1)
    idx = torch.nonzero(finite).squeeze(1)
    if idx.numel() == 0:
        return idx
    return idx[first_per_key(voxel_keys(points[idx], cell))]


class EvidenceStore:
    """Session-local frames reduced to identity codes + ranges + pose, stacked on the GPU.

    Code per pixel, exactly as PhysicalEvidenceStore::ingest: invalid range -> INVALID;
    physical identity > 0 -> identity; otherwise a dynamic pixel or a pixel whose semantic
    label is an object/dynamic class -> UNIDENTIFIED (-1); otherwise BACKGROUND (0).
    """

    def __init__(self, object_semantics: Sequence[int] = (), dynamic_semantics: Sequence[int] = (),
                 max_range: float = 5.0):
        self.object_semantics = torch.as_tensor(sorted(set(object_semantics) | set(dynamic_semantics)),
                                                dtype=torch.int64, device=DEV)
        self.max_range = max_range
        self._stamps: List[int] = []
        self.K: Optional[Intrinsics] = None
        self.code = self.rng = self.T = self.R = None
        self.n = 0

    def _grow(self, h: int, w: int) -> None:
        cap = 0 if self.code is None else self.code.shape[0]
        if self.n < cap:
            return
        new = max(64, cap * 2)
        def grow(t, shape, dtype, fill=0):
            out = torch.full((new,) + shape, fill, dtype=dtype, device=DEV)
            if t is not None:
                out[:cap] = t
            return out
        self.code = grow(self.code, (h, w), torch.int32, INVALID_CODE)
        self.rng = grow(self.rng, (h, w), torch.int32)
        self.T = grow(self.T, (4, 4), torch.float32)
        self.R = grow(self.R, (3, 3), torch.float32)

    def ingest(self, frame: Frame, dynamic_mask=None) -> None:
        K = frame.K
        if self.K is None:
            self.K = K
        h, w = frame.depth.shape
        depth = to_dev(frame.depth)
        u = (torch.arange(w, device=DEV, dtype=torch.float32) + K.offset - K.cx) / K.fx
        v = (torch.arange(h, device=DEV, dtype=torch.float32) + K.offset - K.cy) / K.fy
        scale = torch.sqrt(u[None, :] ** 2 + v[:, None] ** 2 + 1.0)
        rng = depth * scale                                   # optical-Z depth -> range
        valid = torch.isfinite(rng) & (rng > 0) & (rng <= self.max_range)
        mm = torch.where(valid, torch.round(rng * 1000.0), torch.zeros_like(rng))
        valid &= (mm > 0) & (mm <= 65535)
        inst = to_dev(frame.instance, torch.int64)
        code = torch.full((h, w), INVALID_CODE, dtype=torch.int32, device=DEV)
        phys = valid & (inst > 0)
        code[phys] = inst[phys].to(torch.int32)
        rest = valid & ~phys
        unident = torch.zeros_like(rest)
        if dynamic_mask is not None:
            unident |= to_dev(dynamic_mask, torch.bool)
        if frame.semantic is not None and self.object_semantics.numel():
            unident |= torch.isin(to_dev(frame.semantic, torch.int64), self.object_semantics)
        code[rest & unident] = UNIDENTIFIED_CODE
        code[rest & ~unident] = BACKGROUND_CODE
        if self._stamps and frame.stamp_ns <= self._stamps[-1]:
            raise ValueError("frames must arrive in time order")
        self._grow(h, w)
        i = self.n
        self.code[i] = code
        self.rng[i] = torch.where(valid, mm, torch.zeros_like(mm)).to(torch.int32)
        T_world_cam = torch.as_tensor(frame.T_world_cam, dtype=torch.float64)
        self.T[i] = torch.linalg.inv(T_world_cam).to(DEV, torch.float32)
        self.R[i] = T_world_cam[:3, :3].to(DEV, torch.float32)
        self._stamps.append(int(frame.stamp_ns))
        self.n += 1

    # -- time index --------------------------------------------------------------------------
    def window(self, earliest: int, latest: int) -> Tuple[int, int]:
        """Index range [lo, hi) of stored frames with earliest <= stamp <= latest."""
        if earliest > latest or not self._stamps:
            return 0, 0
        lo = int(np.searchsorted(self._stamps, earliest, side="left"))
        hi = int(np.searchsorted(self._stamps, latest, side="right"))
        return lo, max(lo, hi)

    def stamps(self, earliest: int, latest: int) -> List[int]:
        lo, hi = self.window(earliest, latest)
        return self._stamps[lo:hi]

    def stamp_tensor(self, lo: int, hi: int) -> torch.Tensor:
        return torch.as_tensor(self._stamps[lo:hi], dtype=torch.int64, device=DEV)

    def first_stamp(self) -> Optional[int]:
        return self._stamps[0] if self._stamps else None

    # -- projection --------------------------------------------------------------------------
    def project(self, lo: int, hi: int, points: torch.Tensor):
        """Projection of N points into frames [lo, hi): dict of (F, N) tensors.

        etype, pid (int64), measured (float32 range, NaN unavailable), query (float32 range),
        pixel (int64 flat index, -1 outside), view (F, N, 3) unit sensor->point direction in world.
        """
        K = self.K
        T = self.T[lo:hi]
        cam = torch.einsum("fij,nj->fni", T[:, :3, :3], points) + T[:, None, :3, 3]
        z = cam[..., 2]
        uf = K.fx * cam[..., 0] / z + K.cx - K.offset + 0.5
        vf = K.fy * cam[..., 1] / z + K.cy - K.offset + 0.5
        inview = (z > 0) & torch.isfinite(uf) & torch.isfinite(vf)
        u = torch.where(inview, torch.floor(uf), torch.full_like(uf, -1)).to(torch.int64)
        v = torch.where(inview, torch.floor(vf), torch.full_like(vf, -1)).to(torch.int64)
        inview &= (u >= 0) & (u < K.width) & (v >= 0) & (v < K.height)
        F, N = z.shape
        fidx = torch.arange(lo, hi, device=DEV)[:, None].expand(F, N)
        uu, vv = u.clamp(0, K.width - 1), v.clamp(0, K.height - 1)
        code = self.code[fidx, vv, uu]
        mm = self.rng[fidx, vv, uu]
        etype = torch.full((F, N), UNAVAILABLE, dtype=torch.int8, device=DEV)
        et = torch.where(code == INVALID_CODE, INVALID,
                         torch.where(code == UNIDENTIFIED_CODE, UNIDENTIFIED,
                                     torch.where(code == BACKGROUND_CODE, BACKGROUND,
                                                 torch.where(code > 0, PHYSICAL, UNAVAILABLE)))).to(torch.int8)
        etype = torch.where(inview, et, etype)
        pid = torch.where(inview & (code > 0), code.to(torch.int64), torch.zeros_like(code, dtype=torch.int64))
        measured = torch.where(inview & (mm > 0), mm.to(torch.float32) / 1000.0,
                               torch.full_like(z, float("nan")))
        query = torch.linalg.norm(cam, dim=-1)
        view = torch.einsum("fij,fnj->fni", self.R[lo:hi], cam / query[..., None])
        pixel = torch.where(inview, v * K.width + u, torch.full_like(u, -1))
        return dict(etype=etype, pid=pid, measured=measured, query=query, pixel=pixel, view=view)


def classify(p: dict, physical_id: int, tolerance: float) -> torch.Tensor:
    """classifyMeasurement, vectorised over (frames, points)."""
    etype, pid, meas, query = p["etype"], p["pid"], p["measured"], p["query"]
    out = torch.full_like(etype, int(Vote.INVALID))
    unavailable = etype == UNAVAILABLE
    out[unavailable] = int(Vote.UNAVAILABLE)
    ok = (etype != INVALID) & torch.isfinite(meas) & torch.isfinite(query) & (meas > 0) & (query > 0)
    delta = meas - query
    todo = ~unavailable & ok
    same = (etype == PHYSICAL) & (pid > 0) & (pid == physical_id)
    occluded = todo & ((delta < -tolerance) | (~same & (delta < -1e-3)))
    out[occluded] = int(Vote.OCCLUDED)
    todo &= ~occluded
    free = todo & (delta > tolerance)
    out[free] = int(Vote.FREE)
    todo &= ~free
    out[todo & (etype == BACKGROUND)] = int(Vote.BACKGROUND)
    out[todo & (etype == UNIDENTIFIED)] = int(Vote.UNIDENTIFIED)
    out[todo & same] = int(Vote.SUPPORTED)
    out[todo & (etype == PHYSICAL) & ~same] = int(Vote.OTHER)
    return out


@dataclass
class SurfaceEvidence:
    """RayVerificator::SurfaceEvidenceCounts (the fields the state machine reads)."""
    surface_samples: int = 0
    support_rays: int = 0
    contradiction_rays: int = 0
    supported_votes: int = 0
    free_space_votes: int = 0
    replaced_by_other_votes: int = 0
    replaced_by_background_votes: int = 0
    occluded_votes: int = 0
    unobserved_samples: int = 0
    contradicted_surface_samples: int = 0
    latest_support_stamp: int = 0
    absence_coverage_sufficient: bool = False
    absence_llr: float = 0.0
    reliable_samples: int = 0
    reliable_in_view: int = 0
    reliable_seen_through: int = 0


def _latest(mask: torch.Tensor, stamps: torch.Tensor) -> torch.Tensor:
    """Per point, the latest stamp at which `mask` holds over the frames (0 if never)."""
    return torch.where(mask, stamps[:, None], torch.zeros_like(stamps)[:, None]).amax(dim=0)


def count_projected_surface(store: EvidenceStore, physical_id: int, points: torch.Tensor,
                            map_resolution: float, earliest: int, latest: int,
                            depth_tolerance: float, min_absent_surface_fraction: float,
                            cache: Optional[dict] = None) -> SurfaceEvidence:
    """countProjectedPhysicalSurface: one sample per map cell, every (frame, pixel) counted once.

    Every count is a sum, maximum or union over frames, so with `cache` (one dict per state) a
    call whose window start and samples equal the previous call's only projects the new frames;
    the result is identical to evaluating the whole window.
    """
    r = SurfaceEvidence()
    if not (map_resolution > 0) or len(points) == 0:
        return r
    samples = points[dedup_cells(points, map_resolution)]
    r.surface_samples = int(len(samples))
    lo, hi = store.window(earliest, latest)
    if hi <= lo or len(samples) == 0:
        r.unobserved_samples = r.surface_samples
        return r
    key = (physical_id, lo, len(samples), float(samples.double().sum()))
    a = cache.get("agg") if cache is not None and cache.get("key") == key and cache.get("hi", hi + 1) <= hi else None
    if a is None:
        n = len(samples)
        a = dict(coverage=torch.zeros(n, dtype=torch.bool, device=DEV),
                 sample_support=torch.zeros(n, dtype=torch.int64, device=DEV),
                 sample_absence=torch.zeros(n, dtype=torch.int64, device=DEV),
                 support_rays=0, contradiction_rays=0, latest_support_stamp=0, supported_votes=0,
                 free_space_votes=0, replaced_by_background_votes=0, replaced_by_other_votes=0,
                 occluded_votes=0)
        start = lo
    else:
        start = cache["hi"]
    if start < hi:
        stamps = store.stamp_tensor(start, hi)
        p = store.project(start, hi, samples)
        vote = classify(p, physical_id, depth_tolerance)
        a["coverage"] |= (vote != int(Vote.UNAVAILABLE)).any(dim=0)
        sup = vote == int(Vote.SUPPORTED)
        absent = (vote == int(Vote.FREE)) | (vote == int(Vote.BACKGROUND)) | (vote == int(Vote.OTHER))
        a["sample_support"] = torch.maximum(a["sample_support"], _latest(sup, stamps))
        a["sample_absence"] = torch.maximum(a["sample_absence"], _latest(absent, stamps))
        frame = torch.arange(hi - start, device=DEV)[:, None].expand_as(p["pixel"])
        keys = (frame << 32) | p["pixel"]
        a["support_rays"] += int(torch.unique(keys[sup]).numel())
        a["contradiction_rays"] += int(torch.unique(keys[absent]).numel())
        if sup.any():
            a["latest_support_stamp"] = max(a["latest_support_stamp"], int(stamps[sup.any(dim=1)].max()))
        a["supported_votes"] += int(sup.sum())
        a["free_space_votes"] += int((vote == int(Vote.FREE)).sum())
        a["replaced_by_background_votes"] += int((vote == int(Vote.BACKGROUND)).sum())
        a["replaced_by_other_votes"] += int((vote == int(Vote.OTHER)).sum())
        a["occluded_votes"] += int((vote == int(Vote.OCCLUDED)).sum())
    if cache is not None:
        cache.update(key=key, hi=hi, agg=a)
    r.support_rays = a["support_rays"]
    r.contradiction_rays = a["contradiction_rays"]
    r.latest_support_stamp = a["latest_support_stamp"] if a["latest_support_stamp"] else r.latest_support_stamp
    r.supported_votes = a["supported_votes"]
    r.free_space_votes = a["free_space_votes"]
    r.replaced_by_background_votes = a["replaced_by_background_votes"]
    r.replaced_by_other_votes = a["replaced_by_other_votes"]
    r.occluded_votes = a["occluded_votes"]
    r.unobserved_samples = int((~a["coverage"]).sum())
    r.contradicted_surface_samples = int((a["sample_absence"] > a["sample_support"]).sum())
    r.absence_coverage_sufficient = (r.surface_samples > 0 and r.contradicted_surface_samples > 0 and
                                     r.contradicted_surface_samples / r.surface_samples >=
                                     min_absent_surface_fraction)
    return r


def _robust_variance(values: Sequence[float], centre: float) -> float:
    """1.4826 * MAD, squared; centre < 0 means the median (upper median, as nth_element)."""
    if len(values) == 0:
        return 1e-4
    vals = np.asarray(values, dtype=np.float64)
    if centre < 0:
        centre = float(np.partition(vals, len(vals) // 2)[len(vals) // 2])
    dev = np.abs(vals - centre)
    mad = float(np.partition(dev, len(dev) // 2)[len(dev) // 2])
    return max(1e-4, (1.4826 * mad) ** 2)


@dataclass
class SensorStatistics:
    """Pooled look statistics of this sensor: the prior of a state without its own history.

    Process-global in t2 (file-static in projected_physical_evidence.cpp); exported next to
    the map as sensor_statistics.txt and loaded by the next session.
    """
    pooled_n: float = 0.0
    pooled_sum: float = 0.0
    pooled_label_n: float = 0.0
    pooled_label_sum: float = 0.0
    pooled_geo_dev: List[float] = field(default_factory=list)
    pooled_label_dev: List[float] = field(default_factory=list)
    loaded_geo_var: float = -1.0
    loaded_label_var: float = -1.0

    def save(self, path) -> None:
        gv = _robust_variance(self.pooled_geo_dev, -1.0) if len(self.pooled_geo_dev) >= 3 \
            else self.loaded_geo_var
        lv = _robust_variance(self.pooled_label_dev, -1.0) if len(self.pooled_label_dev) >= 3 \
            else self.loaded_label_var
        with open(path, "w") as f:
            f.write(" ".join(repr(float(x)) for x in (self.pooled_n, self.pooled_sum, gv,
                                                       self.pooled_label_n,
                                                       self.pooled_label_sum, lv)) + "\n")

    def load(self, path) -> bool:
        try:
            n, s, gv, ln, ls, lv = (float(x) for x in open(path).read().split()[:6])
        except (OSError, ValueError):
            return False
        self.pooled_n += n
        self.pooled_sum += s
        self.loaded_geo_var = gv
        self.pooled_label_n += ln
        self.pooled_label_sum += ls
        self.loaded_label_var = lv
        return True


class _Records:
    """Per-cell reliability records of one state (AbsenceSample), as GPU columns."""

    FIELDS = (("identity_hits", torch.int32), ("siwi", torch.bool), ("tentative_hits", torch.int32),
              ("tentative_veto", torch.bool), ("last_on_surface", torch.int64),
              ("last_seen_through", torch.int64), ("last_identity", torch.int64),
              ("last_foreign", torch.int64), ("counted", torch.bool))

    def __init__(self):
        self.keys = torch.zeros(0, dtype=torch.int64, device=DEV)     # insertion order
        self.sorted_keys = self.keys
        self.order = torch.zeros(0, dtype=torch.int64, device=DEV)
        for name, dtype in self.FIELDS:
            setattr(self, name, torch.zeros(0, dtype=dtype, device=DEV))

    def ensure(self, cell_keys: torch.Tensor) -> torch.Tensor:
        """Record index per cell key; unseen keys get new records (try_emplace)."""
        n0 = len(self.keys)
        pos = torch.searchsorted(self.sorted_keys, cell_keys)
        hit = (pos < n0) & (self.sorted_keys[pos.clamp(max=max(n0 - 1, 0))] == cell_keys) if n0 \
            else torch.zeros_like(cell_keys, dtype=torch.bool)
        idx = torch.full_like(cell_keys, -1)
        if n0:
            idx[hit] = self.order[pos[hit]]
        new = ~hit
        if new.any():
            fresh = torch.arange(n0, n0 + int(new.sum()), device=DEV)
            idx[new] = fresh
            self.keys = torch.cat([self.keys, cell_keys[new]])
            for name, dtype in self.FIELDS:
                setattr(self, name, torch.cat([getattr(self, name),
                                               torch.zeros(int(new.sum()), dtype=dtype, device=DEV)]))
            s = torch.argsort(self.keys)
            self.sorted_keys, self.order = self.keys[s], s
        return idx


@dataclass
class _AbsenceState:
    processed: int = 0
    ever_identified: bool = False
    history_n: float = 0.0
    history_sum: float = 0.0
    history_sq: float = 0.0
    label_n: float = 0.0
    label_sum: float = 0.0
    label_sq: float = 0.0
    geo_looks: List[float] = field(default_factory=list)
    label_looks: List[float] = field(default_factory=list)
    inherited: bool = False
    cusum: float = 0.0
    records: _Records = field(default_factory=_Records)


K_MIN_IDENTIFIED_SAMPLES = 3
K_MIN_IDENTITY_HITS = 3
K_MIN_SAMPLES_IN_VIEW = 30
K_MAX_ABSENCE_SAMPLES = 1500
K_NONE, K_IN_VIEW_ONLY, K_SEEN_THROUGH, K_ON_SURFACE = -2, -1, 0, 1
MAX_FRAMES_PER_PASS = 256


@dataclass
class EvidenceConfig:
    depth_tolerance: float = 0.3              # ray_verificator.depth_tolerance (room18 config)
    surface_match_tolerance: float = 0.05     # observed-absence sensor tolerance
    max_absence_incidence_deg: float = 60.0
    min_absent_surface_fraction: float = 0.2
    map_resolution: float = 0.05              # active_window.volumetric_map.voxel_size
    # Ablation of the absence decision on the same evidence stream:
    #   "cusum"  t2: object's own Beta history vs uniform, CUSUM, ln 99 (the method)
    #   "single" fix V37: one round with >= 20 % of the surface samples contradicted
    #   "none"   plain V37-like: no absence decision (only the new-site handoff closes)
    decision: str = "cusum"


def _log_present(f, n_, s_, q_):
    if n_ < 1:
        return 0.0, False
    m = min(0.999, max(0.001, s_ / n_))
    f = min(0.995, max(max(0.005, m), f))
    v = max(1e-4, q_ / n_ - (s_ / n_) ** 2)
    c = max(2.0, m * (1 - m) / v - 1)
    a, b = m * c + 1e-3, (1 - m) * c + 1e-3
    return (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) +
            (a - 1) * math.log(f) + (b - 1) * math.log(1 - f)), True


def _prior(n_, s_, own, pn, ps, pdev, loaded_var):
    v0 = _robust_variance(pdev, -1.0) if len(pdev) >= 3 else loaded_var
    pool_mean = ps / pn if pn >= 3 else 0.0
    have_pool = pn >= 3 and v0 > 0
    if not have_pool:
        pool_mean, v0 = 0.05, 0.09
    own_n = n_
    own_m = s_ / own_n if own_n > 0 else 0.0
    own_v = _robust_variance(own, own_m) if own_n >= 3 else 0.0
    m0, k = pool_mean, 3.0
    use_n = own_n if own_n >= 3 else 0.0
    m = (use_n * own_m + k * m0) / (use_n + k)
    v = max(v0, (use_n * (own_v + (own_m - m) ** 2) + k * (v0 + (m0 - m) ** 2)) / (use_n + k))
    return 1.0, m, m * m + v


class ObservedAbsence:
    """applyObservedAbsence and countCurrentPhysicalSurface with their per-state memory."""

    def __init__(self, store: EvidenceStore, config: EvidenceConfig, stats: SensorStatistics):
        self.store = store
        self.config = config
        self.stats = stats
        self.states: Dict[Tuple[int, int], _AbsenceState] = {}
        self._projected: Dict[Tuple[int, int], dict] = {}     # count_projected_surface caches

    def apply(self, physical_id: int, points: torch.Tensor, normals: Optional[torch.Tensor],
              earliest: int, latest: int, counts: SurfaceEvidence, slot: int,
              state_birth: int) -> None:
        cfg, stats = self.config, self.stats
        counts.absence_coverage_sufficient = False
        tolerance = cfg.surface_match_tolerance
        min_cos = math.cos(cfg.max_absence_incidence_deg * math.pi / 180.0)
        keep = dedup_cells(points, 0.5 * tolerance)
        q_pts = points[keep]
        cells = voxel_keys(q_pts, 0.5 * tolerance)
        if normals is not None:
            nrm = normals[keep]
            length = torch.linalg.norm(nrm, dim=1)
            has_normal = torch.isfinite(length) & (length > 1e-12)
            q_nrm = torch.where(has_normal[:, None], nrm / torch.where(has_normal, length, 1.0)[:, None],
                                torch.zeros_like(nrm))
        else:
            has_normal = torch.zeros(len(q_pts), dtype=torch.bool, device=DEV)
            q_nrm = torch.zeros_like(q_pts)
        if len(q_pts) > K_MAX_ABSENCE_SAMPLES:
            stride = len(q_pts) / K_MAX_ABSENCE_SAMPLES
            sel = torch.as_tensor((np.arange(K_MAX_ABSENCE_SAMPLES) * stride).astype(np.int64), device=DEV)
            q_pts, q_nrm, has_normal, cells = q_pts[sel], q_nrm[sel], has_normal[sel], cells[sel]

        state = self.states.setdefault((physical_id, slot), _AbsenceState())
        if latest < state.processed:            # a new session restarts time
            state.processed = 0
            state.ever_identified = False
            state.records = _Records()
        round_start = state.processed + 1
        first = self.store.first_stamp()
        state.inherited = first is not None and state_birth != 0 and state_birth < first
        rec = state.records
        idx = rec.ensure(cells)
        lo_all, hi_all = self.store.window(state.processed + 1, latest)
        for lo in range(lo_all, hi_all, MAX_FRAMES_PER_PASS):
            hi = min(hi_all, lo + MAX_FRAMES_PER_PASS)
            stamps = self.store.stamp_tensor(lo, hi)
            p = self.store.project(lo, hi, q_pts)
            avail = p["etype"] != UNAVAILABLE
            facing = ~has_normal[None, :] | (torch.abs((q_nrm[None] * p["view"]).sum(-1)) >= min_cos)
            meas, query = p["measured"], p["query"]
            measured = avail & (p["etype"] != INVALID) & torch.isfinite(meas) & (meas > 0) & \
                torch.isfinite(query) & (query > 0)
            delta = meas - query
            on = measured & (torch.abs(delta) <= tolerance)
            through = measured & ~on & facing & (delta > tolerance)
            phys = on & (p["etype"] == PHYSICAL) & (p["pid"] > 0)
            identified = phys & (p["pid"] == physical_id)
            foreign = phys & (p["pid"] != physical_id)
            ident_n = identified.sum(dim=1)
            through_n = through.sum(dim=1)
            object_identified = (ident_n >= K_MIN_IDENTIFIED_SAMPLES) & (ident_n > through_n)
            if bool(object_identified.any()):
                state.ever_identified = True
            # judged = on surface or seen through (occluded / in-view-only / none are not verdicts)
            rec.last_on_surface[idx] = torch.maximum(rec.last_on_surface[idx], _latest(on, stamps))
            rec.last_identity[idx] = torch.maximum(rec.last_identity[idx], _latest(identified, stamps))
            rec.last_foreign[idx] = torch.maximum(rec.last_foreign[idx], _latest(foreign, stamps))
            rec.last_seen_through[idx] = torch.maximum(rec.last_seen_through[idx], _latest(through, stamps))
            oi = object_identified[:, None]
            hits = (oi & identified).sum(dim=0).to(torch.int32)
            rec.tentative_hits[idx] = torch.clamp(rec.tentative_hits[idx] + hits, max=65535)
            rec.tentative_veto[idx] |= (oi & (through | foreign)).any(dim=0)
            state.processed = int(stamps[-1])
        state.processed = max(state.processed, latest)

        # One look = this reconciliation round; verdicts are the latest per sample in the round.
        last_id = rec.last_identity[idx]
        own_identity = int(((last_id >= round_start) & (last_id != 0)).sum())
        weak = (~torch.tensor(state.inherited)) & torch.tensor(state.ever_identified) & \
            ((rec.identity_hits[idx] + rec.tentative_hits[idx]) < K_MIN_IDENTITY_HITS)
        eligible = ~rec.siwi[idx] & ~weak.to(DEV)
        counts.reliable_samples = int(eligible.sum())
        last_on, last_th = rec.last_on_surface[idx], rec.last_seen_through[idx]
        last = torch.maximum(last_on, last_th)
        in_view = eligible & (last >= round_start) & (last != 0)
        counts.reliable_in_view = int(in_view.sum())
        seen = in_view & (last_th > last_on)
        on_mask = in_view & ~seen
        seen_through = int(seen.sum())
        on_surface = int(on_mask.sum())
        foreign_on_surface = int((on_mask & (rec.last_foreign[idx] == last_on) & (last_id < last_on)).sum())
        fresh_mask = in_view & ~rec.counted[idx]
        fresh = int(fresh_mask.sum())
        counts.reliable_seen_through = seen_through
        verdicts = on_surface + seen_through
        needed = min(K_MIN_SAMPLES_IN_VIEW, counts.reliable_samples)
        identified_in_place = own_identity >= K_MIN_IDENTIFIED_SAMPLES and own_identity > seen_through

        if verdicts > 0 and verdicts >= needed:
            f_geo = seen_through / verdicts
            f_lab = foreign_on_surface / on_surface if on_surface > 0 else 0.0
            gn, gs, gq = _prior(state.history_n, state.history_sum, state.geo_looks,
                                stats.pooled_n, stats.pooled_sum, stats.pooled_geo_dev, stats.loaded_geo_var)
            lp_geo, ok_geo = _log_present(f_geo, gn, gs, gq)
            # Label disagreement is kept as a statistic only (ok_lab = false in t2).
            if ok_geo:
                weight = min(1.0, fresh / max(1, counts.reliable_samples))
                state.cusum = max(0.0, state.cusum - weight * lp_geo)
                rec.counted[idx[fresh_mask]] = True
                if state.cusum == 0.0:
                    rec.counted[:] = False
            if identified_in_place:
                state.history_n += 1
                state.history_sum += f_geo
                state.history_sq += f_geo * f_geo
                if len(state.geo_looks) < 256:
                    state.geo_looks.append(f_geo)
                if state.history_n == 3 and len(stats.pooled_geo_dev) < 4096:
                    stats.pooled_geo_dev.append(state.history_sum / 3)
                    stats.pooled_n += 1
                    stats.pooled_sum += state.history_sum / 3
                if on_surface >= K_MIN_IDENTIFIED_SAMPLES:
                    state.label_n += 1
                    state.label_sum += f_lab
                    state.label_sq += f_lab * f_lab
                    if len(state.label_looks) < 256:
                        state.label_looks.append(f_lab)
                    if state.label_n == 3 and len(stats.pooled_label_dev) < 4096:
                        stats.pooled_label_dev.append(state.label_sum / 3)
                        stats.pooled_label_n += 1
                        stats.pooled_label_sum += state.label_sum / 3
        if state.cusum == 0.0:
            rec.counted[:] = False
        counts.absence_llr = float(state.cusum)
        counts.absence_coverage_sufficient = state.cusum > math.log(99.0)
        if identified_in_place:
            rec.identity_hits = torch.clamp(rec.identity_hits + rec.tentative_hits, max=65535)
            rec.siwi = rec.siwi | rec.tentative_veto
        rec.tentative_hits.zero_()
        rec.tentative_veto.zero_()
        if counts.absence_coverage_sufficient:
            state.cusum = 0.0                   # the state ends; a successor starts clean

    def measure_state(self, physical_id: int, points: torch.Tensor, normals: Optional[torch.Tensor],
                      last_support: int, latest: int, slot: int, state_birth: int
                      ) -> SurfaceEvidence:
        cfg = self.config
        if last_support >= latest:
            none = SurfaceEvidence()
            self.apply(physical_id, points, normals, latest, latest, none, slot, state_birth)
            none.absence_coverage_sufficient = False
            return none
        earliest = last_support + 1
        measured = count_projected_surface(self.store, physical_id, points, cfg.map_resolution,
                                           earliest, latest, cfg.depth_tolerance,
                                           cfg.min_absent_surface_fraction,
                                           cache=self._projected.setdefault((physical_id, slot), {}))
        if measured.support_rays or measured.contradiction_rays:
            single = measured.absence_coverage_sufficient
            self.apply(physical_id, points, normals, earliest, latest, measured, slot, state_birth)
            self._decide(measured, single)
            return measured
        counts = SurfaceEvidence()      # t2 returns the (empty) proxy counts in this branch
        self.apply(physical_id, points, normals, earliest, latest, counts, slot, state_birth)
        self._decide(counts, False)
        return counts

    def _decide(self, counts: SurfaceEvidence, single: bool) -> None:
        if self.config.decision == "single":
            counts.absence_coverage_sufficient = single
        elif self.config.decision == "none":
            counts.absence_coverage_sufficient = False
