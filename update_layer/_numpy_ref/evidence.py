"""Projected physical evidence and the observed-absence test.

Representation-free port of the t2 decision code (commit 43c663d):
  khronos/src/backend/change_detection/physical_evidence_store.cpp   -> EvidenceStore
  khronos/src/backend/change_detection/projected_physical_evidence.cpp
      classifyMeasurement             -> classify()
      countProjectedPhysicalSurface   -> count_projected_surface()
      applyObservedAbsence            -> ObservedAbsence.apply()
      countCurrentPhysicalSurface     -> ObservedAbsence.measure_state()
      save/loadAbsenceSensorStatistics-> SensorStatistics.save()/load()

The only input about a stored state is a set of surface samples with optional
normals, in world coordinates. How a backend produces them (mesh face centroids,
points, surfels, Gaussians) is the adapter's business. Rules, tolerances and the
single decision constant ln(99) are unchanged. One deliberate difference: t2 first
consults Khronos' mesh-ray index (RayVerificator::countPhysicalSurface) and uses
the projected pixels only when that proxy proposes absence or has no rays; there
is no mesh-ray index here, so the projected pixels always decide (the t2 fallback
branch).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .frames import Frame, Intrinsics

INVALID_CODE = np.iinfo(np.int32).min
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


@dataclass
class StoredFrame:
    stamp: int
    code: np.ndarray          # int32 HxW identity code per pixel
    range_mm: np.ndarray      # uint16 HxW measured range in mm, 0 = invalid
    T_cam_world: np.ndarray   # 4x4
    R_world_cam: np.ndarray   # 3x3
    K: Intrinsics


@dataclass
class Projection:
    etype: np.ndarray         # endpoint class per query point
    physical_id: np.ndarray
    measured: np.ndarray      # measured range at the pixel (m), NaN if unavailable
    query: np.ndarray         # range of the queried point (m)
    pixel: np.ndarray         # flattened pixel index, -1 if not in view
    view_dir: np.ndarray      # Nx3 unit sensor->point direction, world frame


class EvidenceStore:
    """Session-local frames reduced to identity codes + ranges + pose (no images kept).

    Code per pixel, exactly as PhysicalEvidenceStore::ingest: invalid range -> INVALID;
    physical identity > 0 -> identity; otherwise a dynamic pixel or a pixel whose semantic
    label is an object/dynamic class -> UNIDENTIFIED (-1); otherwise BACKGROUND (0).
    """

    def __init__(self, object_semantics: Sequence[int] = (), dynamic_semantics: Sequence[int] = (),
                 max_range: float = 5.0):
        self.frames: Dict[int, StoredFrame] = {}
        self._sorted: List[int] = []
        self.object_semantics = np.asarray(sorted(set(object_semantics) | set(dynamic_semantics)),
                                           dtype=np.int64)
        self.max_range = max_range

    def ingest(self, frame: Frame, dynamic_mask: Optional[np.ndarray] = None) -> None:
        K = frame.K
        h, w = frame.depth.shape
        u = (np.arange(w) + K.offset - K.cx) / K.fx
        v = (np.arange(h) + K.offset - K.cy) / K.fy
        scale = np.sqrt(u[None, :] ** 2 + v[:, None] ** 2 + 1.0)
        rng = frame.depth * scale                      # optical-Z depth -> range along the ray
        valid = np.isfinite(rng) & (rng > 0) & (rng <= self.max_range)
        mm = np.where(valid, np.rint(rng * 1000.0), 0)
        valid &= (mm > 0) & (mm <= 65535)
        code = np.full((h, w), INVALID_CODE, dtype=np.int32)
        inst = frame.instance
        phys = valid & (inst > 0)
        code[phys] = inst[phys]
        rest = valid & ~phys
        unident = np.zeros_like(rest)
        if dynamic_mask is not None:
            unident |= dynamic_mask
        if frame.semantic is not None and self.object_semantics.size:
            unident |= np.isin(frame.semantic, self.object_semantics)
        code[rest & unident] = UNIDENTIFIED_CODE
        code[rest & ~unident] = BACKGROUND_CODE
        T_cam_world = np.linalg.inv(frame.T_world_cam)
        self.frames[frame.stamp_ns] = StoredFrame(
            frame.stamp_ns, code, np.where(valid, mm, 0).astype(np.uint16), T_cam_world,
            frame.T_world_cam[:3, :3].copy(), K)
        self._sorted = sorted(self.frames)

    def stamps(self, earliest: int, latest: int) -> List[int]:
        if earliest > latest or not self._sorted:
            return []
        lo = np.searchsorted(self._sorted, earliest, side="left")
        hi = np.searchsorted(self._sorted, latest, side="right")
        return self._sorted[lo:hi]

    def first_stamp(self) -> Optional[int]:
        return self._sorted[0] if self._sorted else None

    def project(self, stamp: int, points: np.ndarray) -> Projection:
        n = len(points)
        f = self.frames[stamp]
        cam = points @ f.T_cam_world[:3, :3].T + f.T_cam_world[:3, 3]
        u, v, _ = f.K.project(cam)
        inview = u >= 0
        etype = np.full(n, UNAVAILABLE, dtype=np.int8)
        pid = np.zeros(n, dtype=np.int64)
        measured = np.full(n, np.nan, dtype=np.float64)
        pixel = np.full(n, -1, dtype=np.int64)
        query = np.linalg.norm(cam, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            view_dir = (cam / query[:, None]) @ f.R_world_cam.T
        if inview.any():
            uu, vv = u[inview], v[inview]
            code = f.code[vv, uu]
            mm = f.range_mm[vv, uu]
            meas = np.where(mm > 0, mm.astype(np.float64) / 1000.0, np.nan)
            t = np.where(code == INVALID_CODE, INVALID,
                         np.where(code == UNIDENTIFIED_CODE, UNIDENTIFIED,
                                  np.where(code == BACKGROUND_CODE, BACKGROUND,
                                           np.where(code > 0, PHYSICAL, UNAVAILABLE))))
            etype[inview] = t
            pid[inview] = np.where(code > 0, code, 0)
            measured[inview] = meas
            pixel[inview] = vv * f.K.width + uu
        return Projection(etype, pid, measured, query, pixel, view_dir)


def classify(p: Projection, physical_id: int, tolerance: float) -> np.ndarray:
    """classifyMeasurement, vectorised over query points."""
    n = len(p.etype)
    out = np.full(n, Vote.INVALID, dtype=np.int8)
    unavailable = p.etype == UNAVAILABLE
    out[unavailable] = Vote.UNAVAILABLE
    with np.errstate(invalid="ignore"):
        measured_ok = (p.etype != INVALID) & np.isfinite(p.measured) & np.isfinite(p.query) & \
            (p.measured > 0) & (p.query > 0)
        delta = p.measured - p.query
    todo = ~unavailable & measured_ok
    same = (p.etype == PHYSICAL) & (p.physical_id > 0) & (p.physical_id == physical_id)
    occluded = todo & ((delta < -tolerance) | (~same & (delta < -1e-3)))
    out[occluded] = Vote.OCCLUDED
    todo &= ~occluded
    free = todo & (delta > tolerance)
    out[free] = Vote.FREE
    todo &= ~free
    out[todo & (p.etype == BACKGROUND)] = Vote.BACKGROUND
    out[todo & (p.etype == UNIDENTIFIED)] = Vote.UNIDENTIFIED
    out[todo & same] = Vote.SUPPORTED
    out[todo & (p.etype == PHYSICAL) & ~same] = Vote.OTHER
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


def _dedup_cells(points: np.ndarray, cell: float) -> Tuple[np.ndarray, np.ndarray]:
    """First point per grid cell, in input order; returns (indices, integer cells)."""
    finite = np.isfinite(points).all(axis=1)
    idx = np.nonzero(finite)[0]
    keys = np.floor(points[idx] / cell).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    first = np.sort(first)
    return idx[first], keys[first]


def count_projected_surface(store: EvidenceStore, physical_id: int, points: np.ndarray,
                            map_resolution: float, earliest: int, latest: int,
                            depth_tolerance: float, min_absent_surface_fraction: float
                            ) -> SurfaceEvidence:
    """countProjectedPhysicalSurface: one sample per map cell, every (frame, pixel) counted once."""
    r = SurfaceEvidence()
    if not (map_resolution > 0) or len(points) == 0:
        return r
    keep, _ = _dedup_cells(points, map_resolution)
    samples = points[keep]
    r.surface_samples = len(samples)
    stamps = store.stamps(earliest, latest)
    coverage = np.zeros(len(samples), dtype=bool)
    sample_support = np.zeros(len(samples), dtype=np.int64)
    sample_absence = np.zeros(len(samples), dtype=np.int64)
    support_keys, contra_keys = set(), set()
    for k, stamp in enumerate(stamps):
        p = store.project(stamp, samples)
        vote = classify(p, physical_id, depth_tolerance)
        seen = vote != Vote.UNAVAILABLE
        coverage |= seen
        sup = vote == Vote.SUPPORTED
        if sup.any():
            sample_support[sup] = stamp
            r.latest_support_stamp = max(r.latest_support_stamp, stamp)
            support_keys.update(((k << 32) | p.pixel[sup]).tolist())
            r.supported_votes += int(sup.sum())
        for kind, attr in ((Vote.FREE, "free_space_votes"),
                           (Vote.BACKGROUND, "replaced_by_background_votes"),
                           (Vote.OTHER, "replaced_by_other_votes")):
            m = vote == kind
            if m.any():
                sample_absence[m] = stamp
                contra_keys.update(((k << 32) | p.pixel[m]).tolist())
                setattr(r, attr, getattr(r, attr) + int(m.sum()))
        r.occluded_votes += int((vote == Vote.OCCLUDED).sum())
    r.unobserved_samples = int((~coverage).sum())
    r.contradicted_surface_samples = int((sample_absence > sample_support).sum())
    r.absence_coverage_sufficient = (r.surface_samples > 0 and r.contradicted_surface_samples > 0 and
                                     r.contradicted_surface_samples / r.surface_samples >=
                                     min_absent_surface_fraction)
    r.support_rays = len(support_keys)
    r.contradiction_rays = len(contra_keys)
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


@dataclass
class _Samples:
    """Per-cell reliability record of one state (AbsenceSample), stored as columns."""
    index: Dict[Tuple[int, int, int], int] = field(default_factory=dict)
    identity_hits: List[int] = field(default_factory=list)
    seen_through_while_identified: List[bool] = field(default_factory=list)
    tentative_hits: List[int] = field(default_factory=list)
    tentative_veto: List[bool] = field(default_factory=list)
    last_on_surface: List[int] = field(default_factory=list)
    last_seen_through: List[int] = field(default_factory=list)
    last_identity: List[int] = field(default_factory=list)
    last_foreign: List[int] = field(default_factory=list)
    counted: List[bool] = field(default_factory=list)

    def ensure(self, cells: np.ndarray) -> np.ndarray:
        out = np.empty(len(cells), dtype=np.int64)
        for i, c in enumerate(map(tuple, cells.tolist())):
            j = self.index.get(c)
            if j is None:
                j = len(self.identity_hits)
                self.index[c] = j
                self.identity_hits.append(0)
                self.seen_through_while_identified.append(False)
                self.tentative_hits.append(0)
                self.tentative_veto.append(False)
                self.last_on_surface.append(0)
                self.last_seen_through.append(0)
                self.last_identity.append(0)
                self.last_foreign.append(0)
                self.counted.append(False)
            out[i] = j
        return out

    def arrays(self):
        return {k: np.asarray(getattr(self, k)) for k in (
            "identity_hits", "seen_through_while_identified", "tentative_hits", "tentative_veto",
            "last_on_surface", "last_seen_through", "last_identity", "last_foreign", "counted")}

    def store(self, a) -> None:
        for k, v in a.items():
            setattr(self, k, v.tolist())


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
    samples: _Samples = field(default_factory=_Samples)


K_MIN_IDENTIFIED_SAMPLES = 3
K_MIN_IDENTITY_HITS = 3
K_MIN_SAMPLES_IN_VIEW = 30
K_MAX_ABSENCE_SAMPLES = 1500
K_NONE, K_IN_VIEW_ONLY, K_SEEN_THROUGH, K_ON_SURFACE = -2, -1, 0, 1


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


class ObservedAbsence:
    """applyObservedAbsence and countCurrentPhysicalSurface with their per-state memory."""

    def __init__(self, store: EvidenceStore, config: EvidenceConfig, stats: SensorStatistics):
        self.store = store
        self.config = config
        self.stats = stats
        self.states: Dict[Tuple[int, int], _AbsenceState] = {}

    # -- applyObservedAbsence ------------------------------------------------------------
    def apply(self, physical_id: int, points: np.ndarray, normals: Optional[np.ndarray],
              earliest: int, latest: int, counts: SurfaceEvidence, slot: int,
              state_birth: int) -> None:
        cfg, stats = self.config, self.stats
        counts.absence_coverage_sufficient = False
        tolerance = cfg.surface_match_tolerance
        min_cos = math.cos(cfg.max_absence_incidence_deg * math.pi / 180.0)
        cell_size = 0.5 * tolerance
        keep, cells = _dedup_cells(points, cell_size)
        q_pts = points[keep]
        if normals is not None:
            nrm = normals[keep].astype(np.float64)
            length = np.linalg.norm(nrm, axis=1)
            has_normal = np.isfinite(length) & (length > 1e-12)
            q_nrm = np.where(has_normal[:, None], nrm / np.where(has_normal, length, 1.0)[:, None], 0.0)
        else:
            has_normal = np.zeros(len(q_pts), dtype=bool)
            q_nrm = np.zeros_like(q_pts)
        if len(q_pts) > K_MAX_ABSENCE_SAMPLES:
            stride = len(q_pts) / K_MAX_ABSENCE_SAMPLES
            sel = (np.arange(K_MAX_ABSENCE_SAMPLES) * stride).astype(np.int64)
            q_pts, q_nrm, has_normal, cells = q_pts[sel], q_nrm[sel], has_normal[sel], cells[sel]

        state = self.states.setdefault((physical_id, slot), _AbsenceState())
        if latest < state.processed:            # a new session restarts time
            state.processed = 0
            state.ever_identified = False
            state.samples = _Samples()
        round_start = state.processed + 1
        first = self.store.first_stamp()
        state.inherited = first is not None and state_birth != 0 and state_birth < first
        idx = state.samples.ensure(cells)
        S = state.samples.arrays()
        n = len(q_pts)
        for stamp in self.store.stamps(state.processed + 1, latest):
            p = self.store.project(stamp, q_pts)
            avail = p.etype != UNAVAILABLE
            facing = ~has_normal | (np.abs(np.einsum("ij,ij->i", q_nrm, p.view_dir)) >= min_cos)
            with np.errstate(invalid="ignore"):
                measured = avail & (p.etype != INVALID) & np.isfinite(p.measured) & \
                    (p.measured > 0) & np.isfinite(p.query) & (p.query > 0)
                delta = p.measured - p.query
            observed = np.full(n, K_NONE, dtype=np.int8)
            observed[avail & ~measured & facing] = K_IN_VIEW_ONLY
            with np.errstate(invalid="ignore"):
                on = measured & (np.abs(delta) <= tolerance)
                off = measured & ~on & facing
                observed[on] = K_ON_SURFACE
                observed[off] = np.where(delta[off] > tolerance, K_SEEN_THROUGH, K_IN_VIEW_ONLY)
            phys = on & (p.etype == PHYSICAL) & (p.physical_id > 0)
            identified = phys & (p.physical_id == physical_id)
            foreign = phys & (p.physical_id != physical_id)
            identified_samples = int(identified.sum())
            seen_through_samples = int((observed == K_SEEN_THROUGH).sum())
            object_identified = identified_samples >= K_MIN_IDENTIFIED_SAMPLES and \
                identified_samples > seen_through_samples
            if object_identified:
                state.ever_identified = True
            judged = (observed != K_NONE) & (observed != K_IN_VIEW_ONLY)
            j = idx[judged]
            S["last_on_surface"][j[observed[judged] == K_ON_SURFACE]] = stamp
            S["last_identity"][idx[judged & identified]] = stamp
            S["last_foreign"][idx[judged & foreign]] = stamp
            S["last_seen_through"][idx[judged & (observed == K_SEEN_THROUGH)]] = stamp
            if object_identified:
                hit = idx[judged & identified]
                S["tentative_hits"][hit] = np.minimum(S["tentative_hits"][hit] + 1, 65535)
                S["tentative_veto"][idx[judged & ((observed == K_SEEN_THROUGH) | foreign)]] = True
            state.processed = stamp
        state.processed = max(state.processed, latest)

        # One look = this reconciliation round; verdicts are the latest per sample in the round.
        on_surface = seen_through = foreign_on_surface = own_identity = fresh = 0
        fresh_cells = []
        for i in range(n):
            j = idx[i]
            last_id = S["last_identity"][j]
            if last_id >= round_start and last_id != 0:
                own_identity += 1
            if S["seen_through_while_identified"][j]:
                continue
            if (not state.inherited and state.ever_identified and
                    int(S["identity_hits"][j]) + int(S["tentative_hits"][j]) < K_MIN_IDENTITY_HITS):
                continue
            counts.reliable_samples += 1
            last = max(S["last_on_surface"][j], S["last_seen_through"][j])
            if last < round_start or last == 0:
                continue
            counts.reliable_in_view += 1
            if S["last_seen_through"][j] > S["last_on_surface"][j]:
                seen_through += 1
            else:
                on_surface += 1
                if S["last_foreign"][j] == S["last_on_surface"][j] and \
                        S["last_identity"][j] < S["last_on_surface"][j]:
                    foreign_on_surface += 1
            if not S["counted"][j]:
                fresh += 1
                fresh_cells.append(j)
        counts.reliable_seen_through = seen_through
        verdicts = on_surface + seen_through
        needed = min(K_MIN_SAMPLES_IN_VIEW, counts.reliable_samples)
        identified_in_place = own_identity >= K_MIN_IDENTIFIED_SAMPLES and own_identity > seen_through

        def log_present(f, n_, s_, q_):
            if n_ < 1:
                return 0.0, False
            m = min(0.999, max(0.001, s_ / n_))
            f = min(0.995, max(max(0.005, m), f))
            v = max(1e-4, q_ / n_ - (s_ / n_) ** 2)
            c = max(2.0, m * (1 - m) / v - 1)
            a, b = m * c + 1e-3, (1 - m) * c + 1e-3
            return (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) +
                    (a - 1) * math.log(f) + (b - 1) * math.log(1 - f)), True

        def prior(n_, s_, q_, own, pn, ps, pdev, loaded_var):
            v0 = _robust_variance(pdev, -1.0) if len(pdev) >= 3 else loaded_var
            pool_mean = ps / pn if pn >= 3 else 0.0
            have_pool = pn >= 3 and v0 > 0
            if not have_pool:
                pool_mean, v0, have_pool = 0.05, 0.09, True
            own_n = n_
            own_m = s_ / own_n if own_n > 0 else 0.0
            own_v = _robust_variance(own, own_m) if own_n >= 3 else 0.0
            m0, k = pool_mean, 3.0
            use_n = own_n if own_n >= 3 else 0.0
            m = (use_n * own_m + k * m0) / (use_n + k)
            v = max(v0, (use_n * (own_v + (own_m - m) ** 2) + k * (v0 + (m0 - m) ** 2)) / (use_n + k))
            return 1.0, m, m * m + v

        if verdicts > 0 and verdicts >= needed:
            f_geo = seen_through / verdicts
            f_lab = foreign_on_surface / on_surface if on_surface > 0 else 0.0
            gn, gs, gq = prior(state.history_n, state.history_sum, state.history_sq, state.geo_looks,
                               stats.pooled_n, stats.pooled_sum, stats.pooled_geo_dev,
                               stats.loaded_geo_var)
            lp_geo, ok_geo = log_present(f_geo, gn, gs, gq)
            # Label disagreement is kept as a statistic only (ok_lab = false in t2).
            if ok_geo:
                weight = min(1.0, fresh / max(1, counts.reliable_samples))
                state.cusum = max(0.0, state.cusum - weight * lp_geo)
                for j in fresh_cells:
                    S["counted"][j] = True
                if state.cusum == 0.0:
                    S["counted"][:] = False
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
            S["counted"][:] = False
        counts.absence_llr = float(state.cusum)
        counts.absence_coverage_sufficient = state.cusum > math.log(99.0)
        if identified_in_place:
            S["identity_hits"] = np.minimum(S["identity_hits"].astype(np.int64) + S["tentative_hits"], 65535)
            S["seen_through_while_identified"] = S["seen_through_while_identified"] | S["tentative_veto"]
        S["tentative_hits"] = np.zeros_like(S["tentative_hits"])
        S["tentative_veto"] = np.zeros_like(S["tentative_veto"])
        state.samples.store(S)
        if counts.absence_coverage_sufficient:
            state.cusum = 0.0                   # the state ends; a successor starts clean

    # -- countCurrentPhysicalSurface -----------------------------------------------------
    def measure_state(self, physical_id: int, points: np.ndarray, normals: Optional[np.ndarray],
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
                                           cfg.min_absent_surface_fraction)
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
