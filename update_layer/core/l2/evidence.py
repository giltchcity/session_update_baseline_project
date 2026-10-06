"""Observed absence and projected physical evidence: session_core/src/evidence/projected_physical_evidence.cpp
@192c1cf (L2_FINAL2), ported function by function. Line numbers in brackets refer to that file.

Representation: a state's surface is a point set with normals (the layer's fragments) instead of a triangle mesh;
the C++ already handles a face-less mesh (its points, no normal) and otherwise uses face centroids with face
normals, so a point with a finite normal plays the role of a face centroid. Frames are the layer's
EvidenceStore (core/evidence.py), the same content as PhysicalEvidenceStore: per pixel the range in millimetres
and the identity code, projected with the radial query range.

Global C++ state (absence_states, the pooled population, the cell model) belongs to one process = one session;
here it is one AbsenceModel per session, with the population carried by save/load_sensor_statistics as the C++
carries it in sensor_statistics.txt.

Configuration (ray_verificator.h @192c1cf: surface_match_tolerance 0.05, max_absence_incidence_deg 60,
min_absent_surface_fraction 0.2; room18_instance_5cm.yaml:226 depth_tolerance 0.3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..evidence import BACKGROUND, DEV, INVALID, PHYSICAL, UNAVAILABLE, UNIDENTIFIED
from . import ieee

LOG99 = ieee.log(99.0)

# [161-173] smallest sample whose median has a positive breakdown point
def _minimal_robust_sample() -> int:
    n = 1
    while (n - 1) // 2 == 0:
        n += 1
    return n


K_ROBUST_LOOKS = _minimal_robust_sample()          # 3
K_SHARE_VARIANCE_FLOOR = 1e-4                       # [175-185]
K_MIN_SAMPLES_IN_VIEW = 30                          # [297-315]
K_MAX_ABSENCE_SAMPLES = 1500                        # [316-320]
K_COLD_START_SEEN_THROUGH_RATE = 0.05               # [539-550]
K_UNIFORM_SHARE_VARIANCE = 1.0 / 12.0
K_UNIFORM_PRIOR_PSEUDO_COUNT = 2.0                  # [552-554]

# votes [66-67]
V_UNAVAILABLE, V_INVALID, V_OCCLUDED, V_SUPPORTED, V_FREE, V_BACKGROUND, V_OTHER, V_UNIDENTIFIED = range(8)
# verdicts [322]
K_NONE, K_IN_VIEW_ONLY, K_SEEN_THROUGH, K_ON_SURFACE = -2, -1, 0, 1


@dataclass
class EvidenceConfig:
    surface_match_tolerance: float = 0.05
    max_absence_incidence_deg: float = 60.0
    depth_tolerance: float = 0.3
    min_absent_surface_fraction: float = 0.2


# ------------------------------------------------------------------ classification [20-86]
def classify_measurements(p: Dict[str, torch.Tensor], physical_id: int, tolerance: float) -> torch.Tensor:
    """classifyMeasurement for every (frame, point) of a projection: equal-loss argmax of the relation
    probabilities with the original tie order (Supported first, then Background, Other, Unidentified,
    Invalid, Unavailable, Occluded, Free, each replacing only when strictly larger)."""
    etype, pid, meas, query = p["etype"], p["pid"], p["measured"], p["query"]
    quantum = 1.0 / 1000.0
    unavailable = etype == UNAVAILABLE
    invalid = (etype == INVALID) | ~torch.isfinite(meas) | ~torch.isfinite(query) | (meas <= 0) | (query <= 0)
    delta = (meas - query).to(torch.float64)                    # const float delta
    same = (etype == PHYSICAL) & (pid > 0) & (pid == physical_id)
    tol = float(np.float32(tolerance))                          # the float parameter
    near = torch.where(same, torch.full_like(delta, -tol),
                       torch.full_like(delta, -float(np.float32(min(tol, float(np.float32(quantum)))))))
    cdf = lambda b: torch.clamp((b - delta) / quantum + 0.5, 0.0, 1.0)
    occluded = cdf(near)
    through_upper = cdf(torch.full_like(delta, tol))
    free = 1.0 - through_upper
    on = through_upper - occluded
    z = torch.zeros_like(delta)
    supported = torch.where(same, on, z)
    background = torch.where(~same & (etype == BACKGROUND), on, z)
    unidentified = torch.where(~same & (etype == UNIDENTIFIED), on, z)
    other = torch.where(~same & (etype == PHYSICAL), on, z)
    inval = torch.where(~same & (etype != BACKGROUND) & (etype != UNIDENTIFIED) & (etype != PHYSICAL), on, z)
    # unavailable / invalid short-cuts of projectedRelationProbabilities
    one = torch.ones_like(delta)
    occluded, free = torch.where(invalid | unavailable, z, occluded), torch.where(invalid | unavailable, z, free)
    supported = torch.where(invalid | unavailable, z, supported)
    background = torch.where(invalid | unavailable, z, background)
    other = torch.where(invalid | unavailable, z, other)
    unidentified = torch.where(invalid | unavailable, z, unidentified)
    inval = torch.where(unavailable, z, torch.where(invalid, one, inval))
    unav = torch.where(unavailable, one, z)
    vote = torch.full(delta.shape, V_SUPPORTED, dtype=torch.int8, device=delta.device)
    best = supported.clone()
    for cand, prob in ((V_BACKGROUND, background), (V_OTHER, other), (V_UNIDENTIFIED, unidentified),
                       (V_INVALID, inval), (V_UNAVAILABLE, unav), (V_OCCLUDED, occluded), (V_FREE, free)):
        better = prob > best
        vote = torch.where(better, torch.full_like(vote, cand), vote)
        best = torch.where(better, prob, best)
    return vote


# ------------------------------------------------------------------ per-state record [98-146]
class ObjectAbsenceState:
    """Per state: samples keyed by their 2.5 cm cell, the in-place history and Page's statistic."""

    def __init__(self):
        self.processed = 0
        self.ever_identified = False
        self.inherited = False
        self.history_n = 0.0
        self.history_sum = 0.0
        self.looks: List[float] = []
        self.page: List[float] = []
        self.cusum = 0.0
        self.first_window = True
        self.likelihood = (0.0, False, False)   # log_ratio, has_measurement, calibrated_source
        self.likelihood_stamp = 0
        self.index: Dict[Tuple[int, int, int], int] = {}
        n = 0
        self.identity_hits = np.zeros(n, np.int64)
        self.other_obs = np.zeros(n, np.int64)
        self.seen_through_while_identified = np.zeros(n, bool)
        self.tentative_hits = np.zeros(n, np.int64)
        self.tentative_other = np.zeros(n, np.int64)
        self.tentative_veto = np.zeros(n, bool)
        self.last_on_surface = np.zeros(n, np.int64)
        self.last_seen_through = np.zeros(n, np.int64)
        self.last_identity = np.zeros(n, np.int64)
        self.last_look = np.zeros(n, np.int64)

    def rows(self, cells: List[Tuple[int, int, int]]) -> np.ndarray:
        """Row of every cell, creating missing ones (samples.try_emplace)."""
        out = np.empty(len(cells), np.int64)
        new = 0
        for i, c in enumerate(cells):
            r = self.index.get(c)
            if r is None:
                r = len(self.index)
                self.index[c] = r
                new += 1
            out[i] = r
        if new:
            grow = lambda a, v: np.concatenate([a, np.full(new, v, a.dtype)])
            self.identity_hits = grow(self.identity_hits, 0)
            self.other_obs = grow(self.other_obs, 0)
            self.seen_through_while_identified = grow(self.seen_through_while_identified, False)
            self.tentative_hits = grow(self.tentative_hits, 0)
            self.tentative_other = grow(self.tentative_other, 0)
            self.tentative_veto = grow(self.tentative_veto, False)
            self.last_on_surface = grow(self.last_on_surface, 0)
            self.last_seen_through = grow(self.last_seen_through, 0)
            self.last_identity = grow(self.last_identity, 0)
            self.last_look = grow(self.last_look, -1)
        return out


def robust_variance(values, centre: float) -> float:
    """[187-201] 1.4826 * MAD squared, floored; centre < 0: the (upper) median."""
    values = list(values)
    if not values:
        return K_SHARE_VARIANCE_FLOOR
    if centre < 0:
        c = sorted(values)
        centre = c[len(c) // 2]
    dev = sorted(abs(v - centre) for v in values)
    mad = dev[len(dev) // 2]
    return max(K_SHARE_VARIANCE_FLOOR, (1.4826 * mad) * (1.4826 * mad))


@dataclass
class CellModel:                                    # [211-223]
    identified: bool = False
    log_pi: float = ieee.log(0.5)
    log_1mpi: float = ieee.log(0.5)
    log_r: Tuple[float, float] = (0.0, 0.0)
    log_p: Tuple[float, float] = (0.0, 0.0)
    stamp: int = 0


def cell_log_odds(m: CellModel, own: np.ndarray, other: np.ndarray) -> np.ndarray:
    if not m.identified:
        return np.zeros_like(own, dtype=np.float64)
    return m.log_pi - m.log_1mpi + own * (m.log_r[0] - m.log_p[0]) + other * (m.log_r[1] - m.log_p[1])


@dataclass
class PresentBeta:                                  # [578-591]
    m: float = 0.0
    a: float = 0.0
    b: float = 0.0


def present_beta(mean: float, second: float) -> PresentBeta:
    m = min(0.999, max(0.001, mean))
    v = max(K_SHARE_VARIANCE_FLOOR, second - mean * mean)
    c = max(K_UNIFORM_PRIOR_PSEUDO_COUNT, m * (1 - m) / v - 1)
    return PresentBeta(m, m * c + 1e-3, (1 - m) * c + 1e-3)


def present_log_density(beta: PresentBeta, f: float) -> float:     # [607-611]
    f = min(0.995, max(max(0.005, beta.m), f))
    return (math.lgamma(beta.a + beta.b) - math.lgamma(beta.a) - math.lgamma(beta.b) +
            (beta.a - 1) * ieee.log(f) + (beta.b - 1) * ieee.log(1 - f))


def finite_count_log_ratio(k: float, n: float, beta: PresentBeta) -> float:   # [615-623]
    log_present_count = (math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0) +
                         math.lgamma(k + beta.a) + math.lgamma(n - k + beta.b) - math.lgamma(n + beta.a + beta.b) -
                         math.lgamma(beta.a) - math.lgamma(beta.b) + math.lgamma(beta.a + beta.b))
    log_ratio = -ieee.log1p(n) - log_present_count
    if k / n <= beta.a / (beta.a + beta.b):
        log_ratio = min(0.0, log_ratio)
    return log_ratio


@dataclass
class SurfaceEvidenceCounts:
    surface_samples: int = 0
    unobserved_samples: int = 0
    contradicted_surface_samples: int = 0
    supported_votes: int = 0
    free_space_votes: int = 0
    replaced_by_background_votes: int = 0
    replaced_by_other_votes: int = 0
    occluded_votes: int = 0
    support_rays: int = 0
    contradiction_rays: int = 0
    latest_support_stamp: int = 0
    absence_coverage_sufficient: bool = False
    reliable_samples: int = 0
    reliable_in_view: int = 0
    reliable_seen_through: int = 0
    absence_llr: float = 0.0


# ------------------------------------------------------------------ the process-wide model
class AbsenceModel:
    """absence_states + the pooled population + the cell model of one session (one C++ process)."""

    def __init__(self, store, config: Optional[EvidenceConfig] = None, log=None):
        self.store = store
        self.config = config or EvidenceConfig()
        self.states: Dict[Tuple[int, int], ObjectAbsenceState] = {}
        self.pooled_n = 0.0
        self.pooled_sum = 0.0
        self.pooled_geo_dev: List[float] = []
        self.loaded_geo_var = -1.0
        self.cell_model = CellModel()
        self.log = log if log is not None else []

    # --- sensor statistics [698-719]
    def save_sensor_statistics(self, path) -> None:
        gv = robust_variance(self.pooled_geo_dev, -1.0) if len(self.pooled_geo_dev) >= K_ROBUST_LOOKS \
            else self.loaded_geo_var
        with open(path, "w") as f:
            f.write(f"{self.pooled_n!r} {self.pooled_sum!r} {gv!r}\n")

    def load_sensor_statistics(self, path) -> bool:
        try:
            n, s, gv = (float(x) for x in open(path).read().split()[:3])
        except (OSError, ValueError):
            return False
        self.pooled_n += n
        self.pooled_sum += s
        self.loaded_geo_var = gv
        return True

    # --- cell model EM [226-288]
    def estimate_cell_model(self, stamp: int) -> None:
        cm = self.cell_model
        if cm.stamp == stamp:
            return
        cm.stamp = stamp
        hist: Dict[Tuple[int, int], float] = {}
        for st in self.states.values():
            ok = ~st.seen_through_while_identified & ((st.identity_hits + st.other_obs) > 0)
            for a, b in zip(st.identity_hits[ok].tolist(), st.other_obs[ok].tolist()):
                hist[(a, b)] = hist.get((a, b), 0.0) + 1.0
        if not hist:
            return
        cells = [(float(a), float(b), m) for (a, b), m in sorted(hist.items())]
        total = sum(c[2] for c in cells)
        sr, sp, nr = [0.0, 0.0], [0.0, 0.0], 0.0
        for c in cells:
            r = c[0] >= 0.5 * (c[0] + c[1])
            for j in range(2):
                (sr if r else sp)[j] += c[2] * c[j]
            nr += c[2] if r else 0.0
        if nr == 0 or nr == total:
            return
        pi = nr / total
        norm = lambda a: [(a[j] + 0.5) / (a[0] + a[1] + 1.0) for j in range(2)]
        tr, tp = norm(sr), norm(sp)
        for _ in range(1000):
            ar, ap, w_sum = [0.0, 0.0], [0.0, 0.0], 0.0
            lr0, lr1, lp0, lp1 = ieee.log(tr[0]), ieee.log(tr[1]), ieee.log(tp[0]), ieee.log(tp[1])
            for c in cells:
                a = ieee.log(pi) + c[0] * lr0 + c[1] * lr1
                b = ieee.log1p(-pi) + c[0] * lp0 + c[1] * lp1
                w = c[2] / (1.0 + ieee.exp(b - a))
                w_sum += w
                for j in range(2):
                    ar[j] += w * c[j]
                    ap[j] += (c[2] - w) * c[j]
            new_pi = min(max(w_sum / total, 0.5 / (total + 1)), 1.0 - 0.5 / (total + 1))
            tr, tp = norm(ar), norm(ap)
            done = abs(new_pi - pi) < 1e-12
            pi = new_pi
            if done:
                break
        if tr[0] < tp[0]:
            tr, tp = tp, tr
            pi = 1.0 - pi
        cm.identified = True
        cm.log_pi, cm.log_1mpi = ieee.log(pi), ieee.log1p(-pi)
        cm.log_r = (ieee.log(tr[0]), ieee.log(tr[1]))
        cm.log_p = (ieee.log(tp[0]), ieee.log(tp[1]))

    # --- reliability [481-486]
    def reliable(self, st: ObjectAbsenceState, rows: np.ndarray) -> np.ndarray:
        ok = ~st.seen_through_while_identified[rows]
        if st.inherited or not st.ever_identified:
            return ok
        own = (st.identity_hits[rows] + st.tentative_hits[rows]).astype(np.float64)
        other = st.other_obs[rows].astype(np.float64)
        return ok & ((own + other) > 0) & (cell_log_odds(self.cell_model, own, other) >= 0.0)

    # --- in-place model [562-575]
    def present_moments(self, st: ObjectAbsenceState) -> Tuple[float, float]:
        h = float(K_ROBUST_LOOKS)
        v0 = robust_variance(self.pooled_geo_dev, -1.0) if len(self.pooled_geo_dev) >= K_ROBUST_LOOKS \
            else self.loaded_geo_var
        m0 = self.pooled_sum / self.pooled_n if self.pooled_n >= h else 0.0
        if not (self.pooled_n >= h and v0 > 0):
            m0, v0 = K_COLD_START_SEEN_THROUGH_RATE, K_UNIFORM_SHARE_VARIANCE
        own_n = st.history_n
        own_m = st.history_sum / own_n if own_n > 0 else 0.0
        own_v = robust_variance(st.looks, own_m) if own_n >= h else 0.0
        k, use_n = h, (own_n if own_n >= h else 0.0)
        m = (use_n * own_m + k * m0) / (use_n + k)
        v = max(v0, (use_n * (own_v + (own_m - m) ** 2) + k * (v0 + (m0 - m) ** 2)) / (use_n + k))
        return m, m * m + v

    # --- Page with candidate weights [631-648]
    @staticmethod
    def add_look(st: ObjectAbsenceState, judged_rows: np.ndarray, previous_look: np.ndarray, reliable: int,
                 llr: float) -> float:
        t = len(st.page)
        first_since = np.zeros(t + 1, np.int64)
        for p in previous_look.tolist():
            first_since[p + 1] += 1
        first_since = np.cumsum(first_since)
        weight = lambda s: min(1.0, float(first_since[s]) / max(1, reliable))
        strongest = t
        for s in range(t):
            if st.page[s] > (0.0 if strongest == t else st.page[strongest]):
                strongest = s
        st.page.append(0.0)
        for s in range(t + 1):
            st.page[s] += weight(s) * llr
        st.last_look[judged_rows] = t
        st.cusum = max(0.0, max(st.page))
        return weight(strongest)

    def learn_in_place_look(self, st: ObjectAbsenceState, f: float) -> None:   # [653-663]
        st.history_n += 1
        st.history_sum += f
        if len(st.looks) < 256:
            st.looks.append(f)
        h = float(K_ROBUST_LOOKS)
        if st.history_n == h and len(self.pooled_geo_dev) < 4096:
            self.pooled_geo_dev.append(st.history_sum / h)
            self.pooled_n += 1
            self.pooled_sum += st.history_sum / h

    @staticmethod
    def commit_reliability(st: ObjectAbsenceState, in_place: bool) -> None:     # [666-680]
        if in_place:
            st.identity_hits = np.minimum(65535, st.identity_hits + st.tentative_hits)
            st.other_obs = np.minimum(65535, st.other_obs + st.tentative_other)
            st.seen_through_while_identified |= st.tentative_veto
        st.tentative_hits[:] = 0
        st.tentative_other[:] = 0
        st.tentative_veto[:] = False

    # --- queries [331-365]
    @staticmethod
    def absence_queries(points: torch.Tensor, normals: Optional[torch.Tensor], cell_size: float):
        P = points.detach().cpu().numpy().astype(np.float64)
        N = normals.detach().cpu().numpy().astype(np.float64) if normals is not None else np.full_like(P, np.nan)
        finite = np.isfinite(P).all(1)
        P, N = P[finite], N[finite]
        cells = np.floor(P / cell_size).astype(np.int64)
        _, first = np.unique(cells, axis=0, return_index=True)
        first = np.sort(first)                                   # insertion order of the first point per cell
        P, N, cells = P[first], N[first], cells[first]
        if len(P) > K_MAX_ABSENCE_SAMPLES:
            stride = len(P) / K_MAX_ABSENCE_SAMPLES
            idx = np.array([int(i * stride) for i in range(K_MAX_ABSENCE_SAMPLES)])
            P, N, cells = P[idx], N[idx], cells[idx]
        nn = np.linalg.norm(N, axis=1)
        has_normal = np.isfinite(nn) & (nn > 1e-12)
        N = np.where(has_normal[:, None], N / np.where(has_normal, nn, 1.0)[:, None], 0.0)
        return P, N, has_normal, [tuple(c) for c in cells.tolist()]

    # --- classifyFrames [373-474]
    def classify_frames(self, st: ObjectAbsenceState, P, N, has_normal, rows, physical_id: int, tolerance: float,
                        min_cos: float, latest: int) -> None:
        store = self.store
        lo, hi = store.window(st.processed + 1, latest)
        if hi > lo and len(P):
            pts = torch.as_tensor(P, dtype=torch.float32, device=DEV)
            nrm = torch.as_tensor(N, dtype=torch.float32, device=DEV)
            hn = torch.as_tensor(has_normal, device=DEV)
            stamps = store.stamp_tensor(lo, hi).tolist()
            for f0 in range(lo, hi, 64):
                f1 = min(hi, f0 + 64)
                p = store.project(f0, f1, pts)
                et, pid, meas, query, view = p["etype"], p["pid"], p["measured"], p["query"], p["view"]
                avail = et != UNAVAILABLE
                cos_view = torch.where(hn[None, :], (nrm[None] * view).sum(-1).abs(), torch.ones_like(query))
                facing = ~hn[None, :] | (cos_view >= min_cos)
                measured = avail & (et != INVALID) & torch.isfinite(meas) & (meas > 0) & torch.isfinite(query) & \
                    (query > 0)
                delta = meas - query
                on = measured & (delta.abs() <= tolerance)
                physical = (et == PHYSICAL) & (pid > 0)
                ident = on & physical & (pid == physical_id)
                foreign = on & physical & ~ident
                seen = measured & ~on & facing & (delta > tolerance)
                ident_n = ident.sum(1).tolist()
                seen_n = seen.sum(1).tolist()
                on_c, ident_c, foreign_c, seen_c = (x.cpu().numpy() for x in (on, ident, foreign, seen))
                for k in range(f1 - f0):
                    stamp = int(stamps[k + f0 - lo])
                    in_place = ident_n[k] > seen_n[k]
                    if in_place:
                        st.ever_identified = True
                    r_on, r_id, r_seen, r_for = rows[on_c[k]], rows[ident_c[k]], rows[seen_c[k]], rows[foreign_c[k]]
                    st.last_on_surface[r_on] = stamp
                    st.last_identity[r_id] = stamp
                    st.last_seen_through[r_seen] = stamp
                    if in_place:
                        st.tentative_hits[r_id] = np.minimum(65535, st.tentative_hits[r_id] + 1)
                        st.tentative_veto[r_seen] = True
                        st.tentative_veto[r_for] = True
                        r_oth = rows[on_c[k] & ~ident_c[k]]
                        st.tentative_other[r_oth] = np.minimum(65535, st.tentative_other[r_oth] + 1)
                    st.processed = stamp
        st.processed = max(st.processed, latest)

    # --- countProjectedPhysicalSurface [751-821]
    def count_projected_surface(self, physical_id: int, points: torch.Tensor, map_resolution: float,
                                earliest: int, latest: int) -> SurfaceEvidenceCounts:
        res = SurfaceEvidenceCounts()
        if not (math.isfinite(map_resolution) and map_resolution > 0):
            return res
        P = points.detach().cpu().numpy().astype(np.float64)
        P = P[np.isfinite(P).all(1)]
        cells = np.floor(P / map_resolution).astype(np.int64)
        _, first = np.unique(cells, axis=0, return_index=True)
        P = P[np.sort(first)]
        res.surface_samples = len(P)
        lo, hi = self.store.window(earliest, latest)
        if hi <= lo or not len(P):
            res.unobserved_samples = len(P)
            return res
        stamps = self.store.stamp_tensor(lo, hi)
        pts = torch.as_tensor(P, dtype=torch.float32, device=DEV)
        coverage = torch.zeros(len(P), dtype=torch.bool, device=DEV)
        s_sup = torch.zeros(len(P), dtype=torch.int64, device=DEV)
        s_abs = torch.zeros(len(P), dtype=torch.int64, device=DEV)
        sup_keys, con_keys = [], []
        for f0 in range(lo, hi, 64):
            f1 = min(hi, f0 + 64)
            p = self.store.project(f0, f1, pts)
            vote = classify_measurements(p, physical_id, self.config.depth_tolerance)
            st = stamps[f0 - lo:f1 - lo][:, None].expand_as(vote)
            fr = torch.arange(f0 - lo, f1 - lo, device=DEV)[:, None].expand_as(vote)
            coverage |= (vote != V_UNAVAILABLE).any(0)
            sup = vote == V_SUPPORTED
            con = (vote == V_FREE) | (vote == V_BACKGROUND) | (vote == V_OTHER)
            # latest frame of each kind per sample (frames ascend)
            s_sup = torch.maximum(s_sup, torch.where(sup, st, torch.zeros_like(st)).amax(0))
            s_abs = torch.maximum(s_abs, torch.where(con, st, torch.zeros_like(st)).amax(0))
            key = (fr << 32) | p["pixel"].clamp(min=0)
            sup_keys.append(key[sup])
            con_keys.append(key[con])
            res.supported_votes += int(sup.sum())
            res.free_space_votes += int((vote == V_FREE).sum())
            res.replaced_by_background_votes += int((vote == V_BACKGROUND).sum())
            res.replaced_by_other_votes += int((vote == V_OTHER).sum())
            res.occluded_votes += int((vote == V_OCCLUDED).sum())
            if sup.any():
                res.latest_support_stamp = max(res.latest_support_stamp, int(st[sup].max()))
        res.unobserved_samples = int((~coverage).sum())
        res.contradicted_surface_samples = int((s_abs > s_sup).sum())
        res.absence_coverage_sufficient = res.surface_samples > 0 and res.contradicted_surface_samples > 0 and \
            res.contradicted_surface_samples / res.surface_samples >= self.config.min_absent_surface_fraction
        res.support_rays = int(torch.unique(torch.cat(sup_keys)).numel()) if sup_keys else 0
        res.contradiction_rays = int(torch.unique(torch.cat(con_keys)).numel()) if con_keys else 0
        return res

    # --- applyObservedAbsence [823-963]
    def apply_observed_absence(self, physical_id: int, points: torch.Tensor, normals: Optional[torch.Tensor],
                               latest: int, counts: SurfaceEvidenceCounts, state_birth: int, prior_log_odds: float,
                               state_key: int) -> None:
        cfg = self.config
        counts.absence_coverage_sufficient = False
        tolerance = cfg.surface_match_tolerance
        min_cos = math.cos(cfg.max_absence_incidence_deg * math.pi / 180.0)
        P, N, has_normal, cells = self.absence_queries(points, normals, 0.5 * tolerance)
        key = (physical_id, state_key)
        st = self.states.get(key)
        if st is None:
            st = self.states[key] = ObjectAbsenceState()
            if state_birth > 0:
                st.processed = state_birth - 1
        st.likelihood_stamp = latest
        st.likelihood = (0.0, False, True)
        if latest < st.processed:                         # a new session restarts time
            st.processed = 0
            st.ever_identified = False
            st.index = {}
            st.rows([])                                   # samples.clear(): empty per-sample arrays
            for name in ("identity_hits", "other_obs", "seen_through_while_identified", "tentative_hits",
                         "tentative_other", "tentative_veto", "last_on_surface", "last_seen_through",
                         "last_identity", "last_look"):
                setattr(st, name, getattr(st, name)[:0])
        round_start = st.processed + 1
        first = int(self.store._stamps[0]) if self.store.n else None    # timestamps(0, latest).front()
        st.inherited = first is not None and state_birth != 0 and state_birth < first
        rows = st.rows(cells)
        self.classify_frames(st, P, N, has_normal, rows, physical_id, tolerance, min_cos, latest)
        self.estimate_cell_model(latest)
        # summarizeLook [499-520]
        own_identity = int(((st.last_identity[rows] >= round_start) & (st.last_identity[rows] != 0)).sum())
        rel = self.reliable(st, rows)
        reliable = int(rel.sum())
        last = np.maximum(st.last_on_surface[rows], st.last_seen_through[rows])
        judged = rel & (last >= round_start) & (last != 0)
        through = judged & (st.last_seen_through[rows] > st.last_on_surface[rows])
        k, n = int(through.sum()), int(judged.sum())
        counts.reliable_samples, counts.reliable_in_view, counts.reliable_seen_through = reliable, n, k
        in_place = own_identity > k
        if n > 0 and n >= min(K_MIN_SAMPLES_IN_VIEW, reliable):
            f = k / n
            beta = present_beta(*self.present_moments(st))
            llr = -present_log_density(beta, f)
            self.log.append(f"ABSENCE_LOOK inst={physical_id} record={state_key} stamp={latest} k={k} n={n} "
                            f"reliable={reliable} own={own_identity} identified={int(in_place)} "
                            f"inherited={int(st.inherited)} llr={llr:.6g} cusum_before={st.cusum:.6g}")
            jr = rows[judged]
            weight = self.add_look(st, jr, _previous(st, jr), reliable, llr)
            if weight > 0.0:
                st.likelihood = (weight * finite_count_log_ratio(float(k), float(n), beta), True, True)
            if in_place:
                self.learn_in_place_look(st, f)
        elif n > 0:
            self.log.append(f"ABSENCE_UNSCORED inst={physical_id} stamp={latest} k={k} n={n} reliable={reliable} "
                            f"own={own_identity}")
        counts.absence_llr = st.cusum
        before = (prior_log_odds + (st.page[0] if st.page else 0.0)) if st.first_window else -math.inf
        counts.absence_coverage_sufficient = max(before, st.cusum) > LOG99
        if prior_log_odds != 0.0:
            self.log.append(f"ABSENCE_COMMIT inst={physical_id} prior_log_odds={prior_log_odds:.6g} "
                            f"before_session={before:.6g} page_statistic={st.cusum:.6g} "
                            f"committed={int(counts.absence_coverage_sufficient)}")
        self.commit_reliability(st, in_place)
        if counts.absence_coverage_sufficient:
            st.cusum = 0.0
            st.page = []
            st.first_window = False
            st.last_look[:] = -1

    def look_likelihood(self, physical_id: int, state_key: int, stamp: int) -> Tuple[float, bool, bool]:
        """physicalAbsenceLookLikelihood [684-696]."""
        st = self.states.get((physical_id, state_key))
        if st is not None and st.likelihood_stamp == stamp:
            return st.likelihood
        return (0.0, False, False)

    # --- countCurrentPhysicalSurface [965-998] (stored pixels always exist in the layer)
    def count_current_surface(self, physical_id: int, points: torch.Tensor, normals: Optional[torch.Tensor],
                              map_resolution: float, last_support: int, latest: int, state_birth: int,
                              prior_log_odds: float, state_key: int) -> SurfaceEvidenceCounts:
        if last_support >= latest:
            none = SurfaceEvidenceCounts()
            self.apply_observed_absence(physical_id, points, normals, latest, none, state_birth, 0.0, state_key)
            none.absence_coverage_sufficient = False
            return none
        measured = self.count_projected_surface(physical_id, points, map_resolution, last_support + 1, latest)
        self.apply_observed_absence(physical_id, points, normals, latest, measured, state_birth, prior_log_odds,
                                    state_key)
        return measured


def _previous(st: ObjectAbsenceState, judged_rows: np.ndarray) -> np.ndarray:
    """previous_look of the judged samples, in query order (summarizeLook)."""
    return st.last_look[judged_rows].copy()
