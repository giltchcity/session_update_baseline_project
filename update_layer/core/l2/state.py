"""persistent_object_state.cpp @192c1cf (L2_FINAL2), ported line by line onto point-set fragments.

A fragment's geometry is a world-space point set (Nx3 float32, optional normals, the backend element
ids carried with it) instead of a spark_dsg::Mesh in its bounding-box frame. Every geometric test of the
original reads mesh vertices only, through BoundingBox::pointToWorldFrame, so the decisions are the
original's; faces were only concatenated. Float semantics follow the C++ (float32 points; float or
double divisions exactly where the original has them). Nothing is re-derived or tuned: constants,
rules and the comments justifying them are the original's (README section in brackets).

The geometric kernels are module functions; PersistentObjectState calls them through four methods
(_shared_space, _extent_same_site, _off_state_share, _append) so that the consistency test can inject
the values the TSDF run logged (replay_state.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from scipy.special import betainc

from . import ieee

K_STATE_TOLERANCE = 0.10          # kStateTolerance (header): one state's granularity, m
K_GEOMETRY_FACTOR_BOUND = 2.0     # kGeometryFactorBound: B_{M:S} <= 2
K_COPY_VALIDITY_PRIOR = 0.5       # copyInvalidityTerm: indifference value of v
LOG99 = ieee.log(99.0)

_F32 = np.float32


def _g(x: float) -> str:
    """std::ostream << double (default precision 6)."""
    return f"{x:g}"


def _fixed(x: float) -> str:
    """std::to_string(double)."""
    return f"{x:f}"


# ------------------------------------------------------------------------------------ data
@dataclass
class Look:
    stamp: int = 0
    support_rays: int = 0
    reliable_in_view: int = 0
    reliable_seen_through: int = 0
    measured_absence_log_ratio: float = 0.0
    has_measured_absence_likelihood: bool = False
    has_calibrated_absence_source: bool = False


@dataclass
class SurfaceEvidence:
    support_rays: int = 0
    contradiction_rays: int = 0
    surface_samples: int = 0
    absence_coverage_sufficient: bool = False      # set only by the observed-absence test
    supported_votes: int = 0
    free_space_votes: int = 0
    replaced_by_other_votes: int = 0
    replaced_by_background_votes: int = 0
    occluded_votes: int = 0
    unobserved_samples: int = 0
    latest_support_stamp: int = 0                  # actual sensor time, never reducer/check time
    reliable_in_view: int = 0
    reliable_seen_through: int = 0
    measured_absence_log_ratio: float = 0.0
    has_measured_absence_likelihood: bool = False
    has_calibrated_absence_source: bool = False
    reliable_samples: int = 0


@dataclass
class Observation:
    """One geometry-bearing segment of a physical identity (a Khronos object node of the original).
    moved = kHasDynamicHistoryDetail (the tracker watched it move, D1); track_first_seen = the
    kTrackFirstSeenDetail bookkeeping (0: the segment's own first)."""
    identity: int
    points: object                     # Nx3 world, float32
    normals: object = None
    first: int = 0
    last: int = 0
    semantic: int = -1
    moved: bool = False
    reconstruction_frames: int = 0
    elements: object = None
    track_first_seen: int = 0


@dataclass
class Fragment:
    points: object
    normals: object = None
    elements: object = None
    birth_time: int = 0
    uid: int = 0
    track_first_seen: int = 0
    last_support_time: int = 0
    last_confirmed_support: int = 0
    semantic_label: int = -1
    requires_current_session_support: bool = False
    death_time: Optional[int] = None
    reconstruction_frames: int = 0
    looks: List[Look] = field(default_factory=list)

    @property
    def num_vertices(self) -> int:
        return len(self.points)


@dataclass
class PhysicalState:
    fragments: List[Fragment] = field(default_factory=list)
    current: Optional[int] = None
    observed_new: Optional[Fragment] = None
    pending_absence_stamp: int = 0
    b_session: Optional["PhysicalState"] = None
    last_support_rays: int = 0
    last_contradiction_rays: int = 0
    last_surface_samples: int = 0
    last_session_reliable_samples: int = 0
    last_merged_observation_first: int = 0
    ingested_intervals: Set[Tuple[int, int]] = field(default_factory=set)
    has_dynamic_history: bool = False
    mobility_changes: int = 0
    mobility_continuations: int = 0


@dataclass
class SameStatePosterior:
    q: float = 0.0
    factor: float = 1.0               # B_{M:S}, or its bound when not evaluated
    evaluated: bool = False
    off: float = 0.0
    effective_cells: float = 0.0
    same: bool = True

    def __str__(self) -> str:
        return (f" change_prior={_g(self.q)} geometry_factor_or_bound={_g(self.factor)}"
                f" effective_cells={_g(self.effective_cells)} off_share="
                + (_g(self.off) if self.evaluated else "unmeasured"))


@dataclass
class HistoryRecord:
    changes: float = 0.0
    continuations: float = 0.0
    session_changes: float = 0.0
    session_continuations: float = 0.0
    semantic_label: int = -1
    has_fragments: bool = False


@dataclass
class Materialized:
    """applyPhysicalGeometry's write into the merged node: the fragments shown (CURRENT, plus the
    session state when it is the same physical state), and, for an identity whose last state ended
    without a successor, the end of its presence."""
    fragments: List[Fragment]
    present: bool
    presence_end: Optional[int] = None


# ------------------------------------------------------------------------- geometric kernels
def _np32(points) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    return np.ascontiguousarray(points, dtype=_F32).reshape(-1, 3)


_OFF = np.int64(1 << 20)


def _pack(k: np.ndarray) -> np.ndarray:
    """Lexicographic (x, y, z) order of int64 cell keys as one int64 (std::map<std::tuple> order)."""
    k = k + _OFF
    return (k[..., 0] << 42) | (k[..., 1] << 21) | k[..., 2]


def _cells(p: np.ndarray, keys: np.ndarray):
    """Group points by cell key: sorted packed keys, per-cell double sums and counts."""
    packed = _pack(keys)
    uniq, inv = np.unique(packed, return_inverse=True)
    sums = np.zeros((len(uniq), 3), np.float64)
    np.add.at(sums, inv, p.astype(np.float64))
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    return uniq, sums, counts


_NEIGHBOURS = np.array([(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
                       np.int64)


def shared_space_probability(current, candidate, resolution: float, decision_probability: float) -> float:
    """sharedSpaceProbability [M1c]: one quantization witness per occupied voxel. Lack of shared support
    remains unresolved; it never establishes movement or absence."""
    a, b = _np32(current), _np32(candidate)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    res = _F32(resolution)
    ka = np.floor(a / res).astype(np.int64)            # float / float, as p.x() / resolution
    kb = np.floor(b / res).astype(np.int64)
    ref_keys, ref_sum, ref_cnt = _cells(a, ka)
    obs_keys, obs_sum, obs_cnt = _cells(b, kb)
    ref_mean = ref_sum / ref_cnt[:, None]
    point = obs_sum / obs_cnt[:, None]
    # unpack the observed keys to look up their 27 neighbours
    ok = np.stack([(obs_keys >> 42) & 0x1FFFFF, (obs_keys >> 21) & 0x1FFFFF, obs_keys & 0x1FFFFF], 1) - _OFF
    corr = np.zeros(len(obs_keys), np.float64)
    r = float(res)
    for off in _NEIGHBOURS:
        nk = _pack(ok + off)
        pos = np.searchsorted(ref_keys, nk)
        pos_c = np.minimum(pos, len(ref_keys) - 1)
        hit = ref_keys[pos_c] == nk
        if not hit.any():
            continue
        delta = np.abs(point[hit] - ref_mean[pos_c[hit]])
        prob = np.prod(np.maximum(1.0 - delta / r, 0.0), axis=1)
        corr[hit] = np.maximum(corr[hit], prob)
    # sequential accumulation in std::map order, with the original's two early exits
    log_no = 0.0
    for c in corr.tolist():
        if c == 1.0:
            return 1.0
        log_no += ieee.log1p(-c)
        p = -math.expm1(log_no)
        # A monotone lower bound suffices for exactly the same final decision.
        if p > decision_probability:
            return p
    return -math.expm1(log_no)


def extent_same_site_probability(current, candidate, resolution: float) -> float:
    """extentSameSiteProbability [M1d]: probability that the world AABBs share occupied cells under the
    uniform grid-phase model of sharedSpaceProbability."""
    a, b = _np32(current), _np32(candidate)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    a_lo, a_hi, b_lo, b_hi = a.min(0), a.max(0), b.min(0), b.max(0)
    r = float(_F32(resolution))
    probability = 1.0
    for axis in range(3):
        gap = max(0.0, float(a_lo[axis]) - float(b_hi[axis]), float(b_lo[axis]) - float(a_hi[axis]))
        probability *= max(0.0, 1.0 - gap / r)
    return probability


def off_state_share(copy, reference, tolerance: float, map_resolution: float) -> Tuple[float, float]:
    """offStateShare [M1g, M1h] -> (expected off-state share, effective number of cells).

    M1g: radial cell-phase uncertainty has support [-resolution, resolution]; no point beyond
    tolerance + resolution can contribute correspondence mass. M1h: vertex weights are kept; one
    correlated spatial group per occupied resolution cell."""
    c = _np32(copy)
    if len(c) == 0:
        return 0.0, 0.0
    ref = _np32(reference)
    tol = float(_F32(tolerance))                  # the float parameter of the original
    resolution = float(_F32(map_resolution))      # const double resolution = map_resolution_ (float)
    radius = tol + resolution
    radius2 = radius * radius
    certain_radius = tol - resolution
    nearest2 = np.full(len(c), radius2, np.float64)
    if len(ref):
        from scipy.spatial import cKDTree
        _, idx = cKDTree(ref.astype(np.float64)).query(c.astype(np.float64), k=1,
                                                       distance_upper_bound=radius * (1 + 1e-9))
        found = idx < len(ref)
        d = ref[idx[found]] - c[found]                         # float32, (q - p)
        # Eigen's unrolled squaredNorm of a Vector3f: x^2 + (y^2 + z^2) (binary split 1 + 2)
        d2 = (d[:, 0] * d[:, 0] + (d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])).astype(np.float64)
        nearest2[found] = np.minimum(radius2, d2)
    certain = (certain_radius >= 0.0) & (nearest2 <= certain_radius * certain_radius)
    u = (tol - np.sqrt(nearest2)) / resolution
    # These endpoints are the exact support of the difference of two uniforms.
    same = np.where(certain | (u >= 1.0), 1.0,
                    np.where(u <= -1.0, 0.0,
                             np.where(u < 0.0, 0.5 * (1.0 + u) * (1.0 + u), 1.0 - 0.5 * (1.0 - u) * (1.0 - u))))
    expected_off = float(np.cumsum(1.0 - same)[-1])            # sequential, as the loop
    keys = _pack(np.floor(c.astype(np.float64) / resolution).astype(np.int64))   # float / double
    _, counts = np.unique(keys, return_counts=True)
    vertices = float(len(c))
    effective_cells = vertices * vertices / float(np.sum(counts.astype(np.float64) ** 2))
    return expected_off / vertices, effective_cells


def motion_geometry_bayes_factor(off: float, effective_cells: float) -> float:
    """motionGeometryBayesFactor: 2 ibetac(1/2 + n off, 1/2 + n (1 - off), 1/2); ibetac(a, b, x) =
    I_{1-x}(b, a)."""
    alpha = 0.5 + effective_cells * off
    beta = 0.5 + effective_cells * (1.0 - off)
    return 2.0 * float(betainc(beta, alpha, 0.5))


def _cat(a, b, na: int, nb: int, width: int):
    if a is None and b is None:
        return None
    dev = None
    for t in (a, b):
        if isinstance(t, torch.Tensor):
            dev = t.device
    a = a if a is not None else torch.full((na, width), float("nan"), device=dev)
    b = b if b is not None else torch.full((nb, width), float("nan"), device=dev)
    return torch.cat([a, b])


# ---------------------------------------------------------------------------- the registry
class PersistentObjectState:
    def __init__(self, map_resolution: float = 0.05, high_mobility_semantics: Sequence[int] = ()):
        if not map_resolution > 0.0:
            raise ValueError("PersistentObjectState map resolution must be positive")
        self.states: Dict[int, PhysicalState] = {}
        self.event_history: Dict[int, HistoryRecord] = {}
        self.event_stamp = 0
        self.event_open = False
        self.map_resolution = float(_F32(map_resolution))
        self.high_mobility: Set[int] = set(high_mobility_semantics)
        self.log: List[str] = []
        self._next_uid = 1                 # static std::atomic in the original: one per process/session

    # -- geometry hooks -------------------------------------------------------------------------
    def _shared_space(self, current: Fragment, obs: Observation, decision: float) -> float:
        return shared_space_probability(current.points, obs.points, self.map_resolution, decision)

    def _extent_same_site(self, current: Fragment, candidate: Fragment) -> float:
        return extent_same_site_probability(current.points, candidate.points, self.map_resolution)

    def _off_state_share(self, copy: Fragment, reference: Fragment) -> Tuple[float, float]:
        return off_state_share(copy.points, reference.points, K_STATE_TOLERANCE, self.map_resolution)

    @staticmethod
    def _copy_geometry(obs: Observation):
        def clone(t):
            return t.clone() if isinstance(t, torch.Tensor) else (None if t is None else np.array(t))
        el = obs.elements
        if el is None and isinstance(obs.points, torch.Tensor):
            el = torch.zeros(0, dtype=torch.int64, device=obs.points.device)
        return clone(obs.points), clone(obs.normals), clone(el)

    @staticmethod
    def _append(target: Fragment, points, normals, elements) -> None:
        """appendMeshUnion (vertices; mismatched optional fields defaulted, as the original)."""
        n0, n1 = len(target.points), len(points)
        target.normals = _cat(target.normals, normals, n0, n1, 3)
        target.points = torch.cat([target.points, points])
        if target.elements is not None or elements is not None:
            ea = target.elements if target.elements is not None else torch.zeros(0, dtype=torch.int64)
            eb = elements if elements is not None else torch.zeros(0, dtype=torch.int64, device=ea.device)
            target.elements = torch.cat([ea, eb.to(ea.device)])

    def _union(self, target: Fragment, source: Fragment) -> None:
        self._append(target, source.points, source.normals, source.elements)

    # -- M2a: the persistence prior -----------------------------------------------------------
    def set_high_mobility_semantic_labels(self, labels: Sequence[int]) -> None:
        self.high_mobility = set(labels)

    def begin_observation_event(self, stamp: int) -> None:
        if self.event_open and stamp == self.event_stamp:
            return                                        # same evidence snapshot
        self.event_history = {}
        for i, state in self.states.items():
            record = HistoryRecord(changes=float(state.mobility_changes),
                                   continuations=float(state.mobility_continuations))
            if state.b_session is not None:
                record.session_changes = float(state.b_session.mobility_changes)
                record.session_continuations = float(state.b_session.mobility_continuations)
            if state.fragments:
                record.has_fragments = True
                record.semantic_label = (state.fragments[state.current] if state.current is not None
                                         else state.fragments[-1]).semantic_label
            self.event_history[i] = record
        self.event_stamp = stamp
        self.event_open = True

    def change_prior_log_odds(self, physical_instance_id: int, state_slot: int) -> float:
        # q is the prior of one relation between consecutive states [M2a], not an in-session hazard per
        # round, so only the inherited state's cross-session relation carries it.
        if state_slot != 0:
            return 0.0
        state = self.states.get(physical_instance_id)
        if state is None or state.current is None:
            return 0.0
        current = state.fragments[state.current]
        if not current.requires_current_session_support:
            return 0.0
        q = self.state_change_probability(state, current)
        return ieee.log(q) - ieee.log1p(-q)

    def state_change_probability(self, state: PhysicalState, current: Fragment) -> float:
        """[M2a] ontology groups the prior population; only resolved historical relations update the
        Bernoulli probability. The target never trains its own prior."""
        owner, instance_id, registered = state, 0, False
        for i in sorted(self.states):
            root = self.states[i]
            if root is state or root.b_session is state:
                owner, instance_id, registered = root, i, True
                break
        session = owner is not state
        hm = self.high_mobility
        group = current.semantic_label in hm
        # One configured ontology judgment contributes one prior opinion, not a measured movement.
        # Empty ontology contributes no directional information.
        ontology_opinions = 0.0 if not hm else 1.0
        prior_mass = 1.0 + ontology_opinions
        own_m = own_u = class_m = class_u = group_m = group_u = 0.0

        def population(label: int, m: float, u: float) -> None:
            nonlocal class_m, class_u, group_m, group_u
            if label == current.semantic_label:
                class_m += m
                class_u += u
            elif (label in hm) == group:
                group_m += m
                group_u += u

        # [M2a] condition on the history resolved before this observation event (an identity first
        # registered inside the event has none yet); outside an event, on the live registry.
        if self.event_open:
            own = self.event_history.get(instance_id) if registered else None
            if own is not None:
                own_m, own_u = own.changes, own.continuations
                if session:
                    own_m += own.session_changes
                    own_u += own.session_continuations
            for i in sorted(self.event_history):
                record = self.event_history[i]
                if (registered and i == instance_id) or not record.has_fragments:
                    continue
                population(record.semantic_label, record.changes, record.continuations)
        else:
            own_m, own_u = float(state.mobility_changes), float(state.mobility_continuations)
            if session:
                own_m += owner.mobility_changes
                own_u += owner.mobility_continuations
            for i in sorted(self.states):
                other = self.states[i]
                if other is owner or not other.fragments:
                    continue
                fragment = other.fragments[other.current] if other.current is not None else other.fragments[-1]
                population(fragment.semantic_label, float(other.mobility_changes),
                           float(other.mobility_continuations))
        group_mean = (group_m + 0.5 + ontology_opinions * group) / (group_m + group_u + prior_mass)
        alpha = prior_mass * group_mean + class_m
        beta = prior_mass * (1.0 - group_mean) + class_u
        probability = (alpha + own_m) / (alpha + beta + own_m + own_u)
        self.log.append(f"MOBILITY_PRIOR inst={instance_id} session={int(session)} class={current.semantic_label}"
                        f" changes={_g(own_m)} continuations={_g(own_u)} alpha={_g(alpha)} beta={_g(beta)}"
                        f" q={_g(probability)} event_stamp={self.event_stamp if self.event_open else 0}")
        return probability

    # -- fragments ------------------------------------------------------------------------------
    def make_fragment(self, obs: Observation) -> Fragment:
        # Provenance rule: a fragment's geometry is exactly what was observed of *this* state. It is
        # never a previous fragment's mesh re-anchored to a new box, and never a union across states.
        pts, nrm, el = self._copy_geometry(obs)
        uid = self._next_uid
        self._next_uid += 1
        # A direct observation is support for the state it observed. For fragments restored from a
        # previous session this is reset by initialize_from_objects: A's observation timestamps are
        # not evidence in B.
        return Fragment(points=pts, normals=nrm, elements=el, birth_time=obs.first, uid=uid,
                        track_first_seen=obs.track_first_seen if obs.track_first_seen > 0 else obs.first,
                        last_support_time=obs.last, last_confirmed_support=obs.last,
                        requires_current_session_support=False, semantic_label=obs.semantic,
                        reconstruction_frames=obs.reconstruction_frames)

    # -- M1g/M1h/M1i: same physical state ------------------------------------------------------
    def same_state_posterior(self, q: float, measured: Fragment, shape: Fragment) -> SameStatePosterior:
        p = SameStatePosterior(q=q)
        if measured.num_vertices and shape.num_vertices:
            p.factor = K_GEOMETRY_FACTOR_BOUND
            p.evaluated = 1.0 - q < q * p.factor
            if p.evaluated:
                p.off, p.effective_cells = self._off_state_share(measured, shape)
                p.factor = motion_geometry_bayes_factor(p.off, p.effective_cells)
        p.same = 1.0 - q >= q * p.factor
        return p

    @staticmethod
    def copy_invalidity_term(copy: Fragment) -> Tuple[float, bool, int, float]:
        """-> ((1 - v) / v L_{U:N}, calibrated, measured_looks, log_ratio).

        An invalid copy is a surface the rays pass through: each calibrated look already carries
        log p(look | absent) / p(look | present) on the copy's own surface (the observed-absence test's
        per-look ratio). The prior validity v of a session copy has no measured rate, so it is the
        indifference value 1/2 and (1 - v) / v = 1."""
        calibrated, measured_looks, log_ratio = False, 0, 0.0
        for look in copy.looks:
            calibrated = calibrated or look.has_calibrated_absence_source
            if look.has_measured_absence_likelihood:
                log_ratio += look.measured_absence_log_ratio
                measured_looks += 1
        # Without a calibrated look there is no measurement of U and L_{U:N} = 1 (log_ratio = 0).
        term = (1.0 - K_COPY_VALIDITY_PRIOR) / K_COPY_VALIDITY_PRIOR * ieee.exp(log_ratio)
        return term, calibrated, measured_looks, log_ratio

    def session_copy_elsewhere(self, state: PhysicalState, inherited: Fragment,
                               session_reliable_samples: int) -> bool:
        b = state.b_session
        if b is None or b.current is None:
            return False
        if session_reliable_samples == 0:
            return False                                  # no established surface measurement
        copy = b.fragments[b.current]
        if copy.num_vertices == 0:
            return False                                  # no surface correspondence measurement
        # README 1.1, one decision over S (A persists, N is more of A), M (A ended, N is its successor)
        # and U (N is not a valid surface of this identity). Handing over is wrong under S and under U,
        # keeping is wrong under M, all at the same surface loss, so the Bayes action is
        #   hand over  iff  q B_{M:S} > (1 - q) + ((1 - v) / v) L_{U:N},
        # with q the motion prior of the A->N relation, B_{M:S} the M1h geometry factor (<= 2), the U
        # geometry equal to the unrestricted reference model of M1h, and the U odds term from N's own
        # looks (copy_invalidity_term).
        invalid_term, calibrated, measured_looks, _ = self.copy_invalidity_term(copy)
        q = self.state_change_probability(state, inherited)
        stay = (1.0 - q) + invalid_term
        # B_{M:S} <= 2: when even the bound cannot make M the Bayes action, the geometry cannot change
        # the decision and is not evaluated.
        bound_allows = q * K_GEOMETRY_FACTOR_BOUND > stay
        effective_cells, off, factor = 0.0, 0.0, K_GEOMETRY_FACTOR_BOUND
        if bound_allows:
            off, effective_cells = self._off_state_share(copy, inherited)
            factor = motion_geometry_bayes_factor(off, effective_cells)
        elsewhere = bound_allows and q * factor > stay
        self.log.append(f"SAME_STATE inst={inherited.semantic_label}/{copy.num_vertices}v"
                        f" copy_reliable={session_reliable_samples} change_prior={_g(q)}"
                        f" calibrated={int(calibrated)} measured_looks={measured_looks}"
                        f" invalid_term={_g(invalid_term)} effective_cells={_g(effective_cells)}"
                        f" off_share={_fixed(off) if bound_allows else 'unmeasured'}"
                        f" bayes_factor_or_bound={_g(factor)} elsewhere={int(elsewhere)}")
        return elsewhere

    @staticmethod
    def record_look(fragment: Fragment, evidence: SurfaceEvidence, stamp: int) -> None:
        if evidence.surface_samples == 0:
            return                                        # nothing of this fragment was measured
        fragment.looks.append(Look(stamp, evidence.support_rays, evidence.reliable_in_view,
                                   evidence.reliable_seen_through, evidence.measured_absence_log_ratio,
                                   evidence.has_measured_absence_likelihood,
                                   evidence.has_calibrated_absence_source))

    def observed_empty_since(self, fragment: Fragment, since: int, change_probability: float,
                             physical_instance_id: int) -> bool:
        support = judged = seen_through = 0
        log_ratio, measured, calibrated_source = 0.0, False, False
        for look in fragment.looks:
            if look.stamp > since:
                support += look.support_rays
                judged += look.reliable_in_view
                seen_through += look.reliable_seen_through
                log_ratio += look.measured_absence_log_ratio
                measured = measured or look.has_measured_absence_likelihood
                calibrated_source = calibrated_source or look.has_calibrated_absence_source
        # P04, known inconsistency kept as is: this test is called only for a CURRENT state born in
        # this session and still adds logit q, while change_prior_log_odds gives such a state prior 0.
        log_odds = ieee.log(change_probability) - ieee.log1p(-change_probability) + log_ratio
        # Count-only callers supply the original hard observation contract. A calibrated round with no
        # fresh evidence must never take that exact path.
        empty = (measured and support == 0 and log_odds > LOG99) if calibrated_source else \
            (seen_through > 0 and seen_through == judged and support == 0)
        self.log.append(f"EMPTY_INTERVAL_POSTERIOR inst={physical_instance_id} since={since} support={support}"
                        f" judged={judged} seen_through={seen_through} measured={int(measured)}"
                        f" calibrated_source={int(calibrated_source)} log_ratio={_g(log_ratio)}"
                        f" change_prior={_g(change_probability)} log_odds={_g(log_odds)} empty={int(empty)}")
        return empty

    def merge_observation_into_fragment(self, target: Fragment, obs: Observation) -> None:
        self._append(target, *self._copy_geometry(obs))
        target.reconstruction_frames += obs.reconstruction_frames
        target.last_support_time = max(target.last_support_time, obs.last)
        target.last_confirmed_support = max(target.last_confirmed_support, obs.last)
        target.birth_time = min(target.birth_time, obs.first)
        target.track_first_seen = min(target.track_first_seen,
                                      obs.track_first_seen if obs.track_first_seen > 0 else obs.first)

    def merge_observed_new(self, state: PhysicalState, obs: Observation) -> None:
        # One slot, not competing candidates. Every observation that did not belong to CURRENT is
        # unioned here, so geometry that pure-B would have accumulated cannot be lost.
        if state.observed_new is None:
            state.observed_new = self.make_fragment(obs)
            return
        self.merge_observation_into_fragment(state.observed_new, obs)

    def absorb_observed_through(self, state: PhysicalState) -> None:
        if state.current is None or state.observed_new is None:
            return
        # Precondition: a real measurement confirmed CURRENT present through its last_confirmed_support.
        # One physical ID cannot be in two places at one instant, so observations that began no later
        # than that support are more views of the same state.
        current = state.fragments[state.current]
        new = state.observed_new
        if new.birth_time > current.last_confirmed_support:
            return
        state.mobility_continuations += 1
        self._union(current, new)
        current.reconstruction_frames += new.reconstruction_frames
        current.last_support_time = max(current.last_support_time, new.last_support_time)
        current.last_confirmed_support = max(current.last_confirmed_support, new.last_confirmed_support)
        current.birth_time = min(current.birth_time, new.birth_time)
        current.track_first_seen = min(current.track_first_seen, new.track_first_seen)
        state.observed_new = None

    @staticmethod
    def promote_observed_new(state: PhysicalState) -> None:
        if state.observed_new is None or state.current is not None:
            return
        state.fragments.append(state.observed_new)
        state.observed_new = None
        state.current = len(state.fragments) - 1

    @staticmethod
    def close_current(state: PhysicalState, stamp: int) -> None:
        if state.current is None:
            return
        current = state.fragments[state.current]
        # Upper bound, not a measured instant: the state ended somewhere in (last_support, stamp].
        current.death_time = max(stamp, current.last_support_time)
        state.mobility_changes += 1
        state.current = None
        state.has_dynamic_history = True

    def archive_session_state(self, state: PhysicalState, stamp: int) -> None:
        b = state.b_session
        if b is None:
            return
        # Identity conflict or different-site candidate: keep both hypotheses as closed history
        # fragments. Never union them, never delete them.
        if b.current is not None:
            frag = b.fragments[b.current]
            frag.death_time = max(stamp, frag.last_support_time)
            b.current = None                              # archived with the rest below
        self.fold_session_state(state, stamp)

    def hand_over_inherited(self, state: PhysicalState, stamp: int) -> bool:
        self.close_current(state, stamp)
        b = state.b_session
        if b is None or b.current is None:
            return False
        state.fragments.append(b.fragments.pop(b.current))
        b.current = None
        state.current = len(state.fragments) - 1
        return True

    @staticmethod
    def fold_session_state(state: PhysicalState, stamp: int) -> None:
        b = state.b_session
        if b is not None:
            # Nothing of the session state is deleted: its closed fragments and its leftover candidate
            # (a different site: archived, never united) join the identity's history. Its CURRENT, if
            # still set, was just united with the inherited state.
            for i, fragment in enumerate(b.fragments):
                if b.current is not None and i == b.current:
                    continue
                if fragment.death_time is None:
                    fragment.death_time = max(stamp, fragment.last_support_time)
                state.fragments.append(fragment)
            if b.observed_new is not None:
                b.observed_new.death_time = stamp
                state.fragments.append(b.observed_new)
            state.mobility_changes += b.mobility_changes
            state.mobility_continuations += b.mobility_continuations
        state.b_session = None

    # -- ingest (applyPhysicalGeometry / ingestObservation) -------------------------------------
    def ingest_observation(self, state: PhysicalState, obs: Observation, physical_instance_id: int) -> None:
        cur_v = state.fragments[state.current].num_vertices if state.current is not None else 0
        obs_v = state.observed_new.num_vertices if state.observed_new is not None else 0
        self.log.append(f"INGEST inst={physical_instance_id} first={obs.first // 1000000000}s"
                        f" seg_verts={len(obs.points)} cur_verts={cur_v} observed_verts={obs_v}")
        # Nothing established yet: this observation opens the first fragment.
        if state.current is None:
            if state.observed_new is not None:
                self.merge_observed_new(state, obs)
                return
            state.fragments.append(self.make_fragment(obs))
            state.current = len(state.fragments) - 1
            if obs.moved:
                state.mobility_changes += 1               # D1 within the first segment
                state.has_dynamic_history = True
            return
        # NEW_STATE requires direct evidence that the state we hold no longer holds. Tracker motion
        # evidence is exactly that: the object was watched leaving (D1).
        if obs.moved:
            self.close_current(state, obs.first)
            state.fragments.append(self.make_fragment(obs))
            state.current = len(state.fragments) - 1
            state.observed_new = None
            state.pending_absence_stamp = 0
            state.has_dynamic_history = True
            return
        current = state.fragments[state.current]
        if current.requires_current_session_support:
            # Keep inherited and session observations independent until measured evidence resolves
            # their relationship online.
            self.log.append(f"INGEST_DECIDE inst={physical_instance_id} inherited_session_deferred=true")
            if state.b_session is None:
                state.b_session = PhysicalState()
            self.ingest_observation(state.b_session, obs, physical_instance_id)
            return
        # Within one session, two surface maps of the same site refine each other directly when they
        # actually share surface.
        change_probability = self.state_change_probability(state, current)
        shared_probability = self._shared_space(current, obs, change_probability)
        same_session_overlap = shared_probability > change_probability
        self.log.append(f"OVERLAP_POSTERIOR inst={physical_instance_id}"
                        f" shared_probability_lower_bound={_g(shared_probability)}"
                        f" change_prior={_g(change_probability)} same_state={int(same_session_overlap)}")
        # Shared space is not confirmation: if CURRENT was observed empty, with nothing supporting it,
        # while this segment was being observed, one identity cannot be in both places.
        contradicted = same_session_overlap and \
            self.observed_empty_since(current, obs.first, change_probability, physical_instance_id)
        self.log.append(f"INGEST_DECIDE inst={physical_instance_id} same_session_overlap={int(same_session_overlap)}"
                        f" contradicted={int(contradicted)}")
        if same_session_overlap and not contradicted:
            state.pending_absence_stamp = 0
            self.merge_observation_into_fragment(current, obs)
            return
        # Neither current support nor current-session surface overlap: the one replacement slot.
        self.merge_observed_new(state, obs)

    def apply_physical_geometry(self, physical_instance_id: int, segments: Sequence[Observation],
                                merged_first: int) -> Optional[Materialized]:
        """One canonicalization of an identity: its segments (sorted by first, last), locks, ingest,
        then the materialization. merged_first = observationFirstStamp of the merged node."""
        if not segments:
            return None
        state = self.states.setdefault(physical_instance_id, PhysicalState())
        processed_before = bool(state.ingested_intervals)
        to_process = []
        for i, seg in enumerate(segments):
            if processed_before and i == 0 and seg.first == state.last_merged_observation_first:
                continue                                  # anchor lock
            if (seg.first, seg.last) in state.ingested_intervals:
                continue                                  # interval lock
            to_process.append(seg)
        for seg in to_process:
            state.ingested_intervals.add((seg.first, seg.last))
            if len(seg.points) == 0:
                continue                                  # trajectory-only observation
            self.ingest_observation(state, seg, physical_instance_id)
        state.last_merged_observation_first = merged_first
        return self.materialize(physical_instance_id)

    def materialize(self, physical_instance_id: int) -> Optional[Materialized]:
        state = self.states.get(physical_instance_id)
        if state is None:
            return None
        if state.current is not None:
            current = state.fragments[state.current]
            shown = [current]
            b = state.b_session
            if current.requires_current_session_support and b is not None and b.current is not None:
                # The rays of the last round: contradiction outvotes support (P12-P14).
                already_absent = state.last_contradiction_rays > state.last_support_rays
                b_current = b.fragments[b.current]
                # M1i: the same geometry likelihood and persistence prior as M1h.
                posterior = self.same_state_posterior(self.state_change_probability(state, current),
                                                      b_current, current)
                self.log.append(f"MATERIALIZE_POSTERIOR inst={physical_instance_id}{posterior}"
                                f" inherited_verts={current.num_vertices} session_verts={b_current.num_vertices}"
                                f" same_site={int(posterior.same)} already_absent={int(already_absent)}")
                if not already_absent and posterior.same:
                    shown.append(b_current)
            # Presence follows the registry's state decision (P58): present until the state closes.
            return Materialized(shown, True, None)
        if state.fragments:
            # The identity's last state ended with no successor: absent; its presence ends at the
            # midpoint of (last support, closure] (minimum expected risk, uniform departure time).
            session_state = state.b_session is not None and state.b_session.current is not None
            last = None
            for fragment in state.fragments:
                if fragment.death_time is not None and (last is None or fragment.death_time > last.death_time):
                    last = fragment
            end = None
            if not session_state and last is not None:
                seen = max(last.last_support_time, last.last_confirmed_support)
                end = seen + (max(last.death_time, seen) - seen) // 2
            return Materialized([], False, end)
        return None

    # -- direct reports ------------------------------------------------------------------------
    def report_current_contradicted(self, physical_instance_id: int, stamp: int) -> bool:
        state = self.states.get(physical_instance_id)
        if state is None or state.current is None:
            return False
        self.close_current(state, stamp)
        self.promote_observed_new(state)
        return True

    def report_current_supported(self, physical_instance_id: int, stamp: int) -> bool:
        state = self.states.get(physical_instance_id)
        if state is None or state.current is None:
            return False
        current = state.fragments[state.current]
        current.last_support_time = max(current.last_support_time, stamp)
        current.last_confirmed_support = max(current.last_confirmed_support, stamp)
        self.absorb_observed_through(state)
        return True

    # -- terminal round -------------------------------------------------------------------------
    def finalize_pending_absences(self, stamp: int) -> int:
        # Finalization is a callback of the current event, not a measurement.
        self.begin_observation_event(stamp)
        closed = 0
        for i in sorted(self.states):
            state = self.states[i]
            b = state.b_session
            if state.current is None:
                # P01: a session state left without an inherited CURRENT joins the identity at the
                # session end: its CURRENT becomes the identity's CURRENT, everything else its history.
                if b is not None and b.current is not None:
                    state.fragments.append(b.fragments.pop(b.current))
                    b.current = None
                    state.current = len(state.fragments) - 1
                self.fold_session_state(state, stamp)
                if state.current is None:
                    self.promote_observed_new(state)
                state.pending_absence_stamp = 0
                continue
            current = state.fragments[state.current]
            if not current.requires_current_session_support:
                if state.pending_absence_stamp != 0 and state.observed_new is None:
                    self.close_current(state, stamp)
                    closed += 1
                # P01: a session state that outlived its inherited state is archived, not dropped.
                self.archive_session_state(state, stamp)
                state.pending_absence_stamp = 0
                continue
            # Compare the frozen inherited state with the independent B-session state.
            have_b_current = b is not None and b.current is not None
            by_absence = state.last_contradiction_rays > state.last_support_rays
            inherited_absent = by_absence or \
                self.session_copy_elsewhere(state, current, state.last_session_reliable_samples)
            if inherited_absent:
                self.log.append(f"INHERITED_CLOSE inst={i} by_observed_absence={int(by_absence)}"
                                f" by_session_elsewhere={int(not by_absence)} support={state.last_support_rays}"
                                f" contradiction={state.last_contradiction_rays} terminal=1")
                self.hand_over_inherited(state, stamp)
                closed += 1
            elif have_b_current:
                b_current = b.fragments[b.current]
                # M1j: finalization uses the same association posterior as online materialization.
                posterior = self.same_state_posterior(self.state_change_probability(state, current),
                                                      b_current, current)
                self.log.append(f"FINALIZE inst={i}{posterior} inherited_verts={current.num_vertices}"
                                f" session_verts={b_current.num_vertices} same_site={int(posterior.same)}")
                if posterior.same:
                    state.mobility_continuations += 1
                    self._union(current, b_current)
                    current.reconstruction_frames += b_current.reconstruction_frames
                    current.last_support_time = max(current.last_support_time, b_current.last_support_time)
                    current.last_confirmed_support = max(current.last_confirmed_support,
                                                         b_current.last_confirmed_support)
                    current.birth_time = min(current.birth_time, b_current.birth_time)
                    current.track_first_seen = min(current.track_first_seen, b_current.track_first_seen)
                else:
                    # Different site and not absent: identity conflict or a hidden move. Keep the
                    # inherited state CURRENT; archive the B-session hypotheses as closed fragments.
                    self.archive_session_state(state, stamp)
            self.fold_session_state(state, stamp)
            state.pending_absence_stamp = 0
        return closed

    # -- evidence rounds ------------------------------------------------------------------------
    def resolve_current_evidence(self, physical_instance_id: int, inherited_evidence: SurfaceEvidence,
                                 session_evidence: SurfaceEvidence, stamp: int) -> bool:
        self.begin_observation_event(stamp)
        state = self.states.get(physical_instance_id)
        if state is None:
            return False
        cur_v = state.fragments[state.current].num_vertices if state.current is not None else 0
        obs_v = state.observed_new.num_vertices if state.observed_new is not None else 0
        self.log.append(f"EVIDENCE inst={physical_instance_id} inherited_support={inherited_evidence.support_rays}"
                        f" inherited_contradiction={inherited_evidence.contradiction_rays}"
                        f" session_support={session_evidence.support_rays}"
                        f" session_contradiction={session_evidence.contradiction_rays}"
                        f" inherited_absent_flag={int(inherited_evidence.absence_coverage_sufficient)}"
                        f" cur_verts={cur_v} observed_verts={obs_v}")
        # Resolve the independent B-session mini state first. Its D2 decisions are allowed online
        # because both the old and the new observations belong to B.
        b = state.b_session
        if b is not None and b.current is not None:
            self.resolve_support_dominance(b, session_evidence, physical_instance_id, stamp, "SESSION")
        # Keep the inherited geometry separate and evaluate its measured evidence on every
        # reconciliation round, including the terminal round.
        if state.current is not None and state.fragments[state.current].requires_current_session_support:
            inherited = state.fragments[state.current]
            if inherited_evidence.support_rays:
                inherited.last_confirmed_support = max(inherited.last_confirmed_support,
                                                       min(inherited_evidence.latest_support_stamp, stamp))
            state.last_support_rays = inherited_evidence.support_rays
            state.last_contradiction_rays = inherited_evidence.contradiction_rays \
                if inherited_evidence.absence_coverage_sufficient else 0
            state.last_surface_samples = inherited_evidence.surface_samples
            state.last_session_reliable_samples = session_evidence.reliable_samples
            # Online D2/D3 transition: as soon as the B-session state exists and A's old surface is
            # seen through, switch CURRENT to the B state. Do not wait until the end of the session.
            by_absence = state.last_contradiction_rays > inherited_evidence.support_rays
            inherited_absent = by_absence or \
                self.session_copy_elsewhere(state, inherited, session_evidence.reliable_samples)
            if inherited_absent:
                self.log.append(f"INHERITED_CLOSE inst={physical_instance_id} by_observed_absence={int(by_absence)}"
                                f" by_session_elsewhere={int(not by_absence)}"
                                f" support={inherited_evidence.support_rays}"
                                f" contradiction={state.last_contradiction_rays}")
                # Seeing the old site empty closes its state even before the identity is seen
                # elsewhere; so does this session's own established reconstruction standing mostly off
                # the inherited surface (one identity, one pose). P01: the session state joins the
                # identity whether or not it has a CURRENT to hand over.
                self.hand_over_inherited(state, stamp)
                self.fold_session_state(state, stamp)
                state.pending_absence_stamp = 0
                return True
            state.pending_absence_stamp = stamp
            return False
        # A session-local top-level current uses the same support-dominance rule. After an online
        # promotion the current is a normal top-level fragment, so its evidence arrives in the
        # inherited slot (the only non-empty measurement slot).
        if state.current is not None:
            use_inherited_slot = session_evidence.surface_samples == 0 and inherited_evidence.surface_samples > 0
            return self.resolve_support_dominance(state, inherited_evidence if use_inherited_slot
                                                  else session_evidence, physical_instance_id, stamp, "TOP")
        return False

    def resolve_support_dominance(self, b: PhysicalState, evidence: SurfaceEvidence, physical_instance_id: int,
                                  stamp: int, scope: str) -> bool:
        current = b.fragments[b.current]
        self.record_look(current, evidence, stamp)
        support = evidence.support_rays
        contradiction = evidence.contradiction_rays if evidence.absence_coverage_sufficient else 0
        # M1d: a candidate at a different site is the object's current place as soon as nothing
        # supports the old site; the old site is kept as closed history. Without a candidate only the
        # old site seen empty (contradiction outvotes support, P12-P14) closes the state.
        different_site = False
        if b.observed_new is not None:
            same_site_probability = self._extent_same_site(current, b.observed_new)
            q = self.state_change_probability(b, current)
            different_site = same_site_probability < q
            self.log.append(f"EXTENT_POSTERIOR inst={physical_instance_id}"
                            f" same_site_probability={_g(same_site_probability)} change_prior={_g(q)}"
                            f" different_site={int(different_site)}")
        by_new_site = different_site and support == 0
        by_observed_absence = contradiction > support
        if by_new_site or by_observed_absence:
            self.log.append(f"{scope}_CLOSE inst={physical_instance_id} by_new_site={int(by_new_site)}"
                            f" by_observed_absence={int(by_observed_absence)} support={support}"
                            f" contradiction={contradiction}")
            self.close_current(b, stamp)
            self.promote_observed_new(b)
            return True
        if support == 0:
            return False
        # Absorbing a candidate presupposes that CURRENT was confirmed present; a decision at t=20 may
        # only contain support observed at t=5.
        current.last_confirmed_support = max(current.last_confirmed_support,
                                             min(evidence.latest_support_stamp, stamp))
        if b.observed_new is None:
            return False                                  # no candidate to absorb
        # M1k/M1l: CURRENT is the supported measurement; the candidate is the shape hypothesized to
        # explain it, absorbed only when it is the same state.
        posterior = self.same_state_posterior(self.state_change_probability(b, current), current, b.observed_new)
        self.log.append(f"{scope}_ABSORB inst={physical_instance_id}{posterior}"
                        f" candidate_vertices={b.observed_new.num_vertices} support={support}"
                        f" absorb={int(posterior.same)}")
        if posterior.same:
            self.absorb_observed_through(b)
        return False

    # -- session boundary (initializeFromObjects) ---------------------------------------------
    def initialize_from_objects(self, objects: Sequence[dict]) -> None:
        """objects: the previous session's canonical object nodes, each dict(identity, points, normals,
        first, last, semantic, bbox_valid, dynamic, mobility_changes, mobility_continuations)."""
        for obj in objects:
            i = obj["identity"]
            # A node carries only the CURRENT materialization, so a seeded ID starts with exactly one
            # fragment. The previous session's history is not recoverable from the node.
            state = self.states.setdefault(i, PhysicalState())
            state.fragments = []
            state.observed_new = None
            state.pending_absence_stamp = 0
            state.current = None
            if obj.get("bbox_valid", True):
                obs = Observation(identity=i, points=obj["points"], normals=obj.get("normals"),
                                  first=obj["first"], last=obj["last"], semantic=obj["semantic"],
                                  moved=obj.get("dynamic", False), elements=obj.get("elements"),
                                  track_first_seen=obj.get("track_first_seen", 0),
                                  reconstruction_frames=obj.get("reconstruction_frames", 0))
                state.fragments.append(self.make_fragment(obs))
                state.current = 0
                # A's observation is the state we inherit, but it is not a B-ray measurement.
                state.fragments[-1].last_confirmed_support = 0
                state.fragments[-1].requires_current_session_support = True
            state.last_merged_observation_first = obj["first"]
            state.ingested_intervals = {(obj["first"], obj["last"])}
            state.has_dynamic_history = bool(obj.get("dynamic", False))
            state.mobility_changes = obj["mobility_changes"] if "mobility_changes" in obj \
                else (1 if state.has_dynamic_history else 0)
            state.mobility_continuations = obj.get("mobility_continuations", 0)

    def clear(self) -> None:
        self.states.clear()
        self.event_history.clear()
        self.event_open = False
        self.event_stamp = 0

    # -- views ----------------------------------------------------------------------------------
    def num_states(self) -> int:
        return len(self.states)

    def has_state(self, physical_instance_id: int) -> bool:
        return physical_instance_id in self.states

    def tracked_ids(self) -> List[int]:
        return sorted(self.states)

    def current_fragment(self, physical_instance_id: int) -> Optional[Fragment]:
        s = self.states.get(physical_instance_id)
        return None if s is None or s.current is None else s.fragments[s.current]

    def session_current_fragment(self, physical_instance_id: int) -> Optional[Fragment]:
        s = self.states.get(physical_instance_id)
        if s is None or s.b_session is None or s.b_session.current is None:
            return None
        return s.b_session.fragments[s.b_session.current]

    def history_fragments(self, physical_instance_id: int) -> List[Fragment]:
        s = self.states.get(physical_instance_id)
        return [] if s is None else list(s.fragments)

    def observed_new(self, physical_instance_id: int) -> Optional[Fragment]:
        s = self.states.get(physical_instance_id)
        return None if s is None else s.observed_new

    def unresolved_candidates(self, physical_instance_id: int) -> List[Fragment]:
        o = self.observed_new(physical_instance_id)
        return [] if o is None else [o]

    def state_intervals(self) -> Dict[int, List[Tuple[int, Optional[int]]]]:
        """Read-only view for representation backends (layer addition, no decision reads it): (birth,
        death) of every state of every identity, committed or pending; death None = still open."""
        def frags(state: PhysicalState) -> List[Fragment]:
            out = list(state.fragments)
            if state.observed_new is not None:
                out.append(state.observed_new)
            if state.b_session is not None:
                out += frags(state.b_session)
            return out
        return {i: [(f.birth_time, f.death_time) for f in frags(s)] for i, s in self.states.items()}
