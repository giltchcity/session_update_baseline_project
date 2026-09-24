"""Identity -> temporal-fragment state machine.

Representation-free port of t2 (commit 43c663d)
khronos/src/backend/reconciliation/persistent_object_state.cpp. A fragment's
geometry is a world-space point set (with optional normals) instead of a
spark_dsg::Mesh in its bounding-box frame; every geometric test in t2 already
works on the mesh vertices only (voxel co-occupancy, axis-aligned extent, nearest
vertex within a tolerance), so the decisions are unchanged. Mesh faces were only
carried along by concatenation.

States of one physical identity (see the t2 header for the full rationale):
  fragments      closed history + at most one CURRENT
  observed_new   one accumulated replacement slot, materialised nowhere
  b_session      independent session state while CURRENT is inherited
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .evidence import SurfaceEvidence

K_STATE_TOLERANCE = 0.10       # protocol_v1 state granularity (m)
K_ESTABLISHED_SAMPLES = 30     # = RayVerificator kMinSamplesInView


@dataclass
class Observation:
    """One geometry-bearing observation segment of a physical identity.

    t2 receives these from Khronos' tracker/extractor as DSG object nodes; any backend
    can produce them from instance-masked measurements. `moved` is direct evidence that
    the identity was watched moving (Khronos: kHasDynamicHistoryDetail).
    """
    identity: int
    points: np.ndarray                    # Nx3 world
    normals: Optional[np.ndarray]         # Nx3 or None
    first: int
    last: int
    semantic: int = -1
    moved: bool = False
    reconstruction_frames: int = 0
    elements: Optional[np.ndarray] = None  # backend element ids carried with the geometry


@dataclass
class Look:
    stamp: int
    support_rays: int
    reliable_in_view: int
    reliable_seen_through: int


@dataclass
class Fragment:
    points: np.ndarray
    normals: Optional[np.ndarray]
    elements: np.ndarray
    birth_time: int = 0
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
    last_geometric_support: int = 0
    last_surface_samples: int = 0
    last_session_reliable_samples: int = 0
    ingested_intervals: Set[Tuple[int, int]] = field(default_factory=set)
    has_dynamic_history: bool = False


# -- geometric helpers (vertex-only, as in t2) --------------------------------------------
def _keys(points: np.ndarray, resolution: float) -> np.ndarray:
    return np.floor(points / resolution).astype(np.int64)


def _key_set(points: np.ndarray, resolution: float) -> Set[Tuple[int, int, int]]:
    return set(map(tuple, _keys(points, resolution).tolist()))


def surfaces_share_space(a: np.ndarray, b: np.ndarray, resolution: float) -> bool:
    if len(a) == 0 or len(b) == 0:
        return False
    occupied = _key_set(a, resolution)
    return any(k in occupied for k in map(tuple, _keys(b, resolution).tolist()))


def candidate_within_current_extent(cur: np.ndarray, cand: np.ndarray, resolution: float) -> bool:
    if len(cur) == 0 or len(cand) == 0:
        return False
    a_lo, a_hi = cur.min(axis=0), cur.max(axis=0)
    b_lo, b_hi = cand.min(axis=0), cand.max(axis=0)
    return bool(np.all(a_lo - resolution <= b_hi) and np.all(b_lo - resolution <= a_hi))


_NEIGHBOURS = np.array([(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
                       dtype=np.int64)


def shared_surface_samples(cur: np.ndarray, cand: np.ndarray, resolution: float) -> int:
    """Vertices of `cur` in the same or a directly neighbouring voxel as a vertex of `cand`."""
    if len(cur) == 0 or len(cand) == 0:
        return 0
    cand_keys = _key_set(cand, resolution)
    shared = 0
    for k in _keys(cur, resolution):
        if any(tuple(k + d) in cand_keys for d in _NEIGHBOURS):
            shared += 1
    return shared


def off_state_share(copy_pts: np.ndarray, reference: np.ndarray, tolerance: float) -> float:
    """Share of `copy_pts` farther than `tolerance` from every point of `reference`."""
    if len(copy_pts) == 0:
        return 0.0
    if len(reference) == 0:
        return 1.0
    from scipy.spatial import cKDTree
    d, _ = cKDTree(reference).query(copy_pts, k=1, distance_upper_bound=tolerance * (1 + 1e-6))
    return float(np.mean(~(d <= tolerance)))


def _concat(a: Optional[np.ndarray], b: Optional[np.ndarray], na: int, nb: int) -> Optional[np.ndarray]:
    if a is None and b is None:
        return None
    a = a if a is not None else np.full((na, 3), np.nan)
    b = b if b is not None else np.full((nb, 3), np.nan)
    return np.concatenate([a, b])


class PersistentObjectState:
    def __init__(self, map_resolution: float = 0.05, high_mobility_semantics: Sequence[int] = ()):
        self.states: Dict[int, PhysicalState] = {}
        self.map_resolution = map_resolution
        self.high_mobility = set(high_mobility_semantics)
        self.log: List[str] = []

    # -- helpers --------------------------------------------------------------------------
    def is_high_mobility(self, state: PhysicalState, current: Fragment) -> bool:
        if state.has_dynamic_history:
            return True
        if any(f.death_time is not None for f in state.fragments):
            return True
        return current.semantic_label in self.high_mobility

    @staticmethod
    def make_fragment(obs: Observation) -> Fragment:
        elements = obs.elements if obs.elements is not None else np.zeros(0, dtype=np.int64)
        return Fragment(points=obs.points.copy(),
                        normals=None if obs.normals is None else obs.normals.copy(),
                        elements=elements.copy(), birth_time=obs.first, last_support_time=obs.last,
                        last_confirmed_support=obs.last, semantic_label=obs.semantic,
                        reconstruction_frames=obs.reconstruction_frames)

    @staticmethod
    def _append(target: Fragment, pts, nrm, elements) -> None:
        n0 = len(target.points)
        target.normals = _concat(target.normals, nrm, n0, len(pts))
        target.points = np.concatenate([target.points, pts])
        target.elements = np.concatenate([target.elements, elements])

    def merge_observation_into_fragment(self, target: Fragment, obs: Observation) -> None:
        el = obs.elements if obs.elements is not None else np.zeros(0, dtype=np.int64)
        self._append(target, obs.points, obs.normals, el)
        target.reconstruction_frames += obs.reconstruction_frames
        target.last_support_time = max(target.last_support_time, obs.last)
        target.last_confirmed_support = max(target.last_confirmed_support, obs.last)
        target.birth_time = min(target.birth_time, obs.first)

    def merge_observed_new(self, state: PhysicalState, obs: Observation) -> None:
        if state.observed_new is None:
            state.observed_new = self.make_fragment(obs)
            return
        self.merge_observation_into_fragment(state.observed_new, obs)

    def absorb_observed_through(self, state: PhysicalState, stamp: int) -> None:
        if state.current is None or state.observed_new is None:
            return
        if state.observed_new.birth_time > stamp:
            return
        cur = state.fragments[state.current]
        new = state.observed_new
        self._append(cur, new.points, new.normals, new.elements)
        cur.reconstruction_frames += new.reconstruction_frames
        cur.last_support_time = max(cur.last_support_time, new.last_support_time)
        cur.last_confirmed_support = max(cur.last_confirmed_support, new.last_confirmed_support)
        cur.birth_time = min(cur.birth_time, new.birth_time)
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
        cur = state.fragments[state.current]
        cur.death_time = max(stamp, cur.last_support_time)
        state.current = None
        state.has_dynamic_history = True

    @staticmethod
    def archive_session_state(state: PhysicalState, stamp: int) -> None:
        b = state.b_session
        if b is None:
            return
        if b.current is not None:
            frag = b.fragments[b.current]
            frag.death_time = max(stamp, frag.last_support_time)
            state.fragments.append(frag)
            b.current = None
        if b.observed_new is not None:
            b.observed_new.death_time = stamp
            state.fragments.append(b.observed_new)
            b.observed_new = None
        state.b_session = None

    @staticmethod
    def record_look(fragment: Fragment, ev: SurfaceEvidence, stamp: int) -> None:
        if ev.surface_samples == 0:
            return
        fragment.looks.append(Look(stamp, ev.support_rays, ev.reliable_in_view, ev.reliable_seen_through))

    @staticmethod
    def observed_empty_since(fragment: Fragment, since: int) -> bool:
        support = judged = seen = 0
        for look in fragment.looks:
            if look.stamp > since:
                support += look.support_rays
                judged += look.reliable_in_view
                seen += look.reliable_seen_through
        return seen > 0 and seen == judged and support == 0

    def session_copy_elsewhere(self, state: PhysicalState, inherited: Fragment,
                               session_reliable_samples: int) -> bool:
        if state.b_session is None or state.b_session.current is None:
            return False
        if not self.is_high_mobility(state, inherited):
            return False
        if session_reliable_samples < K_ESTABLISHED_SAMPLES:
            return False
        copy_frag = state.b_session.fragments[state.b_session.current]
        off = off_state_share(copy_frag.points, inherited.points, K_STATE_TOLERANCE)
        return off > 0.5

    @staticmethod
    def inherited_evidence_absent(support: int, contradiction: int) -> bool:
        return contradiction > support

    # -- ingest (applyPhysicalGeometry / ingestObservation) --------------------------------
    def ingest(self, obs: Observation) -> None:
        state = self.states.setdefault(obs.identity, PhysicalState())
        if (obs.first, obs.last) in state.ingested_intervals:
            return                                       # interval lock
        state.ingested_intervals.add((obs.first, obs.last))
        if len(obs.points) == 0:
            return                                       # trajectory-only observation
        self._ingest(state, obs)

    def _ingest(self, state: PhysicalState, obs: Observation) -> None:
        if state.current is None:
            if state.observed_new is not None:
                self.merge_observed_new(state, obs)
                return
            state.fragments.append(self.make_fragment(obs))
            state.current = len(state.fragments) - 1
            if obs.moved:
                state.has_dynamic_history = True
            return
        if obs.moved:                                    # watched moving (D1)
            self.close_current(state, obs.first)
            state.fragments.append(self.make_fragment(obs))
            state.current = len(state.fragments) - 1
            state.observed_new = None
            state.pending_absence_stamp = 0
            state.has_dynamic_history = True
            return
        cur = state.fragments[state.current]
        if cur.requires_current_session_support:
            if state.b_session is None:
                state.b_session = PhysicalState()
            self._ingest(state.b_session, obs)
            return
        overlap = surfaces_share_space(cur.points, obs.points, self.map_resolution)
        contradicted = overlap and self.observed_empty_since(cur, obs.first)
        if overlap and not contradicted:
            state.pending_absence_stamp = 0
            self.merge_observation_into_fragment(cur, obs)
            return
        self.merge_observed_new(state, obs)

    # -- materialisation --------------------------------------------------------------------
    def displayed(self, identity: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(points, element ids) the map shows for this identity now (applyPhysicalGeometry)."""
        state = self.states.get(identity)
        if state is None or state.current is None:
            return None
        cur = state.fragments[state.current]
        if cur.requires_current_session_support and state.b_session is not None and \
                state.b_session.current is not None:
            already_absent = self.inherited_evidence_absent(state.last_support_rays,
                                                            state.last_contradiction_rays)
            b_cur = state.b_session.fragments[state.b_session.current]
            shared = shared_surface_samples(cur.points, b_cur.points, self.map_resolution)
            same_site = (not self.is_high_mobility(state, cur) or shared > 0) and \
                not self.session_copy_elsewhere(state, cur, state.last_session_reliable_samples)
            if not already_absent and same_site:
                return (np.concatenate([cur.points, b_cur.points]),
                        np.concatenate([cur.elements, b_cur.elements]))
        return cur.points, cur.elements

    # -- evidence (resolveCurrentEvidence) ---------------------------------------------------
    def resolve_current_evidence(self, identity: int, inherited: SurfaceEvidence,
                                 session: SurfaceEvidence, stamp: int) -> bool:
        state = self.states.get(identity)
        if state is None:
            return False
        if state.b_session is not None:
            b = state.b_session
            if b.current is not None:
                self._resolve_measured(b, identity, session, stamp, "SESSION")
        if state.current is not None and state.fragments[state.current].requires_current_session_support:
            inh = state.fragments[state.current]
            if inherited.support_rays:
                inh.last_confirmed_support = max(inh.last_confirmed_support,
                                                 min(inherited.latest_support_stamp, stamp))
            state.last_support_rays = inherited.support_rays
            state.last_contradiction_rays = inherited.contradiction_rays \
                if inherited.absence_coverage_sufficient else 0
            state.last_surface_samples = inherited.surface_samples
            state.last_session_reliable_samples = session.reliable_samples
            state.last_geometric_support = shared_surface_samples(
                inh.points, state.b_session.fragments[state.b_session.current].points,
                self.map_resolution) if state.b_session is not None and \
                state.b_session.current is not None else 0
            absent = self.inherited_evidence_absent(inherited.support_rays,
                                                    state.last_contradiction_rays) or \
                self.session_copy_elsewhere(state, inh, session.reliable_samples)
            if absent:
                self.close_current(state, stamp)
                self.log.append(f"{stamp} CLOSE_INHERITED inst={identity}")
                if state.b_session is None or state.b_session.current is None:
                    state.pending_absence_stamp = 0
                    return True
                b = state.b_session
                state.fragments.append(b.fragments[b.current])
                b.current = None
                state.current = len(state.fragments) - 1
                if b.observed_new is not None:
                    b.observed_new.death_time = stamp
                    state.fragments.append(b.observed_new)
                    b.observed_new = None
                state.b_session = None
                state.pending_absence_stamp = 0
                return True
            state.pending_absence_stamp = stamp
            return False
        if state.current is not None:
            use_inherited = session.surface_samples == 0 and inherited.surface_samples > 0
            ev = inherited if use_inherited else session
            return self._resolve_measured(state, identity, ev, stamp, "TOP")
        return False

    def _resolve_measured(self, b: PhysicalState, identity: int, ev: SurfaceEvidence, stamp: int,
                          tag: str) -> bool:
        cur = b.fragments[b.current]
        self.record_look(cur, ev, stamp)
        support = ev.support_rays
        contradiction = ev.contradiction_rays if ev.absence_coverage_sufficient else 0
        samples = ev.surface_samples
        geom = shared_surface_samples(cur.points, b.observed_new.points, self.map_resolution) \
            if b.observed_new is not None else 0
        scale = float(samples) if samples > 0 else 1.0
        support_rate = support / scale
        contradiction_rate = contradiction / scale
        different_site = b.observed_new is not None and not candidate_within_current_extent(
            cur.points, b.observed_new.points, self.map_resolution)
        if (different_site and support_rate <= 0.0) or contradiction_rate > support_rate:
            self.log.append(f"{stamp} {tag}_CLOSE inst={identity} by_new_site="
                            f"{int(different_site and support_rate <= 0.0)} by_observed_absence="
                            f"{int(contradiction_rate > support_rate)}")
            self.close_current(b, stamp)
            self.promote_observed_new(b)
            if tag == "SESSION":
                b.has_dynamic_history = True
            return tag == "TOP"
        if support_rate > 0.0:
            cur.last_confirmed_support = max(cur.last_confirmed_support,
                                             min(ev.latest_support_stamp, stamp))
            if not self.is_high_mobility(b, cur) or geom > 0:
                self.absorb_observed_through(b, stamp)
        return False

    # -- direct reports (reportCurrentContradicted / reportCurrentSupported) -----------------
    def report_current_contradicted(self, identity: int, stamp: int) -> bool:
        state = self.states.get(identity)
        if state is None or state.current is None:
            return False
        self.close_current(state, stamp)
        self.promote_observed_new(state)
        return True

    def report_current_supported(self, identity: int, stamp: int) -> bool:
        state = self.states.get(identity)
        if state is None or state.current is None:
            return False
        cur = state.fragments[state.current]
        cur.last_support_time = max(cur.last_support_time, stamp)
        cur.last_confirmed_support = max(cur.last_confirmed_support, stamp)
        self.absorb_observed_through(state, stamp)
        return True

    def unresolved_candidates(self, identity: int) -> List[Fragment]:
        s = self.states.get(identity)
        return [] if s is None or s.observed_new is None else [s.observed_new]

    # -- terminal round (finalizePendingAbsences) -------------------------------------------
    def finalize_pending_absences(self, stamp: int) -> int:
        closed = 0
        for identity, state in self.states.items():
            if state.current is None:
                self.promote_observed_new(state)
                state.pending_absence_stamp = 0
                continue
            cur = state.fragments[state.current]
            if not cur.requires_current_session_support:
                if state.pending_absence_stamp != 0 and state.observed_new is None:
                    self.close_current(state, stamp)
                    closed += 1
                state.pending_absence_stamp = 0
                continue
            have_b = state.b_session is not None and state.b_session.current is not None
            absent = self.inherited_evidence_absent(state.last_support_rays,
                                                    state.last_contradiction_rays) or \
                self.session_copy_elsewhere(state, cur, state.last_session_reliable_samples)
            if absent:
                self.close_current(state, stamp)
                if have_b:
                    b = state.b_session
                    state.fragments.append(b.fragments[b.current])
                    b.current = None
                    state.current = len(state.fragments) - 1
                    if b.observed_new is not None:
                        b.observed_new.death_time = stamp
                        state.fragments.append(b.observed_new)
                        b.observed_new = None
                closed += 1
            elif have_b:
                b = state.b_session
                b_cur = b.fragments[b.current]
                shared = shared_surface_samples(cur.points, b_cur.points, self.map_resolution)
                if not self.is_high_mobility(state, cur) or shared > 0:
                    self._append(cur, b_cur.points, b_cur.normals, b_cur.elements)
                    cur.reconstruction_frames += b_cur.reconstruction_frames
                    cur.last_support_time = max(cur.last_support_time, b_cur.last_support_time)
                    cur.last_confirmed_support = max(cur.last_confirmed_support,
                                                     b_cur.last_confirmed_support)
                    cur.birth_time = min(cur.birth_time, b_cur.birth_time)
                else:
                    self.archive_session_state(state, stamp)
            state.b_session = None
            state.pending_absence_stamp = 0
        return closed

    # -- session boundary (initializeFromObjects) -------------------------------------------
    def seed_inherited(self, identity: int, fragment: Fragment, has_dynamic_history: bool,
                       interval: Tuple[int, int]) -> None:
        """An exported CURRENT fragment of the previous session becomes this session's CURRENT."""
        state = PhysicalState()
        frag = copy.deepcopy(fragment)
        frag.looks = []
        frag.death_time = None
        frag.last_confirmed_support = 0
        frag.requires_current_session_support = True
        state.fragments.append(frag)
        state.current = 0
        state.ingested_intervals = {interval}
        state.has_dynamic_history = has_dynamic_history
        self.states[identity] = state

    # -- views --------------------------------------------------------------------------------
    def tracked_ids(self) -> List[int]:
        return sorted(self.states)

    def current_fragment(self, identity: int) -> Optional[Fragment]:
        s = self.states.get(identity)
        return None if s is None or s.current is None else s.fragments[s.current]

    def session_current_fragment(self, identity: int) -> Optional[Fragment]:
        s = self.states.get(identity)
        if s is None or s.b_session is None or s.b_session.current is None:
            return None
        return s.b_session.fragments[s.b_session.current]

    def history(self, identity: int) -> List[Fragment]:
        s = self.states.get(identity)
        return [] if s is None else list(s.fragments)
