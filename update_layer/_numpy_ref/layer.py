"""Round driver: a posed RGB-D mapping backend plus the representation-free update layer.

Per round (t2 Backend::runChangeDetectionThread order):
  1. judge every stored object state against the round's frames (observed-absence test,
     projected support/contradiction) and resolve it (registry.resolve_current_evidence);
  2. judge stored background elements (element rule, below) and retire the absent ones;
  3. ingest the round's new observations into the identity registry;
  4. emit a snapshot: background elements alive + the object states the registry displays.
Terminal round: 1-3, finalize pending absences, 1 again, finalize, snapshot.

The backend here is a point/surfel map: background elements are voxel-hashed points with a
normal and a semantic label; an object's observation in a round is its instance-masked
points. Swapping the backend changes only integrate() / elements / surface samples.

Element rule (new; t2 has no representation-free background rule -- its background uses
Khronos' mesh-ray detector). Each round an element gets its latest verdict from projection:
on-surface (|range difference| <= the 5 cm sensor tolerance), seen-through (measured beyond it,
facing within the 60 deg incidence limit), or nothing. A seen-through verdict adds ln(1/p_M),
p_M = the sensor's pooled see-through share of present surfaces that the object test learns
(SensorStatistics); an on-surface verdict resets the sum; the element is retired above ln 99.
The same test and threshold as for object states, at element granularity; no new constant.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .evidence import (EvidenceConfig, EvidenceStore, ObservedAbsence, SensorStatistics,
                       SurfaceEvidence, UNAVAILABLE, INVALID)
from .frames import Frame, FlatSession, SessionSpec, backproject
from .registry import Fragment, Observation, PersistentObjectState

OPEN = np.iinfo(np.int64).max
LN99 = math.log(99.0)


@dataclass
class LayerConfig:
    round_s: float                         # reconciliation period (t2: real 2.1 s, synthetic 10.8 s)
    map_resolution: float                  # t2 active-window voxel size (real 0.05, synthetic 0.10)
    high_mobility: Sequence[int] = ()
    object_semantics: Sequence[int] = ()   # label space: object classes (UNIDENTIFIED when no id)
    dynamic_semantics: Sequence[int] = ()  # label space: dynamic classes (never integrated)
    evidence_hz: float = 5.0
    pixel_step: int = 2
    min_cluster_px_full: int = 50          # InstanceForwarding min_cluster_size (full resolution)
    element_voxel: float = 0.02
    object_voxel: float = 0.01
    use_layer: bool = True                 # False = naive memory (keep everything inherited)
    # Observation segments follow Khronos' frontend cadence (active_window.cpp): a track ends
    # after temporal_window unobserved (death extraction); a stable active track emits a chunk
    # every max_buffer_size * store_every_n_frames observations; both carry the trailing buffer
    # window. Configs: temporal_window 3 s, buffer 100 x every 3rd frame, 30 Hz input,
    # tracker min_num_observations 15.
    segments: str = "tracks"               # "tracks" (Khronos cadence) or "rounds"
    temporal_window_s: float = 3.0
    buffer_frames_30hz: int = 300
    min_observations_30hz: int = 15
    element_rule_hz: float = 1.0
    # Element-only baselines (no identity states): object points are elements carrying their
    # identity. "carve" retires an element on its first seen-through verdict (per-frame carving,
    # the representation-level update of DynaMem / Khronos' ray check); "elements" applies the
    # calibrated element test to everything.
    objects_as_elements: bool = False
    single_look: bool = False
    decision: str = "cusum"                 # absence-decision ablation (EvidenceConfig.decision)


def pixel_normals(frame: Frame) -> np.ndarray:
    """Per-pixel world normals from depth differences (NaN where undefined), towards the camera."""
    K, d = frame.K, frame.depth
    h, w = d.shape
    u = (np.arange(w) + K.offset - K.cx) / K.fx
    v = (np.arange(h) + K.offset - K.cy) / K.fy
    P = np.stack([u[None, :] * d, v[:, None] * d, d], axis=-1)
    dx = np.full_like(P, np.nan)
    dy = np.full_like(P, np.nan)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1, :] = P[2:, :] - P[:-2, :]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    flip = np.einsum("ijk,ijk->ij", n, P) > 0
    n[flip] *= -1
    return n @ frame.T_world_cam[:3, :3].T


class ElementStore:
    """Background surface elements of a point/surfel map, voxel-hashed, with lifetimes."""

    def __init__(self, voxel: float):
        self.voxel = voxel
        self.keys = np.zeros(0, dtype=np.int64)      # sorted
        self.order = np.zeros(0, dtype=np.int64)     # element index per sorted key
        self.n = 0
        cap = 1 << 16
        self.sum = np.zeros((cap, 3))
        self.cnt = np.zeros(cap)
        self.nrm = np.zeros((cap, 3))
        self.label = np.zeros(cap, dtype=np.int64)
        self.birth = np.zeros(cap, dtype=np.int64)
        self.last_seen = np.zeros(cap, dtype=np.int64)
        self.death = np.full(cap, OPEN, dtype=np.int64)
        self.inherited = np.zeros(cap, dtype=bool)
        self.cusum = np.zeros(cap)
        self.identity = np.zeros(cap, dtype=np.int64)

    def _grow(self, need: int) -> None:
        cap = len(self.cnt)
        if need <= cap:
            return
        new = max(need, cap * 2)
        for name in ("sum", "cnt", "nrm", "label", "birth", "last_seen", "death", "inherited", "cusum",
                     "identity"):
            a = getattr(self, name)
            b = np.zeros((new,) + a.shape[1:], dtype=a.dtype)
            if name == "death":
                b[:] = OPEN
            b[:cap] = a
            setattr(self, name, b)

    def encode(self, pts: np.ndarray, identity: Optional[np.ndarray] = None) -> np.ndarray:
        # 18 bits per axis (+-2.6 km at 2 cm) and the identity (< 512) in the top 9 bits:
        # one element per (voxel, identity), so objects never share an element with background.
        k = np.floor(pts / self.voxel).astype(np.int64) + (1 << 17)
        key = (k[:, 0] << 36) | (k[:, 1] << 18) | k[:, 2]
        if identity is not None:
            key |= identity.astype(np.int64) << 54
        return key

    def lookup(self, keys: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(self.keys, keys)
        pos = np.minimum(pos, max(len(self.keys) - 1, 0))
        hit = (len(self.keys) > 0) & (self.keys[pos] == keys) if len(self.keys) else np.zeros(len(keys), bool)
        out = np.full(len(keys), -1, dtype=np.int64)
        out[hit] = self.order[pos[hit]]
        return out

    def add(self, pts: np.ndarray, nrm: np.ndarray, lab: np.ndarray, stamp: int,
            inherited: bool = False, birth: Optional[np.ndarray] = None,
            identity: Optional[np.ndarray] = None) -> None:
        if len(pts) == 0:
            return
        keys = self.encode(pts, identity)
        uk, inv = np.unique(keys, return_inverse=True)
        idx = self.lookup(uk)
        # a key whose element was retired starts a new element (the surface re-appeared)
        dead = idx >= 0
        dead[dead] = self.death[idx[dead]] != OPEN
        new = (idx < 0) | dead
        n_new = int(new.sum())
        if n_new:
            self._grow(self.n + n_new)
            fresh = np.arange(self.n, self.n + n_new)
            idx[new] = fresh
            self.birth[fresh] = stamp
            self.death[fresh] = OPEN
            self.inherited[fresh] = inherited
            self.n += n_new
            live_keys = np.concatenate([self.keys[~np.isin(self.keys, uk[new])], uk[new]])
            live_order = np.concatenate([self.order[~np.isin(self.keys, uk[new])], idx[new]])
            s = np.argsort(live_keys, kind="stable")
            self.keys, self.order = live_keys[s], live_order[s]
        el = idx[inv]
        if identity is not None:
            self.identity[el] = identity
        ok = np.isfinite(nrm).all(axis=1)
        np.add.at(self.sum, el, pts)
        np.add.at(self.cnt, el, 1.0)
        np.add.at(self.nrm, el[ok], nrm[ok])
        self.label[el] = lab
        self.last_seen[idx] = stamp
        self.cusum[idx] = 0.0
        if birth is not None:
            self.birth[el] = birth

    def alive(self) -> np.ndarray:
        return np.flatnonzero(self.death[: self.n] == OPEN)

    def xyz(self, ids: np.ndarray) -> np.ndarray:
        return self.sum[ids] / self.cnt[ids, None]

    def normals(self, ids: np.ndarray) -> np.ndarray:
        v = self.nrm[ids]
        l = np.linalg.norm(v, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(l > 1e-9, v / l, np.nan)


@dataclass
class Track:
    obs: List[tuple] = field(default_factory=list)      # (stamp, points, normals)
    count: int = 0
    submitted: int = 0
    last: int = 0
    sem: Dict[int, int] = field(default_factory=dict)


@dataclass
class RoundBuffer:
    pts: Dict[int, List[np.ndarray]] = field(default_factory=dict)
    nrm: Dict[int, List[np.ndarray]] = field(default_factory=dict)
    first: Dict[int, int] = field(default_factory=dict)
    last: Dict[int, int] = field(default_factory=dict)
    frames: Dict[int, int] = field(default_factory=dict)
    sem: Dict[int, Dict[int, int]] = field(default_factory=dict)
    stamps: List[int] = field(default_factory=list)


@dataclass
class SessionResult:
    stamps: List[int]
    display: List[List[Tuple[int, int, List[Tuple[int, int]]]]]  # per snapshot: (identity, semantic, [(uid, n)])
    fragments: Dict[int, Fragment]
    fragment_identity: Dict[int, int]
    elements: ElementStore
    registry: PersistentObjectState
    stats: SensorStatistics
    log: List[str]
    naive: Optional[dict] = None


class UpdateLayerSession:
    def __init__(self, cfg: LayerConfig, prior: Optional[dict] = None,
                 stats: Optional[SensorStatistics] = None):
        self.cfg = cfg
        self.store = EvidenceStore(cfg.object_semantics, cfg.dynamic_semantics)
        self.stats = stats or SensorStatistics()
        self.absence = ObservedAbsence(self.store, EvidenceConfig(map_resolution=cfg.map_resolution,
                                                                  decision=cfg.decision), self.stats)
        self.registry = PersistentObjectState(cfg.map_resolution, cfg.high_mobility)
        self.elements = ElementStore(cfg.element_voxel)
        self.fragments: Dict[int, Fragment] = {}
        self.fragment_identity: Dict[int, int] = {}
        self.display: List = []
        self.stamps: List[int] = []
        self.log: List[str] = []
        self.semantic: Dict[int, int] = {}
        self.naive: Dict[int, Fragment] = {}     # naive memory: one union per identity
        self.tracks: Dict[int, Track] = {}
        self.pending: List[Observation] = []     # extracted segments awaiting the next ingest
        self.prior = prior

    # -- fragment identities (uid follows the Fragment object) ------------------------------
    def _uid(self, frag: Fragment, identity: int) -> int:
        uid = id(frag)
        self.fragments[uid] = frag
        self.fragment_identity[uid] = identity
        return uid

    def seed(self, start_ns: int) -> None:
        """Load the previous session's exported state (D3 across a process boundary)."""
        p = self.prior
        if p is None:
            return
        if len(p["el_xyz"]):
            self.elements.add(p["el_xyz"], p["el_nrm"], p["el_label"], start_ns, inherited=True,
                              birth=None, identity=p.get("el_identity") if self.cfg.objects_as_elements else None)
        for obj in p["objects"]:
            frag = Fragment(points=obj["points"], normals=obj["normals"],
                            elements=np.zeros(0, dtype=np.int64), birth_time=obj["first"],
                            last_support_time=obj["last"], last_confirmed_support=obj["last"],
                            semantic_label=obj["semantic"])
            self.registry.seed_inherited(obj["identity"], frag, obj["dynamic"], (obj["first"], obj["last"]))
            self.semantic[obj["identity"]] = obj["semantic"]
        # the inherited scene is the session's first snapshot, stamped with the previous session's
        # final time (t2 keeps that seed snapshot inside the map)
        self._snapshot(int(p.get("final_stamp", start_ns)))

    # -- per frame --------------------------------------------------------------------------
    def integrate(self, frame: Frame, buf: RoundBuffer) -> None:
        cfg = self.cfg
        dyn = np.zeros(frame.depth.shape, dtype=bool)
        if frame.semantic is not None and len(cfg.dynamic_semantics):
            dyn = (frame.instance <= 0) & np.isin(frame.semantic, np.asarray(cfg.dynamic_semantics))
        self.store.ingest(frame, dyn)
        normals = pixel_normals(frame)
        valid = np.isfinite(frame.depth)
        if cfg.objects_as_elements:
            pts, v, u = backproject(frame, valid & ~dyn)
            lab = frame.semantic[v, u] if frame.semantic is not None else np.zeros(len(v), np.int64)
            self.elements.add(pts, normals[v, u], lab, frame.stamp_ns,
                              identity=np.maximum(frame.instance[v, u], 0))
            buf.stamps.append(frame.stamp_ns)
            return
        bg = valid & (frame.instance <= 0) & ~dyn
        pts, v, u = backproject(frame, bg)
        lab = frame.semantic[v, u] if frame.semantic is not None else np.zeros(len(v), np.int64)
        self.elements.add(pts, normals[v, u], lab, frame.stamp_ns)
        min_px = max(1, cfg.min_cluster_px_full // (cfg.pixel_step ** 2))
        ids, counts = np.unique(frame.instance[valid & (frame.instance > 0)], return_counts=True)
        for i, c in zip(ids.tolist(), counts.tolist()):
            if c < min_px:
                continue
            m = valid & (frame.instance == i)
            p, vv, uu = backproject(frame, m)
            buf.pts.setdefault(i, []).append(p)
            buf.nrm.setdefault(i, []).append(normals[vv, uu])
            tr = self.tracks.setdefault(i, Track())
            tr.obs.append((frame.stamp_ns, p, normals[vv, uu]))
            tr.count += 1
            tr.last = frame.stamp_ns
            buf.first.setdefault(i, frame.stamp_ns)
            buf.last[i] = frame.stamp_ns
            buf.frames[i] = buf.frames.get(i, 0) + 1
            if frame.semantic is not None:
                s, sc = np.unique(frame.semantic[vv, uu], return_counts=True)
                votes = buf.sem.setdefault(i, {})
                for a, b in zip(s.tolist(), sc.tolist()):
                    votes[a] = votes.get(a, 0) + b
                    tr.sem[a] = tr.sem.get(a, 0) + b
        buf.stamps.append(frame.stamp_ns)
        if self.cfg.segments == "tracks":
            self._extract(frame.stamp_ns, terminal=False)

    def _segment(self, i: int, tr: Track, stamp: int) -> None:
        cfg = self.cfg
        window = cfg.buffer_frames_30hz / 30.0 * 1e9
        obs = [o for o in tr.obs if o[0] >= tr.last - window]
        if not obs:
            return
        pts = np.concatenate([o[1] for o in obs])
        nrm = np.concatenate([o[2] for o in obs])
        keys = np.floor(pts / cfg.object_voxel).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        first = np.sort(first)
        sem = max(tr.sem, key=tr.sem.get) if tr.sem else -1
        self.semantic.setdefault(i, sem)
        self.pending.append(Observation(identity=i, points=pts[first], normals=nrm[first],
                                        first=obs[0][0], last=obs[-1][0], semantic=self.semantic[i],
                                        reconstruction_frames=len(obs)))

    def _extract(self, stamp: int, terminal: bool) -> None:
        cfg = self.cfg
        per_frame = 30.0 / cfg.evidence_hz
        chunk = max(1, int(round(cfg.buffer_frames_30hz / per_frame)))
        min_obs = max(1, int(math.ceil(cfg.min_observations_30hz / per_frame)))
        window = cfg.buffer_frames_30hz / 30.0 * 1e9
        for i in list(self.tracks):
            tr = self.tracks[i]
            inactive = terminal or stamp - tr.last > cfg.temporal_window_s * 1e9
            if inactive:
                if tr.count >= min_obs:
                    self._segment(i, tr, stamp)
                del self.tracks[i]
            elif tr.count - tr.submitted >= chunk:
                self._segment(i, tr, stamp)
                tr.submitted = tr.count
                tr.obs = [o for o in tr.obs if o[0] >= tr.last - window]

    # -- per round ---------------------------------------------------------------------------
    def _verify(self, stamp: int) -> None:
        reg = self.registry
        for i in reg.tracked_ids():
            cur, scur = reg.current_fragment(i), reg.session_current_fragment(i)
            if (cur is None or cur.num_vertices == 0) and (scur is None or scur.num_vertices == 0):
                continue
            inh, ses = SurfaceEvidence(), SurfaceEvidence()
            for frag, slot, target in ((cur, 0, "inh"), (scur, 1, "ses")):
                if frag is None or frag.num_vertices == 0:
                    continue
                ev = self.absence.measure_state(
                    i, frag.points, frag.normals,
                    max(frag.last_support_time, frag.last_confirmed_support), stamp, slot,
                    frag.birth_time)
                if target == "inh":
                    inh = ev
                else:
                    ses = ev
            n_log = len(reg.log)
            reg.resolve_current_evidence(i, inh, ses, stamp)
            self.log.extend(reg.log[n_log:])

    def _element_rule(self, stamps: Sequence[int], stamp: int) -> None:
        if not self.cfg.use_layer:
            return
        ids = self.elements.alive()
        if not len(ids):
            return
        pts = self.elements.xyz(ids)
        nrm = self.elements.normals(ids)
        has_n = np.isfinite(nrm).all(axis=1)
        tol = self.absence.config.surface_match_tolerance
        min_cos = math.cos(math.radians(self.absence.config.max_absence_incidence_deg))
        verdict = np.zeros(len(ids), dtype=np.int8)       # 0 none, 1 on surface, 2 seen through
        chosen, last_t = [], None
        for t in stamps:
            if last_t is None or t - last_t >= 1e9 / self.cfg.element_rule_hz:
                chosen.append(t)
                last_t = t
        for t in chosen:
            if t <= 0:
                continue
            p = self.store.project(t, pts)
            with np.errstate(invalid="ignore"):
                measured = (p.etype != UNAVAILABLE) & (p.etype != INVALID) & \
                    np.isfinite(p.measured) & (p.measured > 0)
                delta = p.measured - p.query
                facing = ~has_n | (np.abs(np.einsum("ij,ij->i", np.nan_to_num(nrm), p.view_dir)) >= min_cos)
                on = measured & (np.abs(delta) <= tol)
                through = measured & (delta > tol) & facing
            # only measurements after the element's own last support count
            later = t > self.elements.last_seen[ids]
            verdict[on & later] = 1
            verdict[through & later] = 2
        pn, ps = self.stats.pooled_n, self.stats.pooled_sum
        p_miss = ps / pn if pn >= 3 else 0.05          # uninformative population of prior()
        step = -math.log(min(0.995, max(0.005, p_miss)))
        if self.cfg.single_look:
            step = LN99 + 1.0
        c = self.elements.cusum
        c[ids[verdict == 1]] = 0.0
        through_ids = ids[verdict == 2]
        c[through_ids] += step
        retire = through_ids[c[through_ids] > LN99]
        self.elements.death[retire] = stamp

    def _naive_ingest(self, o: Observation) -> None:
        frag = self.naive.get(o.identity)
        if frag is None:
            self.naive[o.identity] = self.registry.make_fragment(o)
        else:
            self.registry.merge_observation_into_fragment(frag, o)

    def _ingest(self, buf: RoundBuffer) -> None:
        obs = []
        if self.cfg.segments == "tracks":
            obs, self.pending = self.pending, []
            for o in sorted(obs, key=lambda o: (o.first, o.last, o.identity)):
                if self.cfg.use_layer:
                    self.registry.ingest(o)
                else:
                    self._naive_ingest(o)
            return
        for i, chunks in buf.pts.items():
            pts = np.concatenate(chunks)
            nrm = np.concatenate(buf.nrm[i])
            keys = np.floor(pts / self.cfg.object_voxel).astype(np.int64)
            _, first = np.unique(keys, axis=0, return_index=True)
            first = np.sort(first)
            votes = buf.sem.get(i, {})
            sem = max(votes, key=votes.get) if votes else -1
            self.semantic.setdefault(i, sem)
            obs.append(Observation(identity=i, points=pts[first], normals=nrm[first],
                                   first=buf.first[i], last=buf.last[i], semantic=self.semantic[i],
                                   reconstruction_frames=buf.frames[i]))
        for o in sorted(obs, key=lambda o: (o.first, o.last, o.identity)):
            if self.cfg.use_layer:
                self.registry.ingest(o)
            else:
                self._naive_ingest(o)

    def _snapshot(self, stamp: int) -> None:
        reg = self.registry
        entries = []
        if not self.cfg.use_layer:
            for i in sorted(set(reg.states) | set(self.naive)):
                parts = []
                if i in reg.states and reg.states[i].current is not None:
                    inh = reg.states[i].fragments[reg.states[i].current]
                    parts.append((self._uid(inh, i), inh.num_vertices))
                if i in self.naive:
                    parts.append((self._uid(self.naive[i], i), self.naive[i].num_vertices))
                entries.append((i, self.semantic.get(i, -1), parts))
            self.display.append(entries)
            self.stamps.append(stamp)
            return
        for i in reg.tracked_ids():
            state = reg.states[i]
            if state.current is None:
                continue
            cur = state.fragments[state.current]
            parts = [(self._uid(cur, i), cur.num_vertices)]
            shown = reg.displayed(i)
            if shown is not None and len(shown[0]) > cur.num_vertices and state.b_session is not None \
                    and state.b_session.current is not None:
                b = state.b_session.fragments[state.b_session.current]
                parts.append((self._uid(b, i), b.num_vertices))
            entries.append((i, self.semantic.get(i, cur.semantic_label), parts))
        self.display.append(entries)
        self.stamps.append(stamp)

    def run(self, session: FlatSession, max_frames: Optional[int] = None, verbose: bool = True
            ) -> SessionResult:
        cfg = self.cfg
        start_ns = session.stamp_ns(0)
        self.seed(start_ns)
        step = max(1, int(round(30.0 / cfg.evidence_hz)))
        indices = list(range(0, len(session.ids), step))
        if max_frames:
            indices = indices[:max_frames]
        buf = RoundBuffer()
        round_start = session.stamp_ns(indices[0])
        t0 = time.time()
        for n, i in enumerate(indices):
            frame = session.load(i)
            self.integrate(frame, buf)
            last = n == len(indices) - 1
            if frame.stamp_ns - round_start >= cfg.round_s * 1e9 or last:
                stamp = frame.stamp_ns
                if cfg.use_layer:
                    self._verify(stamp)
                self._element_rule(buf.stamps, stamp)
                if last and cfg.segments == "tracks":
                    self._extract(stamp, terminal=True)
                self._ingest(buf)
                if last and cfg.use_layer:
                    self.registry.finalize_pending_absences(stamp)
                    self._verify(stamp)
                    self.registry.finalize_pending_absences(stamp)
                self._snapshot(stamp)
                if verbose:
                    print(f"round {len(self.stamps):3d} t={(stamp - start_ns) / 1e9:7.1f}s "
                          f"frames={n + 1}/{len(indices)} ids={len(self.registry.states)} "
                          f"elements={len(self.elements.alive())} {time.time() - t0:6.1f}s", flush=True)
                buf = RoundBuffer()
                round_start = stamp
        return SessionResult(self.stamps, self.display, self.fragments, self.fragment_identity,
                             self.elements, self.registry, self.stats, self.log,
                             None if cfg.use_layer else self.naive)


def export_state(res: SessionResult) -> dict:
    """What the next session inherits: alive background elements + what each identity shows."""
    ids = res.elements.alive()
    objects = []
    if res.naive is not None:
        for i, parts in ((i, p) for i, _, p in res.display[-1]):
            frags = [res.fragments[uid] for uid, _ in parts]
            pts = np.concatenate([f.points[:n] for f, (_, n) in zip(frags, parts)])
            nrm = np.concatenate([(f.normals if f.normals is not None else np.full((len(f.points), 3), np.nan))[:n]
                                  for f, (_, n) in zip(frags, parts)])
            objects.append(dict(identity=i, points=pts, normals=nrm, semantic=frags[0].semantic_label,
                                first=min(f.birth_time for f in frags),
                                last=max(f.last_support_time for f in frags), dynamic=False))
        return dict(el_xyz=res.elements.xyz(ids), el_nrm=res.elements.normals(ids),
                    el_label=res.elements.label[ids], objects=objects, final_stamp=res.stamps[-1])
    reg = res.registry
    for i in reg.tracked_ids():
        shown = reg.displayed(i)
        cur = reg.current_fragment(i)
        if shown is None or cur is None:
            continue
        pts = shown[0]
        nrm = cur.normals if len(pts) == cur.num_vertices else None
        if nrm is None:
            b = reg.session_current_fragment(i)
            nrm = np.concatenate([cur.normals if cur.normals is not None else np.full((cur.num_vertices, 3), np.nan),
                                  b.normals if b is not None and b.normals is not None
                                  else np.full((len(pts) - cur.num_vertices, 3), np.nan)])
        objects.append(dict(identity=i, points=pts, normals=nrm, semantic=cur.semantic_label,
                            first=cur.birth_time, last=cur.last_support_time,
                            dynamic=reg.states[i].has_dynamic_history))
    return dict(el_xyz=res.elements.xyz(ids), el_nrm=res.elements.normals(ids),
                el_label=res.elements.label[ids], el_identity=res.elements.identity[ids],
                objects=objects, final_stamp=res.stamps[-1])
