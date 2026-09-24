"""Round driver: a posed RGB-D mapping backend plus the representation-free update layer (GPU).

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

All array work runs on the GPU (torch, evidence.DEV); frames are decoded ahead on CPU threads.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .evidence import (DEV, EvidenceConfig, EvidenceStore, INVALID, ObservedAbsence, SensorStatistics,
                       SurfaceEvidence, UNAVAILABLE, first_per_key, to_dev, voxel_keys)
from .frames import Frame, FlatSession
from .registry import Fragment, Observation, PersistentObjectState

OPEN = int(np.iinfo(np.int64).max)
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
    segments: str = "rounds"               # "rounds" (one segment per round) or "tracks"
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
    # t2 frontend's static-surface frame selection (MeshObjectExtractor::selectStaticFrames): an
    # observation keeps only the frames after the newest earlier frame whose instance surface
    # conflicts with the newest frame in either direction (>= 20 of <= 512 samples seen through
    # beyond 5 cm in a 3x3 footprint, judged samples >= half, free share > 0.2). Defaults of t2.
    static_frames: bool = False
    static_tolerance: float = 0.05
    static_max_free_fraction: float = 0.2
    static_min_pixels: int = 20


def pixel_normals(depth: torch.Tensor, K, R_world_cam: torch.Tensor) -> torch.Tensor:
    """Per-pixel world normals from depth differences (NaN where undefined), towards the camera."""
    h, w = depth.shape
    u = (torch.arange(w, device=DEV, dtype=torch.float32) + K.offset - K.cx) / K.fx
    v = (torch.arange(h, device=DEV, dtype=torch.float32) + K.offset - K.cy) / K.fy
    P = torch.stack([u[None, :] * depth, v[:, None] * depth, depth], dim=-1)
    dx = torch.full_like(P, float("nan"))
    dy = torch.full_like(P, float("nan"))
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1, :] = P[2:, :] - P[:-2, :]
    n = torch.cross(dx, dy, dim=-1)
    n = n / torch.linalg.norm(n, dim=-1, keepdim=True)
    flip = (n * P).sum(-1) > 0
    n = torch.where(flip[..., None], -n, n)
    return n @ R_world_cam.T


def backproject(depth: torch.Tensor, K, T_world_cam: torch.Tensor, mask: torch.Tensor):
    """World points and pixel indices of the valid masked depth pixels."""
    valid = torch.isfinite(depth) & mask
    v, u = torch.nonzero(valid, as_tuple=True)
    z = depth[v, u]
    x = (u.to(torch.float32) + K.offset - K.cx) / K.fx * z
    y = (v.to(torch.float32) + K.offset - K.cy) / K.fy * z
    cam = torch.stack([x, y, z], dim=1)
    return cam @ T_world_cam[:3, :3].T + T_world_cam[:3, 3], v, u


class ElementStore:
    """Background surface elements of a point/surfel map, voxel-hashed, with lifetimes (GPU)."""

    COLS = (("sum", (3,), torch.float64, 0), ("cnt", (), torch.float64, 0), ("nrm", (3,), torch.float64, 0),
            ("label", (), torch.int64, 0), ("birth", (), torch.int64, 0), ("last_seen", (), torch.int64, 0),
            ("death", (), torch.int64, OPEN), ("inherited", (), torch.bool, 0),
            ("cusum", (), torch.float64, 0), ("identity", (), torch.int64, 0))

    def __init__(self, voxel: float):
        self.voxel = voxel
        self.keys = torch.zeros(0, dtype=torch.int64, device=DEV)      # sorted
        self.order = torch.zeros(0, dtype=torch.int64, device=DEV)     # element index per sorted key
        self.n = 0
        for name, shape, dtype, fill in self.COLS:
            setattr(self, name, torch.full((1 << 16,) + shape, fill, dtype=dtype, device=DEV))

    def _grow(self, need: int) -> None:
        cap = len(self.cnt)
        if need <= cap:
            return
        new = max(need, cap * 2)
        for name, shape, dtype, fill in self.COLS:
            a = getattr(self, name)
            b = torch.full((new,) + shape, fill, dtype=dtype, device=DEV)
            b[:cap] = a
            setattr(self, name, b)

    def encode(self, pts: torch.Tensor, identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 18 bits per axis (+-2.6 km at 2 cm) and the identity (< 512) in the top 9 bits:
        # one element per (voxel, identity), so objects never share an element with background.
        k = torch.floor(pts / self.voxel).to(torch.int64) + (1 << 17)
        key = (k[:, 0] << 36) | (k[:, 1] << 18) | k[:, 2]
        if identity is not None:
            key = key | (identity.to(torch.int64) << 54)
        return key

    def lookup(self, keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(element index or -1, position in the sorted key table) per key."""
        n = len(self.keys)
        if n == 0:
            return torch.full_like(keys, -1), torch.zeros_like(keys)
        pos = torch.searchsorted(self.keys, keys).clamp(max=n - 1)
        hit = self.keys[pos] == keys
        return torch.where(hit, self.order[pos], torch.full_like(keys, -1)), pos

    def add(self, pts: torch.Tensor, nrm: torch.Tensor, lab: torch.Tensor, stamp: int,
            inherited: bool = False, identity: Optional[torch.Tensor] = None) -> None:
        if len(pts) == 0:
            return
        keys = self.encode(pts, identity)
        uk, inv = torch.unique(keys, return_inverse=True)
        idx, pos = self.lookup(uk)
        known = idx >= 0
        # a key whose element was retired starts a new element (the surface re-appeared)
        dead = known.clone()
        dead[known] = self.death[idx[known]] != OPEN
        new = ~known | dead
        n_new = int(new.sum())
        if n_new:
            self._grow(self.n + n_new)
            fresh = torch.arange(self.n, self.n + n_new, device=DEV)
            idx[new] = fresh
            self.birth[fresh] = stamp
            self.death[fresh] = OPEN
            self.inherited[fresh] = inherited
            self.n += n_new
            self.order[pos[dead]] = idx[dead]                        # re-point existing keys
            fresh_keys = ~known
            if fresh_keys.any():
                self.keys = torch.cat([self.keys, uk[fresh_keys]])
                self.order = torch.cat([self.order, idx[fresh_keys]])
                s = torch.argsort(self.keys)
                self.keys, self.order = self.keys[s], self.order[s]
        el = idx[inv]
        if identity is not None:
            self.identity[el] = identity.to(torch.int64)
        ok = torch.isfinite(nrm).all(dim=1)
        self.sum.index_add_(0, el, pts.to(torch.float64))
        self.cnt.index_add_(0, el, torch.ones(len(el), dtype=torch.float64, device=DEV))
        self.nrm.index_add_(0, el[ok], nrm[ok].to(torch.float64))
        last = torch.full((len(uk),), -1, dtype=torch.int64, device=DEV)
        last.scatter_reduce_(0, inv, torch.arange(len(inv), device=DEV), reduce="amax")
        self.label[idx] = lab.to(torch.int64)[last]                  # the last observation's label
        self.last_seen[idx] = stamp
        self.cusum[idx] = 0.0

    def alive(self) -> torch.Tensor:
        return torch.nonzero(self.death[: self.n] == OPEN).squeeze(1)

    def xyz(self, ids: torch.Tensor) -> torch.Tensor:
        return (self.sum[ids] / self.cnt[ids, None]).to(torch.float32)

    def normals(self, ids: torch.Tensor) -> torch.Tensor:
        v = self.nrm[ids]
        length = torch.linalg.norm(v, dim=1, keepdim=True)
        return torch.where(length > 1e-9, v / length, torch.full_like(v, float("nan"))).to(torch.float32)


class ClosedStateBackground:
    """t2 markClosedObjectBackground (closed_object_background.cpp) with Khronos' RayChangeDetector.

    Background elements near a CLOSED object state (within sqrt(3) * map resolution of its surface,
    not re-integrated after the state closed) are judged per element on the projected pixels after
    max(state support, element support): |range difference| <= 0.5 * resolution + 1 mm is geometric
    support (drops everything before it), beyond is absent, nearer is occluded (inconclusive).
    Absent when some window of window_size bins of temporal_resolution (from a bin with data) has
    an absent share above absence_confidence (room18/synthetic configs: 5 s, 5 bins, 0.6).
    Incremental per round; decisions equal re-evaluating all stored frames every round.
    """

    def __init__(self, layer: "UpdateLayerSession", temporal_resolution_s: float = 5.0,
                 window_size: int = 5, absence_confidence: float = 0.6):
        self.layer = layer
        self.res_ns = int(temporal_resolution_s * 1e9)
        self.window = window_size
        self.conf = absence_confidence
        self.states: Dict[int, dict] = {}         # fragment uid -> candidates and running counts

    def _candidates(self, frag: Fragment, cfg) -> Optional[dict]:
        el = self.layer.elements
        alive = torch.nonzero((el.death[: el.n] == OPEN) & (el.identity[: el.n] <= 0)).squeeze(1)
        if not len(alive) or frag.num_vertices == 0:
            return None
        radius = math.sqrt(3.0) * cfg.map_resolution
        pts = frag.points[torch.isfinite(frag.points).all(dim=1)]
        lo, hi = pts.min(dim=0).values - radius, pts.max(dim=0).values + radius
        xyz = el.xyz(alive)
        box = ((xyz >= lo) & (xyz <= hi)).all(dim=1)
        box &= el.last_seen[alive] <= frag.death_time          # reconstructed after closure: not owned
        ids = alive[box]
        if not len(ids):
            return None
        from scipy.spatial import cKDTree
        d, _ = cKDTree(pts.cpu().numpy()).query(el.xyz(ids).cpu().numpy(), k=1,
                                                distance_upper_bound=radius * (1 + 1e-6))
        near = torch.as_tensor(np.isfinite(d) & (d <= radius), device=DEV)
        ids = ids[near]
        if not len(ids):
            return None
        through = torch.clamp(el.last_seen[ids], min=max(frag.last_support_time, frag.last_confirmed_support))
        return dict(ids=ids, supported=through, last_geo=through.clone(), processed=int(through.min()),
                    bins=torch.zeros((len(ids), 0, 2), dtype=torch.int32, device=DEV), bin0=None)

    def run(self, stamp: int) -> int:
        layer, cfg = self.layer, self.layer.cfg
        el, store = layer.elements, layer.store
        tol = 0.5 * cfg.map_resolution + 1e-3
        retired = 0
        for state in layer.registry.states.values():
            for frag in state.fragments:
                if frag.death_time is None or frag.death_time > stamp or frag.last_support_time >= stamp:
                    continue
                key = id(frag)
                if key not in self.states:
                    self.states[key] = self._candidates(frag, cfg)
                c = self.states[key]
                if c is None:
                    continue
                keep = el.death[c["ids"]] == OPEN
                if not keep.any():
                    continue
                lo, hi = store.window(c["processed"] + 1, stamp)
                c["processed"] = stamp
                if hi <= lo:
                    continue
                stamps = store.stamp_tensor(lo, hi)
                p = store.project(lo, hi, el.xyz(c["ids"]))
                et, meas, query = p["etype"], p["measured"], p["query"]
                ok = (et != UNAVAILABLE) & (et != INVALID) & torch.isfinite(meas) & (meas > 0) & \
                    torch.isfinite(query)
                delta = meas - query
                valid = ok & (stamps[:, None] > c["supported"][None, :])
                present = valid & (torch.abs(delta) <= tol)
                absent = valid & (delta > tol)
                inconcl = valid & (delta < -tol)
                last_present = torch.where(present, stamps[:, None], torch.zeros_like(stamps)[:, None]).amax(0)
                c["last_geo"] = torch.maximum(c["last_geo"], last_present)
                # remove_past: only verdicts after the element's latest geometric support count
                after = stamps[:, None] > c["last_geo"][None, :]
                b = (stamps // self.res_ns)
                if c["bin0"] is None:
                    c["bin0"] = int(b[0])
                nb = int(b[-1]) - c["bin0"] + 1
                if nb > c["bins"].shape[1]:
                    pad = torch.zeros((len(c["ids"]), nb - c["bins"].shape[1], 2), dtype=torch.int32, device=DEV)
                    c["bins"] = torch.cat([c["bins"], pad], dim=1)
                # a new geometric support clears all earlier bins of that element
                reset = last_present > 0
                c["bins"][reset] = 0
                col = (b - c["bin0"]).long()
                for kind, m in ((0, absent & after), (1, inconcl & after)):
                    rows, frames_ = torch.nonzero(m.T, as_tuple=True)
                    if len(rows):
                        flat = c["bins"][:, :, kind].reshape(-1)
                        flat.index_add_(0, rows * c["bins"].shape[1] + col[frames_],
                                        torch.ones(len(rows), dtype=torch.int32, device=DEV))
                        c["bins"][:, :, kind] = flat.view(len(c["ids"]), -1)
                A = c["bins"][:, :, 0].float()
                I = c["bins"][:, :, 1].float()
                cA = torch.nn.functional.pad(torch.cumsum(A, 1), (1, 0))
                cI = torch.nn.functional.pad(torch.cumsum(I, 1), (1, 0))
                nbins = A.shape[1]
                end = torch.clamp(torch.arange(nbins, device=DEV) + self.window, max=nbins)
                wA = cA[:, end] - cA[:, :nbins]
                wI = cI[:, end] - cI[:, :nbins]
                has = (A + I) > 0
                ratio = wA / torch.clamp(wA + wI, min=1)
                absent_now = (has & (ratio > self.conf)).any(dim=1) & keep
                if absent_now.any():
                    ids = c["ids"][absent_now]
                    el.death[ids] = stamp
                    retired += int(absent_now.sum())
        return retired


@dataclass
class Track:
    obs: List[tuple] = field(default_factory=list)      # (stamp, points, normals)
    count: int = 0
    submitted: int = 0
    last: int = 0
    sem: Dict[int, int] = field(default_factory=dict)


@dataclass
class RoundBuffer:
    pts: Dict[int, List[torch.Tensor]] = field(default_factory=dict)
    nrm: Dict[int, List[torch.Tensor]] = field(default_factory=dict)
    first: Dict[int, int] = field(default_factory=dict)
    last: Dict[int, int] = field(default_factory=dict)
    frames: Dict[int, int] = field(default_factory=dict)
    sem: Dict[int, Dict[int, int]] = field(default_factory=dict)
    stamps: List[int] = field(default_factory=list)
    fstamp: Dict[int, List[int]] = field(default_factory=dict)     # per identity: stamp of each frame
    fidx: Dict[int, List[int]] = field(default_factory=dict)       # per identity: evidence-store index


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


def _votes(values: torch.Tensor) -> Dict[int, int]:
    s, c = torch.unique(values, return_counts=True)
    return dict(zip(s.tolist(), c.tolist()))


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
        self.dynamic = torch.as_tensor(list(cfg.dynamic_semantics), dtype=torch.int64, device=DEV)
        self.closed_background = ClosedStateBackground(self)
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
        seed_stamp = int(p.get("final_stamp", start_ns))
        if len(p["el_xyz"]):
            ident = to_dev(p["el_identity"], torch.int64) if self.cfg.objects_as_elements and \
                "el_identity" in p else None
            # inherited elements exist since the previous session: alive in the seed snapshot
            self.elements.add(to_dev(p["el_xyz"]), to_dev(p["el_nrm"]), to_dev(p["el_label"], torch.int64),
                              seed_stamp, inherited=True, identity=ident)
        for obj in p["objects"]:
            frag = Fragment(points=to_dev(obj["points"]), normals=to_dev(obj["normals"]),
                            elements=torch.zeros(0, dtype=torch.int64, device=DEV),
                            birth_time=obj["first"], last_support_time=obj["last"],
                            last_confirmed_support=obj["last"], semantic_label=obj["semantic"])
            self.registry.seed_inherited(obj["identity"], frag, obj["dynamic"], (obj["first"], obj["last"]))
            self.semantic[obj["identity"]] = obj["semantic"]
        # the inherited scene is the session's first snapshot, stamped with the previous session's
        # final time (t2 keeps that seed snapshot inside the map)
        self._snapshot(seed_stamp)

    # -- per frame --------------------------------------------------------------------------
    def integrate(self, frame: Frame, buf: RoundBuffer) -> None:
        cfg = self.cfg
        depth = to_dev(frame.depth)
        inst = to_dev(frame.instance, torch.int64)
        sem = to_dev(frame.semantic, torch.int64) if frame.semantic is not None else None
        dyn = torch.zeros_like(inst, dtype=torch.bool)
        if sem is not None and self.dynamic.numel():
            dyn = (inst <= 0) & torch.isin(sem, self.dynamic)
        self.store.ingest(frame, dyn)
        T = torch.as_tensor(frame.T_world_cam, dtype=torch.float32, device=DEV)
        normals = pixel_normals(depth, frame.K, T[:3, :3])
        valid = torch.isfinite(depth)
        if cfg.objects_as_elements:
            pts, v, u = backproject(depth, frame.K, T, valid & ~dyn)
            lab = sem[v, u] if sem is not None else torch.zeros_like(v)
            self.elements.add(pts, normals[v, u], lab, frame.stamp_ns, identity=inst[v, u].clamp(min=0))
            buf.stamps.append(frame.stamp_ns)
            return
        pts, v, u = backproject(depth, frame.K, T, valid & (inst <= 0) & ~dyn)
        lab = sem[v, u] if sem is not None else torch.zeros_like(v)
        self.elements.add(pts, normals[v, u], lab, frame.stamp_ns)
        min_px = max(1, cfg.min_cluster_px_full // (cfg.pixel_step ** 2))
        obj = valid & (inst > 0)
        ids, counts = torch.unique(inst[obj], return_counts=True)
        seen = []
        for i, c in zip(ids.tolist(), counts.tolist()):
            if c < min_px:
                continue
            p, vv, uu = backproject(depth, frame.K, T, valid & (inst == i))
            seen.append((i, p, normals[vv, uu], _votes(sem[vv, uu]) if sem is not None else {}))
        for i, p, n, fv in seen:
            self._buffer(buf, i, frame.stamp_ns, p, n, fv, self.store.n - 1)
        buf.stamps.append(frame.stamp_ns)
        if cfg.segments == "tracks":
            self._extract(frame.stamp_ns, terminal=False)

    def _buffer(self, buf: RoundBuffer, i: int, stamp: int, p: torch.Tensor, n: torch.Tensor,
                fv: Dict[int, int], fidx: int) -> None:
        buf.pts.setdefault(i, []).append(p)
        buf.nrm.setdefault(i, []).append(n)
        buf.fstamp.setdefault(i, []).append(stamp)
        buf.fidx.setdefault(i, []).append(fidx)
        buf.first.setdefault(i, stamp)
        buf.last[i] = stamp
        buf.frames[i] = buf.frames.get(i, 0) + 1
        tr = self.tracks.setdefault(i, Track())
        if self.cfg.segments == "tracks":
            tr.obs.append((stamp, p, n, fidx))
            tr.count += 1
            tr.last = stamp
        votes = buf.sem.setdefault(i, {})
        for a, b in fv.items():
            votes[a] = votes.get(a, 0) + b
            tr.sem[a] = tr.sem.get(a, 0) + b

    def _surface_compat(self, pairs, identity: int) -> torch.Tensor:
        """t2 compareSurfaceFrames for (source samples, target store index) pairs: (P, 3) counts
        supported / free / sampled. A sample supports when a 3x3 target pixel with this identity
        measures its range within the tolerance; it is free when all 9 pixels are valid and
        measure beyond it (no support)."""
        cfg, st = self.cfg, self.store
        K = st.K
        pts, pair, tgt = [], [], []
        sampled = torch.zeros(len(pairs), dtype=torch.int64, device=DEV)
        for k, (src, f) in enumerate(pairs):
            stride = max(1, (len(src) + 511) // 512)
            q = src[::stride]
            q = q[torch.isfinite(q).all(dim=1)]
            sampled[k] = len(q)
            pts.append(q)
            pair.append(torch.full((len(q),), k, dtype=torch.int64, device=DEV))
            tgt.append(torch.full((len(q),), f, dtype=torch.int64, device=DEV))
        out = torch.zeros(len(pairs), 3, dtype=torch.int64, device=DEV)
        out[:, 2] = sampled
        if not pts:
            return out
        P, pair, tgt = torch.cat(pts), torch.cat(pair), torch.cat(tgt)
        T = st.T[tgt]
        cam = torch.einsum("nij,nj->ni", T[:, :3, :3], P) + T[:, :3, 3]
        dist = torch.linalg.norm(cam, dim=1)
        z = cam[:, 2]
        ok = torch.isfinite(dist) & (dist > 0) & (dist <= st.max_range) & (z > 0)
        uf = K.fx * cam[:, 0] / z + K.cx - K.offset + 0.5
        vf = K.fy * cam[:, 1] / z + K.cy - K.offset + 0.5
        ok &= torch.isfinite(uf) & torch.isfinite(vf)
        u = torch.where(ok, torch.floor(uf), torch.zeros_like(uf)).to(torch.int64)
        v = torch.where(ok, torch.floor(vf), torch.zeros_like(vf)).to(torch.int64)
        ok &= (u >= 1) & (v >= 1) & (u + 1 < K.width) & (v + 1 < K.height)
        # each target pixel is judged once per pair (the first sample that lands on it)
        key = (pair * K.height + v) * K.width + u
        idx = torch.nonzero(ok).squeeze(1)
        if not len(idx):
            return out
        idx = idx[first_per_key(key[idx])]
        dy = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1], device=DEV)
        dx = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1], device=DEV)
        f9 = tgt[idx][:, None].expand(-1, 9)
        vv, uu = v[idx][:, None] + dy, u[idx][:, None] + dx
        rng = st.rng[f9, vv, uu].to(torch.float32) / 1000.0
        code = st.code[f9, vv, uu]
        valid = rng > 0
        delta = rng - dist[idx][:, None]
        tol = cfg.static_tolerance
        support = (valid & (delta.abs() <= tol) & (code == identity)).any(dim=1)
        free = (valid & (delta > tol)).all(dim=1) & ~support
        out[:, 0].index_add_(0, pair[idx], support.to(torch.int64))
        out[:, 1].index_add_(0, pair[idx], free.to(torch.int64))
        return out

    def _static_start(self, i: int, frames: List[torch.Tensor], fidx: List[int]) -> int:
        """Index of the first frame kept (t2 MeshObjectExtractor::selectStaticFrames)."""
        n = len(frames)
        if n < 2:
            return 0
        cfg = self.cfg
        anchor = fidx[-1]
        pairs = [(frames[j], anchor) for j in range(n - 1)] + [(frames[-1], fidx[j]) for j in range(n - 1)]
        c = self._surface_compat(pairs, i)
        judged = c[:, 0] + c[:, 1]
        conflict = (c[:, 1] >= cfg.static_min_pixels) & (judged * 2 >= c[:, 2]) & \
            (c[:, 1].to(torch.float32) > cfg.static_max_free_fraction * judged.to(torch.float32))
        conflict = conflict[: n - 1] | conflict[n - 1:]
        hit = torch.nonzero(conflict).squeeze(1)
        if not len(hit):
            return 0
        j = int(hit.max())
        self.log.append(f"{self.store._stamps[anchor]} STATIC_BOUNDARY inst={i} "
                        f"rejected={self.store._stamps[fidx[j]]} kept={n - j - 1}/{n}")
        return j + 1

    def _observation(self, i: int, pts: torch.Tensor, nrm: torch.Tensor, first: int, last: int,
                     votes: Dict[int, int], frames: int) -> Observation:
        keep = first_per_key(voxel_keys(pts, self.cfg.object_voxel))
        sem = max(votes, key=votes.get) if votes else -1
        self.semantic.setdefault(i, sem)
        return Observation(identity=i, points=pts[keep], normals=nrm[keep], first=first, last=last,
                           semantic=self.semantic[i], reconstruction_frames=frames)

    def _segment(self, i: int, tr: Track) -> None:
        window = self.cfg.buffer_frames_30hz / 30.0 * 1e9
        obs = [o for o in tr.obs if o[0] >= tr.last - window]
        if obs and self.cfg.static_frames and self.cfg.use_layer:
            obs = obs[self._static_start(i, [o[1] for o in obs], [o[3] for o in obs]):]
        if obs:
            self.pending.append(self._observation(i, torch.cat([o[1] for o in obs]),
                                                  torch.cat([o[2] for o in obs]), obs[0][0], obs[-1][0],
                                                  tr.sem, len(obs)))

    def _extract(self, stamp: int, terminal: bool) -> None:
        cfg = self.cfg
        per_frame = 30.0 / cfg.evidence_hz
        chunk = max(1, int(round(cfg.buffer_frames_30hz / per_frame)))
        min_obs = max(1, int(math.ceil(cfg.min_observations_30hz / per_frame)))
        window = cfg.buffer_frames_30hz / 30.0 * 1e9
        for i in list(self.tracks):
            tr = self.tracks[i]
            if terminal or stamp - tr.last > cfg.temporal_window_s * 1e9:
                if tr.count >= min_obs:
                    self._segment(i, tr)
                del self.tracks[i]
            elif tr.count - tr.submitted >= chunk:
                self._segment(i, tr)
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
            for frag, slot in ((cur, 0), (scur, 1)):
                if frag is None or frag.num_vertices == 0:
                    continue
                ev = self.absence.measure_state(
                    i, frag.points, frag.normals,
                    max(frag.last_support_time, frag.last_confirmed_support), stamp, slot,
                    frag.birth_time)
                if slot == 0:
                    inh = ev
                else:
                    ses = ev
            n_log = len(reg.log)
            reg.resolve_current_evidence(i, inh, ses, stamp)
            self.log.extend(reg.log[n_log:])

    def _element_rule(self, stamps: Sequence[int], stamp: int) -> None:
        if not (self.cfg.use_layer or self.cfg.objects_as_elements):
            return
        ids = self.elements.alive()
        if not len(ids):
            return
        pts = self.elements.xyz(ids)
        nrm = self.elements.normals(ids)
        has_n = torch.isfinite(nrm).all(dim=1)
        nrm0 = torch.nan_to_num(nrm)
        tol = self.absence.config.surface_match_tolerance
        min_cos = math.cos(math.radians(self.absence.config.max_absence_incidence_deg))
        verdict = torch.zeros(len(ids), dtype=torch.int8, device=DEV)   # 0 none, 1 on surface, 2 through
        last_seen = self.elements.last_seen[ids]
        chosen, last_t = [], None
        for t in stamps:
            if last_t is None or t - last_t >= 1e9 / self.cfg.element_rule_hz:
                chosen.append(t)
                last_t = t
        for t in chosen:
            lo, hi = self.store.window(t, t)
            if hi <= lo:
                continue
            p = self.store.project(lo, hi, pts)
            et, meas, query = p["etype"][0], p["measured"][0], p["query"][0]
            measured = (et != UNAVAILABLE) & (et != INVALID) & torch.isfinite(meas) & (meas > 0)
            delta = meas - query
            facing = ~has_n | (torch.abs((nrm0 * p["view"][0]).sum(-1)) >= min_cos)
            later = t > last_seen             # only measurements after the element's own last support
            on = measured & (torch.abs(delta) <= tol) & later
            through = measured & (delta > tol) & facing & later
            verdict[on] = 1
            verdict[through] = 2
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
        if self.cfg.segments == "tracks":
            obs, self.pending = self.pending, []
        else:
            obs = []
            for i, chunks in buf.pts.items():
                k = self._static_start(i, chunks, buf.fidx[i]) if self.cfg.static_frames and self.cfg.use_layer else 0
                obs.append(self._observation(i, torch.cat(chunks[k:]), torch.cat(buf.nrm[i][k:]),
                                             buf.fstamp[i][k] if k else buf.first[i], buf.last[i],
                                             buf.sem.get(i, {}), buf.frames[i] - k))
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
        with ThreadPoolExecutor(max_workers=4) as pool:
            ahead = [pool.submit(session.load, i) for i in indices[:8]]
            for n, _ in enumerate(indices):
                frame = ahead[n].result()
                ahead[n] = None
                if n + 8 < len(indices):
                    ahead.append(pool.submit(session.load, indices[n + 8]))
                self.integrate(frame, buf)
                last = n == len(indices) - 1
                if frame.stamp_ns - round_start >= cfg.round_s * 1e9 or last:
                    stamp = frame.stamp_ns
                    if cfg.use_layer:
                        self._verify(stamp)
                        self.closed_background.run(stamp)
                    self._element_rule(buf.stamps, stamp)
                    if last and cfg.segments == "tracks":
                        self._extract(stamp, terminal=True)
                    self._ingest(buf)
                    if last and cfg.use_layer:
                        self.registry.finalize_pending_absences(stamp)
                        self._verify(stamp)
                        self.registry.finalize_pending_absences(stamp)
                        self.closed_background.run(stamp)
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


def _np(t: Optional[torch.Tensor]):
    return None if t is None else t.detach().cpu().numpy()


def export_state(res: SessionResult) -> dict:
    """What the next session inherits: alive background elements + what each identity shows."""
    ids = res.elements.alive()
    objects = []
    if res.naive is not None:
        for i, _, parts in res.display[-1]:
            frags = [res.fragments[uid] for uid, _ in parts]
            pts = torch.cat([f.points[:n] for f, (_, n) in zip(frags, parts)])
            nrm = torch.cat([(f.normals if f.normals is not None
                              else torch.full((len(f.points), 3), float("nan"), device=DEV))[:n]
                             for f, (_, n) in zip(frags, parts)])
            objects.append(dict(identity=i, points=_np(pts), normals=_np(nrm), semantic=frags[0].semantic_label,
                                first=min(f.birth_time for f in frags),
                                last=max(f.last_support_time for f in frags), dynamic=False))
    else:
        reg = res.registry
        for i in reg.tracked_ids():
            shown = reg.displayed(i)
            cur = reg.current_fragment(i)
            if shown is None or cur is None:
                continue
            pts = shown[0]
            parts = [cur]
            b = reg.session_current_fragment(i)
            if len(pts) > cur.num_vertices and b is not None:
                parts.append(b)
            nrm = torch.cat([f.normals if f.normals is not None
                             else torch.full((f.num_vertices, 3), float("nan"), device=DEV) for f in parts])
            objects.append(dict(identity=i, points=_np(pts), normals=_np(nrm), semantic=cur.semantic_label,
                                first=cur.birth_time, last=cur.last_support_time,
                                dynamic=reg.states[i].has_dynamic_history))
    return dict(el_xyz=_np(res.elements.xyz(ids)), el_nrm=_np(res.elements.normals(ids)),
                el_label=_np(res.elements.label[ids]), el_identity=_np(res.elements.identity[ids]),
                objects=objects, final_stamp=res.stamps[-1])
