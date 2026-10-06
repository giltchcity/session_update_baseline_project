"""The update layer: one set of decisions for any map backend (interface.Backend).

The layer reads the same frames as the backend (depth, pose, instance masks) and keeps its own
evidence and object states. It sees the backend's map only through elements() and acts on it only
through retire(). Per reconciliation round (t2 Backend::runChangeDetectionThread order):
  1. judge every stored object state against the round's frames (observed-absence test,
     projected support/contradiction) and resolve it (registry.resolve_current_evidence);
  2. judge the backend's background elements: those near a closed object state
     (ClosedStateBackground) and all of them (element rule, below);
  3. ingest the round's new object observations into the identity registry;
  4. return what to retire: the background elements judged absent, and the object elements that
     are no longer the object (object support, below).
Terminal round: 1-3, finalize pending absences, 1 again, finalize, closed-state background again.

Object states are built from the layer's own instance-masked observations (one segment per
identity per round, after t2's static-surface frame selection), never from the backend's
geometry, so every backend gets the same object decisions. D1 / D2 / D3 (README: visible motion,
hidden change within a run, hidden change across the session gap) all go through the same
evidence: observed absence of the stored state and the handoff to the state observed at the new
site; the frames in which an object is seen moving never enter a state (static-frame selection).
(Marking an observation "moved" when its frames conflict was tried and dropped: on real depth the
conflict also fires on static objects, and closing their state removed them.)
Object support: t2 shows an object as the surface extracted from its selected static frames. A
backend integrates every frame, so an identity's elements are kept only within the sensor
tolerance (+ the element's extent) of that identity's live states (current, in-session and
pending ones); the rest -- the old sites of closed states and what was integrated while the
object moved -- is retired.

Element size: every test of a measured surface against an element allows the sensor tolerance
plus the element's extent (interface.Elements.extent: 0 for points, a surfel's radius, a
Gaussian's 3 sigma, half a cell's diagonal).

Element rule (new; t2 has no representation-free background rule -- its background uses
Khronos' mesh-ray detector). Each round an element gets its latest verdict from projection:
on-surface (|range difference| <= the 5 cm sensor tolerance), seen-through (measured beyond it,
facing within the 60 deg incidence limit), or nothing. A seen-through verdict adds ln(1/p_M),
p_M = the sensor's pooled see-through share of present surfaces that the object test learns
(SensorStatistics); an on-surface verdict or a new backend update of the element resets the
sum; the element is retired above ln 99. The same test and threshold as for object states, at
element granularity; no new constant.

Decision core (LayerConfig.core). "l2": the TSDF core of L2_FINAL2 (session_core @192c1cf: core/l2/evidence.py,
core/l2/state.py, replayed line by line against its khronos.log), driven in the round order of
Backend::runChangeDetectionThread with its session hooks (backend_session.cpp): one observation event per
round, verify, (terminal: finalize), closed-state background, the element rule (reconcile), canonicalize
(ingest + materialize every identity); terminal drain: verify, finalize, closed-state background,
canonicalize. Segments pass the MeshObjectExtractor volume gates first. "t2": the earlier port
(core/evidence.py, core/registry.py), kept as control. Replacing the core does not touch the backends.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..frames import Frame, SessionSpec, dynamic_mask, torch_backproject, torch_pixel_normals
from ..interface import Elements
from .evidence import (DEV, EvidenceConfig, EvidenceStore, INVALID, ObservedAbsence, SensorStatistics,
                       SurfaceEvidence, UNAVAILABLE, first_per_key, to_dev, voxel_keys)
from .registry import Fragment, Observation, PersistentObjectState
from .l2 import evidence as l2_evidence, state as l2_state
from .d1 import D1, D1Config
from . import session_end

LN99 = math.log(99.0)


@dataclass
class LayerConfig:
    round_s: float                         # reconciliation period (t2: real 2.1 s, synthetic 10.8 s)
    map_resolution: float                  # t2 active-window voxel size (real 0.05, synthetic 0.10)
    max_range: float                       # sensor range of the t2 mapper (real 5 m, synthetic 15 m)
    high_mobility: Sequence[int] = ()
    object_semantics: Sequence[int] = ()   # label space: object classes (UNIDENTIFIED when no id)
    dynamic_semantics: Sequence[int] = ()  # label space: dynamic classes (never integrated)
    evidence_hz: float = 5.0
    pixel_step: int = 2
    min_cluster_px_full: int = 50          # InstanceForwarding min_cluster_size (full resolution)
    object_voxel: float = 0.01
    element_rule_hz: float = 1.0
    decision: str = "cusum"                # absence-decision ablation (EvidenceConfig.decision, t2 only)
    # decision core: "l2" = the TSDF core of L2_FINAL2 (session_core @192c1cf, core/l2: replayed line by
    # line against its logs); "t2" = the earlier port (43c663d, core/evidence.py + registry.py), control
    core: str = "t2"
    # MeshObjectExtractor gates of the TSDF runs (session_update_baseline/configs/room18_instance_5cm.yaml:
    # 120-121 and datasets/synthetic_ab/mapping_configs/mapper_mechanism_10cm.yaml:69-70: min_object_volume
    # 0.005, max_object_volume 10 m^3; MeshObjectExtractor drops a reconstruction whose BoundingBox volume is
    # outside them): such a segment is no object reconstruction and never reaches the registry (l2 frontend).
    min_object_volume: float = 0.005
    max_object_volume: float = 10.0
    # active-window truncation distance of the TSDF runs (D1 front end's free-space state, core/d1.py)
    truncation: float = 0.15
    d1: bool = False                       # D1 front end (core/d1.py) with the l2 core
    g5: bool = False                       # session-end memory test (core/session_end.py) with the l2 core
    # object reconstruction voxel of the TSDF runs: the session-end present TSDF (voxel, truncation 2 voxels)
    object_voxel: float = 0.02
    # depthScale's pixel stride (16 px of the TSDF archive's frames = 8 of the layer's evidence frames)
    depth_scale_stride: int = 8
    # t2 frontend's static-surface frame selection (MeshObjectExtractor::selectStaticFrames): an
    # observation keeps only the frames after the newest earlier frame whose instance surface
    # conflicts with the newest frame in either direction (>= 20 of <= 512 samples seen through
    # beyond 5 cm in a 3x3 footprint, judged samples >= half, free share > 0.2). Defaults of t2.
    static_frames: bool = True
    static_tolerance: float = 0.05
    static_max_free_fraction: float = 0.2
    static_min_pixels: int = 20


def _np(t: Optional[torch.Tensor]):
    return None if t is None else t.detach().cpu().numpy()


def _votes(values: torch.Tensor) -> Dict[int, int]:
    s, c = torch.unique(values, return_counts=True)
    return dict(zip(s.tolist(), c.tolist()))


def _empty_ids() -> torch.Tensor:
    return torch.zeros(0, dtype=torch.int64, device=DEV)


class _ById:
    """Rows of the backend's current elements by element id."""

    def __init__(self, ids: torch.Tensor):
        self.sorted, self.perm = torch.sort(ids)

    def rows(self, ids: torch.Tensor) -> torch.Tensor:
        """Row per id, -1 when the backend no longer has it."""
        n = len(self.sorted)
        if n == 0 or len(ids) == 0:
            return torch.full_like(ids, -1)
        pos = torch.searchsorted(self.sorted, ids).clamp(max=n - 1)
        return torch.where(self.sorted[pos] == ids, self.perm[pos], torch.full_like(ids, -1))


class ElementEvidence:
    """The element rule's running sum per backend element id (with the last_update it saw)."""

    def __init__(self):
        self.ids = _empty_ids()                                   # sorted
        self.cusum = torch.zeros(0, dtype=torch.float64, device=DEV)
        self.last = _empty_ids()

    def get(self, ids: torch.Tensor, last_update: torch.Tensor) -> torch.Tensor:
        c = torch.zeros(len(ids), dtype=torch.float64, device=DEV)
        n = len(self.ids)
        if n and len(ids):
            pos = torch.searchsorted(self.ids, ids).clamp(max=n - 1)
            hit = (self.ids[pos] == ids) & (self.last[pos] == last_update)   # updated again: reset
            c[hit] = self.cusum[pos[hit]]
        return c

    def set(self, ids: torch.Tensor, cusum: torch.Tensor, last_update: torch.Tensor) -> None:
        order = torch.argsort(ids)
        self.ids, self.cusum, self.last = ids[order], cusum[order], last_update[order]


class ClosedStateBackground:
    """t2 markClosedObjectBackground (closed_object_background.cpp) with Khronos' RayChangeDetector.

    Background elements near a CLOSED object state (within sqrt(3) * map resolution of its surface,
    not updated by the backend after the state closed) are judged per element on the projected
    pixels after max(state support, element support): |range difference| <= 0.5 * resolution + 1 mm
    is geometric support (drops everything before it), beyond is absent, nearer is occluded
    (inconclusive). Absent when some window of window_size bins of temporal_resolution (from a bin
    with data) has an absent share above absence_confidence (room18/synthetic configs: 5 s, 5 bins,
    0.6). Incremental per round; decisions equal re-evaluating all stored frames every round.
    """

    def __init__(self, layer: "UpdateLayer", temporal_resolution_s: float = 5.0,
                 window_size: int = 5, absence_confidence: float = 0.6):
        self.layer = layer
        self.res_ns = int(temporal_resolution_s * 1e9)
        self.window = window_size
        self.conf = absence_confidence
        self.states: Dict[int, Optional[dict]] = {}   # id(fragment) -> candidates and running counts

    def _candidates(self, frag: Fragment, el: Elements, alive: torch.Tensor) -> Optional[dict]:
        rows = torch.nonzero(alive & (el.identity <= 0)).squeeze(1)
        if not len(rows) or frag.num_vertices == 0:
            return None
        radius = math.sqrt(3.0) * self.layer.cfg.map_resolution + float(el.extent[rows].max())
        pts = frag.points[torch.isfinite(frag.points).all(dim=1)]
        if not len(pts):
            return None
        lo, hi = pts.min(dim=0).values - radius, pts.max(dim=0).values + radius
        xyz = el.xyz[rows]
        box = ((xyz >= lo) & (xyz <= hi)).all(dim=1)
        box &= el.last_update[rows] <= frag.death_time          # rebuilt after closure: not owned
        rows = rows[box]
        if not len(rows):
            return None
        from scipy.spatial import cKDTree
        d, _ = cKDTree(pts.cpu().numpy()).query(el.xyz[rows].cpu().numpy(), k=1,
                                                distance_upper_bound=radius * (1 + 1e-6))
        reach = math.sqrt(3.0) * self.layer.cfg.map_resolution + el.extent[rows].cpu().numpy()
        rows = rows[torch.as_tensor(np.isfinite(d) & (d <= reach), device=DEV)]
        if not len(rows):
            return None
        through = torch.clamp(el.last_update[rows], min=max(frag.last_support_time, frag.last_confirmed_support))
        return dict(ids=el.ids[rows], xyz=el.xyz[rows].clone(), extent=el.extent[rows].clone(),
                    gone=torch.zeros(len(rows), dtype=torch.bool, device=DEV),
                    supported=through, last_geo=through.clone(), processed=int(through.min()),
                    bins=torch.zeros((len(rows), 0, 2), dtype=torch.int32, device=DEV), bin0=None)

    def run(self, stamp: int, el: Elements, by_id: _ById, alive: torch.Tensor) -> torch.Tensor:
        """Element ids judged absent in this round."""
        layer, cfg = self.layer, self.layer.cfg
        store = layer.store
        tol = 0.5 * cfg.map_resolution + 1e-3
        out = []
        for state in layer.registry.states.values():
            for frag in state.fragments:
                if frag.death_time is None or frag.death_time > stamp or frag.last_support_time >= stamp:
                    continue
                key = id(frag)
                if key not in self.states:
                    self.states[key] = self._candidates(frag, el, alive)
                c = self.states[key]
                if c is None:
                    continue
                rows = by_id.rows(c["ids"])
                live = rows >= 0
                live[live.clone()] = alive[rows[live]]
                c["gone"] |= ~live                                # retired or dropped by the backend
                keep = ~c["gone"]
                if not keep.any():
                    continue
                c["xyz"][live] = el.xyz[rows[live]]
                c["extent"][live] = el.extent[rows[live]]
                lo, hi = store.window(c["processed"] + 1, stamp)
                c["processed"] = stamp
                if hi <= lo:
                    continue
                stamps = store.stamp_tensor(lo, hi)
                p = store.project(lo, hi, c["xyz"])
                et, meas, query = p["etype"], p["measured"], p["query"]
                ok = (et != UNAVAILABLE) & (et != INVALID) & torch.isfinite(meas) & (meas > 0) & \
                    torch.isfinite(query)
                delta = meas - query
                tol_e = tol + c["extent"][None, :]
                valid = ok & (stamps[:, None] > c["supported"][None, :])
                present = valid & (torch.abs(delta) <= tol_e)
                absent = valid & (delta > tol_e)
                inconcl = valid & (delta < -tol_e)
                last_present = torch.where(present, stamps[:, None], torch.zeros_like(stamps)[:, None]).amax(0)
                c["last_geo"] = torch.maximum(c["last_geo"], last_present)
                # remove_past: only verdicts after the element's latest geometric support count
                after = stamps[:, None] > c["last_geo"][None, :]
                b = stamps // self.res_ns
                if c["bin0"] is None:
                    c["bin0"] = int(b[0])
                nb = int(b[-1]) - c["bin0"] + 1
                if nb > c["bins"].shape[1]:
                    pad = torch.zeros((len(c["ids"]), nb - c["bins"].shape[1], 2), dtype=torch.int32, device=DEV)
                    c["bins"] = torch.cat([c["bins"], pad], dim=1)
                # a new geometric support clears all earlier bins of that element
                c["bins"][last_present > 0] = 0
                col = (b - c["bin0"]).long()
                for kind, m in ((0, absent & after), (1, inconcl & after)):
                    rr, ff = torch.nonzero(m.T, as_tuple=True)
                    if len(rr):
                        flat = c["bins"][:, :, kind].reshape(-1)
                        flat.index_add_(0, rr * c["bins"].shape[1] + col[ff],
                                        torch.ones(len(rr), dtype=torch.int32, device=DEV))
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
                    out.append(c["ids"][absent_now])
                    c["gone"] |= absent_now
        return torch.cat(out) if out else _empty_ids()


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


class UpdateLayer:
    """Evidence, object states and element verdicts of one session chain.

    start_session(spec, prior) -> observe(frame) for every frame -> decide(stamp, last, elements)
    at every round boundary (returns the element ids to retire) -> end_session(out_dir) -> prior.
    """

    def __init__(self, cfg: LayerConfig):
        self.cfg = cfg

    # -- session ------------------------------------------------------------------------------
    def start_session(self, spec: SessionSpec, prior: Optional[dict] = None) -> None:
        cfg = self.cfg
        self.store = EvidenceStore(cfg.object_semantics, cfg.dynamic_semantics, cfg.max_range)
        self.semantic: Dict[int, int] = {}
        self.buf = RoundBuffer()
        self.el_evidence = ElementEvidence()
        self.closed = ClosedStateBackground(self)
        self.log: List[str] = []
        self.d1 = None
        self.rejected: List[Optional[torch.Tensor]] = []
        if cfg.core == "l2":
            self._start_l2(prior)
            return
        self.stats = SensorStatistics()
        if prior is not None:
            self.stats.load(prior["stats_path"])          # t2: sensor_statistics.txt next to the map
        self.absence = ObservedAbsence(self.store, EvidenceConfig(map_resolution=cfg.map_resolution,
                                                                  decision=cfg.decision), self.stats)
        self.registry = PersistentObjectState(cfg.map_resolution, cfg.high_mobility)
        for obj in (prior or {}).get("objects", []):
            frag = Fragment(points=to_dev(obj["points"]), normals=to_dev(obj["normals"]),
                            elements=_empty_ids(), birth_time=obj["first"], last_support_time=obj["last"],
                            last_confirmed_support=obj["last"], semantic_label=obj["semantic"])
            self.registry.seed_inherited(obj["identity"], frag, obj["dynamic"], (obj["first"], obj["last"]))
            self.semantic[obj["identity"]] = obj["semantic"]

    def _start_l2(self, prior: Optional[dict]) -> None:
        """session_backend.cpp: the absence model loads the previous session's sensor statistics and
        the registry starts from its canonical object nodes (initializeFromObjects)."""
        cfg = self.cfg
        self.absence = l2_evidence.AbsenceModel(self.store, l2_evidence.EvidenceConfig(), log=self.log)
        self.stats = self.absence                        # pooled in-place population (element rule)
        if prior is not None:
            self.absence.load_sensor_statistics(prior["stats_path"])
        self.registry = l2_state.PersistentObjectState(cfg.map_resolution, cfg.high_mobility)
        self.registry.log = self.log
        self.shown: Dict[int, l2_state.Materialized] = {}
        # saveSessionState depth_scales.txt: every earlier session's measured depth scale
        self.previous_depth_scales = list((prior or {}).get("depth_scales", []))
        self.depth_scale_now = None
        objects = []
        for obj in (prior or {}).get("objects", []):
            o = dict(obj)
            o["points"], o["normals"] = to_dev(obj["points"]), to_dev(obj["normals"])
            o["elements"] = _empty_ids()
            objects.append(o)
            self.semantic[obj["identity"]] = obj["semantic"]
        self.registry.initialize_from_objects(objects)
        if cfg.d1:
            self.d1 = D1(D1Config(voxel_size=cfg.map_resolution, truncation=cfg.truncation,
                                  max_range=cfg.max_range, dynamic_labels=tuple(cfg.dynamic_semantics)),
                         log=self.log)

    def motion(self, frame) -> Optional[np.ndarray]:
        """D1 front end on one backend frame (every frame, in order): its motion pixels, or None when
        the core has no D1 front end (t2)."""
        return None if self.d1 is None else self.d1.process(frame)

    def session_end_memory(self, el: Elements) -> Tuple[torch.Tensor, int]:
        """session_refusion step 5 on the backend's elements at the session end: the ids of memory elements
        the session's frames see through (or that are hidden in their band), and the session's first
        stamp (their state ended in the gap before it: a D3 change)."""
        start = self.store.first_stamp()
        if el.created is None or start is None or not len(el):
            return _empty_ids(), 0
        reg = self.registry
        ident = el.identity
        own = torch.zeros(int(ident.max()) + 1 if len(ident) else 1, dtype=torch.bool, device=DEV)
        own[0] = True
        for i in reg.tracked_ids():
            cur = reg.current_fragment(i)
            if cur is not None and cur.num_vertices and cur.birth_time < start and i < len(own):
                own[i] = True            # current, and its state did not begin in this session
        tested = (el.created < start) & own[ident.clamp(min=0)]
        n_object_state = int(((el.created < start) & ~own[ident.clamp(min=0)]).sum())
        idx = torch.nonzero(tested).squeeze(1)
        # steps 2-3: the present of this session's frames and the sensor's depth noise per range bin
        V, F, N = session_end.present_surface(self.store, self.cfg.object_voxel, self.rejected)
        sigma = session_end.noise_table(self.store, V, N, 2.0 * self.cfg.object_voxel, self.rejected)
        self.log.append("SIGMA_CM " + " ".join(f"{x * 100:.3f}" for x in sigma) + f" present_vertices={len(V)}")
        # P41: this session's depth scale; the memory's position error per metre of range, (|s_prev| + |s_now|)
        s_now, s_diag = session_end.depth_scale(self.store, self.cfg.truncation, self.cfg.depth_scale_stride,
                                                self.rejected)
        self.depth_scale_now = s_now
        error_per_metre = abs(s_now) + max([abs(x) for x in self.previous_depth_scales], default=0.0)
        self.log.append("DEPTH_SCALE " + " ".join(f"{k}={v}" for k, v in s_diag.items()))
        ev = session_end.memory_test(self.store, el.xyz[idx], el.extent[idx], el.extent[idx], sigma, self.rejected)
        seen, hidden, displaced = session_end.decide(ev, el.xyz[idx], el.extent[idx], sigma, error_per_metre, (V, F))
        out = el.ids[idx[seen | hidden | displaced]]
        self.log.append(f"MEMORY_TEST start={start} tested={len(idx)} object_state={n_object_state}"
                        f" any_hit={int((ev['hit'] > 0).sum())} any_through={int((ev['through'] > 0).sum())}"
                        f" seen_through={int(seen.sum())} hidden={int(hidden.sum())} displaced={int(displaced.sum())}"
                        f" error_per_metre={error_per_metre:.4g}")
        return out, start

    def _end_l2(self, out_dir: Path) -> dict:
        """saveSessionState: sensor statistics, and one node per identity with fragments (its CURRENT
        materialization, empty when the identity ended without a successor) carrying the mobility counts."""
        stats_path = out_dir / "sensor_statistics.txt"
        self.absence.save_sensor_statistics(stats_path)
        traj = []
        if self.d1 is not None:
            self.d1.finish()
            traj = self.d1.trajectories
        np.savez_compressed(out_dir / "d1_trajectories.npz",
                            track=np.array([t["track"] for t in traj], dtype=np.int64),
                            physical=np.array([-1 if t["physical"] is None else t["physical"] for t in traj],
                                              dtype=np.int64),
                            semantic=np.array([-1 if t["semantic"] is None else t["semantic"] for t in traj],
                                              dtype=np.int64),
                            stamps=np.array([np.array(t["stamps"], dtype=np.int64) for t in traj], dtype=object),
                            positions=np.array([t["positions"] for t in traj], dtype=object))
        if self.d1 is not None:
            self.log.append(f"D1_SUMMARY trajectories={len(traj)} physical="
                            f"{sum(t['physical'] is not None for t in traj)} motion_frames={self.d1.motion_frames}")
            import json
            with open(out_dir / "live_tracks.jsonl", "w") as fh:          # evaluate_live_dynamics.py input
                for rec in self.d1.live:
                    fh.write(json.dumps(rec) + "\n")
        (out_dir / "layer_log.txt").write_text("\n".join(self.log) + "\n")
        reg = self.registry
        objects = []
        for i in reg.tracked_ids():
            st = reg.states[i]
            if not st.fragments:
                continue
            m = reg.materialize(i)
            shown = m.fragments if m is not None and m.present else []
            pts = torch.cat([f.points for f in shown]) if shown else torch.zeros((0, 3), device=DEV)
            nrm = torch.cat([f.normals if f.normals is not None
                             else torch.full((f.num_vertices, 3), float("nan"), device=DEV) for f in shown]) \
                if shown else torch.zeros((0, 3), device=DEV)
            cur = st.fragments[st.current] if st.current is not None else st.fragments[-1]
            objects.append(dict(identity=i, points=_np(pts), normals=_np(nrm), semantic=cur.semantic_label,
                                first=min(f.birth_time for f in st.fragments), last=cur.last_support_time,
                                dynamic=st.has_dynamic_history, mobility_changes=st.mobility_changes,
                                mobility_continuations=st.mobility_continuations, bbox_valid=True))
        scales = self.previous_depth_scales + ([self.depth_scale_now] if self.depth_scale_now is not None else [])
        return dict(objects=objects, stats_path=str(stats_path), core="l2", depth_scales=scales)

    def end_session(self, out_dir) -> dict:
        """What the next session inherits: the displayed object states and the sensor statistics."""
        out_dir = Path(out_dir)
        if self.cfg.core == "l2":
            return self._end_l2(out_dir)
        stats_path = out_dir / "sensor_statistics.txt"
        self.stats.save(stats_path)
        (out_dir / "layer_log.txt").write_text("\n".join(self.log) + "\n")
        reg = self.registry
        objects = []
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
        return dict(objects=objects, stats_path=str(stats_path))

    # -- per frame ----------------------------------------------------------------------------
    def observe(self, frame: Frame) -> None:
        """Evidence of one frame and the instance-masked observations of its objects."""
        cfg = self.cfg
        dyn = dynamic_mask(frame, cfg.dynamic_semantics)
        self.store.ingest(frame, dyn)
        if cfg.core == "l2" and cfg.g5:
            # the session-end test reads the ranges without the rejected pixels (FrameArchive::offer)
            self.rejected.append(torch.as_tensor(dyn, dtype=torch.bool, device=DEV) if dyn.any() else None)
        depth = to_dev(frame.depth)
        inst = to_dev(frame.instance, torch.int64)
        sem = to_dev(frame.semantic, torch.int64) if frame.semantic is not None else None
        T = torch.as_tensor(frame.T_world_cam, dtype=torch.float32, device=DEV)
        normals = torch_pixel_normals(depth, frame.K, T[:3, :3])
        valid = torch.isfinite(depth)
        min_px = max(1, cfg.min_cluster_px_full // (cfg.pixel_step ** 2))
        obj = valid & (inst > 0)
        ids, counts = torch.unique(inst[obj], return_counts=True)
        for i, c in zip(ids.tolist(), counts.tolist()):
            if c < min_px:
                continue
            p, v, u = torch_backproject(depth, frame.K, T, valid & (inst == i))
            self._buffer(i, frame.stamp_ns, p, normals[v, u], _votes(sem[v, u]) if sem is not None else {},
                         self.store.n - 1)
        self.buf.stamps.append(frame.stamp_ns)

    def _buffer(self, i: int, stamp: int, p: torch.Tensor, n: torch.Tensor, fv: Dict[int, int],
                fidx: int) -> None:
        b = self.buf
        b.pts.setdefault(i, []).append(p)
        b.nrm.setdefault(i, []).append(n)
        b.fstamp.setdefault(i, []).append(stamp)
        b.fidx.setdefault(i, []).append(fidx)
        b.first.setdefault(i, stamp)
        b.last[i] = stamp
        b.frames[i] = b.frames.get(i, 0) + 1
        votes = b.sem.setdefault(i, {})
        for a, c in fv.items():
            votes[a] = votes.get(a, 0) + c

    # -- t2 static-surface frame selection ------------------------------------------------------
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
        if self.cfg.core == "l2":
            return l2_state.Observation(identity=i, points=pts[keep], normals=nrm[keep], first=first, last=last,
                                        semantic=self.semantic[i], reconstruction_frames=frames,
                                        elements=_empty_ids())
        return Observation(identity=i, points=pts[keep], normals=nrm[keep], first=first, last=last,
                           semantic=self.semantic[i], reconstruction_frames=frames)

    # -- per round ----------------------------------------------------------------------------
    def _verify_l2(self, stamp: int) -> None:
        """Backend::verifyCurrentObjectStates (backend_session.cpp @192c1cf): every tracked identity's
        CURRENT (slot 0) and session CURRENT (slot 1) faces the stored frames after its last support; the
        inherited slot carries the M2a prior into the observed-absence test; the look's measured ratio
        goes with the counts."""
        reg, cfg = self.registry, self.cfg
        for i in reg.tracked_ids():
            cur, scur = reg.current_fragment(i), reg.session_current_fragment(i)
            if (cur is None or cur.num_vertices == 0) and (scur is None or scur.num_vertices == 0):
                continue
            ev = {0: l2_state.SurfaceEvidence(), 1: l2_state.SurfaceEvidence()}
            for frag, slot in ((cur, 0), (scur, 1)):
                if frag is None or frag.num_vertices == 0:
                    continue
                prior = reg.change_prior_log_odds(i, slot)
                key = frag.uid if frag.uid != 0 else (slot | (1 << 63))       # absenceStateKey
                c = self.absence.count_current_surface(
                    i, frag.points, frag.normals, cfg.map_resolution,
                    max(frag.last_support_time, frag.last_confirmed_support), stamp, frag.birth_time, prior, key)
                ratio, measured, calibrated = self.absence.look_likelihood(i, key, stamp)
                ev[slot] = l2_state.SurfaceEvidence(
                    support_rays=c.support_rays, contradiction_rays=c.contradiction_rays,
                    surface_samples=c.surface_samples, absence_coverage_sufficient=c.absence_coverage_sufficient,
                    supported_votes=c.supported_votes, free_space_votes=c.free_space_votes,
                    replaced_by_other_votes=c.replaced_by_other_votes,
                    replaced_by_background_votes=c.replaced_by_background_votes, occluded_votes=c.occluded_votes,
                    unobserved_samples=c.unobserved_samples, latest_support_stamp=c.latest_support_stamp,
                    reliable_in_view=c.reliable_in_view, reliable_seen_through=c.reliable_seen_through,
                    reliable_samples=c.reliable_samples, measured_absence_log_ratio=ratio,
                    has_measured_absence_likelihood=measured, has_calibrated_absence_source=calibrated)
                self.log.append(f"STATE_EVIDENCE_WINDOW inst={i} slot={slot} after="
                                f"{max(frag.last_support_time, frag.last_confirmed_support)} through={stamp}"
                                f" support={c.support_rays} contradiction={c.contradiction_rays}"
                                f" reliable={c.reliable_samples} reliable_in_view={c.reliable_in_view}"
                                f" reliable_seen_through={c.reliable_seen_through} absence_llr={c.absence_llr:.6g}"
                                f" prior_log_odds={prior:.6g} total_samples={c.surface_samples}"
                                f" absence_coverage_sufficient={int(c.absence_coverage_sufficient)}")
            reg.resolve_current_evidence(i, ev[0], ev[1], stamp)

    def _canonicalize_l2(self) -> None:
        """canonicalizePhysicalObjects: per identity (in id order) its new segment of this round is ingested
        (applyPhysicalGeometry), then every identity is materialized."""
        reg, b = self.registry, self.buf
        new = {}
        for i, chunks in b.pts.items():
            fst, nrm, fidx = b.fstamp[i], b.nrm[i], b.fidx[i]
            moved = False
            # D1 (MeshObjectExtractor): a settled physical track displaced by >= min_dynamic_displacement
            # gives the next segment its motion history, reconstructed only from frames after the motion;
            # while it moves (displaced that far) it yields a trajectory, not a static segment.
            ep = self.d1.motion_episode(i) if self.d1 is not None else None
            if ep is not None:
                keep = [j for j, t in enumerate(fst) if t > ep["last_motion"]]
                if not keep:
                    self.d1.pending[i] = ep                 # no static frame yet: waits for the next segment
                    continue
                chunks, nrm, fidx, fst = [chunks[j] for j in keep], [nrm[j] for j in keep], \
                    [fidx[j] for j in keep], [fst[j] for j in keep]
                moved = True
            elif self.d1 is not None:
                dynamic, _, displacement = self.d1.dynamic_now(i)
                if dynamic and displacement >= self.d1.cfg.min_dynamic_displacement:
                    self.log.append(f"D1_HOLD inst={i} displacement={displacement:.4g} frames={len(fst)}")
                    continue
            k = self._static_start(i, chunks, fidx) if self.cfg.static_frames else 0
            obs = self._observation(i, torch.cat(chunks[k:]), torch.cat(nrm[k:]), fst[k], fst[-1],
                                    b.sem.get(i, {}), len(chunks) - k)
            obs.moved = moved
            if moved:
                self.log.append(f"D1_SEGMENT inst={i} first={obs.first} last_motion={ep['last_motion']}")
            new[i] = obs
        self.buf = RoundBuffer()
        cfg = self.cfg
        for i in list(new):
            p = new[i].points
            p = p[torch.isfinite(p).all(dim=1)]
            volume = float(torch.prod(p.max(dim=0).values - p.min(dim=0).values)) if len(p) else 0.0
            if not cfg.min_object_volume <= volume <= cfg.max_object_volume:
                self.log.append(f"SEGMENT_DROPPED inst={i} first={new[i].first} volume={volume:.6g}"
                                f" points={len(p)}")
                del new[i]
        for i in sorted(set(reg.tracked_ids()) | set(new)):
            if i in new:
                st = reg.states.get(i)
                first = new[i].first if st is None or not st.fragments else \
                    min([new[i].first] + [f.birth_time for f in st.fragments])
                self.shown[i] = reg.apply_physical_geometry(i, [new[i]], first)
            else:
                self.shown[i] = reg.materialize(i)

    def _verify(self, stamp: int) -> None:
        if self.cfg.core == "l2":
            return self._verify_l2(stamp)
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

    def _element_rule(self, stamp: int, el: Elements, alive: torch.Tensor) -> torch.Tensor:
        rows = torch.nonzero(alive & (el.identity <= 0)).squeeze(1)
        if not len(rows):
            return _empty_ids()
        ids, pts, last_seen = el.ids[rows], el.xyz[rows], el.last_update[rows]
        nrm, ext = el.normal[rows], el.extent[rows]
        has_n = torch.isfinite(nrm).all(dim=1)
        nrm0 = torch.nan_to_num(nrm)
        tol = self.absence.config.surface_match_tolerance
        min_cos = math.cos(math.radians(self.absence.config.max_absence_incidence_deg))
        verdict = torch.zeros(len(ids), dtype=torch.int8, device=DEV)   # 0 none, 1 on surface, 2 through
        chosen, last_t = [], None
        for t in self.buf.stamps:
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
            verdict[measured & (torch.abs(delta) <= tol + ext) & later] = 1
            verdict[measured & (delta > tol + ext) & facing & later] = 2
        pn, ps = self.stats.pooled_n, self.stats.pooled_sum
        p_miss = ps / pn if pn >= 3 else 0.05          # uninformative population of prior()
        step = -math.log(min(0.995, max(0.005, p_miss)))
        c = self.el_evidence.get(ids, last_seen)
        c[verdict == 1] = 0.0
        c[verdict == 2] += step
        retire = (verdict == 2) & (c > LN99)
        self.el_evidence.set(ids[~retire], c[~retire], last_seen[~retire])
        return ids[retire]

    def _ingest(self) -> None:
        b = self.buf
        obs = []
        for i, chunks in b.pts.items():
            k = self._static_start(i, chunks, b.fidx[i]) if self.cfg.static_frames else 0
            obs.append(self._observation(i, torch.cat(chunks[k:]), torch.cat(b.nrm[i][k:]),
                                         b.fstamp[i][k] if k else b.first[i], b.last[i],
                                         b.sem.get(i, {}), b.frames[i] - k))
        for o in sorted(obs, key=lambda o: (o.first, o.last, o.identity)):
            self.registry.ingest(o)

    @staticmethod
    def _open_fragments(state) -> list:
        """Every not-yet-closed fragment of a physical state, including the in-session (B) state."""
        out = [f for f in state.fragments if f.death_time is None]
        if state.observed_new is not None:
            out.append(state.observed_new)
        if state.b_session is not None:
            out += UpdateLayer._open_fragments(state.b_session)
        return out

    def _support(self, el: Elements, alive: torch.Tensor) -> torch.Tensor:
        """Object support (module notes): elements of a tracked identity farther than the sensor
        tolerance from every live state of that identity."""
        from scipy.spatial import cKDTree
        reg = self.registry
        tol = self.absence.config.surface_match_tolerance
        obj = alive & (el.identity > 0)
        if not obj.any():
            return _empty_ids()
        out = []
        for i in torch.unique(el.identity[obj]).tolist():
            state = reg.states.get(i)
            if state is None:
                continue                                  # never observed by the layer: no belief
            rows = torch.nonzero(obj & (el.identity == i)).squeeze(1)
            live = [f.points for f in self._open_fragments(state) if f.num_vertices]
            if not live:
                out.append(el.ids[rows])                  # the identity shows nothing now
                continue
            pts = torch.cat(live)
            pts = pts[torch.isfinite(pts).all(dim=1)]
            reach = tol + el.extent[rows].cpu().numpy()
            d, _ = cKDTree(pts.cpu().numpy()).query(el.xyz[rows].cpu().numpy(), k=1,
                                                    distance_upper_bound=float(reach.max()) * (1 + 1e-6))
            far = torch.as_tensor(~(np.isfinite(d) & (d <= reach)), device=el.ids.device)
            if far.any():
                out.append(el.ids[rows[far]])
        return torch.cat(out) if out else _empty_ids()

    def state_intervals(self) -> Dict[int, list]:
        """(birth, death) of every object state, for backends whose map is estimated per state
        (interface.Backend.consumes_state_intervals)."""
        return self.registry.state_intervals()

    def decide(self, stamp: int, last: bool, el: Elements, object_support: bool = True) -> Dict[str, torch.Tensor]:
        """One reconciliation round at `stamp`; the element ids to retire, by reason. object_support=False
        for a backend that takes the state intervals instead: its object geometry follows the states."""
        reg = self.registry
        by_id = _ById(el.ids)
        alive = torch.ones(len(el), dtype=torch.bool, device=DEV)

        def drop(ids: torch.Tensor) -> None:
            r = by_id.rows(ids)
            alive[r[r >= 0]] = False

        if self.cfg.core == "l2":
            # Backend::runChangeDetectionThread with the session hooks (192c1cf): sessionBeforeReconcile =
            # one observation event, verify, (terminal: finalize), closed-state background; reconcile (the
            # element rule); sessionAfterReconcile = canonicalize, (terminal: verify, finalize, closed-state
            # background, canonicalize = TERMINAL_STATE_DRAIN).
            reg.begin_observation_event(stamp)
            self._verify_l2(stamp)
            if last:
                reg.finalize_pending_absences(stamp)
            closed_bg = [self.closed.run(stamp, el, by_id, alive)]
            drop(closed_bg[0])
            rule = self._element_rule(stamp, el, alive)
            drop(rule)
            self._canonicalize_l2()
            if last:
                self._verify_l2(stamp)
                reg.finalize_pending_absences(stamp)
                closed_bg.append(self.closed.run(stamp, el, by_id, alive))
                drop(closed_bg[-1])
                self._canonicalize_l2()
            objects = self._support(el, alive) if object_support else _empty_ids()
            out = dict(closed_background=torch.cat(closed_bg), element_rule=rule, object_support=objects)
            self.log.append(f"{stamp} RETIRE " + " ".join(f"{k}={len(v)}" for k, v in out.items()))
            return out
        self._verify(stamp)
        closed_bg = [self.closed.run(stamp, el, by_id, alive)]
        drop(closed_bg[0])
        rule = self._element_rule(stamp, el, alive)
        drop(rule)
        self._ingest()
        if last:
            reg.finalize_pending_absences(stamp)
            self._verify(stamp)
            reg.finalize_pending_absences(stamp)
            closed_bg.append(self.closed.run(stamp, el, by_id, alive))
            drop(closed_bg[-1])
        objects = self._support(el, alive) if object_support else _empty_ids()
        self.buf = RoundBuffer()
        out = dict(closed_background=torch.cat(closed_bg), element_rule=rule, object_support=objects)
        self.log.append(f"{stamp} RETIRE " + " ".join(f"{k}={len(v)}" for k, v in out.items()))
        return out
