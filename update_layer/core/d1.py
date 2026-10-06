"""D1 front end of the layer: the TSDF version's active-window motion pipeline, without a TSDF map.

Ported from the user's Khronos fork at 192c1cf (session_update_baseline/ports/mapping_core/khronos) and
the Hydra integrator it calls (ros2_ws/src/hydra), per frame in ActiveWindow::spinOnce order:

  1. FreeSpaceMotionDetector (free_space_motion_detector.cpp; adapted by Khronos from Dynablox, Schmid
     et al., RA-L 2023, sec. III): the frame's points that fall into ever-free voxels of the free-space
     state built from the frames before are motion seeds; seeds grow through 26-connected ever-free
     voxels and take in their point-occupied neighbours; clusters closer than min_separation_distance
     voxels merge; clusters below min_cluster_size pixels are dropped.
  2. InstanceForwarding (instance_forwarding.cpp): one cluster per physical instance id (valid pixels,
     >= min_cluster_size); promoteDynamicSemanticClusters: a motion cluster that mostly (>= 0.5) covers
     a dynamic class or a static class is removed, and every 8-connected component of a dynamic class
     becomes a dynamic cluster with that class; ids and the dynamic image are rebuilt.
  3. ExternalTracker (external_tracker.cpp): physical tracks keyed by the instance id; a physical track
     is dynamic while a dynamic cluster overlaps its pixels (IoU >= min_cross_iou), and settles after
     settle_time without; dynamic-only tracks associate by bounding-box centre within
     max_dynamic_distance.
  4. ProjectiveIntegrator::updateMap (Hydra projective_integrator.cpp, InterpolatorAdaptive, Camera)
     with the integration mask of ActiveWindow::updateMap (dynamic labels + this frame's dynamic
     clusters; the mask acts only inside the truncation band, so free space in front of a moving body
     is still observed), then TrackingIntegrator::updateBlocks (tracking_integrator.cpp: last occupied,
     active window, ever-free with the 18-neighbourhood) and resetInactive.

The voxel state carries only what these decisions read (fused distance and weight, last observed /
occupied stamps, ever-free, to-remove); it is the layer's, not the backend's map, so every backend gets
the same D1 decisions. MeshObjectExtractor's D1 rules (computeDynamicDisplacement,
extractDynamicObject, appendDynamicHistory) give the outputs: trajectories of dynamic tracks and of
physical tracks displaced by >= min_dynamic_displacement, and the motion episodes of physical tracks.
Config values: the L2_FINAL2 runs (glog of real_a and synthetic session_a).

Known differences, all at the representation boundary: per-voxel updates are vectorised (Jacobi) where
the C++ updates blocks in parallel threads (ever-free reads neighbours being updated in the same pass:
order-dependent there too); interpolateID ties of the mask are broken towards "masked".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .evidence import DEV

_BIG = 1 << 20


@dataclass
class D1Config:
    voxel_size: float                       # active window volumetric_map.voxel_size
    truncation: float                       # volumetric_map.truncation_distance
    max_range: float                        # sensor / detector max range
    min_range: float = 0.1                  # sensor min_range
    voxels_per_side: int = 16
    max_depth_difference: float = 0.05      # InterpolatorAdaptive.max_depth_difference_m
    weight_dropoff_epsilon: float = -1.0    # [voxels when negative]
    min_measurement_weight: float = 1e-4
    max_weight: float = 1e5
    # TrackingIntegrator
    temporal_buffer: float = 2.0
    temporal_window: float = 3.0
    occupancy_threshold: float = -1.5       # [voxels when negative]
    # FreeSpaceMotionDetector
    motion_min_cluster: int = 3000
    motion_max_cluster: int = 1000000
    min_separation: float = 2.0             # voxels
    # InstanceForwarding
    instance_min_cluster: int = 50
    dynamic_overlap: float = 0.5
    # ExternalTracker
    min_cross_iou: float = 0.1
    max_dynamic_distance: float = 1.0
    settle_time: float = 1.0
    min_num_observations: int = 15
    # MeshObjectExtractor
    min_dynamic_displacement: float = 1.0
    accept_semantic_dynamic_tracks: bool = True
    dynamic_labels: Tuple[int, ...] = ()


def _pack(k: torch.Tensor) -> torch.Tensor:
    k = k + _BIG
    return (k[..., 0] << 42) | (k[..., 1] << 21) | k[..., 2]


_N26 = torch.tensor([(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                     if (dx, dy, dz) != (0, 0, 0)], dtype=torch.int64)
_N18 = torch.tensor([o for o in _N26.tolist() if sum(abs(x) for x in o) <= 2], dtype=torch.int64)


def _sec(ns: torch.Tensor) -> torch.Tensor:
    return ns.to(torch.float64) * 1e-9


# --------------------------------------------------------------------------------- tracks
@dataclass
class Track:
    id: int
    physical_instance_id: Optional[int]
    is_dynamic: bool = False
    has_dynamic_history: bool = False
    last_motion_seen: int = 0
    first_seen: int = 0
    last_seen: int = 0
    last_centroid: Optional[np.ndarray] = None
    semantics: Optional[int] = None
    is_active: bool = True
    observations: List[Tuple[int, int, int]] = field(default_factory=list)   # stamp, semantic id, dynamic id
    # per observation with a dynamic cluster: (stamp, centroid of the cluster's points)
    dynamic_centroids: List[Tuple[int, np.ndarray]] = field(default_factory=list)


class D1:
    """Per-session D1 front end. process(frame) -> the frame's dynamic pixels (bool HxW)."""

    FIRST_DYNAMIC_TRACK_ID = 1 << 16

    def __init__(self, cfg: D1Config, log: Optional[list] = None):
        self.cfg = cfg
        self.log = log if log is not None else []
        V = cfg.voxels_per_side
        self.V3 = V ** 3
        lin = torch.arange(self.V3, device=DEV)
        self.local = torch.stack([lin % V, (lin // V) % V, lin // (V * V)], dim=1)      # x fastest
        self.slots: Dict[Tuple[int, int, int], int] = {}
        self.free_slots: List[int] = []
        cap = 64
        self._alloc(cap)
        self._lookup_dirty = True
        self.tracks: List[Track] = []
        self.next_dynamic_id = self.FIRST_DYNAMIC_TRACK_ID
        self.trajectories: List[dict] = []                     # finished dynamic tracks (4D)
        self.pending: Dict[int, dict] = {}                     # identity -> motion episode not yet handed over
        self.motion_frames = 0

    # -- block pool -------------------------------------------------------------------------
    def _alloc(self, cap: int) -> None:
        V3 = self.V3

        def grow(t, fill, dtype):
            new = torch.full((cap, V3), fill, dtype=dtype, device=DEV)
            if t is not None:
                new[:len(t)] = t
            return new
        old = getattr(self, "dist", None)
        n_old = 0 if old is None else len(old)
        self.dist = grow(getattr(self, "dist", None), 0.0, torch.float32)
        self.weight = grow(getattr(self, "weight", None), 0.0, torch.float32)
        self.last_obs = grow(getattr(self, "last_obs", None), 0, torch.int64)
        self.last_occ = grow(getattr(self, "last_occ", None), 0, torch.int64)
        self.ever_free = grow(getattr(self, "ever_free", None), False, torch.bool)
        self.to_remove = grow(getattr(self, "to_remove", None), False, torch.bool)
        self.active = grow(getattr(self, "active", None), False, torch.bool)
        bk = torch.zeros((cap, 3), dtype=torch.int64, device=DEV)
        if n_old:
            bk[:n_old] = self.block_key
        self.block_key = bk
        self.free_slots += list(range(cap - 1, n_old - 1, -1))

    def _new_block(self, key: Tuple[int, int, int]) -> int:
        if not self.free_slots:
            self._alloc(2 * len(self.dist))
        s = self.free_slots.pop()
        self.slots[key] = s
        self.dist[s] = 0.0
        self.weight[s] = 0.0
        self.last_obs[s] = 0
        self.last_occ[s] = 0
        self.ever_free[s] = False
        self.to_remove[s] = False
        self.active[s] = False
        self.block_key[s] = torch.tensor(key, device=DEV)
        self._lookup_dirty = True
        return s

    def _remove_block(self, key) -> None:
        s = self.slots.pop(key)
        self.free_slots.append(s)
        self._lookup_dirty = True

    def _lookup_table(self):
        if self._lookup_dirty:
            if self.slots:
                keys = torch.tensor(list(self.slots.keys()), dtype=torch.int64, device=DEV)
                slots = torch.tensor(list(self.slots.values()), dtype=torch.int64, device=DEV)
                packed = _pack(keys)
                order = torch.argsort(packed)
                self._lk, self._ls = packed[order], slots[order]
            else:
                self._lk = torch.zeros(0, dtype=torch.int64, device=DEV)
                self._ls = torch.zeros(0, dtype=torch.int64, device=DEV)
            self._lookup_dirty = False
        return self._lk, self._ls

    def _voxel(self, g: torch.Tensor):
        """Global voxel indices (N, 3) -> (slot or -1, linear local index)."""
        V = self.cfg.voxels_per_side
        b = torch.div(g, V, rounding_mode="floor")
        loc = g - b * V
        lin = loc[:, 0] + V * (loc[:, 1] + V * loc[:, 2])
        lk, ls = self._lookup_table()
        if not len(lk):
            return torch.full((len(g),), -1, dtype=torch.int64, device=DEV), lin
        pk = _pack(b)
        pos = torch.searchsorted(lk, pk).clamp(max=len(lk) - 1)
        slot = torch.where(lk[pos] == pk, ls[pos], torch.full_like(pk, -1))
        return slot, lin

    # -- per frame --------------------------------------------------------------------------
    def process(self, frame) -> np.ndarray:
        cfg = self.cfg
        stamp = int(frame.stamp_ns)
        K = frame.K
        depth = torch.as_tensor(frame.depth, dtype=torch.float32, device=DEV)
        H, W = depth.shape
        T = torch.as_tensor(frame.T_world_cam, dtype=torch.float32, device=DEV)
        uu = torch.arange(W, device=DEV, dtype=torch.float32)[None, :].expand(H, W)
        vv = torch.arange(H, device=DEV, dtype=torch.float32)[:, None].expand(H, W)
        xr = (uu + K.offset - K.cx) / K.fx
        yr = (vv + K.offset - K.cy) / K.fy
        valid_depth = torch.isfinite(depth) & (depth > 0)
        z = torch.where(valid_depth, depth, torch.zeros_like(depth))
        cam = torch.stack([xr * z, yr * z, z], dim=-1)
        rng = torch.linalg.norm(cam, dim=-1)                   # Hydra range image (0 where invalid)
        world = cam @ T[:3, :3].T + T[:3, 3]
        label = torch.as_tensor(frame.semantic, dtype=torch.int64, device=DEV) if frame.semantic is not None \
            else torch.zeros((H, W), dtype=torch.int64, device=DEV)
        inst = torch.as_tensor(frame.instance, dtype=torch.int64, device=DEV)
        dyn_label = torch.zeros_like(label, dtype=torch.bool)
        for d in cfg.dynamic_labels:
            dyn_label |= label == d

        # 1. motion detection on the free-space state of the frames before this one
        dynamic_clusters = self._motion_clusters(world, rng, valid_depth)
        # 2. instance clusters and dynamic-semantic promotion
        object_valid = valid_depth & (rng <= cfg.max_range)          # isValidObjectMeasurementPixel
        semantic_clusters = self._instance_clusters(inst, label, object_valid)
        dynamic_clusters = self._promote(dynamic_clusters, label, dyn_label, object_valid)
        dynamic_image = torch.zeros((H, W), dtype=torch.int32, device=DEV)
        for k, c in enumerate(dynamic_clusters):
            c["id"] = k + 1
            dynamic_image.view(-1)[c["pix"]] = k + 1
        for c in semantic_clusters + dynamic_clusters:
            p = world.view(-1, 3)[c["pix"]]
            c["center"] = ((p.min(0).values + p.max(0).values) / 2).cpu().numpy()   # BoundingBox centre
            c["mean"] = p.mean(0).cpu().numpy()                                       # computeCentroid
        # 3. tracking
        self._track(stamp, semantic_clusters, dynamic_clusters, dynamic_image)
        # 4. integration with the integration mask, then tracking / ever-free / inactive reset
        mask = dyn_label | (dynamic_image > 0)
        self._integrate(stamp, T, K, rng, mask, H, W)
        if dynamic_clusters:
            self.motion_frames += 1
        return (dynamic_image > 0).cpu().numpy()

    # -- 1. FreeSpaceMotionDetector -------------------------------------------------------------
    def _motion_clusters(self, world, rng, valid) -> List[dict]:
        cfg = self.cfg
        ok = valid & (rng > 0) & (rng <= cfg.max_range)
        pix = torch.nonzero(ok.view(-1)).squeeze(1)
        if not len(pix) or not self.slots:
            return []
        p = world.view(-1, 3)[pix]
        g = torch.floor(p / cfg.voxel_size).to(torch.int64)
        slot, lin = self._voxel(g)
        inb = slot >= 0                                          # tracking_layer.getBlockPtr(p_W)
        if not inb.any():
            return []
        pix, g, slot, lin = pix[inb], g[inb], slot[inb], lin[inb]
        seed_pt = self.ever_free[slot, lin]
        if not seed_pt.any():
            return []
        # point-occupied voxels and seeds
        key = _pack(g)
        vkeys, vinv = torch.unique(key, return_inverse=True)
        is_seed = torch.zeros(len(vkeys), dtype=torch.bool, device=DEV)
        is_seed[vinv[seed_pt]] = True
        vg = torch.zeros((len(vkeys), 3), dtype=torch.int64, device=DEV)
        vg[vinv] = g
        # nodes: seeds and point voxels 26-adjacent to a seed (the clusters' voxels); final clusters are
        # the connected components of the nodes under "closer than min_separation" (mergeClusters)
        nb = vg[:, None, :] + _N26.to(DEV)[None]
        pos = torch.searchsorted(vkeys, _pack(nb)).clamp(max=len(vkeys) - 1)
        nb_exists = vkeys[pos] == _pack(nb)
        nb_idx = torch.where(nb_exists, pos, torch.full_like(pos, -1))
        adj_seed = (nb_exists & is_seed[pos]).any(dim=1)
        node = is_seed | adj_seed
        if cfg.min_separation > 2.0:
            raise NotImplementedError("min_separation > 2 voxels needs a wider merge neighbourhood")
        # min_separation 2 voxels: norm < 2 <=> 26-neighbourhood (sqrt(3) < 2 <= 2) -> components
        label = torch.where(node, torch.arange(len(vkeys), device=DEV), torch.full((len(vkeys),), _BIG * 4,
                                                                                     device=DEV, dtype=torch.int64))
        nb_node = (nb_idx >= 0) & node[nb_idx.clamp(min=0)]
        while True:
            nl = torch.where(nb_node, label[nb_idx.clamp(min=0)], torch.full_like(nb_idx, _BIG * 4)).min(dim=1).values
            new = torch.where(node, torch.minimum(label, nl), label)
            if torch.equal(new, label):
                break
            label = new
        clusters = []
        pt_label = label[vinv]
        in_cluster = node[vinv]
        ids = torch.unique(pt_label[in_cluster])
        for cid in ids.tolist():
            sel = in_cluster & (pt_label == cid)
            n = int(sel.sum())
            if n < cfg.motion_min_cluster or n > cfg.motion_max_cluster:
                continue
            clusters.append(dict(pix=pix[sel], semantics=None))
        return clusters

    # -- 2. InstanceForwarding ---------------------------------------------------------------
    def _instance_clusters(self, inst, label, valid) -> List[dict]:
        cfg = self.cfg
        m = (inst > 0) & valid
        pix = torch.nonzero(m.view(-1)).squeeze(1)
        if not len(pix):
            return []
        ids = inst.view(-1)[pix]
        out = []
        for i in torch.unique(ids).tolist():
            sel = pix[ids == i]
            if len(sel) < cfg.instance_min_cluster:
                continue
            labs, cnt = torch.unique(label.view(-1)[sel], return_counts=True)
            out.append(dict(id=int(i), pix=sel, semantics=int(labs[torch.argmax(cnt)])))
        return out

    def _promote(self, clusters, label, dyn_label, valid) -> List[dict]:
        cfg = self.cfg
        kept = []
        for c in clusters:
            lab = label.view(-1)[c["pix"]]
            dynamic = dyn_label.view(-1)[c["pix"]]
            total = float(len(c["pix"]))
            static = (~dynamic) & (lab != 0)
            if float(dynamic.sum()) / total >= cfg.dynamic_overlap or float(static.sum()) / total >= cfg.dynamic_overlap:
                continue
            kept.append(c)
        if cfg.dynamic_labels:
            from scipy.ndimage import label as cc
            lab_np = label.cpu().numpy()
            valid_np = valid.cpu().numpy()
            for d in cfg.dynamic_labels:
                comp, n = cc(lab_np == d, structure=np.ones((3, 3), dtype=bool))
                if not n:
                    continue
                comp = np.where(valid_np, comp, 0).ravel()
                order = np.argsort(comp, kind="stable")
                cs = comp[order]
                starts = np.searchsorted(cs, np.arange(1, n + 1))
                ends = np.searchsorted(cs, np.arange(1, n + 1), side="right")
                for a, b in zip(starts, ends):
                    if b - a < cfg.instance_min_cluster:
                        continue
                    kept.append(dict(pix=torch.as_tensor(order[a:b], device=DEV), semantics=int(d)))
        return kept

    # -- 3. ExternalTracker --------------------------------------------------------------------
    def _track(self, stamp, semantic_clusters, dynamic_clusters, dynamic_image) -> None:
        cfg = self.cfg
        used_dynamic = set()
        flat_dyn = dynamic_image.view(-1)
        for obs in semantic_clusters:
            best, best_iou = None, cfg.min_cross_iou
            for d in dynamic_clusters:
                if d["id"] in used_dynamic:
                    continue
                if len(obs["pix"]) <= len(d["pix"]):
                    inter = int((flat_dyn[obs["pix"]] == d["id"]).sum())
                else:
                    inter = int(torch.isin(d["pix"], obs["pix"]).sum())
                union = len(obs["pix"]) + len(d["pix"]) - inter
                iou = inter / union if union else 0.0
                if iou >= best_iou and iou > 0.0:
                    best_iou, best = iou, d
            track = next((t for t in self.tracks if t.id == obs["id"] and t.id < self.FIRST_DYNAMIC_TRACK_ID), None)
            if track is None:
                track = Track(id=obs["id"], physical_instance_id=obs["id"], first_seen=stamp)
                self.tracks.append(track)
            self._update_physical(track, obs, best, stamp)
            if best is not None:
                used_dynamic.add(best["id"])
        for track in self.tracks:
            if not track.is_dynamic or track.id < self.FIRST_DYNAMIC_TRACK_ID:
                continue
            best, best_d = None, cfg.max_dynamic_distance
            for d in dynamic_clusters:
                if d["id"] in used_dynamic:
                    continue
                dist = float(np.linalg.norm(d["center"] - track.last_centroid))
                if dist <= best_d:
                    best_d, best = dist, d
            if best is not None:
                self._update_dynamic(track, best, stamp)
                used_dynamic.add(best["id"])
        for d in dynamic_clusters:
            if d["id"] not in used_dynamic:
                used_dynamic.add(d["id"])
                track = Track(id=self.next_dynamic_id, physical_instance_id=None, is_dynamic=True,
                              has_dynamic_history=True, last_motion_seen=stamp, first_seen=stamp)
                self.next_dynamic_id += 1
                self.tracks.append(track)
                self._update_dynamic(track, d, stamp)
        # updateTrackingDuration; inactive dynamic-only tracks leave the window as trajectories
        window = int(cfg.temporal_window * 1e9)
        min_time = stamp - window if stamp > window else 0
        keep = []
        for t in self.tracks:
            t.is_active = t.last_seen >= min_time
            if not t.is_active:
                # extractInactiveObjects: the track leaves the active window with its extraction
                if t.physical_instance_id is None:
                    self._finish_dynamic(t)
                else:
                    self._episode(t)
                continue
            keep.append(t)
        self.tracks = keep

    def _update_physical(self, track: Track, obs: dict, dyn: Optional[dict], stamp: int) -> None:
        if dyn is not None:
            track.is_dynamic = True
            track.has_dynamic_history = True
            track.last_motion_seen = stamp
            track.dynamic_centroids.append((stamp, dyn["mean"]))
        elif track.is_dynamic and track.physical_instance_id and \
                stamp >= track.last_motion_seen + int(self.cfg.settle_time * 1e9):
            # still identified but no longer overlapping motion: static-current again at the new pose
            track.is_dynamic = False
            self.log.append(f"D1_SETTLED inst={track.id} stamp={stamp} last_motion={track.last_motion_seen}"
                            f" displacement={self.displacement(track):.4g}")
            self._episode(track)
        if obs.get("semantics") is not None:
            track.semantics = obs["semantics"]
        track.last_centroid = dyn["center"] if dyn is not None else obs["center"]
        track.last_seen = stamp
        track.observations.append((stamp, obs["id"], dyn["id"] if dyn is not None else -1))

    def _update_dynamic(self, track: Track, obs: dict, stamp: int) -> None:
        track.is_dynamic = True
        track.has_dynamic_history = True
        track.last_motion_seen = stamp
        if obs.get("semantics") is not None:
            track.semantics = obs["semantics"]
        track.last_centroid = obs["center"]
        track.last_seen = stamp
        track.observations.append((stamp, -1, obs["id"]))
        track.dynamic_centroids.append((stamp, obs["mean"]))

    # -- MeshObjectExtractor D1 rules --------------------------------------------------------------
    @staticmethod
    def displacement(track: Track) -> float:
        """computeDynamicDisplacement: largest distance of a motion-cluster centroid from the first."""
        if len(track.dynamic_centroids) < 2:
            return 0.0
        first = track.dynamic_centroids[0][1]
        return float(max(np.linalg.norm(c - first) for _, c in track.dynamic_centroids[1:]))

    def _finish_dynamic(self, track: Track) -> None:
        """extractDynamicObject for a dynamic-only track leaving the window: its trajectory, unless it
        moved less than min_dynamic_displacement and is not of a dynamic class."""
        cfg = self.cfg
        if not track.dynamic_centroids:
            return
        moved = self.displacement(track) >= cfg.min_dynamic_displacement
        semantic_dynamic = track.semantics is not None and track.semantics in cfg.dynamic_labels
        if not moved and not (cfg.accept_semantic_dynamic_tracks and semantic_dynamic):
            self.log.append(f"D1_DROPPED track={track.id} displacement={self.displacement(track):.4g}")
            return
        self.trajectories.append(dict(track=track.id, physical=None, semantic=track.semantics,
                                      stamps=[s for s, _ in track.dynamic_centroids],
                                      positions=np.stack([c for _, c in track.dynamic_centroids])))

    def _episode(self, t: Track) -> None:
        """appendDynamicHistory + the has_dynamic_history of the segment extracted after the motion: a
        physical track displaced by >= min_dynamic_displacement (below it, the static fallback clears
        the history). The episode waits for the layer's next segment of the identity."""
        if not t.has_dynamic_history or not t.dynamic_centroids:
            return
        d = self.displacement(t)
        if d < self.cfg.min_dynamic_displacement:
            self.log.append(f"D1_STATIC_FALLBACK inst={t.id} displacement={d:.4g}")
            return
        traj = dict(track=t.id, physical=t.physical_instance_id, semantic=t.semantics,
                    stamps=[s for s, _ in t.dynamic_centroids],
                    positions=np.stack([c for _, c in t.dynamic_centroids]))
        if self.pending.get(t.physical_instance_id, {}).get("last_motion") == t.last_motion_seen:
            return
        self.pending[t.physical_instance_id] = dict(last_motion=t.last_motion_seen, trajectory=traj)
        self.trajectories.append(traj)
        self.log.append(f"D1_EPISODE inst={t.physical_instance_id} last_motion={t.last_motion_seen}"
                        f" displacement={d:.4g} samples={len(t.dynamic_centroids)}")

    def motion_episode(self, identity: int) -> Optional[dict]:
        """The pending motion episode of an identity (handed over once)."""
        return self.pending.pop(identity, None)

    def dynamic_now(self, identity: int) -> Tuple[bool, int, float]:
        """(the identity's track is dynamic now, its last motion stamp, its displacement so far)."""
        t = next((t for t in self.tracks if t.physical_instance_id == identity), None)
        if t is None:
            return False, 0, 0.0
        return t.is_dynamic, (t.last_motion_seen if t.has_dynamic_history else 0), self.displacement(t)

    def finish(self) -> None:
        """finishMapping: every track leaves the window."""
        for t in self.tracks:
            if t.physical_instance_id is None:
                self._finish_dynamic(t)
        self.tracks = [t for t in self.tracks if t.physical_instance_id is not None]

    # -- 4. ProjectiveIntegrator + TrackingIntegrator ---------------------------------------------
    def _frustum_blocks(self, T, K, rng, H, W) -> torch.Tensor:
        """Blocks a voxel update can touch: along every 4th pixel's ray up to min(range + truncation,
        max range), sampled at half a block (findBlocksInViewFrustum allocates the frustum; blocks
        that receive no update are removed again, as there)."""
        cfg = self.cfg
        bs = cfg.voxel_size * cfg.voxels_per_side
        s = 4
        r = rng[::s, ::s]
        uu = torch.arange(0, W, s, device=DEV, dtype=torch.float32)[None, :].expand_as(r)
        vv = torch.arange(0, H, s, device=DEV, dtype=torch.float32)[:, None].expand_as(r)
        d = torch.stack([(uu + K.offset - K.cx) / K.fx, (vv + K.offset - K.cy) / K.fy, torch.ones_like(r)], -1)
        d = d / torch.linalg.norm(d, dim=-1, keepdim=True)
        far = torch.where(r > 0, torch.clamp(r + cfg.truncation, max=cfg.max_range),
                          torch.full_like(r, cfg.max_range))
        steps = torch.arange(0.0, cfg.max_range + bs, bs / 2, device=DEV)
        keys = []
        for t0 in steps.split(16):
            pts = d[..., None, :] * t0[None, None, :, None]
            ok = t0[None, None, :] <= far[..., None]
            pw = pts[ok] @ T[:3, :3].T + T[:3, 3]
            keys.append(torch.unique(torch.floor(pw / bs).to(torch.int64), dim=0))
        keys = torch.unique(torch.cat(keys), dim=0)
        return torch.unique(torch.cat([keys + o for o in
                                       torch.tensor([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1)
                                                     for c in (-1, 0, 1)], device=DEV)]), dim=0)

    def _integrate(self, stamp, T, K, rng, mask, H, W) -> None:
        cfg = self.cfg
        vs, trunc, V = cfg.voxel_size, cfg.truncation, cfg.voxels_per_side
        blocks = self._frustum_blocks(T, K, rng, H, W)
        new = []
        bslots = []
        for key in map(tuple, blocks.tolist()):
            s = self.slots.get(key)
            if s is None:
                s = self._new_block(key)
                new.append(key)
            bslots.append(s)
        bslots = torch.tensor(bslots, dtype=torch.int64, device=DEV)
        g = blocks[:, None, :] * V + self.local[None]                         # (B, V3, 3)
        p = (g.to(torch.float32) + 0.5) * vs
        R, t = T[:3, :3], T[:3, 3]
        pc = (p - t) @ R                                                       # sensor_T_world * p
        vr = torch.linalg.norm(pc, dim=-1)
        max_img = float(rng[rng > 0].max()) if (rng > 0).any() else 0.0
        ok = (vr >= cfg.min_range) & (vr <= cfg.max_range) & (vr <= max_img) & (pc[..., 2] > 0)
        zc = torch.where(pc[..., 2] > 0, pc[..., 2], torch.ones_like(vr))
        u = pc[..., 0] * K.fx / zc + K.cx - K.offset
        v = pc[..., 1] * K.fy / zc + K.cy - K.offset
        ok &= (u >= 0) & (u <= W) & (v >= 0) & (v <= H)
        # InterpolatorAdaptive
        u0, v0 = torch.floor(u).to(torch.int64), torch.floor(v).to(torch.int64)
        rmin = torch.full_like(vr, float("inf"))
        rmax = torch.full_like(vr, -float("inf"))
        corners = []
        for du, dv in ((0, 0), (0, 1), (1, 0), (1, 1)):                   # u_offset_, v_offset_
            cu, cv = u0 + du, v0 + dv
            inside = (cu >= 0) & (cv >= 0) & (cu < W) & (cv < H)
            rr = rng[cv.clamp(0, H - 1), cu.clamp(0, W - 1)]
            corners.append((cu, cv, rr))
            rmin = torch.where(inside, torch.minimum(rmin, rr), rmin)
            rmax = torch.where(inside, torch.maximum(rmax, rr), rmax)
        nearest = (rmax - rmin) > cfg.max_depth_difference
        un, vn = torch.floor(u + 0.5).to(torch.int64), torch.floor(v + 0.5).to(torch.int64)   # std::round, u >= 0
        ok_n = (un >= 0) & (un < W) & (vn >= 0) & (vn < H)
        ok_b = (u0 >= 0) & (v0 >= 0) & (u0 < W - 1) & (v0 < H - 1)
        ok &= torch.where(nearest, ok_n, ok_b)
        du_, dv_ = u - u0.to(torch.float32), v - v0.to(torch.float32)
        w = [(1 - du_) * (1 - dv_), (1 - du_) * dv_, du_ * (1 - dv_), du_ * dv_]
        d_b = sum(wi * c[2] for wi, c in zip(w, corners))
        d_n = rng[vn.clamp(0, H - 1), un.clamp(0, W - 1)]
        d_s = torch.where(nearest, d_n, d_b)
        ok &= torch.isfinite(d_s) & (d_s > 0)
        sdf = torch.clamp(d_s - vr, max=trunc)
        ok &= sdf >= -trunc
        inside_band = sdf.abs() < trunc
        m_n = mask[vn.clamp(0, H - 1), un.clamp(0, W - 1)]
        m_b = sum(wi * mask[c[1].clamp(0, H - 1), c[0].clamp(0, W - 1)].to(torch.float32)
                  for wi, c in zip(w, corners))
        masked = torch.where(nearest, m_n, m_b >= 0.5)
        ok &= ~(inside_band & masked)
        weight = (K.fx * K.fy * (vs / zc) ** 2) / zc ** 2              # computeRayDensity / depth^2
        eps = cfg.weight_dropoff_epsilon if cfg.weight_dropoff_epsilon > 0 else -cfg.weight_dropoff_epsilon * vs
        weight = torch.where(sdf < -eps, weight * (trunc + sdf) / (trunc - eps), weight)
        weight = torch.clamp(weight, min=cfg.min_measurement_weight)
        sl = bslots[:, None].expand_as(vr)
        lin = torch.arange(self.V3, device=DEV)[None].expand_as(vr)
        s_ok, l_ok = sl[ok], lin[ok]
        pw, mw, ms = self.weight[s_ok, l_ok], weight[ok], sdf[ok]
        self.dist[s_ok, l_ok] = (self.dist[s_ok, l_ok] * pw + ms * mw) / (pw + mw)
        self.weight[s_ok, l_ok] = torch.clamp(pw + mw, max=cfg.max_weight)
        self.last_obs[s_ok, l_ok] = stamp
        updated_rows = ok.any(dim=1)
        updated = bslots[updated_rows]
        for key, row in zip(map(tuple, blocks.tolist()), updated_rows.tolist()):
            if not row and key in new:
                self._remove_block(key)
        self._update_tracking(stamp, updated)

    def _update_tracking(self, stamp: int, updated_slots: torch.Tensor) -> None:
        cfg = self.cfg
        if not self.slots:
            return
        alive = torch.tensor(list(self.slots.values()), dtype=torch.int64, device=DEV)
        thr = -cfg.occupancy_threshold * cfg.voxel_size if cfg.occupancy_threshold < 0 else cfg.occupancy_threshold
        occ = self.dist[alive] < thr
        lo = self.last_occ[alive]
        self.last_occ[alive] = torch.where(occ, torch.full_like(lo, stamp), lo)
        t = stamp * 1e-9
        # updateTrackingDuration: active while observed within the temporal window; a voxel that leaves
        # it is to be removed
        active = _sec(self.last_obs[alive]) >= t - cfg.temporal_window
        self.to_remove[alive] |= self.active[alive] & ~active
        self.active[alive] = active
        # ever-free for the blocks updated by this frame (neighbours read the state before this pass)
        if len(updated_slots):
            us = torch.unique(updated_slots)
            free = self._is_free(us, t)
            cand = free & ~self.ever_free[us]
            if cand.any():
                rows, cols = torch.nonzero(cand, as_tuple=True)
                g = self.block_key[us[rows]] * cfg.voxels_per_side + self.local[cols]
                ok = torch.ones(len(g), dtype=torch.bool, device=DEV)
                for o in _N18.to(DEV):
                    s2, l2 = self._voxel(g + o)
                    exists = s2 >= 0
                    s2c, l2c = s2.clamp(min=0), l2
                    nfree = (_sec(self.last_occ[s2c, l2c]) < t - cfg.temporal_buffer) & (self.last_obs[s2c, l2c] != 0)
                    ok &= exists & (self.ever_free[s2c, l2c] | nfree)
                self.ever_free[us[rows[ok]], cols[ok]] = True
        # resetInactive: blocks without active voxels, or whose voxels are all to be removed
        has_active = active.any(dim=1)
        all_removed = self.to_remove[alive].all(dim=1)
        drop = (~has_active) | all_removed
        if drop.any():
            inv = {s: k for k, s in self.slots.items()}
            for s in alive[drop].tolist():
                self._remove_block(inv[s])

    def _is_free(self, slots: torch.Tensor, t: float) -> torch.Tensor:
        return (_sec(self.last_occ[slots]) < t - self.cfg.temporal_buffer) & (self.last_obs[slots] != 0)
