"""Object front end for a predicted segmentation: Khronos' detection and tracking for input without physical ids.

With an annotated segmentation the frame's instance image already names the physical object (Khronos
InstanceForwarding + ExternalTracker, core/d1.py). A segmentation network names its segments per frame only, so
before the backend and the layer see a frame its segments are tracked into physical ids the way Khronos does it
(Khronos @192c1cf, the user's fork, ports/mapping_core/khronos):
  1. InstanceForwarding (object_detection/instance_forwarding.cpp): one cluster per segment of an object class,
     pixels with a measured range inside the sensor range, clusters below min_cluster_size pixels dropped;
  2. MaxIoUTracker (tracking/max_iou_tracker.cpp), track_by voxels, semantic_association assign_cluster: every
     track still in the temporal window takes, greedily in track order, the cluster of its own class whose
     voxels overlap the track's last voxels with the highest IoU above min_semantic_iou; the track keeps the
     voxels of its last observation. Parameters of Khronos' uHumans2 config (khronos_ros/config/mapper/
     uHumans2.yaml:61-77): min_cluster_size 50 px, voxel_size 0.2 m, min_semantic_iou 0.25, temporal window 3 s.
  3. A cluster no active track takes is a new track in Khronos; the active window then forgets a track 3 s after
     its last observation, and Khronos' backend associates the objects it extracted later (same class, same place).
     Here that association is done on the tracks: an unclaimed cluster joins the track of its class that holds
     most of its voxels (all voxels the track was ever observed in, share of the cluster's voxels >= 0.5), so an
     object seen again later, or in the next session, keeps its physical id; otherwise it starts a new track.
  4. A segmentation network names regions of its own frame only (a class-agnostic one such as SAM's: a region may be a
     whole object, a part of one or several, and its number means nothing in the next frame), so, as GaME uses SAM's
     masks, a region is read in its own frame only. The front keeps records: per record its surface, as the mean point
     of its measurements in each place (voxels of place_voxel; records share a place without displacing each other).
     A pixel whose measurement lies on a record's surface (the nearest surface point in its place or a face-adjacent
     one, within the layer's surface tolerance) is that record, however the region cut it or numbered it. A record
     still at its place goes on; where it is gone the place shows what is behind it, never the record. New surface
     (on no record) is organised by its region: a region on no record is a new record (a new place, a new state; who
     it was before is not asked); a region on one record of this session is more of it; a region beside one record
     of an earlier session is a record of this session (an earlier session's record is what it was at its places:
     its surface is the layer's object, front.hold, and this session never adds to it); a region over several
     records or background leaves its new surface to a later frame whose region shows one.
The output replaces the frame's instance image (track id + 1, 0 = none); the semantic image is the network's.
Moving objects: none in the datasets this runs on (no dynamic classes); MaxIoUTracker's dynamic tracks are not
ported.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .frames import Frame, backproject


@dataclass
class FrontConfig:
    object_semantics: Sequence[int]
    max_range: float
    min_cluster_size: int = 50          # pixels at full resolution
    voxel_size: float = 0.2             # m
    min_semantic_iou: float = 0.25
    temporal_window_s: float = 3.0
    min_reid_share: float = 0.5
    place_voxel: float = 0.0            # 4. above (0: off): identities kept by place, voxels of this size (m)
    surface_tolerance: float = 0.05     # 4.: on a kept surface along the ray (EvidenceConfig.surface_match_tolerance)


@dataclass
class Track:
    id: int
    semantic: int
    last_seen: int
    last_voxels: set
    all_voxels: set = field(default_factory=set)
    observations: int = 0


BACKGROUND_PLACE = -2      # [4.] a place an earlier session saw but whose segment the layer kept as no object


def _place_keys(pts: np.ndarray, voxel: float) -> np.ndarray:
    k = np.floor(np.asarray(pts, np.float64) / voxel).astype(np.int64) + (1 << 20)
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


_FACES = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
_FACE_OFFSETS = [(dx << 42) + (dy << 21) + dz for dx, dy, dz in _FACES]


class KhronosObjectFront:
    def __init__(self, cfg: FrontConfig):
        self.cfg = cfg
        self.objects = set(int(c) for c in cfg.object_semantics)
        self.tracks: List[Track] = []
        self.cells: Dict[int, Dict[int, list]] = {}   # [4.] place -> record (track index | BACKGROUND_PLACE) -> [sum xyz, n]
        self.log: Dict[str, int] = dict(frames=0, clusters=0, tracked=0, reidentified=0, new=0)
        self.session_first = 0             # the first track of the current session (begin_session)

    def begin_session(self) -> None:
        """A new session (visit) begins: the records made so far are earlier sessions' (4.)."""
        self.session_first = len(self.tracks)

    def hold(self, objects: Dict[int, np.ndarray]) -> None:
        """4.: an earlier session's record is its object as the layer keeps it (identity -> world points); a surface of an
        earlier session that the layer keeps as no object (no reconstruction, a flat patch, a reconstruction beyond the
        object volume) is background, as an annotated segmentation's stuff is. This session's records keep their own."""
        if self.cfg.place_voxel <= 0:
            return
        kept = {i - 1 for i in objects if 0 <= i - 1 < self.session_first}
        for d in self.cells.values():                         # an earlier record the layer no longer keeps: background
            for h in [h for h in d if 0 <= h < self.session_first]:
                sx, sy, sz, n = d.pop(h)
                if h not in kept:
                    b = d.setdefault(BACKGROUND_PLACE, [0.0, 0.0, 0.0, 0])
                    b[0] += sx; b[1] += sy; b[2] += sz; b[3] += n
        for i, pts in sorted(objects.items()):
            t_idx = i - 1                                     # the front's output is track id + 1
            if not 0 <= t_idx < self.session_first or not len(pts):
                continue
            self._add(_place_keys(pts, self.cfg.place_voxel), np.asarray(pts, np.float64), t_idx)

    def _add(self, keys: np.ndarray, pts: np.ndarray, holder: int) -> None:
        """Measurements of a record's surface: the mean point per place."""
        uk, inv = np.unique(keys, return_inverse=True)
        n = np.bincount(inv)
        sm = np.stack([np.bincount(inv, pts[:, j]) for j in range(3)], 1)
        for key, c, (x, y, z) in zip(uk.tolist(), n.tolist(), sm.tolist()):
            e = self.cells.setdefault(key, {}).setdefault(holder, [0.0, 0.0, 0.0, 0])
            e[0] += x; e[1] += y; e[2] += z; e[3] += c

    def state(self) -> dict:
        return dict(tracks=[(t.id, t.semantic, t.last_seen, sorted(t.last_voxels), sorted(t.all_voxels), t.observations)
                            for t in self.tracks], cells={k: dict(d) for k, d in self.cells.items()})

    def load(self, state: dict) -> None:
        self.tracks = [Track(i, sem, last, set(map(tuple, lv)), set(map(tuple, av)), n)
                       for i, sem, last, lv, av, n in state["tracks"]]
        self.cells = {int(k): {int(h): list(e) for h, e in d.items()} for k, d in state.get("cells", {}).items()}
        self.session_first = len(self.tracks)

    def _clusters(self, frame: Frame) -> List[dict]:
        """InstanceForwarding: one cluster per segment of an object class, in-range pixels, >= min_cluster_size."""
        inst, sem = frame.instance, frame.semantic
        in_range = np.isfinite(frame.depth) & (frame.depth <= self.cfg.max_range)
        out = []
        for i in np.unique(inst[(inst > 0) & in_range]):
            m = (inst == i) & in_range
            if m.sum() < self.cfg.min_cluster_size and self.cfg.place_voxel <= 0:
                continue
            labels, counts = np.unique(sem[m], return_counts=True)
            c = int(labels[np.argmax(counts)])
            if c not in self.objects:
                continue
            pts, _, _ = backproject(frame, m)
            vox = set(map(tuple, np.floor(pts / self.cfg.voxel_size).astype(np.int64).tolist()))
            out.append(dict(segment=int(i), semantic=c, mask=m, voxels=vox))
        return out

    def _new_record(self, semantic: int, stamp: int) -> int:
        self.tracks.append(Track(len(self.tracks), semantic, stamp, set()))
        self.log["new"] += 1
        return len(self.tracks) - 1

    def _measure(self, pts: np.ndarray) -> np.ndarray:
        """4.: per point, the record whose surface point in its place or a face-adjacent one is nearest within the
        surface tolerance (track index or BACKGROUND_PLACE), -1 where none is."""
        out = np.full(len(pts), -1, np.int64)
        if not self.cells or not len(pts):
            return out
        ek, eh, em = [], [], []
        for key, d in self.cells.items():
            for h, (x, y, z, n) in sorted(d.items()):          # background first: a record wins a tie
                ek.append(key); eh.append(h); em.append((x / n, y / n, z / n))
        ek, eh, em = np.asarray(ek, np.int64), np.asarray(eh, np.int64), np.asarray(em, np.float64)
        order = np.argsort(ek, kind="stable")
        ek, eh, em = ek[order], eh[order], em[order]
        uk, start, count = np.unique(ek, return_index=True, return_counts=True)
        keys = _place_keys(pts, self.cfg.place_voxel)
        best = np.full(len(pts), self.cfg.surface_tolerance)
        for off in _FACE_OFFSETS:
            q = keys + off
            pos = np.clip(np.searchsorted(uk, q), 0, len(uk) - 1)
            hit = uk[pos] == q
            for j in range(int(count.max())):
                sel = hit & (count[pos] > j)
                if not sel.any():
                    break
                e = start[pos[sel]] + j
                d = np.linalg.norm(pts[sel] - em[e], axis=1)
                better = d <= best[sel]
                idx = np.nonzero(sel)[0][better]
                best[idx] = d[better]
                out[idx] = eh[e[better]]
        return out

    def _associate(self, clusters: List[dict], stamp: int) -> Dict[int, int]:
        """2-3: Khronos' tracking by voxel IoU within the temporal window, then re-identification by place, else new."""
        cfg = self.cfg
        claimed: Dict[int, int] = {}                              # cluster index -> track index
        window = stamp - int(cfg.temporal_window_s * 1e9)
        first = self.session_first if cfg.place_voxel > 0 else 0  # 4.: an earlier session's identity is not extended
        for t_idx, t in enumerate(self.tracks):                   # assignClustersToStaticTrack
            if t.last_seen < window or t_idx < first:
                continue
            best, best_iou = None, cfg.min_semantic_iou
            for c_idx, c in enumerate(clusters):
                if c_idx in claimed or c["semantic"] != t.semantic:
                    continue
                inter = len(c["voxels"] & t.last_voxels)
                iou = inter / (len(c["voxels"]) + len(t.last_voxels) - inter)
                if iou > best_iou:
                    best, best_iou = c_idx, iou
            if best is not None:
                claimed[best] = t_idx
                self.log["tracked"] += 1
        for c_idx, c in enumerate(clusters):                      # same class, same place: the object's track
            if c_idx in claimed:
                continue
            best, best_share = None, cfg.min_reid_share
            for t_idx, t in enumerate(self.tracks):
                if t.semantic != c["semantic"] or t_idx < first:
                    continue
                share = len(c["voxels"] & t.all_voxels) / len(c["voxels"])
                if share >= best_share:
                    best, best_share = t_idx, share
            if best is None:
                self.tracks.append(Track(len(self.tracks), c["semantic"], stamp, set()))
                best = len(self.tracks) - 1
                self.log["new"] += 1
            else:
                self.log["reidentified"] += 1
            claimed[c_idx] = best
        return claimed

    def _process_by_place(self, frame: Frame, clusters: List[dict]) -> np.ndarray:
        """4.: the frame's record image: a pixel on a record's surface is that record; new surface by its region."""
        cfg, stamp = self.cfg, frame.stamp_ns
        out = np.zeros(frame.instance.shape, np.int32)
        valid = np.isfinite(frame.depth) & (frame.depth <= cfg.max_range)
        P, V, U = backproject(frame, valid)
        meas = np.full(frame.depth.shape, -1, np.int64)
        meas[V, U] = self._measure(P)
        semantic_of = np.array([t.semantic for t in self.tracks] + [0], np.int64)
        point = np.full(frame.depth.shape + (3,), np.nan)
        point[V, U] = P
        seen: Dict[int, list] = {}
        out[V, U] = np.where(meas[V, U] >= 0, meas[V, U] + 1, 0)  # a pixel on a record's surface is that record
        for c in clusters:
            m = c["mask"] & valid
            mv = meas[m]
            mv = np.where((mv >= 0) & (semantic_of[np.maximum(mv, 0)] != c["semantic"]), -1, mv)
            new = m & (meas == -1)
            told = sorted(set(np.unique(mv[mv != -1]).tolist()))
            rec = None
            if new.sum() and not told:                         # a region on no record's surface: a new record
                if new.sum() >= cfg.min_cluster_size:
                    rec = self._new_record(c["semantic"], stamp)
            elif new.sum() and len(told) == 1 and told[0] >= self.session_first:
                rec = told[0]                                 # more of this session's record its region shows
            elif new.sum() and len(told) == 1 and told[0] >= 0:  # beside an earlier session's record: this session's
                if new.sum() >= cfg.min_cluster_size:
                    rec = self._new_record(c["semantic"], stamp)
            # several records, or background, in the region: its new surface waits for a region of one
            if rec is not None:
                out[new] = rec + 1
                self._add(_place_keys(point[new], cfg.place_voxel), point[new], rec)
                out[c["mask"] & ~valid] = rec + 1
            for t in np.unique(out[m]).tolist():
                if t > 0:
                    seen.setdefault(t - 1, []).append(point[m & (out == t)])
        on = (out > 0) & valid & (meas >= self.session_first) & (meas == out - 1)   # this session's records keep
        for r in np.unique(meas[on]).tolist():                                         # what they measure
            mm = on & (meas == r)
            self._add(_place_keys(point[mm], cfg.place_voxel), point[mm], r)
        for t_idx, pl in seen.items():
            t = self.tracks[t_idx]
            vox = set(map(tuple, np.floor(np.concatenate(pl) / cfg.voxel_size).astype(np.int64).tolist()))
            t.last_seen, t.last_voxels = stamp, vox
            t.observations += 1
        self.log["frames"] += 1
        self.log["clusters"] += len(clusters)
        self._last_measure = meas
        return out

    def process(self, frame: Frame) -> np.ndarray:
        """The frame's physical-id image (track id + 1). Call once per frame, in time order."""
        stamp = frame.stamp_ns
        clusters = self._clusters(frame)
        if self.cfg.place_voxel > 0:
            out = self._process_by_place(frame, clusters)
            dump = os.environ.get("FRONT_DUMP")                    # inspection: the records, what each pixel measures
            if dump and frame.index % 25 == 0:
                np.savez_compressed(os.path.join(dump, f"{frame.stamp_ns}.npz"), out=out, measure=self._last_measure,
                                    segment=frame.instance, depth=frame.depth)
            return out
        claimed = self._associate(clusters, stamp)
        out = np.zeros(frame.instance.shape, np.int32)
        for c_idx, t_idx in claimed.items():
            c, t = clusters[c_idx], self.tracks[t_idx]
            t.last_seen, t.last_voxels = stamp, c["voxels"]       # updateTrack: the last observation's voxels
            t.all_voxels |= c["voxels"]
            t.observations += 1
            out[c["mask"]] = t.id + 1
        self.log["frames"] += 1
        self.log["clusters"] += len(clusters)
        return out
