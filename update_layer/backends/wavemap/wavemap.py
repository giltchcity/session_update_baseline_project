"""wavemap (Reijgwart et al., RSS 2023; ETH ASL) as a backend.

=====================================================================================================
CHANGES TO PUBLISHED wavemap (pywavemap 2.2.1, FT/baselines/wavemap @db4f2ff): no code changes.
  [W3] rows 2 and 4 (own update off): the previous map is kept apart as the memory, which the integration
       does not touch; each session builds its own map with the published measurement model.
Input: run.py's common input (every frame, real 960x540 / synthetic 680x480), people as invalid depth.
  Map resolution 2 cm cells (since 2026-10-01 20:10; before: 5 cm as in its example, with the layer's
  5 Hz 480x270 / 340x240 frames), the same as the points backend's voxels.
=====================================================================================================

pywavemap 2.2.1 as published (FT/baselines/wavemap @db4f2ff, pip install ./library/python),
configured as its depth-camera example (examples/python/mapping/full_pipeline.py, written for the
Flat dataset family our synthetic set belongs to): hashed chunked wavelet octree (the example's 5 cm
cells; here 2 cm, the common map resolution, see Input above), continuous-ray measurement model (range
sigma 1 cm, occupied scaling 0.4, free scaling 0.2), thresholding every 5 s, the dataset's sensor range.
Frames: run.py's common input; dynamic pixels (people) are not integrated.

Own update: wavemap's change handling is its free-space evidence. Off (rows 2 and 4): the elements of the
previous map are kept apart as the memory, which the integration does not touch and only retire() ends;
each session builds its own map (fresh_present: the session-end test of the TSDF version's refusion).

Elements: the occupied cells (log-odds > 0) among the cells that contain a measured surface point,
i.e. the map's surface (cells behind it that the beam model also occupies are not reported), then the
memory still alive; id = cell index (16 bits per axis) + a generation that grows each time the cell is
retired and at each session that builds its own map; xyz = cell centre; extent = half the cell's diagonal;
no normal; identity and semantic label = the majority of the latest frame's points in the cell; last_update
= that frame's stamp; created = the stamp of the first frame that measured a surface point in the cell in its
current generation (a cell the beam model occupies again after a retirement without a surface point in it:
the latest integrated frame when it is first reported occupied; memory elements keep theirs, so they are the
session's memory for the session-end test).
retire() sets the cell's log-odds to the map's minimum (free) with addToCellValue (the chunked block's
setCellValue subtracts the parent's average, hashed_chunked_wavelet_octree_block.cc:45-58) and ends a memory
element; a cell becomes occupied again only through new measurements, with a new id; its lifetime in the 4D
timeline ends at the retirement stamp (SnapshotRecorder.end: a session-end memory retirement at the session
start ends it before the session's snapshots, as for GaME / points).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pywavemap as wm
import torch

from ...eval.scenelist import ElementLifetimes, SnapshotRecorder
from ...frames import Frame, SessionSpec, backproject, dynamic_mask
from ...interface import Backend, DatasetInfo, Elements

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BIAS = 1 << 15                          # cell index -> 16-bit field
MASK48 = (1 << 48) - 1


class _State:
    """The wavemap map, its pipeline, the surface-cell table and the memory (rows 2 and 4)."""

    def __init__(self, cell: float):
        self.map = wm.Map.create({"type": "hashed_chunked_wavelet_octree", "min_cell_width": {"meters": cell}})
        self.pipeline = wm.Pipeline(self.map)
        self.pipeline.add_operation({"type": "threshold_map", "once_every": {"seconds": 5.0}})
        self.integrator = False
        # surface cells: sorted packed index, with generation / creation and last frame stamp / labels
        self.keys = np.zeros(0, np.int64)
        self.gen = np.zeros(0, np.int64)
        self.born = np.zeros(0, np.int64)       # creation stamp of the current generation; -1 = not yet re-created
        self.last = np.zeros(0, np.int64)
        self.now = 0                            # the latest integrated frame's stamp
        self.identity = np.zeros(0, np.int64)
        self.semantic = np.zeros(0, np.int64)
        # cells new since the table was last read, sorted (key, last, identity, semantic, born): merged into the table
        # once per read instead of at every frame
        self.pending = [np.zeros(0, np.int64) for _ in range(5)]
        self.memory: Optional[dict] = None      # the previous map's elements (sorted ids) and which are alive
        # per cell key (sorted), the lowest generation this map may give it: above every id the key had in earlier
        # sessions, so an id is never reused
        self.floor = (np.zeros(0, np.int64), np.zeros(0, np.int64))


def _pack(idx: np.ndarray) -> np.ndarray:
    b = idx.astype(np.int64) + BIAS
    return (b[:, 0] << 32) | (b[:, 1] << 16) | b[:, 2]


def _unpack(keys: np.ndarray) -> np.ndarray:
    return np.stack([(keys >> 32) & 0xFFFF, (keys >> 16) & 0xFFFF, keys & 0xFFFF], 1) - BIAS


def _majority(cells: np.ndarray, values: np.ndarray):
    """Per unique cell (sorted): the most frequent value (ties: the smallest)."""
    assert len(values) == 0 or (values.min() >= 0 and values.max() < 1 << 16)
    k = np.sort((cells.astype(np.uint64) << np.uint64(16)) | values.astype(np.uint64))
    run = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    counts = np.diff(np.r_[run, len(k)])
    rc, rv = (k[run] >> np.uint64(16)).astype(np.int64), (k[run] & np.uint64(0xFFFF)).astype(np.int64)
    first = np.flatnonzero(np.r_[True, rc[1:] != rc[:-1]])                  # each cell's runs, by value
    seg = np.repeat(np.arange(len(first)), np.diff(np.r_[first, len(rc)]))
    best = np.flatnonzero(counts == np.maximum.reduceat(counts, first)[seg])
    best = best[np.r_[True, seg[best][1:] != seg[best][:-1]]]               # the first (smallest value) per cell
    return rc[best], rv[best]


class WavemapBackend(Backend):
    name = "wavemap"
    CHANGES = ("own update off (rows 2 and 4): the previous map kept apart as the memory",
               "retire: addToCellValue(min - value)")

    def __init__(self, info: DatasetInfo, own_update: bool, work_dir=None, cell: float = 0.02):
        super().__init__(info, own_update, work_dir)
        self.fresh_present = not own_update
        self.cell = cell
        self.state: Optional[_State] = None
        self._rows_now = None                   # _occupied(self.state) until the map changes

    def start_session(self, spec: SessionSpec, prior: Optional[_State]) -> None:
        self._rows_now = None
        if prior is None or self.own_update:
            self.state = prior if prior is not None else _State(self.cell)
        else:
            # the previous map is the memory; this session's map starts empty, its cells with new ids
            ids, xyz, identity, semantic, born, last = self._current(prior)
            st = _State(self.cell)
            keys = np.concatenate([prior.floor[0], prior.keys, ids & MASK48])
            gens = np.concatenate([prior.floor[1], prior.gen + 1, (ids >> 48) + 1])
            o = np.lexsort((-gens, keys))
            keys, gens = keys[o], gens[o]
            first = np.r_[True, keys[1:] != keys[:-1]] if len(keys) else np.zeros(0, bool)
            st.floor = (keys[first], gens[first])
            o = np.argsort(ids)
            st.memory = dict(ids=ids[o], xyz=xyz[o], identity=identity[o], semantic=semantic[o], born=born[o],
                             last=last[o], alive=np.ones(len(ids), bool))
            self.state = st
        self.recorder = SnapshotRecorder()

    def end_session(self) -> _State:
        return self.state

    def _add_integrator(self, frame: Frame) -> None:
        K = frame.K
        h, w = frame.depth.shape
        self.state.pipeline.add_integrator("integrator", {
            "projection_model": {"type": "pinhole_camera_projector", "width": w, "height": h,
                                 "fx": K.fx, "fy": K.fy, "cx": K.cx - K.offset, "cy": K.cy - K.offset},
            "measurement_model": {"type": "continuous_ray", "range_sigma": {"meters": 0.01},
                                  "scaling_free": 0.2, "scaling_occupied": 0.4},
            "integration_method": {"type": "hashed_chunked_wavelet_integrator",
                                   "min_range": {"meters": self.info.depth_range[0]},
                                   "max_range": {"meters": self.info.depth_range[1]}}})
        self.state.integrator = True

    def integrate(self, frame: Frame) -> None:
        st = self.state
        self._rows_now = None
        if not st.integrator:
            self._add_integrator(frame)
        st.now = int(frame.stamp_ns)
        keep = np.isfinite(frame.depth) & ~dynamic_mask(frame, self.info.dynamic_semantics)
        depth = np.where(keep, frame.depth, 0.0).astype(np.float32)
        pose = wm.Pose(np.asfortranarray(frame.T_world_cam.astype(np.float32)))
        st.pipeline.run_pipeline(["integrator"], wm.PosedImage(pose, wm.Image(np.asfortranarray(depth.T))))
        # the cells this frame measured a surface in
        pts, v, u = backproject(frame, keep)
        if not len(pts):
            return
        cells = _pack(np.floor(pts / self.cell).astype(np.int64))
        uc, ident = _majority(cells, frame.instance[v, u].astype(np.int64).clip(min=0))
        sem = frame.semantic[v, u].astype(np.int64) if frame.semantic is not None else np.zeros(len(v), np.int64)
        _, label = _majority(cells, sem)
        pos = np.searchsorted(st.keys, uc)
        pos_c = np.minimum(pos, max(len(st.keys) - 1, 0))
        hit = (st.keys[pos_c] == uc) if len(st.keys) else np.zeros(len(uc), bool)
        rows = pos_c[hit]
        st.last[rows], st.identity[rows], st.semantic[rows] = frame.stamp_ns, ident[hit], label[hit]
        st.born[rows[st.born[rows] < 0]] = frame.stamp_ns
        new = ~hit
        if new.any():
            uc, ident, label = uc[new], ident[new], label[new]
            pk, pl, pi, ps, pb = st.pending
            pp = np.minimum(np.searchsorted(pk, uc), max(len(pk) - 1, 0))
            ph = (pk[pp] == uc) if len(pk) else np.zeros(len(uc), bool)
            pl[pp[ph]], pi[pp[ph]], ps[pp[ph]] = frame.stamp_ns, ident[ph], label[ph]
            if (~ph).any():
                at, nn = np.searchsorted(pk, uc[~ph]), int((~ph).sum())
                st.pending = [np.insert(pk, at, uc[~ph]), np.insert(pl, at, np.full(nn, frame.stamp_ns, np.int64)),
                              np.insert(pi, at, ident[~ph]), np.insert(ps, at, label[~ph]),
                              np.insert(pb, at, np.full(nn, frame.stamp_ns, np.int64))]

    @staticmethod
    def _merge(st: _State) -> None:
        """The pending new cells into the sorted table (generation: the key's floor, 0 for a key never seen)."""
        pk, pl, pi, ps, pb = st.pending
        if not len(pk):
            return
        fk, fg = st.floor
        gen = np.zeros(len(pk), np.int64)
        if len(fk):
            fp = np.minimum(np.searchsorted(fk, pk), len(fk) - 1)
            gen = np.where(fk[fp] == pk, fg[fp], 0)
        at = np.searchsorted(st.keys, pk)
        st.keys, st.gen = np.insert(st.keys, at, pk), np.insert(st.gen, at, gen)
        st.last, st.identity = np.insert(st.last, at, pl), np.insert(st.identity, at, pi)
        st.semantic, st.born = np.insert(st.semantic, at, ps), np.insert(st.born, at, pb)
        st.pending = [np.zeros(0, np.int64) for _ in range(5)]

    @classmethod
    def _occupied(cls, st: _State):
        cls._merge(st)
        if not len(st.keys):
            return np.zeros(0, np.int64)
        values = st.map.get_cell_values(np.ascontiguousarray(_unpack(st.keys).astype(np.int32)))
        rows = np.flatnonzero(np.asarray(values).reshape(-1) > 0)
        pending = rows[st.born[rows] < 0]          # occupied again after a retirement, no surface point yet
        st.born[pending] = st.now
        return rows

    def _current(self, st: _State, rows: Optional[np.ndarray] = None):
        """The map now, as (ids, xyz, identity, semantic, created, last update): this session's occupied surface cells
        (`rows`, read when not given), then the memory still alive."""
        rows = self._occupied(st) if rows is None else rows
        parts = [((st.gen[rows] << 48) | st.keys[rows], (_unpack(st.keys[rows]) + 0.5) * self.cell, st.identity[rows],
                  st.semantic[rows], st.born[rows], st.last[rows])]
        if st.memory is not None:
            m, a = st.memory, st.memory["alive"]
            parts.append((m["ids"][a], m["xyz"][a], m["identity"][a], m["semantic"][a], m["born"][a], m["last"][a]))
        return [np.concatenate(c) for c in zip(*parts)]

    def _now(self):
        """_current(self.state), its cells read once between two changes of the map."""
        if self._rows_now is None:
            self._rows_now = self._occupied(self.state)
        return self._current(self.state, self._rows_now)

    def elements(self) -> Elements:
        ids, xyz, identity, _, born, last = self._now()
        n = len(ids)
        t = lambda a, d: torch.as_tensor(a, dtype=d, device=DEV)
        return Elements(t(ids, torch.int64), t(xyz, torch.float32), torch.full((n, 3), float("nan"), device=DEV),
                        t(identity, torch.int64), t(last, torch.int64),
                        torch.full((n,), 0.5 * float(np.sqrt(3.0)) * self.cell, device=DEV), t(born, torch.int64),
                        torch.full((n,), 0.5 * self.cell, device=DEV))         # h = half a cell edge

    def retire(self, ids: torch.Tensor, stamp: int) -> None:
        st = self.state
        self._merge(st)
        ids = ids.cpu().numpy().astype(np.int64)
        if st.memory is not None and len(st.memory["ids"]):
            # a memory element ends (a cell of this session's map never has a memory element's id)
            m = st.memory
            pos = np.minimum(np.searchsorted(m["ids"], ids), len(m["ids"]) - 1)
            hit = (m["ids"][pos] == ids) & m["alive"][pos]
            m["alive"][pos[hit]] = False
            self.recorder.end(ids[hit], stamp)
            ids = ids[~hit]
        keys, gen = ids & MASK48, ids >> 48
        pos = np.minimum(np.searchsorted(st.keys, keys), max(len(st.keys) - 1, 0))
        ok = (st.keys[pos] == keys) & (st.gen[pos] == gen)
        low = st.map.min_log_odds
        cells = _unpack(keys[ok]).astype(np.int32)
        values = np.asarray(st.map.get_cell_values(np.ascontiguousarray(cells))).reshape(-1) if len(cells) else []
        for idx, v in zip(cells, values):
            st.map.add_to_cell_value(np.ascontiguousarray(idx), float(low - v))
        st.gen[pos[ok]] += 1
        st.born[pos[ok]] = -1
        self.recorder.end(ids[ok], stamp)
        # read-back (2026-10-06): a retired cell must not be occupied any more. A minimal pywavemap test
        # (plane, 5/200 frames, every other / half of the cells retired, with and without threshold_map)
        # read back 0% occupied; the earlier 25-45% "still occupied" of the old runs is checked here.
        if len(cells):
            back = np.asarray(st.map.get_cell_values(np.ascontiguousarray(cells))).reshape(-1)
            n_bad = int((back > 0).sum())
            self.retire_readback = getattr(self, "retire_readback", [0, 0, 0])
            self.retire_readback[0] += int(len(ids))
            self.retire_readback[1] += int(len(cells))
            self.retire_readback[2] += n_bad
            if n_bad:
                print(f"wavemap retire read-back: {n_bad} of {len(cells)} cells still occupied", flush=True)
        stale = int(len(ids) - ok.sum())
        if stale:
            print(f"wavemap retire: {stale} ids of an older generation (cell re-occupied since) skipped", flush=True)
        if self._rows_now is not None and len(cells):
            # addToCellValue changes only the octree block of the cell (HashedChunkedWaveletOctree::addToCellValue): the
            # cells of the touched blocks are read again, the others keep their reading
            side = 1 << st.map.tree_height
            touched = np.unique(_pack(np.floor_divide(cells.astype(np.int64), side)))
            member = np.isin(_pack(np.floor_divide(_unpack(st.keys), side)), touched)
            again = np.flatnonzero(member)
            v = np.asarray(st.map.get_cell_values(np.ascontiguousarray(_unpack(st.keys[again]).astype(np.int32)))).reshape(-1)
            occ = again[v > 0]
            st.born[occ[st.born[occ] < 0]] = st.now
            self._rows_now = np.union1d(self._rows_now[~member[self._rows_now]], occ)

    def snapshot(self, stamp: int) -> None:
        ids, xyz, identity, semantic, _, _ = self._now()
        self.recorder.record(stamp, ids, xyz.astype(np.float32), np.full((len(ids), 3), np.nan, np.float32),
                             semantic, identity)

    def timeline(self) -> ElementLifetimes:
        return self.recorder.timeline()
