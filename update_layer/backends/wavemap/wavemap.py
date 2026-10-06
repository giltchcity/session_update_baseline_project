"""wavemap (Reijgwart et al., RSS 2023; ETH ASL) as a backend.

=====================================================================================================
CHANGES TO PUBLISHED wavemap (pywavemap 2.2.1, FT/baselines/wavemap @db4f2ff): no code changes.
  [W1] rows 2 and 4 (own update off): measurement model scaling_free = 1e-6 instead of 0.2
       (its configuration requires > 0), so a cell that is seen through is no longer lowered.
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

Own update: wavemap's change handling is its free-space evidence. Off (rows 2 and 4): free scaling
1e-6 (its configuration requires > 0), so a cell that is seen through is no longer lowered.

Elements: the occupied cells (log-odds > 0) among the cells that contain a measured surface point,
i.e. the map's surface (cells behind it that the beam model also occupies are not reported);
id = cell index (16 bits per axis) + a generation that grows each time the cell is retired;
xyz = cell centre; extent = half the cell's diagonal; no normal; identity and semantic label =
the majority of the latest frame's points in the cell; last_update = that frame's stamp. retire() sets the cell's log-odds to the
map's minimum (free); it becomes occupied again only through new measurements, with a new id.
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
    """The wavemap map, its pipeline and the surface-cell table (carried across sessions)."""

    def __init__(self, cell: float):
        self.map = wm.Map.create({"type": "hashed_chunked_wavelet_octree", "min_cell_width": {"meters": cell}})
        self.pipeline = wm.Pipeline(self.map)
        self.pipeline.add_operation({"type": "threshold_map", "once_every": {"seconds": 5.0}})
        self.integrator = False
        # surface cells: sorted packed index, with generation / last frame stamp / labels
        self.keys = np.zeros(0, np.int64)
        self.gen = np.zeros(0, np.int64)
        self.last = np.zeros(0, np.int64)
        self.identity = np.zeros(0, np.int64)
        self.semantic = np.zeros(0, np.int64)


def _pack(idx: np.ndarray) -> np.ndarray:
    b = idx.astype(np.int64) + BIAS
    return (b[:, 0] << 32) | (b[:, 1] << 16) | b[:, 2]


def _unpack(keys: np.ndarray) -> np.ndarray:
    return np.stack([(keys >> 32) & 0xFFFF, (keys >> 16) & 0xFFFF, keys & 0xFFFF], 1) - BIAS


def _majority(cells: np.ndarray, values: np.ndarray):
    """Per unique cell (sorted): the most frequent value."""
    order = np.lexsort((values, cells))
    c, v = cells[order], values[order]
    run = np.flatnonzero(np.r_[True, (c[1:] != c[:-1]) | (v[1:] != v[:-1])])
    counts = np.diff(np.r_[run, len(c)])
    rc, rv = c[run], v[run]
    best = np.lexsort((-counts, rc))                       # per cell, the largest run first
    first = np.flatnonzero(np.r_[True, rc[best][1:] != rc[best][:-1]])
    return rc[best][first], rv[best][first]


class WavemapBackend(Backend):
    name = "wavemap"
    CHANGES = ("W1 own update off = scaling_free 1e-6 (rows 2 and 4)",)

    def __init__(self, info: DatasetInfo, own_update: bool, work_dir=None, cell: float = 0.02):
        super().__init__(info, own_update, work_dir)
        self.cell = cell
        self.state: Optional[_State] = None

    def start_session(self, spec: SessionSpec, prior: Optional[_State]) -> None:
        self.state = prior if prior is not None else _State(self.cell)
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
                                  "scaling_free": 0.2 if self.own_update else 1e-6, "scaling_occupied": 0.4},
            "integration_method": {"type": "hashed_chunked_wavelet_integrator",
                                   "min_range": {"meters": self.info.depth_range[0]},
                                   "max_range": {"meters": self.info.depth_range[1]}}})
        self.state.integrator = True

    def integrate(self, frame: Frame) -> None:
        st = self.state
        if not st.integrator:
            self._add_integrator(frame)
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
        new = ~hit
        if new.any():
            keys = np.concatenate([st.keys, uc[new]])
            o = np.argsort(keys, kind="stable")
            st.keys = keys[o]
            st.gen = np.concatenate([st.gen, np.zeros(int(new.sum()), np.int64)])[o]
            st.last = np.concatenate([st.last, np.full(int(new.sum()), frame.stamp_ns, np.int64)])[o]
            st.identity = np.concatenate([st.identity, ident[new]])[o]
            st.semantic = np.concatenate([st.semantic, label[new]])[o]

    def _occupied(self):
        st = self.state
        if not len(st.keys):
            return np.zeros(0, np.int64)
        values = st.map.get_cell_values(np.ascontiguousarray(_unpack(st.keys).astype(np.int32)))
        return np.flatnonzero(np.asarray(values).reshape(-1) > 0)

    def elements(self) -> Elements:
        st = self.state
        rows = self._occupied()
        ids = (st.gen[rows] << 48) | st.keys[rows]
        xyz = (_unpack(st.keys[rows]) + 0.5) * self.cell
        t = lambda a, d: torch.as_tensor(a, dtype=d, device=DEV)
        return Elements(t(ids, torch.int64), t(xyz, torch.float32),
                        torch.full((len(rows), 3), float("nan"), device=DEV),
                        t(st.identity[rows], torch.int64), t(st.last[rows], torch.int64),
                        torch.full((len(rows),), 0.5 * float(np.sqrt(3.0)) * self.cell, device=DEV))

    def retire(self, ids: torch.Tensor, stamp: int) -> None:
        st = self.state
        ids = ids.cpu().numpy().astype(np.int64)
        keys, gen = ids & MASK48, ids >> 48
        pos = np.minimum(np.searchsorted(st.keys, keys), max(len(st.keys) - 1, 0))
        ok = (st.keys[pos] == keys) & (st.gen[pos] == gen)
        low = st.map.min_log_odds
        cells = _unpack(keys[ok]).astype(np.int32)
        for idx in cells:
            st.map.set_cell_value(np.ascontiguousarray(idx), low)
        st.gen[pos[ok]] += 1
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

    def snapshot(self, stamp: int) -> None:
        st = self.state
        rows = self._occupied()
        ids = (st.gen[rows] << 48) | st.keys[rows]
        xyz = ((_unpack(st.keys[rows]) + 0.5) * self.cell).astype(np.float32)
        self.recorder.record(stamp, ids, xyz, np.full((len(rows), 3), np.nan, np.float32),
                             st.semantic[rows], st.identity[rows])

    def timeline(self) -> ElementLifetimes:
        return self.recorder.timeline()
