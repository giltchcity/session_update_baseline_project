"""The contract between the update layer and a mapping backend.

A backend is an existing mapper that turns posed RGB-D frames into geometry (a TSDF, points,
surfels, 3D Gaussians, occupancy cells, ...). The update layer never looks inside it; everything
goes through the calls below.

Elements are the smallest pieces of the backend's map that can be kept or retired on their own
(a point, a surfel, a Gaussian, an occupied cell, ...). Each element has
  id            a stable integer id while the element lives (a retired id is never reused)
  xyz           a surface position in the common world frame, metres
  normal        a unit surface normal, NaN when the backend has none (sign may be arbitrary)
  identity      the physical instance id it belongs to, 0 = background
  last_update   the stamp (ns) of the last measurement that updated it (its latest support)
  extent        how far the element's geometry reaches from xyz, metres (0: a point; a surfel's
                radius, a Gaussian's 3 sigma, half a cell's diagonal). Every "does the measured
                surface lie at this element" test of the layer allows sensor tolerance + extent.

Who does what
  backend  fuses frames the way it was published; owns ALL geometry, also across sessions
           (end_session() returns its state, start_session() takes it back); executes retire();
           answers elements(); records snapshot()s and turns them into surface samples for
           evaluation (timeline()).
  layer    reads the same frames for its evidence (depth, pose, instance masks); keeps the object
           states and the per-element evidence; decides which element ids to retire.

Rows of the comparison (the same four for every backend)
  1 scratch  every session starts from an empty map; backend as published (own update on); no layer
  2 naive    the previous session's map is carried over; own update off; no layer
  3 own      the previous map is carried over; own update on (the backend's published change
             handling); no layer
  4 layer    the previous map is carried over; own update off; the update layer decides
  5 own+layer  the previous map is carried over; own update on AND the update layer (since 2026-10-01).
             t2 is this case: TSDF fusion carves what a frame sees through on its own (it cannot be
             switched off) and the layer adds hidden / cross-session / object-level changes. Row 4
             isolates the layer; on the synthetic set it misses exactly the visible changes t2 gets
             (mixed A 3/0/1 vs t2 4/0/0, mixed B 1/0/1 vs 2/0/0: the old site is not seen again, or
             the change is 3 s before the deadline), which per-frame own updates handle.

Frames: integrate(frame) is called for every frame the layer reads (the layer's rate). A backend
may integrate at its own rate and resolution in between (it loads what it needs through
frame.index); on return its map must include everything up to frame.stamp_ns. Pixels of dynamic
classes (people) are never integrated by any backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from .frames import Frame, SessionSpec

ROWS = {1: "scratch", 2: "naive", 3: "own", 4: "layer", 5: "own+layer"}


@dataclass
class Elements:
    ids: torch.Tensor           # (N,) int64
    xyz: torch.Tensor           # (N, 3) float32, metres, world frame
    normal: torch.Tensor        # (N, 3) float32, NaN when unknown
    identity: torch.Tensor      # (N,) int64, 0 = background
    last_update: torch.Tensor   # (N,) int64 ns
    extent: Optional[torch.Tensor] = None   # (N,) float32 metres; None = points (0)

    def __post_init__(self):
        if self.extent is None:
            self.extent = torch.zeros(len(self.ids), dtype=torch.float32, device=self.ids.device)

    def __len__(self) -> int:
        return int(self.ids.numel())

    def select(self, mask: torch.Tensor) -> "Elements":
        return Elements(self.ids[mask], self.xyz[mask], self.normal[mask], self.identity[mask],
                        self.last_update[mask], self.extent[mask])


@dataclass
class DatasetInfo:
    """What every backend needs to know about the data, identical for all of them."""
    name: str                            # "synthetic" | "real"
    dynamic_semantics: Sequence[int]     # classes never integrated (people)
    depth_range: tuple                   # (min, max) metres, the reference mapper's sensor range
    sessions: Sequence[SessionSpec] = () # every session of the chain, in order


class Backend:
    """Base class: every method must be implemented by a backend adapter."""

    name = "backend"
    # Everything that behaves differently from the published method, one short label each (the
    # full account is at the top of the adapter's module). run.py writes them into every run.json.
    CHANGES: tuple = ()

    def __init__(self, info: DatasetInfo, own_update: bool, work_dir: Optional[Path] = None):
        self.info = info
        self.own_update = own_update
        self.work_dir = work_dir                # where the backend may keep its own logs

    def start_session(self, spec: SessionSpec, prior: Optional[Any]) -> None:
        """Begin a session. prior = what end_session() returned last time, or None (empty map)."""
        raise NotImplementedError

    def integrate(self, frame: Frame) -> None:
        """Bring the map up to frame.stamp_ns (see the module notes on rates)."""
        raise NotImplementedError

    def elements(self) -> Elements:
        """All live elements (background and objects)."""
        raise NotImplementedError

    def retire(self, ids: torch.Tensor, stamp: int) -> None:
        """Remove these elements from the map; they must not come back on their own."""
        raise NotImplementedError

    # Optional, for representations whose map is estimated from stored observations (3DGS keyframes):
    # the object states' time intervals instead of per-element object retirements. Each element then
    # belongs to the state of its identity that was open when the element was created, and the map at
    # time t holds the elements whose state is alive at t; every stored observation supervises the map
    # of its own time. False = the layer retires the elements of ended states (object support).
    consumes_state_intervals = False

    def set_state_intervals(self, intervals: dict, stamp: int) -> None:
        """intervals: identity -> [(birth_ns, death_ns or None), ...] of all its states (layer.state_intervals)."""

    def snapshot(self, stamp: int) -> None:
        """Record the map as it is now (called at every round boundary, after retire(), and once
        at the session start for the inherited map)."""
        raise NotImplementedError

    def timeline(self):
        """This session's snapshots: an object with stamps() and scene(t) = the EvaluationScene
        (update_layer.eval.scene) of the latest snapshot at or before t, i.e. background surface
        samples plus one object per identity. It is pickled and read by the scoring environments,
        so it holds numpy data only (update_layer.eval.scenelist). How the map becomes surface
        samples is the backend's converter."""
        raise NotImplementedError

    def end_session(self) -> Any:
        """Finish the session and return the state the next session starts from."""
        raise NotImplementedError

    # Optional checkpointing (run.py --resume): the prior as a torch.save-able dict and back, so a
    # chain can restart after its last finished session. None = not supported.
    def prior_state(self, prior: Any) -> Optional[dict]:
        return None

    def prior_from_state(self, state: dict) -> Any:
        raise NotImplementedError
