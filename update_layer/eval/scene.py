"""EvaluationScene: the representation-independent view of a map that every metric reads.

A map at query time t is reduced to world-space surface samples:
  background        samples of everything not attributed to an object, with a semantic label
  objects           one entry per displayed object state: identity, class, samples, presence

Presence follows Khronos' isPresent (khronos_attribute_utils.cpp:294): present at t when the
last appearance (first_observed_ns) before t is later than the last disappearance
(last_observed_ns) before t. A displayed object may be non-present (Khronos keeps such
nodes); the harness counts their surfaces as shown (export_cleanup_binary, cleanup_mesh).

How a representation produces samples is its converter's business (TSDF mesh: area samples;
points/surfels: the elements; 3DGS: depth rendered from fixed evaluation cameras and
back-projected). The metrics never see the representation.

ElementTimeline stores a whole session compactly as elements with display lifetimes
[birth, death): the scene at t is the elements alive at t. A backend with an update layer
writes it directly (every retire / create is a lifetime bound).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

INVALID_LABEL = 0xFFFFFFFF
OPEN = np.iinfo(np.int64).max


@dataclass
class SceneObject:
    id: str                         # unique per displayed state (node id in Khronos maps)
    instance_id: int                # physical identity (details.instance_id)
    semantic: int
    points: np.ndarray              # (M, 3) float32 world surface samples
    present: bool
    first_observed_ns: List[int] = field(default_factory=list)
    last_observed_ns: List[int] = field(default_factory=list)
    observation_first_ns: int = 0
    observation_last_ns: int = 0
    trajectory_timestamps: List[int] = field(default_factory=list)
    trajectory_positions: List[List[float]] = field(default_factory=list)
    normals: Optional[np.ndarray] = None        # (M, 3) unit normals, NaN where unknown


@dataclass
class EvaluationScene:
    t_ns: int
    background: np.ndarray          # (N, 3) float32
    background_label: np.ndarray    # (N,) uint32, INVALID_LABEL when unknown
    objects: List[SceneObject]
    background_normal: Optional[np.ndarray] = None   # (N, 3), NaN where unknown


def is_present(first: Sequence[int], last: Sequence[int], t: int) -> bool:
    appeared = [x for x in first if x <= t]
    if not appeared:
        return False
    gone = [x for x in last if x <= t]
    return not gone or max(appeared) > max(gone)


class ElementTimeline:
    """Elements with display lifetimes; states give object identity and presence.

    Arrays (npz):
      stamps          (S,)   snapshot times the map was emitted at (online query grid)
      el_xyz          (E, 3) float32
      el_label        (E,)   uint32 semantic label
      el_state        (E,)   int32 index into the state table, -1 = background
      el_birth/death  (E,)   int64 display interval [birth, death)
      st_instance     (T,)   int64 physical identity of each object state
      st_semantic     (T,)   int64
      st_first/st_last(T,)   int64 presence interval; st_last = OPEN while alive
      st_obs_first/last (T,) int64 observation bounds
    """

    def __init__(self, arrays: Dict[str, np.ndarray], meta: Optional[dict] = None):
        self.a = arrays
        self.meta = meta or {}

    @staticmethod
    def load(path) -> "ElementTimeline":
        path = Path(path)
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return ElementTimeline(arrays, meta)

    def save(self, path) -> None:
        path = Path(path)
        np.savez_compressed(path, **self.a)
        path.with_suffix(".json").write_text(json.dumps(self.meta, indent=2))

    def stamps(self) -> List[int]:
        return [int(s) for s in self.a["stamps"]]

    def scene(self, t_ns: int, background: bool = True) -> EvaluationScene:
        a = self.a
        alive = (a["el_birth"] <= t_ns) & (t_ns < a["el_death"])
        bg = alive & (a["el_state"] < 0)
        objects = []
        obj = alive & (a["el_state"] >= 0)
        states = a["el_state"][obj]
        xyz = a["el_xyz"][obj]
        order = np.argsort(states, kind="stable")
        states, xyz = states[order], xyz[order]
        cuts = np.flatnonzero(np.diff(states)) + 1
        for chunk_states, chunk in zip(np.split(states, cuts), np.split(xyz, cuts)):
            if not len(chunk_states):
                continue
            s = int(chunk_states[0])
            first = [int(a["st_first"][s])]
            last = [] if a["st_last"][s] == OPEN else [int(a["st_last"][s])]
            objects.append(SceneObject(
                id=str(s), instance_id=int(a["st_instance"][s]), semantic=int(a["st_semantic"][s]),
                points=chunk.astype(np.float32), present=is_present(first, last, t_ns),
                first_observed_ns=first, last_observed_ns=last,
                observation_first_ns=int(a["st_obs_first"][s]),
                observation_last_ns=int(a["st_obs_last"][s])))
        if background:
            return EvaluationScene(t_ns, a["el_xyz"][bg].astype(np.float32),
                                   a["el_label"][bg].astype(np.uint32), objects)
        return EvaluationScene(t_ns, np.zeros((0, 3), np.float32), np.zeros(0, np.uint32), objects)
