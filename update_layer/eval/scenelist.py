"""Saved session timelines: stamps() / scene(t) = the map at the latest snapshot at or before t.

  SceneListTimeline  one EvaluationScene per snapshot (maps evaluated by rendering, e.g. 3DGS)
  ElementLifetimes   elements with display lifetimes [birth, death) (points, surfels, cells)

Kept free of heavy imports (numpy only) so the scoring environments can unpickle them.
"""
from __future__ import annotations

import bisect
import pickle
from typing import List

import numpy as np

from .scene import EvaluationScene, SceneObject

UINT64_MAX = 18446744073709551615


class SceneListTimeline:
    """stamps() / scene(t) like LayerTimeline: the latest snapshot at or before t."""

    def __init__(self, stamps: List[int], scenes: List[EvaluationScene]):
        self._stamps, self.scenes = list(stamps), list(scenes)

    def stamps(self) -> List[int]:
        return self._stamps

    def scene(self, t_ns: int, background: bool = True) -> EvaluationScene:
        k = max(0, bisect.bisect_right(self._stamps, t_ns) - 1)
        s = self.scenes[k]
        if background:
            return s
        return EvaluationScene(s.t_ns, np.zeros((0, 3), np.float32), np.zeros(0, np.uint32), s.objects)

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)

    @staticmethod
    def load(path) -> "SceneListTimeline":
        with open(path, "rb") as f:
            return pickle.load(f)


class ElementLifetimes:
    """Every element a session displayed, with its lifetime in snapshots.

    Built from lifetimes [birth, death) in ns; stored per element as the first snapshot that shows
    it and the first that no longer does (elements no snapshot shows are dropped), with float16
    normals. scene(t): the elements shown by the latest snapshot at or before t; identity 0 is
    background, every other identity one object (class = its elements' majority label).
    """

    def __init__(self, stamps: List[int], xyz: np.ndarray, normal: np.ndarray, label: np.ndarray,
                 identity: np.ndarray, birth: np.ndarray, death: np.ndarray):
        self._stamps = [int(x) for x in stamps]
        st = np.asarray(self._stamps, dtype=np.int64)
        first = np.searchsorted(st, np.asarray(birth, dtype=np.int64), side="left")
        end = np.searchsorted(st, np.asarray(death, dtype=np.int64), side="left")
        keep = first < end
        self.xyz = np.ascontiguousarray(np.asarray(xyz, dtype=np.float32)[keep])
        self.normal = np.asarray(normal, dtype=np.float16)[keep]
        self.label = np.asarray(label, dtype=np.uint32)[keep]
        self.identity = np.asarray(identity, dtype=np.int64)[keep].astype(np.uint16)
        self.first = first[keep].astype(np.int32)
        self.end = end[keep].astype(np.int32)

    def __setstate__(self, state):
        if "birth" in state:                    # pickled before snapshot-indexed lifetimes
            self.__init__(state["_stamps"], state["xyz"], state["normal"], state["label"],
                          state["identity"], state["birth"], state["death"])
        else:
            self.__dict__.update(state)

    def stamps(self) -> List[int]:
        return self._stamps

    def scene(self, t_ns: int, background: bool = True) -> EvaluationScene:
        k = max(0, bisect.bisect_right(self._stamps, t_ns) - 1)
        alive = (self.first <= k) & (k < self.end)
        objects = []
        obj = alive & (self.identity > 0)
        for identity in np.unique(self.identity[obj]):
            m = obj & (self.identity == identity)
            first = self._stamps[int(self.first[m].min())]
            objects.append(SceneObject(
                id=str(int(identity)), instance_id=int(identity),
                semantic=int(np.bincount(self.label[m].astype(np.int64)).argmax()),
                points=self.xyz[m], present=True, first_observed_ns=[first],
                last_observed_ns=[UINT64_MAX], observation_first_ns=first,
                observation_last_ns=int(t_ns), normals=self.normal[m].astype(np.float32)))
        if not background:
            return EvaluationScene(t_ns, np.zeros((0, 3), np.float32), np.zeros(0, np.uint32), objects)
        bg = alive & (self.identity == 0)
        return EvaluationScene(t_ns, self.xyz[bg], self.label[bg], objects, self.normal[bg].astype(np.float32))

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)


class SnapshotRecorder:
    """Builds ElementLifetimes from what a backend shows at each snapshot: an element (by id) is
    born at the first snapshot that shows it and dies at the first that no longer does; a change
    of its identity ends it and starts a new lifetime; its other attributes are those of the last
    snapshot that showed it."""

    OPEN = np.iinfo(np.int64).max

    def __init__(self):
        self.stamps: List[int] = []
        self.ids = np.zeros(0, np.int64)            # sorted
        self.row = np.zeros(0, np.int64)            # row of each sorted id in the arrays
        self.a = dict(xyz=np.zeros((0, 3), np.float32), normal=np.zeros((0, 3), np.float32),
                      label=np.zeros(0, np.int64), identity=np.zeros(0, np.int64),
                      birth=np.zeros(0, np.int64), death=np.zeros(0, np.int64))

    def record(self, stamp: int, ids: np.ndarray, xyz: np.ndarray, normal: np.ndarray, label: np.ndarray,
               identity: np.ndarray) -> None:
        self.stamps.append(int(stamp))
        ids = np.asarray(ids, dtype=np.int64)
        attrs = dict(xyz=xyz, normal=normal, label=label, identity=identity)
        a = self.a
        identity = np.asarray(identity, dtype=np.int64)
        relabel = np.zeros(0, np.int64)                 # positions (in self.ids) of relabelled ids
        if len(self.ids):
            pos = np.minimum(np.searchsorted(self.ids, ids), len(self.ids) - 1)
            hit = self.ids[pos] == ids
            rows = self.row[pos[hit]]
            gone = a["death"] == self.OPEN
            gone[rows] = False
            same = a["identity"][rows] == identity[hit]
            gone[rows[~same]] = True                    # identity changed: that lifetime ends
            a["death"][gone] = stamp
            keep = np.flatnonzero(hit)[same]
            for k, v in attrs.items():
                a[k][rows[same]] = np.asarray(v)[keep]
            relabel = pos[hit][~same]
            cont = np.zeros(len(ids), bool)
            cont[keep] = True
        else:
            cont = np.zeros(len(ids), bool)
        new = np.nonzero(~cont)[0]                      # new ids and relabelled ones
        if len(new):
            n_rows = len(a["death"])
            for k, v in attrs.items():
                a[k] = np.concatenate([a[k], np.asarray(v)[new].astype(a[k].dtype)])
            a["birth"] = np.concatenate([a["birth"], np.full(len(new), stamp, np.int64)])
            a["death"] = np.concatenate([a["death"], np.full(len(new), self.OPEN, np.int64)])
            new_rows_all = n_rows + np.arange(len(new))
            if len(relabel):                            # same id, new row
                is_rel = np.zeros(len(ids), bool)
                is_rel[np.flatnonzero(hit)[~same]] = True
                self.row[relabel] = new_rows_all[is_rel[new]]
                fresh = ~is_rel[new]
            else:
                fresh = np.ones(len(new), bool)
            order = np.argsort(ids[new][fresh], kind="stable")
            new_ids, new_rows = ids[new][fresh][order], new_rows_all[fresh][order]
            if len(new_ids) and len(self.ids) and new_ids[0] < self.ids[-1]:
                all_ids = np.concatenate([self.ids, new_ids])
                o = np.argsort(all_ids, kind="stable")
                self.ids, self.row = all_ids[o], np.concatenate([self.row, new_rows])[o]
            elif len(new_ids):
                self.ids = np.concatenate([self.ids, new_ids])
                self.row = np.concatenate([self.row, new_rows])

    def end(self, ids: np.ndarray, stamp: int) -> None:
        """The backend retired these elements at `stamp`: their open lifetimes end then (never before their
        birth), also when the snapshots since showed them (a session-end retirement dated at the session start)."""
        ids = np.asarray(ids, dtype=np.int64)
        if not len(self.ids) or not len(ids):
            return
        pos = np.minimum(np.searchsorted(self.ids, ids), len(self.ids) - 1)
        rows = self.row[pos[self.ids[pos] == ids]]
        a = self.a
        rows = rows[a["death"][rows] == self.OPEN]
        a["death"][rows] = np.maximum(int(stamp), a["birth"][rows])

    def timeline(self) -> ElementLifetimes:
        a = self.a
        return ElementLifetimes(self.stamps, a["xyz"], a["normal"], a["label"], a["identity"], a["birth"],
                                a["death"])


class TimelineTail:
    """A timeline stored as another timeline file's first `keep` snapshots plus its own snapshots (the post_ref
    timeline: the pre_ref snapshots with the final one re-rendered after the refinement). `base` is a file name in
    the directory of the file that holds this object (symlinks resolved). load_timeline returns the full
    SceneListTimeline; the content is identical to storing every snapshot."""

    def __init__(self, base: str, keep: int, stamps: List[int], scenes: list):
        self.base, self.keep, self._stamps, self.scenes = base, int(keep), list(stamps), list(scenes)

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)


def load_timeline(path):
    """Any saved timeline (stamps() / scene(t)); the pickle names its own class."""
    import os
    with open(path, "rb") as f:
        tl = pickle.load(f)
    if isinstance(tl, TimelineTail):
        base = load_timeline(os.path.join(os.path.dirname(os.path.realpath(path)), tl.base))
        return SceneListTimeline(base.stamps()[:tl.keep] + tl._stamps, base.scenes[:tl.keep] + tl.scenes)
    return tl
