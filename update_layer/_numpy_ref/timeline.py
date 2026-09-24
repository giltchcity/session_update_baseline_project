"""A session run of (backend + update layer) as an EvaluationScene source.

Scene at t = background elements alive at t (birth <= t < death) plus the object states the
registry displayed at the latest snapshot <= t (Khronos getDsgPtr semantics). One object per
physical identity (Khronos canonicalises to one node per physical id).
"""
from __future__ import annotations

import bisect
import pickle
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, "/home/jixian/Desktop/FT")
from eval.scene.scene import EvaluationScene, SceneObject  # noqa: E402

UINT64_MAX = 18446744073709551615


class LayerTimeline:
    def __init__(self, res):
        el = res.elements
        n = el.n
        self.el_xyz = (el.sum[:n] / el.cnt[:n, None]).astype(np.float32)
        self.el_label = el.label[:n].astype(np.uint32)
        self.el_normal = el.normals(np.arange(n)).astype(np.float32)
        self.el_birth = el.birth[:n].copy()
        self.el_death = el.death[:n].copy()
        self.el_inherited = el.inherited[:n].copy()
        self.el_identity = el.identity[:n].copy()
        self._stamps: List[int] = list(res.stamps)
        self.display = res.display
        self.frag_points = {uid: f.points for uid, f in res.fragments.items()}
        self.frag_birth = {uid: f.birth_time for uid, f in res.fragments.items()}
        self.frag_normals = {uid: (f.normals if f.normals is not None
                                   else np.full((len(f.points), 3), np.nan)) for uid, f in res.fragments.items()}

    def stamps(self) -> List[int]:
        return self._stamps

    def snapshot_index(self, t_ns: int) -> int:
        return max(0, bisect.bisect_right(self._stamps, t_ns) - 1)

    def scene(self, t_ns: int, background: bool = True) -> EvaluationScene:
        k = self.snapshot_index(t_ns)
        objects = []
        for identity, semantic, parts in self.display[k]:
            pts = np.concatenate([self.frag_points[uid][:n] for uid, n in parts]).astype(np.float32)
            fn = getattr(self, "frag_normals", {})
            nrm = np.concatenate([fn[uid][:n] if uid in fn else np.full((n, 3), np.nan)
                                  for uid, n in parts]).astype(np.float32)
            first = int(self.frag_birth[parts[0][0]])
            objects.append(SceneObject(id=str(identity), instance_id=int(identity), semantic=int(semantic),
                                       points=pts, present=True, first_observed_ns=[first],
                                       last_observed_ns=[UINT64_MAX], observation_first_ns=first,
                                       observation_last_ns=int(t_ns), normals=nrm))
        el_identity = getattr(self, "el_identity", None)
        if el_identity is not None and (el_identity > 0).any():
            # element-only baselines: an object is the alive elements carrying its identity
            alive = (self.el_birth <= t_ns) & (t_ns < self.el_death)
            for identity in np.unique(el_identity[alive & (el_identity > 0)]):
                m = alive & (el_identity == identity)
                first = int(self.el_birth[m].min())
                objects.append(SceneObject(id=str(int(identity)), instance_id=int(identity),
                                           semantic=int(np.bincount(self.el_label[m].astype(np.int64)).argmax()),
                                           points=self.el_xyz[m], present=True, first_observed_ns=[first],
                                           last_observed_ns=[UINT64_MAX], observation_first_ns=first,
                                           observation_last_ns=int(t_ns), normals=self.el_normal[m]))
        if background:
            alive = (self.el_birth <= t_ns) & (t_ns < self.el_death)
            if el_identity is not None:
                alive &= el_identity <= 0
            el_normal = getattr(self, "el_normal", None)
            return EvaluationScene(t_ns, self.el_xyz[alive], self.el_label[alive], objects,
                                   None if el_normal is None else el_normal[alive])
        return EvaluationScene(t_ns, np.zeros((0, 3), np.float32), np.zeros(0, np.uint32), objects)

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path) -> "LayerTimeline":
        with open(path, "rb") as f:
            return pickle.load(f)
