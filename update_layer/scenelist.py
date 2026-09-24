"""EvaluationScenes stored per snapshot (for maps evaluated by rendering, e.g. 3DGS).

Kept free of heavy imports so the scoring environments can unpickle it.
"""
from __future__ import annotations

import bisect
import pickle
import sys
from typing import List

import numpy as np

sys.path.insert(0, "/home/jixian/Desktop/FT")
from eval.scene.scene import EvaluationScene  # noqa: E402


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
