"""CPU tests of lifecycle ordering and permanent decisions at the runner/backend boundary.

The fixture does no mapping or optimization. It exposes active/dormant IDs so the real
run_chain can be checked for pre-integration events, 1 Hz evidence queries and unchanged
permanent retirement calls, then writes actual run.json and CPU checkpoints.
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from update_layer import run
from update_layer.eval.scenelist import SceneListTimeline
from update_layer.interface import Backend, Elements


def _ids(values=()):
    return torch.tensor(list(values), dtype=torch.int64)


class _Frames:
    def __init__(self, _spec, pixel_step):
        self.ids = list(range(31))

    def stamp_ns(self, index):
        return 1_000_000_000 + round(index * 1_000_000_000 / 30)

    def load(self, index):
        return SimpleNamespace(index=index, stamp_ns=self.stamp_ns(index))


class _LifecycleBackend(Backend):
    CHANGES = ()

    def __init__(self, events, supported=True):
        self.events = events
        self.supports_reversible_elements = supported
        self.active = {11, 42, 77, 88}
        self.dormant = set()
        self.permanent = set()
        self.surface_weights = {uid: 1.0 for uid in self.active}
        self.stamps, self.scenes = [], []
        self.lifecycle_stats = {"suppressed": 0, "reactivated": 0, "positive_reactivations": 0}

    def start_session(self, spec, _prior):
        self.events.append(("start", spec.name))

    def _view(self, ids, decision=False):
        ids = sorted(ids)
        n = len(ids)
        identities = {11: 11, 42: 900, 77: 77, 88: 901}
        return Elements(_ids(ids), torch.zeros((n, 3)), torch.full((n, 3), float("nan")),
                        _ids([0 if decision and uid in (42, 88) else identities[uid] for uid in ids]),
                        _ids([0] * n), created=_ids([1] * n),
                        suppressed=torch.tensor([uid in self.dormant for uid in ids], dtype=torch.bool),
                        state_birth=_ids([1] * n),
                        surface_weight=torch.tensor([self.surface_weights[uid] for uid in ids], dtype=torch.float32))

    def elements(self):
        return self._view(self.active, decision=True)

    def evidence_elements(self):
        if not self.supports_reversible_elements:
            raise AssertionError("an unsupported backend must not receive lifecycle evidence queries")
        result = self._view(self.active | self.dormant)
        self.events.append(("evidence", tuple(result.ids.tolist())))
        return result

    def decision_elements(self):
        if not self.supports_reversible_elements:
            raise AssertionError("the legacy runner path should use elements()")
        result = self._view(self.active | self.dormant, decision=True)
        self.events.append(("decision_view", tuple(result.ids.tolist())))
        return result

    def update_surface_weights(self, ids, weights):
        self.events.append(("weights", tuple(ids.tolist()), tuple(weights.tolist())))
        for uid, weight in zip(ids.tolist(), weights.tolist(), strict=True):
            self.surface_weights[uid] = weight

    def suppress(self, ids, stamp):
        changed = self.active.intersection(ids.tolist()) - self.permanent
        self.active -= changed
        self.dormant |= changed
        self.lifecycle_stats["suppressed"] += len(changed)
        self.events.append(("suppress", stamp, tuple(sorted(changed))))
        return len(changed)

    def reactivate(self, ids, stamp):
        changed = self.dormant.intersection(ids.tolist()) - self.permanent
        self.dormant -= changed
        self.active |= changed
        self.lifecycle_stats["reactivated"] += len(changed)
        self.events.append(("reactivate", stamp, tuple(sorted(changed))))
        return len(changed)

    def integrate(self, frame):
        if self.supports_reversible_elements and frame.index == 6:
            # A backend may also recover a row at a keyframe between the layer's 1 Hz checks.
            self.surface_weights[77] = 1.0
            self.lifecycle_stats["positive_reactivations"] += self.reactivate(_ids([77]), frame.stamp_ns)
        self.events.append(("integrate", frame.index, tuple(sorted(self.active))))

    def retire(self, ids, stamp):
        self.events.append(("retire", stamp, tuple(ids.tolist())))
        self.permanent.update(ids.tolist())
        self.active.difference_update(ids.tolist())
        self.dormant.difference_update(ids.tolist())

    def snapshot(self, stamp):
        self.stamps.append(stamp)
        self.scenes.append({"active": sorted(self.active)})

    def timeline(self):
        return SceneListTimeline(list(self.stamps), list(self.scenes))

    def end_session(self):
        return self

    def prior_state(self, _prior):
        return {"active": sorted(self.active), "dormant": sorted(self.dormant),
                "permanent": sorted(self.permanent)}


class _LifecycleLayer:
    def __init__(self, events):
        self.events = events
        self.reversible_elements = False
        self.last_element_stamp = None
        self.decided_once = False

    def start_session(self, *_args):
        pass

    def motion(self, frame):
        self.events.append(("motion", frame.index))
        return np.zeros((1, 1), dtype=bool)

    def element_observation_due(self, stamp):
        assert self.reversible_elements
        due = self.last_element_stamp is None or stamp - self.last_element_stamp >= 1_000_000_000
        self.events.append(("due", stamp, due))
        return due

    def update_elements(self, frame, elements):
        assert frame.stamp_ns in run.MOTION, "motion must be available before evidence evaluation"
        assert elements.identity[elements.ids == 88].item() == 901, "support evidence needs physical identity"
        if 42 in elements.ids.tolist():
            assert elements.identity[elements.ids == 42].item() == 900
        self.events.append(("update", frame.index, tuple(elements.ids.tolist())))
        self.last_element_stamp = frame.stamp_ns
        if frame.index == 0:
            return {"suppress": _ids([11, 42, 77, 88]), "reactivate": _ids(),
                    "weights": torch.zeros_like(elements.surface_weight)}
        weights = elements.surface_weight.clone()
        weights[elements.ids == 11] = 2.0
        return {"suppress": _ids(), "reactivate": _ids([11]), "weights": weights}

    def observe(self, frame):
        self.events.append(("observe", frame.index))

    def decide(self, stamp, last, elements, **_kwargs):
        self.events.append(("decide", stamp, tuple(elements.ids.tolist())))
        if not self.decided_once:
            assert 42 in elements.ids.tolist(), "dormant rows must remain visible to permanent closure checks"
            assert elements.identity[elements.ids == 42].item() == 0, "permanent checks keep the O2 mapping"
            if self.reversible_elements:
                at_42 = elements.ids == 42
                assert bool(elements.suppressed[at_42].all())
            self.decided_once = True
            return {"closed_background": _ids([42]),
                    "element_rule": _ids() if self.reversible_elements else _ids([11])}
        return {"closed_background": _ids(), "element_rule": _ids()}

    def session_end_memory(self, elements, *_args):
        self.events.append(("memory", tuple(elements.ids.tolist())))
        if self.reversible_elements:
            assert 88 in elements.ids.tolist(), "G5 must also see suppressed memory"
            assert bool(elements.suppressed[elements.ids == 88].all())
            assert elements.identity[elements.ids == 88].item() == 0, "G5 keeps the O2 identity mapping"
        return _ids(), 1_000_000_000

    def end_session(self, _directory):
        return None


class ElementLifecycleRunnerTests(unittest.TestCase):
    def _run(self, supported):
        events = []
        backend = _LifecycleBackend(events, supported)
        layer = _LifecycleLayer(events)
        cfg = SimpleNamespace(round_s=0.5, evidence_hz=5.0, pixel_step=2)
        spec = SimpleNamespace(name="synthetic_b")
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with mock.patch.object(run, "dataset_config", return_value=(cfg, SimpleNamespace(), [spec])), \
                    mock.patch.object(run, "FlatSession", _Frames), \
                    mock.patch.object(run, "UpdateLayer", return_value=layer), \
                    mock.patch.object(run, "make_backend", return_value=backend), \
                    contextlib.redirect_stdout(io.StringIO()):
                run.run_chain("game", 4, "synthetic", out, verbose=False, g5=True)
            report = json.loads((out / "session_b/run.json").read_text())
        return backend, layer, events, report

    def test_evidence_and_reactivation_precede_integrate_without_moving_observe(self):
        backend, layer, events, report = self._run(True)
        self.assertTrue(layer.reversible_elements)
        for index in (0, 30):
            motion = next(i for i, event in enumerate(events) if event[:2] == ("motion", index))
            update = next(i for i, event in enumerate(events) if event[:2] == ("update", index))
            integrate = next(i for i, event in enumerate(events) if event[:2] == ("integrate", index))
            observe = next(i for i, event in enumerate(events) if event[:2] == ("observe", index))
            self.assertLess(motion, update)
            self.assertLess(update, integrate)
            self.assertLess(integrate, observe)
            weights = next(i for i in range(update + 1, integrate) if events[i][0] == "weights")
            action = next(i for i in range(update + 1, integrate) if events[i][0] in ("reactivate", "suppress"))
            self.assertLess(weights, action)
        integrated = {event[1]: event[2] for event in events if event[0] == "integrate"}
        self.assertNotIn(11, integrated[0])
        self.assertIn(11, integrated[30])
        self.assertIn(77, integrated[6])
        # Six 5 Hz evidence frames over one second, but only two 1 Hz lifecycle observations.
        self.assertEqual(len([event for event in events if event[0] == "due"]), 6)
        self.assertEqual(len([event for event in events if event[0] == "update"]), 2)
        middle_motion = next(i for i, event in enumerate(events) if event[:2] == ("motion", 6))
        middle_integrate = next(i for i, event in enumerate(events) if event[:2] == ("integrate", 6))
        self.assertFalse(any(event[0] == "evidence" for event in events[middle_motion:middle_integrate]))
        lifecycle = report["element_lifecycle"]
        self.assertEqual(lifecycle["observations"], 2)
        self.assertEqual(lifecycle["suppressed"], 4)
        self.assertEqual(lifecycle["reactivated"], 1)
        self.assertGreaterEqual(lifecycle["host_seconds"], 0)
        self.assertEqual(report["backend_element_lifecycle"]["positive_reactivations"], 1)
        self.assertEqual(report["backend_element_lifecycle"]["reactivated"], 2)
        # The permanent reason is still applied at the round and is never converted into a lifecycle event.
        self.assertEqual(report["retired_by_layer"]["closed_background"], 1)
        self.assertEqual(report["retired_by_layer"]["element_rule"], 0)
        self.assertEqual(backend.permanent, {42})
        self.assertNotIn(42, backend.active | backend.dormant)
        self.assertIn(88, backend.dormant)
        self.assertEqual(backend.surface_weights[11], 2.0)
        self.assertEqual(backend.surface_weights[77], 1.0)
        self.assertEqual(backend.surface_weights[88], 0.0)

    def test_backend_without_capability_preserves_permanent_rule_path(self):
        backend, layer, events, report = self._run(False)
        self.assertFalse(layer.reversible_elements)
        self.assertFalse(any(event[0] in ("due", "update", "evidence", "decision_view", "weights", "suppress", "reactivate")
                             for event in events))
        self.assertEqual(report["retired_by_layer"]["element_rule"], 1)
        self.assertEqual(report["retired_by_layer"]["closed_background"], 1)
        self.assertEqual(backend.permanent, {11, 42})
        self.assertFalse(report["element_lifecycle"]["enabled"])

    def test_elements_selection_keeps_suppression_and_state_membership(self):
        backend = _LifecycleBackend([])
        backend.suppress(_ids([11]), 10)
        elements = backend.evidence_elements()
        selected = elements.select(elements.ids == 11)
        self.assertEqual(selected.ids.tolist(), [11])
        self.assertEqual(selected.suppressed.tolist(), [True])
        self.assertEqual(selected.state_birth.tolist(), [1])
        self.assertEqual(selected.surface_weight.tolist(), [1.0])
        legacy = Elements(_ids([1]), torch.zeros((1, 3)), torch.zeros((1, 3)), _ids([0]), _ids([0]))
        self.assertEqual(legacy.suppressed.tolist(), [False])
        self.assertIsNone(legacy.state_birth)
        self.assertIsNone(legacy.surface_weight)


if __name__ == "__main__":
    unittest.main()
