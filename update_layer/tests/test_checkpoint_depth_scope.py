"""CPU regression for preserving depth-loss scope across checkpoint restoration.

Executes the actual save, restore and finish_session method bodies. Only CUDA
transfers and renderer/model construction are replaced by CPU test doubles.
No optimizer step, production file edit, configuration or training is performed.
"""
import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

CODE = Path(__file__).resolve().parents[2]
GAME = "update_layer/backends/game/game.py"
GROWTH = "update_layer/backends/game/growth.py"
ROWS = ("identity", "last_update", "created", "state_birth", "death_state",
        "death_evidence", "death_prune", "frozen")


class Model:
    def restore(self, data, options):
        self.data = data

    def capture(self):
        return self.data


class Map:
    def __init__(self, config):
        self.config = config
        self.has_prior = False
        self.gaussian_model = Model()
        self.opt_params = None

    def optimize_model(self, **kwargs):
        return {"has_prior": self.has_prior, "refinement": kwargs["refinement"]}


def methods(game, growth):
    ns = dict(TrackedGaME=Map, ROWS=ROWS, _to_gpu=lambda value: value,
              _label_weight_store=lambda value: value, LabelTable=torch.Tensor)
    exec(compile(growth, GROWTH, "exec"), ns)
    cls = next(n for n in ast.parse(game).body if isinstance(n, ast.ClassDef)
               and n.name == "GameBackend")
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in {"prior_state", "prior_from_state", "finish_session"}]
    exec("from __future__ import annotations\n" + "\n".join(ast.unparse(n) for n in selected), ns)
    return ns


def state(created, start=100):
    n = len(created)
    s = {k: torch.zeros(n, dtype=torch.int64) for k in ROWS}
    s.update(created=torch.tensor(created, dtype=torch.int64), frozen=torch.zeros(n, dtype=torch.bool),
             uid=torch.arange(n), model=(torch.tensor([1.25]),), keyframes={0: {}}, estimated_poses={},
             ignored_frames=set(), occlusion_masks={}, last_keyframe_id=0, next_uid=n, frame_counter=1,
             now=120, kf_stamp={0: 100}, timed=True, tracked_labels=None, session_start=start,
             support_tol=.05, label_ids=[], label_weight=torch.zeros((n, 1)), semantic_of={})
    return s


def backend():
    return SimpleNamespace(config={"refinement_iters": 1}, tolerance=.05, scale=1., semantic_of={})


def test_checkpoint_roundtrip_preserves_depth_scope():
    torch.set_num_threads(1)
    fixed = methods((CODE/GAME).read_text(), (CODE/GROWTH).read_text())
    cases = [("fresh_empty_A", state([]), False),
             ("A_created_at_boundary", state([100, 105]), False),
             ("B_inherits_A", state([50, 100, 110]), True),
             ("C_inherits_AB", state([1, 80, 100]), True)]
    legacy = state([5, 150]); del legacy["session_start"]
    legacy["session_starts"] = [1, 100]
    cases.append(("legacy_recorded_boundary", legacy, True))
    empty_legacy = state([]); del empty_legacy["session_start"]
    empty_legacy["session_starts"] = []
    cases.append(("legacy_empty_boundary", empty_legacy, False))
    removed = state([110]); removed["has_prior"] = True
    cases.append(("saved_scope_survives_removed_prior", removed, True))
    explicit = state([100]); explicit["has_prior"] = False
    cases.append(("saved_fresh_scope", explicit, False))
    results = []
    # CPU placement is the only alteration to the checkpoint restoration bodies.
    with patch.object(torch.Tensor, "cuda", lambda self, *a, **kw: self):
        for name, s, expected in cases:
            be = backend()
            be.game = fixed["prior_from_state"](be, s)
            assert be.game.has_prior is expected, name
            assert torch.equal(be.game.created, s["created"]), name
            assert be.game.support_tol == .05, name
            if be.game.session_start is not None:
                report = fixed["finish_session"](be, 150)
                assert report == {"has_prior": expected, "refinement": True}, name
            # Use the actual checkpoint serializer and a real torch save/load round trip.
            saved = fixed["prior_state"](be, be.game)
            buf = io.BytesIO(); torch.save(saved, buf); buf.seek(0)
            loaded = torch.load(buf, weights_only=False)
            restored = fixed["prior_from_state"](be, loaded)
            assert restored.has_prior is expected, name
            assert torch.equal(restored.gaussian_model.data[0], s["model"][0]), name
            results.append(dict(case=name, expected=expected, restored=restored.has_prior))
    report = dict(passed=True, scope=__doc__, cases=results)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    test_checkpoint_roundtrip_preserves_depth_scope()
