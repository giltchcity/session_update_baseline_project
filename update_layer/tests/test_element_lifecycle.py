"""Behavioral regressions for sensor-time evidence; no renderer or GT-score stand-ins."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from update_layer.core.element_lifecycle import (SurfaceWeightConfig, SurfaceWeightEvidence,
                                                  classify_elements)
from update_layer.core.layer import LayerConfig, UpdateLayer
from update_layer.frames import Frame, Intrinsics, MOTION
from update_layer.interface import Elements

SECOND = 1_000_000_000


def elements(points=((0.0, 0.0, 2.0),), identity=7, suppressed=False, uid=42):
    n = len(points)
    out = Elements(torch.arange(uid, uid + n, dtype=torch.int64), torch.tensor(points, dtype=torch.float32),
                   torch.full((n, 3), float("nan")), torch.full((n,), identity, dtype=torch.int64),
                   torch.zeros(n, dtype=torch.int64), torch.zeros(n))
    out.suppressed = torch.full((n,), suppressed, dtype=torch.bool)
    return out


def frame(depth=2.0, identity=7, stamp=0, offset=0.0):
    K = Intrinsics(64, 48, 60.0, 60.0, 32.0, 24.0, offset)
    return Frame(0, stamp, np.full((48, 64), depth, np.float32), np.full((48, 64), identity, np.int32),
                 None, np.eye(4), K)


def observations(kind, n=1):
    return dict(hit=torch.full((n,), kind == "hit", dtype=torch.bool),
                support=torch.full((n,), 1.0 if kind == "hit" else 0.0),
                through_weight=torch.full((n,), 1.0 if kind == "through" else 0.0))


def test_support_is_not_discarded_and_real_removal_is_eventually_suppressed():
    state, el = SurfaceWeightEvidence(), elements()
    # A stable surface retains eight effective observations. A short miss burst must not erase it.
    for t in range(10):
        assert not len(state.update(t * SECOND, el, observations("hit"))["suppress"])
    assert state.weights.item() == 8
    for t in range(10, 13):
        assert not len(state.update(t * SECOND, el, observations("through"))["suppress"])
    before = state.weights.clone()
    state.update(13 * SECOND, el, observations("unknown"))
    assert torch.equal(state.weights, before)
    # A genuinely removed surface with sustained reliable free space still leaves the current map.
    for t in range(14, 19):
        assert not len(state.update(t * SECOND, el, observations("through"))["suppress"])
    action = state.update(19 * SECOND, el, observations("through"))
    assert action["suppress"].tolist() == [42]


def test_dormant_negative_debt_is_bounded_and_strict_hit_reuses_uid():
    state, el = SurfaceWeightEvidence(), elements(suppressed=True)
    for t in range(50):
        state.update(t * SECOND, el, observations("through"))
    assert state.weights.item() == -1
    result = state.update(50 * SECOND, el, observations("hit"))
    assert result["reactivate"].tolist() == [42]
    assert state.weights.item() >= 1


def test_keyframe_restore_between_events_clears_old_negative_state():
    state, el = SurfaceWeightEvidence(), elements(suppressed=True)
    state.update(0, el, observations("through"))
    # The backend's strict positive query restored this same uid on an unsampled keyframe.
    el.suppressed[:] = False
    result = state.update(SECOND, el, observations("through"))
    assert not len(result["suppress"])
    assert state.weights.item() == 0


def test_weak_unlabeled_support_cannot_charge_dormant_confidence():
    state, el = SurfaceWeightEvidence(), elements(suppressed=True)
    weak = dict(hit=torch.tensor([False]), support=torch.tensor([0.5]), through_weight=torch.tensor([0.0]))
    for t in range(20):
        result = state.update(t * SECOND, el, weak)
        assert not len(result["reactivate"])
    assert state.weights.item() == -1


def test_new_densified_uid_keeps_canonical_parent_confidence():
    state = SurfaceWeightEvidence()
    child = elements(uid=84)
    child.surface_weight = torch.tensor([8.0])  # propagated by the backend clone hook
    result = state.update(0, child, observations("through"))
    assert not len(result["suppress"]) and result["weights"].item() == 7


def test_sensor_time_is_invariant_to_round_grouping_duplicate_calls_and_session_gap():
    def drive(round_ns):
        layer = UpdateLayer(LayerConfig(round_s=round_ns / SECOND, map_resolution=0.05, max_range=5.0))
        layer.reversible_elements = True
        layer.start_session(SimpleNamespace(depth_range=(0.1, 5.0)))
        state, el, actions = layer.surface_evidence, elements(), []
        stamp = 0
        round_start = 0
        sequence = ["hit"] * 8 + ["through"] * 20 + ["hit"] * 8
        # Round boundaries have no input to this state machine. Calls every 0.2 seconds
        # retain the same 1 Hz observation timestamps for both original round settings.
        for k, kind in enumerate(sequence):
            stamp = k * 200_000_000
            result = state.update(stamp, el, observations(kind))
            for action in ("suppress", "reactivate"):
                if len(result[action]):
                    actions.append((stamp, action, result[action].tolist()))
                    el.suppressed[:] = action == "suppress"
            assert not any(len(v) for v in state.update(stamp, el, observations(kind)).values())
            if stamp - round_start >= round_ns or k == len(sequence) - 1:
                # Execute the actual reconciliation entry point with an empty object registry.
                # The old per-round element CUSUM must neither re-count nor permanently retire.
                permanent = layer.decide(stamp, k == len(sequence) - 1, el)
                assert not any(len(v) for v in permanent.values())
                round_start = stamp
        return state, actions
    a, events_a = drive(2_100_000_000)
    b, events_b = drive(10_800_000_000)
    assert events_a == events_b and events_a
    assert torch.equal(a.weights, b.weights)
    old = a.weights.clone()
    a.update(10**18, elements(), observations("through"))
    assert a.weights.item() == max(-1, old.item() - 1)  # one observation, not elapsed-time carving


def test_checkpoint_retains_weights_clock_and_rejects_invalid_or_oversize_metadata():
    state, el = SurfaceWeightEvidence(), elements()
    for t in range(4):
        state.update(t * SECOND, el, observations("hit"))
    restored = SurfaceWeightEvidence(state=state.state())
    assert torch.equal(restored.weights, state.weights)
    assert not restored.due(3 * SECOND)
    assert restored.due(4 * SECOND)
    bad = state.state()
    bad["weights"] = torch.tensor([float("nan")])
    with pytest.raises(ValueError):
        SurfaceWeightEvidence(state=bad)
    for invalid_stamp in (None, "3000000000", 3.0, -1, 2**64):
        bad = state.state()
        bad["last_stamp"] = invalid_stamp
        with pytest.raises(ValueError):
            SurfaceWeightEvidence(state=bad)
    with pytest.raises(ValueError, match="parameters"):
        SurfaceWeightEvidence(config=SurfaceWeightConfig(max_weight=9), state=state.state())
    small = SurfaceWeightEvidence(config=SurfaceWeightConfig(max_state_bytes=1))
    with pytest.raises(RuntimeError, match="budget"):
        small.update(0, el, observations("hit"))
    assert small.last_stamp is None and small.ids is None


def test_permanently_removed_ids_leave_weight_index_and_relabel_starts_clean():
    state, el = SurfaceWeightEvidence(), elements()
    for t in range(8):
        state.update(t * SECOND, el, observations("hit"))
    el.identity[:] = 9
    state.update(8 * SECOND, el, observations("unknown"))
    assert state.weights.item() == 1
    empty = elements(points=()).select(torch.zeros(0, dtype=torch.bool))
    state.update(9 * SECOND, empty, observations("unknown", 0))
    assert len(state.ids) == 0


@pytest.mark.parametrize("offset", [0.0, 0.5])
def test_geometry_identity_occlusion_and_invalid_observations(offset):
    el = elements()
    f = frame(offset=offset)
    assert classify_elements(f, el, 0.05, 5)["hit"].tolist() == [True]
    foreign = classify_elements(replace(f, instance=np.full_like(f.instance, 8)), el, 0.05, 5)
    assert not foreign["hit"].any() and not foreign["support"].any()
    for depth in (1.0, float("nan"), 0.0, 6.0):
        result = classify_elements(replace(f, depth=np.full_like(f.depth, depth)), el, 0.05, 5)
        assert not result["hit"].any() and not result["through_weight"].any()
    through = classify_elements(replace(f, depth=np.full_like(f.depth, 4.0)), el, 0.05, 5)
    assert through["through_weight"].item() == 1


def test_rounded_pixel_uses_its_actual_measured_ray_for_recovery():
    f = frame()
    # A point is close to the outer half of its selected pixel. Construct a measured
    # RANGE just outside the tolerance; scaling both z values by the point ray would
    # incorrectly admit it. Pixel-center convention must be honored after rounding.
    u = 48
    cam = torch.tensor([[(u - f.K.cx - 0.49) / f.K.fx * 2.0, 0.0, 2.0]])
    qrange = torch.linalg.vector_norm(cam, dim=1).item()
    actual_scale = np.sqrt(1 + ((u - f.K.cx) / f.K.fx) ** 2)
    f.depth[:] = (qrange + 0.0501) / actual_scale
    el = elements(points=cam.tolist(), suppressed=True)
    assert not classify_elements(f, el, 0.05, 5)["hit"].any()


def test_thin_surface_nearby_veto_and_holes_reduce_instead_of_erase_reliability():
    f = frame(depth=4.0, identity=0)
    el = elements(identity=0)
    f.depth[:, 33] = 2.0  # projected center sees through, one neighboring pixel still holds the thin surface
    assert classify_elements(f, el, 0.05, 5)["through_weight"].item() == 0
    f.depth[:] = 4.0
    f.depth[23, 31] = np.nan  # diagonal hole leaves usable depth derivatives and a coherent center
    weight = classify_elements(f, el, 0.05, 5)["through_weight"].item()
    assert 0 < weight < 1
    f.depth[:] = np.nan
    f.depth[24, 32] = 4
    assert classify_elements(f, el, 0.05, 5)["through_weight"].item() == 0


def test_d1_pixels_are_unknown_and_known_identity_can_recover_a_thin_center():
    f, el = frame(), elements(suppressed=True)
    f.depth[:] = 4
    f.depth[24, 32] = 2
    assert classify_elements(f, el, 0.05, 5)["hit"].item()  # exact identity corroborates the thin center
    MOTION[f.stamp_ns] = np.ones(f.depth.shape, dtype=bool)
    try:
        result = classify_elements(f, el, 0.05, 5)
        assert not result["hit"].any() and not result["through_weight"].any()
    finally:
        MOTION.pop(f.stamp_ns, None)


def test_foreign_label_cannot_recover_nearby_old_object_and_grazing_normal_cannot_carve():
    f, el = frame(identity=9), elements(suppressed=True)
    assert not classify_elements(f, el, 0.05, 5)["hit"].any()
    el.normal[:] = torch.tensor([1.0, 0.0, 0.0])
    result = classify_elements(frame(depth=4.0), el, 0.05, 5)
    assert not result["through_weight"].any()
