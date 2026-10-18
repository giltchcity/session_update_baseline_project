"""The state intervals on the backend's clock (UpdateLayer.state_intervals).

[O2] A state the layer first saw at its frame t owns the rows created after the layer's frame before t
(UpdateLayer._owned_from): the backend's keyframes between two layer frames (synthetic B 2795, the layer's 2790 held out
and 2796 the TV's first sighting) show the state; a birth at or before the session's first layer frame (an inherited
state) stays. An identity is one object: a state that has ended ended no later than its identity's next state was born
(the TV cabinet's earlier state closed at 64.8 s, the next one born 51.8 s); an open state stays open."""
from types import SimpleNamespace

from update_layer.core.evidence import EvidenceStore
from update_layer.core.layer import UpdateLayer

S = 1_000_000_000


def _layer(stamps, intervals=None):
    store = EvidenceStore()
    store._stamps = list(stamps)                       # the layer's frames of this session
    lay = SimpleNamespace(store=store, registry=SimpleNamespace(state_intervals=lambda: intervals or {}))
    lay._owned_from = lambda t: UpdateLayer._owned_from(lay, t)
    return lay


def test_owned_from_the_layer_frame_before():
    lay = _layer([10 * S, 12 * S, 16 * S])
    assert UpdateLayer._owned_from(lay, 16 * S) == 12 * S + 1
    assert UpdateLayer._owned_from(lay, 12 * S) == 10 * S + 1
    assert UpdateLayer._owned_from(lay, 10 * S) == 10 * S      # the session's first frame
    assert UpdateLayer._owned_from(lay, 3 * S) == 3 * S        # an inherited state's birth
    assert UpdateLayer._owned_from(_layer([]), 16 * S) == 16 * S


def test_an_ended_state_ends_when_the_next_is_born():
    lay = _layer([10 * S, 12 * S, 16 * S, 20 * S],
                 {55: [(3 * S, 20 * S), (16 * S, None)],                  # closed after the next state was born
                  7: [(3 * S, 10 * S), (16 * S, None)],                   # closed before: stays
                  9: [(3 * S, None), (12 * S, None)]})                    # still open beside a pending one: stays open
    out = UpdateLayer.state_intervals(lay)
    assert out[55] == [(3 * S, 12 * S + 1), (12 * S + 1, None)]
    assert out[7] == [(3 * S, 10 * S), (12 * S + 1, None)]
    assert out[9] == [(3 * S, None), (10 * S + 1, None)]
