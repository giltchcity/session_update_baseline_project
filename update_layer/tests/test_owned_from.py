"""[O2] A state the layer first saw at its frame t owns the rows created after the layer's frame before t (UpdateLayer._owned_from):
the backend's keyframes between two layer frames (synthetic B 2795, the layer's 2790 held out and 2796 the TV's first
sighting) show the state; a birth at or before the session's first layer frame (an inherited state) stays."""
from types import SimpleNamespace

from update_layer.core.evidence import EvidenceStore
from update_layer.core.layer import UpdateLayer

S = 1_000_000_000


def test_owned_from_the_layer_frame_before():
    store = EvidenceStore()
    store._stamps = [10 * S, 12 * S, 16 * S]          # the layer's frames of this session
    lay = SimpleNamespace(store=store)
    assert UpdateLayer._owned_from(lay, 16 * S) == 12 * S + 1
    assert UpdateLayer._owned_from(lay, 12 * S) == 10 * S + 1
    assert UpdateLayer._owned_from(lay, 10 * S) == 10 * S      # the session's first frame
    assert UpdateLayer._owned_from(lay, 3 * S) == 3 * S        # an inherited state's birth
    assert UpdateLayer._owned_from(SimpleNamespace(store=EvidenceStore()), 16 * S) == 16 * S
