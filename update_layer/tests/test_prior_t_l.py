"""The prior carries the earliest owned birth (t_L) of an identity's states, and the next session's inherited state
is owned from it: rows created between t_L and the fragment's first observation stay the state's (real_row4f B 10-08)."""
import torch
from update_layer.core.l2 import state as l2


def test_inherited_state_owned_from_t_l():
    reg = l2.PersistentObjectState(0.05, [])
    pts = torch.zeros((10, 3))
    reg.initialize_from_objects([dict(identity=19, points=pts, normals=None, first=1_000_000_000, last=131_000_000_000,
                                      semantic=139, track_first_seen=200_000_000, bbox_valid=True)])
    (birth, death), = reg.state_intervals()[19]
    assert birth == 200_000_000 and death is None
    reg2 = l2.PersistentObjectState(0.05, [])
    reg2.initialize_from_objects([dict(identity=19, points=pts, normals=None, first=1_000_000_000, last=131_000_000_000,
                                       semantic=139, bbox_valid=True)])
    (birth2, _), = reg2.state_intervals()[19]
    assert birth2 == 1_000_000_000          # without t_L the state starts at its first observation (the old defect)
