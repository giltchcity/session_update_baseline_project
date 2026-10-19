"""Regressions for A-internal history surviving B/C state reconciliation.

The next layer registry inherits current nodes only. These CPU tests cover closed-state
resurrection, the later A state appearing too early, and legitimate decisions inside B.
"""
import torch

from update_layer.backends.game import t1

MAX = t1.INT64_MAX


def reconcile(created, births, ends, intervals, start, evidence=None):
    """Test fixture for one identity; production supplies these arrays from the map."""
    c, b, d = (torch.tensor(v) for v in (created, births, ends))
    e = d if evidence is None else torch.minimum(d, torch.tensor(evidence))
    return t1.state_membership(torch.full_like(c, 55), c, {55: intervals} if intervals else {},
                               start, prior=(b, d, e))


def test_inheritance_preserves_closed_state_and_later_live_birth():
    b, d = reconcile([10, 25, 45], [5, 20, MAX], [20, MAX, MAX], [(5, None)], 40)
    assert b.tolist() == [5, 20, 5]
    assert d.tolist() == [20, MAX, MAX]


def test_continuing_state_is_selected_at_boundary_not_creation():
    b, d = reconcile([25, 65], [20, MAX], [MAX, MAX], [(35, 60), (60, None)], 40)
    assert b.tolist() == [20, 60]
    assert d.tolist() == [60, MAX]


def test_state_born_at_session_start_ends_the_inherited_state():
    b, d = reconcile([25, 45], [20, MAX], [MAX, MAX], [(5, 40), (40, None)], 40)
    assert b.tolist() == [20, 40] and d.tolist() == [40, MAX]


def test_absent_identity_retains_both_closed_and_live_history():
    b, d = reconcile([10, 25], [5, 20], [20, MAX], [], 40)
    assert b.tolist() == [5, 20] and d.tolist() == [20, MAX]


def test_evidence_closed_before_session_retains_membership():
    b, d = reconcile([10], [5], [MAX], [(30, None)], 40, evidence=[20])
    assert b.tolist() == [5] and d.tolist() == [MAX]


def test_evidence_end_inside_session_does_not_block_layer_revision():
    b, d = reconcile([10], [5], [MAX], [(5, 50), (50, None)], 40, evidence=[51])
    assert b.tolist() == [5] and d.tolist() == [50]


def test_current_session_tv_gets_backdated_state_ownership():
    b, d = reconcile([51], [MAX], [MAX], [(5, 50), (50, None)], 40)
    assert b.tolist() == [50] and d.tolist() == [MAX]


def test_current_session_closed_state_can_still_receive_revised_end():
    b, d = reconcile([45, 55], [40, 50], [54, MAX], [(40, 50), (50, None)], 40)
    assert b.tolist() == [40, 50] and d.tolist() == [50, MAX]


def test_first_known_state_can_end_untracked_carried_rows_in_this_session():
    b, d = reconcile([10, 55], [MAX, MAX], [MAX, MAX], [(50, None)], 40)
    assert b.tolist() == [MAX, 50] and d.tolist() == [50, MAX]


def test_incomplete_registry_cannot_end_a_continuing_row_in_the_past():
    b, d = reconcile([10], [5], [MAX], [(5, 30)], 40)
    assert b.tolist() == [5] and d.tolist() == [MAX]


def test_two_handoffs_preserve_the_entire_previous_history():
    c = torch.tensor([10, 25, 65]); ident = torch.full_like(c, 55)
    b, d = reconcile(c.tolist(), [5, 20, MAX], [20, MAX, MAX], [(5, 60), (60, None)], 40)
    before = (b.clone(), d.clone())
    nb, nd = t1.state_membership(ident, c, {55: [(5, 130), (130, None)]}, 100, prior=(b, d, d))
    assert nb.tolist() == [5, 20, 60] and nd.tolist() == [20, 60, 130]
    for t in (0, 5, 15, 20, 30, 40, 59, 60, 99):
        old_live = t1.alive_at(t, ident, c, b, d, torch.full_like(d, MAX))
        new_live = t1.alive_at(t, ident, c, nb, nd, torch.full_like(d, MAX))
        assert torch.equal(old_live, new_live)
    assert torch.equal(b, before[0]) and torch.equal(d, before[1])
