"""A state ends when it was observed to end (10-22, synthetic B 517293c: the dining table, last seen at 48.8 s and seen
gone between 75.6 and 81.3 s, ended at its 86.4 s close, so the map of 81 s drew it): the evidence's first frame after
the window's last support that saw the surface gone, the registry's export of it, and the layer's cap by a successor
only (a state born while the earlier one was still seen is another hypothesis of the same object)."""
import torch
from update_layer.core.l2 import evidence as ev
from update_layer.core.l2 import state as l2
from update_layer.core.layer import UpdateLayer

S = 1_000_000_000


class _Store:
    """Frames at the given stamps; project() hands classify_measurements the frame range."""

    def __init__(self, stamps):
        self.s = torch.tensor(stamps, dtype=torch.int64, device=ev.DEV)

    def window(self, a, b):
        idx = ((self.s >= a) & (self.s <= b)).nonzero().flatten()
        return (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, 0)

    def stamp_tensor(self, lo, hi):
        return self.s[lo:hi]

    def project(self, f0, f1, pts):
        return {"frames": (f0, f1), "pixel": torch.zeros((f1 - f0, len(pts)), dtype=torch.int64, device=ev.DEV)}


def _first_contradiction(votes, monkeypatch, earliest=1):
    store = _Store([k * S for k in sorted(votes)])
    keys = sorted(votes)
    monkeypatch.setattr(ev, "classify_measurements", lambda p, pid, tol: torch.tensor(
        [[votes[keys[k]]] * p["pixel"].shape[1] for k in range(*p["frames"])], device=ev.DEV))
    res = ev.AbsenceModel(store).count_projected_surface(7, torch.tensor([[0.0, 0, 0], [1.0, 0, 0]]), 0.05,
                                                         earliest * S, max(votes) * S)
    return res.latest_support_stamp // S, res.first_contradiction_stamp // S


def test_first_contradiction_after_the_last_support(monkeypatch):
    # seen gone at 1 s, supported at 2 s, unobserved at 3 s, gone at 4 and 5 s: it ended at 4 s
    assert _first_contradiction({1: ev.V_FREE, 2: ev.V_SUPPORTED, 3: ev.V_UNAVAILABLE, 4: ev.V_BACKGROUND,
                                 5: ev.V_FREE, 6: ev.V_UNAVAILABLE}, monkeypatch) == (2, 4)
    # occluded or unobserved is no evidence of its end; replaced by another object is
    assert _first_contradiction({1: ev.V_OCCLUDED, 2: ev.V_UNAVAILABLE, 3: ev.V_OTHER}, monkeypatch) == (0, 3)
    assert _first_contradiction({1: ev.V_SUPPORTED, 2: ev.V_OCCLUDED}, monkeypatch) == (1, 0)


def _closed(first_absence, close_at):
    reg = l2.PersistentObjectState(0.05, [])
    reg.initialize_from_objects([dict(identity=5, points=torch.zeros((10, 3)), normals=None, first=1 * S, last=40 * S,
                                      semantic=1, bbox_valid=True)])
    st = reg.states[5]
    st.fragments[st.current].first_absence = first_absence
    l2.PersistentObjectState.close_current(st, close_at)
    (birth, end, seen), = reg.state_intervals()[5]
    return birth // S, end // S, seen // S


def test_export_ends_at_the_first_absence():
    assert _closed(52 * S, 60 * S) == (1, 52, 40)          # seen gone at 52 s, closed at the round of 60 s
    assert _closed(0, 60 * S) == (1, 60, 40)               # closed without seeing it gone: its close


def test_cap_by_a_successor_only():
    lay = UpdateLayer.__new__(UpdateLayer)
    lay._owned_from = lambda t: t

    class _Reg:
        @staticmethod
        def state_intervals():
            return {9: [(0, 80, 46), (4, 80, 47)],          # a hypothesis born while the first was still seen
                    7: [(0, 90, 30), (50, None, 60)],       # a successor born after the first was last seen
                    3: [(0, None, 20), (25, None, 30)]}     # both open: nothing ended
    lay.registry = _Reg()
    assert lay.state_intervals() == {9: [(0, 80), (4, 80)], 7: [(0, 50), (50, None)], 3: [(0, None), (25, None)]}
