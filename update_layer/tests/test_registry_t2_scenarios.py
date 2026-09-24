"""t2 C++ scenarios (khronos/tests/test_temporal_fragment_states.cpp @43c663d) on the Python port.

Each test feeds the same segments and evidence as the C++ test and checks the same outcome.
C++ meshes are in a bounding-box frame centred at `center`; here points are world = local + center.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from update_layer.evidence import SurfaceEvidence  # noqa: E402
from update_layer.registry import Observation, PersistentObjectState  # noqa: E402

S = 1_000_000_000


def seg(identity, first, last, local, center=(0, 0, 0), moved=False):
    pts = np.asarray(local, dtype=np.float64) + np.asarray(center, dtype=np.float64)
    return Observation(identity=identity, points=pts, normals=None, first=first, last=last, moved=moved)


def world(frag):
    return frag.points.detach().cpu().numpy()


FRONT = [(-0.40, 0, 0), (-0.39, 0, 0)]
BACK = [(0.40, 0, 0), (0.39, 0, 0)]


def test_A_disjoint_observation_stays_unresolved():
    r = PersistentObjectState()
    r.ingest(seg(701, 1 * S, 1 * S, FRONT))
    r.ingest(seg(701, 3 * S, 3 * S, BACK))
    cur = r.current_fragment(701)
    assert cur.num_vertices == 2
    assert np.allclose(world(cur), FRONT)
    assert len(r.history(701)) == 1
    assert len(r.unresolved_candidates(701)) == 1


def test_A_prime_confirmed_current_absorbs_disjoint_view():
    r = PersistentObjectState()
    r.ingest(seg(702, 1 * S, 1 * S, FRONT))
    r.ingest(seg(702, 3 * S, 3 * S, BACK))
    assert r.report_current_supported(702, 3 * S)
    cur = r.current_fragment(702)
    assert cur.num_vertices == 4
    assert len(r.history(702)) == 1
    assert not r.unresolved_candidates(702)


OLD = [(0, 0, 0), (0.01, 0, 0)]
NEW = [(0, 0, 0), (0.02, 0, 0), (0.03, 0, 0)]


def test_E_watched_motion_opens_new_state():
    r = PersistentObjectState()
    r.ingest(seg(703, 1 * S, 1 * S, OLD))
    r.ingest(seg(703, 4 * S, 4 * S, NEW, center=(5, 0, 0), moved=True))
    h = r.history(703)
    assert len(h) == 2 and h[0].death_time is not None and h[1].death_time is None
    assert np.allclose(world(r.current_fragment(703)), np.asarray(NEW) + [5, 0, 0])


def relocation(identity, contradiction_first):
    r = PersistentObjectState()
    r.ingest(seg(identity, 1 * S, 1 * S, OLD))
    if contradiction_first:
        r.report_current_contradicted(identity, 8 * S)
        r.ingest(seg(identity, 9 * S, 9 * S, NEW, center=(5, 0, 0)))
    else:
        r.ingest(seg(identity, 9 * S, 9 * S, NEW, center=(5, 0, 0)))
        r.report_current_contradicted(identity, 8 * S)
    h = r.history(identity)
    return r, h


def test_FGJ_relocation_is_order_invariant_and_provenance_clean():
    outs = []
    for identity, first in ((704, True), (705, False)):
        r, h = relocation(identity, first)
        assert len(h) == 2
        assert h[0].death_time is not None
        assert not r.unresolved_candidates(identity)
        cur = world(r.current_fragment(identity))
        assert np.allclose(cur, np.asarray(NEW) + [5, 0, 0])
        assert np.allclose(world(h[0]), OLD)
        outs.append(cur)
    assert np.allclose(outs[0], outs[1])


def look(support, seen_through):
    return SurfaceEvidence(surface_samples=10, support_rays=support, reliable_in_view=10,
                           reliable_seen_through=seen_through)


OLD_SITE = [(0.00, 0, 0), (0.01, 0, 0)]
MOVED = [(0.02, 0, 0), (0.60, 0, 0), (0.61, 0, 0)]


def overlap(identity, support, seen):
    r = PersistentObjectState()
    r.ingest(seg(identity, 1 * S, 2 * S, OLD_SITE))
    r.resolve_current_evidence(identity, look(support, seen), SurfaceEvidence(), 6 * S)
    r.ingest(seg(identity, 5 * S, 7 * S, MOVED))
    return r.current_fragment(identity).num_vertices, len(r.unresolved_candidates(identity))


def test_KKK_shared_space_merges_unless_observed_empty():
    assert overlap(801, 0, 10) == (2, 1)      # K
    assert overlap(802, 5, 0) == (5, 0)       # K'
    assert overlap(804, 0, 4) == (5, 0)       # K''


def test_L_absorb_requires_support():
    r = PersistentObjectState()
    r.ingest(seg(803, 1 * S, 2 * S, OLD_SITE))
    r.resolve_current_evidence(803, look(0, 10), SurfaceEvidence(), 6 * S)
    r.ingest(seg(803, 5 * S, 7 * S, MOVED))
    assert len(r.unresolved_candidates(803)) == 1
    r.resolve_current_evidence(803, look(0, 0), SurfaceEvidence(), 8 * S)
    assert len(r.unresolved_candidates(803)) == 1
    assert r.current_fragment(803).num_vertices == 2
    r.resolve_current_evidence(803, look(4, 0), SurfaceEvidence(), 9 * S)
    assert not r.unresolved_candidates(803)
    assert r.current_fragment(803).num_vertices == 5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
