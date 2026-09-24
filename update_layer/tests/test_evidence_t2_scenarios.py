"""t2 evidence semantics on the Python port.

1. Pixel coding of PhysicalEvidenceStore::ingest (khronos/tests/test_physical_identity_evidence.cpp,
   testRleProjectionAndCopyOnWrite @43c663d): 4x2 camera fx = fy = cx = cy = 1, range 1 everywhere;
   row 0 = background, background, I7, I7; row 1 = semantic object without id, dynamic pixel,
   zero range, NaN range.
2. classifyMeasurement vote table (projected_physical_evidence.cpp:18-44).
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402
from update_layer.evidence import (BACKGROUND, DEV, INVALID, PHYSICAL, UNAVAILABLE, UNIDENTIFIED,  # noqa: E402
                                   EvidenceStore, Vote, classify)
from update_layer.frames import Frame, Intrinsics  # noqa: E402

K = Intrinsics(4, 2, 1.0, 1.0, 1.0, 1.0)
T1, T2 = 10_000_000_000, 20_000_000_000
OBJECT_SEMANTIC = 42


def pixel_point(u, v, depth=1.0):
    return torch.tensor([[(u - 1.0) * depth, (v - 1.0) * depth, depth]], dtype=torch.float32, device=DEV)


def typed_frame(stamp):
    rng = np.ones((2, 4), np.float32)
    rng[1, 2] = 0.0
    rng[1, 3] = np.nan
    # the store receives optical-Z depth; range = depth * |(x/z, y/z, 1)|
    u = np.arange(4) - 1.0
    v = np.arange(2) - 1.0
    scale = np.sqrt(u[None, :] ** 2 + v[:, None] ** 2 + 1)
    depth = rng / scale
    semantic = np.ones((2, 4), np.int32)
    semantic[1, 0] = OBJECT_SEMANTIC
    instance = np.zeros((2, 4), np.int32)
    instance[0, 2] = instance[0, 3] = 7
    dynamic = np.zeros((2, 4), bool)
    dynamic[1, 1] = True
    return Frame(0, stamp, depth.astype(np.float32), instance, semantic, np.eye(4), K), dynamic


def test_pixel_coding():
    store = EvidenceStore(object_semantics=[OBJECT_SEMANTIC], max_range=10.0)
    frame, dynamic = typed_frame(T1)
    store.ingest(frame, dynamic)
    expect = [((0, 0), BACKGROUND, 0), ((2, 0), PHYSICAL, 7), ((0, 1), UNIDENTIFIED, 0),
              ((1, 1), UNIDENTIFIED, 0), ((2, 1), INVALID, 0), ((3, 1), INVALID, 0)]
    for (u, v), etype, pid in expect:
        p = store.project(0, 1, pixel_point(u, v))
        assert int(p["etype"][0, 0]) == etype and int(p["pid"][0, 0]) == pid, ((u, v), p["etype"], p["pid"])
    assert store.stamps(0, T2) == [T1]
    p = store.project(0, 1, torch.tensor([[100.0, 0.0, 1.0]], device=DEV))
    assert int(p["etype"][0, 0]) == UNAVAILABLE


def proj(etype, pid, measured, query):
    t = lambda x, d: torch.tensor([[x]], dtype=d, device=DEV)
    return dict(etype=t(etype, torch.int8), pid=t(pid, torch.int64), measured=t(measured, torch.float32),
                query=t(query, torch.float32))


def test_vote_table():
    tol = 0.3
    cases = [
        (proj(PHYSICAL, 7, 1.0, 1.0), Vote.SUPPORTED),       # same identity on the surface
        (proj(PHYSICAL, 9, 1.0, 1.0), Vote.OTHER),           # another identity, same depth
        (proj(BACKGROUND, 0, 1.0, 1.0), Vote.BACKGROUND),    # background at the surface
        (proj(UNIDENTIFIED, 0, 1.0, 1.0), Vote.UNIDENTIFIED),
        (proj(BACKGROUND, 0, 2.0, 1.0), Vote.FREE),          # measured beyond the surface
        (proj(BACKGROUND, 0, 0.9, 1.0), Vote.OCCLUDED),      # a nearer surface
        (proj(PHYSICAL, 7, 0.9, 1.0), Vote.SUPPORTED),       # own identity within tolerance
        (proj(PHYSICAL, 7, 0.6, 1.0), Vote.OCCLUDED),        # own identity beyond tolerance, nearer
        (proj(INVALID, 0, math.nan, 1.0), Vote.INVALID),
        (proj(UNAVAILABLE, 0, math.nan, 1.0), Vote.UNAVAILABLE),
        (proj(BACKGROUND, 0, 1.0 - 2e-3, 1.0), Vote.OCCLUDED),  # different id nearer by > 1 mm
    ]
    for p, want in cases:
        got = int(classify(p, 7, tol)[0, 0])
        assert got == want, (p, got, want)


if __name__ == "__main__":
    test_pixel_coding()
    print("PASS test_pixel_coding")
    test_vote_table()
    print("PASS test_vote_table")
