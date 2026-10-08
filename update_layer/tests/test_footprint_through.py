"""Footprint hit / seen-through test (fork session_refusion.cpp:1150-1206) on the evidence store and in the l2 look.

A thin bar 2 px wide at 2 m in front of a wall at 4 m. A sample displaced 2 px (3.3 cm, within the 5 cm tolerance)
from the bar reads the wall at its own pixel: the single-pixel rule saw through it, the footprint rule hits it. A
sample far from the bar is still seen through; a depth hole or the image border inside the footprint blocks."""
import math
import numpy as np
import torch

from update_layer.frames import Frame, Intrinsics
from update_layer.core.evidence import EvidenceStore, DEV
from update_layer.core.l2 import evidence as l2


def _scene():
    W, H, fx = 128, 96, 120.0
    K = Intrinsics(width=W, height=H, fx=fx, fy=fx, cx=64.0, cy=48.0, offset=0.0)
    depth = np.full((H, W), 4.0, np.float32)
    depth[:, 60:62] = 2.0                         # the bar
    depth[48, 100] = np.nan                       # a depth hole
    frame = Frame(index=0, stamp_ns=10, depth=depth, instance=np.zeros((H, W), np.int32), semantic=None,
                  T_world_cam=np.eye(4), K=K)
    store = EvidenceStore()
    store.ingest(frame)
    def at(u, v, z):                              # the point whose ray is pixel (u, v) at optical depth z
        return [(u - K.cx) / fx * z, (v - K.cy) / fx * z, z]
    pts = torch.tensor([at(60, 48, 2.0),          # on the bar
                        at(63, 48, 2.0),          # 2 px beside the bar: own pixel reads the wall
                        at(80, 48, 2.0),          # nothing there: the wall is seen through it
                        at(101, 48, 2.0),         # a depth hole in the footprint
                        at(1, 48, 2.0)],          # the image border in the footprint
                       dtype=torch.float32, device=DEV)
    return store, pts


def test_store_footprint():
    store, pts = _scene()
    p = store.project(0, 1, pts)
    delta = p["measured"] - p["query"]
    single_through = (delta > 0.05)[0].tolist()
    assert single_through == [False, True, True, True, True]
    fp = store.footprint(0, 1, pts, 0.05, p)
    assert fp["hit"][0].tolist() == [True, True, False, False, False]
    assert fp["through"][0].tolist() == [False, False, True, False, False]


def test_l2_look_uses_footprint():
    store, pts = _scene()
    model = l2.AbsenceModel(store)
    P, N, has_normal, cells = model.absence_queries(pts, None, 0.025)
    st = l2.ObjectAbsenceState()
    rows = st.rows(cells)
    model.classify_frames(st, P, N, has_normal, rows, 7, 0.05, math.cos(math.radians(60)), 10)
    assert st.last_on_surface[rows].tolist() == [10, 10, 0, 0, 0]
    assert st.last_seen_through[rows].tolist() == [0, 0, 10, 0, 0]
