#!/usr/bin/env python
"""What is in the far-bathroom box of a synthetic B map, and where did it come from? (CPU, A and B checkpoints, nothing trained.)
Box = records/probe_matrix_20261010.py's far-bathroom box (x 11.0..16.2, y -5.4..-2.0 m). Rows alive at B's end, split by origin (created before B = A-born, else B-born) and by distance to the
exact GT surface of B's last frame (<= 5 cm = on the surface). For A-born rows: displacement since A's end, whether the row was ended (pruned / layer) in B, and, for the rows that were ON the surface
at A's end, what happened to them (still on / moved off / ended).
  python records/probe_bathroom_rows_20261013.py RUN_DIR ...
"""
import sys
from pathlib import Path
import numpy as np, open3d as o3d, torch
from scipy.spatial import cKDTree
INT = np.iinfo(np.int64).max
D0 = "/home/jixian/Desktop/FT/datasets/synthetic_ab"
gb = np.load(f"{D0}/session_b/gt/temporal_gt.npz"); ids_b = list(gb["instance_ids"]); lbm = np.load(f"{D0}/gt_geometry/b_local_meshes.npz")
Tb1 = gb["T_world_object"][-1]; vis = gb["render_visible"][-1]; pts = []
for j, i in enumerate(ids_b):
    if f"{i:04d}_vertices" not in lbm.files or not vis[j]: continue
    V = lbm[f"{i:04d}_vertices"].astype(np.float64); F = lbm[f"{i:04d}_faces"].astype(np.int32)
    V = (np.c_[V, np.ones(len(V))] @ Tb1[j].T)[:, :3]
    mm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    pts.append(np.asarray(mm.sample_points_uniformly(number_of_points=max(50, int(mm.get_surface_area() * 4000))).points))
tree = cKDTree(np.vstack(pts))
X0, X1, Y0, Y1 = 11.0, 16.2, -5.4, -2.0
for run in sys.argv[1:]:
    run = Path(run)
    a = torch.load(run / "checkpoint_a.pt", weights_only=False, mmap=True, map_location="cpu")["backend"]
    b = torch.load(run / "checkpoint_b.pt", weights_only=False, mmap=True, map_location="cpu")["backend"]
    ss = int(b["session_starts"][-1]) if "session_starts" in b else int(b["session_start"])
    uidA, uidB = a["uid"].numpy(), b["uid"].numpy(); createdB = b["created"].numpy()
    ended = (b["death_prune"].numpy() < INT) | (b["death_evidence"].numpy() < INT) | (b["death_state"].numpy() < INT)
    XB = b["model"][1].detach().numpy().astype(np.float32) / 10.0
    srt = np.argsort(uidA); pos = np.clip(np.searchsorted(uidA[srt], uidB), 0, len(uidA) - 1); inA = uidA[srt][pos] == uidB
    XA = np.zeros_like(XB); XA[inA] = a["model"][1].detach().numpy()[srt[pos[inA]]].astype(np.float32) / 10.0
    inbox_B = (XB[:, 0] >= X0) & (XB[:, 0] <= X1) & (XB[:, 1] >= Y0) & (XB[:, 1] <= Y1)
    inbox_A = inA & (XA[:, 0] >= X0) & (XA[:, 0] <= X1) & (XA[:, 1] >= Y0) & (XA[:, 1] <= Y1)
    dB = np.full(len(XB), 9.0); idx = np.nonzero(inbox_B | inbox_A)[0]
    dB[idx], _ = tree.query(XB[idx], k=1, distance_upper_bound=1.0)
    dA = np.full(len(XB), 9.0); ia = np.nonzero(inbox_A)[0]
    dA[ia], _ = tree.query(XA[ia], k=1, distance_upper_bound=1.0)
    aborn = createdB < ss
    alive = ~ended
    print(f"\n=== {run.name}: far-bathroom box")
    print(f"  alive rows now in the box: {int((alive & inbox_B).sum())}   A-born {int((alive & inbox_B & aborn).sum())}   B-born {int((alive & inbox_B & ~aborn).sum())}")
    for nm, m in (("A-born", alive & inbox_B & aborn), ("B-born", alive & inbox_B & ~aborn)):
        if m.any(): print(f"    {nm}: off the GT surface (> 5 cm) {100*(dB[m] > 0.05).mean():5.1f} %  ({int((dB[m] > 0.05).sum())} rows)")
    s = aborn & inbox_A & inA                     # A-born rows that were inside the box at A's end
    on_A = s & (dA <= 0.05)
    mv = np.linalg.norm(XB - XA, axis=1)
    print(f"  A-born rows in the box at A's end: {int(s.sum())}; on the GT surface then: {int(on_A.sum())} ({100*on_A.sum()/max(s.sum(),1):.1f} %)")
    print(f"    of the rows on the surface at A's end: ended in B {100*ended[on_A].mean():5.1f} %, alive & still on the surface {100*(alive[on_A] & (dB[on_A] <= 0.05)).mean():5.1f} %, alive & moved off (> 5 cm) {100*(alive[on_A] & (dB[on_A] > 0.05)).mean():5.1f} %")
    al = on_A & alive
    LOOPEND = 1
    print(f"    alive ones: median displacement {100*np.median(mv[al]):.1f} cm, > 5 cm {100*(mv[al] > 0.05).mean():.1f} %, > 20 cm {100*(mv[al] > 0.20).mean():.1f} %, left the box {100*(~inbox_B[al]).mean():.1f} %")
    # cause of the ends of the on-surface bathroom rows
    dp, de, ds_ = b["death_prune"].numpy(), b["death_evidence"].numpy(), b["death_state"].numpy()
    idn = b["identity"].numpy()
    e = on_A & ended
    first = np.minimum(np.minimum(dp, de), ds_)
    cause = np.where(first == dp, 0, np.where(first == de, 1, 2))
    print(f"    ended rows ({int(e.sum())}): GaME opacity prune {100*(cause[e]==0).mean():.1f} %, layer evidence {100*(cause[e]==1).mean():.1f} %, object-state end {100*(cause[e]==2).mean():.1f} %;"
          f" object-labelled {100*(idn[e] > 0).mean():.1f} %, background {100*(idn[e] <= 0).mean():.1f} %")
    t = (first[e] - ss) / 1e9
    import os
    for cut in (45, 70, 100):
        tt = (np.minimum(np.minimum(dp, de), ds_)[on_A] - ss) / 1e9
        print(f"    ENDED_BY {cut} s after B's start: {100*np.mean((tt <= cut) & ended[on_A]):.1f} % of the on-surface bathroom rows")
    print(f"    time of the end after B's start: median {np.median(t):.0f} s, 10-90 % {np.percentile(t,10):.0f}..{np.percentile(t,90):.0f} s")

print("exit 0")
