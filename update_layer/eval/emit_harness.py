"""Write the real-data harness intermediates from any EvaluationScene source.

The validated harness (results/current_version_20260921/tools/evaluate_real_session.sh) reads
three per-snapshot files produced from a Khronos .4dmap by C++ tools. These functions write
the same files from scene samples, so D2/D3, the cleanup check and the ghost % run through
the unchanged harness Python for every representation:

  snapshots.jsonl     tools/abc_eval_final_baseline/export_timeline.cpp
  cleanup_probes.bin  tools/abc_eval_protocol_v1/src/export_cleanup_binary.cpp
  cleanup_mesh.jsonl  tools/abc_eval_v2/cleanup_mesh.cpp

Differences, all deliberate:
  * cleanup_mesh: the nearest surface *sample* within 10 cm replaces the exact closest point on a
    triangle (samples must be dense; see check_density).
  * cleanup_probes.bin: the background label is that of the nearest downsampled sample. The C++
    exporter indexes the pre-downsample label array with the post-downsample index
    (export_cleanup_binary.cpp: global_labels[gidx] after global_pts = downsample(...)).
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .scene import INVALID_LABEL, EvaluationScene

SceneAt = Callable[[int], EvaluationScene]


def first_per_voxel(points: np.ndarray, voxel: float) -> np.ndarray:
    """Indices of the first point in each voxel, in input order (C++ downsample())."""
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    keys = np.floor(points / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.sort(first)


def snapshot_record(index: int, scene: EvaluationScene) -> dict:
    objs = []
    for o in scene.objects:
        pts = o.points.astype(np.float64)
        n = len(pts)
        stride = max(1, (n + 1499) // 1500)
        row = {"id": o.id, "semantic": o.semantic, "first_observed_ns": o.first_observed_ns,
               "last_observed_ns": o.last_observed_ns, "present": bool(o.present),
               "mesh_vertices": n, "observation_first_ns": o.observation_first_ns,
               "observation_last_ns": o.observation_last_ns,
               "details": {"instance_id": [o.instance_id]}, "centroid": None,
               "surface": pts[::stride].tolist(),
               "trajectory_timestamps": list(o.trajectory_timestamps),
               "trajectory_positions": [list(p) for p in o.trajectory_positions]}
        if n:
            row["centroid"] = pts.mean(axis=0).tolist()
            row["bbox_min"] = pts.min(axis=0).tolist()
            row["bbox_max"] = pts.max(axis=0).tolist()
        objs.append(row)
    return {"index": index, "stamp_ns": int(scene.t_ns), "objects": objs}


def write_snapshots(stamps: Sequence[int], scene_at: SceneAt, out_dir, provenance: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "snapshots.jsonl").open("w") as f:
        for i, t in enumerate(stamps):
            f.write(json.dumps(snapshot_record(i, scene_at(t))) + "\n")
    (out_dir / "COMPLETE.json").write_text(json.dumps(
        {"map": provenance, "snapshots": len(stamps), "provenance": "evaluation_scene"}, indent=2))


def _u32(f, v): f.write(struct.pack("<I", v))
def _u64(f, v): f.write(struct.pack("<Q", v))
def _f32(f, v): f.write(struct.pack("<f", v))


def _string(f, s: str):
    b = s.encode()
    _u32(f, len(b))
    f.write(b)


def write_cleanup_binary(stamps: Sequence[int], scene_at: SceneAt, probes_json, out_path) -> None:
    groups = json.loads(Path(probes_json).read_text())
    with open(out_path, "wb") as f:
        f.write(b"ABCCLN1\0")
        _u32(f, len(stamps))
        for ix, t in enumerate(stamps):
            scene = scene_at(t)
            keep = first_per_voxel(scene.background, 0.03)
            g_pts, g_lab = scene.background[keep], scene.background_label[keep]
            owners, o_pts, o_own = [], [], []
            for o in scene.objects:
                if len(o.points) == 0:
                    continue
                o_own.append(np.full(len(o.points), len(owners), dtype=np.int64))
                o_pts.append(o.points)
                owners.append((o.id, o.semantic))
            if o_pts:
                o_pts = np.concatenate(o_pts)
                o_own = np.concatenate(o_own)
                k = first_per_voxel(o_pts, 0.02)
                o_pts, o_own = o_pts[k], o_own[k]
            g_tree = cKDTree(g_pts) if len(g_pts) else None
            o_tree = cKDTree(o_pts) if len(o_pts) else None
            _u32(f, ix)
            _u64(f, int(t))
            _u32(f, len(groups))
            for g in groups:
                q = np.asarray(g["points"], dtype=np.float32).reshape(-1, 3)
                _string(f, g["event_id"])
                _u32(f, len(q))
                gd = np.full(len(q), math.inf, np.float32)
                gl = np.full(len(q), INVALID_LABEL, np.uint32)
                od = np.full(len(q), math.inf, np.float32)
                oi = np.full(len(q), INVALID_LABEL, np.uint32)
                if g_tree is not None and len(q):
                    d, i = g_tree.query(q)
                    gd, gl = d.astype(np.float32), g_lab[i].astype(np.uint32)
                if o_tree is not None and len(q):
                    d, i = o_tree.query(q)
                    od, oi = d.astype(np.float32), o_own[i].astype(np.uint32)
                f.write(np.rec.fromarrays([gd, gl, od, oi], formats="<f4,<u4,<f4,<u4").tobytes())
            _u32(f, len(owners))
            for oid, sem in owners:
                _string(f, oid)
                _u32(f, int(sem) & 0xFFFFFFFF)


def write_cleanup_mesh(stamps: Sequence[int], scene_at: SceneAt, probes_json, out_path,
                       radius: float = 0.10) -> None:
    """Per snapshot and probe: nearest shown sample within `radius` (cleanup_mesh.cpp layout).

    Layer 0 = background (owner "0"), 1 = present object, 2 = displayed non-present object;
    layer 2 competes only in "history_events", layers 0/1 only in "events".
    """
    groups = json.loads(Path(probes_json).read_text())
    probes = [np.asarray(g["points"], dtype=np.float64).reshape(-1, 3) for g in groups]
    limit = radius * radius + 1e-12
    with open(out_path, "w") as f:
        for ix, t in enumerate(stamps):
            scene = scene_at(t)
            layers = [(scene.background.astype(np.float64), 0, "0")]
            layers += [(o.points.astype(np.float64), 1 if o.present else 2, o.id) for o in scene.objects]
            trees = [(cKDTree(p), p, layer, owner) for p, layer, owner in layers if len(p)]
            row = {"index": ix, "stamp_ns": int(t), "faces": 0,
                   "vertices": int(sum(len(p) for p, _, _ in layers)), "invalid_faces": 0,
                   "events": {}, "history_events": {}}
            for g, q in zip(groups, probes):
                out = {}
                for key, want in (("events", (0, 1)), ("history_events", (2,))):
                    best = np.full(len(q), limit)
                    rec: List = [None] * len(q)
                    for tree, pts, layer, owner in trees:
                        if layer not in want or not len(q):
                            continue
                        d, i = tree.query(q, distance_upper_bound=radius)
                        for k in np.flatnonzero(np.isfinite(d)):
                            d2 = float(d[k]) ** 2
                            if d2 < best[k]:
                                best[k] = d2
                                p = pts[i[k]]
                                rec[k] = [math.sqrt(d2), float(p[0]), float(p[1]), float(p[2]),
                                          layer, owner]
                    out[key] = rec
                row["events"][g["event_id"]] = out["events"]
                row["history_events"][g["event_id"]] = out["history_events"]
            f.write(json.dumps(row) + "\n")


def check_density(points: np.ndarray, spacing: float = 0.01) -> float:
    """Median nearest-neighbour spacing of the samples (the nearest-sample error scale)."""
    if len(points) < 2:
        return math.inf
    d, _ = cKDTree(points).query(points, k=2)
    return float(np.median(d[:, 1]))
