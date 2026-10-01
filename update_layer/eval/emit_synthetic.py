"""Write the synthetic native export (complete_native_export.cpp layout) from any scene source.

The validated synthetic scripts (/mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts, driven by
eval/synthetic_base_v38 and eval_tools_full_20260922) read, per snapshot, meshes.jsonl rows and
world-frame <prefix>.f32 vertices / .u32 faces / .rgb colours. They need triangles: closest-
triangle distances for the state/change tests, area sampling for Mesh F1.

Point samples are written as surfels: one equilateral triangle per sample, centred on it, in
its tangent plane (normal from the representation; +z when unknown), with area = spacing^2 so
that area sampling weights every sample by the surface it stands for. Samples are first
reduced to one per voxel of that spacing, so repeated observations do not add area.

The surfel error is measurable on a TSDF map: score its mesh, then its area samples written as
surfels (eval/scene/parity_surfels.py).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .emit_harness import first_per_voxel
from .scene import EvaluationScene


def surfel_triangles(points: np.ndarray, normals: Optional[np.ndarray], spacing: float):
    """(3N, 3) float32 vertices and (N, 3) uint32 faces; triangle area = spacing^2."""
    n = len(points)
    if n == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32)
    nrm = np.zeros((n, 3))
    nrm[:, 2] = 1.0
    if normals is not None:
        ok = np.isfinite(normals).all(axis=1)
        length = np.linalg.norm(normals, axis=1)
        ok &= length > 1e-9
        nrm[ok] = normals[ok] / length[ok, None]
    a = np.zeros((n, 3))
    a[:, 0] = 1.0
    parallel = np.abs(nrm[:, 0]) > 0.9
    a[parallel] = [0.0, 1.0, 0.0]
    e1 = np.cross(nrm, a)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(nrm, e1)
    radius = math.sqrt(4.0 * spacing * spacing / (3.0 * math.sqrt(3.0)))
    verts = np.empty((n, 3, 3))
    for k, theta in enumerate((math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 + 4 * math.pi / 3)):
        verts[:, k] = points + radius * (math.cos(theta) * e1 + math.sin(theta) * e2)
    faces = np.arange(3 * n, dtype=np.uint32).reshape(n, 3)
    return verts.reshape(-1, 3).astype(np.float32), faces


def _write(out: Path, prefix: str, points, normals, spacing) -> tuple:
    keep = first_per_voxel(points, spacing)
    verts, faces = surfel_triangles(points[keep], None if normals is None else normals[keep], spacing)
    verts.tofile(out / f"{prefix}.f32")
    faces.tofile(out / f"{prefix}.u32")
    np.full((len(verts), 3), 180, np.uint8).tofile(out / f"{prefix}.rgb")
    return len(verts), len(faces)


def write_native(source, out_dir, bg_spacing: float, obj_spacing: float) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "meshes.jsonl").open("w") as meta:
        for ix, t in enumerate(source.stamps()):
            scene: EvaluationScene = source.scene(t)
            row = {"index": ix, "stamp_ns": int(t), "objects": [],
                   "background_prefix": f"{ix}_background"}
            for o in scene.objects:
                prefix = f"{ix}_{o.id}"
                nv, nf = _write(out, prefix, o.points, o.normals, obj_spacing) if len(o.points) else (0, 0)
                row["objects"].append({
                    "id": o.id, "semantic": o.semantic, "present": bool(o.present),
                    "current": nv > 0, "vertices": nv, "faces": nf, "prefix": prefix,
                    "first_observed_ns": o.first_observed_ns, "last_observed_ns": o.last_observed_ns,
                    "trajectory_timestamps": list(o.trajectory_timestamps),
                    "trajectory_positions": [list(p) for p in o.trajectory_positions]})
            _write(out, row["background_prefix"], scene.background, scene.background_normal, bg_spacing)
            meta.write(json.dumps(row) + "\n")
