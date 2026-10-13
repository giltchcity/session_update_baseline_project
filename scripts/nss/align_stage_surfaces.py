#!/usr/bin/env python3
"""Align NSS point-cloud fragments and merge them into stage-wise surfaces.

This is a lightweight adapter/probing tool for NSS. It does not depend on
Open3D: NSS PLY files contain only binary x/y/z doubles, so numpy is enough.
The output PLYs are merged surface point clouds, not watertight meshes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


STAGE_COLORS = {
    1: (230, 57, 70),
    2: (42, 157, 143),
    3: (69, 123, 157),
    4: (244, 162, 97),
    5: (131, 56, 236),
}


def is_zero_transform(transform: np.ndarray) -> bool:
    return np.allclose(transform, 0.0)


def read_xyz_ply(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: reached EOF before end_header")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        fmt = None
        vertex_count = None
        properties = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[:2] == ["format", "binary_little_endian"]:
                fmt = "binary_little_endian"
            elif parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
            elif parts[0] == "element":
                in_vertex = False
            elif in_vertex and parts[0] == "property":
                properties.append((parts[1], parts[2]))

        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: expected binary_little_endian PLY, got {fmt}")
        if vertex_count is None:
            raise ValueError(f"{path}: no vertex count in header")
        if properties[:3] != [("double", "x"), ("double", "y"), ("double", "z")]:
            raise ValueError(f"{path}: expected first properties double x/y/z, got {properties[:3]}")

        dtype_fields = []
        type_map = {
            "double": "<f8",
            "float": "<f4",
            "uchar": "u1",
            "uint8": "u1",
            "int": "<i4",
            "uint": "<u4",
        }
        for ply_type, name in properties:
            if ply_type not in type_map:
                raise ValueError(f"{path}: unsupported PLY property type {ply_type}")
            dtype_fields.append((name, type_map[ply_type]))

        arr = np.fromfile(f, dtype=np.dtype(dtype_fields), count=vertex_count)

    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64, copy=False)


def write_xyzrgb_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64)
    if colors is None:
        colors = np.full((len(points), 3), 200, dtype=np.uint8)
    else:
        colors = np.asarray(colors, dtype=np.uint8)
    if len(points) != len(colors):
        raise ValueError("points and colors length mismatch")

    dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ])
    out = np.empty(len(points), dtype=dtype)
    out["x"] = points[:, 0].astype(np.float32)
    out["y"] = points[:, 1].astype(np.float32)
    out["z"] = points[:, 2].astype(np.float32)
    out["red"] = colors[:, 0]
    out["green"] = colors[:, 1]
    out["blue"] = colors[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment aligned by scripts/nss/align_stage_surfaces.py\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        out.tofile(f)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homog = np.ones((len(points), 4), dtype=np.float64)
    homog[:, :3] = points
    return (transform @ homog.T).T[:, :3]


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0.0 or len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    unique_idx.sort()
    return points[unique_idx], colors[unique_idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--point-cloud-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--voxel-size", type=float, default=0.075,
                        help="Downsample merged outputs; 0 disables downsampling.")
    parser.add_argument("--write-fragments", action="store_true",
                        help="Also write every aligned fragment separately.")
    args = parser.parse_args()

    graph = json.loads(args.graph.read_text())
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_points: dict[int, list[np.ndarray]] = defaultdict(list)
    stage_colors: dict[int, list[np.ndarray]] = defaultdict(list)
    all_points = []
    all_colors = []
    manifest = {
        "graph": str(args.graph),
        "name": graph.get("name"),
        "voxel_size": args.voxel_size,
        "nodes": [],
        "stages": {},
    }

    for node in graph["nodes"]:
        transform = np.asarray(node["global_transform"], dtype=np.float64)
        is_outlier = is_zero_transform(transform)
        entry = {
            "id": node["id"],
            "name": node["name"],
            "stage": node["stage"],
            "spot": node["spot"],
            "points_declared": node.get("points"),
            "outlier_zero_transform": bool(is_outlier),
        }
        if is_outlier:
            manifest["nodes"].append(entry)
            continue

        ply_path = args.point_cloud_dir / node["name"]
        pts = read_xyz_ply(ply_path)
        aligned = transform_points(pts, transform)
        color = np.asarray(STAGE_COLORS.get(node["stage"], (180, 180, 180)), dtype=np.uint8)
        colors = np.repeat(color[None, :], len(aligned), axis=0)

        if args.write_fragments:
            frag_path = out_dir / "fragments" / f"node_{node['id']:03d}_stage_{node['stage']}_{Path(node['name']).stem}.ply"
            write_xyzrgb_ply(frag_path, aligned, colors)
            entry["aligned_fragment"] = str(frag_path)

        entry["points_read"] = int(len(pts))
        entry["points_aligned"] = int(len(aligned))
        manifest["nodes"].append(entry)
        stage_points[node["stage"]].append(aligned)
        stage_colors[node["stage"]].append(colors)
        all_points.append(aligned)
        all_colors.append(colors)

    for stage in sorted(stage_points):
        pts = np.vstack(stage_points[stage])
        cols = np.vstack(stage_colors[stage])
        raw_count = len(pts)
        pts_ds, cols_ds = voxel_downsample(pts, cols, args.voxel_size)
        path = out_dir / f"stage_{stage}_merged.ply"
        write_xyzrgb_ply(path, pts_ds, cols_ds)
        manifest["stages"][str(stage)] = {
            "raw_points": int(raw_count),
            "downsampled_points": int(len(pts_ds)),
            "path": str(path),
        }

    if all_points:
        pts = np.vstack(all_points)
        cols = np.vstack(all_colors)
        raw_count = len(pts)
        pts_ds, cols_ds = voxel_downsample(pts, cols, args.voxel_size)
        all_path = out_dir / "all_stages_colored.ply"
        write_xyzrgb_ply(all_path, pts_ds, cols_ds)
        manifest["all_stages"] = {
            "raw_points": int(raw_count),
            "downsampled_points": int(len(pts_ds)),
            "path": str(all_path),
        }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
