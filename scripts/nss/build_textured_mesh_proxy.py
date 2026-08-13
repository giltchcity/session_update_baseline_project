#!/usr/bin/env python3
"""Bake OBJ textures to vertex colors and decimate into a light triangle mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def bake_texture_colors(mesh: o3d.geometry.TriangleMesh) -> None:
    triangles = np.asarray(mesh.triangles)
    triangle_uvs = np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)
    material_ids = np.asarray(mesh.triangle_material_ids)
    if len(triangle_uvs) != len(triangles) or not mesh.textures:
        raise RuntimeError("OBJ has no usable triangle UVs/textures")

    color_sum = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
    color_count = np.zeros(len(mesh.vertices), dtype=np.int32)
    for material_id in np.unique(material_ids):
        if material_id < 0 or material_id >= len(mesh.textures):
            continue
        triangle_indices = np.flatnonzero(material_ids == material_id)
        if not len(triangle_indices):
            continue
        image = np.asarray(mesh.textures[int(material_id)])
        if image.size == 0:
            continue
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        if image.shape[2] < 3:
            image = np.repeat(image[..., :1], 3, axis=2)
        image = image[..., :3].astype(np.float64) / 255.0
        height, width = image.shape[:2]
        uvs = triangle_uvs[triangle_indices].reshape(-1, 2)
        x = np.clip(np.rint(uvs[:, 0] * (width - 1)), 0, width - 1).astype(np.int64)
        y = np.clip(np.rint((1.0 - uvs[:, 1]) * (height - 1)), 0, height - 1).astype(np.int64)
        colors = image[y, x]
        vertices = triangles[triangle_indices].reshape(-1)
        np.add.at(color_sum, vertices, colors)
        np.add.at(color_count, vertices, 1)

    valid = color_count > 0
    vertex_colors = np.full((len(mesh.vertices), 3), 0.65, dtype=np.float64)
    vertex_colors[valid] = color_sum[valid] / color_count[valid, None]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # The proxy now carries appearance in vertex colors, so heavy texture images
    # and per-corner UV data can be released before decimation.
    mesh.triangle_uvs = o3d.utility.Vector2dVector()
    mesh.triangle_material_ids = o3d.utility.IntVector()
    mesh.textures = []


def build_proxy(source: Path, output: Path, target_triangles: int) -> None:
    print(f"LOADING_TEXTURED_OBJ {source}", flush=True)
    mesh = o3d.io.read_triangle_mesh(str(source), enable_post_processing=False)
    if not mesh.has_triangles():
        raise RuntimeError(f"No triangles loaded from {source}")
    print(
        f"LOADED vertices={len(mesh.vertices)} triangles={len(mesh.triangles)} "
        f"textures={len(mesh.textures)}",
        flush=True,
    )
    bake_texture_colors(mesh)
    if len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output), mesh, write_ascii=False):
        raise RuntimeError(f"Failed to write {output}")
    print(
        f"SAVED_PROXY {output} vertices={len(mesh.vertices)} "
        f"triangles={len(mesh.triangles)} colors={mesh.has_vertex_colors()}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-triangles", type=int, default=120000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        print(f"CACHE_EXISTS {args.output}")
        return
    build_proxy(args.source, args.output, args.target_triangles)


if __name__ == "__main__":
    main()
