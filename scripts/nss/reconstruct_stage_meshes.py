#!/usr/bin/env python3
"""Reconstruct diagnostic stage-wise meshes from aligned NSS point clouds.

NSS officially evaluates point-cloud registration, not mesh reconstruction.
The Poisson meshes produced here are only for visual inspection / adapter
prototyping: Poisson may close holes, smooth surfaces, and hallucinate caps.
Use the aligned PLY point clouds for metric claims unless a later experiment
explicitly defines mesh reconstruction as part of the method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=Path,
        help="Directory containing stage_*_merged.ply point clouds.",
    )
    parser.add_argument(
        "--ply",
        action="append",
        type=Path,
        help="Specific stage point cloud. Can be passed multiple times.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory for meshes. Defaults to <dir>/meshes.",
    )
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--normal-radius", type=float, default=0.35)
    parser.add_argument("--normal-max-nn", type=int, default=50)
    parser.add_argument("--orient-k", type=int, default=50)
    parser.add_argument(
        "--density-quantile",
        type=float,
        default=0.03,
        help="Remove Poisson vertices below this density quantile; set 0 to disable.",
    )
    parser.add_argument(
        "--bbox-padding",
        type=float,
        default=0.15,
        help="Crop reconstructed mesh to point-cloud AABB plus this padding.",
    )
    parser.add_argument(
        "--min-component-triangles",
        type=int,
        default=50,
        help="Remove connected triangle components smaller than this.",
    )
    return parser.parse_args()


def collect_inputs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.ply:
        paths.extend(args.ply)
    if args.dir:
        paths.extend(sorted(args.dir.glob("stage_*_merged.ply")))
    if not paths:
        raise SystemExit("Pass --dir DIR or --ply FILE.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit("Missing input PLY(s): " + ", ".join(missing))
    return paths


def output_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        return args.out_dir
    if args.dir:
        return args.dir / "meshes"
    return args.ply[0].parent / "meshes"


def clean_mesh(mesh: o3d.geometry.TriangleMesh, min_triangles: int) -> o3d.geometry.TriangleMesh:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    if min_triangles > 0 and len(mesh.triangles) > 0:
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels = np.asarray(labels)
        counts = np.asarray(counts)
        remove = counts[labels] < min_triangles
        mesh.remove_triangles_by_mask(remove)
        mesh.remove_unreferenced_vertices()

    return mesh


def reconstruct(path: Path, args: argparse.Namespace) -> tuple[o3d.geometry.TriangleMesh, dict]:
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"{path}: empty point cloud")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.normal_radius,
            max_nn=args.normal_max_nn,
        )
    )
    if args.orient_k > 0:
        pcd.orient_normals_consistent_tangent_plane(args.orient_k)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=args.poisson_depth,
    )
    densities_np = np.asarray(densities)

    if args.density_quantile > 0.0 and len(densities_np):
        threshold = float(np.quantile(densities_np, args.density_quantile))
        mesh.remove_vertices_by_mask(densities_np < threshold)
    else:
        threshold = float("nan")

    point_bbox = pcd.get_axis_aligned_bounding_box()
    bbox = point_bbox
    if args.bbox_padding > 0.0:
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=point_bbox.min_bound - args.bbox_padding,
            max_bound=point_bbox.max_bound + args.bbox_padding,
        )
    mesh = mesh.crop(bbox)
    mesh = clean_mesh(mesh, args.min_component_triangles)
    mesh.compute_vertex_normals()

    stats = {
        "input": str(path),
        "input_points": int(len(pcd.points)),
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
        "poisson_depth": args.poisson_depth,
        "normal_radius": args.normal_radius,
        "normal_max_nn": args.normal_max_nn,
        "orient_k": args.orient_k,
        "density_quantile": args.density_quantile,
        "density_threshold": threshold,
        "bbox_padding_m_each_side": args.bbox_padding,
        "point_bbox_min": point_bbox.min_bound.tolist(),
        "point_bbox_max": point_bbox.max_bound.tolist(),
        "bbox_min": bbox.min_bound.tolist(),
        "bbox_max": bbox.max_bound.tolist(),
    }
    return mesh, stats


def main() -> None:
    args = parse_args()
    inputs = collect_inputs(args)
    out_dir = output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "method": "open3d_poisson",
        "official_nss_metric": False,
        "note": (
            "Diagnostic mesh reconstruction only. NSS official evaluation is "
            "point-cloud registration; Poisson can close holes, smooth surfaces, "
            "and hallucinate caps."
        ),
        "outputs": [],
    }
    for path in inputs:
        print(f"RECONSTRUCT {path}")
        mesh, stats = reconstruct(path, args)
        out_path = out_dir / path.name.replace("_merged.ply", "_mesh.ply")
        ok = o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=False)
        if not ok:
            raise RuntimeError(f"failed to write {out_path}")
        stats["mesh"] = str(out_path)
        manifest["outputs"].append(stats)
        print(f"WROTE {out_path} vertices={stats['vertices']} triangles={stats['triangles']}")

    manifest_path = out_dir / "mesh_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"WROTE {manifest_path}")


if __name__ == "__main__":
    main()
