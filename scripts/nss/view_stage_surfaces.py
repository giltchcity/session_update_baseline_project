#!/usr/bin/env python3
"""Interactive Open3D viewer for aligned NSS stage point clouds."""

from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ply",
        action="append",
        type=Path,
        help="PLY file to show. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Directory containing all_stages_colored.ply, stage_*_merged.ply, or stage_*_mesh.ply files.",
    )
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument(
        "--background",
        choices=("dark", "white"),
        default="dark",
        help="Viewer background color. Dark makes untextured/light meshes visible.",
    )
    return parser.parse_args()


def collect_inputs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.ply:
        paths.extend(args.ply)
    if args.dir:
        combined = args.dir / "all_stages_colored.ply"
        if combined.exists():
            paths.append(combined)
        else:
            paths.extend(sorted(args.dir.glob("stage_*_merged.ply")))
            paths.extend(sorted(args.dir.glob("stage_*_mesh.ply")))
    if not paths:
        raise SystemExit("Pass --ply FILE or --dir DIR.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit("Missing PLY file(s): " + ", ".join(missing))
    return paths


def main() -> None:
    args = parse_args()
    geoms = []
    for path in collect_inputs(args):
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            if not mesh.has_vertex_colors():
                mesh.paint_uniform_color([0.72, 0.72, 0.72])
            print(f"LOADED_MESH {path} vertices={len(mesh.vertices)} triangles={len(mesh.triangles)}")
            geoms.append(mesh)
            continue

        cloud = o3d.io.read_point_cloud(str(path))
        if cloud.is_empty():
            print(f"SKIP_EMPTY {path}")
            continue
        print(f"LOADED_CLOUD {path} points={len(cloud.points)}")
        geoms.append(cloud)

    if not geoms:
        raise SystemExit("No non-empty point clouds to display.")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="NSS stage surfaces", width=1400, height=900)
    for geom in geoms:
        vis.add_geometry(geom)
    render = vis.get_render_option()
    render.point_size = args.point_size
    render.background_color = [1.0, 1.0, 1.0] if args.background == "white" else [0.02, 0.02, 0.025]
    render.show_coordinate_frame = True
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
