#!/usr/bin/env python3
"""Display OBJ triangle meshes in their stored coordinates without textures."""

from __future__ import annotations

import argparse
from array import array
from pathlib import Path
import time

import numpy as np
import open3d as o3d


COLORS = [
    [0.90, 0.12, 0.10],
    [0.05, 0.75, 0.90],
    [0.20, 0.78, 0.25],
    [0.92, 0.65, 0.08],
    [0.62, 0.25, 0.85],
    [0.95, 0.35, 0.65],
]


def read_obj_geometry(path: Path) -> o3d.geometry.TriangleMesh:
    """Load exact OBJ vertex/face geometry while deliberately skipping textures."""
    vertices = array("d")
    triangles = array("i")
    vertex_count = 0

    with path.open("r", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                values = line.split()
                vertices.extend((float(values[1]), float(values[2]), float(values[3])))
                vertex_count += 1
            elif line.startswith("f "):
                indices = []
                for token in line.split()[1:]:
                    index = int(token.split("/", 1)[0])
                    indices.append(index - 1 if index > 0 else vertex_count + index)
                for i in range(1, len(indices) - 1):
                    triangles.extend((indices[0], indices[i], indices[i + 1]))

    vertex_array = np.frombuffer(vertices, dtype=np.float64).reshape(-1, 3)
    triangle_array = np.frombuffer(triangles, dtype=np.int32).reshape(-1, 3)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertex_array),
        o3d.utility.Vector3iVector(triangle_array),
    )
    mesh.compute_triangle_normals()
    print(f"{path}: vertices={len(vertex_array)} triangles={len(triangle_array)}")
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", action="append", required=True)
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--transform", action="append", default=[])
    parser.add_argument("--topdown", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--hide-window", action="store_true")
    parser.add_argument("--exit-after-screenshot", action="store_true")
    args = parser.parse_args()

    if args.transform and len(args.transform) != len(args.mesh):
        raise SystemExit("--transform must be omitted or supplied once per --mesh")

    meshes = []
    for i, mesh_path in enumerate(args.mesh):
        mesh = read_obj_geometry(Path(mesh_path))
        if args.transform:
            transform = np.loadtxt(args.transform[i], dtype=np.float64).reshape(4, 4)
            mesh.transform(transform)
            print(f"Applied transform: {args.transform[i]}")
        mesh.paint_uniform_color(COLORS[i % len(COLORS)])
        meshes.append(mesh)

    points = np.vstack(
        [
            np.asarray(mesh.get_axis_aligned_bounding_box().get_box_points())
            for mesh in meshes
        ]
    )
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0

    labels = [args.name[i] if i < len(args.name) else f"mesh{i + 1}" for i in range(len(meshes))]
    for i, label in enumerate(labels):
        print(f"COLOR {COLORS[i % len(COLORS)]}: {label}")

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(
        window_name=" | ".join(labels),
        width=args.width,
        height=args.height,
        visible=not args.hide_window,
    )
    for mesh in meshes:
        visualizer.add_geometry(mesh)

    view = visualizer.get_view_control()
    if args.topdown:
        view.set_lookat(center)
        view.set_front([0.0, 0.0, -1.0])
        view.set_up([0.0, 1.0, 0.0])
        view.set_zoom(0.72)
    render = visualizer.get_render_option()
    render.background_color = np.asarray([1.0, 1.0, 1.0])
    render.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
    for _ in range(20):
        visualizer.poll_events()
        visualizer.update_renderer()
        time.sleep(0.05)

    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        visualizer.capture_screen_image(str(args.screenshot), do_render=True)
        print(f"SAVED_SCREENSHOT {args.screenshot}")

    if not args.hide_window and not args.exit_after_screenshot:
        visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()
