#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("obj", type=Path)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--unlit", action="store_true", help="Use unlit shader; useful for checking texture colors.")
    parser.add_argument("--white", action="store_true", help="Use a white background instead of dark gray.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.obj.exists():
        raise SystemExit(f"Missing OBJ: {args.obj}")

    mesh = o3d.io.read_triangle_model(str(args.obj))
    if not mesh.meshes:
        raise SystemExit(f"Failed to load triangle model: {args.obj}")

    print(f"LOADED_MODEL {args.obj}")
    print(f"mesh_count={len(mesh.meshes)} material_count={len(mesh.materials)}")

    app = gui.Application.instance
    app.initialize()

    window = app.create_window(f"Textured OBJ: {args.obj.name}", args.width, args.height)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    boxes = []
    for i, part in enumerate(mesh.meshes):
        material = mesh.materials[part.material_idx] if 0 <= part.material_idx < len(mesh.materials) else rendering.MaterialRecord()
        if args.unlit:
            material.shader = "defaultUnlit"
        elif not material.shader:
            material.shader = "defaultLit"
        scene.scene.add_geometry(f"mesh_{i}", part.mesh, material)
        boxes.append(part.mesh.get_axis_aligned_bounding_box())

    if not boxes:
        raise SystemExit("Loaded model has no mesh boxes.")
    min_bound = np.min(np.vstack([box.min_bound for box in boxes]), axis=0)
    max_bound = np.max(np.vstack([box.max_bound for box in boxes]), axis=0)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    center = bbox.get_center()
    extent = bbox.get_extent()
    print(f"bbox_min={min_bound.tolist()}")
    print(f"bbox_max={max_bound.tolist()}")
    print(f"bbox_extent={extent.tolist()}")

    bg = [1.0, 1.0, 1.0, 1.0] if args.white else [0.12, 0.12, 0.13, 1.0]
    scene.scene.set_background(bg)
    scene.scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS, [0.0, -1.0, -1.0])
    scene.scene.show_axes(True)
    scene.setup_camera(60.0, bbox, center)

    def on_layout(layout_context: gui.LayoutContext) -> None:
        scene.frame = window.content_rect

    window.set_on_layout(on_layout)

    app.run()


if __name__ == "__main__":
    main()
