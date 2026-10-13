#!/usr/bin/env python3
"""P1/P2/P3 inspection view for the final A and B maps.

Background mesh keeps its original RGB colors; object meshes are colored by
physical instance (fixed palette) so moved/duplicated objects are easy to
spot. Two windows in order: A final, then B final.

Usage:
    conda run -n 3d_vsg python scripts/check_ab_final.py \
        [--mesh-dir runs/ab_final_mesh]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d
from open3d.visualization import gui, rendering


def show_stage(app, title: str, bg_path: Path, obj_path: Path) -> None:
    window = app.create_window(title, 1500, 950)
    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    scene_widget.scene.set_background([0.06, 0.07, 0.10, 1.0])
    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    bg_mesh = o3d.io.read_triangle_mesh(str(bg_path))
    bg_mesh.compute_vertex_normals()
    scene_widget.scene.add_geometry("background", bg_mesh, material)
    bounds = bg_mesh.get_axis_aligned_bounding_box()
    if obj_path.exists():
        obj_mesh = o3d.io.read_triangle_mesh(str(obj_path))
        if not obj_mesh.is_empty():
            obj_mesh.compute_vertex_normals()
            scene_widget.scene.add_geometry("objects", obj_mesh, material)
            bounds += obj_mesh.get_axis_aligned_bounding_box()
    window.add_child(scene_widget)
    scene_widget.setup_camera(60, bounds, bounds.get_center())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mesh-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "runs" / "ab_final_mesh",
    )
    args = ap.parse_args()
    app = gui.Application.instance
    app.initialize()
    print("===== A final (background: original color, objects: instance color) =====")
    show_stage(app, "A final - check P0 (bg original, objects instance color)",
               args.mesh_dir / "A_p0fix_bg.ply",
               args.mesh_dir / "A_p0fix_objects.ply")
    app.run()
    print("===== B final =====")
    show_stage(app, "B final - check P0 (bg original, objects instance color)",
               args.mesh_dir / "B_p0fix_bg.ply",
               args.mesh_dir / "B_p0fix_objects.ply")
    app.run()


if __name__ == "__main__":
    main()
