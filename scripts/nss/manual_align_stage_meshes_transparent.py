#!/usr/bin/env python3
"""Pairwise NSS mesh alignment with per-mesh transparency."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering

from manual_align_stage_meshes import axis_rotation, translation


class TransparentPairAligner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.fixed = int(args.pair[0])
        self.moving = int(args.pair[1])
        self.transform = np.eye(4)
        if args.initial_transform and args.initial_transform.exists():
            self.transform = np.loadtxt(args.initial_transform).reshape(4, 4)
        self.translation_step = args.translation_step
        self.rotation_step_deg = args.rotation_step_deg
        self.fixed_transparent = False
        self.visibility_mode = 0

        self.meshes = {
            self.fixed: self.load_proxy(self.fixed),
            self.moving: self.load_proxy(self.moving),
        }
        self.local_moving_center = self.meshes[self.moving].get_axis_aligned_bounding_box().get_center()

        app = gui.Application.instance
        app.initialize()
        self.window = app.create_window(
            f"NSS Stage{self.fixed} fixed | Stage{self.moving} moving | O: transparency",
            args.width,
            args.height,
        )
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.window.renderer)
        self.widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.window.add_child(self.widget)
        self.window.set_on_layout(self.on_layout)
        self.widget.set_on_key(self.on_key)

        self.fixed_name = f"stage{self.fixed}"
        self.moving_name = f"stage{self.moving}"
        self.widget.scene.add_geometry(
            self.fixed_name, self.meshes[self.fixed], self.make_material(1.0)
        )
        self.widget.scene.add_geometry(
            self.moving_name, self.meshes[self.moving], self.make_material(1.0)
        )
        self.widget.scene.set_geometry_transform(self.moving_name, self.transform)
        self.reset_topdown()
        self.print_help()
        self.print_state()

    def load_proxy(self, stage: int) -> o3d.geometry.TriangleMesh:
        path = self.args.proxy_dir / f"Bldg{self.args.building}_Stage{stage}_textured_proxy.ply"
        mesh = o3d.io.read_triangle_mesh(str(path))
        if not mesh.has_triangles():
            raise RuntimeError(f"Missing or empty proxy: {path}")
        mesh.compute_vertex_normals()
        print(
            f"LOADED Stage{stage}: {path} vertices={len(mesh.vertices)} "
            f"triangles={len(mesh.triangles)}",
            flush=True,
        )
        return mesh

    @staticmethod
    def make_material(alpha: float) -> rendering.MaterialRecord:
        material = rendering.MaterialRecord()
        material.shader = "defaultLit" if alpha >= 0.999 else "defaultLitTransparency"
        material.base_color = [1.0, 1.0, 1.0, alpha]
        material.has_alpha = alpha < 0.999
        return material

    def on_layout(self, _context) -> None:
        self.widget.frame = self.window.content_rect

    def moving_world_center(self) -> np.ndarray:
        point = np.append(self.local_moving_center, 1.0)
        return (self.transform @ point)[:3]

    def apply_delta(self, delta: np.ndarray) -> None:
        self.transform = delta @ self.transform
        self.widget.scene.set_geometry_transform(self.moving_name, self.transform)
        self.widget.force_redraw()
        self.print_state()

    def move(self, dx: float, dy: float, dz: float) -> None:
        self.apply_delta(translation(dx, dy, dz))

    def rotate_yaw(self, direction: float) -> None:
        center = self.moving_world_center()
        rotation = axis_rotation("z", np.deg2rad(direction * self.rotation_step_deg))
        delta = translation(*center) @ rotation @ translation(*(-center))
        self.apply_delta(delta)

    def toggle_fixed_transparency(self) -> None:
        self.fixed_transparent = not self.fixed_transparent
        alpha = self.args.fixed_alpha if self.fixed_transparent else 1.0
        self.widget.scene.modify_geometry_material(self.fixed_name, self.make_material(alpha))
        self.widget.force_redraw()
        print(f"FIXED_STAGE{self.fixed}_OPACITY {alpha:.2f}", flush=True)

    def cycle_visibility(self) -> None:
        self.visibility_mode = (self.visibility_mode + 1) % 3
        fixed_visible = self.visibility_mode in (0, 1)
        moving_visible = self.visibility_mode in (0, 2)
        self.widget.scene.show_geometry(self.fixed_name, fixed_visible)
        self.widget.scene.show_geometry(self.moving_name, moving_visible)
        labels = ["BOTH", f"STAGE{self.fixed}_ONLY", f"STAGE{self.moving}_ONLY"]
        print(f"VISIBILITY {labels[self.visibility_mode]}", flush=True)

    def reset_topdown(self) -> None:
        fixed_box = self.meshes[self.fixed].get_axis_aligned_bounding_box()
        moving_box = self.meshes[self.moving].get_axis_aligned_bounding_box()
        moving_points = np.asarray(moving_box.get_box_points())
        moving_world = (self.transform[:3, :3] @ moving_points.T).T + self.transform[:3, 3]
        all_points = np.vstack([np.asarray(fixed_box.get_box_points()), moving_world])
        minimum, maximum = all_points.min(axis=0), all_points.max(axis=0)
        center = (minimum + maximum) / 2.0
        extent = float(np.max(maximum - minimum))
        bounds = o3d.geometry.AxisAlignedBoundingBox(minimum, maximum)
        self.widget.setup_camera(60.0, bounds, center)
        self.widget.scene.camera.look_at(center, center + [0.0, 0.0, extent * 1.5], [0.0, 1.0, 0.0])
        self.widget.force_redraw()
        print("VIEW TOPDOWN_RESET", flush=True)

    def save(self) -> None:
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        identity_path = self.args.output_dir / f"stage{self.fixed}_to_stage{self.fixed}.txt"
        pair_path = self.args.output_dir / f"stage{self.moving}_to_stage{self.fixed}.txt"
        np.savetxt(identity_path, np.eye(4), fmt="%.12f")
        np.savetxt(pair_path, self.transform, fmt="%.12f")
        print(f"SAVED {pair_path}", flush=True)
        if self.fixed != 1:
            fixed_to_stage1_path = self.args.output_dir / f"stage{self.fixed}_to_stage1.txt"
            if fixed_to_stage1_path.exists():
                fixed_to_stage1 = np.loadtxt(fixed_to_stage1_path).reshape(4, 4)
                composed_path = self.args.output_dir / f"stage{self.moving}_to_stage1.txt"
                np.savetxt(composed_path, fixed_to_stage1 @ self.transform, fmt="%.12f")
                print(f"SAVED_COMPOSED {composed_path}", flush=True)

    def print_state(self) -> None:
        yaw = np.rad2deg(np.arctan2(self.transform[1, 0], self.transform[0, 0]))
        xyz = self.transform[:3, 3]
        print(
            f"Stage{self.moving}: translation=[{xyz[0]:.3f}, {xyz[1]:.3f}, "
            f"{xyz[2]:.3f}] yaw={yaw:.2f}deg",
            flush=True,
        )

    def print_help(self) -> None:
        print(
            f"""
Stage{self.fixed} is fixed; only Stage{self.moving} moves.
  O                 toggle fixed Stage opacity 100% / {self.args.fixed_alpha * 100:.0f}%
  A / D or Q / E    rotate moving mesh yaw left/right
  Arrow keys        move moving mesh X/Y
  PageUp/PageDown   move moving mesh Z
  [ / ]             halve/double translation step
  , / .             halve/double rotation step
  V                 both / fixed only / moving only
  T                 reset top-down camera
  P                 save transform
""",
            flush=True,
        )

    def on_key(self, event: gui.KeyEvent):
        if event.type != gui.KeyEvent.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        key = event.key
        if key == gui.KeyName.O:
            self.toggle_fixed_transparency()
        elif key in (gui.KeyName.A, gui.KeyName.Q):
            self.rotate_yaw(1.0)
        elif key in (gui.KeyName.D, gui.KeyName.E):
            self.rotate_yaw(-1.0)
        elif key == gui.KeyName.LEFT:
            self.move(-self.translation_step, 0.0, 0.0)
        elif key == gui.KeyName.RIGHT:
            self.move(self.translation_step, 0.0, 0.0)
        elif key == gui.KeyName.UP:
            self.move(0.0, self.translation_step, 0.0)
        elif key == gui.KeyName.DOWN:
            self.move(0.0, -self.translation_step, 0.0)
        elif key == gui.KeyName.PAGE_UP:
            self.move(0.0, 0.0, self.translation_step)
        elif key == gui.KeyName.PAGE_DOWN:
            self.move(0.0, 0.0, -self.translation_step)
        elif key == gui.KeyName.LEFT_BRACKET:
            self.translation_step = max(0.001, self.translation_step * 0.5)
            print(f"TRANSLATION_STEP {self.translation_step:.4f}m", flush=True)
        elif key == gui.KeyName.RIGHT_BRACKET:
            self.translation_step = min(10.0, self.translation_step * 2.0)
            print(f"TRANSLATION_STEP {self.translation_step:.4f}m", flush=True)
        elif key == gui.KeyName.COMMA:
            self.rotation_step_deg = max(0.05, self.rotation_step_deg * 0.5)
            print(f"ROTATION_STEP {self.rotation_step_deg:.3f}deg", flush=True)
        elif key == gui.KeyName.PERIOD:
            self.rotation_step_deg = min(45.0, self.rotation_step_deg * 2.0)
            print(f"ROTATION_STEP {self.rotation_step_deg:.3f}deg", flush=True)
        elif key == gui.KeyName.V:
            self.cycle_visibility()
        elif key == gui.KeyName.T:
            self.reset_topdown()
        elif key == gui.KeyName.P:
            self.save()
        else:
            return gui.Widget.EventCallbackResult.IGNORED
        return gui.Widget.EventCallbackResult.HANDLED

    def run(self) -> None:
        gui.Application.instance.run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--building", type=int, default=1)
    parser.add_argument("--pair", choices=["12", "13", "23"], required=True)
    parser.add_argument("--proxy-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-transform", type=Path)
    parser.add_argument("--translation-step", type=float, default=0.25)
    parser.add_argument("--rotation-step-deg", type=float, default=0.5)
    parser.add_argument("--fixed-alpha", type=float, default=0.35)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1200)
    args = parser.parse_args()
    TransparentPairAligner(args).run()


if __name__ == "__main__":
    main()
