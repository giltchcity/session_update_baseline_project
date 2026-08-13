#!/usr/bin/env python3
"""Interactively align one NSS OBJ triangle mesh against another stage."""

from __future__ import annotations

import argparse
from array import array
from pathlib import Path

import numpy as np
import open3d as o3d

from view_two_meshes_overlay import COLORS, read_obj_geometry


KEY_LEFT = 263
KEY_RIGHT = 262
KEY_DOWN = 264
KEY_UP = 265
KEY_PAGE_UP = 266
KEY_PAGE_DOWN = 267


def read_single_atlas_obj(path: Path, texture_path: Path) -> o3d.geometry.TriangleMesh:
    """Read a triangular single-atlas OBJ without invoking Assimp."""
    vertices = array("d")
    texture_coordinates: list[tuple[float, float]] = []
    triangles = array("i")
    triangle_uvs = array("d")
    vertex_count = 0
    with path.open("r", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                fields = line.split()
                vertices.extend((float(fields[1]), float(fields[2]), float(fields[3])))
                vertex_count += 1
            elif line.startswith("vt "):
                fields = line.split()
                texture_coordinates.append((float(fields[1]), float(fields[2])))
            elif line.startswith("f "):
                corners = []
                for token in line.split()[1:]:
                    fields = token.split("/")
                    vertex_index = int(fields[0])
                    uv_index = int(fields[1])
                    if vertex_index < 0:
                        vertex_index = vertex_count + vertex_index + 1
                    if uv_index < 0:
                        uv_index = len(texture_coordinates) + uv_index + 1
                    corners.append((vertex_index - 1, texture_coordinates[uv_index - 1]))
                for i in range(1, len(corners) - 1):
                    for corner in (corners[0], corners[i], corners[i + 1]):
                        triangles.append(corner[0])
                        triangle_uvs.extend(corner[1])

    vertex_array = np.frombuffer(vertices, dtype=np.float64).reshape(-1, 3)
    triangle_array = np.frombuffer(triangles, dtype=np.int32).reshape(-1, 3)
    used_vertices = np.unique(triangle_array)
    remap = np.full(len(vertex_array), -1, dtype=np.int32)
    remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int32)
    vertex_array = vertex_array[used_vertices]
    triangle_array = remap[triangle_array]
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertex_array),
        o3d.utility.Vector3iVector(triangle_array),
    )
    mesh.triangle_uvs = o3d.utility.Vector2dVector(
        np.frombuffer(triangle_uvs, dtype=np.float64).reshape(-1, 2)
    )
    mesh.triangle_material_ids = o3d.utility.IntVector(
        np.zeros(len(mesh.triangles), dtype=np.int32)
    )
    mesh.textures = [o3d.io.read_image(str(texture_path))]
    mesh.compute_vertex_normals()
    print(
        f"LOADED_UV_ATLAS_PROXY {path}: vertices={len(mesh.vertices)} "
        f"triangles={len(mesh.triangles)} texture={texture_path}"
    )
    return mesh


def translation(dx: float, dy: float, dz: float) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = [dx, dy, dz]
    return result


def axis_rotation(axis: str, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    result = np.eye(4)
    if axis == "x":
        result[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == "y":
        result[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    else:
        result[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return result


class ManualAligner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.visualizer = o3d.visualization.VisualizerWithKeyCallback()
        self.meshes: dict[int, o3d.geometry.TriangleMesh] = {}
        self.wireframes: dict[int, o3d.geometry.LineSet] = {}
        self.texture_colors: dict[int, np.ndarray] = {}
        self.texture_images: dict[int, list[o3d.geometry.Image]] = {}
        self.transforms = {args.fixed_stage: np.eye(4), args.moving_stage: np.eye(4)}
        self.fixed = args.fixed_stage
        self.selected = args.moving_stage
        self.visibility_mode = 0
        self.translation_step = args.translation_step
        self.rotation_step_deg = args.rotation_step_deg
        self.opening_transform = np.eye(4)

    def load(self) -> None:
        paths = {}
        for stage in (self.fixed, self.selected):
            paths[stage] = (
                self.args.raw_root
                / f"Bldg{self.args.building}_Stage{stage}"
                / "mesh"
                / f"Bldg{self.args.building}_Stage{stage}.obj"
            )
        for stage, path in paths.items():
            proxy = self.args.proxy_dir / f"Bldg{self.args.building}_Stage{stage}_textured_proxy.ply"
            uv_proxy_dir = self.args.uv_proxy_root / f"Bldg{self.args.building}_Stage{stage}"
            uv_proxy = uv_proxy_dir / "proxy.obj"
            atlas = uv_proxy_dir / "atlas.jpg"
            if self.args.uv_atlas and uv_proxy.exists() and atlas.exists() and not self.args.no_proxy:
                mesh = read_single_atlas_obj(uv_proxy, atlas)
            elif proxy.exists() and not self.args.no_proxy:
                mesh = o3d.io.read_triangle_mesh(str(proxy))
                mesh.compute_vertex_normals()
                print(
                    f"LOADED_TEXTURED_PROXY {proxy}: vertices={len(mesh.vertices)} "
                    f"triangles={len(mesh.triangles)}"
                )
            else:
                mesh = read_obj_geometry(path)
                mesh.paint_uniform_color(COLORS[stage - 1])
            self.texture_colors[stage] = np.asarray(mesh.vertex_colors).copy()
            self.texture_images[stage] = list(mesh.textures)
            if self.args.pseudo_colors:
                mesh.textures = []
                mesh.paint_uniform_color(COLORS[stage - 1])
            transform_path = self.args.output_dir / f"stage{stage}_to_stage{self.fixed}.txt"
            initial_path = self.args.initial_transform or transform_path
            if stage == self.selected and initial_path.exists() and not self.args.ignore_saved:
                transform = np.loadtxt(initial_path).reshape(4, 4)
                mesh.transform(transform)
                self.transforms[stage] = transform
                print(f"LOADED Stage{stage} transform: {initial_path}")
            self.meshes[stage] = mesh
            wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            wireframe.paint_uniform_color(COLORS[stage - 1])
            self.wireframes[stage] = wireframe
        self.opening_transform = self.transforms[self.selected].copy()

    def selected_center(self) -> np.ndarray:
        return self.meshes[self.selected].get_axis_aligned_bounding_box().get_center()

    def apply_delta(self, delta: np.ndarray) -> None:
        mesh = self.meshes[self.selected]
        mesh.transform(delta)
        wireframe = self.wireframes[self.selected]
        wireframe.transform(delta)
        self.transforms[self.selected] = delta @ self.transforms[self.selected]
        self.visualizer.update_geometry(mesh)
        self.visualizer.update_geometry(wireframe)
        self.print_state()

    def move(self, dx: float, dy: float, dz: float):
        def callback(_visualizer):
            self.apply_delta(translation(dx, dy, dz))
            return False

        return callback

    def rotate(self, axis: str, direction: float):
        def callback(_visualizer):
            center = self.selected_center()
            angle = np.deg2rad(direction * self.rotation_step_deg)
            delta = translation(*center) @ axis_rotation(axis, angle) @ translation(*(-center))
            self.apply_delta(delta)
            return False

        return callback

    def change_translation_step(self, factor: float):
        def callback(_visualizer):
            self.translation_step = np.clip(self.translation_step * factor, 0.001, 10.0)
            print(f"TRANSLATION_STEP {self.translation_step:.4f} m")
            return False

        return callback

    def change_rotation_step(self, factor: float):
        def callback(_visualizer):
            self.rotation_step_deg = np.clip(self.rotation_step_deg * factor, 0.05, 45.0)
            print(f"ROTATION_STEP {self.rotation_step_deg:.3f} deg")
            return False

        return callback

    def save(self, _visualizer=None):
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        identity_path = self.args.output_dir / f"stage{self.fixed}_to_stage{self.fixed}.txt"
        pair_path = self.args.output_dir / f"stage{self.selected}_to_stage{self.fixed}.txt"
        np.savetxt(identity_path, np.eye(4), fmt="%.12f")
        np.savetxt(pair_path, self.transforms[self.selected], fmt="%.12f")
        print(f"SAVED {identity_path}")
        print(f"SAVED {pair_path}")
        if self.fixed != 1:
            fixed_to_stage1_path = self.args.output_dir / f"stage{self.fixed}_to_stage1.txt"
            if fixed_to_stage1_path.exists():
                fixed_to_stage1 = np.loadtxt(fixed_to_stage1_path).reshape(4, 4)
                moving_to_stage1 = fixed_to_stage1 @ self.transforms[self.selected]
                composed_path = self.args.output_dir / f"stage{self.selected}_to_stage1.txt"
                np.savetxt(composed_path, moving_to_stage1, fmt="%.12f")
                print(f"SAVED_COMPOSED {composed_path}")
        return False

    def cycle_visibility(self, _visualizer=None):
        self.visibility_mode = (self.visibility_mode + 1) % 4
        reference = self.meshes[self.fixed]
        moving = self.meshes[self.selected]
        reference_wire = self.wireframes[self.fixed]
        moving_wire = self.wireframes[self.selected]
        self.visualizer.remove_geometry(reference, reset_bounding_box=False)
        self.visualizer.remove_geometry(moving, reset_bounding_box=False)
        self.visualizer.remove_geometry(reference_wire, reset_bounding_box=False)
        self.visualizer.remove_geometry(moving_wire, reset_bounding_box=False)
        if self.visibility_mode == 0:
            self.visualizer.add_geometry(reference, reset_bounding_box=False)
            self.visualizer.add_geometry(moving, reset_bounding_box=False)
        elif self.visibility_mode == 1:
            self.visualizer.add_geometry(reference, reset_bounding_box=False)
        elif self.visibility_mode == 2:
            self.visualizer.add_geometry(moving, reset_bounding_box=False)
        else:
            self.visualizer.add_geometry(reference, reset_bounding_box=False)
            self.visualizer.add_geometry(moving_wire, reset_bounding_box=False)
        labels = [
            "BOTH_SOLID",
            f"STAGE{self.fixed}_ONLY",
            f"STAGE{self.selected}_ONLY",
            "FIXED_TEXTURE_PLUS_MOVING_WIREFRAME",
        ]
        print(f"VISIBILITY {labels[self.visibility_mode]}")
        return False

    def toggle_wireframe(self, _visualizer=None):
        render = self.visualizer.get_render_option()
        render.mesh_show_wireframe = not render.mesh_show_wireframe
        print(f"WIREFRAME {render.mesh_show_wireframe}")
        return False

    def reset_alignment(self, _visualizer=None):
        current = self.transforms[self.selected]
        delta = self.opening_transform @ np.linalg.inv(current)
        self.apply_delta(delta)
        print("ALIGNMENT_RESET_TO_OPENING_TRANSFORM")
        return False

    def toggle_colors(self, _visualizer=None):
        self.args.pseudo_colors = not self.args.pseudo_colors
        for stage, mesh in self.meshes.items():
            if self.args.pseudo_colors:
                mesh.textures = []
                mesh.paint_uniform_color(COLORS[stage - 1])
            else:
                if self.texture_images[stage]:
                    mesh.vertex_colors = o3d.utility.Vector3dVector()
                    mesh.textures = self.texture_images[stage]
                else:
                    mesh.vertex_colors = o3d.utility.Vector3dVector(self.texture_colors[stage])
            self.visualizer.update_geometry(mesh)
        print(f"COLOR_MODE {'PSEUDO_STAGE_COLORS' if self.args.pseudo_colors else 'TEXTURE_COLORS'}")
        return False

    def reset_topdown(self, _visualizer=None):
        center = self.meshes[self.fixed].get_axis_aligned_bounding_box().get_center()
        view = self.visualizer.get_view_control()
        view.set_lookat(center)
        view.set_front([0.0, 0.0, -1.0])
        view.set_up([0.0, 1.0, 0.0])
        view.set_zoom(0.62)
        print("VIEW TOPDOWN_RESET")
        return False

    def print_state(self) -> None:
        transform = self.transforms[self.selected]
        yaw = np.rad2deg(np.arctan2(transform[1, 0], transform[0, 0]))
        xyz = transform[:3, 3]
        print(
            f"Stage{self.selected}: translation=[{xyz[0]:.3f}, {xyz[1]:.3f}, "
            f"{xyz[2]:.3f}] yaw~={yaw:.2f} deg"
        )

    @staticmethod
    def print_help() -> None:
        print(
            """
Pairwise manual alignment controls (the first stage in --pair is fixed):
  Arrow keys        move X/Y in the top-down world view
  PageUp/PageDown   move Z up/down
  A / D or Q / E    rotate the MOVING mesh yaw left/right around its own center
  I / K             pitch around X (usually leave unchanged)
  J / L             roll around Y (usually leave unchanged)
  [ / ]             halve/double translation step
  , / .             halve/double rotation step
  P                 save all 4x4 transforms
  R                 reset moving mesh to opening transform
  V                 both solid / fixed only / moving only / wireframe overlay
  C                 texture colors / stage colors
  M                 toggle triangle wireframe
  T                 reset camera to top-down view
  H                 print this help
  Esc / close       exit (press P first)
"""
        )

    def run(self) -> None:
        self.load()
        self.visualizer.create_window(
            f"Manual NSS pair: Stage{self.fixed} fixed | Stage{self.selected} moving",
            width=self.args.width,
            height=self.args.height,
        )
        self.visualizer.add_geometry(self.meshes[self.fixed])
        self.visualizer.add_geometry(self.meshes[self.selected])

        step = lambda: self.translation_step
        self.visualizer.register_key_callback(KEY_LEFT, lambda v: self.move(-step(), 0, 0)(v))
        self.visualizer.register_key_callback(KEY_RIGHT, lambda v: self.move(step(), 0, 0)(v))
        self.visualizer.register_key_callback(KEY_UP, lambda v: self.move(0, step(), 0)(v))
        self.visualizer.register_key_callback(KEY_DOWN, lambda v: self.move(0, -step(), 0)(v))
        self.visualizer.register_key_callback(KEY_PAGE_UP, lambda v: self.move(0, 0, step())(v))
        self.visualizer.register_key_callback(KEY_PAGE_DOWN, lambda v: self.move(0, 0, -step())(v))
        self.visualizer.register_key_callback(ord("Q"), self.rotate("z", 1))
        self.visualizer.register_key_callback(ord("E"), self.rotate("z", -1))
        self.visualizer.register_key_callback(ord("A"), self.rotate("z", 1))
        self.visualizer.register_key_callback(ord("D"), self.rotate("z", -1))
        self.visualizer.register_key_callback(ord("I"), self.rotate("x", 1))
        self.visualizer.register_key_callback(ord("K"), self.rotate("x", -1))
        self.visualizer.register_key_callback(ord("J"), self.rotate("y", 1))
        self.visualizer.register_key_callback(ord("L"), self.rotate("y", -1))
        self.visualizer.register_key_callback(ord("["), self.change_translation_step(0.5))
        self.visualizer.register_key_callback(ord("]"), self.change_translation_step(2.0))
        self.visualizer.register_key_callback(ord(","), self.change_rotation_step(0.5))
        self.visualizer.register_key_callback(ord("."), self.change_rotation_step(2.0))
        self.visualizer.register_key_callback(ord("P"), self.save)
        self.visualizer.register_key_callback(ord("R"), self.reset_alignment)
        self.visualizer.register_key_callback(ord("V"), self.cycle_visibility)
        self.visualizer.register_key_callback(ord("C"), self.toggle_colors)
        self.visualizer.register_key_callback(ord("M"), self.toggle_wireframe)
        self.visualizer.register_key_callback(ord("T"), self.reset_topdown)
        self.visualizer.register_key_callback(ord("H"), lambda _v: (self.print_help() or False))

        self.reset_topdown()
        render = self.visualizer.get_render_option()
        render.background_color = np.asarray([1.0, 1.0, 1.0])
        render.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
        render.line_width = 1.5

        self.print_help()
        self.print_state()
        self.visualizer.run()
        self.visualizer.destroy_window()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--building", type=int, default=1)
    parser.add_argument("--pair", choices=["12", "13", "23"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--proxy-dir", type=Path, default=Path("runs/nss/textured_mesh_proxies")
    )
    parser.add_argument(
        "--uv-proxy-root", type=Path, default=Path("runs/nss/uv_atlas_mesh_proxies")
    )
    parser.add_argument(
        "--uv-atlas", action="store_true", help="Opt in to the experimental UV-atlas proxy"
    )
    parser.add_argument("--no-proxy", action="store_true")
    parser.add_argument("--pseudo-colors", action="store_true")
    parser.add_argument("--initial-transform", type=Path)
    parser.add_argument("--translation-step", type=float, default=0.25)
    parser.add_argument("--rotation-step-deg", type=float, default=1.0)
    parser.add_argument("--ignore-saved", action="store_true")
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1200)
    args = parser.parse_args()
    args.fixed_stage = int(args.pair[0])
    args.moving_stage = int(args.pair[1])
    ManualAligner(args).run()


if __name__ == "__main__":
    main()
