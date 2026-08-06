#!/usr/bin/env python3
"""Interactive Open3D player for the Base1 A/B process sequence."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering


class ProcessViewer:
    def __init__(
        self,
        manifest: Path,
        frame_seconds: float,
        trajectories_file: Path | None,
        sensor_views_file: Path | None,
    ) -> None:
        self.root = manifest.parent
        with manifest.open(newline="") as stream:
            self.frames = list(csv.DictReader(stream))
        if not self.frames:
            raise RuntimeError("Sequence manifest has no frames")

        self.trajectories = {"A": {}, "B": {}}
        if trajectories_file and trajectories_file.exists():
            self.trajectories = json.loads(trajectories_file.read_text())["sessions"]
        self.sensor_views = {"A": {}, "B": {}}
        if sensor_views_file and sensor_views_file.exists():
            self.sensor_views = json.loads(sensor_views_file.read_text())["sessions"]
        self.a_final_frame = next(
            frame for frame in reversed(self.frames) if frame["session"] == "A"
        )
        self.a_history_mesh = o3d.io.read_triangle_mesh(
            str(self.root / self.a_final_frame["ply"])
        )
        if self.a_history_mesh.is_empty():
            raise RuntimeError("Session A history mesh is empty")
        self.a_history_mesh.compute_vertex_normals()

        self.frame_seconds = frame_seconds
        self.index = 0
        self.playing = False
        self.closed = False
        self.last_advance = time.monotonic()
        self.camera_initialized = False
        self.loaded_map_path: Path | None = None
        self.overlay_names: list[str] = []
        self.history_visible = True
        self.trajectories_visible = True
        self.sensor_range_visible = True
        self.sensor_max_range_m = 5.0
        self.current_session = "A"
        self.sensor_slot_paths = {"A": None, "B": None}

        app = gui.Application.instance
        self.window = app.create_window("Base1: Session A -> Session B", 1600, 900)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.96, 0.97, 0.98, 1.0])

        em = self.window.theme.font_size
        self.panel = gui.Vert(0.4 * em, gui.Margins(em, 0.7 * em, em, 0.7 * em))
        self.title = gui.Label("")
        self.details = gui.Label("")
        self.timeline = gui.Slider(gui.Slider.INT)
        self.timeline.set_limits(0, len(self.frames) - 1)
        self.timeline.int_value = 0
        self.timeline.set_on_value_changed(lambda value: self.show_frame(int(value)))

        controls = gui.Horiz(0.5 * em)
        previous_button = gui.Button("<")
        previous_button.set_on_clicked(lambda: self.step(-1))
        self.play_button = gui.Button("Play")
        self.play_button.set_on_clicked(self.toggle_play)
        next_button = gui.Button(">")
        next_button.set_on_clicked(lambda: self.step(1))
        restart_button = gui.Button("Restart")
        restart_button.set_on_clicked(lambda: self.show_frame(0))
        jump_b_button = gui.Button("Jump to Session B")
        jump_b_button.set_on_clicked(self.jump_to_b)
        for widget in (previous_button, self.play_button, next_button, restart_button, jump_b_button):
            controls.add_child(widget)

        layer_controls = gui.Horiz(0.8 * em)
        self.history_checkbox = gui.Checkbox("Session A history")
        self.history_checkbox.checked = True
        self.history_checkbox.set_on_checked(self.set_history_visible)
        self.trajectory_checkbox = gui.Checkbox("Robot trajectories")
        self.trajectory_checkbox.checked = True
        self.trajectory_checkbox.set_on_checked(self.set_trajectories_visible)
        self.sensor_range_checkbox = gui.Checkbox("5 m sensor range")
        self.sensor_range_checkbox.checked = True
        self.sensor_range_checkbox.set_on_checked(self.set_sensor_range_visible)
        self.legend = gui.Label(
            "Map: current solid | A history translucent blue   "
            "Robot: A cyan | B orange | 5 m envelope yellow   "
            "Dynamic entities: magenta"
        )
        layer_controls.add_child(self.history_checkbox)
        layer_controls.add_child(self.trajectory_checkbox)
        layer_controls.add_child(self.sensor_range_checkbox)

        self.panel.add_child(self.title)
        self.panel.add_child(self.details)
        self.panel.add_child(self.legend)
        self.panel.add_child(self.timeline)
        self.panel.add_child(controls)
        self.panel.add_child(layer_controls)
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.sensor_a_label = gui.Label("VISIT 1 | current RGB")
        self.sensor_a_label.background_color = gui.Color(0.05, 0.08, 0.12, 0.92)
        self.sensor_a_image = gui.ImageWidget()
        self.sensor_a_image.background_color = gui.Color(0.03, 0.04, 0.06, 0.92)
        self.sensor_b_label = gui.Label("VISIT 2 | current RGB")
        self.sensor_b_label.background_color = gui.Color(0.05, 0.08, 0.12, 0.92)
        self.sensor_b_image = gui.ImageWidget()
        self.sensor_b_image.background_color = gui.Color(0.03, 0.04, 0.06, 0.92)
        for widget in (
            self.sensor_a_label,
            self.sensor_a_image,
            self.sensor_b_label,
            self.sensor_b_image,
        ):
            self.window.add_child(widget)
        self.window.set_on_layout(self.on_layout)
        self.window.set_on_close(self.on_close)
        self.show_frame(0)
        threading.Thread(target=self.play_loop, daemon=True).start()

    def on_layout(self, context: gui.LayoutContext) -> None:
        rect = self.window.content_rect
        panel_height = int(8.0 * context.theme.font_size)
        self.panel.frame = gui.Rect(rect.x, rect.y, rect.width, panel_height)
        self.scene_widget.frame = gui.Rect(
            rect.x, rect.y + panel_height, rect.width, rect.height - panel_height
        )
        inset_width = min(360, max(260, int(rect.width * 0.23)))
        inset_height = int(inset_width * 2 / 3)
        label_height = int(1.7 * context.theme.font_size)
        gap = 14
        margin = 18
        top = rect.y + panel_height + margin
        right_x = rect.x + rect.width - margin - inset_width
        if self.current_session == "B":
            left_x = right_x - gap - inset_width
        else:
            left_x = right_x
        self.sensor_a_label.frame = gui.Rect(left_x, top, inset_width, label_height)
        self.sensor_a_image.frame = gui.Rect(
            left_x, top + label_height, inset_width, inset_height
        )
        self.sensor_b_label.frame = gui.Rect(right_x, top, inset_width, label_height)
        self.sensor_b_image.frame = gui.Rect(
            right_x, top + label_height, inset_width, inset_height
        )

    def frame_path(self, index: int) -> Path:
        return self.root / self.frames[index]["ply"]

    def clear_overlays(self) -> None:
        for name in self.overlay_names:
            self.scene_widget.scene.remove_geometry(name)
        self.overlay_names.clear()

    def set_history_visible(self, visible: bool) -> None:
        self.history_visible = visible
        self.show_frame(self.index)

    def set_trajectories_visible(self, visible: bool) -> None:
        self.trajectories_visible = visible
        self.show_frame(self.index)

    def set_sensor_range_visible(self, visible: bool) -> None:
        self.sensor_range_visible = visible
        self.show_frame(self.index)

    def add_session_a_history(self, session: str) -> None:
        self.scene_widget.scene.remove_geometry("session_a_history")
        if session != "B" or not self.history_visible:
            return
        material = rendering.MaterialRecord()
        material.shader = "defaultLitTransparency"
        material.base_color = [0.16, 0.62, 0.95, 0.20]
        self.scene_widget.scene.add_geometry(
            "session_a_history", self.a_history_mesh, material
        )

    def add_robot_trajectory(
        self,
        session_name: str,
        query_time_ns: int | None,
        show_full: bool,
    ) -> int:
        trajectory = self.trajectories.get(session_name, {})
        positions = trajectory.get("positions", [])
        timestamps = [int(value) for value in trajectory.get("timestamps_ns", [])]
        if not positions or len(positions) != len(timestamps):
            return 0

        if show_full:
            visible_count = len(positions)
        elif query_time_ns is not None:
            visible_count = bisect.bisect_right(timestamps, query_time_ns)
        else:
            visible_count = 0
        visible_count = max(0, min(visible_count, len(positions)))
        if visible_count == 0:
            return 0

        color = [0.02, 0.72, 0.92] if session_name == "A" else [1.0, 0.46, 0.05]
        name_prefix = f"robot_{session_name.lower()}"
        if visible_count > 1:
            points = o3d.utility.Vector3dVector(positions[:visible_count])
            lines = o3d.utility.Vector2iVector(
                [[index, index + 1] for index in range(visible_count - 1)]
            )
            line_set = o3d.geometry.LineSet(points, lines)
            line_set.colors = o3d.utility.Vector3dVector(
                [color for _ in range(visible_count - 1)]
            )
            material = rendering.MaterialRecord()
            material.shader = "unlitLine"
            material.line_width = 6.0
            name = f"{name_prefix}_trajectory"
            self.scene_widget.scene.add_geometry(name, line_set, material)
            self.overlay_names.append(name)

        marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.13, resolution=12)
        marker.translate(positions[visible_count - 1])
        marker.compute_vertex_normals()
        marker.paint_uniform_color(color)
        marker_material = rendering.MaterialRecord()
        marker_material.shader = "defaultLit"
        name = f"{name_prefix}_position"
        self.scene_widget.scene.add_geometry(name, marker, marker_material)
        self.overlay_names.append(name)
        return visible_count

    def add_robot_trajectories(self, frame: dict[str, str]) -> tuple[int, int]:
        if not self.trajectories_visible:
            return 0, 0
        query_time_ns = int(float(frame["session_time_s"]) * 1.0e9)
        session = frame["session"]
        if session == "A":
            return self.add_robot_trajectory("A", query_time_ns, False), 0
        return (
            self.add_robot_trajectory("A", None, True),
            self.add_robot_trajectory("B", query_time_ns, False),
        )

    def add_sensor_range(self, frame: dict[str, str]) -> float | None:
        if not self.sensor_range_visible:
            return None
        session = frame["session"]
        trajectory = self.trajectories.get(session, {})
        positions = trajectory.get("positions", [])
        timestamps = [int(value) for value in trajectory.get("timestamps_ns", [])]
        if not positions or len(positions) != len(timestamps):
            return None

        query_time_ns = int(float(frame["session_time_s"]) * 1.0e9)
        pose_index = bisect.bisect_right(timestamps, query_time_ns) - 1
        if pose_index < 0:
            return None
        center = positions[min(pose_index, len(positions) - 1)]

        ring_samples = 128
        ring_points = []
        for sample in range(ring_samples):
            angle = 2.0 * 3.141592653589793 * sample / ring_samples
            ring_points.append(
                [
                    center[0] + self.sensor_max_range_m * math.cos(angle),
                    center[1] + self.sensor_max_range_m * math.sin(angle),
                    center[2],
                ]
            )
        lines = [[index, (index + 1) % ring_samples] for index in range(ring_samples)]
        ring = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(ring_points),
            o3d.utility.Vector2iVector(lines),
        )
        color = [0.98, 0.78, 0.06]
        ring.colors = o3d.utility.Vector3dVector([color for _ in lines])
        material = rendering.MaterialRecord()
        material.shader = "unlitLine"
        material.line_width = 2.0
        name = "sensor_max_range"
        self.scene_widget.scene.add_geometry(name, ring, material)
        self.overlay_names.append(name)
        return self.sensor_max_range_m

    def add_dynamic_tracks(self, frame: dict[str, str]) -> int:
        overlay_name = frame.get("dynamic_history", "") or frame.get("overlay", "")
        if not overlay_name:
            return 0
        overlay = json.loads((self.root / overlay_name).read_text())
        tracks = overlay.get("dynamic_tracks", [])
        query_time_ns = int(float(frame["session_time_s"]) * 1.0e9)
        track_material = rendering.MaterialRecord()
        track_material.shader = "unlitLine"
        track_material.line_width = 4.0
        bbox_material = rendering.MaterialRecord()
        bbox_material.shader = "unlitLine"
        bbox_material.line_width = 2.0
        point_material = rendering.MaterialRecord()
        point_material.shader = "defaultUnlit"
        point_material.point_size = 3.0
        color = [0.95, 0.05, 0.55]

        active_tracks = 0
        for index, track in enumerate(tracks):
            positions = track.get("positions", [])
            timestamps = [int(value) for value in track.get("timestamps_ns", [])]
            if not positions or len(positions) != len(timestamps) or query_time_ns < timestamps[0]:
                continue

            current_index = min(
                bisect.bisect_left(timestamps, query_time_ns),
                len(timestamps) - 1,
            )
            visible_positions = positions[: current_index + 1]
            if len(visible_positions) > 1:
                points = o3d.utility.Vector3dVector(visible_positions)
                lines = o3d.utility.Vector2iVector(
                    [[i, i + 1] for i in range(len(visible_positions) - 1)]
                )
                trajectory = o3d.geometry.LineSet(points, lines)
                trajectory.colors = o3d.utility.Vector3dVector(
                    [color for _ in range(len(visible_positions) - 1)]
                )
                name = f"dynamic_track_{index}"
                self.scene_widget.scene.add_geometry(name, trajectory, track_material)
                self.overlay_names.append(name)

            # A dynamic entity is current only while its recorded motion is active.
            # The past trajectory remains as history after it leaves the scene.
            if query_time_ns > timestamps[-1] + 300_000_000:
                continue
            active_tracks += 1
            center = positions[current_index]
            point_frames = track.get("point_frames", [])
            if current_index < len(point_frames) and point_frames[current_index]:
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(point_frames[current_index])
                point_cloud.paint_uniform_color(color)
                name = f"dynamic_points_{index}"
                self.scene_widget.scene.add_geometry(name, point_cloud, point_material)
                self.overlay_names.append(name)
            extent = track.get("bbox_dimensions", [0.5, 0.5, 1.7])
            minimum = [center[i] - 0.5 * extent[i] for i in range(3)]
            maximum = [center[i] + 0.5 * extent[i] for i in range(3)]
            bbox = o3d.geometry.AxisAlignedBoundingBox(minimum, maximum)
            bbox.color = color
            bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
            name = f"dynamic_bbox_{index}"
            self.scene_widget.scene.add_geometry(name, bbox_lines, bbox_material)
            self.overlay_names.append(name)
        return active_tracks

    @staticmethod
    def nearest_timestamp_index(timestamps: list[int], query_time_ns: int) -> int:
        if not timestamps:
            return -1
        insertion = bisect.bisect_left(timestamps, query_time_ns)
        if insertion <= 0:
            return 0
        if insertion >= len(timestamps):
            return len(timestamps) - 1
        before = insertion - 1
        if query_time_ns - timestamps[before] <= timestamps[insertion] - query_time_ns:
            return before
        return insertion

    def update_sensor_slot(
        self,
        slot: str,
        session: str,
        image_index: int,
        label: str,
    ) -> None:
        data = self.sensor_views.get(session, {})
        paths = data.get("paths", [])
        if image_index < 0 or image_index >= len(paths):
            return
        relative_path = paths[image_index]
        label_widget = self.sensor_a_label if slot == "A" else self.sensor_b_label
        image_widget = self.sensor_a_image if slot == "A" else self.sensor_b_image
        label_widget.text = label
        if self.sensor_slot_paths[slot] == relative_path:
            return
        image = o3d.io.read_image(str(self.root / relative_path))
        if image.is_empty():
            return
        image_widget.update_image(image)
        self.sensor_slot_paths[slot] = relative_path

    def update_sensor_views(self, frame: dict[str, str]) -> None:
        session = frame["session"]
        query_time_ns = int(float(frame["session_time_s"]) * 1.0e9)
        self.current_session = session

        current_data = self.sensor_views.get(session, {})
        current_times = [int(value) for value in current_data.get("timestamps_ns", [])]
        current_index = self.nearest_timestamp_index(current_times, query_time_ns)
        if session == "A":
            self.sensor_a_label.visible = current_index >= 0
            self.sensor_a_image.visible = current_index >= 0
            self.sensor_b_label.visible = False
            self.sensor_b_image.visible = False
            if current_index >= 0:
                sensor_time = current_times[current_index] / 1.0e9
                self.update_sensor_slot(
                    "A",
                    "A",
                    current_index,
                    f"VISIT 1 | current RGB | t={sensor_time:.2f}s",
                )
            self.window.set_needs_layout()
            return

        has_current = current_index >= 0
        self.sensor_b_label.visible = has_current
        self.sensor_b_image.visible = has_current
        self.sensor_a_label.visible = False
        self.sensor_a_image.visible = False
        if not has_current:
            self.window.set_needs_layout()
            return

        current_positions = current_data.get("positions", [])
        visit_a = self.sensor_views.get("A", {})
        visit_a_positions = np.asarray(visit_a.get("positions", []), dtype=float)
        if current_index < len(current_positions) and len(visit_a_positions):
            current_position = np.asarray(current_positions[current_index], dtype=float)
            distances = np.linalg.norm(visit_a_positions - current_position, axis=1)
            visit_a_index = int(np.argmin(distances))
            distance = float(distances[visit_a_index])
            visit_a_times = [
                int(value) for value in visit_a.get("timestamps_ns", [])
            ]
            visit_a_time = visit_a_times[visit_a_index] / 1.0e9
            self.sensor_a_label.visible = True
            self.sensor_a_image.visible = True
            self.update_sensor_slot(
                "A",
                "A",
                visit_a_index,
                f"VISIT 1 | nearest prior RGB | t={visit_a_time:.2f}s | d={distance:.2f}m",
            )

        current_time = current_times[current_index] / 1.0e9
        self.update_sensor_slot(
            "B",
            "B",
            current_index,
            f"VISIT 2 | current RGB | t={current_time:.2f}s",
        )
        self.window.set_needs_layout()

    def show_frame(self, index: int) -> None:
        self.index = max(0, min(index, len(self.frames) - 1))
        frame = self.frames[self.index]
        map_path = self.frame_path(self.index)
        mesh = None
        if map_path != self.loaded_map_path:
            mesh = o3d.io.read_triangle_mesh(str(map_path))
            if mesh.is_empty():
                raise RuntimeError(f"Empty mesh: {map_path}")
            mesh.compute_vertex_normals()
            self.scene_widget.scene.remove_geometry("map")
            material = rendering.MaterialRecord()
            material.shader = "defaultLit"
            self.scene_widget.scene.add_geometry("map", mesh, material)
            self.loaded_map_path = map_path
        self.clear_overlays()
        self.add_session_a_history(frame["session"])
        a_poses, b_poses = self.add_robot_trajectories(frame)
        sensor_range = self.add_sensor_range(frame)
        dynamic_tracks = self.add_dynamic_tracks(frame)
        self.update_sensor_views(frame)
        if not self.camera_initialized:
            assert mesh is not None
            bounds = mesh.get_axis_aligned_bounding_box()
            self.scene_widget.setup_camera(58.0, bounds, bounds.get_center())
            self.camera_initialized = True

        session = frame["session"]
        phase = (
            "within-session reconstruction + offline detected-motion replay"
            if session == "A"
            else "cross-session reconstruction using A memory"
        )
        self.title.text = (
            f"Session {session} | {phase} | checkpoint "
            f"{int(frame['session_checkpoint']) + 1}/13"
        )
        if session == "B":
            self.details.text = (
                f"source t={float(frame['source_time_s']):.2f}s   "
                f"A-base={frame['cross_session_prior_vertices']}  "
                f"B-evidence={frame['cross_session_current_vertices']}  "
                f"A absent/persistent/unobserved="
                f"{frame['cross_session_prior_absent_vertices']}/"
                f"{frame['cross_session_prior_persistent_vertices']}/"
                f"{frame['cross_session_prior_unobserved_vertices']}  "
                f"B-new={frame['cross_session_current_injected_vertices']}  "
                f"final={frame['final_vertices']}  robot poses A/B={a_poses}/{b_poses}  "
                f"range={sensor_range or 0:g}m  dynamic tracks={dynamic_tracks}"
            )
        else:
            self.details.text = (
                f"source t={float(frame['source_time_s']):.2f}s   "
                f"vertices {frame['initial_vertices']} -> {frame['final_vertices']}   "
                f"removed={frame['removed_vertices']}   injected={frame['injected_vertices']}  "
                f"robot poses A={a_poses}  range={sensor_range or 0:g}m  "
                f"dynamic tracks={dynamic_tracks}"
            )
        self.timeline.int_value = self.index
        self.last_advance = time.monotonic()

    def step(self, delta: int) -> None:
        self.playing = False
        self.play_button.text = "Play"
        self.show_frame(self.index + delta)

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.text = "Pause" if self.playing else "Play"
        self.last_advance = time.monotonic()

    def jump_to_b(self) -> None:
        for index, frame in enumerate(self.frames):
            if frame["session"] == "B":
                self.show_frame(index)
                return

    def advance_if_due(self) -> None:
        if not self.playing or time.monotonic() - self.last_advance < self.frame_seconds:
            return
        if self.index + 1 >= len(self.frames):
            self.playing = False
            self.play_button.text = "Play"
            return
        self.show_frame(self.index + 1)

    def play_loop(self) -> None:
        app = gui.Application.instance
        while not self.closed:
            time.sleep(0.05)
            if self.playing:
                app.post_to_main_thread(self.window, self.advance_if_due)

    def on_close(self) -> bool:
        self.closed = True
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "sequence_manifest.csv",
    )
    parser.add_argument("--frame-seconds", type=float, default=0.8)
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=None,
        help="Extracted A/B trajectory JSON (default: next to manifest)",
    )
    parser.add_argument(
        "--sensor-views",
        type=Path,
        default=None,
        help="Extracted A/B RGB manifest (default: next to manifest)",
    )
    args = parser.parse_args()
    trajectories = args.trajectories
    if trajectories is None:
        trajectories = args.manifest.resolve().parent / "session_trajectories.json"
    sensor_views = args.sensor_views
    if sensor_views is None:
        sensor_views = args.manifest.resolve().parent / "sensor_views.json"

    gui.Application.instance.initialize()
    ProcessViewer(
        args.manifest.resolve(),
        args.frame_seconds,
        trajectories.resolve(),
        sensor_views.resolve(),
    )
    try:
        gui.Application.instance.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
