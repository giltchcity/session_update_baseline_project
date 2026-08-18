#!/usr/bin/env python3
"""Interactive Open3D player for the recursive 0->A->B chain.

Parses the two .4dmap files directly through the resident 4dmap_mesh_server
(C++ reader) over a subprocess pipe -- no intermediate files are produced.
Drag one slider across the whole 0->A->B timeline: frames 0..na-1 = session A,
na..na+nb-1 = session B (B[0] is the continuation of A[-1]).

Usage:
    conda run -n 3d_vsg python session_update_baseline/scripts/view_ab_chain_4dmap.py \
        [--map-a runs/session_update_ab_chain_20260814/session_a/final.4dmap] \
        [--map-b runs/session_update_ab_chain_20260814/session_b/final.4dmap] \
        [--pose-a <A rgbd dir>] [--pose-b <B rgbd dir>] \
        [--frame-seconds 0.4]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import struct
import subprocess
import threading
import time
import bisect
from collections import OrderedDict
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering

_SERVER = (
    Path(__file__).resolve().parent.parent
    / "build_canonical" / "4dmap_mesh_server"
)
# Runtime library roots required by the canonical build; order matters
# (first entry wins per library).
_LD_LIBRARY_PATH = [
    "/home/jixian/ros2_ws/install/hydra/lib",
    "/home/jixian/ros2_ws/install/teaserpp/lib",
    "/home/jixian/ros2_ws/install/spark_dsg/lib",
    "/home/jixian/ros2_ws/install/kimera_pgmo/lib",
    "/home/jixian/ros2_ws/install/config_utilities/lib",
    "/home/jixian/ros2_ws/install/kimera_rpgo/lib",
    "/opt/ros/jazzy/lib/x86_64-linux-gnu",
    "/home/jixian/ros2_ws/install/pose_graph_tools/lib",
    "/home/jixian/Desktop/FT/session_update_baseline/.canonical_mapping/install/lib",
]

_HEADER = struct.Struct("<IIIQ")  # magic, n_vertices, n_faces, timestamp_ns
_MAGIC = 0x314D4241


def load_pose_trajectory(rgbd_dir: Path | None):
    """Camera-to-world positions from an RGBD session dir (timestamps.csv +
    per-frame <ImageID>_pose.txt). Returns (positions Nx3, stamps) or None."""
    if not rgbd_dir or not rgbd_dir.is_dir():
        return None
    rows = []
    with (rgbd_dir / "timestamps.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append(row)
    positions = []
    stamps = []
    for row in rows:
        image_id = row.get("ImageID", row.get("image_id", ""))
        pose_file = rgbd_dir / f"{image_id}_pose.txt"
        if not pose_file.is_file():
            continue
        matrix = []
        with pose_file.open() as stream:
            for line in stream:
                matrix.append([float(v) for v in line.split()])
                if len(matrix) == 4:
                    positions.append([matrix[0][3], matrix[1][3], matrix[2][3]])
                    matrix = []
        stamp = row.get("TimeStamp", row.get("timestamp", ""))
        stamps.append(int(stamp))
    if not positions:
        return None
    return np.asarray(positions, dtype=np.float64), np.asarray(stamps, dtype=np.int64)


def load_rgbd_images(rgbd_dir: Path | None):
    """(paths, stamps_ns) for the <ImageID>_color.png files of an RGBD session
    dir, sorted by timestamp. None when the dir has no usable images."""
    if not rgbd_dir or not rgbd_dir.is_dir():
        return None
    rows = []
    with (rgbd_dir / "timestamps.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append(row)
    paths = []
    stamps = []
    for row in rows:
        image_id = row.get("ImageID", row.get("image_id", ""))
        image_path = rgbd_dir / f"{image_id}_color.png"
        if not image_path.is_file():
            continue
        paths.append(str(image_path))
        stamps.append(int(row.get("TimeStamp", row.get("timestamp", "0"))))
    if not paths:
        return None
    return paths, np.asarray(stamps, dtype=np.int64)


class MapServer:
    """Resident 4dmap_mesh_server subprocess: JSON metadata line, then binary
    frame responses. Request/response serialized; safe for one worker thread.

    Construction spawns the subprocess and returns immediately; the metadata
    line (written only after the maps finish loading, ~1 min for 16 GiB) is
    read lazily by the first caller. Loading progress arrives on stderr and
    is forwarded to an optional callback."""

    def __init__(
        self, map_a: str, map_b: str, stride: int = 1, on_progress=None
    ) -> None:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = ":".join(_LD_LIBRARY_PATH) + ":" + env.get(
            "LD_LIBRARY_PATH", ""
        )
        self.proc = subprocess.Popen(
            [_SERVER, "--map_a", map_a, "--map_b", map_b, "--stride", str(stride)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self.meta = None
        self.sessions = []
        self.frames = {}
        self.stamps = {}
        self._meta_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._closed = False
        self._on_progress = on_progress
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self) -> None:
        for line in self.proc.stderr:
            text = line.decode(errors="replace").rstrip()
            if self._on_progress:
                self._on_progress(text)

    def _ensure_meta(self) -> None:
        with self._meta_lock:
            if self.meta is None:
                self.meta = json.loads(self.proc.stdout.readline().decode())
                self.sessions = [n for n in ("a", "b") if n in self.meta]
                self.frames = {
                    n: self.meta[n]["frames"] for n in self.sessions
                }
                self.stamps = {
                    n: self.meta[n]["stamps"] for n in self.sessions
                }

    def request(self, session: str, index: int):
        """Returns ((verts Nx3 f4, faces Mx3 u4, colors Nx3 u1), ts) or
        (None, ts) when the snapshot has no mesh. Blocks until the maps are
        loaded on first use."""
        self._ensure_meta()
        with self._request_lock:
            self.proc.stdin.write(f"{session} {index}\n".encode())
            self.proc.stdin.flush()
            magic, nv, nf, ts = _HEADER.unpack(
                self.proc.stdout.read(_HEADER.size)
            )
            if magic != _MAGIC:
                raise RuntimeError(f"bad magic 0x{magic:x} from server")
            if nv == 0:
                return None, ts
            verts = np.frombuffer(
                self.proc.stdout.read(12 * nv), dtype="<f4"
            ).reshape(-1, 3)
            faces = np.frombuffer(
                self.proc.stdout.read(12 * nf), dtype="<u4"
            ).reshape(-1, 3)
            colors = np.frombuffer(
                self.proc.stdout.read(3 * nv), dtype="u1"
            ).reshape(-1, 3)
            return (verts, faces, colors), ts

    def request_dynamic(self, session: str, index: int):
        """D1 temporal layer for one snapshot: JSON list of dynamic object
        nodes with trajectory positions, per-sample timestamps and per-sample
        point clouds (world frame). [] when the snapshot has none."""
        self._ensure_meta()
        with self._request_lock:
            self.proc.stdin.write(f"DYNPTS {session} {index}\n".encode())
            self.proc.stdin.flush()
            return json.loads(self.proc.stdout.readline().decode())

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            with self._request_lock:
                self.proc.stdin.write(b"QUIT\n")
                self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


class ChainPlayer:
    def __init__(
        self,
        server: MapServer,
        pose_a,
        pose_b,
        frame_seconds: float,
        rgbd_a=None,
        rgbd_b=None,
    ) -> None:
        self.server = server
        # Map metadata arrives only after the server loads the maps (~1 min);
        # the window opens immediately and the timeline is armed when ready.
        self.na = 0
        self.nb = 0
        self.total = 0
        self.chain_start_ns = 0
        self.chain_end_ns = 0
        self.frame_seconds = frame_seconds
        self.index = 0
        self.playing = False
        self.last_advance = 0.0
        self.show_trajectories = True
        self.show_dynamics = True
        self.updating_timeline = False
        self.overlay_names = []
        self.overlay_meshes = {}
        self.dynamic_names = []
        self.rgbd_a = rgbd_a
        self.rgbd_b = rgbd_b
        self.current_session = "A"
        self.image_cache = OrderedDict()
        self.image_cache_limit = 8
        self.sensor_slot_paths = {"A": None, "B": None}

        app = gui.Application.instance
        # 1280x800: llvmpipe (WSLg on Win10 has no GPU GL) costs pixels ~ O(area);
        # 1600x1000 roughly doubles the software rasterization cost of the view.
        self.window = app.create_window("0 -> A -> B recursive chain", 1280, 800)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.05, 0.07, 0.10, 1.0])
        self.scene_widget.scene.show_axes(True)

        em = self.window.theme.font_size
        self.panel = gui.Vert(0.4 * em, gui.Margins(em, 0.5 * em, em, 0.5 * em))
        self.title = gui.Label("")
        self.details = gui.Label("")
        self.title.text_color = gui.Color(1.0, 1.0, 1.0)
        self.details.text_color = gui.Color(0.85, 0.9, 0.95)

        self.timeline = gui.Slider(gui.Slider.INT)
        self.timeline.set_limits(0, 0)
        self.timeline.set_on_value_changed(self.on_timeline_changed)

        controls = gui.Horiz(0.4 * em)
        prev_button = gui.Button("<")
        prev_button.set_on_clicked(lambda: self.step(-1))
        self.play_button = gui.Button("Play")
        self.play_button.set_on_clicked(self.toggle_play)
        next_button = gui.Button(">")
        next_button.set_on_clicked(lambda: self.step(1))
        restart_button = gui.Button("Restart")
        restart_button.set_on_clicked(lambda: self.jump_to(0))
        jump_a_button = gui.Button("Jump A end")
        jump_a_button.set_on_clicked(lambda: self.jump_to(self.na - 1))
        jump_b_button = gui.Button("Jump B start")
        jump_b_button.set_on_clicked(lambda: self.jump_to(self.na))
        end_button = gui.Button("End")
        end_button.set_on_clicked(lambda: self.jump_to(self.total - 1))
        for widget in (
            prev_button,
            self.play_button,
            next_button,
            restart_button,
            jump_a_button,
            jump_b_button,
            end_button,
        ):
            controls.add_child(widget)

        layer_controls = gui.Horiz(0.8 * em)
        self.trajectory_checkbox = gui.Checkbox("Camera trajectories")
        self.trajectory_checkbox.checked = True
        self.trajectory_checkbox.set_on_checked(self.set_trajectories_visible)
        self.dynamics_checkbox = gui.Checkbox("D1 动态层 (轨迹+时序点云)")
        self.dynamics_checkbox.checked = True
        self.dynamics_checkbox.set_on_checked(self.set_dynamics_visible)
        for widget in (
            self.trajectory_checkbox,
            self.dynamics_checkbox,
        ):
            layer_controls.add_child(widget)

        self.panel.add_child(self.title)
        self.panel.add_child(self.details)
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

        self.pose_a = pose_a
        self.pose_b = pose_b
        self.add_camera_trajectories()

        # Loading progress from the server's stderr -> details label.
        self.title.text = "正在加载地图 (共 ~16 GB)…"
        self.details.text = "准备中…"
        app.post_to_main_thread(
            self.window,
            lambda: self._set_loading_progress("准备中…"),
        )
        self.server._on_progress = self._on_server_progress

        # Background loader: coalesces rapid slider drags to the latest frame.
        self.want_index = 0
        self.loaded_index = -1
        self.rendered_index = -1
        self.camera_fitted = False
        self.want_changed_at = 0.0
        self.stop_loader = False
        self.loader = threading.Thread(target=self.load_loop, daemon=True)
        self.loader.start()

        threading.Thread(target=self.play_loop, daemon=True).start()

    def _on_server_progress(self, line: str) -> None:
        """Server stderr line (any thread): reflect loading progress on the
        details label via the GUI thread."""
        if self.stop_loader:
            return
        app = gui.Application.instance
        if app is not None:
            app.post_to_main_thread(
                self.window, lambda l=line: self._set_loading_progress(l)
            )

    def _set_loading_progress(self, line: str) -> None:
        if self.total == 0:
            self.details.text = line

    def _init_from_meta(self) -> None:
        """Called on the GUI thread once map metadata is available."""
        if self.total != 0:
            return
        self.na = self.server.frames["a"]
        self.nb = self.server.frames["b"]
        self.total = self.na + self.nb
        self.chain_start_ns = self.server.stamps["a"][0]
        # B[0] is the A[-1] continuation frame sharing A's clock; the real B
        # timeline (and RGBD matching) starts at B[1].
        b_stamps = self.server.stamps["b"]
        self.chain_b_start_ns = b_stamps[1] if len(b_stamps) > 1 else b_stamps[0]
        self.chain_end_ns = self.server.stamps["b"][-1]
        self.timeline.set_limits(0, self.total - 1)
        self.title.text = "加载完成,拖动时间轴浏览 0 → A → B"
        self.details.text = ""
        self.want_index = 0
        self.scene_widget.force_redraw()

    # ---- GUI callbacks -------------------------------------------------

    def on_layout(self, context: gui.LayoutContext) -> None:
        rect = self.window.content_rect
        panel_height = 10 * context.theme.font_size
        self.panel.frame = gui.Rect(rect.x, rect.y, rect.width, panel_height)
        scene_rect = gui.Rect(
            rect.x, rect.y + panel_height, rect.width, rect.height - panel_height
        )
        self.scene_widget.frame = scene_rect
        inset_width = min(320, max(240, int(rect.width * 0.22)))
        inset_height = int(inset_width * 2 / 3)
        label_height = int(1.7 * context.theme.font_size)
        gap = 12
        margin = 14
        top = scene_rect.y + margin
        right_x = rect.x + rect.width - margin - inset_width
        left_x = right_x - gap - inset_width
        self.sensor_a_label.frame = gui.Rect(
            left_x, top, inset_width, label_height
        )
        self.sensor_a_image.frame = gui.Rect(
            left_x, top + label_height, inset_width, inset_height
        )
        self.sensor_b_label.frame = gui.Rect(
            right_x, top, inset_width, label_height
        )
        self.sensor_b_image.frame = gui.Rect(
            right_x, top + label_height, inset_width, inset_height
        )

    def on_close(self) -> bool:
        self.playing = False
        self.stop_loader = True
        self.server.shutdown()
        return True

    def on_timeline_changed(self, value: float) -> None:
        if not self.updating_timeline:
            self.jump_to(int(value))

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.text = "Pause" if self.playing else "Play"
        self.last_advance = time.monotonic()

    def step(self, delta: int) -> None:
        self.jump_to(self.index + delta)

    def jump_to(self, index: int) -> None:
        index = max(0, min(self.total - 1, index))
        self.updating_timeline = True
        self.timeline.int_value = index
        self.updating_timeline = False
        self.want_index = index
        # Timestamp of the last requested jump; the loader debounces on it so
        # a slider drag never starts a (slow) mesh request per tick -- the
        # frame only swaps once the drag has settled.
        self.want_changed_at = time.monotonic()

    def advance_if_due(self) -> None:
        if not self.playing:
            return
        now = time.monotonic()
        if now - self.last_advance < self.frame_seconds:
            return
        self.last_advance = now
        # Advance from the last RENDERED frame so playback never skips frames
        # while the server is still catching up.
        if self.rendered_index >= self.total - 1:
            self.playing = False
            self.play_button.text = "Play"
            return
        self.jump_to(self.rendered_index + 1)

    def play_loop(self) -> None:
        app = gui.Application.instance
        while True:
            time.sleep(0.02)
            if self.playing:
                app.post_to_main_thread(self.window, self.advance_if_due)

    def set_trajectories_visible(self, visible: bool) -> None:
        self.show_trajectories = visible
        self.add_camera_trajectories()

    def set_dynamics_visible(self, visible: bool) -> None:
        self.show_dynamics = visible
        app = gui.Application.instance
        app.post_to_main_thread(
            self.window, lambda: self.clear_dynamics()
        )

    # ---- scene content --------------------------------------------------

    def load_loop(self) -> None:
        app = gui.Application.instance
        # Wait for the maps to load before requesting anything (session
        # mapping depends on the metadata; the window meanwhile shows the
        # server's loading progress on stderr).
        if self.server.meta is None:
            try:
                self.server._ensure_meta()
            except Exception as exc:  # server died
                print(f"server request failed: {exc}", flush=True)
                app.post_to_main_thread(self.window, lambda: self.on_load_error())
                return
            app.post_to_main_thread(self.window, self._init_from_meta)
            # The session mapping below must not wait for the async GUI post:
            # with na still 0 the first timeline index would be misrouted to
            # session b and render the heaviest frame instead of A's first.
            self.na = self.server.frames["a"]
            self.nb = self.server.frames["b"]
        while not self.stop_loader:
            index = self.want_index
            if index == self.loaded_index:
                time.sleep(0.03)
                continue
            # Debounce: while the timeline is being dragged (or any rapid
            # jump), wait until the target has been stable for a beat before
            # requesting the mesh. One render per settle, not one per tick.
            if self.want_changed_at and (
                time.monotonic() - self.want_changed_at < 0.35
            ):
                time.sleep(0.05)
                continue
            self.loaded_index = index
            if index < self.na:
                session, local = "a", index
            else:
                session, local = "b", index - self.na
            try:
                payload, ts = self.server.request(session, local)
                dyn = self.server.request_dynamic(session, local)
            except Exception as exc:  # server died
                print(f"server request failed: {exc}", flush=True)
                app.post_to_main_thread(self.window, lambda: self.on_load_error())
                break
            app.post_to_main_thread(
                self.window,
                lambda p=payload, t=ts, i=index, d=dyn: self.render_frame(
                    i, t, p, d
                ),
            )

    def on_load_error(self) -> None:
        self.title.text = "ERROR: mesh server died"
        self.playing = False
        self.play_button.text = "Play"

    def clear_overlays(self) -> None:
        for name in self.overlay_names:
            self.scene_widget.scene.remove_geometry(name)
        self.overlay_names.clear()
        self.overlay_meshes.clear()

    def clear_dynamics(self) -> None:
        for name in self.dynamic_names:
            self.scene_widget.scene.remove_geometry(name)
        self.dynamic_names.clear()

    def render_dynamics(self, nodes, ts) -> None:
        """D1 temporal layer, restored from the last working visualization:
        every track carries its full timestamped trajectory and per-timestamp
        point clouds; the sample at (or just before) the queried time is drawn,
        so a person at time t shows that person's point cloud at time t."""
        self.clear_dynamics()
        if not self.show_dynamics or not nodes:
            return
        query_time_ns = int(ts)
        track_material = rendering.MaterialRecord()
        track_material.shader = "unlitLine"
        track_material.line_width = 2.0
        bbox_material = rendering.MaterialRecord()
        bbox_material.shader = "unlitLine"
        bbox_material.line_width = 2.0
        point_material = rendering.MaterialRecord()
        point_material.shader = "defaultUnlit"
        point_material.point_size = 3.0
        color = [0.95, 0.05, 0.55]
        for node in nodes:
            node_id = node.get("id", 0)
            positions = node.get("positions", [])
            timestamps = [int(value) for value in node.get("timestamps_ns", [])]
            if not positions or len(positions) != len(timestamps):
                continue
            if query_time_ns < timestamps[0]:
                continue
            # Only entities active at the queried time.
            if query_time_ns > timestamps[-1] + 300_000_000:
                continue
            current_index = min(
                bisect.bisect_left(timestamps, query_time_ns),
                len(timestamps) - 1,
            )
            traj = np.asarray(positions, dtype=np.float64)
            visible_positions = traj[: current_index + 1]
            if len(visible_positions) > 1:
                lines = o3d.geometry.LineSet(
                    o3d.utility.Vector3dVector(visible_positions),
                    o3d.utility.Vector2iVector(
                        [[i, i + 1] for i in range(len(visible_positions) - 1)]
                    ),
                )
                lines.colors = o3d.utility.Vector3dVector(
                    [color for _ in range(len(visible_positions) - 1)]
                )
                name = f"dyn_traj_{node_id}"
                self.scene_widget.scene.add_geometry(name, lines, track_material)
                self.dynamic_names.append(name)
            point_frames = node.get("point_frames", [])
            if current_index < len(point_frames) and point_frames[current_index]:
                cloud = np.asarray(point_frames[current_index], dtype=np.float64)
                if cloud.ndim == 2 and len(cloud):
                    pcd = o3d.geometry.PointCloud(
                        o3d.utility.Vector3dVector(cloud)
                    )
                    pcd.paint_uniform_color(color)
                    name = f"dyn_cloud_{node_id}"
                    self.scene_widget.scene.add_geometry(
                        name, pcd, point_material
                    )
                    self.dynamic_names.append(name)
            center = positions[current_index]
            extent = node.get("bbox_dimensions", [0.5, 0.5, 1.7])
            minimum = [center[i] - 0.5 * extent[i] for i in range(3)]
            maximum = [center[i] + 0.5 * extent[i] for i in range(3)]
            bbox = o3d.geometry.AxisAlignedBoundingBox(minimum, maximum)
            bbox.color = color
            bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
            name = f"dyn_bbox_{node_id}"
            self.scene_widget.scene.add_geometry(name, bbox_lines, bbox_material)
            self.dynamic_names.append(name)

    # ---- current-frame RGB insets (VISIT 1 / VISIT 2) -----------------

    def nearest_timestamp_index(self, timestamps, query_time_ns: int) -> int:
        if timestamps is None or len(timestamps) == 0:
            return -1
        insertion = bisect.bisect_left(timestamps, query_time_ns)
        if insertion <= 0:
            return 0
        if insertion >= len(timestamps):
            return len(timestamps) - 1
        before = insertion - 1
        if (
            query_time_ns - timestamps[before]
            <= timestamps[insertion] - query_time_ns
        ):
            return before
        return insertion

    def load_rgb_image(self, path: str):
        image = self.image_cache.pop(path, None)
        if image is None:
            image = o3d.io.read_image(path)
        self.image_cache[path] = image
        while len(self.image_cache) > self.image_cache_limit:
            self.image_cache.popitem(last=False)
        return image

    def update_sensor_slot(self, slot, session, image_index, label) -> None:
        data = self.rgbd_a if session == "A" else self.rgbd_b
        if data is None:
            return
        paths, _ = data
        if image_index < 0 or image_index >= len(paths):
            return
        relative_path = paths[image_index]
        label_widget = (
            self.sensor_a_label if slot == "A" else self.sensor_b_label
        )
        image_widget = (
            self.sensor_a_image if slot == "A" else self.sensor_b_image
        )
        label_widget.text = label
        if self.sensor_slot_paths[slot] == relative_path:
            return
        image = self.load_rgb_image(relative_path)
        if image.is_empty():
            return
        image_widget.update_image(image)
        self.sensor_slot_paths[slot] = relative_path

    def update_rgb_insets(self, index: int, ts: int) -> None:
        """Sync the camera-view RGB insets to the current chain timestamp."""
        session = "A" if index < self.na else "B"
        self.current_session = session
        in_b = session == "B"
        self.sensor_a_label.visible = True
        self.sensor_a_image.visible = True
        self.sensor_b_label.visible = in_b
        self.sensor_b_image.visible = in_b
        for slot, sess in (("A", "A"), ("B", "B")):
            data = self.rgbd_a if sess == "A" else self.rgbd_b
            if data is None:
                continue
            _, stamps = data
            if sess == "B" and not in_b:
                continue
            # 4dmap stamps are Unix-ns; RGBD timestamps.csv is session-relative
            # (starts at 0). Align by each session's first 4dmap stamp.
            offset = self.chain_start_ns if sess == "A" else self.chain_b_start_ns
            idx = self.nearest_timestamp_index(stamps, ts - offset)
            if idx < 0:
                continue
            visit = 1 if sess == "A" else 2
            self.update_sensor_slot(
                slot,
                sess,
                idx,
                f"VISIT {visit} | current RGB | "
                f"t={stamps[idx] * 1.0e-9:.2f}s",
            )
        self.window.set_needs_layout()

    def add_camera_trajectories(self) -> None:
        self.clear_overlays()
        if self.show_trajectories:
            for data, color in (
                (self.pose_a, [0.1, 0.7, 0.95]),
                (self.pose_b, [1.0, 0.55, 0.1]),
            ):
                if data is None:
                    continue
                positions, _ = data
                if len(positions) < 2:
                    continue
                lines = o3d.geometry.LineSet(
                    o3d.utility.Vector3dVector(positions),
                    o3d.utility.Vector2iVector(
                        [[i, i + 1] for i in range(len(positions) - 1)]
                    ),
                )
                lines.colors = o3d.utility.Vector3dVector(
                    [color for _ in range(len(positions) - 1)]
                )
                material = rendering.MaterialRecord()
                material.shader = "unlitLine"
                material.line_width = 2.5
                name = f"trajectory_{len(self.overlay_names)}"
                self.scene_widget.scene.add_geometry(name, lines, material)
                self.overlay_names.append(name)

    def render_frame(self, index: int, ts: int, payload, dyn=None) -> None:
        if self.stop_loader or index != self.loaded_index:
            return
        if payload is None:
            # Snapshot without a mesh: keep the previous geometry on screen.
            label = f"Session {self.session_label(index)} | ts {ts} (no mesh)"
            self.title.text = label
            self.details.text = self.timeline_text(index, ts)
            self.scene_widget.force_redraw()
            return

        # Mesh construction happens on the GUI thread: creating open3d
        # geometry off-thread has proven unreliable in this environment.
        verts, faces, colors = payload
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts.astype(np.float64)),
            o3d.utility.Vector3iVector(faces.astype(np.int32)),
        )
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            colors.astype(np.float64) / 255.0
        )
        # NOTE: no compute_vertex_normals() -- the material is defaultUnlit so
        # normals are unused, and computing them over millions of vertices per
        # frame costs hundreds of ms on the software (llvmpipe) GL path.

        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        try:
            self.scene_widget.scene.remove_geometry("frame_mesh")
            self.scene_widget.scene.add_geometry("frame_mesh", mesh, material)
            self.add_camera_trajectories()
            self.render_dynamics(dyn or [], ts)

            if not self.camera_fitted:
                verts64 = verts.astype(np.float64)
                # 0-vertex or non-finite snapshots (e.g. frames with no mesh)
                # must not crash the camera fit; leave the camera alone.
                if verts64.shape[0] >= 3 and np.all(np.isfinite(verts64)):
                    self.camera_fitted = True
                    bounds = o3d.geometry.AxisAlignedBoundingBox(
                        verts64.min(axis=0), verts64.max(axis=0)
                    )
                    self.scene_widget.setup_camera(
                        60, bounds, bounds.get_center()
                    )

            self.rendered_index = index
            self.title.text = self.session_label(index)
            self.details.text = self.timeline_text(index, ts)
            self.update_rgb_insets(index, ts)
            self.scene_widget.force_redraw()
        except Exception:
            # One failed render (e.g. a display/window quirk) must not kill the
            # player; keep the timeline responsive. Full traceback for
            # diagnosing render failures.
            import traceback

            traceback.print_exc()

    def session_label(self, index: int) -> str:
        if index < self.na:
            return (
                f"Session A | step {index + 1}/{self.na} "
                f"| timeline {index + 1}/{self.total}"
            )
        local = index - self.na
        return (
            f"Session B | step {local + 1}/{self.nb} "
            f"| timeline {index + 1}/{self.total}"
        )

    def timeline_text(self, index: int, ts: int) -> str:
        rel_ns = ts - self.chain_start_ns
        return f"t = +{rel_ns * 1.0e-9:.2f} s of chain  | ts {ts}"


def _on_sigterm(signum, frame):
    # A killed GUI app must not run Open3D teardown while loader/play threads
    # are still posting to a destroyed window (pure virtual call crash).
    os._exit(0)


def main() -> None:
    signal.signal(signal.SIGTERM, _on_sigterm)
    default_root = Path(
        "/home/jixian/Desktop/FT/session_update_baseline_project/"
        "session_update_baseline/runs"
    )
    parser = argparse.ArgumentParser(description="0->A->B chain viewer")
    parser.add_argument(
        "--map-a",
        type=Path,
        default=default_root / "v4_a/final.4dmap",
    )
    parser.add_argument(
        "--map-b",
        type=Path,
        default=default_root / "v32_b/final.4dmap",
    )
    parser.add_argument(
        "--pose-a",
        type=Path,
        default=Path(
            "/home/jixian/Desktop/FT/datasets/local_ab/rgbd/"
            "session_a_20260809_204010_201_flat_rgbd_30hz_1080p"
        ),
    )
    parser.add_argument(
        "--pose-b",
        type=Path,
        default=Path(
            "/home/jixian/Desktop/FT/datasets/local_ab/rgbd/"
            "session_b_20260810_030502_620_flat_rgbd_30hz_1080p"
        ),
    )
    parser.add_argument("--frame-seconds", type=float, default=0.4)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="vertex stride for the server-side decimation; higher = "
        "lighter/laggier-free but coarser. 1 = full resolution.",
    )
    args = parser.parse_args()

    server = MapServer(str(args.map_a), str(args.map_b), stride=args.stride)
    pose_a = load_pose_trajectory(args.pose_a)
    pose_b = load_pose_trajectory(args.pose_b)
    rgbd_a = load_rgbd_images(args.pose_a)
    rgbd_b = load_rgbd_images(args.pose_b)

    app = gui.Application.instance
    app.initialize()
    player = ChainPlayer(
        server, pose_a, pose_b, args.frame_seconds, rgbd_a, rgbd_b
    )
    try:
        app.run()
    except KeyboardInterrupt:
        # Ctrl+C while the maps are still loading must also kill the resident
        # 16 GB map server, not leave it hanging around.
        player.on_close()
        server.shutdown()
        raise


if __name__ == "__main__":
    main()
