#!/usr/bin/env python3
"""Render NSS stage meshes into a Panoptic-Mapping flat-style RGB-D dataset.

This is an adapter/probing tool, not an official NSS benchmark step. It renders
diagnostic meshes produced from GT-aligned NSS fragments into virtual camera
observations. The first useful sanity check is simply whether Panoptic's flat
player can ingest these images without depth/pose/time issues.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


STAGE_COLORS = {
    1: np.array([230, 57, 70], dtype=np.uint8),
    2: np.array([42, 157, 143], dtype=np.uint8),
    3: np.array([69, 123, 157], dtype=np.uint8),
    4: np.array([244, 162, 97], dtype=np.uint8),
    5: np.array([131, 56, 236], dtype=np.uint8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-dir", type=Path)
    parser.add_argument(
        "--mesh-transform",
        type=Path,
        help="Optional 4x4 transform applied to every input mesh before rendering.",
    )
    parser.add_argument(
        "--mesh",
        action="append",
        default=[],
        metavar="STAGE=PATH",
        help="Explicit stage mesh. May be repeated and supports official NSS OBJ files.",
    )
    parser.add_argument("--stages", nargs="+", type=int, help="Optional stage IDs to render, e.g. --stages 5 or --stages 1 2 5.")
    parser.add_argument("--graph", type=Path, help="NSS pose graph JSON. Required for --camera-mode nss_nodes/nss_path.")
    parser.add_argument("--graph-scene", default="Bldg3_Scene1", help="Scene name when the official graph JSON contains a scene list.")
    parser.add_argument("--node-stages", nargs="+", type=int, help="Only use graph nodes from these stages as camera anchors.")
    parser.add_argument(
        "--graph-axis-signs",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        metavar=("SX", "SY", "SZ"),
        help="Axis mapping from NSS graph coordinates to raw-mesh coordinates.",
    )
    parser.add_argument(
        "--graph-position-offset",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="Translation added after graph-axis mapping; use to place virtual cameras in raw-mesh coordinates.",
    )
    parser.add_argument("--waypoints", type=Path, help="Manual waypoint JSON. Required for --camera-mode manual_waypoints.")
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--run-name", default="run_virtual")
    parser.add_argument("--camera-mode", choices=["orbit", "nss_nodes", "nss_path", "manual_waypoints"], default="orbit")
    parser.add_argument("--views-per-node", type=int, default=4)
    parser.add_argument("--path-min-spacing", type=float, default=0.35)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fx", type=float, default=420.0)
    parser.add_argument("--fy", type=float, default=420.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=20.0)
    parser.add_argument("--cameras", type=int, default=8)
    parser.add_argument("--height-offset", type=float, default=1.6)
    parser.add_argument("--radius-scale", type=float, default=0.85)
    parser.add_argument("--timestamp-step-ns", type=int, default=500_000_000)
    parser.add_argument(
        "--pose-stride",
        type=int,
        default=1,
        help="Render every Nth generated pose. Useful for an end-to-end smoke test.",
    )
    parser.add_argument("--segmentation-mode", choices=["single", "stage"], default="single")
    parser.add_argument("--filter-poses", action="store_true")
    parser.add_argument("--filter-min-valid-ratio", type=float, default=0.03)
    parser.add_argument("--filter-max-valid-ratio", type=float, default=0.95)
    parser.add_argument("--filter-min-median-depth", type=float, default=0.2)
    parser.add_argument("--filter-max-median-depth", type=float, default=15.0)
    parser.add_argument(
        "--manual-yaw-offsets-deg",
        nargs="+",
        type=float,
        default=[0.0],
        help=(
            "Extra yaw offsets for --camera-mode manual_waypoints. "
            "For example: --manual-yaw-offsets-deg 0 -90 90 180."
        ),
    )
    parser.add_argument(
        "--manual-yaw-sweep-amplitude-deg",
        type=float,
        default=0.0,
        help=(
            "If nonzero for --camera-mode manual_waypoints, replace discrete yaw offsets "
            "with a smooth sinusoidal yaw sweep along path distance."
        ),
    )
    parser.add_argument(
        "--manual-yaw-sweep-period-m",
        type=float,
        default=3.0,
        help="Path distance in meters for one full smooth yaw sweep cycle.",
    )
    parser.add_argument(
        "--filter-require-all-stages",
        action="store_true",
        help="Keep a pose only if every rendered stage satisfies valid-ratio/depth thresholds.",
    )
    return parser.parse_args()


def parse_stage(path: Path) -> int:
    # stage_5_mesh.ply -> 5
    parts = path.stem.split("_")
    if len(parts) < 2 or parts[0] != "stage":
        raise ValueError(f"cannot parse stage from {path}")
    return int(parts[1])


def parse_explicit_meshes(values: list[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--mesh expects STAGE=PATH, got: {value}")
        stage_text, path_text = value.split("=", 1)
        stage = int(stage_text)
        path = Path(path_text)
        if not path.is_file():
            raise SystemExit(f"stage {stage} mesh does not exist: {path}")
        result[stage] = path
    return result


def read_meshes(
    mesh_dir: Path | None,
    explicit_meshes: dict[int, Path],
    mesh_transform: np.ndarray | None = None,
    keep_stages: set[int] | None = None,
) -> tuple[dict[int, o3d.geometry.TriangleMesh], dict[int, str]]:
    meshes = {}
    sources = dict(explicit_meshes)
    if mesh_dir is not None:
        for path in sorted(mesh_dir.glob("stage_*_mesh.ply")):
            sources.setdefault(parse_stage(path), path)
    for stage, path in sorted(sources.items()):
        if keep_stages is not None and stage not in keep_stages:
            continue
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.triangles) == 0:
            raise ValueError(f"{path}: empty mesh")
        if mesh_transform is not None:
            mesh.transform(mesh_transform)
        mesh.compute_vertex_normals()
        meshes[stage] = mesh
    if not meshes:
        suffix = "" if keep_stages is None else f" for stages {sorted(keep_stages)}"
        raise SystemExit(f"No input meshes found{suffix}; use --mesh-dir or repeated --mesh STAGE=PATH")
    return meshes, {stage: str(sources[stage]) for stage in meshes}


def combined_bbox(meshes: dict[int, o3d.geometry.TriangleMesh]) -> o3d.geometry.AxisAlignedBoundingBox:
    points = []
    for mesh in meshes.values():
        points.append(np.asarray(mesh.vertices))
    all_points = np.vstack(points)
    return o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.min(all_points, axis=0),
        max_bound=np.max(all_points, axis=0),
    )


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return world_T_camera with OpenCV camera axes: x right, y down, z forward."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = down
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


def yaw_pose(eye: np.ndarray, yaw: float) -> np.ndarray:
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    return look_at(eye, eye + forward)


def yaw_pitch_pose(eye: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    forward = np.array([
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
    ])
    return look_at(eye, eye + forward)


def generate_poses(bbox: o3d.geometry.AxisAlignedBoundingBox, count: int, height_offset: float, radius_scale: float) -> list[np.ndarray]:
    center = bbox.get_center()
    extent = bbox.get_extent()
    radius = max(float(extent[0]), float(extent[1])) * radius_scale
    z = float(bbox.min_bound[2] + height_offset)
    target = center.copy()
    target[2] = min(float(center[2]), z)

    poses = []
    for i in range(count):
        angle = 2.0 * math.pi * i / count
        eye = np.array([
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
            z,
        ])
        poses.append(look_at(eye, target))
    return poses


def load_graph_scene(graph_path: Path, scene_name: str) -> dict:
    data = json.loads(graph_path.read_text())
    if isinstance(data, dict):
        return data
    if not isinstance(data, list):
        raise SystemExit(f"Unsupported graph JSON root in {graph_path}: {type(data).__name__}")
    matches = [scene for scene in data if scene.get("name") == scene_name]
    if len(matches) != 1:
        names = [scene.get("name") for scene in data]
        raise SystemExit(f"Scene {scene_name!r} not found uniquely in {graph_path}; available={names}")
    return matches[0]


def node_transform(node: dict) -> np.ndarray:
    value = node.get("global_transform", node.get("tsfm"))
    if value is None:
        raise SystemExit(f"Node {node.get('id')} has neither global_transform nor tsfm")
    return np.asarray(value, dtype=np.float64)


def load_nss_node_poses(
    graph_path: Path,
    views_per_node: int,
    scene_name: str,
    node_stages: set[int] | None,
    axis_signs: np.ndarray,
    position_offset: np.ndarray,
) -> tuple[list[np.ndarray], list[dict]]:
    if graph_path is None:
        raise SystemExit("--graph is required for --camera-mode nss_nodes")
    graph = load_graph_scene(graph_path, scene_name)
    poses = []
    metadata = []
    yaws = [2.0 * math.pi * i / views_per_node for i in range(views_per_node)]
    for node in graph["nodes"]:
        stage = int(node["stage"])
        if node_stages is not None and stage not in node_stages:
            continue
        transform = node_transform(node)
        if np.allclose(transform, 0.0):
            continue
        eye = transform[:3, 3] * axis_signs + position_offset
        for view_index, yaw in enumerate(yaws):
            poses.append(yaw_pose(eye, yaw))
            metadata.append({
                "source": "nss_node",
                "node_id": int(node["id"]),
                "node_name": node["name"],
                "node_stage": stage,
                "view_index": int(view_index),
                "yaw_rad": float(yaw),
            })
    return poses, metadata


def load_nss_path_poses(
    graph_path: Path,
    min_spacing: float,
    scene_name: str,
    node_stages: set[int] | None,
    axis_signs: np.ndarray,
    position_offset: np.ndarray,
) -> tuple[list[np.ndarray], list[dict]]:
    if graph_path is None:
        raise SystemExit("--graph is required for --camera-mode nss_path")
    graph = load_graph_scene(graph_path, scene_name)
    nodes = []
    for node in graph["nodes"]:
        stage = int(node["stage"])
        if node_stages is not None and stage not in node_stages:
            continue
        transform = node_transform(node)
        if np.allclose(transform, 0.0):
            continue
        nodes.append({
            "id": int(node["id"]),
            "name": node["name"],
            "stage": stage,
            "position": transform[:3, 3] * axis_signs + position_offset,
        })
    if len(nodes) < 2:
        raise SystemExit("Need at least two inlier nodes for nss_path")

    # Greedy nearest-neighbor ordering gives a continuous inspection path from
    # unordered NSS graph nodes. This is a camera-path adapter, not NSS GT odom.
    remaining = set(range(len(nodes)))
    current = min(remaining, key=lambda i: (nodes[i]["position"][0], nodes[i]["position"][1]))
    order = [current]
    remaining.remove(current)
    while remaining:
        cur_pos = nodes[current]["position"]
        current = min(remaining, key=lambda i: float(np.linalg.norm(nodes[i]["position"][:2] - cur_pos[:2])))
        order.append(current)
        remaining.remove(current)

    filtered_order = []
    last_pos = None
    for idx in order:
        pos = nodes[idx]["position"]
        if last_pos is None or np.linalg.norm(pos[:2] - last_pos[:2]) >= min_spacing:
            filtered_order.append(idx)
            last_pos = pos

    poses = []
    metadata = []
    for path_index, idx in enumerate(filtered_order):
        pos = nodes[idx]["position"]
        if path_index + 1 < len(filtered_order):
            next_pos = nodes[filtered_order[path_index + 1]]["position"]
            direction = next_pos - pos
        elif path_index > 0:
            prev_pos = nodes[filtered_order[path_index - 1]]["position"]
            direction = pos - prev_pos
        else:
            direction = np.array([1.0, 0.0, 0.0])
        yaw = math.atan2(direction[1], direction[0])
        poses.append(yaw_pose(pos, yaw))
        metadata.append({
            "source": "nss_path",
            "path_index": int(path_index),
            "node_id": nodes[idx]["id"],
            "node_name": nodes[idx]["name"],
            "node_stage": nodes[idx]["stage"],
            "yaw_rad": float(yaw),
        })
    return poses, metadata


def load_manual_waypoint_poses(
    path: Path,
    yaw_offsets_deg: list[float],
    yaw_sweep_amplitude_deg: float,
    yaw_sweep_period_m: float,
) -> tuple[list[np.ndarray], list[dict]]:
    if path is None:
        raise SystemExit("--waypoints is required for --camera-mode manual_waypoints")
    data = json.loads(path.read_text())
    waypoints = data.get("waypoints", [])
    if len(waypoints) < 2:
        raise SystemExit(f"{path}: need at least two waypoints")

    z = float(data.get("camera_z", 1.25))
    step = float(data.get("step_m", 0.35))
    pitch = math.radians(float(data.get("pitch_deg", -5.0)))
    xy = np.asarray([[float(w["x"]), float(w["y"])] for w in waypoints], dtype=np.float64)

    if data.get("preinterpolated", False):
        samples = [point for point in xy]
        segment_ids = list(range(len(xy)))
        distances = [float(row.get("distance_m", 0.0)) for row in waypoints]
        if not any(distances[1:]):
            distances = [0.0]
            for start, end in zip(xy, xy[1:]):
                distances.append(distances[-1] + float(np.linalg.norm(end - start)))
    else:
        samples = []
        segment_ids = []
        distances = []
        path_distance = 0.0
        for i in range(len(xy) - 1):
            start = xy[i]
            end = xy[i + 1]
            delta = end - start
            length = float(np.linalg.norm(delta))
            if length < 1e-6:
                continue
            n = max(1, int(math.ceil(length / step)))
            for j in range(n):
                if samples and j == 0:
                    continue
                alpha = j / n
                samples.append(start * (1.0 - alpha) + end * alpha)
                segment_ids.append(i)
                distances.append(path_distance + alpha * length)
            path_distance += length
        samples.append(xy[-1])
        segment_ids.append(len(xy) - 2)
        distances.append(path_distance)

    poses = []
    metadata = []
    samples_arr = np.asarray(samples)
    smooth_sweep = abs(float(yaw_sweep_amplitude_deg)) > 1e-9
    if smooth_sweep and yaw_sweep_period_m <= 0.0:
        raise SystemExit("--manual-yaw-sweep-period-m must be positive")
    yaw_offsets = [math.radians(float(offset)) for offset in yaw_offsets_deg]
    for i, point_xy in enumerate(samples_arr):
        if i + 1 < len(samples_arr):
            direction = samples_arr[i + 1] - point_xy
        elif i > 0:
            direction = point_xy - samples_arr[i - 1]
        else:
            direction = np.array([1.0, 0.0])
        stored_yaw = waypoints[i].get("yaw_rad") if i < len(waypoints) else None
        base_yaw = float(stored_yaw) if stored_yaw is not None else math.atan2(direction[1], direction[0])
        eye = np.array([point_xy[0], point_xy[1], z], dtype=np.float64)
        if smooth_sweep:
            offset_deg = float(yaw_sweep_amplitude_deg) * math.sin(
                2.0 * math.pi * distances[i] / float(yaw_sweep_period_m)
            )
            offset_rad = math.radians(offset_deg)
            yaw = base_yaw + offset_rad
            poses.append(yaw_pitch_pose(eye, yaw, pitch))
            metadata.append({
                "source": "manual_waypoints",
                "waypoint_file": str(path),
                "sample_index": int(i),
                "segment_index": int(segment_ids[i]),
                "path_distance_m": float(distances[i]),
                "base_yaw_rad": float(base_yaw),
                "yaw_mode": "smooth_sine",
                "yaw_sweep_amplitude_deg": float(yaw_sweep_amplitude_deg),
                "yaw_sweep_period_m": float(yaw_sweep_period_m),
                "yaw_offset_deg": float(offset_deg),
                "yaw_rad": float(yaw),
                "pitch_rad": float(pitch),
            })
            continue

        for offset_deg, offset_rad in zip(yaw_offsets_deg, yaw_offsets):
            yaw = base_yaw + offset_rad
            poses.append(yaw_pitch_pose(eye, yaw, pitch))
            metadata.append({
                "source": "manual_waypoints",
                "waypoint_file": str(path),
                "sample_index": int(i),
                "segment_index": int(segment_ids[i]),
                "path_distance_m": float(distances[i]),
                "base_yaw_rad": float(base_yaw),
                "yaw_mode": "discrete_offsets",
                "yaw_offset_deg": float(offset_deg),
                "yaw_rad": float(yaw),
                "pitch_rad": float(pitch),
            })
    return poses, metadata


def make_rays(pose: np.ndarray, width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> o3d.core.Tensor:
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    dirs_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    rot = pose[:3, :3].astype(np.float32)
    trans = pose[:3, 3].astype(np.float32)
    dirs_world = dirs_cam @ rot.T
    origins = np.broadcast_to(trans, dirs_world.shape)
    rays = np.concatenate([origins, dirs_world], axis=-1).astype(np.float32)
    return o3d.core.Tensor(rays)


def build_scene(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def render_mesh(
    scene: o3d.t.geometry.RaycastingScene,
    pose: np.ndarray,
    stage: int,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    near: float,
    far: float,
    seg_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    ans = scene.cast_rays(make_rays(pose, width, height, fx, fy, cx, cy))
    depth = ans["t_hit"].numpy().astype(np.float32)
    hit = np.isfinite(depth) & (depth >= near) & (depth <= far)
    depth_out = np.where(hit, depth, 0.0).astype(np.float32)

    color = np.full((height, width, 3), 255, dtype=np.uint8)
    base = STAGE_COLORS.get(stage, np.array([180, 180, 180], dtype=np.uint8))
    if np.any(hit):
        shade = np.clip(1.0 - depth_out / far, 0.25, 1.0)
        shaded = (base[None, None, :].astype(np.float32) * shade[:, :, None]).astype(np.uint8)
        color[hit] = shaded[hit]

    seg = np.zeros((height, width), dtype=np.uint8)
    seg[hit] = seg_id
    stats = {
        "hit_pixels": int(np.count_nonzero(hit)),
        "hit_ratio": float(np.mean(hit)),
        "depth_min": float(depth_out[hit].min()) if np.any(hit) else None,
        "depth_median": float(np.median(depth_out[hit])) if np.any(hit) else None,
        "depth_max": float(depth_out[hit].max()) if np.any(hit) else None,
    }
    return color, depth_out, seg, stats


def write_pose(path: Path, pose: np.ndarray) -> None:
    with path.open("w") as f:
        for row in range(4):
            f.write(" ".join(f"{pose[row, col]:.9f}" for col in range(4)))
            f.write("\n")


def write_label_csv(path: Path, stages: list[int], segmentation_mode: str) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["InstanceID", "ClassID", "PanopticID", "MeshID", "InfraredID", "R", "G", "B", "Name", "Size"])
        writer.writerow([0, 0, 0, 0, 0, 255, 255, 255, "Background", "XL"])
        if segmentation_mode == "single":
            writer.writerow([1, 1, 0, 1, 1, 180, 180, 180, "NSS_Surface", "XL"])
        else:
            for idx, stage in enumerate(stages, start=1):
                color = STAGE_COLORS.get(stage, np.array([180, 180, 180], dtype=np.uint8))
                writer.writerow([idx, idx, 0, idx, idx, int(color[0]), int(color[1]), int(color[2]), f"NSS_Stage_{stage}", "XL"])


def write_class_csv(path: Path, stages: list[int], segmentation_mode: str) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ClassID", "Name"])
        writer.writerow([0, "Background"])
        if segmentation_mode == "single":
            writer.writerow([1, "NSS Surface"])
        else:
            for idx, stage in enumerate(stages, start=1):
                writer.writerow([idx, f"NSS Stage {stage}"])


def write_intrinsics(path: Path, args: argparse.Namespace) -> None:
    path.write_text(
        "\n".join(
            [
                f"Res_x: {args.width}",
                f"Res_y: {args.height}",
                f"u: {args.cx}",
                f"v: {args.cy}",
                f"f_x: {args.fx}",
                f"f_y: {args.fy}",
            ]
        )
        + "\n"
    )


def select_poses(
    scenes: dict[int, o3d.t.geometry.RaycastingScene],
    poses: list[np.ndarray],
    pose_metadata: list[dict],
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[dict]]:
    if not args.filter_poses:
        return poses, [{**meta, "kept": True, "filter_reason": "filter_disabled"} for meta in pose_metadata]

    kept_poses = []
    camera_manifest = []
    for pose_id, (pose, meta) in enumerate(zip(poses, pose_metadata)):
        per_stage = []
        for stage in sorted(scenes):
            seg_id = 1 if args.segmentation_mode == "single" else sorted(scenes).index(stage) + 1
            _, _, _, stats = render_mesh(
                scenes[stage],
                pose,
                stage,
                args.width,
                args.height,
                args.fx,
                args.fy,
                args.cx,
                args.cy,
                args.near,
                args.far,
                seg_id,
            )
            per_stage.append({"stage": stage, **stats})

        valid_ratios = np.asarray([s["hit_ratio"] for s in per_stage], dtype=float)
        med_depths = np.asarray([
            s["depth_median"] if s["depth_median"] is not None else np.inf
            for s in per_stage
        ], dtype=float)
        aggregate_valid = float(np.median(valid_ratios))
        aggregate_depth = float(np.median(med_depths[np.isfinite(med_depths)])) if np.any(np.isfinite(med_depths)) else None
        finite_depths = np.isfinite(med_depths)
        per_stage_ok = (
            (valid_ratios >= args.filter_min_valid_ratio)
            & (valid_ratios <= args.filter_max_valid_ratio)
            & finite_depths
            & (med_depths >= args.filter_min_median_depth)
            & (med_depths <= args.filter_max_median_depth)
        )
        aggregate_ok = (
            aggregate_valid >= args.filter_min_valid_ratio
            and aggregate_valid <= args.filter_max_valid_ratio
            and aggregate_depth is not None
            and args.filter_min_median_depth <= aggregate_depth <= args.filter_max_median_depth
        )
        kept = bool(np.all(per_stage_ok)) if args.filter_require_all_stages else bool(aggregate_ok)
        reason = "kept"
        if aggregate_valid < args.filter_min_valid_ratio:
            reason = "low_valid_ratio"
        elif aggregate_valid > args.filter_max_valid_ratio:
            reason = "high_valid_ratio"
        elif aggregate_depth is None:
            reason = "no_depth"
        elif aggregate_depth < args.filter_min_median_depth:
            reason = "too_close"
        elif aggregate_depth > args.filter_max_median_depth:
            reason = "too_far"
        elif args.filter_require_all_stages and not np.all(per_stage_ok):
            reason = "some_stage_failed"

        entry = {
            "pose_id": pose_id,
            **meta,
            "position": pose[:3, 3].tolist(),
            "kept": bool(kept),
            "filter_reason": reason,
            "aggregate_valid_ratio_median": aggregate_valid,
            "aggregate_depth_median": aggregate_depth,
            "per_stage_ok": per_stage_ok.tolist(),
            "per_stage": per_stage,
        }
        camera_manifest.append(entry)
        if kept:
            kept_poses.append(pose)

    if not kept_poses:
        raise SystemExit("Pose filtering removed every candidate pose.")
    return kept_poses, camera_manifest


def main() -> None:
    args = parse_args()
    mesh_transform = None
    if args.mesh_transform is not None:
        mesh_transform = np.loadtxt(args.mesh_transform, dtype=np.float64).reshape(4, 4)
        if not np.all(np.isfinite(mesh_transform)) or not np.allclose(
            mesh_transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8
        ):
            raise SystemExit(f"Invalid mesh transform: {args.mesh_transform}")
    meshes, mesh_sources = read_meshes(
        args.mesh_dir,
        parse_explicit_meshes(args.mesh),
        mesh_transform,
        set(args.stages) if args.stages else None,
    )
    bbox = combined_bbox(meshes)
    scenes = {stage: build_scene(mesh) for stage, mesh in meshes.items()}
    if args.camera_mode == "orbit":
        poses = generate_poses(bbox, args.cameras, args.height_offset, args.radius_scale)
        pose_metadata = [
            {"source": "orbit", "pose_index": i}
            for i in range(len(poses))
        ]
    elif args.camera_mode == "nss_nodes":
        poses, pose_metadata = load_nss_node_poses(
            args.graph,
            args.views_per_node,
            args.graph_scene,
            set(args.node_stages) if args.node_stages else None,
            np.asarray(args.graph_axis_signs, dtype=np.float64),
            np.asarray(args.graph_position_offset, dtype=np.float64),
        )
    elif args.camera_mode == "nss_path":
        poses, pose_metadata = load_nss_path_poses(
            args.graph,
            args.path_min_spacing,
            args.graph_scene,
            set(args.node_stages) if args.node_stages else None,
            np.asarray(args.graph_axis_signs, dtype=np.float64),
            np.asarray(args.graph_position_offset, dtype=np.float64),
        )
    else:
        poses, pose_metadata = load_manual_waypoint_poses(
            args.waypoints,
            args.manual_yaw_offsets_deg,
            args.manual_yaw_sweep_amplitude_deg,
            args.manual_yaw_sweep_period_m,
        )
    if args.pose_stride < 1:
        raise SystemExit("--pose-stride must be >= 1")
    poses = poses[:: args.pose_stride]
    pose_metadata = pose_metadata[:: args.pose_stride]
    poses, camera_manifest = select_poses(scenes, poses, pose_metadata, args)

    run_dir = args.out_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    stages = sorted(meshes.keys())
    write_label_csv(args.out_root / "groundtruth_labels.csv", stages, args.segmentation_mode)
    write_class_csv(
        args.out_root / "groundtruth_labels_classes.csv", stages, args.segmentation_mode
    )
    write_intrinsics(args.out_root / "Intrinsics.txt", args)

    timestamps = []
    manifest = {
        "source_mesh_dir": str(args.mesh_dir) if args.mesh_dir else None,
        "source_meshes": mesh_sources,
        "mesh_transform_file": str(args.mesh_transform) if args.mesh_transform else None,
        "mesh_transform": mesh_transform.tolist() if mesh_transform is not None else None,
        "graph": str(args.graph) if args.graph else None,
        "graph_scene": args.graph_scene,
        "node_stages": args.node_stages,
        "graph_axis_signs": args.graph_axis_signs,
        "graph_position_offset": args.graph_position_offset,
        "run_dir": str(run_dir),
        "stages": sorted(meshes.keys()),
        "camera_mode": args.camera_mode,
        "segmentation_mode": args.segmentation_mode,
        "filter_poses": args.filter_poses,
        "pose_stride": args.pose_stride,
        "camera": {
            "width": args.width,
            "height": args.height,
            "fx": args.fx,
            "fy": args.fy,
            "cx": args.cx,
            "cy": args.cy,
            "near": args.near,
            "far": args.far,
            "poses_per_stage": len(poses),
        },
        "bbox_min": bbox.min_bound.tolist(),
        "bbox_max": bbox.max_bound.tolist(),
        "frames": [],
    }

    frame_id = 0
    timestamp_ns = 1_000_000_000
    for stage in sorted(meshes):
        seg_id = 1 if args.segmentation_mode == "single" else stages.index(stage) + 1
        for pose_index, pose in enumerate(poses):
            image_id = f"{frame_id:06d}"
            color, depth, seg, stats = render_mesh(
                scenes[stage],
                pose,
                stage,
                args.width,
                args.height,
                args.fx,
                args.fy,
                args.cx,
                args.cy,
                args.near,
                args.far,
                seg_id,
            )
            Image.fromarray(color, mode="RGB").save(run_dir / f"{image_id}_color.png")
            Image.fromarray(depth, mode="F").save(run_dir / f"{image_id}_depth.tiff")
            Image.fromarray(seg, mode="L").save(run_dir / f"{image_id}_segmentation.png")
            write_pose(run_dir / f"{image_id}_pose.txt", pose)
            timestamps.append((image_id, timestamp_ns))
            manifest["frames"].append({
                "image_id": image_id,
                "stage": stage,
                "pose_index": pose_index,
                "seg_id": seg_id,
                **stats,
            })
            print(
                f"FRAME {image_id} stage={stage} pose={pose_index} "
                f"hit_ratio={stats['hit_ratio']:.3f} depth={stats['depth_min']}..{stats['depth_max']}"
            )
            frame_id += 1
            timestamp_ns += args.timestamp_step_ns

    with (run_dir / "timestamps.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ImageID", "TimeStamp"])
        writer.writerows(timestamps)

    (args.out_root / "render_manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.out_root / "camera_manifest.json").write_text(json.dumps(camera_manifest, indent=2))
    print(f"WROTE {run_dir}")
    print(f"WROTE {args.out_root / 'groundtruth_labels.csv'}")
    print(f"WROTE {args.out_root / 'groundtruth_labels_classes.csv'}")
    print(f"WROTE {args.out_root / 'Intrinsics.txt'}")
    print(f"WROTE {args.out_root / 'render_manifest.json'}")
    print(f"WROTE {args.out_root / 'camera_manifest.json'}")


if __name__ == "__main__":
    main()
