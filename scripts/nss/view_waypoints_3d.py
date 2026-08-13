#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--waypoints", required=True, type=Path)
    parser.add_argument("--ply", type=Path, help="Optional point cloud to display.")
    parser.add_argument("--mesh", type=Path, help="Optional mesh to display.")
    parser.add_argument("--max-points", type=int, default=180000)
    parser.add_argument("--frustum-depth", type=float, default=0.8)
    parser.add_argument("--frustum-width", type=float, default=0.45)
    parser.add_argument("--frustum-height", type=float, default=0.32)
    parser.add_argument("--display-stride", type=int, default=1, help="Show every Nth camera marker/frustum while keeping the full path line.")
    return parser.parse_args()


def read_waypoint_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    if "waypoints" not in data or not data["waypoints"]:
        raise SystemExit(f"{path}: no waypoints")
    return data


def make_line_set(points: np.ndarray, color: tuple[float, float, float], closed: bool = False) -> o3d.geometry.LineSet:
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    if closed and len(points) > 2:
        lines.append([len(points) - 1, 0])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color), (len(lines), 1)))
    return line_set


def make_spheres(points: np.ndarray, radius: float, color: tuple[float, float, float]) -> list[o3d.geometry.TriangleMesh]:
    out = []
    for point in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
        sphere.translate(point)
        sphere.paint_uniform_color(color)
        out.append(sphere)
    return out


def make_frustum(origin: np.ndarray, yaw: float, pitch: float, depth: float, width: float, height: float) -> o3d.geometry.LineSet:
    forward = np.array([
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
    ])
    forward = forward / np.linalg.norm(forward)
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    center = origin + forward * depth
    corners = np.array([
        center - right * width - up * height,
        center + right * width - up * height,
        center + right * width + up * height,
        center - right * width + up * height,
    ])
    pts = np.vstack([origin, corners])
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array([0.1, 0.3, 1.0]), (len(lines), 1)))
    return ls


def waypoint_positions(data: dict) -> np.ndarray:
    z = float(data.get("camera_z", 1.25))
    pts = np.array([[float(w["x"]), float(w["y"]), z] for w in data["waypoints"]], dtype=float)
    return pts


def waypoint_yaws(points: np.ndarray, data: dict) -> np.ndarray:
    stored = [row.get("yaw_rad") for row in data["waypoints"]]
    if all(value is not None for value in stored):
        return np.asarray(stored, dtype=float)
    yaws = []
    for i, point in enumerate(points):
        if i + 1 < len(points):
            direction = points[i + 1] - point
        elif i > 0:
            direction = point - points[i - 1]
        else:
            direction = np.array([1.0, 0.0, 0.0])
        yaws.append(math.atan2(direction[1], direction[0]))
    return np.asarray(yaws)


def main() -> None:
    args = parse_args()
    data = read_waypoint_json(args.waypoints)
    waypoints = waypoint_positions(data)
    yaws = waypoint_yaws(waypoints, data)
    pitch = math.radians(float(data.get("pitch_deg", -5.0)))

    geoms: list[o3d.geometry.Geometry] = []
    if args.mesh:
        mesh = o3d.io.read_triangle_mesh(str(args.mesh))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.65, 0.65, 0.65])
        geoms.append(mesh)
    if args.ply:
        pcd = o3d.io.read_point_cloud(str(args.ply))
        if len(pcd.points) > args.max_points:
            pcd = pcd.random_down_sample(args.max_points / len(pcd.points))
        pcd.paint_uniform_color([0.05, 0.05, 0.05])
        geoms.append(pcd)

    geoms.append(make_line_set(waypoints, (1.0, 0.0, 0.0), closed=bool(data.get("closed", False))))
    stride = max(1, args.display_stride)
    displayed_points = waypoints[::stride]
    displayed_yaws = yaws[::stride]
    geoms.extend(make_spheres(displayed_points, radius=0.08, color=(0.0, 0.8, 0.1)))
    for point, yaw in zip(displayed_points, displayed_yaws):
        geoms.append(make_frustum(point, yaw, pitch, args.frustum_depth, args.frustum_width, args.frustum_height))

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    geoms.append(axes)

    print(f"WAYPOINTS {len(waypoints)}")
    print(f"DISPLAYED_CAMERAS {len(displayed_points)} stride={stride}")
    print(f"BBOX min={waypoints.min(axis=0)} max={waypoints.max(axis=0)}")
    print("Green spheres = camera centers; red line = waypoint path; blue pyramids = view direction.")
    o3d.visualization.draw_geometries(geoms, window_name=f"Waypoints 3D: {args.waypoints.name}")


if __name__ == "__main__":
    main()
