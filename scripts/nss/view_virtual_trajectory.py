#!/usr/bin/env python3
"""View NSS virtual RGB-D camera poses as frustums over stage meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from render_virtual_flat_dataset import load_nss_node_poses


STAGE_COLORS = {
    1: [0.90, 0.22, 0.27],
    2: [0.16, 0.62, 0.56],
    3: [0.27, 0.48, 0.62],
    4: [0.96, 0.64, 0.38],
    5: [0.51, 0.22, 0.93],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-dir", type=Path)
    parser.add_argument("--mesh", action="append", default=[], metavar="STAGE=PATH")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--camera-manifest", type=Path, help="Show all candidate poses, including filtered poses.")
    parser.add_argument("--graph", type=Path, help="Official NSS pose graph JSON.")
    parser.add_argument("--graph-scene", default="Bldg3_Scene1")
    parser.add_argument("--node-stage", type=int)
    parser.add_argument("--views-per-node", type=int, default=1)
    parser.add_argument("--graph-axis-signs", nargs=3, type=float, default=[-1.0, 1.0, 1.0])
    parser.add_argument("--graph-position-offset", nargs=3, type=float, default=[0.0, 0.0, 1.25])
    parser.add_argument("--stage", type=int, default=5, help="Stage mesh to show, or 0 for all meshes.")
    parser.add_argument("--poses", type=int, default=0, help="Limit number of poses; 0 means all.")
    parser.add_argument("--frustum-depth", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fx", type=float, default=420.0)
    parser.add_argument("--fy", type=float, default=420.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    return parser.parse_args()


def parse_stage(path: Path) -> int:
    return int(path.stem.split("_")[1])


def load_meshes(mesh_dir: Path | None, explicit: list[str], stage: int) -> list[o3d.geometry.TriangleMesh]:
    meshes = []
    sources = []
    if mesh_dir is not None:
        sources.extend((parse_stage(path), path) for path in sorted(mesh_dir.glob("stage_*_mesh.ply")))
    for item in explicit:
        if "=" not in item:
            raise SystemExit(f"--mesh expects STAGE=PATH, got {item}")
        stage_text, path_text = item.split("=", 1)
        sources.append((int(stage_text), Path(path_text)))
    for mesh_stage, path in sources:
        if stage != 0 and mesh_stage != stage:
            continue
        mesh = o3d.io.read_triangle_mesh(str(path))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(STAGE_COLORS.get(mesh_stage, [0.7, 0.7, 0.7]))
        meshes.append(mesh)
    if not meshes:
        raise SystemExit("No meshes selected; use --mesh-dir or --mesh STAGE=PATH")
    return meshes


def load_poses(run_dir: Path, limit: int) -> list[np.ndarray]:
    paths = sorted(run_dir.glob("*_pose.txt"))
    if limit > 0:
        paths = paths[:limit]
    poses = []
    for path in paths:
        pose = np.loadtxt(path)
        if pose.shape == (4, 4):
            poses.append(pose)
    if not poses:
        raise SystemExit(f"No poses found in {run_dir}")
    return poses


def load_manifest_poses(path: Path, limit: int) -> tuple[list[np.ndarray], list[bool]]:
    entries = json.loads(path.read_text())
    if limit > 0:
        entries = entries[:limit]
    poses = []
    kept = []
    for entry in entries:
        yaw = float(entry["yaw_rad"])
        position = np.asarray(entry["position"], dtype=np.float64)
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = down
        pose[:3, 2] = forward
        pose[:3, 3] = position
        poses.append(pose)
        kept.append(bool(entry["kept"]))
    if not poses:
        raise SystemExit(f"No candidate poses found in {path}")
    return poses, kept


def make_frustum(
    pose: np.ndarray,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth: float,
    color: list[float],
) -> o3d.geometry.LineSet:
    corners_px = np.array([
        [0.0, 0.0],
        [width - 1.0, 0.0],
        [width - 1.0, height - 1.0],
        [0.0, height - 1.0],
    ])
    corners_cam = []
    for u, v in corners_px:
        corners_cam.append([(u - cx) / fx * depth, (v - cy) / fy * depth, depth])
    points_cam = np.vstack([np.zeros((1, 3)), np.asarray(corners_cam)])
    points_world = (pose[:3, :3] @ points_cam.T).T + pose[:3, 3]
    lines = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color(color)
    return ls


def make_trajectory(poses: list[np.ndarray]) -> tuple[o3d.geometry.PointCloud, o3d.geometry.LineSet]:
    pts = np.asarray([pose[:3, 3] for pose in poses])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([1.0, 0.0, 0.0])

    lines = np.asarray([[i, i + 1] for i in range(len(pts) - 1)], dtype=np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    if len(lines):
        ls.lines = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color([1.0, 0.0, 0.0])
    return pcd, ls


def main() -> None:
    args = parse_args()
    meshes = load_meshes(args.mesh_dir, args.mesh, args.stage)
    if args.camera_manifest:
        poses, kept = load_manifest_poses(args.camera_manifest, args.poses)
    elif args.graph:
        poses, _ = load_nss_node_poses(
            args.graph,
            args.views_per_node,
            args.graph_scene,
            {args.node_stage} if args.node_stage is not None else None,
            np.asarray(args.graph_axis_signs, dtype=np.float64),
            np.asarray(args.graph_position_offset, dtype=np.float64),
        )
        if args.poses > 0:
            poses = poses[: args.poses]
        kept = [True] * len(poses)
    else:
        if args.run_dir is None:
            raise SystemExit("--run-dir is required unless --camera-manifest is used")
        poses = load_poses(args.run_dir, args.poses)
        kept = [True] * len(poses)
    print(f"LOADED poses={len(poses)}")
    for i, pose in enumerate(poses):
        print(f"POSE {i:03d} xyz={pose[:3, 3]}")

    candidate_points = o3d.geometry.PointCloud()
    candidate_points.points = o3d.utility.Vector3dVector(np.asarray([pose[:3, 3] for pose in poses]))
    candidate_points.colors = o3d.utility.Vector3dVector(
        np.asarray([[0.1, 0.9, 0.1] if is_kept else [0.9, 0.1, 0.1] for is_kept in kept])
    )
    frustums = [
        make_frustum(
            pose,
            args.width,
            args.height,
            args.fx,
            args.fy,
            args.cx,
            args.cy,
            args.frustum_depth,
            [0.1, 0.9, 0.1] if is_kept else [0.9, 0.1, 0.1],
        )
        for pose, is_kept in zip(poses, kept)
    ]
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    o3d.visualization.draw_geometries(meshes + [candidate_points, frame] + frustums)


if __name__ == "__main__":
    main()
