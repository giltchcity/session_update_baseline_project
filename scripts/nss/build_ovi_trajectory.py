#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Replica-style trajectory for OVI-MAP.")
    parser.add_argument("--waypoints", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fx", type=float, default=420.0)
    parser.add_argument("--fy", type=float, default=420.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=20.0)
    parser.add_argument("--yaw-offsets-deg", nargs="+", type=float, default=[0.0])
    parser.add_argument("--yaw-sweep-amplitude-deg", type=float, default=0.0)
    parser.add_argument("--yaw-sweep-period-m", type=float, default=3.0)
    parser.add_argument("--preview-mesh", type=Path)
    parser.add_argument("--visibility-mesh", type=Path)
    parser.add_argument("--validate-width", type=int, default=160)
    parser.add_argument("--validate-height", type=int, default=120)
    parser.add_argument("--filter-min-hit-ratio", type=float)
    parser.add_argument("--filter-min-depth-median", type=float)
    parser.add_argument("--filter-max-depth-median", type=float)
    parser.add_argument("--max-preview-points", type=int, default=180_000)
    return parser.parse_args()


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
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


def yaw_pitch_pose(eye: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    forward = np.array(
        [
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        ]
    )
    return look_at(eye, eye + forward)


def load_waypoint_poses(
    path: Path,
    yaw_offsets_deg: list[float],
    yaw_sweep_amplitude_deg: float,
    yaw_sweep_period_m: float,
) -> tuple[dict, list[np.ndarray], list[dict]]:
    data = json.loads(path.read_text())
    waypoints = data.get("waypoints", [])
    if len(waypoints) < 2:
        raise SystemExit(f"{path}: need at least two waypoints")

    z = float(data.get("camera_z", 1.35))
    step = float(data.get("step_m", 0.5))
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
        path_distance = float(distances[-1])
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

    smooth_sweep = abs(float(yaw_sweep_amplitude_deg)) > 1e-9
    if smooth_sweep and yaw_sweep_period_m <= 0.0:
        raise SystemExit("--yaw-sweep-period-m must be positive")

    poses = []
    metadata = []
    samples_arr = np.asarray(samples)
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
            offsets = [(offset_deg, math.radians(offset_deg), "smooth_sine")]
        else:
            offsets = [(float(d), r, "discrete_offsets") for d, r in zip(yaw_offsets_deg, yaw_offsets)]

        for offset_deg, offset_rad, yaw_mode in offsets:
            yaw = base_yaw + offset_rad
            poses.append(yaw_pitch_pose(eye, yaw, pitch))
            metadata.append(
                {
                    "frame_id": len(poses) - 1,
                    "source": "manual_waypoints",
                    "waypoint_file": str(path),
                    "sample_index": int(i),
                    "segment_index": int(segment_ids[i]),
                    "path_distance_m": float(distances[i]),
                    "x": float(eye[0]),
                    "y": float(eye[1]),
                    "z": float(eye[2]),
                    "base_yaw_rad": float(base_yaw),
                    "yaw_mode": yaw_mode,
                    "yaw_offset_deg": float(offset_deg),
                    "yaw_rad": float(yaw),
                    "pitch_rad": float(pitch),
                }
            )
    data["expanded_path_length_m"] = float(path_distance)
    return data, poses, metadata


def write_pose(path: Path, pose: np.ndarray) -> None:
    with path.open("w") as f:
        for row in range(4):
            f.write(" ".join(f"{pose[row, col]:.9f}" for col in range(4)))
            f.write("\n")


def write_traj(path: Path, poses: list[np.ndarray]) -> None:
    with path.open("w") as f:
        for pose in poses:
            for row in range(4):
                f.write(" ".join(f"{pose[row, col]:.9f}" for col in range(4)))
                f.write("\n")


def make_rays(
    pose: np.ndarray,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> o3d.core.Tensor:
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    dirs_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    rot = pose[:3, :3].astype(np.float32)
    trans = pose[:3, 3].astype(np.float32)
    dirs_world = dirs_cam @ rot.T
    origins = np.broadcast_to(trans, dirs_world.shape)
    rays = np.concatenate([origins, dirs_world], axis=-1).astype(np.float32)
    return o3d.core.Tensor(rays)


def validate_visibility(
    mesh_path: Path,
    poses: list[np.ndarray],
    args: argparse.Namespace,
) -> list[dict]:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise SystemExit(f"Failed to load validation mesh: {mesh_path}")
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    sx = args.validate_width / args.width
    sy = args.validate_height / args.height
    fx = args.fx * sx
    fy = args.fy * sy
    cx = args.cx * sx
    cy = args.cy * sy

    stats = []
    for i, pose in enumerate(poses):
        rays = make_rays(pose, args.validate_width, args.validate_height, fx, fy, cx, cy)
        depth = scene.cast_rays(rays)["t_hit"].numpy().astype(np.float32)
        hit = np.isfinite(depth) & (depth >= args.near) & (depth <= args.far)
        values = depth[hit]
        stats.append(
            {
                "frame_id": i,
                "hit_ratio": float(np.mean(hit)),
                "depth_min": float(values.min()) if values.size else None,
                "depth_median": float(np.median(values)) if values.size else None,
                "depth_max": float(values.max()) if values.size else None,
            }
        )
    return stats


def write_metadata_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def filter_poses(
    poses: list[np.ndarray],
    metadata: list[dict],
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[dict], list[int]]:
    keep_indices = []
    for i, row in enumerate(metadata):
        keep = True
        hit_ratio = row.get("hit_ratio")
        depth_median = row.get("depth_median")
        if args.filter_min_hit_ratio is not None:
            keep = keep and hit_ratio is not None and float(hit_ratio) >= args.filter_min_hit_ratio
        if args.filter_min_depth_median is not None:
            keep = keep and depth_median is not None and float(depth_median) >= args.filter_min_depth_median
        if args.filter_max_depth_median is not None:
            keep = keep and depth_median is not None and float(depth_median) <= args.filter_max_depth_median
        if keep:
            keep_indices.append(i)

    if not keep_indices:
        raise SystemExit("Trajectory filtering removed every pose.")

    filtered_poses = [poses[i] for i in keep_indices]
    filtered_metadata = []
    for new_id, old_id in enumerate(keep_indices):
        row = dict(metadata[old_id])
        row["source_frame_id"] = row["frame_id"]
        row["frame_id"] = new_id
        filtered_metadata.append(row)
    return filtered_poses, filtered_metadata, keep_indices


def mesh_vertices_for_preview(path: Path, max_points: int) -> np.ndarray | None:
    if path is None:
        return None
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    vertices = np.asarray(mesh.vertices)
    if len(vertices) == 0:
        return None
    if len(vertices) > max_points:
        rng = np.random.default_rng(3)
        vertices = vertices[rng.choice(len(vertices), size=max_points, replace=False)]
    return vertices


def write_preview(path: Path, mesh_points: np.ndarray | None, poses: list[np.ndarray], waypoint_data: dict) -> None:
    pose_xyz = np.asarray([pose[:3, 3] for pose in poses])
    waypoints = np.asarray(
        [[float(w["x"]), float(w["y"])] for w in waypoint_data.get("waypoints", [])], dtype=np.float64
    )
    fig, ax = plt.subplots(figsize=(12, 7), dpi=170)
    if mesh_points is not None:
        ax.scatter(mesh_points[:, 0], mesh_points[:, 1], s=0.08, c="#1d1d1d", alpha=0.24, linewidths=0)
    ax.plot(pose_xyz[:, 0], pose_xyz[:, 1], color="#e63946", linewidth=1.4, label="expanded poses")
    ax.scatter(pose_xyz[:, 0], pose_xyz[:, 1], s=5, c="#e63946", alpha=0.75)
    if len(waypoints):
        ax.plot(waypoints[:, 0], waypoints[:, 1], "o-", color="#2a9d8f", linewidth=1.0, markersize=3, label="waypoints")
    arrow_step = max(1, len(poses) // 28)
    for pose in poses[::arrow_step]:
        start = pose[:3, 3]
        forward = pose[:3, 2]
        ax.arrow(
            start[0],
            start[1],
            forward[0] * 0.45,
            forward[1] * 0.45,
            color="#457b9d",
            width=0.025,
            head_width=0.18,
            length_includes_head=True,
            alpha=0.9,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#dddddd", linewidth=0.35)
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    waypoint_data, poses, metadata = load_waypoint_poses(
        args.waypoints,
        args.yaw_offsets_deg,
        args.yaw_sweep_amplitude_deg,
        args.yaw_sweep_period_m,
    )
    candidate_pose_count = len(poses)

    scene_dir = args.out_root / args.scene
    pose_dir = scene_dir / "pose"
    results_dir = scene_dir / "results"
    pose_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    visibility_stats: list[dict] = []
    if args.visibility_mesh is not None:
        visibility_stats = validate_visibility(args.visibility_mesh, poses, args)
        visibility_by_frame = {row["frame_id"]: row for row in visibility_stats}
        metadata = [{**row, **visibility_by_frame.get(row["frame_id"], {})} for row in metadata]

    keep_indices = list(range(len(poses)))
    filtering_enabled = any(
        threshold is not None
        for threshold in (
            args.filter_min_hit_ratio,
            args.filter_min_depth_median,
            args.filter_max_depth_median,
        )
    )
    if filtering_enabled:
        if not visibility_stats:
            raise SystemExit("Visibility filters require --visibility-mesh.")
        poses, metadata, keep_indices = filter_poses(poses, metadata, args)

    cam_params = {
        "camera": {
            "h": args.height,
            "w": args.width,
            "fx": args.fx,
            "fy": args.fy,
            "cx": args.cx,
            "cy": args.cy,
            "scale": args.depth_scale,
        }
    }
    (args.out_root / "cam_params.json").write_text(json.dumps(cam_params, indent=2))
    write_traj(scene_dir / "traj.txt", poses)
    for i, pose in enumerate(poses):
        write_pose(pose_dir / f"{i:06d}.txt", pose)

    write_metadata_csv(scene_dir / "pose_metadata.csv", metadata)
    mesh_points = mesh_vertices_for_preview(args.preview_mesh, args.max_preview_points) if args.preview_mesh else None
    write_preview(scene_dir / "trajectory_preview.png", mesh_points, poses, waypoint_data)

    hit_ratios = np.asarray(
        [float(row["hit_ratio"]) for row in metadata if row.get("hit_ratio") not in (None, "")],
        dtype=np.float64,
    )
    med_depths = np.asarray(
        [
            float(row["depth_median"])
            for row in metadata
            if row.get("depth_median") not in (None, "")
        ],
        dtype=np.float64,
    )
    manifest = {
        "scene": args.scene,
        "waypoints": str(args.waypoints.resolve()),
        "trajectory_format": "Replica/OVI-MAP compatible traj.txt: row-major 4x4 world_T_camera per frame",
        "camera_frame": "OpenCV camera axes: x right, y down, z forward",
        "scene_dir": str(scene_dir.resolve()),
        "cam_params": str((args.out_root / "cam_params.json").resolve()),
        "traj_txt": str((scene_dir / "traj.txt").resolve()),
        "pose_dir": str(pose_dir.resolve()),
        "results_dir": str(results_dir.resolve()),
        "pose_metadata_csv": str((scene_dir / "pose_metadata.csv").resolve()),
        "preview_png": str((scene_dir / "trajectory_preview.png").resolve()),
        "num_candidate_poses": candidate_pose_count,
        "num_poses": len(poses),
        "kept_source_frame_ids": keep_indices,
        "path_length_m": waypoint_data.get("expanded_path_length_m"),
        "camera": cam_params["camera"],
        "yaw_sweep_amplitude_deg": args.yaw_sweep_amplitude_deg,
        "yaw_sweep_period_m": args.yaw_sweep_period_m,
        "visibility_mesh": str(args.visibility_mesh.resolve()) if args.visibility_mesh else None,
        "filter": {
            "enabled": filtering_enabled,
            "min_hit_ratio": args.filter_min_hit_ratio,
            "min_depth_median": args.filter_min_depth_median,
            "max_depth_median": args.filter_max_depth_median,
        },
        "visibility": {
            "validated": bool(args.visibility_mesh),
            "resolution": [args.validate_width, args.validate_height],
            "hit_ratio_min": float(hit_ratios.min()) if hit_ratios.size else None,
            "hit_ratio_median": float(np.median(hit_ratios)) if hit_ratios.size else None,
            "hit_ratio_max": float(hit_ratios.max()) if hit_ratios.size else None,
            "depth_median_min": float(np.nanmin(med_depths)) if med_depths.size else None,
            "depth_median_median": float(np.nanmedian(med_depths)) if med_depths.size else None,
            "depth_median_max": float(np.nanmax(med_depths)) if med_depths.size else None,
        },
    }
    (scene_dir / "trajectory_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
