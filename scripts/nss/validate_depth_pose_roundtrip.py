#!/usr/bin/env python3
"""Check whether virtual depth + pose backprojects onto the source stage mesh.

This is a diagnostic adapter validator. It does not evaluate NSS registration or
Panoptic mapping quality. It only checks whether the flat dataset pose/depth
convention is self-consistent under a ROS optical camera interpretation:

    x = (u - cx) * z / fx, y = (v - cy) * z / fy, z = z
    world_point = world_T_camera * camera_point

If this fails, Panoptic can still ingest the dataset but will integrate surfaces
in the wrong world locations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path, help="Dataset root containing render_manifest.json and run dir.")
    parser.add_argument("--run-name", default="run_virtual")
    parser.add_argument("--mesh-dir", type=Path, help="Directory containing stage_<id>_mesh.ply. Defaults to render_manifest source_meshes.")
    parser.add_argument("--frames", type=int, default=12, help="Maximum frames to sample.")
    parser.add_argument("--points-per-frame", type=int, default=5000)
    parser.add_argument("--fx", type=float, help="Override fx. Defaults to render_manifest camera.fx.")
    parser.add_argument("--fy", type=float, help="Override fy. Defaults to render_manifest camera.fy.")
    parser.add_argument("--cx", type=float, help="Override cx. Defaults to render_manifest camera.cx.")
    parser.add_argument("--cy", type=float, help="Override cy. Defaults to render_manifest camera.cy.")
    parser.add_argument("--conventions", nargs="+", default=["opencv"], choices=["opencv", "opengl", "ros_base"])
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def read_timestamps(run_dir: Path) -> list[str]:
    with (run_dir / "timestamps.csv").open() as f:
        reader = csv.DictReader(f)
        return [row["ImageID"] for row in reader]


def parse_stage_mesh(path: Path) -> int:
    parts = path.stem.split("_")
    if len(parts) < 2 or parts[0] != "stage":
        raise ValueError(f"cannot parse stage id from {path}")
    return int(parts[1])


def read_mesh_pointclouds(
    mesh_dir: Path | None,
    manifest: dict,
    sample_count: int = 200_000,
) -> tuple[dict[int, o3d.geometry.PointCloud], dict[int, str]]:
    clouds: dict[int, o3d.geometry.PointCloud] = {}
    sources: dict[int, Path] = {}
    if mesh_dir is not None:
        for path in sorted(mesh_dir.glob("stage_*_mesh.ply")):
            sources[parse_stage_mesh(path)] = path
    else:
        sources = {int(stage): Path(path) for stage, path in manifest.get("source_meshes", {}).items()}
    for stage, path in sorted(sources.items()):
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.triangles) == 0:
            raise ValueError(f"{path}: empty mesh")
        # Uniform sampling is enough for nearest-neighbor convention checks.
        n = min(sample_count, max(20_000, len(mesh.vertices)))
        clouds[stage] = mesh.sample_points_uniformly(number_of_points=n)
    if not clouds:
        raise SystemExit("No source meshes found; pass --mesh-dir or use a render manifest with source_meshes")
    return clouds, {stage: str(path) for stage, path in sources.items()}


def frame_stage_map(root: Path) -> dict[str, int]:
    manifest = json.loads((root / "render_manifest.json").read_text())
    return {frame["image_id"]: int(frame["stage"]) for frame in manifest["frames"]}


def choose_frame_ids(ids: list[str], max_frames: int) -> list[str]:
    if max_frames <= 0 or max_frames >= len(ids):
        return ids
    indices = np.linspace(0, len(ids) - 1, max_frames, dtype=int)
    return [ids[i] for i in indices]


def backproject(depth: np.ndarray, pose: np.ndarray, fx: float, fy: float, cx: float, cy: float, convention: str, max_points: int) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    vs, us = np.nonzero(valid)
    if len(us) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if len(us) > max_points:
        rng = np.random.default_rng(13)
        keep = rng.choice(len(us), size=max_points, replace=False)
        us = us[keep]
        vs = vs[keep]
    z = depth[vs, us].astype(np.float64)
    x = (us.astype(np.float64) - cx) * z / fx
    y = (vs.astype(np.float64) - cy) * z / fy

    if convention == "opencv":
        pts_cam = np.stack([x, y, z], axis=1)
    elif convention == "opengl":
        # Common rendering convention: x right, y up, camera looks along -Z.
        pts_cam = np.stack([x, -y, -z], axis=1)
    elif convention == "ros_base":
        # Robot base-like convention: x forward, y left, z up, from optical xyz.
        pts_cam = np.stack([z, -x, -y], axis=1)
    else:
        raise ValueError(convention)

    pts_h = np.concatenate([pts_cam, np.ones((len(pts_cam), 1), dtype=np.float64)], axis=1)
    return (pose @ pts_h.T).T[:, :3]


def nn_stats(points: np.ndarray, cloud: o3d.geometry.PointCloud) -> dict:
    if len(points) == 0:
        return {"count": 0, "median": None, "mean": None, "p90": None, "within_5cm": None, "within_20cm": None}
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    distances = np.asarray(pcd.compute_point_cloud_distance(cloud))
    return {
        "count": int(len(points)),
        "median": float(np.median(distances)),
        "mean": float(np.mean(distances)),
        "p90": float(np.percentile(distances, 90)),
        "within_5cm": float(np.mean(distances < 0.05)),
        "within_20cm": float(np.mean(distances < 0.20)),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.root / args.run_name
    manifest = json.loads((args.root / "render_manifest.json").read_text())
    camera = manifest["camera"]
    fx = args.fx if args.fx is not None else float(camera["fx"])
    fy = args.fy if args.fy is not None else float(camera["fy"])
    cx = args.cx if args.cx is not None else float(camera["cx"])
    cy = args.cy if args.cy is not None else float(camera["cy"])

    ids = choose_frame_ids(read_timestamps(run_dir), args.frames)
    stages_by_frame = frame_stage_map(args.root)
    clouds, mesh_sources = read_mesh_pointclouds(args.mesh_dir, manifest)

    report = {
        "root": str(args.root),
        "run_dir": str(run_dir),
        "mesh_dir": str(args.mesh_dir) if args.mesh_dir else None,
        "mesh_sources": mesh_sources,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "manifest_intrinsics": {
            "fx": float(camera["fx"]),
            "fy": float(camera["fy"]),
            "cx": float(camera["cx"]),
            "cy": float(camera["cy"]),
        },
        "sampled_frames": ids,
        "conventions": {},
    }

    print(f"ROOT {args.root}")
    print(f"RUN_DIR {run_dir}")
    print(f"INTRINSICS fx={fx} fy={fy} cx={cx} cy={cy}")

    for convention in args.conventions:
        frame_results = []
        all_medians = []
        all_p90 = []
        print(f"\nCONVENTION {convention}")
        for image_id in ids:
            stage = stages_by_frame[image_id]
            depth = np.asarray(Image.open(run_dir / f"{image_id}_depth.tiff"), dtype=np.float32)
            pose = np.loadtxt(run_dir / f"{image_id}_pose.txt")
            points = backproject(depth, pose, fx, fy, cx, cy, convention, args.points_per_frame)
            stats = nn_stats(points, clouds[stage])
            stats.update({"image_id": image_id, "stage": stage})
            frame_results.append(stats)
            if stats["median"] is not None:
                all_medians.append(stats["median"])
                all_p90.append(stats["p90"])
            print(
                f"  frame={image_id} stage={stage} n={stats['count']} "
                f"median={stats['median']} p90={stats['p90']} "
                f"<5cm={stats['within_5cm']} <20cm={stats['within_20cm']}"
            )

        summary = {
            "median_of_medians": float(np.median(all_medians)) if all_medians else None,
            "median_p90": float(np.median(all_p90)) if all_p90 else None,
            "frames": frame_results,
        }
        report["conventions"][convention] = summary
        print(f"SUMMARY {convention}: median_of_medians={summary['median_of_medians']} median_p90={summary['median_p90']}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"\nWROTE {args.report}")


if __name__ == "__main__":
    main()
