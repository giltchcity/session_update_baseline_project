#!/usr/bin/env python3
"""Estimate planar stage alignment and save transforms for OBJ visualization.

The registration uses the raw stage point clouds only as a geometric solver. The
saved matrices are intended to be applied to the original triangle meshes. They
are derived alignment candidates, not official NSS ground-truth transforms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def load_stage(root: Path, building: int, stage: int, voxel_size: float):
    path = root / f"Bldg{building}_Stage{stage}" / "pointcloud" / f"Bldg{building}_Stage{stage}.ply"
    cloud = o3d.io.read_point_cloud(str(path))
    if not cloud.has_points():
        raise RuntimeError(f"Empty or unreadable point cloud: {path}")
    cloud = cloud.voxel_down_sample(voxel_size)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3.0, max_nn=50)
    )
    print(f"LOADED Stage{stage}: {len(cloud.points)} points at {voxel_size:.3f}m")
    return cloud, path


def centered_yaw_transform(source, target, yaw_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target.get_center() - rotation @ source.get_center()
    return transform


def project_to_planar(transform: np.ndarray) -> tuple[np.ndarray, float]:
    yaw = float(np.arctan2(transform[1, 0], transform[0, 0]))
    c, s = np.cos(yaw), np.sin(yaw)
    planar = np.eye(4)
    planar[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    planar[:3, 3] = transform[:3, 3]
    return planar, float(np.rad2deg(yaw))


def align(source, target, args):
    best = None
    for yaw_deg in np.arange(args.yaw_min, args.yaw_max + args.yaw_step * 0.5, args.yaw_step):
        transform = centered_yaw_transform(source, target, float(yaw_deg))
        result = o3d.pipelines.registration.evaluate_registration(
            source, target, args.coarse_distance, transform
        )
        score = (result.fitness, -result.inlier_rmse)
        if best is None or score > best[0]:
            best = (score, float(yaw_deg), transform)

    transform = best[2]
    levels = []
    for distance in args.icp_distances:
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            distance,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=args.icp_iterations),
        )
        transform = result.transformation
        levels.append(
            {
                "max_correspondence_m": distance,
                "fitness": result.fitness,
                "inlier_rmse_m": result.inlier_rmse,
            }
        )

    planar, yaw_deg = project_to_planar(transform)
    strict = o3d.pipelines.registration.evaluate_registration(
        source, target, args.icp_distances[-1], planar
    )
    return planar, {
        "coarse_yaw_deg": best[1],
        "final_yaw_deg": yaw_deg,
        "translation_m": planar[:3, 3].tolist(),
        "levels": levels,
        "strict_fitness": strict.fitness,
        "strict_inlier_rmse_m": strict.inlier_rmse,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--building", type=int, default=1)
    parser.add_argument("--reference-stage", type=int, default=1)
    parser.add_argument("--stage", type=int, action="append", required=True)
    parser.add_argument("--voxel-size", type=float, default=0.5)
    parser.add_argument("--yaw-min", type=float, default=-20.0)
    parser.add_argument("--yaw-max", type=float, default=20.0)
    parser.add_argument("--yaw-step", type=float, default=2.0)
    parser.add_argument("--coarse-distance", type=float, default=2.0)
    parser.add_argument(
        "--icp-distances", type=float, nargs="+", default=[2.0, 1.0, 0.5, 0.25]
    )
    parser.add_argument("--icp-iterations", type=int, default=60)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference, reference_path = load_stage(
        args.raw_root, args.building, args.reference_stage, args.voxel_size
    )
    identity_path = args.output_dir / f"stage{args.reference_stage}_to_stage{args.reference_stage}.txt"
    np.savetxt(identity_path, np.eye(4), fmt="%.12f")

    summary = {
        "status": "AUTO_ICP_CANDIDATE_NOT_OFFICIAL_GT",
        "building": args.building,
        "reference_stage": args.reference_stage,
        "reference_pointcloud": str(reference_path),
        "voxel_size_m": args.voxel_size,
        "stages": {},
    }
    for stage in args.stage:
        source, source_path = load_stage(args.raw_root, args.building, stage, args.voxel_size)
        transform, diagnostics = align(source, reference, args)
        output_path = args.output_dir / f"stage{stage}_to_stage{args.reference_stage}.txt"
        np.savetxt(output_path, transform, fmt="%.12f")
        diagnostics["source_pointcloud"] = str(source_path)
        diagnostics["matrix_file"] = str(output_path)
        summary["stages"][str(stage)] = diagnostics
        print(
            f"ALIGNED Stage{stage}: yaw={diagnostics['final_yaw_deg']:.3f}deg "
            f"translation={diagnostics['translation_m']} "
            f"strict_fitness={diagnostics['strict_fitness']:.4f} "
            f"strict_rmse={diagnostics['strict_inlier_rmse_m']:.4f}m"
        )
        del source

    stage2_path = args.output_dir / "stage2_to_stage1.txt"
    stage3_path = args.output_dir / "stage3_to_stage1.txt"
    if args.reference_stage == 1 and stage2_path.exists() and stage3_path.exists():
        stage2_to_stage1 = np.loadtxt(stage2_path).reshape(4, 4)
        stage3_to_stage1 = np.loadtxt(stage3_path).reshape(4, 4)
        stage3_to_stage2 = np.linalg.inv(stage2_to_stage1) @ stage3_to_stage1
        stage3_to_stage2_path = args.output_dir / "stage3_to_stage2.txt"
        np.savetxt(stage3_to_stage2_path, stage3_to_stage2, fmt="%.12f")
        summary["derived_pair_transform"] = {
            "pair": "23",
            "matrix_file": str(stage3_to_stage2_path),
            "formula": "inverse(stage2_to_stage1) @ stage3_to_stage1",
        }

    summary_path = args.output_dir / "alignment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"SAVED {summary_path}")


if __name__ == "__main__":
    main()
