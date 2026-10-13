#!/usr/bin/env python3
"""Build a dense out-and-back trajectory from a contiguous source-path interval."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--end-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def trajectory_yaws(points: np.ndarray) -> np.ndarray:
    directions = np.empty_like(points)
    directions[0] = points[1] - points[0]
    directions[-1] = points[-1] - points[-2]
    directions[1:-1] = points[2:] - points[:-2]
    return np.unwrap(np.arctan2(directions[:, 1], directions[:, 0]))


def main() -> None:
    args = parse_args()
    source = json.loads(args.source.read_text())
    rows = source["waypoints"]
    if not 0 <= args.start_index < args.end_index < len(rows):
        raise SystemExit(
            f"invalid interval [{args.start_index}, {args.end_index}] for {len(rows)} poses"
        )

    forward = rows[args.start_index : args.end_index + 1]
    selected = forward + forward[-2::-1]
    points = np.asarray([[float(row["x"]), float(row["y"])] for row in selected])
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distances = np.r_[0.0, np.cumsum(segment_lengths)]
    yaws = trajectory_yaws(points)

    output = {
        "name": f"{args.source.stem}_turnback_{args.start_index}_{args.end_index}",
        "frame": source.get("frame", "world"),
        "camera_z": float(source.get("camera_z", 1.25)),
        "pitch_deg": float(source.get("pitch_deg", -5.0)),
        "step_m": float(source.get("step_m", np.median(segment_lengths))),
        "closed": True,
        "preinterpolated": True,
        "source_waypoints": str(args.source),
        "selection": {
            "strategy": "common_visible_contiguous_out_and_back",
            "source_start_index": args.start_index,
            "source_end_index": args.end_index,
            "forward_poses": len(forward),
            "total_poses": len(selected),
            "path_length_m": float(distances[-1]),
        },
        "waypoints": [
            {
                "x": float(point[0]),
                "y": float(point[1]),
                "distance_m": float(distance),
                "yaw_rad": float(math.atan2(math.sin(yaw), math.cos(yaw))),
            }
            for point, distance, yaw in zip(points, distances, yaws)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        f"TURNBACK_TRAJECTORY output={args.output} poses={len(selected)} "
        f"length_m={distances[-1]:.3f} return_error_m={np.linalg.norm(points[-1] - points[0]):.6f}"
    )


if __name__ == "__main__":
    main()
