#!/usr/bin/env python3
"""Smooth a hand-drawn XY path and resample it at uniform arc length."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--step-m", type=float, default=0.15)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--closed", action="store_true")
    parser.add_argument("--yaw-smoothing-m", type=float, default=0.75)
    return parser.parse_args()


def chaikin(points: np.ndarray, iterations: int, closed: bool) -> np.ndarray:
    current = np.asarray(points, dtype=np.float64)
    for _ in range(max(0, iterations)):
        if closed:
            following = np.roll(current, -1, axis=0)
            q = 0.75 * current + 0.25 * following
            r = 0.25 * current + 0.75 * following
            current = np.column_stack((q, r)).reshape(-1, 2)
        else:
            result = [current[0]]
            for start, end in zip(current, current[1:]):
                result.extend((0.75 * start + 0.25 * end, 0.25 * start + 0.75 * end))
            result.append(current[-1])
            current = np.asarray(result)
    return current


def resample(points: np.ndarray, step_m: float, closed: bool) -> tuple[np.ndarray, np.ndarray, float]:
    if step_m <= 0.0:
        raise ValueError("--step-m must be positive")
    path = np.vstack((points, points[0])) if closed else points
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 1.0e-8))
    path = path[keep]
    if len(path) < 2:
        raise ValueError("path collapsed after removing duplicate points")
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
    total = float(cumulative[-1])
    count = max(2, int(math.ceil(total / step_m)))
    distances = np.linspace(0.0, total, count, endpoint=not closed)
    x = np.interp(distances, cumulative, path[:, 0])
    y = np.interp(distances, cumulative, path[:, 1])
    return np.column_stack((x, y)), distances, total


def smooth_yaws(points: np.ndarray, closed: bool, sigma_samples: float) -> np.ndarray:
    if closed:
        tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    else:
        tangent = np.empty_like(points)
        tangent[1:-1] = points[2:] - points[:-2]
        tangent[0] = points[1] - points[0]
        tangent[-1] = points[-1] - points[-2]
    yaw = np.arctan2(tangent[:, 1], tangent[:, 0])
    if sigma_samples <= 0.0:
        return yaw
    radius = max(1, int(math.ceil(3.0 * sigma_samples)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma_samples) ** 2)
    kernel /= kernel.sum()
    unit = np.column_stack((np.cos(yaw), np.sin(yaw)))
    if closed:
        filtered = sum(weight * np.roll(unit, int(offset), axis=0) for offset, weight in zip(offsets, kernel))
    else:
        padded = np.pad(unit, ((radius, radius), (0, 0)), mode="edge")
        filtered = np.column_stack(
            [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)]
        )
    return np.arctan2(filtered[:, 1], filtered[:, 0])


def yaw_stats(yaw: np.ndarray) -> dict[str, float]:
    yaw = np.unwrap(yaw)
    delta = np.abs(np.diff(yaw))
    return {
        "yaw_step_median_deg": float(np.degrees(np.median(delta))) if len(delta) else 0.0,
        "yaw_step_p95_deg": float(np.degrees(np.percentile(delta, 95))) if len(delta) else 0.0,
        "yaw_step_max_deg": float(np.degrees(np.max(delta))) if len(delta) else 0.0,
    }


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text())
    raw = np.asarray([[float(row["x"]), float(row["y"])] for row in data["waypoints"]])
    if len(raw) < 3:
        raise SystemExit("need at least three waypoints")

    smoothed_control = chaikin(raw, args.iterations, args.closed)
    samples, distances, length = resample(smoothed_control, args.step_m, args.closed)
    yaw = smooth_yaws(samples, args.closed, args.yaw_smoothing_m / args.step_m)
    result = {
        "name": f"{data.get('name', args.input.stem)}_smooth",
        "frame": data.get("frame", "world"),
        "camera_z": float(data.get("camera_z", 1.25)),
        "pitch_deg": float(data.get("pitch_deg", -5.0)),
        "step_m": float(args.step_m),
        "closed": bool(args.closed),
        "preinterpolated": True,
        "source_waypoints": str(args.input),
        "smoothing": {
            "method": "chaikin_then_uniform_arc_length",
            "iterations": int(args.iterations),
            "raw_waypoints": int(len(raw)),
            "smoothed_control_points": int(len(smoothed_control)),
            "output_samples": int(len(samples)),
            "path_length_m": length,
            "yaw_smoothing_m": float(args.yaw_smoothing_m),
            **yaw_stats(yaw),
        },
        "waypoints": [
            {
                "x": float(point[0]),
                "y": float(point[1]),
                "distance_m": float(distance),
                "yaw_rad": float(frame_yaw),
            }
            for point, distance, frame_yaw in zip(samples, distances, yaw)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **result["smoothing"]}, indent=2))


if __name__ == "__main__":
    main()
