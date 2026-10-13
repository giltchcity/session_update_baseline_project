#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def load_points(path: Path, transform: np.ndarray | None, max_points: int) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise RuntimeError(f"empty point cloud: {path}")
    if transform is not None:
        cloud.transform(transform)
    points = np.asarray(cloud.points, dtype=np.float64)
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if len(points) > max_points:
        rng = np.random.default_rng(0)
        points = points[rng.choice(len(points), max_points, replace=False)]
    return points


def load_trajectory(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    with (run_dir / "timestamps.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    positions = []
    forwards = []
    for row in rows:
        pose = np.loadtxt(run_dir / f"{row['ImageID']}_pose.txt").reshape(4, 4)
        positions.append(pose[:3, 3])
        forwards.append(pose[:3, 2])
    return np.asarray(positions), np.asarray(forwards)


def density_image(points: np.ndarray, bounds: tuple[float, float, float, float], bins: int) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    histogram, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=bins,
        range=((xmin, xmax), (ymin, ymax)),
    )
    density = np.log1p(histogram)
    maximum = float(density.max())
    if maximum > 0.0:
        density /= maximum
    return density.T


def draw_trajectory(
    axis,
    positions: np.ndarray,
    forwards: np.ndarray,
    color: str,
    label: str,
    linestyle: str,
) -> None:
    axis.plot(
        positions[:, 0],
        positions[:, 1],
        color=color,
        linewidth=2.0,
        linestyle=linestyle,
        label=label,
        zorder=4,
    )
    step = max(1, len(positions) // 28)
    samples = slice(None, None, step)
    axis.quiver(
        positions[samples, 0],
        positions[samples, 1],
        forwards[samples, 0],
        forwards[samples, 1],
        color=color,
        angles="xy",
        scale_units="xy",
        scale=0.65,
        width=0.004,
        zorder=5,
    )
    axis.scatter(
        positions[0, 0],
        positions[0, 1],
        s=70,
        c=color,
        marker="o",
        edgecolors="white",
        linewidths=1.2,
        zorder=6,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-pointcloud", type=Path, required=True)
    parser.add_argument("--stage2-pointcloud", type=Path, required=True)
    parser.add_argument("--stage2-transform", type=Path, required=True)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=350_000)
    parser.add_argument("--bins", type=int, default=700)
    args = parser.parse_args()

    transform = np.loadtxt(args.stage2_transform, dtype=np.float64).reshape(4, 4)
    points_a = load_points(args.stage1_pointcloud, None, args.max_points)
    points_b = load_points(args.stage2_pointcloud, transform, args.max_points)
    trajectory_a, forward_a = load_trajectory(args.run_a)
    trajectory_b, forward_b = load_trajectory(args.run_b)

    all_xy = np.vstack((points_a[:, :2], points_b[:, :2], trajectory_a[:, :2], trajectory_b[:, :2]))
    xmin, ymin = np.percentile(all_xy, 0.2, axis=0)
    xmax, ymax = np.percentile(all_xy, 99.8, axis=0)
    padding = 0.03 * max(xmax - xmin, ymax - ymin)
    bounds = (xmin - padding, xmax + padding, ymin - padding, ymax + padding)
    extent = (bounds[0], bounds[1], bounds[2], bounds[3])

    density_a = density_image(points_a, bounds, args.bins)
    density_b = density_image(points_b, bounds, args.bins)

    figure, axes = plt.subplots(1, 3, figsize=(18, 6.5), constrained_layout=True)
    figure.canvas.manager.set_window_title("Bldg1 Stage1 / Stage2 floor plans and trajectories")

    axes[0].imshow(1.0 - density_a, extent=extent, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
    draw_trajectory(axes[0], trajectory_a, forward_a, "#d7191c", "Session A trajectory", "-")
    axes[0].set_title("Stage1 floor plan + Session A")

    axes[1].imshow(1.0 - density_b, extent=extent, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
    draw_trajectory(axes[1], trajectory_b, forward_b, "#00a6ca", "Session B trajectory", "--")
    axes[1].set_title("Aligned Stage2 floor plan + Session B")

    axes[2].scatter(points_a[:, 0], points_a[:, 1], s=0.15, c="#d7191c", alpha=0.18, rasterized=True)
    axes[2].scatter(points_b[:, 0], points_b[:, 1], s=0.15, c="#00a6ca", alpha=0.18, rasterized=True)
    draw_trajectory(axes[2], trajectory_a, forward_a, "#d7191c", "A: solid", "-")
    draw_trajectory(axes[2], trajectory_b, forward_b, "#00a6ca", "B: dashed", "--")
    axes[2].set_title("Overlay: Stage1 red / Stage2 cyan")

    for axis in axes:
        axis.set_xlim(bounds[0], bounds[1])
        axis.set_ylim(bounds[2], bounds[3])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("world x [m]")
        axis.set_ylabel("world y [m]")
        axis.legend(loc="upper right")
        axis.grid(alpha=0.15)

    plt.show()


if __name__ == "__main__":
    main()
