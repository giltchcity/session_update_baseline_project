#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    parser.add_argument("--camera-z", type=float, default=1.25)
    parser.add_argument("--pitch-deg", type=float, default=-5.0)
    parser.add_argument("--step-m", type=float, default=0.35)
    parser.add_argument("--max-points", type=int, default=150000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pcd = o3d.io.read_point_cloud(str(args.ply))
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise SystemExit("empty point cloud")

    mask = np.ones(len(pts), dtype=bool)
    if args.z_min is not None:
        mask &= pts[:, 2] >= args.z_min
    if args.z_max is not None:
        mask &= pts[:, 2] <= args.z_max
    pts = pts[mask]
    if len(pts) == 0:
        raise SystemExit("z filter removed every point")

    if len(pts) > args.max_points:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), args.max_points, replace=False)]

    waypoints: list[tuple[float, float]] = []

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(pts[:, 0], pts[:, 1], s=0.2, c="black", alpha=0.35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Left click: add waypoint | u: undo | s: save | q: quit")
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    (line,) = ax.plot([], [], "r-o", linewidth=1.5, markersize=4)

    def redraw() -> None:
        if waypoints:
            arr = np.asarray(waypoints)
            line.set_data(arr[:, 0], arr[:, 1])
        else:
            line.set_data([], [])
        fig.canvas.draw_idle()

    def save() -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": args.out.stem,
            "frame": "world",
            "camera_z": args.camera_z,
            "pitch_deg": args.pitch_deg,
            "step_m": args.step_m,
            "waypoints": [{"x": float(x), "y": float(y)} for x, y in waypoints],
        }
        args.out.write_text(json.dumps(data, indent=2))
        print(f"SAVED {args.out} waypoints={len(waypoints)}")

    def onclick(event) -> None:
        if event.inaxes != ax or event.button != 1:
            return
        waypoints.append((event.xdata, event.ydata))
        print(f"ADD {len(waypoints) - 1:02d}: x={event.xdata:.3f}, y={event.ydata:.3f}")
        redraw()

    def onkey(event) -> None:
        if event.key == "u":
            if waypoints:
                removed = waypoints.pop()
                print(f"UNDO x={removed[0]:.3f}, y={removed[1]:.3f}")
                redraw()
        elif event.key == "s":
            save()
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)
    plt.show()

    # Closing the window is also a completed edit. Always persist the current
    # path, even when --out already exists from an earlier drawing.
    if waypoints:
        save()


if __name__ == "__main__":
    main()
