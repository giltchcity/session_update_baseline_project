#!/usr/bin/env python3
"""Create a 16:9 top-down page for two Khronos session trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--context-mesh", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    trajectories = json.loads(args.trajectories.read_text())["sessions"]
    mesh = o3d.io.read_triangle_mesh(str(args.context_mesh))
    if mesh.is_empty():
        raise RuntimeError(f"Empty context mesh: {args.context_mesh}")
    vertices = np.asarray(mesh.vertices)
    if len(vertices) > 120_000:
        rng = np.random.default_rng(0)
        vertices = vertices[rng.choice(len(vertices), 120_000, replace=False)]

    a = np.asarray(trajectories["A"]["positions"])
    b = np.asarray(trajectories["B"]["positions"])
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        s=0.28,
        c="#c8ced4",
        alpha=0.32,
        linewidths=0,
        rasterized=True,
        label="Final map footprint",
    )
    ax.plot(a[:, 0], a[:, 1], color="#05b8e8", linewidth=3.0, label="Session A")
    ax.plot(b[:, 0], b[:, 1], color="#ff7518", linewidth=3.0, label="Session B")
    ax.scatter(a[0, 0], a[0, 1], s=90, c="#05b8e8", marker="o", edgecolors="white")
    ax.scatter(a[-1, 0], a[-1, 1], s=100, c="#05b8e8", marker="X", edgecolors="white")
    ax.scatter(b[0, 0], b[0, 1], s=90, c="#ff7518", marker="o", edgecolors="white")
    ax.scatter(b[-1, 0], b[-1, 1], s=100, c="#ff7518", marker="X", edgecolors="white")
    ax.annotate("A start", a[0, :2], xytext=(8, 8), textcoords="offset points")
    ax.annotate("A end", a[-1, :2], xytext=(8, 8), textcoords="offset points")
    ax.annotate("B start", b[0, :2], xytext=(8, -16), textcoords="offset points")
    ax.annotate("B end", b[-1, :2], xytext=(8, -16), textcoords="offset points")
    ax.set_title("Office Session A / Session B Robot Trajectories", fontsize=20)
    ax.set_xlabel("World X [m]")
    ax.set_ylabel("World Y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d8dde2", linewidth=0.7)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout(pad=1.2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"SAVED {args.out}")


if __name__ == "__main__":
    main()
