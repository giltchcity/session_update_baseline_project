#!/usr/bin/env python3
"""Repair deterministic flat-dataset timestamps without preserving hard links."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--start-ns", type=int, default=1_000_000_000)
    parser.add_argument("--step-ns", type=int, default=500_000_000)
    args = parser.parse_args()
    for run_dir in args.runs:
        image_ids = sorted(path.name[:-9] for path in run_dir.glob("*_pose.txt"))
        if not image_ids:
            raise RuntimeError(f"No pose files in {run_dir}")
        target = run_dir / "timestamps.csv"
        temporary = run_dir / ".timestamps.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("ImageID", "TimeStamp"))
            for index, image_id in enumerate(image_ids):
                writer.writerow((image_id, args.start_ns + index * args.step_ns))
        os.replace(temporary, target)
        print(f"TIMESTAMPS_REPAIRED run={run_dir} frames={len(image_ids)}")


if __name__ == "__main__":
    main()
