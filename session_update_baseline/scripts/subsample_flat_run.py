#!/usr/bin/env python3
"""Create a hard-linked sparse view of a flat RGB-D run."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be positive")

    with (args.input_run / "timestamps.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    selected = list(range(0, len(rows), args.stride))
    if selected[-1] != len(rows) - 1:
        selected.append(len(rows) - 1)

    args.output_run.mkdir(parents=True, exist_ok=True)
    for index in selected:
        image_id = rows[index]["ImageID"]
        for suffix in ("_color.png", "_depth.tiff", "_pose.txt"):
            link_or_copy(args.input_run / f"{image_id}{suffix}", args.output_run / f"{image_id}{suffix}")

    with (args.output_run / "timestamps.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows[index] for index in selected)
    print(f"SUBSAMPLED_FLAT_RUN input={len(rows)} output={len(selected)} stride={args.stride}")


if __name__ == "__main__":
    main()
