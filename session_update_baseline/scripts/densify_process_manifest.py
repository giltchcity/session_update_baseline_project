#!/usr/bin/env python3
"""Densify an A/B visualization timeline without inventing map snapshots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-s", type=float, default=0.1)
    return parser.parse_args()


def interpolate_row(left: dict[str, str], right: dict[str, str], value: float) -> dict[str, str]:
    row = dict(left)
    left_time = float(left["session_time_s"])
    right_time = float(right["session_time_s"])
    ratio = 0.0 if right_time <= left_time else (value - left_time) / (right_time - left_time)
    row["session_time_s"] = f"{value:.9f}"
    source_time = float(left["source_time_s"]) + ratio * (
        float(right["source_time_s"]) - float(left["source_time_s"])
    )
    row["source_time_s"] = f"{source_time:.9f}"
    return row


def main() -> None:
    args = parse_args()
    if args.step_s <= 0:
        raise SystemExit("--step-s must be positive")

    with args.input.open(newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    if not source_rows:
        raise SystemExit("input manifest is empty")

    dense_rows: list[dict[str, str]] = []
    sessions: list[list[dict[str, str]]] = []
    for row in source_rows:
        if not sessions or sessions[-1][0]["session"] != row["session"]:
            sessions.append([])
        sessions[-1].append(row)

    for rows in sessions:
        for index, left in enumerate(rows[:-1]):
            right = rows[index + 1]
            current = float(left["session_time_s"])
            stop = float(right["session_time_s"])
            while current < stop - 1.0e-9:
                dense_rows.append(interpolate_row(left, right, current))
                current += args.step_s
        dense_rows.append(dict(rows[-1]))

    for index, row in enumerate(dense_rows):
        row["frame"] = str(index)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dense_rows[0]))
        writer.writeheader()
        writer.writerows(dense_rows)
    args.output.with_suffix(".json").write_text(json.dumps(dense_rows, indent=2))
    print(
        f"DENSE_MANIFEST_READY {args.output} "
        f"source_frames={len(source_rows)} dense_frames={len(dense_rows)} step_s={args.step_s}"
    )


if __name__ == "__main__":
    main()
