#!/usr/bin/env python3
"""Expand sparse map checkpoints onto a dense visualization timeline.

Only timing fields are interpolated. Each dense row keeps the most recent real
mesh checkpoint, so this never invents intermediate reconstruction geometry.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-s", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.step_s <= 0.0:
        raise SystemExit("--step-s must be positive")

    with args.input.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or not rows:
        raise SystemExit(f"empty manifest: {args.input}")

    dense_rows: list[dict[str, str]] = []
    sessions = list(dict.fromkeys(row["session"] for row in rows))
    for session in sessions:
        session_rows = [row for row in rows if row["session"] == session]
        for index, row in enumerate(session_rows):
            if index + 1 == len(session_rows):
                dense_rows.append(dict(row))
                continue

            next_row = session_rows[index + 1]
            start_t = float(row["session_time_s"])
            end_t = float(next_row["session_time_s"])
            start_source = float(row["source_time_s"])
            end_source = float(next_row["source_time_s"])
            duration = end_t - start_t
            if duration <= 0.0:
                raise SystemExit(
                    f"non-increasing time in session {session}: {start_t} -> {end_t}"
                )

            time_s = start_t
            # Bag/checkpoint times carry nanosecond-scale rounding. Keep a
            # microsecond margin so a value such as 17.220001999999998 does
            # not create a duplicate immediately before 17.220002.
            while time_s < end_t - 1.0e-6:
                alpha = (time_s - start_t) / duration
                dense = dict(row)
                dense["session_time_s"] = f"{time_s:.9f}"
                dense["source_time_s"] = (
                    f"{start_source + alpha * (end_source - start_source):.9f}"
                )
                dense_rows.append(dense)
                time_s += args.step_s

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dense_rows)

    counts = {
        session: sum(row["session"] == session for row in dense_rows)
        for session in sessions
    }
    print(f"DENSE_MANIFEST_READY {args.output}")
    print(f"rows={len(dense_rows)} sessions={counts} step_s={args.step_s}")


if __name__ == "__main__":
    main()
