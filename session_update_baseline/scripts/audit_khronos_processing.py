#!/usr/bin/env python3
"""Record how many published frames actually entered Khronos ActiveWindow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_processed_stamps(path: Path) -> list[int]:
    with path.open(newline="") as stream:
        rows = csv.DictReader(stream)
        return [int(row["timestamp(ns)"]) for row in rows]


def stamp_digest(stamps: list[int]) -> str:
    payload = "\n".join(str(stamp) for stamp in stamps).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--playback-manifest", type=Path, required=True)
    parser.add_argument("--active-window-timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    playback = json.loads(args.playback_manifest.read_text())
    stamps = read_processed_stamps(args.active_window_timing)
    published = int(playback["frames_published"])
    processed = len(stamps)
    result = {
        "frames_available": int(playback["frames_available"]),
        "frames_encountered": int(playback["frames_encountered"]),
        "frames_published": published,
        "frames_processed_by_active_window": processed,
        "processed_fraction_of_published": processed / published if published else 0.0,
        "first_processed_timestamp_ns": stamps[0] if stamps else None,
        "last_processed_timestamp_ns": stamps[-1] if stamps else None,
        "processed_timestamp_sha256": stamp_digest(stamps),
        "evidence": {
            "timer": "active_window/all",
            "source": str(args.active_window_timing),
            "meaning": (
                "One row is emitted at ActiveWindow::spinOnce entry before motion detection, "
                "object detection, tracking, and volumetric integration."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
