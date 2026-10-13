#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
from pathlib import Path


FRAME_SUFFIXES = (
    "_color.png",
    "_depth.tiff",
    "_segmentation.png",
    "_pose.txt",
)


def read_timestamps(run_dir: Path) -> list[tuple[str, int]]:
    with (run_dir / "timestamps.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [(row["ImageID"], int(row["TimeStamp"])) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Join virtual-flat sessions without duplicating frame data.")
    parser.add_argument("--session", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sessions = [path.resolve() for path in args.session]
    if len(sessions) < 2:
        raise SystemExit("at least two --session arguments are required")
    if args.out.exists():
        if not args.force:
            raise SystemExit(f"output already exists: {args.out}")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    all_rows: list[tuple[str, int]] = []
    boundaries = []
    next_timestamp = None
    output_index = 0

    for session_index, run_dir in enumerate(sessions):
        rows = read_timestamps(run_dir)
        if not rows:
            raise RuntimeError(f"empty timestamps: {run_dir}")
        source_times = [stamp for _, stamp in rows]
        positive_deltas = [
            current - previous
            for previous, current in zip(source_times, source_times[1:])
            if current > previous
        ]
        step_ns = int(statistics.median(positive_deltas)) if positive_deltas else 500_000_000
        output_start = output_index
        source_start = int(source_times[0])
        if next_timestamp is None:
            next_timestamp = source_start

        for source_id, source_stamp in rows:
            output_id = f"{output_index:06d}"
            output_stamp = next_timestamp + (int(source_stamp) - source_start)
            for suffix in FRAME_SUFFIXES:
                source_path = run_dir / f"{source_id}{suffix}"
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                os.symlink(source_path, args.out / f"{output_id}{suffix}")
            all_rows.append((output_id, output_stamp))
            output_index += 1

        last_output_stamp = all_rows[-1][1]
        next_timestamp = last_output_stamp + step_ns
        boundaries.append(
            {
                "session_index": session_index,
                "source": str(run_dir),
                "output_frame_start": output_start,
                "output_frame_end": output_index - 1,
                "frame_count": len(rows),
                "output_timestamp_start": all_rows[output_start][1],
                "output_timestamp_end": last_output_stamp,
            }
        )

    for metadata_name in ("Intrinsics.txt", "intrinsics.yaml", "groundtruth_labels.csv"):
        for metadata_root in (sessions[0], sessions[0].parent):
            source_path = metadata_root / metadata_name
            if source_path.exists():
                os.symlink(source_path, args.out / metadata_name)
                break

    with (args.out / "timestamps.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("ImageID", "TimeStamp"))
        writer.writerows(all_rows)

    manifest = {
        "frame_count": len(all_rows),
        "storage": "symlinks_to_source_frames",
        "sessions": boundaries,
    }
    (args.out / "combined_session_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"COMBINED_SESSION_READY out={args.out} frames={len(all_rows)}")
    for boundary in boundaries:
        print(
            "SESSION_BOUNDARY "
            f"index={boundary['session_index']} "
            f"frames={boundary['output_frame_start']}..{boundary['output_frame_end']} "
            f"source={boundary['source']}"
        )


if __name__ == "__main__":
    main()
