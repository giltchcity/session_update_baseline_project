#!/usr/bin/env python3
"""Prepare the Base1 A->B viewer directly from the saved 4D maps."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


def read_playback(path: Path) -> dict:
    return json.loads(path.read_text())


def read_flat(run_dir: Path, playback: dict, transform: np.ndarray) -> dict:
    timestamps = []
    paths = []
    positions = []
    base_ns = int(playback["output_bounds_ns"][0])
    with (run_dir / "timestamps.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            image_id = row["ImageID"]
            source_ns = int(row["TimeStamp"])
            pose = np.loadtxt(run_dir / f"{image_id}_pose.txt").reshape(4, 4)
            pose = transform @ pose
            timestamps.append(base_ns + source_ns)
            paths.append(str((run_dir / f"{image_id}_color.png").resolve()))
            positions.append(pose[:3, 3].tolist())
    return {
        "timestamps_ns": timestamps,
        "paths": paths,
        "positions": positions,
    }


def export_sequence(
    exporter: Path, map_file: Path, output: Path, start_ns: int, end_ns: int
) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(exporter),
            "export-mesh",
            "--map_file",
            str(map_file),
            "--output_sequence_dir",
            str(output),
            "--include_faces",
            "true",
            "--include_object_meshes",
            "true",
            "--sequence_period_s",
            "2.0",
            "--sequence_start_ns",
            str(start_ns),
            "--sequence_end_ns",
            str(end_ns),
        ],
        check=True,
    )
    with (output / "map_sequence.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def export_dynamic_history(exporter: Path, dsg_file: Path, output: Path) -> dict:
    subprocess.run(
        [
            str(exporter),
            "export-mesh",
            "--dsg_file",
            str(dsg_file),
            "--output_view_json",
            str(output),
        ],
        check=True,
    )
    return json.loads(output.read_text())


def export_final(
    exporter: Path, map_file: Path, ply: Path, overlay: Path
) -> None:
    subprocess.run(
        [
            str(exporter),
            "export-mesh",
            "--map_file",
            str(map_file),
            "--output_ply",
            str(ply),
            "--output_view_json",
            str(overlay),
            "--map_time",
            "latest",
            "--include_faces",
            "true",
            "--include_object_meshes",
            "true",
        ],
        check=True,
    )


def evidence_fields(path: Path | None) -> dict:
    keys = [
        "initial_vertices",
        "removed_vertices",
        "injected_vertices",
        "final_vertices",
        "prior_matched_objects",
        "prior_restored_objects",
        "prior_absent_objects",
        "prior_unobserved_objects",
        "cross_session_prior_vertices",
        "cross_session_current_vertices",
        "cross_session_prior_absent_vertices",
        "cross_session_prior_persistent_vertices",
        "cross_session_prior_unobserved_vertices",
        "cross_session_current_injected_vertices",
    ]
    data = json.loads(path.read_text()) if path and path.exists() else {}
    return {key: data.get(key, 0) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-map", type=Path, required=True)
    parser.add_argument("--b-map", type=Path, required=True)
    parser.add_argument("--a-dsg", type=Path, required=True)
    parser.add_argument("--b-dsg", type=Path, required=True)
    parser.add_argument("--improved-map", type=Path)
    parser.add_argument("--evidence-summary", type=Path)
    parser.add_argument("--a-flat", type=Path, required=True)
    parser.add_argument("--b-flat", type=Path, required=True)
    parser.add_argument("--a-playback", type=Path, required=True)
    parser.add_argument("--b-playback", type=Path, required=True)
    parser.add_argument("--b-transform", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = Path("/home/jixian/Desktop/FT/session_update_baseline")
    output = args.output_dir.resolve()
    frames = output / "frames"
    if output.exists():
        shutil.rmtree(output)
    frames.mkdir(parents=True)

    exporter = root / "scripts" / "run_base1_khronos_env.sh"
    a_playback = read_playback(args.a_playback)
    b_playback = read_playback(args.b_playback)
    a_start, a_end = [int(value) for value in a_playback["output_bounds_ns"]]
    b_start, b_end = [int(value) for value in b_playback["output_bounds_ns"]]
    a_rows = export_sequence(exporter, args.a_map.resolve(), frames / "A", a_start, a_end)
    b_rows = export_sequence(exporter, args.b_map.resolve(), frames / "B", b_start, b_end)
    a_history_path = frames / "A" / "dynamic_history.json"
    b_history_path = frames / "B" / "dynamic_history.json"
    a_history = export_dynamic_history(exporter, args.a_dsg.resolve(), a_history_path)
    b_history = export_dynamic_history(exporter, args.b_dsg.resolve(), b_history_path)
    common = evidence_fields(None)
    rows = []

    def append_rows(session: str, exported: list[dict], start: int, end: int) -> None:
        history_path = a_history_path if session == "A" else b_history_path
        history = a_history if session == "A" else b_history
        for exported_row in exported:
            stamp = int(exported_row["timestamp_ns"])
            if stamp < start or stamp > end:
                continue
            index = len(rows)
            session_index = sum(row["session"] == session for row in rows)
            base = frames / session
            rows.append(
                {
                    "frame": index,
                    "session": session,
                    "phase": "within_session" if session == "A" else "cross_session_with_A_memory",
                    "session_checkpoint": session_index,
                    "session_time_s": stamp / 1.0e9,
                    "source_time_s": (stamp - start) / 1.0e9,
                    "ply": str((base / exported_row["ply"]).relative_to(output)),
                    "overlay": str((base / exported_row["overlay"]).relative_to(output)),
                    "dynamic_history": str(history_path.relative_to(output)),
                    "dynamic_tracks": sum(
                        bool(track.get("timestamps_ns"))
                        and int(track["timestamps_ns"][0]) <= stamp
                        and stamp <= int(track["timestamps_ns"][-1]) + 300_000_000
                        for track in history.get("dynamic_tracks", [])
                    ),
                    "display_object_meshes": True,
                    **common,
                }
            )

    append_rows("A", a_rows, a_start, a_end)
    append_rows("B", b_rows, b_start, b_end)

    if args.improved_map:
        final_ply = frames / "B" / "improved_final.ply"
        final_overlay = frames / "B" / "improved_final.json"
        export_final(exporter, args.improved_map.resolve(), final_ply, final_overlay)
        final_evidence = evidence_fields(args.evidence_summary)
        rows.append(
            {
                "frame": len(rows),
                "session": "B",
                "phase": "cross_session_reconciled_final",
                "session_checkpoint": sum(row["session"] == "B" for row in rows),
                "session_time_s": b_end / 1.0e9,
                "source_time_s": (b_end - b_start) / 1.0e9,
                "ply": str(final_ply.relative_to(output)),
                # Keep the actual B dynamic history at the final query time.
                "overlay": rows[-1]["overlay"],
                "dynamic_history": str(b_history_path.relative_to(output)),
                "dynamic_tracks": rows[-1]["dynamic_tracks"],
                "display_object_meshes": True,
                **final_evidence,
            }
        )

    if not rows or not any(row["session"] == "A" for row in rows) or not any(
        row["session"] == "B" for row in rows
    ):
        raise RuntimeError("Saved 4D maps did not yield both A and B sequence states")

    with (output / "sequence_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "sequence_manifest.json").write_text(json.dumps(rows, indent=2))

    transform = np.loadtxt(args.b_transform).reshape(4, 4)
    a_sensor = read_flat(args.a_flat, a_playback, np.eye(4))
    b_sensor = read_flat(args.b_flat, b_playback, transform)
    sensor_views = {
        "schema": "base1_sensor_views_v1",
        "sessions": {"A": a_sensor, "B": b_sensor},
    }
    (output / "sensor_views.json").write_text(json.dumps(sensor_views))
    trajectories = {
        "schema": "base1_session_trajectories_v1",
        "sessions": {
            "A": {
                "source": "Session A input pose files",
                "certainty": "DIRECT_INPUT_TRAJECTORY",
                "pose_count": len(a_sensor["positions"]),
                "timestamps_ns": a_sensor["timestamps_ns"],
                "positions": a_sensor["positions"],
            },
            "B": {
                "source": "Session B input pose files transformed by B_to_A",
                "certainty": "DIRECT_REGISTERED_INPUT_TRAJECTORY",
                "pose_count": len(b_sensor["positions"]),
                "timestamps_ns": b_sensor["timestamps_ns"],
                "positions": b_sensor["positions"],
            },
        },
    }
    (output / "session_trajectories.json").write_text(json.dumps(trajectories))
    shutil.copy2(root / "scripts" / "view_office_ab_process.py", output / "view_process.py")
    (output / "README.txt").write_text(
        "Run:\n"
        f"/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python {output / 'view_process.py'} "
        f"--manifest {output / 'sequence_manifest.csv'} --frame-seconds 0.8\n"
    )
    print(
        f"PROCESS_SEQUENCE_READY {output} frames={len(rows)} "
        f"A={sum(row['session'] == 'A' for row in rows)} "
        f"B={sum(row['session'] == 'B' for row in rows)}"
    )


if __name__ == "__main__":
    main()
