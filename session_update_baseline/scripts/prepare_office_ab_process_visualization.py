#!/usr/bin/env python3
"""Materialize the stored A/B Khronos checkpoints as a Base1 mesh sequence."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path


STAMP_RE = re.compile(r"_([0-9]+)\.json$")


def run(command: list[str], cwd: Path) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def checkpoint_stamps(run_dir: Path) -> list[int]:
    stamps = []
    for path in (run_dir / "maps").glob("dsg_*.json"):
        match = STAMP_RE.search(path.name)
        if match:
            stamps.append(int(match.group(1)))
    return sorted(stamps)


def source_time_s(adapter_manifest: dict, session_time_ns: int) -> float:
    bag_start_ns = int(adapter_manifest["source_bag_bounds_ns"][0])
    output_start_ns = int(adapter_manifest["output_start_ns"])
    session_delta_ns = session_time_ns - output_start_ns
    if adapter_manifest["reverse"]:
        source_ns = int(adapter_manifest["selected_source_bounds_ns"][1]) - session_delta_ns
    else:
        source_ns = int(adapter_manifest["selected_source_bounds_ns"][0]) + session_delta_ns
    return (source_ns - bag_start_ns) / 1.0e9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/home/jixian/Desktop/FT"))
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--a-adapter-manifest", type=Path, required=True)
    parser.add_argument("--b-adapter-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--a-run-dir", type=Path)
    parser.add_argument("--a-map", type=Path)
    parser.add_argument("--a-changes", type=Path)
    parser.add_argument("--b-run-dir", type=Path)
    parser.add_argument("--b-map", type=Path)
    parser.add_argument("--b-changes", type=Path)
    parser.add_argument("--prior-map", type=Path)
    parser.add_argument("--prior-memory", type=Path)
    parser.add_argument(
        "--global-mesh-only",
        action="store_true",
        help="Do not include persistent private object meshes in display PLY files.",
    )
    parser.add_argument("--keep-intermediate-maps", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    round_dir = args.round_dir.resolve()
    output_dir = args.output_dir.resolve()
    frames_dir = output_dir / "frames"
    diagnostics_dir = output_dir / "diagnostics"
    frames_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    a_run_dir = (args.a_run_dir or round_dir / "A_khronos_v3").resolve()
    a_map = (args.a_map or a_run_dir / "final.4dmap").resolve()
    a_changes = (args.a_changes or a_run_dir / "object_changes.csv").resolve()
    b_run_dir = (args.b_run_dir or round_dir / "B_khronos_from_scratch").resolve()
    b_map = (args.b_map or b_run_dir / "final.4dmap").resolve()
    b_changes = (args.b_changes or b_run_dir / "object_changes.csv").resolve()
    prior_map = (args.prior_map or round_dir / "A_ours" / "improved_final.4dmap").resolve()
    prior_memory = (
        args.prior_memory or round_dir / "A_ours" / "object_memory.json"
    ).resolve()

    sessions = [
        {
            "name": "A",
            "phase": "within_session",
            "run_dir": a_run_dir,
            "map": a_map,
            "changes": a_changes,
            "adapter": json.loads(args.a_adapter_manifest.read_text()),
            "prior_map": None,
            "prior_memory": None,
        },
        {
            "name": "B",
            "phase": "cross_session_with_A_memory",
            "run_dir": b_run_dir,
            "map": b_map,
            "changes": b_changes,
            "adapter": json.loads(args.b_adapter_manifest.read_text()),
            "prior_map": prior_map,
            "prior_memory": prior_memory,
        },
    ]

    runner = workspace / "session_update_baseline" / "scripts" / "run_session_update_baseline.sh"
    exporter = workspace / "session_update_baseline" / "scripts" / "run_base1_khronos_env.sh"
    manifest_rows: list[dict] = []
    frame_number = 0

    for session in sessions:
        stamps = checkpoint_stamps(session["run_dir"])
        if not stamps:
            raise RuntimeError(f"No checkpoints found in {session['run_dir'] / 'maps'}")

        dynamic_history_path = diagnostics_dir / f"{session['name']}_dynamic_history.json"
        run(
            [
                str(exporter),
                "export-mesh",
                "--map_file",
                str(session["map"]),
                "--output_view_json",
                str(dynamic_history_path),
                "--map_time",
                "latest",
            ],
            workspace,
        )
        dynamic_history = json.loads(dynamic_history_path.read_text())

        for index, stamp in enumerate(stamps):
            diagnostic_dir = diagnostics_dir / f"{session['name']}_{index:02d}"
            command = [
                str(runner),
                "--map_file",
                str(session["map"]),
                "--output_dir",
                str(diagnostic_dir),
                "--mode",
                "full",
                "--dynamic_mode",
                "within_session" if session["name"] == "A" else "cross_session",
                "--map_time",
                f"index:{index}",
                "--object_changes_csv",
                str(session["changes"]),
                "--object_move_decision",
                "hard",
                "--object_injection_policy",
                "repair_candidates",
                "--object_surface_support_distance_m",
                "0.1",
                "--object_alignment_policy",
                "none",
                "--object_distance_m",
                "0.05",
                "--bbox_margin_m",
                "0.05",
                "--injection_min_separation_m",
                "0.08",
                "--min_object_mesh_vertices",
                "20",
                "--repair_global_vertex_ratio_threshold",
                "0.05",
                "--require_same_label",
                "false",
                "--require_bbox_containment",
                "true",
            ]
            if session["name"] == "B":
                command += [
                    "--object_cleanup_reobservation_gate",
                    "true",
                    "--object_cleanup_support_distance_m",
                    "0.08",
                ]
            if session["prior_map"]:
                command += ["--prior_map", str(session["prior_map"])]
            if session["prior_memory"]:
                command += ["--prior_object_memory", str(session["prior_memory"])]
            run(command, workspace)

            frame_path = frames_dir / f"frame_{frame_number:03d}_{session['name']}_{index:02d}.ply"
            overlay_path = (
                frames_dir / f"frame_{frame_number:03d}_{session['name']}_{index:02d}.json"
            )
            export_command = [
                str(exporter),
                "export-mesh",
                "--map_file",
                str(diagnostic_dir / "improved_final.4dmap"),
                "--output_ply",
                str(frame_path),
                "--map_time",
                "latest",
                "--include_faces",
                "true",
            ]
            if not args.global_mesh_only:
                export_command += ["--include_object_meshes", "true"]
            run(export_command, workspace)
            # Dynamic tracks must come from this session's observation map. The
            # reconciled output may contain restored prior object history, which
            # is memory rather than motion observed in the current session.
            run(
                [
                    str(exporter),
                    "export-mesh",
                    "--map_file",
                    str(diagnostic_dir / "original_final.4dmap"),
                    "--output_view_json",
                    str(overlay_path),
                    "--map_time",
                    "latest",
                ],
                workspace,
            )

            evidence = json.loads((diagnostic_dir / "evidence_summary.json").read_text())
            overlay = json.loads(overlay_path.read_text())
            active_dynamic_tracks = sum(
                bool(track.get("timestamps_ns"))
                and int(track["timestamps_ns"][0]) <= stamp
                and stamp <= int(track["timestamps_ns"][-1]) + 300_000_000
                for track in dynamic_history.get("dynamic_tracks", [])
            )
            manifest_rows.append(
                {
                    "frame": frame_number,
                    "session": session["name"],
                    "phase": session["phase"],
                    "session_checkpoint": index,
                    "session_time_s": stamp / 1.0e9,
                    "source_time_s": source_time_s(session["adapter"], stamp),
                    "ply": str(frame_path.relative_to(output_dir)),
                    "overlay": str(overlay_path.relative_to(output_dir)),
                    "dynamic_history": str(dynamic_history_path.relative_to(output_dir)),
                    "dynamic_tracks": active_dynamic_tracks,
                    "display_object_meshes": not args.global_mesh_only,
                    "initial_vertices": evidence["initial_vertices"],
                    "removed_vertices": evidence["removed_vertices"],
                    "injected_vertices": evidence["injected_vertices"],
                    "final_vertices": evidence["final_vertices"],
                    "prior_matched_objects": evidence["prior_matched_objects"],
                    "prior_restored_objects": evidence["prior_restored_objects"],
                    "prior_absent_objects": evidence.get("prior_absent_objects", 0),
                    "prior_unobserved_objects": evidence.get("prior_unobserved_objects", 0),
                    "cross_session_prior_vertices": evidence.get(
                        "cross_session_prior_vertices", 0
                    ),
                    "cross_session_current_vertices": evidence.get(
                        "cross_session_current_vertices", 0
                    ),
                    "cross_session_prior_absent_vertices": evidence.get(
                        "cross_session_prior_absent_vertices", 0
                    ),
                    "cross_session_prior_persistent_vertices": evidence.get(
                        "cross_session_prior_persistent_vertices", 0
                    ),
                    "cross_session_prior_unobserved_vertices": evidence.get(
                        "cross_session_prior_unobserved_vertices", 0
                    ),
                    "cross_session_current_injected_vertices": evidence.get(
                        "cross_session_current_injected_vertices", 0
                    ),
                }
            )
            frame_number += 1

            if not args.keep_intermediate_maps:
                (diagnostic_dir / "original_final.4dmap").unlink(missing_ok=True)
                (diagnostic_dir / "improved_final.4dmap").unlink(missing_ok=True)

    fieldnames = list(manifest_rows[0])
    with (output_dir / "sequence_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output_dir / "sequence_manifest.json").write_text(json.dumps(manifest_rows, indent=2))

    viewer_source = workspace / "session_update_baseline" / "scripts" / "view_office_ab_process.py"
    shutil.copy2(viewer_source, output_dir / "view_process.py")
    print(f"PROCESS_SEQUENCE_READY {output_dir}")
    print(f"frames={len(manifest_rows)}")


if __name__ == "__main__":
    main()
