#!/usr/bin/env python3
"""Finalize reviewed physical-instance labels INTO the dataset (one copy, runnable).

Applies the reviewed room-wide identity rules (corrected_map from
build_reviewed_ab_physical_instances.py) to the source instance labels in
place, so the packaged dataset needs no extra script: the labels in
datasets/local_ab/instance_labels/ ARE the authoritative I1-I20 map for both
A and B.

History: the reviewed artifact (SAM2-refined maps + manifest) was produced on
2026-08-14 but never connected to the run chain and was later deleted.  The
source labels it was built from still carry legacy IDs (notably I9 black
gaming laptop, merged into I7 by the 2026-08-14 user decision).  This script
applies the same reviewed rules to the source labels and writes the reviewed
ledger manifest next to them, so the run chain consumes one self-contained
instance_labels directory.

The SAM2 local overrides (bag boundary refinements) are lost artifacts; the
identity rules (what they override for) are fully preserved here.  This is
recorded in the manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_reviewed_ab_physical_instances import corrected_map, ENTITY_CATALOG

SESSIONS = ("session_a", "session_b")
CATALOG_EXPECTED = {
    "session_a": [1, 2, 3, 4, 5, 6, 7, 10, 11, 13, 14, 16, 17, 18, 19, 20],
    "session_b": [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 16, 17, 18, 19],
}


def frame_number(path: Path) -> int:
    return int(path.stem.split("_")[0])


def process_one(args: tuple[Path, Path, str]) -> dict[str, object]:
    source_path, output_path, session = args
    frame = frame_number(source_path)
    source = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if source is None:
        raise FileNotFoundError(source_path)
    result, changes = corrected_map(session, frame, source)
    if not cv2.imwrite(
        str(output_path), result, [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ):
        raise OSError(output_path)
    return {
        "frame": frame,
        "changes": [(old, new, pixels) for (old, new), pixels in changes.items()],
    }


def process_session(
    instance_root: Path, staging: Path, session: str, workers: int
) -> dict[str, object]:
    source_dir = instance_root / session
    output_dir = staging / session
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = sorted(source_dir.glob("*_segmentation.png"))
    if not source_paths:
        raise FileNotFoundError(source_dir)

    tasks = [
        (path, output_dir / path.name, session) for path in source_paths
    ]
    total_changes: Counter = Counter()
    active_ids: set[int] = set()
    audit_path = output_dir / "frame_audit.jsonl"
    with ProcessPoolExecutor(max_workers=workers) as pool, audit_path.open(
        "w", encoding="utf-8"
    ) as audit_file:
        for record in pool.map(process_one, tasks):
            frame = int(record["frame"])
            changes = record["changes"]
            audit_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            for old, new, pixels in changes:
                total_changes[(old, new)] += pixels
    # Active IDs are those still present after the remap: re-scan outputs.
    for output_path in sorted(output_dir.glob("*.png")):
        img = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
        active_ids.update(int(v) for v in np.unique(img) if v)
    return {
        "source_instance_dir": str(source_dir.resolve()),
        "local_override_frame_count": 0,
        "canonical_override_frame_count": 0,
        "frame_count": len(source_paths),
        "frame_range": [0, len(source_paths) - 1],
        "active_instance_ids": sorted(active_ids),
        "temporal_cross_id_alerts": [],
        "relabel_pixel_totals": [
            {"source_id": old, "destination_id": new, "pixels": pixels}
            for (old, new), pixels in sorted(total_changes.items())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-root", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    sessions: dict[str, object] = {}
    for session in SESSIONS:
        print(f"processing {session} ...", flush=True)
        sessions[session] = process_session(
            args.instance_root, args.staging, session, args.workers
        )
        print(
            f"  {session}: {sessions[session]['frame_count']} frames, "
            f"active IDs {sessions[session]['active_instance_ids']}",
            flush=True,
        )

    validation = {
        "frame_counts_match_source": sessions["session_a"]["frame_count"] == 4003
        and sessions["session_b"]["frame_count"] == 4041,
        "forbidden_instance_ids_absent": all(
            instance_id not in session_data["active_instance_ids"]
            for instance_id in (8, 15)
            for session_data in sessions.values()
        ),
        "session_a_has_all_except_bed_absorbed_i12": sessions["session_a"][
            "active_instance_ids"
        ] == CATALOG_EXPECTED["session_a"],
        "session_b_has_all_except_mac_and_merged_laptop": sessions["session_b"][
            "active_instance_ids"
        ] == CATALOG_EXPECTED["session_b"],
        "computer_ids_are_7_and_20_with_mac_only_in_a": 7 in sessions["session_a"][
            "active_instance_ids"
        ]
        and 20 in sessions["session_a"]["active_instance_ids"]
        and 9 not in sessions["session_a"]["active_instance_ids"]
        and 9 not in sessions["session_b"]["active_instance_ids"]
        and 20 not in sessions["session_b"]["active_instance_ids"],
    }
    manifest = {
        "schema": "reviewed_ab_physical_instances/v1",
        "status": "finalized_into_dataset_20260815",
        "principle": "one physical room object has one ID across both A and B videos; this directory is the single authoritative instance map.",
        "entity_count": len(ENTITY_CATALOG),
        "entity_catalog": [
            {
                "instance_id": instance_id,
                "semantic_id": semantic_id,
                "physical_name": name,
                "physical_name_zh": name_zh,
            }
            for instance_id, (semantic_id, name, name_zh) in ENTITY_CATALOG.items()
        ],
        "reviewed_corrections": {
            "inactive_ids_removed": [8, 15],
            "single_chair_id": 10,
            "apparel_id": 11,
            "bag_ids": [12, 13, 14, 16, 17],
            "computer_ids": [7, 20],
            "notes": [
                "2026-08-14 user decision: I9 (black gaming laptop) merged into I7 (white desktop monitor) across A and B.  Remap: source 9 -> 7 in both sessions (A keeps source 9 -> 20 on MacBook frames 1962-1980), source 8 -> 7 on laptop-labeled A intervals, source 7 -> 7 in B.",
                "Finalized into the dataset on 2026-08-15: the reviewed rules from build_reviewed_ab_physical_instances.py were applied to the source labels in place so the packaged dataset runs directly.",
                "SAM2 local bag refinements (local_sam2_refinement_*_20260814) were deleted with the 2026-08-14 run artifacts; their identity-level rules are all preserved in the remap above.",
            ],
        },
        "sessions": sessions,
        "validation": validation,
    }
    manifest_path = args.staging / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ok = all(
        validation[key]
        for key in (
            "frame_counts_match_source",
            "forbidden_instance_ids_absent",
            "session_a_has_all_except_bed_absorbed_i12",
            "session_b_has_all_except_mac_and_merged_laptop",
            "computer_ids_are_7_and_20_with_mac_only_in_a",
        )
    )
    print(json.dumps(validation, indent=2))
    print(f"manifest: {manifest_path}")
    print("VALIDATION_OK" if ok else "VALIDATION_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
