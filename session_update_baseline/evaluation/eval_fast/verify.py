#!/usr/bin/env python3
"""Compare an eval_fast full run to archived native exports and numeric summaries."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def numeric_leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from numeric_leaves(child, f"{prefix}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from numeric_leaves(child, f"{prefix}/{index}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value


def same_number(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--archive", required=True, type=Path,
                    help="Old real_abc directory")
    ap.add_argument("--geometry-ply", type=Path,
                    help="Optional old viewer mesh for byte-identical geometry proof")
    args = ap.parse_args()
    run = args.run.resolve(strict=True)
    archive = args.archive.resolve(strict=True)
    provenance = json.loads((run / "PROVENANCE.json").read_text())
    if provenance["mode"] != "full":
        ap.error("Only complete timelines may be compared to full archived results")
    tag = f"{provenance['name']}_{provenance['session']}"
    ra = run / "real_abc"
    paths = [
        (ra / f"objects/{tag}/results/static_objects.csv",
         archive / f"objects/{tag}/results/static_objects.csv"),
        (ra / f"exports/{tag}/snapshots.jsonl",
         archive / f"exports/{tag}/snapshots.jsonl"),
    ]
    if provenance["session"] != "a":
        paths.extend([
            (ra / f"exports/{tag}/cleanup_probes.bin",
             archive / f"exports/{tag}/cleanup_probes.bin"),
            (ra / f"exports/{tag}/cleanup_mesh.jsonl",
             archive / f"exports/{tag}/cleanup_mesh.jsonl"),
        ])
    exports = {}
    for produced, old in paths:
        exports[produced.name] = {
            "present": produced.is_file() and old.is_file(),
            "exact": produced.is_file() and old.is_file() and sha(produced) == sha(old),
        }
    if args.geometry_ply:
        new_mesh = ra / f"exports/{tag}/geometry_viewer.ply"
        old_mesh = args.geometry_ply.resolve(strict=True)
        exports["geometry_viewer.ply"] = {
            "present": new_mesh.is_file(),
            "exact": new_mesh.is_file() and sha(new_mesh) == sha(old_mesh),
        }

    new_summary = json.loads((ra / f"summary/{tag}/SUMMARY.json").read_text())
    old_summary = json.loads((archive / f"summary/{tag}/SUMMARY.json").read_text())
    numeric = {}
    for section in ("object", "object_raw_native", "change", "geometry_1cm", "ghost"):
        new = dict(numeric_leaves(new_summary.get(section)))
        old = dict(numeric_leaves(old_summary.get(section)))
        wrong = [path for path in sorted(set(new) | set(old))
                 if path not in new or path not in old or not same_number(new[path], old[path])]
        numeric[section] = {"fields": len(set(new) | set(old)),
                            "exact": not wrong, "mismatches": wrong[:20]}

    success = all(item["exact"] for item in exports.values()) and \
        all(item["exact"] for item in numeric.values())
    report = {"schema": "eval-fast-parity-v1", "run": str(run),
              "archive": str(archive), "native_exports": exports,
              "numeric_summary": numeric, "passed": success}
    (run / "PARITY_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
