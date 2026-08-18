#!/usr/bin/env python3
"""Measure the frozen A->B acceptance KPIs from inspect_session_state geometry dumps.

The question this answers is not "how many vertices survived" -- vertex count alone says
nothing about whether a surface is intact -- but the three properties the contract fixes:

  KPI-1  OLD CURRENT SURVIVAL
         For a moved object, how much of the geometry it had at its OLD site is still being
         published as CURRENT? Reported as the fraction of session-A CURRENT points for that
         physical ID that still have a session-B CURRENT point of the same ID within one map
         voxel. Target for moved IDs: 0. Any static ID must stay at ~100%: that column is the
         over-deletion guard, and it is the one that must never regress.

  KPI-2  NEW CURRENT QUALITY vs PURE-B
         For the same physical ID, how much of what session B could reconstruct on its own
         does the incremental A->B map actually carry? Reported symmetrically:
           recall     what fraction of pure-B's surface the incremental map covers
           precision  what fraction of the incremental map's surface pure-B also has
         Recall below ~0.9 means inherited memory damaged what B observed for itself.
         Precision well below recall means the incremental map carries surface that B never
         saw -- inherited geometry that should not be CURRENT.

  Extent/centroid are reported alongside so a collapse in spatial extent is visible even when
  the point counts look reasonable.

Coverage is evaluated at the map's own reconstruction resolution: a point is "covered" when
some point of the other set lies within one voxel of it. That is the scale the map is built
at, not a tunable match distance.

Usage:
    python3 scripts/measure_ab_kpi.py \
        --session-a runs/v3_a/state --session-b runs/v3_b/state \
        [--pure-b runs/pure_b/state] [--resolution 0.05] [--json out.json]

Each --session-* argument is a state directory (or a .4dmap); the script runs
inspect_session_state --dump-geometry on it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Frozen ground truth for the local A/B pair, from the reviewed instance manifest.
MOVED_IDS = {2: "cabinet/luggage", 3: "table", 10: "chair", 11: "apparel",
             13: "checkered bag", 14: "blue backpack", 18: "basket/blanket", 19: "fan"}
STATIC_IDS = {1: "bed", 4: "shelf", 5: "desk", 6: "wardrobe", 7: "computer",
              16: "wardrobe-top bag", 17: "white bag"}


def find_inspector(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    found = shutil.which("inspect_session_state")
    if found:
        return found
    for candidate in (Path("build_canonical/inspect_session_state"),
                      Path(".canonical_mapping/build/khronos/inspect_session_state")):
        if candidate.is_file():
            return str(candidate.resolve())
    raise SystemExit("inspect_session_state not found; pass --inspector")


def dump_objects(inspector: str, state: Path) -> Dict[int, List[Tuple[float, float, float]]]:
    """physical_id -> world points of that ID's CURRENT private mesh (all nodes merged)."""
    # inspect_session_state takes the flag first and reads a .4dmap, not a state directory.
    target = state
    if target.is_dir():
        for candidate in (target / "final.4dmap", target / "state" / "final.4dmap"):
            if candidate.is_file():
                target = candidate
                break
        else:
            raise SystemExit(f"no final.4dmap under {state}")
    raw = subprocess.check_output([inspector, "--dump-geometry", str(target)], text=True)
    payload = json.loads(raw)
    objects = payload.get("objects", payload if isinstance(payload, list) else [])
    by_id: Dict[int, List[Tuple[float, float, float]]] = {}
    for record in objects:
        pid = int(record.get("physical_id", 0) or 0)
        if pid == 0:
            continue
        points = [(float(p[0]), float(p[1]), float(p[2]))
                  for p in record.get("mesh_world_points", [])]
        by_id.setdefault(pid, []).extend(points)
    return by_id


def voxel_set(points, resolution: float):
    return {(int(x // resolution), int(y // resolution), int(z // resolution))
            for x, y, z in points}


def covered_fraction(query, reference, resolution: float) -> float:
    """Fraction of `query` points lying within one voxel of some `reference` point.

    Occupancy is compared on the map's own voxel grid, and a query voxel counts as covered if
    the reference occupies it or any of its 26 neighbours -- so a point sitting just across a
    grid boundary from its match is not counted as missing.
    """
    if not query:
        return float("nan")
    if not reference:
        return 0.0
    reference_voxels = voxel_set(reference, resolution)
    hits = 0
    for x, y, z in query:
        vx, vy, vz = int(x // resolution), int(y // resolution), int(z // resolution)
        found = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if (vx + dx, vy + dy, vz + dz) in reference_voxels:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        hits += found
    return hits / len(query)


def extent(points) -> Tuple[float, float, float]:
    if not points:
        return (0.0, 0.0, 0.0)
    xs, ys, zs = zip(*points)
    return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))


def centroid(points) -> Tuple[float, float, float]:
    if not points:
        return (float("nan"),) * 3
    n = len(points)
    xs, ys, zs = zip(*points)
    return (sum(xs) / n, sum(ys) / n, sum(zs) / n)


def spatial_clusters(points, radius: float = 0.75):
    """Greedy spatial clustering of world points into site clusters.

    A reference mesh of a moved object in a ghost-free baseline should be one
    cluster. When the reference itself unions several positions (pure-B's
    reconciler can merge distinct sites of one physical ID), the clusters let
    the caller compare against the site that actually matches the query.
    Returns a list of (centroid, [points]).
    """
    clusters = []
    for point in points:
        placed = False
        for cluster in clusters:
            cx, cy, cz = cluster[0]
            if (point[0] - cx) ** 2 + (point[1] - cy) ** 2 + (point[2] - cz) ** 2 <= radius * radius:
                cluster[1].append(point)
                n = len(cluster[1])
                cluster[0] = tuple(
                    (c * (n - 1) + p) / n
                    for c, p in zip(cluster[0], point))
                placed = True
                break
        if not placed:
            clusters.append([point, [point]])
    return [(c[0], c[1]) for c in clusters]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session-a", required=True, type=Path)
    parser.add_argument("--session-b", required=True, type=Path)
    parser.add_argument("--pure-b", type=Path, default=None)
    parser.add_argument("--resolution", type=float, default=0.05,
                        help="map reconstruction resolution in metres (default 5 cm)")
    parser.add_argument("--inspector", default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    inspector = find_inspector(args.inspector)
    a_objects = dump_objects(inspector, args.session_a)
    b_objects = dump_objects(inspector, args.session_b)
    pure_objects = dump_objects(inspector, args.pure_b) if args.pure_b else {}

    rows = []
    for pid, name in sorted({**MOVED_IDS, **STATIC_IDS}.items()):
        is_moved = pid in MOVED_IDS
        a_points = a_objects.get(pid, [])
        b_points = b_objects.get(pid, [])
        pure_points = pure_objects.get(pid, [])

        # KPI-1: how much of A's CURRENT geometry is still published as CURRENT in B.
        residue = covered_fraction(a_points, b_points, args.resolution)
        row = {
            "physical_id": pid,
            "name": name,
            "class": "moved" if is_moved else "static",
            "a_points": len(a_points),
            "b_points": len(b_points),
            "old_geometry_still_current": residue,
            "b_extent": extent(b_points),
            "b_centroid": centroid(b_points),
        }
        if pure_points:
            # KPI-2: symmetric surface coverage against what B alone could rebuild.
            row["pure_b_points"] = len(pure_points)
            row["recall_of_pure_b"] = covered_fraction(pure_points, b_points, args.resolution)
            row["precision_vs_pure_b"] = covered_fraction(b_points, pure_points, args.resolution)
            row["pure_b_extent"] = extent(pure_points)
            # Site-aware recall for moved objects: pure-B can itself union
            # several positions of a moved ID (its reconciler merges segments
            # of one physical ID), which inflates the baseline and deflates
            # naive recall. Compare B's CURRENT only against the pure-B site
            # cluster nearest to B's current centroid.
            if is_moved and b_points and len(pure_points) > len(b_points):
                clusters = spatial_clusters(pure_points)
                b_center = centroid(b_points)
                nearest = min(clusters, key=lambda c: sum(
                    (a - b) ** 2 for a, b in zip(c[0], b_center)))
                row["site_recall"] = covered_fraction(nearest[1], b_points,
                                                      args.resolution)
                row["pure_b_sites"] = len(clusters)
            else:
                row["site_recall"] = row["recall_of_pure_b"]
                row["pure_b_sites"] = 1
        rows.append(row)

    header = f"{'ID':>4} {'name':<18} {'class':<7} {'A pts':>7} {'B pts':>7} {'old→CURRENT':>12}"
    if pure_objects:
        header += f" {'pure-B':>7} {'recall':>7} {'siteR':>7} {'prec':>7}"
    print(header)
    print("-" * len(header))
    for row in rows:
        line = (f"{row['physical_id']:>4} {row['name']:<18} {row['class']:<7} "
                f"{row['a_points']:>7} {row['b_points']:>7} "
                f"{row['old_geometry_still_current']:>11.1%}")
        if "recall_of_pure_b" in row:
            line += (f" {row['pure_b_points']:>7} {row['recall_of_pure_b']:>6.1%} "
                     f" {row['site_recall']:>6.1%} "
                     f"{row['precision_vs_pure_b']:>6.1%}")
        print(line)

    moved = [r for r in rows if r["class"] == "moved" and r["a_points"]]
    static = [r for r in rows if r["class"] == "static" and r["a_points"]]
    print()
    if moved:
        worst = max(moved, key=lambda r: r["old_geometry_still_current"])
        print(f"moved   old-geometry-still-CURRENT: mean "
              f"{sum(r['old_geometry_still_current'] for r in moved) / len(moved):.1%}, "
              f"worst {worst['name']} {worst['old_geometry_still_current']:.1%}   (target 0%)")
    if static:
        weakest = min(static, key=lambda r: r["old_geometry_still_current"])
        print(f"static  retained:                   mean "
              f"{sum(r['old_geometry_still_current'] for r in static) / len(static):.1%}, "
              f"weakest {weakest['name']} {weakest['old_geometry_still_current']:.1%}  "
              f"(target 100%)")
    if pure_objects:
        scored = [r for r in rows if "recall_of_pure_b" in r and r["pure_b_points"]]
        if scored:
            below = [r for r in scored if r["recall_of_pure_b"] < 0.9]
            print(f"recall of pure-B:                   mean "
                  f"{sum(r['recall_of_pure_b'] for r in scored) / len(scored):.1%}; "
                  f"below 90%: {', '.join(r['name'] for r in below) or 'none'}")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
