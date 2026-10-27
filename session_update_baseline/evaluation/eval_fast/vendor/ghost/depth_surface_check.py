"""Classify actual predicted surfaces against time-restricted RGBD evidence.

This is an offline diagnostic, not a published cleanup rate. Layer and prediction
identity are intentionally not inputs to the geometric decision. Hidden points
are unknown, not free. The caller must restrict frames to a stable scene state.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


COLORS = {
    "current_supported": (40, 190, 90),
    "contradicted_free_space": (235, 45, 40),
    "conflicting_evidence": (235, 160, 25),
    "unknown": (145, 145, 155),
}


def project_evidence(points, depth, camera_to_world, intrinsics, tolerance,
                     semantics=None, instances=None, edge_range=0.05):
    """Return per-point votes: 0 unknown, 1 support, 2 free, 3 occluded.

    Each point is independently projected into the measured image; predicted
    foreground cannot hide another predicted point from this evaluation.
    A complete, smooth 3x3 depth neighborhood is required to reject silhouettes.
    """
    n = len(points)
    votes = np.zeros(n, dtype=np.uint8)
    observed_ids = np.full(n, -1, dtype=np.int16)
    q = (points - camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
    valid_z = np.isfinite(q).all(1) & (q[:, 2] > 0.1) & (q[:, 2] <= 5.0)
    candidates = np.flatnonzero(valid_z)
    if not len(candidates):
        return votes, observed_ids
    z = q[candidates, 2]
    u = np.rint(intrinsics["f_x"] * q[candidates, 0] / z + intrinsics["u"]).astype(int)
    v = np.rint(intrinsics["f_y"] * q[candidates, 1] / z + intrinsics["v"]).astype(int)
    inside = (u >= 1) & (v >= 1) & (u < depth.shape[1] - 1) & (v < depth.shape[0] - 1)
    candidates, u, v, z = candidates[inside], u[inside], v[inside], z[inside]
    if not len(candidates):
        return votes, observed_ids
    patch = np.stack([depth[v + dy, u + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    good = np.isfinite(patch).all(0) & (patch > 0.1).all(0) & (patch <= 5.0).all(0)
    good &= np.ptp(patch, axis=0) <= edge_range
    if semantics is not None:
        good &= np.stack([semantics[v + dy, u + dx] != 12
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1)]).all(0)
    lo, hi = np.min(patch, axis=0), np.max(patch, axis=0)
    support = good & (np.max(np.abs(patch - z), axis=0) <= tolerance)
    free = good & (lo > z + tolerance)
    occluded = good & (hi < z - tolerance)
    votes[candidates[support]] = 1
    votes[candidates[free]] = 2
    votes[candidates[occluded]] = 3
    if instances is not None:
        ids = np.stack([instances[v + dy, u + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        same_id = (ids == ids[4]).all(0)
        use = support & same_id
        observed_ids[candidates[use]] = ids[4, use]
    return votes, observed_ids


def classify_counts(support, free, min_votes=3):
    labels = np.full(len(support), "unknown", dtype=object)
    a, b = support >= min_votes, free >= min_votes
    labels[a & ~b] = "current_supported"
    labels[b & ~a] = "contradicted_free_space"
    labels[a & b] = "conflicting_evidence"
    return labels


def write_ply(path, points, labels):
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(points))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, label in zip(points, labels):
            f.write("%.6f %.6f %.6f %d %d %d\n" % (*p, *COLORS[label]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surface-csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[2] / "results/abc_eval_final_baseline")
    ap.add_argument("--old-state", default="A_I13_static")
    ap.add_argument("--session", default="B", choices=list("ABC"))
    ap.add_argument("--start-s", type=float, required=True, help="Beginning of a scene interval with stable geometry in the evaluated region.")
    ap.add_argument("--end-s", type=float, required=True, help="No future frames beyond this point are used.")
    ap.add_argument("--frame-period-s", type=float, default=1.0)
    ap.add_argument("--old-distance", type=float, default=0.15)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--tolerance", type=float, default=0.10)
    ap.add_argument("--min-votes", type=int, default=3)
    ap.add_argument("--include-nonpresent", action="store_true", help="Also diagnose historical/nonpresent object mesh; never interpret this as the default current surface policy.")
    args = ap.parse_args()
    if args.end_s < args.start_s or min(args.frame_period_s, args.old_distance, args.voxel, args.tolerance, args.min_votes) <= 0:
        ap.error("Invalid interval or nonpositive parameter.")
    args.output.mkdir(parents=True, exist_ok=True)
    states = json.loads((args.reference / "gt_objects/states.json").read_text())
    old = next(s for s in states if s["id"] == args.old_state)
    old_cloud = np.load(args.reference / "gt_objects" / Path(old["surface"]).name)
    with args.surface_csv.open() as f:
        rows = list(csv.DictReader(f))
    excluded_nonpresent_samples = sum(r["layer"] == "object_nonpresent" for r in rows) if not args.include_nonpresent else 0
    if not args.include_nonpresent:
        rows = [r for r in rows if r["layer"] != "object_nonpresent"]
    points = np.array([[float(r[k]) for k in ("x", "y", "z")] for r in rows], dtype=float).reshape(-1, 3)
    old_distance = cKDTree(old_cloud).query(points)[0]
    keep = np.flatnonzero(old_distance <= args.old_distance)
    # One representative per layer and spatial voxel; preserve layer evidence,
    # but do not claim a union area or compare methods by these raw point counts.
    seen = set()
    unique = []
    for i in keep:
        key = (rows[i]["layer"], *np.floor(points[i] / args.voxel).astype(int))
        if key not in seen:
            seen.add(key)
            unique.append(i)
    points = points[unique]
    old_distance = old_distance[unique]
    rows = [rows[i] for i in unique]
    cfg = json.loads((args.reference / "input_manifest.json").read_text())["sessions"][args.session]
    rgbd, transform = Path(cfg["rgbd"]), np.array(cfg["transform"])
    with (rgbd / "timestamps.csv").open() as f:
        frames = list(csv.DictReader(f))
    selected, last = [], float("-inf")
    for frame in frames:
        t = int(frame["TimeStamp"]) / 1e9
        if args.start_s <= t <= args.end_s and t - last >= args.frame_period_s - 1e-8:
            selected.append((frame["ImageID"], t))
            last = t
    if not selected:
        raise ValueError("No sensor frames in the requested interval")
    support, free, occluded = [np.zeros(len(points), dtype=np.int32) for _ in range(3)]
    id_votes = np.zeros((len(points), 256), dtype=np.uint16)
    first_support_frame = np.full(len(points), -1, dtype=np.int32)
    first_free_frame = np.full(len(points), -1, dtype=np.int32)
    for index, (stem, t) in enumerate(selected):
        with Image.open(rgbd / (stem + "_depth.tiff")) as image:
            depth = np.array(image, dtype=np.float32)
        with Image.open(Path(cfg["semantic"]) / (stem + "_segmentation.png")) as image:
            semantic = np.array(image)
        with Image.open(Path(cfg["instance"]) / (stem + "_segmentation.png")) as image:
            instance = np.array(image)
        pose = transform @ np.loadtxt(rgbd / (stem + "_pose.txt"))
        votes, ids = project_evidence(points, depth, pose, cfg["intrinsics"], args.tolerance, semantic, instance)
        first_support_frame[(votes == 1) & (first_support_frame < 0)] = int(stem)
        first_free_frame[(votes == 2) & (first_free_frame < 0)] = int(stem)
        support += votes == 1
        free += votes == 2
        occluded += votes == 3
        known = np.flatnonzero((ids >= 0) & (ids < 256))
        id_votes[known, ids[known]] += 1
        if index % 20 == 0 or index == len(selected) - 1:
            print("depth evidence", index + 1, "/", len(selected), "points", len(points), flush=True)
    labels = classify_counts(support, free, args.min_votes)
    dominant = id_votes.argmax(1)
    dominant_count = id_votes.max(1)
    dominant[dominant_count < args.min_votes] = -1
    # Auxiliary GT distances must not silently compare an earlier window to the
    # final state. Only references known to hold throughout the chosen window
    # participate. They do not determine the depth-based class of any point.
    current_states = [s for s in states if s["session"] == args.session
        and s["state_validity"]["known_present_from_s"] <= args.start_s
        and s["state_validity"]["known_present_through_s"] >= args.end_s]
    current_points, current_ids = [], []
    for st in current_states:
        cloud = np.load(args.reference / "gt_objects" / Path(st["surface"]).name)
        current_points.append(cloud)
        current_ids.extend([st["object_id"]] * len(cloud))
    if current_points:
        gt_dist, gt_idx = cKDTree(np.concatenate(current_points)).query(points)
        nearest_current_id = np.asarray(current_ids)[gt_idx]
    else:
        gt_dist = np.full(len(points), np.inf)
        nearest_current_id = np.full(len(points), -1)
    output_rows = []
    for i, row in enumerate(rows):
        output_rows.append(dict(x=points[i, 0], y=points[i, 1], z=points[i, 2],
            layer=row["layer"], prediction_owner=row.get("owner", ""), label=labels[i],
            support_frames=int(support[i]), free_frames=int(free[i]), occluded_frames=int(occluded[i]),
            supporting_current_instance_id=int(dominant[i]), supporting_id_votes=int(dominant_count[i]),
            first_support_frame=int(first_support_frame[i]), first_free_frame=int(first_free_frame[i]),
            old_reference_distance_m=float(old_distance[i]), current_object_reference_distance_m=float(gt_dist[i]),
            nearest_current_object_id=int(nearest_current_id[i])))
    fieldnames = list(output_rows[0]) if output_rows else ["x", "y", "z", "layer", "prediction_owner", "label"]
    with (args.output / "classified_points.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    write_ply(args.output / "classified_points.ply", points, labels)
    replacements = [r for r in output_rows if r["label"] == "current_supported"
                    and r["supporting_current_instance_id"] > 0
                    and r["supporting_current_instance_id"] != old["object_id"]]
    summary = dict(protocol="actual_predicted_surface_vs_time_restricted_depth_v1_diagnostic",
        source_surface_csv=str(args.surface_csv), old_state=args.old_state, session=args.session,
        evidence_interval_s=[args.start_s, args.end_s], selected_frames=len(selected),
        selected_first_last_s=[selected[0][1], selected[-1][1]], tolerance_m=args.tolerance,
        include_nonpresent=args.include_nonpresent, excluded_nonpresent_samples=excluded_nonpresent_samples,
        min_distinct_sampled_frames=args.min_votes, depth_edge_max_range_m=0.05,
        old_surface_neighborhood_m=args.old_distance, voxel_sampling_m=args.voxel,
        candidate_sample_count=len(points), counts=dict(Counter(labels)),
        auxiliary_reference_states=[s["id"] for s in current_states],
        by_layer={layer: dict(Counter(r["label"] for r in output_rows if r["layer"] == layer))
                  for layer in sorted(set(r["layer"] for r in rows))},
        legal_current_other_object_sample_count=len(replacements),
        legal_current_other_object_ids=dict(Counter(r["supporting_current_instance_id"] for r in replacements)),
        meaning="Red: geometry contradicted by measured free space near old reference; it may be stale geometry or reconstruction error. Green: measured current surface. Orange: inconsistent observations. Gray: insufficient/unobservable.",
        limitations=["Offline stable-interval diagnostic; no online latency score.",
            "No proof of triangle ancestry or hidden object identity from coincident geometry.",
            "Not a cleanup success rate; initial representation and reference coverage are not a shared denominator.",
            "Per-layer voxel samples can overlap; counts are not unique surface area or cross-method ranking.",
            "Independent frames can share depth/pose errors; tolerance and state masks need validation before publication.",
            "Current object reference nearest neighbor is diagnostic only, not used to waive free-space evidence."])
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (args.output / "evidence_frames.json").write_text(json.dumps(selected, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
