"""Export disputed site identities from one completed visit's predicted RGB-D observations."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from predicted_site_association import SiteEvidence, Surface, TriangleSurface, name_regions

# This reader only decodes the timestamps stored in the source map.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval_fast/vendor/pack"))
from zpk_snapshot import stamps

RAY_SLICE = re.compile(
    r"STATE_SLICE inst=(\d+) stamp=(\d+).*?inherited_supported=(\d+)"
    r" .*?inherited_repl_other=(\d+)"
)


def ray_slices(path):
    counts = defaultdict(Counter)
    with path.open() as stream:
        for line in stream:
            match = RAY_SLICE.search(line)
            if match:
                track, stamp, own, replaced = map(int, match.groups())
                counts[stamp][track] += own - replaced
    return sorted(counts.items())


def load_sites(snapshot, start, resolution):
    sites = []
    for row in snapshot["objects"]:
        ids = row["details"].get("instance_id")
        if not ids or not row["present"] or not row["surface"]:
            continue
        points = np.asarray(row["surface"], dtype=np.float64)
        if not row["faces"]:
            mesh = None
        else:
            mesh = TriangleSurface(points, row["faces"])
        _, first = np.unique(np.floor(points / resolution), axis=0, return_index=True)
        sites.append(Surface(int(ids[0]), int(row["semantic"]), points[first], mesh))
    if int(snapshot["stamp_ns"]) < start:
        raise ValueError("Source snapshot predates the visit")
    return sites


def source_views(rgbd, start, map_stamps):
    with (rgbd / "timestamps.csv").open() as stream:
        frames = list(csv.DictReader(stream))
    clock = start + np.asarray([int(row["TimeStamp"]) for row in frames])
    if np.any(np.diff(clock) < 0):
        raise ValueError("Source camera timestamps are unordered")
    ends = np.asarray([v for v in map_stamps if v >= start], dtype=np.int64)
    middle = (np.r_[start, ends[:-1]] + ends) // 2
    before = np.searchsorted(clock, np.union1d(ends, middle), side="right") - 1
    return [(int(clock[i]), frames[i]["ImageID"]) for i in np.unique(before[before >= 0])]


def visible(points, pose, camera, depth, resolution):
    cam = (points - pose[:3, 3]) @ pose[:3, :3]
    idx = np.flatnonzero(cam[:, 2] > 0)
    fx, fy, cx, cy = camera
    u = np.floor(fx * cam[idx, 0] / cam[idx, 2] + cx).astype(int)
    v = np.floor(fy * cam[idx, 1] / cam[idx, 2] + cy).astype(int)
    h, w = depth.shape
    inside = (u >= 0) & (v >= 0) & (u < w) & (v < h)
    idx, v, u = idx[inside], v[inside], u[inside]
    z = depth[v, u]
    hit = (z > 0) & np.isfinite(z) & (np.abs(z - cam[idx, 2]) <= resolution)
    return idx[hit], v[hit], u[hit]


def regions(labels, depth, connectivity):
    pixels = np.arange(depth.size, dtype=np.int32).reshape(depth.shape)
    valid = np.isfinite(depth) & (depth > 0)
    edges = []
    for left, right in [(np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])]:
        with np.errstate(invalid="ignore"):
            linked = valid[left] & valid[right] & (labels[left] == labels[right])
            linked &= np.abs(depth[left] - depth[right]) <= connectivity
        edges.append((pixels[left][linked], pixels[right][linked]))
    a, b = (np.concatenate(edge) for edge in zip(*edges))
    graph = coo_matrix((np.ones(len(a), np.uint8), (a, b)), shape=(depth.size, depth.size))
    return connected_components(graph.tocsr(), directed=False)[1].reshape(depth.shape)


def build(args):
    resolution = args.resolution
    start = int(json.loads(args.manifest.read_text())["session_start_ns"])
    with args.source_surfaces.open() as src:
        snap = json.loads(next(src))
    times = stamps(str(args.source_map))
    if not times or int(snap["stamp_ns"]) != int(times[-1]):
        raise ValueError("Mesh export and map have different end timestamps")
    sites = load_sites(snap, start, resolution)
    views = source_views(args.rgbd, start, times)
    packet = dict(schema="predicted_site_handoff/v1", resolution_m=resolution,
                  source_stamp_ns=int(times[-1]), source_views=len(views), disputed_sites=[])
    if not sites:
        return packet
    prior = Counter()
    if args.prior_log:
        for _, delta in ray_slices(args.prior_log):
            prior.update(delta)
    states = [SiteEvidence(s, sites, prior[s.track] > 0) for s in sites]
    labels_by_id = {s.track: s.label for s in sites}
    labels_by_site = np.array([s.label for s in sites])
    points = np.concatenate([s.points for s in sites])
    bounds = np.r_[0, np.cumsum([len(s.points) for s in sites])]
    identity = np.repeat(np.arange(len(sites)), np.diff(bounds))
    to_world = np.loadtxt(args.to_world) if args.to_world else np.eye(4)
    cfg = dict(line.split(":", 1) for line in (args.rgbd / "Intrinsics.txt").read_text().splitlines() if ":" in line)
    camera = [float(cfg[k]) for k in ("f_x", "f_y", "u", "v")]
    for stamp, frame in views:
        pose = to_world @ np.loadtxt(args.rgbd / f"{frame}_pose.txt")
        depth = cv2.imread(str(args.rgbd / f"{frame}_depth.tiff"), cv2.IMREAD_UNCHANGED)
        raw = cv2.imread(str(args.predictions / f"{frame}_segmentation.png"), cv2.IMREAD_UNCHANGED)
        if raw is None or depth is None or raw.shape != depth.shape or depth.dtype != np.float32:
            raise ValueError(f"Invalid depth or prediction at {frame}")
        idx, v, u = visible(points, pose, camera, depth, resolution)
        stride = args.pixel_stride
        labels = raw[::stride, ::stride]
        segment = regions(labels, depth[::stride, ::stride], args.depth_connectivity)
        region = segment[v // stride, u // stride]
        raw_label = labels[v // stride, u // stride]
        named = name_regions(region, identity[idx], raw_label == labels_by_site[identity[idx]],
                             region, len(sites))
        splits = np.searchsorted(idx, bounds)
        for i, state in enumerate(states):
            part = slice(splits[i], splits[i+1])
            state.observe(stamp, idx[part] - bounds[i], raw_label[part], named[part])
    balance = Counter()
    for stamp, delta in ray_slices(args.source_log):
        if views and stamp <= views[-1][0]:
            balance.update(delta)
    for state in states:
        claim = state.claim(balance, resolution)
        if claim is None:
            continue
        site = state.handoff(claim, resolution)
        packet["disputed_sites"].append(dict(
            source_id=site.surface.track, source_class=site.surface.label,
            explaining_id=claim.host, explaining_class=labels_by_id[claim.host],
            points_world_m=site.surface.points.tolist(), shared_indices=np.flatnonzero(claim.shared).tolist(),
            independent_indices=np.flatnonzero(site.independent).tolist(),
        ))
    return packet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("rgbd", "predictions", "source-map", "source-surfaces", "source-log", "manifest"):
        parser.add_argument("--" + field, required=True, type=Path)
    parser.add_argument("--to-world", type=Path)
    parser.add_argument("--prior-log", type=Path)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--depth-connectivity", type=float, default=0.10)
    parser.add_argument("--pixel-stride", type=int, default=4)
    cv2.setNumThreads(1)
    print(json.dumps(build(parser.parse_args())))
