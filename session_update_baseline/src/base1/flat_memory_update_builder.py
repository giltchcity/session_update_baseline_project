#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fin:
        return list(csv.DictReader(fin))


def as_int(row: dict[str, str], key: str, fallback: int = 0) -> int:
    try:
        return int(row.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def as_float(row: dict[str, str], key: str, fallback: float = math.nan) -> float:
    try:
        return float(row.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def parse_ply_header(path: Path) -> tuple[str, int, list[tuple[str, str]], int, int]:
    header_bytes = 0
    header_lines: list[str] = []
    with path.open("rb") as fin:
        while True:
            raw = fin.readline()
            if not raw:
                raise ValueError(f"missing PLY end_header: {path}")
            header_bytes += len(raw)
            line = raw.decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

    fmt = ""
    vertex_count = -1
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertex = True
        elif parts[0] == "element" and parts[1] != "vertex":
            in_vertex = False
        elif in_vertex and parts[0] == "property" and len(parts) >= 3:
            properties.append((parts[1], parts[-1]))

    if not fmt or vertex_count < 0:
        raise ValueError(f"invalid PLY header: {path}")
    return fmt, vertex_count, properties, header_bytes, len(header_lines)


def ply_dtype(properties: list[tuple[str, str]]) -> np.dtype:
    type_map = {
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
        "uchar": "u1",
        "uint8": "u1",
        "char": "i1",
        "int8": "i1",
        "ushort": "<u2",
        "uint16": "<u2",
        "short": "<i2",
        "int16": "<i2",
        "uint": "<u4",
        "uint32": "<u4",
        "int": "<i4",
        "int32": "<i4",
    }
    fields = []
    for ply_type, name in properties:
        if ply_type not in type_map:
            raise ValueError(f"unsupported PLY property type: {ply_type}")
        fields.append((name, type_map[ply_type]))
    return np.dtype(fields)


def load_ply_points(path: Path) -> np.ndarray:
    fmt, vertex_count, properties, header_bytes, header_lines = parse_ply_header(path)
    prop_names = [name for _, name in properties]
    has_origin = {"origin_x", "origin_y", "origin_z"}.issubset(prop_names)
    has_color = {"red", "green", "blue"}.issubset(prop_names)
    if vertex_count == 0:
        width = 3 + (3 if has_origin else 0) + (3 if has_color else 0)
        return np.empty((0, width), dtype=np.float32)
    for name in ("x", "y", "z"):
        if name not in prop_names:
            raise ValueError(f"{path} is missing {name}")

    if fmt == "ascii":
        data = np.loadtxt(str(path), skiprows=header_lines, max_rows=vertex_count, ndmin=2)
        cols = [prop_names.index(axis) for axis in ("x", "y", "z")]
        if has_origin:
            cols += [prop_names.index(axis) for axis in ("origin_x", "origin_y", "origin_z")]
        if has_color:
            cols += [prop_names.index(axis) for axis in ("red", "green", "blue")]
        return np.asarray(data[:, cols], dtype=np.float32)
    if fmt != "binary_little_endian":
        raise ValueError(f"unsupported PLY format {fmt}: {path}")

    with path.open("rb") as fin:
        fin.seek(header_bytes)
        data = np.fromfile(fin, dtype=ply_dtype(properties), count=vertex_count)

    cols = [data["x"], data["y"], data["z"]]
    if has_origin:
        cols += [data["origin_x"], data["origin_y"], data["origin_z"]]
    if has_color:
        cols += [data["red"], data["green"], data["blue"]]
    return np.column_stack(cols).astype(np.float32, copy=False)


def write_binary_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] not in (3, 6, 9):
        raise ValueError(f"expected Nx3, Nx6, or Nx9 points for {path}, got {pts.shape}")
    has_origin = pts.shape[1] in (6, 9)
    has_color = pts.shape[1] == 9
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
    ).encode("ascii")
    if has_origin:
        header += (
            "property float origin_x\n"
            "property float origin_y\n"
            "property float origin_z\n"
        ).encode("ascii")
    if has_color:
        header += (
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
        ).encode("ascii")
    header += "end_header\n".encode("ascii")
    with path.open("wb") as fout:
        fout.write(header)
        if not has_color:
            np.asarray(pts, dtype="<f4").tofile(fout)
            return
        structured = np.empty(
            len(pts),
            dtype=np.dtype(
                [
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("origin_x", "<f4"),
                    ("origin_y", "<f4"),
                    ("origin_z", "<f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                ]
            ),
        )
        structured["x"] = pts[:, 0].astype("<f4")
        structured["y"] = pts[:, 1].astype("<f4")
        structured["z"] = pts[:, 2].astype("<f4")
        structured["origin_x"] = pts[:, 3].astype("<f4")
        structured["origin_y"] = pts[:, 4].astype("<f4")
        structured["origin_z"] = pts[:, 5].astype("<f4")
        colors = np.clip(np.rint(pts[:, 6:9]), 0, 255).astype(np.uint8)
        structured["red"] = colors[:, 0]
        structured["green"] = colors[:, 1]
        structured["blue"] = colors[:, 2]
        structured.tofile(fout)


def voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if len(points) == 0 or voxel_size_m <= 0.0:
        return points.astype(np.float32, copy=False)
    cells = np.floor(points[:, :3] / voxel_size_m).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    indices.sort()
    return points[indices].astype(np.float32, copy=False)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def logsumexp2(a: float, b: float) -> float:
    m = max(a, b)
    if not math.isfinite(m):
        return m
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def log_normal_pdf(x: float, mean: float, variance: float) -> float:
    variance = max(variance, 1.0e-9)
    return -0.5 * (math.log(2.0 * math.pi * variance) + ((x - mean) ** 2) / variance)


def log_uniform_pdf(x: float, lo: float, hi: float) -> float:
    if x < lo or x > hi or hi <= lo:
        return -math.inf
    return -math.log(hi - lo)


def log_beta_fn(alpha: float, beta: float) -> float:
    alpha = max(alpha, 1.0e-6)
    beta = max(beta, 1.0e-6)
    return math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)


def beta_second_moment(alpha: float, beta: float) -> float:
    denom = (alpha + beta) * (alpha + beta + 1.0)
    if denom <= 0.0:
        return 0.0
    return alpha * (alpha + 1.0) / denom


def beta_from_moments(mean: float, second: float) -> tuple[float, float]:
    mean = clamp(mean, 1.0e-5, 1.0 - 1.0e-5)
    variance = max(1.0e-8, second - mean * mean)
    concentration = mean * (1.0 - mean) / variance - 1.0
    if not math.isfinite(concentration) or concentration <= 1.0e-6:
        concentration = 1.0e-6
    return mean * concentration, (1.0 - mean) * concentration


@dataclass
class PocdState:
    mu: float
    sigma: float
    alpha: float
    beta: float

    @property
    def stationarity(self) -> float:
        denom = self.alpha + self.beta
        return self.alpha / denom if denom > 0.0 else 0.5


@dataclass
class PocdUpdate:
    prior: PocdState
    posterior: PocdState
    delta: float
    semantic_stationary: int
    adaptive_k: float
    inlier_weight: float
    outlier_weight: float
    inlier_mean: float
    inlier_sigma: float
    geometry_log_likelihood_inlier: float
    geometry_log_likelihood_outlier: float


def pocd_bayesian_update(
    prior: PocdState,
    delta: float,
    semantic_stationary: int,
    adaptive_k: float,
    tau_m: float,
    delta_max_m: float,
) -> PocdUpdate:
    """POCD Gaussian-Beta update with two-component moment matching.

    Implements the RSS'22 POCD model:
    q(l, v) = N(l | mu, sigma^2) Beta(v | alpha, beta),
    p(Delta | l, v) = v N(Delta | l, tau^2) +
                      (1-v) U(Delta | -Delta_max, Delta_max),
    p(s | v) = Bernoulli(s | v)^k.
    """
    semantic_stationary = 1 if semantic_stationary else 0
    adaptive_k = max(0.0, adaptive_k)
    tau_m = max(tau_m, 1.0e-6)
    delta_max_m = max(delta_max_m, tau_m)
    sigma = max(prior.sigma, 1.0e-6)
    sigma2 = sigma * sigma
    tau2 = tau_m * tau_m

    gamma2 = 1.0 / (1.0 / sigma2 + 1.0 / tau2)
    gamma = math.sqrt(gamma2)
    inlier_mean = gamma2 * (prior.mu / sigma2 + delta / tau2)

    a1 = prior.alpha + adaptive_k * semantic_stationary + 1.0
    b1 = prior.beta + adaptive_k * (1 - semantic_stationary)
    a2 = prior.alpha + adaptive_k * semantic_stationary
    b2 = prior.beta + adaptive_k * (1 - semantic_stationary) + 1.0

    log_geom_inlier = log_normal_pdf(delta, prior.mu, sigma2 + tau2)
    log_geom_outlier = log_uniform_pdf(delta, -delta_max_m, delta_max_m)
    if not math.isfinite(log_geom_outlier):
        log_geom_outlier = log_uniform_pdf(
            clamp(delta, -delta_max_m, delta_max_m),
            -delta_max_m,
            delta_max_m,
        )
    log_prior_beta = log_beta_fn(prior.alpha, prior.beta)
    log_c1 = log_geom_inlier + log_beta_fn(a1, b1) - log_prior_beta
    log_c2 = log_geom_outlier + log_beta_fn(a2, b2) - log_prior_beta
    norm = logsumexp2(log_c1, log_c2)
    if not math.isfinite(norm):
        inlier_weight = 0.0
        outlier_weight = 1.0
    else:
        inlier_weight = math.exp(log_c1 - norm)
        outlier_weight = math.exp(log_c2 - norm)

    mu_new = inlier_weight * inlier_mean + outlier_weight * prior.mu
    l_second = (
        inlier_weight * (gamma2 + inlier_mean * inlier_mean)
        + outlier_weight * (sigma2 + prior.mu * prior.mu)
    )
    sigma_new = math.sqrt(max(1.0e-8, l_second - mu_new * mu_new))

    v_mean = (
        inlier_weight * (a1 / (a1 + b1))
        + outlier_weight * (a2 / (a2 + b2))
    )
    v_second = (
        inlier_weight * beta_second_moment(a1, b1)
        + outlier_weight * beta_second_moment(a2, b2)
    )
    alpha_new, beta_new = beta_from_moments(v_mean, v_second)
    posterior = PocdState(mu=mu_new, sigma=sigma_new, alpha=alpha_new, beta=beta_new)
    return PocdUpdate(
        prior=prior,
        posterior=posterior,
        delta=delta,
        semantic_stationary=semantic_stationary,
        adaptive_k=adaptive_k,
        inlier_weight=inlier_weight,
        outlier_weight=outlier_weight,
        inlier_mean=inlier_mean,
        inlier_sigma=gamma,
        geometry_log_likelihood_inlier=log_geom_inlier,
        geometry_log_likelihood_outlier=log_geom_outlier,
    )


def point_stats(points: np.ndarray) -> dict[str, Any]:
    if len(points) == 0:
        nan = float("nan")
        return {
            "points": 0,
            "bbox_min": [nan, nan, nan],
            "bbox_max": [nan, nan, nan],
            "centroid": [nan, nan, nan],
            "extent": [nan, nan, nan],
        }
    xyz = points[:, :3]
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    centroid = xyz.mean(axis=0)
    return {
        "points": int(len(points)),
        "bbox_min": [float(x) for x in mn],
        "bbox_max": [float(x) for x in mx],
        "centroid": [float(x) for x in centroid],
        "extent": [float(x) for x in (mx - mn)],
    }


def build_grid(points: np.ndarray, cell_size: float) -> dict[tuple[int, int, int], list[int]]:
    if len(points) == 0:
        return {}
    cells = np.floor(points[:, :3] / cell_size).astype(np.int64)
    grid: dict[tuple[int, int, int], list[int]] = {}
    for idx, cell in enumerate(cells):
        key = (int(cell[0]), int(cell[1]), int(cell[2]))
        grid.setdefault(key, []).append(idx)
    return grid


def voxel_key(point: np.ndarray, voxel_size_m: float) -> tuple[int, int, int]:
    cell = np.floor(point[:3] / voxel_size_m).astype(np.int64)
    return int(cell[0]), int(cell[1]), int(cell[2])


def local_ray_tsdf(
    points: np.ndarray,
    voxel_size_m: float,
    truncation_m: float,
    max_points: int,
) -> dict[tuple[int, int, int], float]:
    points = deterministic_sample(points, max_points)
    tsdf: dict[tuple[int, int, int], float] = {}
    if len(points) == 0:
        return tsdf
    voxel_size_m = max(voxel_size_m, 1.0e-3)
    truncation_m = max(truncation_m, voxel_size_m)
    depth_steps = max(1, int(math.ceil(truncation_m / voxel_size_m)))
    for sample in points:
        surface = sample[:3].astype(np.float32)
        if sample.shape[0] >= 6:
            origin = sample[3:6].astype(np.float32)
            ray = surface - origin
            depth = float(np.linalg.norm(ray))
            if not math.isfinite(depth) or depth <= 1.0e-6:
                direction = None
            else:
                direction = ray / depth
        else:
            origin = None
            direction = None

        if direction is None or origin is None:
            base = voxel_key(surface, voxel_size_m)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        key = (base[0] + dx, base[1] + dy, base[2] + dz)
                        center = (np.asarray(key, dtype=np.float32) + 0.5) * voxel_size_m
                        sdf = float(np.linalg.norm(center - surface))
                        if sdf <= truncation_m:
                            old = tsdf.get(key)
                            if old is None or abs(sdf) < abs(old):
                                tsdf[key] = sdf
            continue

        for step in range(-depth_steps, depth_steps + 1):
            probe = surface + direction * (step * voxel_size_m)
            key = voxel_key(probe, voxel_size_m)
            center = (np.asarray(key, dtype=np.float32) + 0.5) * voxel_size_m
            voxel_depth = float(np.dot(center - origin, direction))
            if voxel_depth <= 0.0:
                continue
            signed_distance = clamp(depth - voxel_depth, -truncation_m, truncation_m)
            old = tsdf.get(key)
            if old is None or abs(signed_distance) < abs(old):
                tsdf[key] = signed_distance
    return tsdf


def tsdf_change_measure(
    prior_points: np.ndarray,
    current_points: np.ndarray,
    voxel_size_m: float,
    truncation_m: float,
    lambda_diff: float,
    max_points: int,
) -> tuple[float, int]:
    prior_tsdf = local_ray_tsdf(prior_points, voxel_size_m, truncation_m, max_points)
    current_tsdf = local_ray_tsdf(current_points, voxel_size_m, truncation_m, max_points)
    overlap = set(prior_tsdf) & set(current_tsdf)
    if not overlap:
        return float("nan"), 0
    diff = sum(abs(current_tsdf[key] - prior_tsdf[key]) for key in overlap)
    return lambda_diff * diff / len(overlap), len(overlap)


def semantic_stationarity_class(name: str, class_name: str, class_id: int) -> int:
    text = f"{name} {class_name} {class_id}".lower()
    dynamic_tokens = (
        "person",
        "human",
        "robot",
        "cart",
        "trolley",
        "vehicle",
        "animal",
    )
    return 0 if any(token in text for token in dynamic_tokens) else 1


def adaptive_stationarity_weight(
    semantic_stationary: int,
    delta_m: float,
    tau_m: float,
    unobserved_visible: bool,
) -> float:
    if unobserved_visible:
        return 0.0
    if not semantic_stationary:
        return 1.0
    if abs(delta_m) <= tau_m:
        return 1.0
    if abs(delta_m) <= 2.0 * tau_m:
        return 0.5
    return 0.0


def deterministic_sample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices]


def fraction_points_within(
    query: np.ndarray,
    target: np.ndarray,
    threshold: float,
    max_query_points: int,
    max_target_points: int,
) -> float:
    query = deterministic_sample(query[:, :3], max_query_points)
    target = deterministic_sample(target[:, :3], max_target_points)
    if len(query) == 0:
        return 1.0
    if len(target) == 0:
        return 0.0
    grid = build_grid(target, threshold)
    threshold_sq = threshold * threshold
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    hits = 0
    query_cells = np.floor(query / threshold).astype(np.int64)
    for point, cell in zip(query, query_cells):
        base = (int(cell[0]), int(cell[1]), int(cell[2]))
        found = False
        for dx, dy, dz in offsets:
            bucket = grid.get((base[0] + dx, base[1] + dy, base[2] + dz))
            if not bucket:
                continue
            candidates = target[bucket]
            d2 = np.sum((candidates - point) ** 2, axis=1)
            if np.any(d2 <= threshold_sq):
                found = True
                break
        if found:
            hits += 1
    return hits / len(query)


def supported_points_within(
    query: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> np.ndarray:
    if len(query) == 0:
        return query
    if len(target) == 0:
        return query[:0]
    tree = cKDTree(target[:, :3])
    distances, _ = tree.query(
        query[:, :3],
        k=1,
        distance_upper_bound=threshold,
        workers=-1,
    )
    keep = np.isfinite(distances)
    return query[keep]


def free_cell_key(point: np.ndarray, cell_size: float) -> tuple[int, int, int]:
    cell = np.floor(point / cell_size).astype(np.int64)
    return int(cell[0]), int(cell[1]), int(cell[2])


def build_current_free_space(
    objects: dict[int, dict[str, Any]],
    cell_size_m: float,
    surface_clearance_m: float,
    ray_stride: int,
) -> set[tuple[int, int, int]]:
    free_cells: set[tuple[int, int, int]] = set()
    step_m = max(cell_size_m * 0.75, 0.03)
    for obj in objects.values():
        points = obj["points"]
        if points.shape[1] < 6:
            continue
        rays = points[:: max(1, ray_stride)]
        if len(rays) == 0:
            continue
        origins = rays[:, 3:6]
        surfaces = rays[:, 0:3]
        vectors = surfaces - origins
        depths = np.linalg.norm(vectors, axis=1)
        valid = np.isfinite(depths) & (depths > surface_clearance_m + step_m)
        origins = origins[valid]
        surfaces = surfaces[valid]
        depths = depths[valid]
        vectors = vectors[valid]
        for origin, vector, depth in zip(origins, vectors, depths):
            direction = vector / depth
            max_t = depth - surface_clearance_m
            t = step_m
            while t < max_t:
                free_cells.add(free_cell_key(origin + direction * t, cell_size_m))
                t += step_m
    return free_cells


def free_space_fraction(
    points: np.ndarray,
    free_cells: set[tuple[int, int, int]],
    cell_size_m: float,
    max_points: int,
) -> float:
    sampled = deterministic_sample(points[:, :3], max_points)
    if len(sampled) == 0:
        return 0.0
    hits = 0
    for point in sampled:
        if free_cell_key(point, cell_size_m) in free_cells:
            hits += 1
    return hits / len(sampled)


def load_object_map(map_dir: Path) -> dict[int, dict[str, Any]]:
    rows = read_csv_rows(map_dir / "object_summary.csv")
    memory_by_label: dict[int, dict[str, Any]] = {}
    memory_path = map_dir / "object_memory.json"
    if memory_path.exists():
        try:
            payload = json.loads(memory_path.read_text())
            for item in payload.get("objects", []):
                label_id = int(item.get("label_id", -1))
                if label_id >= 0:
                    memory_by_label[label_id] = item
        except (json.JSONDecodeError, TypeError, ValueError):
            memory_by_label = {}
    objects: dict[int, dict[str, Any]] = {}
    for row in rows:
        label_id = as_int(row, "label_id", -1)
        points_file = row.get("points_file", "")
        points = load_ply_points(map_dir / points_file) if points_file else np.empty((0, 3), dtype=np.float32)
        objects[label_id] = {
            "row": row,
            "points": points,
            "memory": memory_by_label.get(label_id, {}),
        }
    return objects


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def output_row_from_stats(
    meta: dict[str, str],
    stats: dict[str, Any],
    points_file: str,
    session_state: str,
) -> dict[str, Any]:
    return {
        "label_id": as_int(meta, "label_id", -1),
        "name": meta.get("name", ""),
        "class_id": as_int(meta, "class_id", -1),
        "class_name": meta.get("class_name", "unknown"),
        "panoptic_id": as_int(meta, "panoptic_id", -1),
        "mesh_id": as_int(meta, "mesh_id", -1),
        "size": meta.get("size", ""),
        "points": stats["points"],
        "bbox_min_x": stats["bbox_min"][0],
        "bbox_min_y": stats["bbox_min"][1],
        "bbox_min_z": stats["bbox_min"][2],
        "bbox_max_x": stats["bbox_max"][0],
        "bbox_max_y": stats["bbox_max"][1],
        "bbox_max_z": stats["bbox_max"][2],
        "centroid_x": stats["centroid"][0],
        "centroid_y": stats["centroid"][1],
        "centroid_z": stats["centroid"][2],
        "extent_x": stats["extent"][0],
        "extent_y": stats["extent"][1],
        "extent_z": stats["extent"][2],
        "points_file": points_file,
        "session_state": session_state,
    }


def memory_object_from_stats(
    row: dict[str, Any],
    stats: dict[str, Any],
    decision: str,
    belief: PocdState,
) -> dict[str, Any]:
    return {
        "label_id": row["label_id"],
        "name": row["name"],
        "class_id": row["class_id"],
        "class_name": row["class_name"],
        "points": int(stats["points"]),
        "bbox_min": stats["bbox_min"],
        "bbox_max": stats["bbox_max"],
        "centroid": stats["centroid"],
        "extent": stats["extent"],
        "points_file": row["points_file"],
        "stationarity_mu": belief.mu,
        "stationarity_sigma": belief.sigma,
        "stationarity_alpha": belief.alpha,
        "stationarity_beta": belief.beta,
        "stationarity_expectation": belief.stationarity,
        "session_state": decision,
    }


def prior_belief_from_memory(memory: dict[str, Any], v0: float, tau_m: float) -> PocdState:
    alpha = float(memory.get("stationarity_alpha", 2.0 * v0))
    beta = float(memory.get("stationarity_beta", 2.0 * (1.0 - v0)))
    mu = float(memory.get("stationarity_mu", 0.0))
    sigma = float(memory.get("stationarity_sigma", max(tau_m, 1.0e-3)))
    if alpha <= 0.0 or beta <= 0.0:
        alpha = 2.0 * v0
        beta = 2.0 * (1.0 - v0)
    return PocdState(mu=mu, sigma=max(sigma, 1.0e-3), alpha=alpha, beta=beta)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Base1 A->B memory-updated object map from two Base1 flat clean maps."
    )
    parser.add_argument("--a-map-dir", type=Path, required=True)
    parser.add_argument("--b-map-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--object-voxel-size-m", type=float, default=0.03)
    parser.add_argument("--map-voxel-size-m", type=float, default=0.03)
    parser.add_argument("--association-distance-m", type=float, default=0.10)
    parser.add_argument("--stable-current-agreement", type=float, default=0.55)
    parser.add_argument("--free-space-cell-size-m", type=float, default=0.10)
    parser.add_argument("--free-space-conflict-ratio", type=float, default=0.20)
    parser.add_argument("--ray-stride", type=int, default=1)
    parser.add_argument("--max-association-points", type=int, default=8000)
    parser.add_argument("--max-visibility-points", type=int, default=12000)
    parser.add_argument("--pocd-v0", type=float, default=0.67)
    parser.add_argument("--pocd-theta-stat", type=float, default=0.40)
    parser.add_argument("--pocd-delta-max-m", type=float, default=4.0)
    parser.add_argument("--pocd-tau-m", type=float, default=0.20)
    parser.add_argument("--pocd-lambda-diff", type=float, default=1.6)
    parser.add_argument("--pocd-tsdf-voxel-size-m", type=float, default=0.05)
    parser.add_argument("--pocd-max-tsdf-points", type=int, default=6000)
    parser.add_argument(
        "--filter-background-prior",
        action="store_true",
        help="Experimental: materialize only B-supported prior background geometry.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    objects_dir = args.out_dir / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)

    a_objects = load_object_map(args.a_map_dir)
    b_objects = load_object_map(args.b_map_dir)
    needs_absence_evidence = bool(set(a_objects) - set(b_objects))
    free_cells = (
        build_current_free_space(
            b_objects,
            args.free_space_cell_size_m,
            args.association_distance_m,
            args.ray_stride,
        )
        if needs_absence_evidence
        else set()
    )

    output_rows: list[dict[str, Any]] = []
    memory_objects: list[dict[str, Any]] = []
    object_beliefs: list[dict[str, Any]] = []
    visibility_audit: list[dict[str, Any]] = []
    association_audit: list[dict[str, Any]] = []
    selected_points: list[np.ndarray] = []
    decision_counts: Counter[str] = Counter()

    for label_id in sorted(set(a_objects) | set(b_objects)):
        a_obj = a_objects.get(label_id)
        b_obj = b_objects.get(label_id)
        a_points = a_obj["points"] if a_obj else np.empty((0, 6), dtype=np.float32)
        b_points = b_obj["points"] if b_obj else np.empty((0, 6), dtype=np.float32)
        meta = (b_obj or a_obj)["row"]
        panoptic_id = as_int(meta, "panoptic_id", -1)
        name = meta.get("name", f"ID_{label_id}")

        free_ratio = free_space_fraction(
            a_points,
            free_cells,
            args.free_space_cell_size_m,
            args.max_visibility_points,
        ) if a_obj else 0.0
        current_agrees_with_prior = (
            fraction_points_within(
                b_points,
                a_points,
                args.association_distance_m,
                args.max_association_points,
                args.max_association_points,
            )
            if a_obj and b_obj
            else 0.0
        )
        prior_covered_by_current = (
            fraction_points_within(
                a_points,
                b_points,
                args.association_distance_m,
                args.max_association_points,
                args.max_association_points,
            )
            if a_obj and b_obj
            else 0.0
        )
        prior_supported_points = (
            supported_points_within(
                a_points,
                b_points,
                args.association_distance_m,
            )
            if a_obj and b_obj
            else np.empty((0, a_points.shape[1]), dtype=a_points.dtype)
        )
        prior_supported_count = int(len(prior_supported_points))
        prior_support_ratio = (
            prior_supported_count / len(a_points)
            if a_obj and len(a_points) > 0
            else 0.0
        )

        class_id = as_int(meta, "class_id", -1)
        class_name = meta.get("class_name", "unknown")
        prior_memory = a_obj.get("memory", {}) if a_obj else {}
        prior_belief = prior_belief_from_memory(
            prior_memory,
            args.pocd_v0,
            args.pocd_tau_m,
        )
        semantic_stationary = semantic_stationarity_class(name, class_name, class_id)
        pocd_delta = 0.0
        pocd_delta_source = "none"
        pocd_tsdf_overlap = 0
        adaptive_k = 0.0

        if b_obj and not a_obj:
            pocd_delta_source = "new_current_zero_delta"
            adaptive_k = adaptive_stationarity_weight(
                semantic_stationary,
                pocd_delta,
                args.pocd_tau_m,
                False,
            )
            pocd_update = pocd_bayesian_update(
                prior_belief,
                pocd_delta,
                semantic_stationary,
                adaptive_k,
                args.pocd_tau_m,
                args.pocd_delta_max_m,
            )
            decision = "new_current_bayesian"
            final_points = b_points
        elif a_obj and not b_obj:
            visible_unobserved = free_ratio >= args.free_space_conflict_ratio
            if visible_unobserved:
                pocd_delta = args.pocd_delta_max_m
                pocd_delta_source = "visible_prior_missing_delta_max"
            else:
                pocd_delta = 0.0
                pocd_delta_source = "hidden_unobserved_no_geometry_update"
            adaptive_k = adaptive_stationarity_weight(
                semantic_stationary,
                pocd_delta,
                args.pocd_tau_m,
                visible_unobserved,
            )
            pocd_update = pocd_bayesian_update(
                prior_belief,
                pocd_delta,
                semantic_stationary,
                adaptive_k,
                args.pocd_tau_m,
                args.pocd_delta_max_m,
            )
            if visible_unobserved and (
                pocd_update.posterior.stationarity < args.pocd_theta_stat
                or pocd_update.outlier_weight > pocd_update.inlier_weight
            ):
                decision = "absent_prior_dropped_bayesian"
            else:
                decision = "unobserved_prior_memory_only_bayesian"
            final_points = np.empty((0, a_points.shape[1]), dtype=np.float32)
        elif b_obj and a_obj and panoptic_id == 0:
            pocd_delta_source = "background_passthrough"
            pocd_update = pocd_bayesian_update(
                prior_belief,
                pocd_delta,
                1,
                adaptive_k,
                args.pocd_tau_m,
                args.pocd_delta_max_m,
            )
            if not args.filter_background_prior:
                decision = "background_union"
                final_points = np.concatenate([b_points, a_points], axis=0)
            elif prior_supported_count == len(a_points):
                decision = "background_union"
                final_points = np.concatenate([b_points, a_points], axis=0)
            elif prior_supported_count == 0:
                decision = "background_current_only_support_filter"
                final_points = b_points
            else:
                decision = "background_current_supported_prior"
                final_points = np.concatenate([b_points, prior_supported_points], axis=0)
        elif b_obj and a_obj:
            pocd_delta, pocd_tsdf_overlap = tsdf_change_measure(
                a_points,
                b_points,
                args.pocd_tsdf_voxel_size_m,
                args.pocd_tau_m,
                args.pocd_lambda_diff,
                args.pocd_max_tsdf_points,
            )
            if not math.isfinite(pocd_delta):
                agreement = 0.5 * (current_agrees_with_prior + prior_covered_by_current)
                pocd_delta = min(
                    args.pocd_delta_max_m,
                    args.pocd_lambda_diff * max(0.0, 1.0 - agreement) * args.pocd_tau_m,
                )
                pocd_delta_source = "agreement_fallback_no_tsdf_overlap"
            else:
                pocd_delta_source = "local_ray_tsdf_difference"
            adaptive_k = adaptive_stationarity_weight(
                semantic_stationary,
                pocd_delta,
                args.pocd_tau_m,
                False,
            )
            pocd_update = pocd_bayesian_update(
                prior_belief,
                pocd_delta,
                semantic_stationary,
                adaptive_k,
                args.pocd_tau_m,
                args.pocd_delta_max_m,
            )
            if (
                pocd_update.posterior.stationarity >= args.pocd_theta_stat
                and pocd_update.inlier_weight >= pocd_update.outlier_weight
            ):
                if prior_supported_count == len(a_points):
                    decision = "persistent_union_bayesian"
                    final_points = np.concatenate([b_points, a_points], axis=0)
                elif prior_supported_count == 0:
                    decision = "changed_current_only_support_filter_bayesian"
                    final_points = b_points
                else:
                    decision = "persistent_current_supported_prior_bayesian"
                    final_points = np.concatenate([b_points, prior_supported_points], axis=0)
            else:
                decision = "changed_current_only_bayesian"
                final_points = b_points
        else:
            pocd_update = pocd_bayesian_update(
                prior_belief,
                pocd_delta,
                semantic_stationary,
                adaptive_k,
                args.pocd_tau_m,
                args.pocd_delta_max_m,
            )
            decision = "empty"
            final_points = np.empty((0, 3), dtype=np.float32)

        posterior_belief = pocd_update.posterior

        decision_counts[decision] += 1

        association_audit.append(
            {
                "label_id": label_id,
                "name": name,
                "class_id": class_id,
                "class_name": class_name,
                "panoptic_id": panoptic_id,
                "a_points": int(len(a_points)),
                "b_points": int(len(b_points)),
                "current_agrees_with_prior": current_agrees_with_prior,
                "prior_covered_by_current": prior_covered_by_current,
                "prior_supported_by_current_points": prior_supported_count,
                "prior_support_ratio": prior_support_ratio,
                "pocd_delta_m": pocd_delta,
                "pocd_delta_source": pocd_delta_source,
                "pocd_tsdf_overlap_voxels": pocd_tsdf_overlap,
                "pocd_inlier_weight": pocd_update.inlier_weight,
                "pocd_outlier_weight": pocd_update.outlier_weight,
                "stationarity_before": prior_belief.stationarity,
                "stationarity_after": posterior_belief.stationarity,
                "decision": decision,
            }
        )
        visibility_audit.append(
            {
                "label_id": label_id,
                "name": name,
                "class_id": class_id,
                "class_name": class_name,
                "panoptic_id": panoptic_id,
                "a_points": int(len(a_points)),
                "b_points": int(len(b_points)),
                "b_free_space_fraction_of_a": free_ratio,
                "free_space_cells": len(free_cells),
                "pocd_delta_m": pocd_delta,
                "pocd_delta_source": pocd_delta_source,
                "stationarity_after": posterior_belief.stationarity,
                "prior_supported_by_current_points": prior_supported_count,
                "prior_support_ratio": prior_support_ratio,
                "decision": decision,
            }
        )
        object_beliefs.append(
            {
                "label_id": label_id,
                "name": name,
                "class_id": class_id,
                "class_name": class_name,
                "panoptic_id": panoptic_id,
                "prior_present": bool(a_obj),
                "current_present": bool(b_obj),
                "pocd_mu_before": prior_belief.mu,
                "pocd_sigma_before": prior_belief.sigma,
                "pocd_alpha_before": prior_belief.alpha,
                "pocd_beta_before": prior_belief.beta,
                "stationarity_before": prior_belief.stationarity,
                "pocd_delta_m": pocd_delta,
                "pocd_delta_source": pocd_delta_source,
                "pocd_tsdf_overlap_voxels": pocd_tsdf_overlap,
                "semantic_stationary": semantic_stationary,
                "adaptive_k": adaptive_k,
                "pocd_inlier_weight": pocd_update.inlier_weight,
                "pocd_outlier_weight": pocd_update.outlier_weight,
                "pocd_inlier_mean": pocd_update.inlier_mean,
                "pocd_inlier_sigma": pocd_update.inlier_sigma,
                "pocd_log_likelihood_inlier": pocd_update.geometry_log_likelihood_inlier,
                "pocd_log_likelihood_outlier": pocd_update.geometry_log_likelihood_outlier,
                "pocd_mu_after": posterior_belief.mu,
                "pocd_sigma_after": posterior_belief.sigma,
                "pocd_alpha_after": posterior_belief.alpha,
                "pocd_beta_after": posterior_belief.beta,
                "stationarity_after": posterior_belief.stationarity,
                "current_agrees_with_prior": current_agrees_with_prior,
                "prior_covered_by_current": prior_covered_by_current,
                "prior_supported_by_current_points": prior_supported_count,
                "prior_support_ratio": prior_support_ratio,
                "prior_free_space_conflict": free_ratio,
                "decision": decision,
            }
        )

        if len(final_points) == 0:
            continue
        final_points = voxel_downsample(final_points, args.object_voxel_size_m)
        stats = point_stats(final_points)
        object_file = objects_dir / f"{label_id:03d}_{safe_name(name)}.ply"
        write_binary_ply(object_file, final_points)
        if len(final_points) > 0:
            selected_points.append(final_points)
        out_row = output_row_from_stats(
            meta,
            stats,
            str(object_file.relative_to(args.out_dir)),
            decision,
        )
        output_rows.append(out_row)
        memory_objects.append(memory_object_from_stats(out_row, stats, decision, posterior_belief))

    map_points = (
        np.concatenate(selected_points, axis=0)
        if selected_points
        else np.empty((0, 6), dtype=np.float32)
    )
    map_points = voxel_downsample(map_points, args.map_voxel_size_m)
    write_binary_ply(args.out_dir / "map.ply", map_points)

    fieldnames = [
        "label_id",
        "name",
        "class_id",
        "class_name",
        "panoptic_id",
        "mesh_id",
        "size",
        "points",
        "bbox_min_x",
        "bbox_min_y",
        "bbox_min_z",
        "bbox_max_x",
        "bbox_max_y",
        "bbox_max_z",
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "extent_x",
        "extent_y",
        "extent_z",
        "points_file",
        "session_state",
    ]
    write_csv(args.out_dir / "object_summary.csv", output_rows, fieldnames)
    write_csv(
        args.out_dir / "association_audit.csv",
        association_audit,
        [
            "label_id",
            "name",
            "class_id",
            "class_name",
            "panoptic_id",
            "a_points",
            "b_points",
            "current_agrees_with_prior",
            "prior_covered_by_current",
            "prior_supported_by_current_points",
            "prior_support_ratio",
            "pocd_delta_m",
            "pocd_delta_source",
            "pocd_tsdf_overlap_voxels",
            "pocd_inlier_weight",
            "pocd_outlier_weight",
            "stationarity_before",
            "stationarity_after",
            "decision",
        ],
    )
    write_csv(
        args.out_dir / "visibility_audit.csv",
        visibility_audit,
        [
            "label_id",
            "name",
            "class_id",
            "class_name",
            "panoptic_id",
            "a_points",
            "b_points",
            "b_free_space_fraction_of_a",
            "free_space_cells",
            "pocd_delta_m",
            "pocd_delta_source",
            "stationarity_after",
            "prior_supported_by_current_points",
            "prior_support_ratio",
            "decision",
        ],
    )
    write_csv(
        args.out_dir / "object_beliefs.csv",
        object_beliefs,
        [
            "label_id",
            "name",
            "class_id",
            "class_name",
            "panoptic_id",
            "prior_present",
            "current_present",
            "pocd_mu_before",
            "pocd_sigma_before",
            "pocd_alpha_before",
            "pocd_beta_before",
            "stationarity_before",
            "pocd_delta_m",
            "pocd_delta_source",
            "pocd_tsdf_overlap_voxels",
            "semantic_stationary",
            "adaptive_k",
            "pocd_inlier_weight",
            "pocd_outlier_weight",
            "pocd_inlier_mean",
            "pocd_inlier_sigma",
            "pocd_log_likelihood_inlier",
            "pocd_log_likelihood_outlier",
            "pocd_mu_after",
            "pocd_sigma_after",
            "pocd_alpha_after",
            "pocd_beta_after",
            "stationarity_after",
            "current_agrees_with_prior",
            "prior_covered_by_current",
            "prior_supported_by_current_points",
            "prior_support_ratio",
            "prior_free_space_conflict",
            "decision",
        ],
    )

    output_map = {
        "format": "base1_flat_memory_update_map_v1",
        "map_type": "object_point_map",
        "a_map_dir": str(args.a_map_dir),
        "b_map_dir": str(args.b_map_dir),
        "map_ply": "map.ply",
        "objects_dir": "objects",
        "config": {
            "object_voxel_size_m": args.object_voxel_size_m,
            "map_voxel_size_m": args.map_voxel_size_m,
            "association_distance_m": args.association_distance_m,
            "stable_current_agreement": args.stable_current_agreement,
            "free_space_cell_size_m": args.free_space_cell_size_m,
            "free_space_conflict_ratio": args.free_space_conflict_ratio,
            "ray_stride": args.ray_stride,
            "max_association_points": args.max_association_points,
            "max_visibility_points": args.max_visibility_points,
            "pocd_v0": args.pocd_v0,
            "pocd_theta_stat": args.pocd_theta_stat,
            "pocd_delta_max_m": args.pocd_delta_max_m,
            "pocd_tau_m": args.pocd_tau_m,
            "pocd_lambda_diff": args.pocd_lambda_diff,
            "pocd_tsdf_voxel_size_m": args.pocd_tsdf_voxel_size_m,
            "pocd_max_tsdf_points": args.pocd_max_tsdf_points,
        },
        "summary": {
            "a_objects": len(a_objects),
            "b_objects": len(b_objects),
            "output_objects": len(memory_objects),
            "map_points": int(len(map_points)),
            "free_space_cells": len(free_cells),
            "decision_counts": dict(decision_counts),
        },
        "objects": memory_objects,
    }
    (args.out_dir / "map.base1map.json").write_text(
        json.dumps(output_map, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "object_memory.json").write_text(
        json.dumps(
            {
                "format": "base1_flat_object_memory_v1",
                "source_map": "map.base1map.json",
                "a_map_dir": str(args.a_map_dir),
                "b_map_dir": str(args.b_map_dir),
                "objects": memory_objects,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (args.out_dir / "object_beliefs.json").write_text(
        json.dumps(object_beliefs, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "evidence_summary.json").write_text(
        json.dumps(
            {
                "method": "base1_flat_memory_update_builder",
                "inputs": {
                    "a_map_dir": str(args.a_map_dir),
                    "b_map_dir": str(args.b_map_dir),
                },
                "outputs": {
                    "map": str(args.out_dir / "map.base1map.json"),
                    "map_ply": str(args.out_dir / "map.ply"),
                    "object_memory": str(args.out_dir / "object_memory.json"),
                    "object_summary": str(args.out_dir / "object_summary.csv"),
                    "association_audit": str(args.out_dir / "association_audit.csv"),
                    "visibility_audit": str(args.out_dir / "visibility_audit.csv"),
                    "object_beliefs": str(args.out_dir / "object_beliefs.csv"),
                },
                "summary": output_map["summary"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(f"WROTE {args.out_dir / 'map.base1map.json'}")
    print(f"WROTE {args.out_dir / 'map.ply'} points={len(map_points)}")
    print(f"WROTE {args.out_dir / 'object_summary.csv'} objects={len(memory_objects)}")
    print(f"DECISIONS {dict(decision_counts)}")


if __name__ == "__main__":
    main()
