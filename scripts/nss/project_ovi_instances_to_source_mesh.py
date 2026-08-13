#!/usr/bin/env python3
"""Project OVI-MAP instance colors from its raw mesh back to an NSS source OBJ."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-obj", required=True, type=Path)
    parser.add_argument("--ovi-mesh", required=True, type=Path)
    parser.add_argument("--inst-sem", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sample-stride", type=int, default=32)
    parser.add_argument("--max-dist", type=float, default=0.12)
    parser.add_argument("--min-component-faces", type=int, default=150)
    return parser.parse_args()


def read_obj_geometry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z, *_ = line.split()
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                refs = line.split()[1:]
                ids: list[int] = []
                for ref in refs:
                    raw = ref.split("/", 1)[0]
                    idx = int(raw)
                    if idx < 0:
                        idx = len(vertices) + idx + 1
                    ids.append(idx - 1)
                for i in range(1, len(ids) - 1):
                    faces.append((ids[0], ids[i], ids[i + 1]))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def load_color_to_instance(path: Path) -> dict[tuple[int, int, int], int]:
    with path.open("rb") as f:
        inst_sem = pickle.load(f)
    color_to_instance: dict[tuple[int, int, int], int] = {}
    for inst_id, info in inst_sem.items():
        color = tuple(int(x) for x in np.asarray(info["color"]).reshape(-1)[:3])
        color_to_instance[color] = int(inst_id)
    return color_to_instance


def read_ply_header(f) -> tuple[list[str], int, int]:
    header: list[str] = []
    vertex_count = face_count = None
    while True:
        line = f.readline()
        if not line:
            raise RuntimeError("Unexpected EOF while reading PLY header")
        header.append(line)
        if line.startswith("element vertex "):
            vertex_count = int(line.split()[-1])
        elif line.startswith("element face "):
            face_count = int(line.split()[-1])
        elif line.strip() == "end_header":
            break
    if vertex_count is None or face_count is None:
        raise RuntimeError("PLY header is missing vertex or face count")
    return header, vertex_count, face_count


def sample_ovi_vertices(
    path: Path, color_to_instance: dict[tuple[int, int, int], int], stride: int
) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float, float]] = []
    labels: list[int] = []
    colors: list[tuple[int, int, int]] = []
    with path.open("r", buffering=1024 * 1024) as f:
        _, vertex_count, face_count = read_ply_header(f)
        if vertex_count != face_count * 3:
            raise RuntimeError(
                f"Expected OVI PLY to store three unique vertices per face, "
                f"got {vertex_count=} and {face_count=}."
            )
        for face_id in range(face_count):
            lines = [f.readline(), f.readline(), f.readline()]
            if face_id % stride != 0:
                continue
            for line in lines:
                parts = line.split()
                color = (int(parts[6]), int(parts[7]), int(parts[8]))
                inst_id = color_to_instance.get(color, 0)
                if inst_id == 0:
                    continue
                points.append((float(parts[0]), float(parts[1]), float(parts[2])))
                labels.append(inst_id)
                colors.append(color)
    if not points:
        raise RuntimeError("No labeled OVI vertices were sampled")
    return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.uint16)


def build_instance_colors(labels: np.ndarray) -> dict[int, tuple[int, int, int]]:
    unique = sorted(int(x) for x in np.unique(labels) if int(x) != 0)
    rng = np.random.default_rng(20260707)
    return {
        inst_id: tuple(int(v) for v in rng.integers(35, 256, size=3))
        for inst_id in unique
    }


def labels_to_colors(labels: np.ndarray, color_map: dict[int, tuple[int, int, int]]) -> np.ndarray:
    colors = np.full((labels.shape[0], 3), 190, dtype=np.uint8)
    for inst_id, color in color_map.items():
        colors[labels == inst_id] = np.asarray(color, dtype=np.uint8)
    colors[labels == 0] = np.asarray([90, 90, 90], dtype=np.uint8)
    return colors


def face_majority_labels(vertex_labels: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertex_labels[faces]
    out = np.zeros((faces.shape[0],), dtype=np.uint16)
    for i, vals in enumerate(tri):
        nonzero = [int(v) for v in vals if int(v) != 0]
        if not nonzero:
            continue
        out[i] = Counter(nonzero).most_common(1)[0][0]
    return out


def face_neighbors(faces: np.ndarray) -> list[list[int]]:
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_id, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            edge_to_faces[(int(u), int(v))].append(face_id)
    neighbors: list[list[int]] = [[] for _ in range(faces.shape[0])]
    for face_ids in edge_to_faces.values():
        if len(face_ids) < 2:
            continue
        for face_id in face_ids:
            neighbors[face_id].extend(other for other in face_ids if other != face_id)
    return neighbors


def filter_small_components(
    face_labels: np.ndarray, neighbors: list[list[int]], min_faces: int
) -> tuple[np.ndarray, dict[str, int]]:
    filtered = face_labels.copy()
    visited = np.zeros(face_labels.shape[0], dtype=bool)
    removed_components = 0
    removed_faces = 0
    for start in range(face_labels.shape[0]):
        label = int(face_labels[start])
        if label == 0 or visited[start]:
            continue
        queue = deque([start])
        visited[start] = True
        comp: list[int] = []
        border_labels: list[int] = []
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nb in neighbors[cur]:
                nb_label = int(face_labels[nb])
                if nb_label == label and not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
                elif nb_label != label and nb_label != 0:
                    border_labels.append(nb_label)
        if len(comp) >= min_faces:
            continue
        removed_components += 1
        removed_faces += len(comp)
        replacement = 0
        if border_labels:
            replacement = Counter(border_labels).most_common(1)[0][0]
        filtered[np.asarray(comp, dtype=np.int32)] = replacement
    return filtered, {
        "removed_components": removed_components,
        "removed_faces": removed_faces,
    }


def vertex_labels_from_faces(face_labels: np.ndarray, faces: np.ndarray, num_vertices: int) -> np.ndarray:
    votes: list[Counter[int]] = [Counter() for _ in range(num_vertices)]
    for label, tri in zip(face_labels, faces):
        label_i = int(label)
        if label_i == 0:
            continue
        for v in tri:
            votes[int(v)][label_i] += 1
    out = np.zeros((num_vertices,), dtype=np.uint16)
    for i, counter in enumerate(votes):
        if counter:
            out[i] = counter.most_common(1)[0][0]
    return out


def write_labeled_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, labels: np.ndarray, colors: np.ndarray) -> None:
    vertex = np.empty(
        vertices.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("label", "u2"),
        ],
    )
    vertex["x"] = vertices[:, 0]
    vertex["y"] = vertices[:, 1]
    vertex["z"] = vertices[:, 2]
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]
    vertex["label"] = labels
    face = np.empty(faces.shape[0], dtype=[("vertex_indices", "O")])
    face["vertex_indices"] = [row for row in faces.astype(np.int32)]
    PlyData(
        [PlyElement.describe(vertex, "vertex"), PlyElement.describe(face, "face")],
        text=False,
    ).write(path)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_vertices, source_faces = read_obj_geometry(args.source_obj)
    color_to_instance = load_color_to_instance(args.inst_sem)
    sample_points, sample_labels = sample_ovi_vertices(
        args.ovi_mesh, color_to_instance, args.sample_stride
    )

    tree = cKDTree(sample_points)
    dists, indices = tree.query(source_vertices, k=1, workers=-1)
    raw_vertex_labels = sample_labels[indices]
    raw_vertex_labels[dists > args.max_dist] = 0

    instance_colors = build_instance_colors(sample_labels)
    raw_face_labels = face_majority_labels(raw_vertex_labels, source_faces)
    neighbors = face_neighbors(source_faces)
    filtered_face_labels, filter_stats = filter_small_components(
        raw_face_labels, neighbors, args.min_component_faces
    )
    filtered_vertex_labels = vertex_labels_from_faces(
        filtered_face_labels, source_faces, source_vertices.shape[0]
    )

    raw_colors = labels_to_colors(raw_vertex_labels, instance_colors)
    filtered_colors = labels_to_colors(filtered_vertex_labels, instance_colors)

    raw_out = args.out_dir / "source_mesh_ovi_instance_projected_raw.ply"
    filtered_out = args.out_dir / "source_mesh_ovi_instance_projected_filtered.ply"
    write_labeled_ply(raw_out, source_vertices, source_faces, raw_vertex_labels, raw_colors)
    write_labeled_ply(
        filtered_out, source_vertices, source_faces, filtered_vertex_labels, filtered_colors
    )

    stats = {
        "source_vertices": int(source_vertices.shape[0]),
        "source_faces": int(source_faces.shape[0]),
        "sample_points": int(sample_points.shape[0]),
        "sample_stride": int(args.sample_stride),
        "max_dist": float(args.max_dist),
        "assigned_vertices": int(np.count_nonzero(raw_vertex_labels)),
        "assigned_vertex_fraction": float(np.count_nonzero(raw_vertex_labels) / raw_vertex_labels.shape[0]),
        "raw_instances": int(len(np.unique(raw_vertex_labels[raw_vertex_labels != 0]))),
        "filtered_instances": int(len(np.unique(filtered_vertex_labels[filtered_vertex_labels != 0]))),
        "distance_percentiles": {
            str(p): float(np.percentile(dists, p)) for p in (50, 75, 90, 95, 99)
        },
        **filter_stats,
        "raw_output": str(raw_out),
        "filtered_output": str(filtered_out),
    }
    with (args.out_dir / "projection_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
