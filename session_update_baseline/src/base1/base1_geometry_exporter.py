#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PLY_TYPES = {
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fin:
        return list(csv.DictReader(fin))


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


def load_geometry_points(path: Path) -> np.ndarray:
    fmt, vertex_count, properties, header_bytes, header_lines = parse_ply_header(path)
    names = [name for _, name in properties]
    if vertex_count == 0:
        return np.empty((0, 6), dtype=np.float32)
    for name in ("x", "y", "z"):
        if name not in names:
            raise ValueError(f"{path} missing {name}")
    has_color = {"red", "green", "blue"}.issubset(names)
    cols = [names.index(axis) for axis in ("x", "y", "z")]
    if has_color:
        cols += [names.index(axis) for axis in ("red", "green", "blue")]

    if fmt == "ascii":
        data = np.loadtxt(
            str(path),
            skiprows=header_lines,
            max_rows=vertex_count,
            usecols=cols,
            ndmin=2,
        )
        out = np.asarray(data, dtype=np.float32)
    elif fmt == "binary_little_endian":
        dtype = np.dtype([(name, PLY_TYPES[ply_type]) for ply_type, name in properties])
        with path.open("rb") as fin:
            fin.seek(header_bytes)
            data = np.fromfile(fin, dtype=dtype, count=vertex_count)
        arrays = [data["x"], data["y"], data["z"]]
        if has_color:
            arrays += [data["red"], data["green"], data["blue"]]
        out = np.column_stack(arrays).astype(np.float32, copy=False)
    else:
        raise ValueError(f"unsupported PLY format {fmt}: {path}")

    if out.shape[1] == 3:
        colors = np.full((len(out), 3), 180.0, dtype=np.float32)
        out = np.concatenate([out[:, :3], colors], axis=1)
    return out


def voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if len(points) == 0 or voxel_size_m <= 0.0:
        return points.astype(np.float32, copy=False)
    cells = np.floor(points[:, :3] / voxel_size_m).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    indices.sort()
    return points[indices].astype(np.float32, copy=False)


def write_geometry_ply(path: Path, points: np.ndarray, binary: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 6:
        raise ValueError(f"expected Nx6 xyzrgb points for {path}, got {pts.shape}")
    if not binary:
        with path.open("w") as fout:
            fout.write("ply\n")
            fout.write("format ascii 1.0\n")
            fout.write(f"element vertex {len(pts)}\n")
            fout.write("property float x\n")
            fout.write("property float y\n")
            fout.write("property float z\n")
            fout.write("property uchar red\n")
            fout.write("property uchar green\n")
            fout.write("property uchar blue\n")
            fout.write("end_header\n")
            colors = np.clip(np.rint(pts[:, 3:6]), 0, 255).astype(np.uint8)
            for point, color in zip(pts[:, :3], colors):
                fout.write(
                    f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )
        return

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    out = np.empty(
        len(pts),
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        ),
    )
    out["x"] = pts[:, 0]
    out["y"] = pts[:, 1]
    out["z"] = pts[:, 2]
    colors = np.clip(np.rint(pts[:, 3:6]), 0, 255).astype(np.uint8)
    out["red"] = colors[:, 0]
    out["green"] = colors[:, 1]
    out["blue"] = colors[:, 2]
    with path.open("wb") as fout:
        fout.write(header)
        out.tofile(fout)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Base1 final object geometry without Panoptic TSDF/mesh rematerialization."
    )
    parser.add_argument("--object-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--combined-name", default="base1_final_geometry.ply")
    parser.add_argument("--namespace-dir-name", default="namespaces")
    parser.add_argument("--voxel-size-m", type=float, default=0.0)
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Write binary PLY. Default is ASCII for compatibility with existing eval scripts.",
    )
    args = parser.parse_args()

    map_dir = args.object_summary.parent
    out_dir = args.out_dir
    namespace_dir = out_dir / args.namespace_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    namespace_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    combined_chunks: list[np.ndarray] = []
    for row in read_rows(args.object_summary):
        name = row.get("name", f"label_{row.get('label_id', '')}")
        rel_file = row.get("points_file", "")
        if not rel_file:
            continue
        src = map_dir / rel_file
        points = load_geometry_points(src)
        if args.voxel_size_m > 0.0:
            points = voxel_downsample(points, args.voxel_size_m)
        out_file = namespace_dir / f"{name}.ply"
        write_geometry_ply(out_file, points, args.binary)
        if len(points) > 0:
            combined_chunks.append(points)
        manifest_rows.append(
            {
                "label_id": row.get("label_id", ""),
                "name": name,
                "class_id": row.get("class_id", ""),
                "class_name": row.get("class_name", ""),
                "panoptic_id": row.get("panoptic_id", ""),
                "session_state": row.get("session_state", row.get("state", "")),
                "source_points_file": rel_file,
                "export_ply": str(out_file.relative_to(out_dir)),
                "points": int(len(points)),
            }
        )

    combined = (
        np.concatenate(combined_chunks, axis=0)
        if combined_chunks
        else np.empty((0, 6), dtype=np.float32)
    )
    if args.voxel_size_m > 0.0:
        combined = voxel_downsample(combined, args.voxel_size_m)
    combined_path = out_dir / args.combined_name
    write_geometry_ply(combined_path, combined, args.binary)

    manifest_path = out_dir / "base1_geometry_manifest.csv"
    with manifest_path.open("w", newline="") as fout:
        fieldnames = [
            "label_id",
            "name",
            "class_id",
            "class_name",
            "panoptic_id",
            "session_state",
            "source_points_file",
            "export_ply",
            "points",
        ]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "format": "base1_final_geometry_export_v1",
        "role": "formal_base1_geometry_no_panoptic_rematerialization",
        "object_summary": str(args.object_summary),
        "combined_ply": str(combined_path),
        "namespace_dir": str(namespace_dir),
        "manifest": str(manifest_path),
        "voxel_size_m": args.voxel_size_m,
        "ply_encoding": "binary_little_endian" if args.binary else "ascii",
        "objects": len(manifest_rows),
        "combined_points": int(len(combined)),
    }
    (out_dir / "base1_geometry_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(f"WROTE {combined_path} points={len(combined)}")
    print(f"WROTE {namespace_dir} objects={len(manifest_rows)}")
    print(f"WROTE {manifest_path}")


if __name__ == "__main__":
    main()
