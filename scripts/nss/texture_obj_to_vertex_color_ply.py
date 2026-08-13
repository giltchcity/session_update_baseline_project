#!/usr/bin/env python3
"""Convert a textured OBJ into a binary PLY with sampled vertex colors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--mtl", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def read_mtl(path: Path) -> dict[str, Path]:
    material_to_texture: dict[str, Path] = {}
    current: str | None = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("newmtl "):
                current = line.split(maxsplit=1)[1]
            elif current and line.startswith("map_Kd "):
                tex = line.split(maxsplit=1)[1]
                material_to_texture[current] = (path.parent / tex).resolve()
    return material_to_texture


def sample_texture(
    material: str | None,
    uv: tuple[float, float] | None,
    material_to_texture: dict[str, Path],
    texture_cache: dict[str, np.ndarray],
) -> tuple[int, int, int]:
    if not material or material not in material_to_texture or uv is None:
        return (185, 185, 185)
    tex_path = material_to_texture[material]
    key = str(tex_path)
    if key not in texture_cache:
        image = Image.open(tex_path).convert("RGB")
        texture_cache[key] = np.asarray(image, dtype=np.uint8)
    tex = texture_cache[key]
    h, w = tex.shape[:2]
    u = uv[0] % 1.0
    v = uv[1] % 1.0
    x = min(w - 1, max(0, int(round(u * (w - 1)))))
    y = min(h - 1, max(0, int(round((1.0 - v) * (h - 1)))))
    r, g, b = tex[y, x]
    return (int(r), int(g), int(b))


def main() -> None:
    args = parse_args()
    obj_path = args.obj
    mtl_path = args.mtl
    if mtl_path is None:
        mtl_path = None

    positions: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    out_positions: list[tuple[float, float, float]] = []
    out_colors: list[tuple[int, int, int]] = []
    out_faces: list[tuple[int, int, int]] = []
    texture_cache: dict[str, np.ndarray] = {}
    material_to_texture: dict[str, Path] = {}
    current_material: str | None = None

    with obj_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("mtllib ") and mtl_path is None:
                mtl_path = (obj_path.parent / line.split(maxsplit=1)[1].strip()).resolve()
                material_to_texture = read_mtl(mtl_path)
            elif line.startswith("v "):
                _, x, y, z, *_ = line.split()
                positions.append((float(x), float(y), float(z)))
            elif line.startswith("vt "):
                parts = line.split()
                uvs.append((float(parts[1]), float(parts[2])))
            elif line.startswith("usemtl "):
                current_material = line.split(maxsplit=1)[1].strip()
            elif line.startswith("f "):
                refs = line.split()[1:]
                corners: list[tuple[int, int | None]] = []
                for ref in refs:
                    parts = ref.split("/")
                    v_idx = int(parts[0])
                    if v_idx < 0:
                        v_idx = len(positions) + v_idx + 1
                    vt_idx = None
                    if len(parts) > 1 and parts[1]:
                        vt_idx = int(parts[1])
                        if vt_idx < 0:
                            vt_idx = len(uvs) + vt_idx + 1
                    corners.append((v_idx - 1, None if vt_idx is None else vt_idx - 1))

                for i in range(1, len(corners) - 1):
                    tri = [corners[0], corners[i], corners[i + 1]]
                    base = len(out_positions)
                    for pos_idx, uv_idx in tri:
                        uv = None if uv_idx is None else uvs[uv_idx]
                        out_positions.append(positions[pos_idx])
                        out_colors.append(
                            sample_texture(current_material, uv, material_to_texture, texture_cache)
                        )
                    out_faces.append((base, base + 1, base + 2))

    vertices_np = np.asarray(out_positions, dtype=np.float32)
    colors_np = np.asarray(out_colors, dtype=np.uint8)
    faces_np = np.asarray(out_faces, dtype=np.int32)

    vertex = np.empty(
        vertices_np.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"] = vertices_np[:, 0]
    vertex["y"] = vertices_np[:, 1]
    vertex["z"] = vertices_np[:, 2]
    vertex["red"] = colors_np[:, 0]
    vertex["green"] = colors_np[:, 1]
    vertex["blue"] = colors_np[:, 2]

    face = np.empty(faces_np.shape[0], dtype=[("vertex_indices", "O")])
    face["vertex_indices"] = [row for row in faces_np]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(vertex, "vertex"), PlyElement.describe(face, "face")],
        text=False,
    ).write(args.out)
    print("wrote", args.out)
    print("vertices", vertices_np.shape[0], "faces", faces_np.shape[0], "textures", len(texture_cache))


if __name__ == "__main__":
    main()
