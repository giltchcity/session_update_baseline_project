#!/usr/bin/env python3
"""Build a decimated OBJ with one high-resolution texture atlas.

The raw NSS OBJ uses hundreds of 2048px textures. This tool first remaps all
material UVs into one atlas, then runs MeshLab's texture-aware decimator. The
result remains a triangle mesh with real UV texture mapping, not vertex color.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image
import pymeshlab


def parse_materials(path: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    current = None
    for raw in path.read_text(errors="ignore").splitlines():
        fields = raw.strip().split(maxsplit=1)
        if not fields:
            continue
        if fields[0] == "newmtl" and len(fields) == 2:
            current = fields[1]
        elif fields[0] == "map_Kd" and len(fields) == 2 and current:
            result[current] = path.parent / fields[1]
    if not result:
        raise RuntimeError(f"No textured materials in {path}")
    return result


def collect_uvs(path: Path) -> list[tuple[float, float]]:
    result = []
    with path.open("r", errors="ignore") as stream:
        for line in stream:
            if line.startswith("vt "):
                fields = line.split()
                result.append((float(fields[1]), float(fields[2])))
    if not result:
        raise RuntimeError(f"No UV coordinates in {path}")
    return result


def build_atlas(
    materials: dict[str, Path], output: Path, tile_size: int, padding: int
) -> dict[str, tuple[int, int]]:
    names = sorted(materials)
    columns = math.ceil(math.sqrt(len(names)))
    rows = math.ceil(len(names) / columns)
    width, height = columns * tile_size, rows * tile_size
    atlas = Image.new("RGB", (width, height), (128, 128, 128))
    tile_map = {}
    inner = tile_size - 2 * padding
    for index, name in enumerate(names):
        column, row = index % columns, index // columns
        with Image.open(materials[name]) as image:
            image = image.convert("RGB").resize((inner, inner), Image.Resampling.LANCZOS)
            x, y = column * tile_size + padding, row * tile_size + padding
            atlas.paste(image, (x, y))
            # Duplicate edge pixels into the padding to reduce UV seam bleeding.
            if padding:
                atlas.paste(image.crop((0, 0, inner, 1)).resize((inner, padding)), (x, y - padding))
                atlas.paste(
                    image.crop((0, inner - 1, inner, inner)).resize((inner, padding)),
                    (x, y + inner),
                )
                atlas.paste(image.crop((0, 0, 1, inner)).resize((padding, inner)), (x - padding, y))
                atlas.paste(
                    image.crop((inner - 1, 0, inner, inner)).resize((padding, inner)),
                    (x + inner, y),
                )
        tile_map[name] = (column, row)
    output.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(output, quality=95, subsampling=0, optimize=True)
    print(f"SAVED_ATLAS {output} size={width}x{height} materials={len(names)}", flush=True)
    return tile_map


def remap_uv(
    uv: tuple[float, float], tile: tuple[int, int], tile_size: int, padding: int,
    atlas_width: int, atlas_height: int,
) -> tuple[float, float]:
    u, v = uv
    column, row = tile
    inner = tile_size - 2 * padding
    x = column * tile_size + padding + np.clip(u, 0.0, 1.0) * (inner - 1)
    y_from_top = row * tile_size + padding + (1.0 - np.clip(v, 0.0, 1.0)) * (inner - 1)
    return x / atlas_width, 1.0 - y_from_top / atlas_height


def rewrite_obj_to_atlas(
    source: Path,
    output: Path,
    materials: dict[str, Path],
    tiles: dict[str, tuple[int, int]],
    tile_size: int,
    padding: int,
) -> None:
    old_uvs = collect_uvs(source)
    columns = math.ceil(math.sqrt(len(materials)))
    rows = math.ceil(len(materials) / columns)
    atlas_width, atlas_height = columns * tile_size, rows * tile_size
    uv_map: dict[tuple[int, str], int] = {}
    new_uvs: list[tuple[float, float]] = []

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=output.parent) as face_stream:
        face_path = Path(face_stream.name)
        current_material = None
        with source.open("r", errors="ignore") as stream:
            for line in stream:
                if line.startswith("usemtl "):
                    current_material = line.strip().split(maxsplit=1)[1]
                    if current_material not in materials:
                        raise RuntimeError(f"Unknown material {current_material}")
                elif line.startswith("f "):
                    if current_material is None:
                        raise RuntimeError("Face appeared before usemtl")
                    remapped = []
                    for token in line.split()[1:]:
                        fields = token.split("/")
                        if len(fields) < 2 or not fields[1]:
                            raise RuntimeError("Face has no texture coordinate")
                        old_index = int(fields[1])
                        if old_index < 0:
                            old_index = len(old_uvs) + old_index + 1
                        key = (old_index, current_material)
                        if key not in uv_map:
                            uv_map[key] = len(new_uvs) + 1
                            new_uvs.append(
                                remap_uv(
                                    old_uvs[old_index - 1],
                                    tiles[current_material],
                                    tile_size,
                                    padding,
                                    atlas_width,
                                    atlas_height,
                                )
                            )
                        fields[1] = str(uv_map[key])
                        remapped.append("/".join(fields))
                    face_stream.write("f " + " ".join(remapped) + "\n")

    mtl_name = output.name + ".mtl"
    with output.open("w") as out:
        out.write(f"mtllib {mtl_name}\n")
        with source.open("r", errors="ignore") as stream:
            for line in stream:
                if line.startswith("v ") or line.startswith("vn "):
                    out.write(line)
        for u, v in new_uvs:
            out.write(f"vt {u:.10f} {v:.10f}\n")
        out.write("usemtl atlas\n")
        with face_path.open("r") as faces:
            shutil.copyfileobj(faces, out)
    face_path.unlink()
    (output.parent / mtl_name).write_text(
        "newmtl atlas\nKa 0 0 0\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd atlas.jpg\n"
    )
    print(f"SAVED_ATLAS_OBJ {output} uvs={len(new_uvs)}", flush=True)


def decimate(source: Path, output: Path, target_faces: int) -> None:
    mesh_set = pymeshlab.MeshSet()
    mesh_set.load_new_mesh(str(source))
    mesh_set.apply_filter(
        "meshing_decimation_quadric_edge_collapse_with_texture",
        targetfacenum=target_faces,
        preserveboundary=True,
        preservenormal=True,
        optimalplacement=True,
    )
    mesh = mesh_set.current_mesh()
    print(
        f"DECIMATED vertices={mesh.vertex_number()} faces={mesh.face_number()} "
        f"wedge_uv={mesh.has_wedge_tex_coord()}",
        flush=True,
    )
    mesh_set.save_current_mesh(str(output), save_textures=False, save_wedge_texcoord=True)
    output_mtl = output.parent / (output.name + ".mtl")
    output_mtl.write_text(
        "newmtl material_0\nKa 0 0 0\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd atlas.jpg\n"
    )
    print(f"SAVED_DECIMATED_OBJ {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--target-faces", type=int, default=150000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    proxy = args.output_dir / "proxy.obj"
    if proxy.exists() and not args.force:
        print(f"CACHE_EXISTS {proxy}")
        return
    mtl = args.source.with_suffix(".mtl")
    materials = parse_materials(mtl)
    atlas_path = args.output_dir / "atlas.jpg"
    tiles = build_atlas(materials, atlas_path, args.tile_size, args.padding)
    atlas_source = args.output_dir / "atlas_source.obj"
    rewrite_obj_to_atlas(
        args.source, atlas_source, materials, tiles, args.tile_size, args.padding
    )
    decimate(atlas_source, proxy, args.target_faces)


if __name__ == "__main__":
    main()
