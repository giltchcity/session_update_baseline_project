#!/usr/bin/env bash
# Builds the history tools here (default: hist_measure hist_convert piece_map_cut piece_map_check piece_map_plain) against a build's
# khronos (KHRONOS_BUILD, the SESSION_UPDATE_CANONICAL_BUILD parent of a build_canonical.sh build), with the flags and
# link line of its export_4dmap_mesh_ply target (as eval/chain_fill/build.sh does for the 290e110 build), plus zstd;
# the port's own headers come first. PieceHistory comes from that build's libkhronos.
# usage: [KHRONOS_BUILD=DIR] build.sh [TOOL ...]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
port="$(cd "$here/../../ports/mapping_core/khronos" && pwd)"
target=${KHRONOS_BUILD:-/home/jixian/Desktop/FT/wt_split_final_build}/baseline/CMakeFiles/export_4dmap_mesh_ply.dir
# spark_dsg was built against the system nlohmann json; json_include/nlohmann points at it (see below).
mkdir -p "$here/json_include"; [[ -e "$here/json_include/nlohmann" ]] || ln -s /usr/include/nlohmann "$here/json_include/nlohmann"
python3 - "$here" "$target" "$port" "${@:-hist_measure hist_convert piece_map_cut piece_map_check piece_map_plain}" <<'PY'
import pathlib, shlex, subprocess, sys
here, target, port = map(pathlib.Path, sys.argv[1:4])
tools = ' '.join(sys.argv[4:]).split()
flags = {}
for line in (target / 'flags.make').read_text().splitlines():
    if line.startswith(('CXX_DEFINES =', 'CXX_INCLUDES =', 'CXX_FLAGS =')):
        name, text = line.split('=', 1); flags[name.strip()] = shlex.split(text)
# spark_dsg was built against the system nlohmann json (3.11.3); the build's -isystem conda include holds 3.11.2, whose
# ABI tag does not link with spark_dsg's Metadata(json), so the system copy comes first (json_include/nlohmann -> it).
compile0 = ['/usr/bin/c++', *flags['CXX_DEFINES'], f'-I{here / "json_include"}', f'-I{port / "include"}',
            *flags['CXX_INCLUDES'], *flags['CXX_FLAGS']]
link0 = shlex.split((target / 'link.txt').read_text())
for tool in tools:
    objs = []
    for src in [here / f'{tool}.cpp']:
        obj = here / f'{src.stem}.o'
        subprocess.run([*compile0, '-c', str(src), '-o', str(obj)], check=True)
        objs.append(str(obj))
    link = list(link0)
    at = link.index('CMakeFiles/export_4dmap_mesh_ply.dir/app/export_4dmap_mesh_ply.cpp.o')
    link[at:at + 1] = objs
    link[link.index('-o') + 1] = str(here / tool)
    subprocess.run(link + ['-L/home/jixian/ros2_ws/install/spark_dsg/lib', '-lspark_dsg', '-lzstd'], check=True, cwd=target.parent.parent)
PY
