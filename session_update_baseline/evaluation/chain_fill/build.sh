#!/usr/bin/env bash
# Builds chain_fill, timeline_export and timeline_to_4dmap (or the ones named) against the clean 290e110 build's khronos
# (the flags and link line of its export_4dmap_mesh_ply).
# usage: build.sh [TOOL ...]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
target=/home/jixian/Desktop/FT/wt_290e110_build/baseline/CMakeFiles/export_4dmap_mesh_ply.dir
python3 - "$here" "$target" "${@:-chain_fill timeline_export timeline_to_4dmap}" <<'PY'
import pathlib, shlex, subprocess, sys
here, target = map(pathlib.Path, sys.argv[1:3])
tools = ' '.join(sys.argv[3:]).split()
flags = {}
for line in (target / 'flags.make').read_text().splitlines():
    if line.startswith(('CXX_DEFINES =', 'CXX_INCLUDES =', 'CXX_FLAGS =')):
        name, text = line.split('=', 1); flags[name.strip()] = shlex.split(text)
link0 = shlex.split((target / 'link.txt').read_text())
for tool in tools:
    obj = here / f'{tool}.o'
    subprocess.run(['/usr/bin/c++', *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'], *flags['CXX_FLAGS'], '-c', str(here / f'{tool}.cpp'), '-o', str(obj)], check=True)
    link = list(link0)
    link[link.index('CMakeFiles/export_4dmap_mesh_ply.dir/app/export_4dmap_mesh_ply.cpp.o')] = str(obj)
    link[link.index('-o') + 1] = str(here / tool)
    subprocess.run(link, check=True, cwd=target.parent.parent)
PY
