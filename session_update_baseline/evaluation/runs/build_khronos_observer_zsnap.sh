#!/usr/bin/env bash
# The synthetic Khronos observer (datasets/synthetic_ab/evaluation_complete_20260917/khronos_observer.cpp, the official
# session node with the synthetic evaluation's recording hooks) built against the zsnap build, so the synthetic runs hold
# their 4D-map snapshots compressed as every other Khronos run (user 10-31). The recipe of
# Synthetic_ABC/scripts/prepare_native_observers.py ('khronos'): the session node target's compile flags and link line,
# here from build_artifacts/official_khronos_63faadde_zsnap/adapter instead of ..._abc/adapter; the source unchanged.
# Output: build_artifacts/official_khronos_63faadde_zsnap/synthetic_observer/khronos_observer.
set -eo pipefail
FT=/home/jixian/Desktop/FT
SRC=$FT/datasets/synthetic_ab/evaluation_complete_20260917/khronos_observer.cpp
BUILD=$FT/build_artifacts/official_khronos_63faadde_zsnap/adapter
CM=$BUILD/CMakeFiles/official_khronos_session_node.dir
OUT=$FT/build_artifacts/official_khronos_63faadde_zsnap/synthetic_observer
mkdir -p $OUT
python3 - "$SRC" "$CM" "$OUT" "$BUILD" <<'PY'
import shlex, subprocess, sys
from pathlib import Path
src, cm, out, build = (Path(a) for a in sys.argv[1:])
flags = {}
for line in (cm / 'flags.make').read_text().splitlines():
    if line.startswith('CXX_') and ' = ' in line:
        k, v = line.split(' = ', 1); flags[k] = shlex.split(v)
obj = out / 'khronos_observer.o'
subprocess.run(['/usr/bin/c++', *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'], '-DSYNTHETIC_COMMON_AABB', '-O2', '-std=c++17',
                '-c', str(src), '-o', str(obj)], check=True)
link = shlex.split((cm / 'link.txt').read_text())
link[next(i for i, x in enumerate(link) if x.endswith('.cpp.o'))] = str(obj)
link[link.index('-o') + 1] = str(out / 'khronos_observer')
subprocess.run(link, cwd=build, check=True)
print('BUILT', out / 'khronos_observer')
PY
