"""historical_object_eval.cpp of datasets/synthetic_ab/evaluation_complete_20260917 with the static-object evaluation
of the real data, Flat and 3RScan (Surface, max_matching_distance 0.5 m, surface_subsampling_rate 100, semantics
required, compensate_missegmentation false; one setting for every dataset, 2026-10-29) instead of the published
synthetic harness's 2 m / every point / compensation on (eval/synthetic_object_eval_m050 changed the distance only),
built for both map readers with the recipe of /mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts/compile_historical_object_eval.py, into this
directory (the 2 m binaries in evaluation_complete_20260917 are untouched).
usage: python3 build.py
"""
from pathlib import Path
import shlex, subprocess
ROOT = Path('/home/jixian/Desktop/FT'); D = Path(__file__).resolve().parent; cpp = D / 'historical_object_eval.cpp'
for method, build, target, ev in [
        ('ours', ROOT / 'session_update_baseline_project/session_update_baseline/build_canonical', 'stream_4dmap_copy',
         ROOT / 'session_update_baseline_project/session_update_baseline/ports/mapping_core/khronos_eval'),
        ('khronos', ROOT / 'build_artifacts/official_khronos_63faadde_abc/mesh_adapter', 'official_mesh_export',
         ROOT / 'build_artifacts/official_khronos_63faadde_abc/src/khronos_eval')]:
    cm = build / f'CMakeFiles/{target}.dir'; flags = {}
    for line in (cm / 'flags.make').read_text().splitlines():
        if line.startswith('CXX_') and ' = ' in line:
            k, v = line.split(' = ', 1); flags[k] = shlex.split(v)
    link = shlex.split((cm / 'link.txt').read_text()); tail = link[link.index('-o') + 2:]
    subprocess.run(['/usr/bin/c++', *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'], '-I' + str(ev / 'include'), '-O2', '-std=c++17',
                    str(cpp), str(ev / 'src/object_evaluator.cpp'), '-o', str(D / f'{method}_historical_object_eval'), *tail],
                   cwd=build, check=True)
    print('BUILT', method, flush=True)
