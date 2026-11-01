"""Our session node with the live-track recorder of the 09-17 synthetic evaluation (Synthetic_ABC/scripts/
prepare_native_observers.py, unchanged recipe): app/session_khronos_node.cpp of WORK with native_eval_observer.h's
synthetic_eval::record(data, tracks) after each processed frame and synthetic_eval::terminal() at the end, compiled with
the session_khronos_node flags of BUILD; plus copies of run_session.sh / run_khronos_session_strict.sh that start it.
The mapping is unchanged; with SYN_EVAL_OUT set, each frame's tracks go to SYN_EVAL_OUT/live_tracks.jsonl.
usage: build_ours_observer.py WORK BUILD OUT_DIR"""
import shlex, subprocess, sys
from pathlib import Path
WORK, BUILD, OUT = (Path(a).resolve() for a in sys.argv[1:4])
HEADER = Path('/mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts/native_eval_observer.h')
OUT.mkdir(parents=True, exist_ok=True)
text = f'#include "{HEADER}"\n' + (WORK / 'app/session_khronos_node.cpp').read_text()
needle = '[processed_pub](const auto&, const auto& data, const auto&) {'
assert text.count(needle) == 1
text = text.replace(needle, '[processed_pub](const auto&, const auto& data, const auto& tracks) {\n        synthetic_eval::record(data,tracks);')
end = 'manager.finishMappingAndSaveCallback(nullptr, nullptr);'
assert text.count(end) == 1
text = text.replace(end, end + '\n        synthetic_eval::terminal();')
cpp = OUT / 'ours_observer.cpp'; cpp.write_text(text)
cm = BUILD / 'baseline/CMakeFiles/session_khronos_node.dir'; flags = {}
for line in (cm / 'flags.make').read_text().splitlines():
    if line.startswith('CXX_') and ' = ' in line:
        k, v = line.split(' = ', 1); flags[k] = shlex.split(v)
obj = OUT / 'ours_observer.o'
subprocess.run(['/usr/bin/c++', *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'], '-O2', '-std=c++17', '-c', str(cpp), '-o', str(obj)], check=True)
link = shlex.split((cm / 'link.txt').read_text()); first = next(i for i, x in enumerate(link) if x.endswith('.cpp.o')); link[first] = str(obj)
link[link.index('-o') + 1] = str(OUT / 'ours_observer')
subprocess.run(link, cwd=BUILD / 'baseline', check=True)
root = 'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"'
strict = (WORK / 'scripts/run_khronos_session_strict.sh').read_text()
assert strict.count(root) == 1 and strict.count('"${BASE1_BUILD_DIR}/session_khronos_node"') == 1
strict = strict.replace(root, f'ROOT="{WORK}"').replace('"${BASE1_BUILD_DIR}/session_khronos_node"', f'"{OUT}/ours_observer"')
(OUT / 'run_ours_strict.sh').write_text(strict); (OUT / 'run_ours_strict.sh').chmod(0o755)
session = (WORK / 'scripts/run_session.sh').read_text()
runner = 'STRICT_RUNNER="${ROOT}/scripts/run_khronos_session_strict.sh"'
assert session.count(root) == 1 and session.count(runner) == 1
session = session.replace(root, f'ROOT="{WORK}"').replace(runner, f'STRICT_RUNNER="{OUT}/run_ours_strict.sh"')
(OUT / 'run_ours_session.sh').write_text(session); (OUT / 'run_ours_session.sh').chmod(0o755)
print('BUILT', OUT / 'ours_observer')
