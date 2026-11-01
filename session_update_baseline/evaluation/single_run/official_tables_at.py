"""The official tables.py 4D averages of the background at another of the config's thresholds (user 11-01: 5 cm;
the paper reports 0.2 m): tables.py unchanged, its metric list set to Accuracy / Completeness / F1 at THRESHOLD.
usage: official_tables_at.py THRESHOLD EVAL_DIR [EVAL_DIR ...]"""
import os, sys
PLOTTING = '/home/jixian/Desktop/FT/build_artifacts/official_khronos_63faadde_abc/src/khronos_eval/plotting'
t, dirs = sys.argv[1], [os.path.abspath(d) for d in sys.argv[2:]]
os.chdir(PLOTTING); sys.path.insert(0, PLOTTING)
import tables  # noqa: E402
tables.METRICS = [f'Accuracy@{t}', f'Completeness@{t}', f'F1@{t}']
for d in dirs:
    sys.argv = ['tables.py', os.path.dirname(d), '-m', f'{os.path.basename(d)}:{os.path.basename(d)}']
    tables.main()
