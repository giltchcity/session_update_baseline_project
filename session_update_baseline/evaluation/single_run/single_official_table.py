"""The single-run table in the TESSE table's metrics (user 11-01): background at 5 cm (official mesh evaluation, the
config's 0.05 threshold; ours with the background, the objects and the historical objects as TESSE), static objects,
dynamic objects and changes (official 4D, tables.py), each Pre / Rec / F1, and the final map's G1 (F1 at 5 cm).
Ours: eval/single_official/ours_{real_S_fix1101,syn_S}_mv(_bghist); Khronos: eval/single_official/khronos_{real,syn}_S.
usage: single_official_table.py"""
import glob, json, subprocess, sys
from pathlib import Path
R = Path('/home/jixian/Desktop/FT/wt_takeover_runs'); E = R / 'eval/single_official'
PY = '/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python'


def row(text, label):
    line = text.strip().splitlines()[-1]   # tables.py: the label, then each value in a 20-wide column (a longer label
    assert line.startswith(label), (line, label)   # runs into the first value)
    return line[len(label):].split()


def tables(d):
    return row((E / d / 'TABLES_4D.txt').read_text(), d)


def background(d):
    return row(subprocess.run([PY, str(R / 'analysis/official_tables_at.py'), '0.05', str(E / d)], capture_output=True, text=True, check=True).stdout, d)


def g1(pattern):
    return 100 * json.loads(Path(glob.glob(str(R / pattern))[0]).read_text())['F1_05cm']


SYN_G1 = {}
for line in (R / 'eval/obj100_single_synthetic_obs/FULL_TABLE.md').read_text().splitlines():
    if line.startswith('| Mesh F1@5cm'):
        SYN_G1['a' if '| A |' in line else 'b'] = float(line.split('|')[3])
KHR_SYN_G1 = {'a': 93.64, 'b': 86.63}   # eval/obj100_khronos_clean_synthetic_5cm (the 11-01 table's clean-build rows)
for line in (R / 'eval/obj100_khronos_clean_synthetic_5cm/FULL_TABLE.md').read_text().splitlines():
    if line.startswith('| Mesh F1@5cm'):
        KHR_SYN_G1['a' if '| A |' in line else 'b'] = float(line.split('|')[3])

print('| 访问 | 方法 | 背景（5 cm）Pre / Rec / F1 | 静态物体 Pre / Rec / F1 | 动态物体 Pre / Rec / F1 | 变化 Pre / Rec / F1 | final G1 |')
print('|---|---|---|---|---|---|---|')
for kind, visits in (('real', 'abc'), ('syn', 'ab')):
    for s in visits:
        ours = f'ours_real_{s}_fix1101_mv' if kind == 'real' else f'ours_syn_{s}_mv'
        rows = [('我们', background(ours + '_bghist'), tables(ours),
                 g1(f'eval/m050_single_fix1101_{s}/real_abc/geometry/*/geometry_viewer_G1.json') if kind == 'real' else SYN_G1[s]),
                ('Khronos', background(f'khronos_{kind}_{s}'), tables(f'khronos_{kind}_{s}'),
                 g1(f'eval/khronos_pervisit_semantic_{s}/real_abc/geometry/*/geometry_viewer_G1.json') if kind == 'real' else KHR_SYN_G1[s])]
        for k, (name, bg, t, g) in enumerate(rows):
            cells = [' / '.join(bg), ' / '.join(t[3:6]), ' / '.join(t[6:9]), ' / '.join(t[9:12]), f'{g:.1f}']
            print(f"| {('真实 ' if kind == 'real' else '合成 ') + s.upper() if k == 0 else ''} | {name} | " + ' | '.join(cells) + ' |')
