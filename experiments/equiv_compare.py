"""Compare a run of the cleaned tree with the run of the old tree on the same input (CPU, reads logs and reports only).
原理思想：删死代码只做常量折叠和删除不可达分支，行为应当不变。GPU 光栅化的反向有原子加，同一份代码重跑也不逐位相同（10-14 记录：
150 帧合成链两次 3,193,395 vs 3,193,551 行），所以比较的是：每会话帧数、层的退役计数、每轮打印的元素数与 F3E 剪枝/压力诊断行
（行数、免疫数、提议数）、峰值显存、关键帧数，以及精修的采样序列哈希（Python RNG 的消耗只取决于代码路径，必须完全相同）和
16 个视角的 MAE。两边逐项列出，差异在上面那个噪声量级之内算等价。
F3E 两行打印在清理树 a13bfc2 里删了（监工 10:53），新侧没有这两行；改由层的决策日志补上同一层面的对照：每会话 layer_log.txt
的记录类型计数、决策记录（类型 + inst）的多重集、学到的传感器统计（sensor_statistics.txt）和 D1 轨迹数。关键帧两边的日志都
没有记；它只由相机位姿决定（C1/C2：平移 > 1.25 单位或任一欧拉角 > 50 度），同一输入两边相同。
usage: equiv_compare.py OLD_RUN_DIR NEW_RUN_DIR
"""
import glob
import json
import re
import sys
from pathlib import Path

OLD, NEW = Path(sys.argv[1]), Path(sys.argv[2])


def runjsons(d):
    out = {}
    for p in sorted(glob.glob(str(d / 'map' / 'session_*' / 'run.json'))):
        j = json.load(open(p))
        out[Path(p).parent.name] = {k: j.get(k) for k in ('frames', 'retired_by_layer', 'peak_gpu_gb', 'maxrss_gb', 'seconds')}
    return out


def diag(d):
    txt = (d / 'run.log').read_text(errors='ignore') if (d / 'run.log').exists() else ''
    rounds = re.findall(r'(\w+_\w) t=\s*([\d.]+)s frames=(\d+)/(\d+) elements=(\d+) retired=(\{[^}]*\})', txt)
    prune = re.findall(r'F3E_PRUNE call (\d+) rows=(\d+) immune=(\d+) contradicted=(\d+) proposed=(\d+)', txt)
    pressure = re.findall(r'F3E_PRESSURE call (\d+) rows=(\d+) immune_before=(\d+) released_now=(\d+)', txt)
    return rounds, prune, pressure


DECISIONS = ('INHERITED_CLOSE', 'INGEST_DECIDE', 'FINALIZE', 'SAME_STATE', 'MATERIALIZE_POSTERIOR', 'RETIRE', 'SEGMENT_DROPPED',
             'ABSENCE_UNSCORED')


def layer(d):
    out = {}
    for p in sorted(glob.glob(str(d / 'map' / 'session_*' / 'layer_log.txt'))):
        sd = Path(p).parent
        kinds, decisions = {}, []
        for line in open(p, errors='ignore'):
            kind = next((t for t in line.split() if re.fullmatch(r'[A-Z][A-Z_]+', t)), None)
            if kind is None:
                continue
            kinds[kind] = kinds.get(kind, 0) + 1
            if kind in DECISIONS:
                inst = next((t for t in line.split() if t.startswith('inst=')), '')
                decisions.append(f'{kind} {inst}')
        st = sd / 'sensor_statistics.txt'
        tr = sd / 'live_tracks.jsonl'
        out[sd.name] = dict(kinds=kinds, decisions=sorted(decisions), stats=st.read_text().split() if st.exists() else None,
                            tracks=sum(1 for _ in open(tr)) if tr.exists() else None)
    return out


def screen(d):
    js = glob.glob(str(d / 'short*_*.json'))
    if not js:
        return None
    j = json.load(open(js[0]))
    after = {k: round(v['photo_MAE_255'], 3) for k, v in j['after'].items() if k != 'alive_object_rows'}
    return dict(sampling_sha256=j.get('sampling_sha256'), seconds=round(j.get('seconds', 0)), iterations=j.get('iterations'),
                peak_allocated_GB=round(j.get('peak_allocated_GB', 0), 2), alive_object_rows=j['after'].get('alive_object_rows'),
                after=after, sh=j.get('active_sh_degree_at_end'))


print(f'old: {OLD}\nnew: {NEW}')
so, sn = screen(OLD), screen(NEW)
if so or sn:
    print('refinement:')
    for k in ('sampling_sha256', 'iterations', 'seconds', 'peak_allocated_GB', 'sh', 'alive_object_rows'):
        print(f'  {k:18s} old {so.get(k) if so else None}  new {sn.get(k) if sn else None}  {"SAME" if so and sn and so.get(k) == sn.get(k) else ""}')
    if so and sn:
        for v in so['after']:
            a, b = so['after'][v], sn['after'].get(v)
            print(f'  MAE {v:6s} old {a:7.3f} new {b:7.3f} diff {b - a:+.3f}')
else:
    ro, rn = runjsons(OLD), runjsons(NEW)
    for s in sorted(set(ro) | set(rn)):
        print(f'{s}: old {ro.get(s)}\n{"":{len(s)}}  new {rn.get(s)}')
    (r_o, p_o, q_o), (r_n, p_n, q_n) = diag(OLD), diag(NEW)
    print(f'rounds old {len(r_o)} new {len(r_n)}')
    for a, b in zip(r_o, r_n):
        print(f'  {a[0]} t={a[1]:>7s} frames={a[2]}/{a[3]} elements old {a[4]} new {b[4]} ({int(b[4]) - int(a[4]):+d}) retired old {a[5]} new {b[5]}')
    lo, ln = layer(OLD), layer(NEW)
    for s in sorted(set(lo) | set(ln)):
        a, b = lo.get(s), ln.get(s)
        if not a or not b:
            print(f'{s} layer log: old {bool(a)} new {bool(b)}')
            continue
        print(f'{s} layer log record counts (old / new):')
        for k in sorted(set(a['kinds']) | set(b['kinds'])):
            x, y = a['kinds'].get(k, 0), b['kinds'].get(k, 0)
            print(f'  {k:24s} {x:6d} / {y:6d} {"SAME" if x == y else "DIFF"}')
        same = a['decisions'] == b['decisions']
        print(f'  decision records (type + inst) multiset: {"SAME" if same else "DIFF"} ({len(a["decisions"])} / {len(b["decisions"])})')
        if not same:
            from collections import Counter
            ca, cb = Counter(a['decisions']), Counter(b['decisions'])
            print('    only old:', dict(ca - cb))
            print('    only new:', dict(cb - ca))
        print(f'  sensor statistics old {a["stats"]}\n                     new {b["stats"]}')
        print(f'  D1 tracks old {a["tracks"]} new {b["tracks"]}')
    print(f'F3E_PRUNE lines old {len(p_o)} new {len(p_n)} (the prints are removed in the cleaned tree since a13bfc2)')
    for a, b in list(zip(p_o, p_n))[:12]:
        print(f'  call {a[0]}: rows {a[1]} / {b[1]} immune {a[2]} / {b[2]} contradicted {a[3]} / {b[3]} proposed {a[4]} / {b[4]}')
print('EQUIV_COMPARE_DONE')
