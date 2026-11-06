"""The detail pages of the results table (RESULTS_DETAIL_TSDF_20261029.md): every result of the TSDF side's latest
batch, per dataset and per visit, read from the evaluation outputs (nothing typed in by hand): ours, the Khronos version
of the main table, the other Khronos versions run (continuous semantic, continuous with instances and MaxIoU, every
visit from an empty map), and Panoptic; with the metrics the main table does not use.
usage: results_detail.py OUT.md
"""
import csv, glob, json, re, sys
from pathlib import Path

R = Path('/home/jixian/Desktop/FT/wt_takeover_runs'); E = R / 'eval'; FT = Path('/home/jixian/Desktop/FT')
sys.path.insert(0, str(FT / 'eval_fast/vendor/harness'))
import object_weighted_nr as ow  # noqa: E402

out = []
w = out.append
f1 = lambda tp, fp, fn: 200 * tp / (2 * tp + fp + fn) if tp + fp + fn else float('nan')
fmt = lambda x, d=2: '—' if x is None or x != x else f'{x:.{d}f}'


def summary(d):
    f = next(iter(glob.glob(f'{E}/{d}/real_abc/summary/*/SUMMARY.txt')), None)
    if not f:
        return {}
    s = {}
    for line in open(f):
        if m := re.search(r'OBJECT  weighted\+NR .*P/R/F1 = ([\d.]+) / ([\d.]+) / ([\d.]+)', line): s['obj_w'] = tuple(map(float, m.groups()))
        elif m := re.search(r'OBJECT  unweighted\+NR .*P/R/F1 = ([\d.]+) / ([\d.]+) / ([\d.]+)', line): s['obj_u'] = tuple(map(float, m.groups()))
        elif m := re.search(r'OBJECT  raw native .*P/R/F1 = ([\d.]+) / ([\d.]+) / ([\d.]+)', line): s['obj_raw'] = tuple(map(float, m.groups()))
        elif m := re.search(r'native ChangeF1 \(diag\)  = (\S+)', line): s['native_chg'] = None if m.group(1) == 'None' else float(m.group(1))
        elif m := re.search(r'CHANGE  D2 .*TP/FP/FN (\d+)/(\d+)/(\d+)', line): s['d2'] = tuple(map(int, m.groups()))
        elif m := re.search(r'CHANGE  D3 (\w\w) .*TP/FP/FN (\d+)/(\d+)/(\d+)   \[raw excl I2 [\d/]+, incl I2 (\d+)/(\d+)/(\d+)\]', line):
            s['d3'] = tuple(map(int, m.groups()[1:4])); s['d3_incl'] = tuple(map(int, m.groups()[4:7]))
        elif m := re.search(r'G1 \(primary, room crop\) F1@5/10/20 = ([\d.]+) / ([\d.]+) / ([\d.]+)   P/R@5 = ([\d.]+) / ([\d.]+)   area ([\d.]+)', line):
            s['g1'] = tuple(map(float, m.groups()))
        elif m := re.search(r'viewer policy, 1 cm ref +F1@5/10/20 = ([\d.]+) / ([\d.]+) / ([\d.]+)', line): s['g2'] = tuple(map(float, m.groups()))
        elif m := re.search(r'GHOST .* = ([\d.]+) %', line): s['ghost'] = float(m.group(1))
    return s


def weighted(json_path):
    j = json.load(open(json_path)); rows = list(csv.DictReader(open(j['rows'])))
    res, per = ow.object_metrics(rows, None)
    a = sum(q['active_weight_s'] for q in per if q['F1'] == q['F1']); num = sum(q['active_weight_s'] * q['F1'] for q in per if q['F1'] == q['F1'])
    return dict(P=res['P_weighted'], R=res['R_weighted'], F1=res['F1_weighted'], F1u=res['F1_unweighted'], active=a, num=num, queries=len(rows))


def wpath(d):
    return next(iter(glob.glob(f'{E}/{d}/**/OBJECT_WEIGHTED_NR.json', recursive=True)), None)


# ------------------------------------------------------------------------------------------------------ real room
REAL = [
    ('我们', dict(a='m050_states_a', b='m050_once_b', c='m050_states_c'), None),
    ('Khronos 连续 + 实例，ExternalTracker（主表）', {s: f'khronos_cont_real_inst_external_{s}' for s in 'abc'}, None),
    ('Khronos 连续 + 实例，MaxIoU', {s: f'khronos_cont_real_inst_maxiou_{s}' for s in 'abc'},
     dict(a='m050_khronos_inst_a', b='m050obj_khronos_inst_b', c='m050obj_khronos_inst_c')),
    ('Khronos 连续，语义', {s: f'm050_khronos_cont_{s}' for s in 'abc'}, None),
    ('Khronos 每次访问从空图建（从零，第 1 行）', {s: f'khronos_pervisit_semantic_{s}' for s in 'abc'}, {s: f'm050_khronos_pv_{s}' for s in 'abc'}),
]
w('# 明细（TSDF 这边，2026-10-29 最新一批）\n')
w('总表的补充。数字都由 analysis/results_detail.py 从评测输出直接读出，物体匹配 0.5 m，单位 %。"主表"标出总表用的那一项；其余是没进总表的指标、版本和行，只作补充。\n')
w('## 1. 真实房间（A / B / C）\n')
w('### 1.1 几何\n')
w('G1：每次访问结束时的地图，对该次访问 1 cm 参考、同一个房间裁剪框（eval/geometry_g1.py），主表用 F1@5cm。G2：查看器策略、1 cm 参考，不裁剪，诊断用。\n')
w('| 方法 | 访问 | G1 F1@5 | P@5 | R@5 | F1@10 | F1@20 | 面积 m² | G2 F1@5 / 10 / 20 |')
w('|---|---|---|---|---|---|---|---|---|')
for name, dirs, _ in REAL:
    for s in 'abc':
        g = summary(dirs[s]); g1 = g.get('g1'); g2 = g.get('g2')
        if not g1 and (gj := next(iter(glob.glob(f'{E}/{dirs[s]}/real_abc/geometry/*/geometry_viewer_G1.json')), None)):
            x = json.load(open(gj))   # a run scored without the summary step: its G1 file
            g1 = (100 * x['F1_05cm'], 100 * x['F1_10cm'], 100 * x['F1_20cm'], 100 * x['P_05cm'], 100 * x['R_05cm'], x['pred_area_m2'])
        if g1:
            w(f'| {name} | {s.upper()} | {g1[0]:.2f} | {g1[3]:.2f} | {g1[4]:.2f} | {g1[1]:.2f} | {g1[2]:.2f} | {g1[5]:.1f} | '
              f'{" / ".join(f"{x:.2f}" for x in g2) if g2 else "—"} |')
for S in 'ABC':   # Panoptic: its 10-29 rerun (analysis/panoptic_score.sh), each visit's final map
    d = json.load(open(E / f'panoptic_rerun_real/geometry/{S.lower()}/G1.json'))
    w(f'| Panoptic | {S} | {100*d["F1_05cm"]:.2f} | {100*d["P_05cm"]:.2f} | {100*d["R_05cm"]:.2f} | {100*d["F1_10cm"]:.2f} | {100*d["F1_20cm"]:.2f} | '
      f'{fmt(d.get("pred_area_m2"), 1)} | — |')
w('')
w('### 1.2 物体\n')
w('在线物体：每张图在自己的时刻，官方评测器（Surface、0.5 m、每 100 点抽 1、要求同语义、不补偿）。按时长加权 + NR 修正是已发表 Table 1 的算法（object_weighted_nr.py）；'
  '"一条时间轴"把三次访问的查询放在一起、按同样的权重加权，是主表的物体 F1。官方 ChangeF1（诊断）是同一评测器逐次查询的变化 F1，按会话打分，跨会话的变化不在它的真值里。\n')
w('| 方法 | 访问 | 加权 P / R / F1 | 不加权 F1 | 原始官方 F1 | 官方 ChangeF1（诊断） | 查询数 | 一条时间轴 F1（主表） |')
w('|---|---|---|---|---|---|---|---|')
for name, dirs, objdirs in REAL:
    num = den = 0.0; rows = []
    for s in 'abc':
        g = summary(dirs[s]); od = (objdirs or dirs)[s]; wp = wpath(od)
        if not wp:
            continue
        x = weighted(wp); num += x['num']; den += x['active']
        raw = summary(od).get('obj_raw') if objdirs else g.get('obj_raw')
        rows.append(f'| {name} | {s.upper()} | {x["P"]:.2f} / {x["R"]:.2f} / {x["F1"]:.2f} | {x["F1u"]:.2f} | '
                    f'{fmt(raw[2]) if raw else "—"} | {fmt(g.get("native_chg"))} | {x["queries"]} |')
    for i, r in enumerate(rows):
        w(r + (f' {100*num/den:.2f} |' if i == 0 else ' |'))
num = den = 0.0; rows = []
for s in 'abc':   # Panoptic: the maps its rerun saved during each visit, NR from its own objects (analysis/panoptic_score.sh)
    p = E / f'panoptic_rerun_real/{s}'
    rr = list(csv.DictReader(open(p / 'objects/post/native_unique_online_rows.csv'))); res, per = ow.object_metrics(rr, ow.nr_sequence(str(p / 'objects.jsonl')))
    num += sum(q['active_weight_s'] * q['F1'] for q in per if q['F1'] == q['F1']); den += sum(q['active_weight_s'] for q in per if q['F1'] == q['F1'])
    sm = json.load(open(p / 'objects/post/NATIVE_ONLINE_SUMMARY.json'))['summary']; chg = sm['ChangeF1']['mean']
    rows.append(f'| Panoptic | {s.upper()} | {res["P_weighted"]:.2f} / {res["R_weighted"]:.2f} / {res["F1_weighted"]:.2f} | {res["F1_unweighted"]:.2f} | '
                f'{100*sm["ObjectF1"]["mean"]:.2f} | {fmt(None if chg is None else 100*chg)} | {len(rr)} |')
for i, r in enumerate(rows):
    w(r + (f' {100*num/den:.2f} |' if i == 0 else ' |'))
w('')
w('### 1.3 变化\n')
w('已发表的状态协议（eval_state_change_h.py）：D2 = 访问内的变化，D3 = 自上次访问以来的变化（共同对照集，不含 I2；"含 I2"只作参考）。主表的变化 F1 = 所有访问的 D2 + D3 事件合计。'
  '旧址残留：已离开物体的表面中，本次访问看到是空的部分仍被地图显示的比例，越低越好。\n')
w('| 方法 | 访问 | D2 TP/FP/FN，F1 | D3 TP/FP/FN，F1 | D3 含 I2 | 旧址残留 % | 合计 TP/FP/FN，F1（主表） |')
w('|---|---|---|---|---|---|---|')
for name, dirs, _ in REAL:
    tot = [0, 0, 0]; rows = []
    for s in 'abc':
        g = summary(dirs[s]); d2 = g.get('d2'); d3 = g.get('d3')
        for c in (d2, d3):
            if c: tot = [a + b for a, b in zip(tot, c)]
        rows.append(f'| {name} | {s.upper()} | {"/".join(map(str, d2)) + f", {f1(*d2):.2f}" if d2 else "—"} | '
                    f'{"/".join(map(str, d3)) + f", {f1(*d3):.2f}" if d3 else "—"} | {"/".join(map(str, g["d3_incl"])) if g.get("d3_incl") else "—"} | {fmt(g.get("ghost"))} |')
    complete = all(summary(dirs[s]).get('d2') for s in 'abc')
    for i, r in enumerate(rows):
        w(r + ((f' {"/".join(map(str, tot))}，{f1(*tot):.2f} |' if complete else ' —（不是每次访问都评过） |') if i == 0 else ' |'))
pan = {}
for s in 'abc':   # Panoptic's rerun: D2 every visit, D3 from B on (its common set without I2, object 2, as the others' D3)
    p = E / f'panoptic_rerun_real/changes/{s}'
    rows = list(csv.DictReader(open(p / 'STATE_CHANGE_SUMMARY.csv')))
    d2 = next(r for r in rows if r['group'].startswith('D2') and r['component'] == 'combined')
    d3 = d3i = None
    if s != 'a':
        det = [r for r in csv.DictReader(open(p / 'STATE_CHANGE_DETAILS.csv')) if r['scope'] == 'D3' and r['object_id'] != '2']
        tp = fp = fn = 0
        for r in det:
            T = lambda k: r[k] == 'True'
            if T('expected_appeared'): tp += T('pred_appeared'); fn += not T('pred_appeared')
            if T('expected_disappeared'): tp += T('pred_disappeared_obj'); fn += not T('pred_disappeared_obj')
            elif T('pred_disappeared_obj'): fp += 1
        d3c = next(r for r in rows if r['group'].startswith('D3') and r['component'] == 'combined')
        d3, d3i = (tp, fp, fn), (int(d3c['TP']), int(d3c['FP']), int(d3c['FN']))
    pan[s] = ((int(d2['TP']), int(d2['FP']), int(d2['FN'])), d3, d3i)
tot = [sum(pan[s][0][i] + (pan[s][1][i] if pan[s][1] else 0) for s in 'abc') for i in range(3)]
for s in 'abc':
    d2, d3, d3i = pan[s]
    w(f'| Panoptic | {s.upper()} | {"/".join(map(str, d2))}, {f1(*d2):.2f} | {"/".join(map(str, d3)) + f", {f1(*d3):.2f}" if d3 else "—"} | '
      f'{"/".join(map(str, d3i)) if d3i else "—"} | — |' + (f' {"/".join(map(str, tot))}，{f1(*tot):.2f} |' if s == 'a' else ' |'))
w('')

# ------------------------------------------------------------------------------------------------------ synthetic
def full_table(d):
    rows = {}
    for line in open(E / d / 'FULL_TABLE.md'):
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) == 5 and c[0] and not c[0].startswith('**') and c[0].lower() != 'metric' and not set(c[0]) <= set('-'):
            rows[(c[0], c[1])] = c[2]
    return rows


ours, kh = full_table('obj100_states_synthetic'), full_table('obj100_khronos_cont_synthetic_5cm')
ours_pub, kh_pub = full_table('states_synthetic'), full_table('m050_khronos_cont_synthetic')
w('## 2. 合成（A / B）\n')
w('已发表的合成评测（analysis/run_synthetic_eval.py，按 evaluation_complete_20260917 的原步骤），29 个单元。物体用统一设置（0.5 m、每 100 点抽 1、要求同语义、不补偿，'
  'eval/synthetic_object_eval_official）；"发表设置"两列是原来的抽样 1、开缺分割补偿，只作对照。主表：G1 = Mesh F1@5cm（按访问），物体 F1 = A、B 的在线物体 F1'
  '（全时长）按时长 139.5 s / 118.5 s 合并，变化 F1 = D2 State + D3 State 合计。\n')
w('| 指标 | 访问 | 我们 | Khronos 连续 + 实例 | 我们（发表设置） | Khronos（发表设置） |')
w('|---|---|---|---|---|---|')
for (metric, sess), v in ours.items():
    w(f'| {metric} | {sess} | {v} | {kh.get((metric, sess), "—")} | {ours_pub.get((metric, sess), "—") if "Object" in metric else ""} | '
      f'{kh_pub.get((metric, sess), "—") if "Object" in metric else ""} |')
w('')
oa, ob = float(ours[('Online Object F1, full duration incl. initial empty map (%)', 'A')]), float(ours[('Online Object F1, full duration incl. initial empty map (%)', 'A→B')])
ka, kb = float(kh[('Online Object F1, full duration incl. initial empty map (%)', 'A')]), float(kh[('Online Object F1, full duration incl. initial empty map (%)', 'A→B')])
cnt = lambda t, k: tuple(map(int, t[(k[0], k[1])].split('/')))
co = [sum(x) for x in zip(cnt(ours, ('D2 State TP/FP/FN', 'A')), cnt(ours, ('D2 State TP/FP/FN', 'A→B')), cnt(ours, ('D3 State TP/FP/FN', 'A→B')))]
ck = [sum(x) for x in zip(cnt(kh, ('D2 State TP/FP/FN', 'A')), cnt(kh, ('D2 State TP/FP/FN', 'A→B')), cnt(kh, ('D3 State TP/FP/FN', 'A→B')))]
w(f'主表的合并值：物体 F1 我们 {(oa*139.5+ob*118.5)/258:.2f}、Khronos {(ka*139.5+kb*118.5)/258:.2f}；变化 我们 {"/".join(map(str, co))}（{f1(*co):.2f}）、'
  f'Khronos {"/".join(map(str, ck))}（{f1(*ck):.2f}）。\n')
P = E / 'panoptic_rerun_synthetic_5cm'   # Panoptic's 11-01 rerun at 5 cm (objects and background) through the same evaluation, in its "ours" slot (analysis/panoptic_score.sh)
pg = {(r['session'], float(r['threshold_m'])): float(r['F1']) for r in csv.DictReader(open(P / 'harness/geometry/FINAL_GEOMETRY.csv')) if r['scope'] == 'all'}
po = {r['session']: r for r in csv.DictReader(open(P / 'harness/online/ONLINE_OBJECT_SUMMARY.csv'))}
pc = {(r['session'], r['kind']): tuple(int(r[k]) for k in ('TP', 'FP', 'FN'))
      for r in csv.DictReader(open(P / 'harness_changes/ours/state_eval/group_scores/GROUP_STATE_SUMMARY.csv')) if r['kind'] in ('D2_hidden', 'D3') and r['session'] in ('a', 'b')}
w('Panoptic（11-01 按 5 cm 重跑，物体和背景都 5 cm（10-29 那次背景 10 cm），同一套评测；它不原生存时空地图，在线物体用运行中多存的图）：\n')
w('| 访问 | Mesh F1@5 / 10 / 20 cm | 在线物体 P / R / F1（全时长） | D2 State TP/FP/FN | D3 State TP/FP/FN |')
w('|---|---|---|---|---|')
for s in 'ab':
    o, g = po[s], ' / '.join(f'{100*pg[(s, t)]:.2f}' for t in (0.05, 0.1, 0.2))
    w(f'| {s.upper()} | {g} | {100*float(o["Precision"]):.2f} / {100*float(o["Recall"]):.2f} / {100*float(o["F1"]):.2f} | '
      f'{"/".join(map(str, pc[(s, "D2_hidden")]))} | {"/".join(map(str, pc[(s, "D3")])) if (s, "D3") in pc else "—"} |')
pt = [sum(v[i] for v in pc.values()) for i in range(3)]
w(f'\nPanoptic 主表的合并值：物体 F1 {100*(float(po["a"]["F1"])*139.5 + float(po["b"]["F1"])*118.5)/258:.2f}；变化 {"/".join(map(str, pt))}（{f1(*pt):.2f}）。\n')

# ------------------------------------------------------------------------------------------------------ Flat
w('## 3. Flat（A / B）\n')
w('G1：每次访问结束时的地图，对 Flat 点参考（flat_{1,2}_gt_10000.ply），2 cm 采样，不裁剪。物体和变化用 GaME、wavemap 同一套 Flat 脚本'
  '（runs/update_layer_game_T1_20261006/analysis/wm_flat_eval_20261028）：物体是官方评测器 + 每次访问的真值（每次访问看到的所有物体实例，60 个），变化是'
  'flat_change_eval.py 的打分（搬动按 0.10 m 门槛，15 个事件），我们只换了读入（analysis/flat_change_maps.py）。\n')
w('| 方法 | 访问 | G1 F1@5 | P@5 | R@5 | F1@10 | F1@20 |')
w('|---|---|---|---|---|---|---|')
for name, files in (('我们（主表）', ('g1final_tk_flat_a', 'g1final_flat_b_resupport')), ('Khronos 连续，语义（主表）', ('khronos_cont_flat_a', 'khronos_cont_flat_b')),
                    ('Khronos 每次访问从空图建（从零）', ('g1final_khronos_flat_a', 'g1final_khronos_flat_b')), ('Panoptic', ('panoptic_rerun_flat/g1_a', 'panoptic_rerun_flat/g1_b'))):
    for s, d in zip('AB', files):
        g = json.load(open(E / d / 'G1_flat.json'))
        w(f'| {name} | {s} | {100*g["F1_05cm"]:.2f} | {100*g["P_05cm"]:.2f} | {100*g["R_05cm"]:.2f} | {100*g["F1_10cm"]:.2f} | {100*g["F1_20cm"]:.2f} |')
w('')
w('| 方法 | 访问 | 在线物体 P / R / F1（不加权） | 加权 F1 | 一条时间轴 F1（主表） |')
w('|---|---|---|---|---|')
for name, base in (('我们', 'flat_objects_changes/objects_ours_{}'), ('Khronos 连续，语义', 'flat_objects_changes/objects_khronos_{}'),
                   ('Panoptic', 'panoptic_rerun_flat/{}/objects')):
    num = den = 0.0; rows = []
    for s in 'ab':
        p = E / base.format(s)
        sm = json.load(open(p / 'post/NATIVE_ONLINE_SUMMARY.json'))['summary']
        rr = list(csv.DictReader(open(p / 'post/native_unique_online_rows.csv'))); res, per = ow.object_metrics(rr, None)
        num += sum(q['active_weight_s'] * q['F1'] for q in per if q['F1'] == q['F1']); den += sum(q['active_weight_s'] for q in per if q['F1'] == q['F1'])
        rows.append(f'| {name} | {s.upper()} | {100*sm["ObjectPrecision"]["mean"]:.2f} / {100*sm["ObjectRecall"]["mean"]:.2f} / {100*sm["ObjectF1"]["mean"]:.2f} | {res["F1_weighted"]:.2f} |')
    for i, r in enumerate(rows):
        w(r + (f' {100*num/den:.2f} |' if i == 0 else ' |'))
w('')
fc = json.load(open(E / 'flat_objects_changes/FLAT_CHANGE.json'))
fcb = json.load(open(E / 'flat_objects_changes/FLAT_CHANGE_before_gate.json'))
w('| 方法 | 变化 TP/FP/FN，F1（主表，0.10 m 门槛） | 门槛之前 | 判错的事件 |')
w('|---|---|---|---|')
pfc = json.load(open(E / 'panoptic_rerun_flat/FLAT_CHANGE.json'))['panoptic']   # its rerun, scored after the gate only
for name, x, y in (('我们', fc['ours'], fcb['ours']), ('Khronos 连续，语义', fc['khronos'], fcb['khronos']), ('Panoptic', pfc, None)):
    wrong = '；'.join(f"{'漏' if e['expected'] else '误'} {e['name']}（{e['kind']}，{e['component']}）" for e in x['events'] if e['expected'] != e['predicted'])
    before = f'{y["TP"]}/{y["FP"]}/{y["FN"]}，{y["F1"]:.2f}' if y else '—'
    w(f'| {name} | {x["TP"]}/{x["FP"]}/{x["FN"]}，{x["F1"]:.2f} | {before} | {wrong} |')
w('')

# ------------------------------------------------------------------------------------------------------ 3RScan
w('## 4. 3RScan（8 间，27 次访问）\n')
w('物体、变化：官方评测器的在线切片（每张图在自己的时刻），0.5 m；主表取每间的 weighted（按快照当值时长），房间间简单平均，只平均有分的房间。'
  '"我们（主表）"经 chain_fill 10-30 版（只声明新出现）；"我们（10-29 版）"是 chain_fill 还按我们的计数声明搬动时的分（总表 10-30 之前用的），只作对照。'
  'G1：每次访问结束时的地图对本次参考，F1@5cm；主表 = 每间先平均自己的几次访问、再对房间平均。\n')
def geo(r, fn):
    return [float(m.group(1)) for l in open(R / '3rscan' / r / fn) for m in [re.search(r'visit \d+ geometry @5cm P/R/F1 \[[\d.]+, [\d.]+, ([\d.]+)\]', l)] if m]
rooms = sorted(p.name for p in (R / '3rscan').glob('p3_*'))
w('| 房间 | 访问数 | G1 我们（各次） | G1 Khronos 实例（各次） | G1 Khronos 语义（各次） | G1 Panoptic（各次） |')
w('|---|---|---|---|---|---|')
for r in rooms:
    a, b, c = geo(r, 'eval_ours_290e110.log'), geo(r, 'eval_khronos_inst_ext_geom.log'), geo(r, 'eval_khronos_scene.log')
    p = [v['5cm'][2] for v in json.load(open(R / '3rscan' / r / 'panoptic_rerun/CHAIN_EVAL.json'))['geometry']]
    w(f'| {r[3:]} | {len(a)} | {" / ".join(f"{x:.1f}" for x in a)} | {" / ".join(f"{x:.1f}" for x in b)} | {" / ".join(f"{x:.1f}" for x in c)} | '
      f'{" / ".join(f"{x:.1f}" for x in p)} |')
w('')
METH = [('我们（主表）', 'ours_290e110/khronos_metrics'), ('我们（10-29 版）', 'ours_290e110/khronos_metrics_counter_moves'),
        ('Khronos 连续 + 实例，ExternalTracker（主表）', 'khronos_inst_ext/khronos_metrics'), ('Khronos 连续 + 实例，MaxIoU', 'khronos_inst/khronos_metrics'),
        ('Khronos 连续，语义', 'khronos_scene/khronos_metrics'), ('Panoptic', 'panoptic_rerun/khronos_metrics')]
w('| 房间 | ' + ' | '.join(f'{n} 物体 / 变化（weighted）' for n, _ in METH) + ' |')
w('|---|' + '---|' * len(METH))
agg = {n: ([], []) for n, _ in METH}
for r in rooms:
    cells = []
    for n, d in METH:
        x = json.load(open(R / '3rscan' / r / d / 'METRICS.json'))['metrics']
        o, c = x['ObjectF1']['weighted'], x['ChangeF1']['weighted']
        if o is not None: agg[n][0].append(o)
        if c is not None: agg[n][1].append(c)
        cells.append(f'{fmt(o, 1)} / {fmt(c, 1)}')
    w(f'| {r[3:]} | ' + ' | '.join(cells) + ' |')
w('| **8 间平均（有分的房间数）** | ' + ' | '.join(f'{sum(o)/len(o):.1f} ({len(o)}) / {sum(c)/len(c):.1f} ({len(c)})' for o, c in agg.values()) + ' |')
w('')
w('上表的物体是主表的物体 F1；上表的变化是官方 Khronos 的变化指标（读方法自己写的出现 / 消失记录），只放在明细。主表的变化 F1 用下面的事件评测'
  '（eval/threerscan_chain_eval.py，和真实、合成、Flat 同一类）：每次访问对上一次访问，真值状态一半以上的点落在地图同一物体表面 10 cm 内才算被表示，8 间所有事件合计。\n')
w('| 房间 | 我们 TP/FP/FN，F1 | Khronos 连续 + 实例，ExternalTracker TP/FP/FN，F1 | Panoptic TP/FP/FN，F1 |')
w('|---|---|---|---|')
tot = {'ours_290e110': [0, 0, 0], 'khronos_inst_ext': [0, 0, 0], 'panoptic_rerun': [0, 0, 0]}
for r in rooms:
    cells = []
    for d in tot:
        x = json.load(open(R / '3rscan' / r / d / 'CHAIN_EVAL.json'))
        tot[d] = [a + b for a, b in zip(tot[d], (x['TP'], x['FP'], x['FN']))]
        cells.append(f'{x["TP"]}/{x["FP"]}/{x["FN"]}，{f1(x["TP"], x["FP"], x["FN"]):.1f}')
    w(f'| {r[3:]} | ' + ' | '.join(cells) + ' |')
w('| **8 间合计（主表）** | ' + ' | '.join(f'{"/".join(map(str, t))}，{f1(*t):.1f}' for t in tot.values()) + ' |')
w('')

# ------------------------------------------------------------------------------------------------------ TESSE
w('## 5. TESSE（office / apartment）\n')
w('Khronos 论文表 I 的官方指标，官方 4D（每张图在所有更早的图时刻查询，官方 tables.py 的 4D 平均）。"我们（主表）"经 chain_fill 10-30 版（只声明新出现，和 3RScan 同一版）；"我们（10-29 版）"是 chain_fill 还按我们的计数声明搬动时的分（总表 10-30 之前用的），只作对照。\n')
w('| 场景 | 方法 | 背景 Pre / Rec / F1 | 静态物体 Pre / Rec / F1 | 动态物体 Pre / Rec / F1 | 变化 Pre / Rec / F1 |')
w('|---|---|---|---|---|---|')
for scene, ev, rows in (('office', 'tesse_paper_qj', (('我们（主表）', 'ours_fill1101_chain4dAppear_5m_all_m050'), ('我们（10-29 版）', 'ours_resupport_chain4dA_5m_all_m050'),
                                                     ('Khronos 连续 + 实例，5 cm（主表）', 'khronos_inst_5cm_all_m050'), ('Khronos 连续 + 实例，10 cm', 'khronos_inst_5m_all_m050'), ('Khronos 语义', 'khronos_5m_all_m050'))),
                        ('apartment', 'tesse_apt_qj', (('我们（主表）', 'ours_fill1101_chain4dAppear_5m_all_m050'), ('我们（10-29 版）', 'ours_chain4dA_5m_all_m050'),
                                                       ('Khronos 连续 + 实例，5 cm（主表）', 'khronos_inst_5cm_all_m050'), ('Khronos 连续 + 实例，10 cm', 'khronos_inst_5m_all_m050'), ('Khronos 语义', 'khronos_5m_all_m050')))):
    for name, d in rows:
        t = (E / ev / d / 'TABLES_4D.txt').read_text().splitlines()
        i = next(k for k, l in enumerate(t) if l.startswith('Data'))
        v = re.findall(r'\d+\.\d', t[i + 1][len(d):] if t[i + 1].startswith(d) else t[i + 1])[-12:]
        w(f'| {scene} | {name} | {" / ".join(v[0:3])} | {" / ".join(v[3:6])} | {" / ".join(v[6:9])} | {" / ".join(v[9:12])} |')
w('')
Path(sys.argv[1]).write_text('\n'.join(out) + '\n')
print('written', sys.argv[1], len(out), 'lines')
