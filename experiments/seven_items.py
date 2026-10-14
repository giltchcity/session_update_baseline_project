"""The user's 'not worse than pre-ref' items of the synthetic B refinement as one table (supervisor 12:04 / 12:14), CPU, reads tool outputs.
Items: bathroom (bathroom_struct: within 5 cm, veil in front of the GT), kitchen wall (b2625), chairs (chair, chair_white), curtain
(curtain_box, b1830 / b1845 / b1860), pot (pot_ROI, pot_ring at b3525), table-top residue (rows of the untracked items 79 80 81 117 170
181 alive at the end, table_items_count.py), the 15 views of views15 (each view, and their mean).
Columns: R11p pre_ref (the tools' fixed 'pre-ref' column), r21, SH (r21 + GaME's SH ramp, 14f3855), this chain's own pre_ref, this
chain's refined map. Each refined map is judged against the pre_ref it was refined from (r21 and SH: R11p B; this chain: its own B),
every measure strictly (no tolerance: the tool renders are deterministic forward passes); this chain's refined map is also shown
against R11p's pre_ref. Lower is better except 'within 5 cm'. A missing measurement prints PENDING and is not a pass.
usage: seven_items.py REFINE_DIR      (syn_chain_4ca6976/refine_<tree>)
"""
import re
import sys
from pathlib import Path

RR = Path('/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006')
REF = Path(sys.argv[1]) if len(sys.argv) > 1 else None
SRC = {'SH': RR / 'exp_sh_r11p_14f3855', 'chain_pre': RR / 'syn_chain_4ca6976/map', 'chain_final': REF}
COLS = ('R11p_pre', 'r21', 'SH', 'chain_pre', 'chain_final')
VIEWS = ('a2700', 'a2850', 'a2985', 'b1830', 'b1845', 'b1860', 'b2190', 'b2625', 'b345', 'b1395', 'b2955', 'b3135', 'b3480', 'b3525', 'b3540')


def views15(d):
    p = d / 'views15_final.out' if d else None
    if not p or not p.exists():
        return None
    rows = [l.split() for l in p.read_text(errors='ignore').splitlines() if l.strip()]
    head = next((r for r in rows if r and r[0] == 'view'), None)
    if head is None:
        return None
    out = {}
    for r in rows:
        if len(r) == len(head) and r[0] != 'view':
            out[r[0]] = dict(zip(head[1:], map(float, r[1:])))
    return out


def bathroom(d):
    p = d / 'bathroom_struct_final.out' if d else None
    if not p or not p.exists():
        return None
    out = {}
    for l in p.read_text(errors='ignore').splitlines():
        m = re.match(r'(\S+)\s+\| echo\s+([\d.]+)% \| within 5 cm\s+([\d.]+)% \| in front of GT >5 cm \(fog/veil\)\s+([\d.]+)%', l)
        if m:
            out[m.group(1)] = {'within5': float(m.group(3)), 'veil': float(m.group(4))}
    return out or None


def items():
    out = {}
    for p in sorted((RR / 'code_mine_20261014/runs_quick3k').glob('table_items_count_*.txt')):
        for l in p.read_text(errors='ignore').splitlines():
            m = re.match(r'(\S+)\s+alive rows\s+(\d+)\s+table items\s+(\d+)', l)
            if m:
                out[m.group(1)] = int(m.group(3))
    return out


V = {k: views15(d) for k, d in SRC.items()}
B = {k: bathroom(d) for k, d in SRC.items()}
T = items()


def col_values(get):
    """get(table, column label) over the tool outputs -> the five columns."""
    vals = {}
    for src in ('SH', 'chain_final', 'chain_pre'):            # the fixed columns are the same in every output: take the first present
        if 'R11p_pre' not in vals:
            v = get(src, ('preref', 'pre-ref'))
            if v is not None:
                vals['R11p_pre'] = v
        if 'r21' not in vals:
            v = get(src, ('r21',))
            if v is not None:
                vals['r21'] = v
    for src in ('SH', 'chain_pre', 'chain_final'):
        v = get(src, ('final',))
        if v is not None:
            vals[src] = v
    return vals


def v15(key):
    def get(src, labels):
        t = V.get(src)
        if not t or key not in t:
            return None
        return next((t[key][l] for l in labels if l in t[key]), None)
    return col_values(get)


def bath(field):
    def get(src, labels):
        t = B.get(src)
        if not t:
            return None
        return next((t[l][field] for l in labels if l in t), None)
    return col_values(get)


def table_rows():
    m = {'pre_ref_R11p': 'R11p_pre', 'r21': 'r21', 'SH': 'SH', 'chain_pre_ref': 'chain_pre', 'chain_final': 'chain_final'}
    return {m[k]: v for k, v in T.items() if k in m}


def verdict(v, ref, higher):
    if v is None or ref is None:
        return 'PENDING'
    ok = v >= ref if higher else v <= ref
    return ('HOLDS' if ok else 'NOT') + f' ({v - ref:+.2f})'


measures = [('bathroom', 'within 5 cm %', bath('within5'), True), ('bathroom', 'veil (in front > 5 cm) %', bath('veil'), False),
            ('kitchen wall', 'b2625 MAE', v15('b2625'), False),
            ('chairs', 'chair MAE', v15('chair'), False), ('chairs', 'chair_white MAE', v15('chair_white'), False),
            ('curtain', 'curtain_box MAE', v15('curtain_box'), False)] + \
           [('curtain', f'{k} MAE', v15(k), False) for k in ('b1830', 'b1845', 'b1860')] + \
           [('pot', 'pot_ROI MAE (b3525)', v15('pot_ROI'), False), ('pot', 'pot_ring MAE (b3525)', v15('pot_ring'), False),
            ('table residue', 'item rows alive at the end', table_rows(), False)] + \
           [('15 views', f'{k} MAE', v15(k), False) for k in VIEWS]
mean = {}
for c in COLS:
    xs = [v15(k).get(c) for k in VIEWS]
    if all(x is not None for x in xs):
        mean[c] = sum(xs) / len(xs)
measures.append(('15 views', 'mean of the 15 MAE', mean, False))

fmt = lambda x: 'PENDING' if x is None else (f'{x:.2f}' if isinstance(x, float) else str(x))
print('Synthetic B, "not worse than the pre_ref it was refined from" (r21 / SH from R11p B; this chain from its own B)')
print('%-14s %-28s' % ('item', 'measure') + ''.join('%12s' % c for c in COLS) + '   %-17s %-17s %-17s %-17s' % ('r21 vs R11p_pre', 'SH vs R11p_pre', 'final vs own pre', 'final vs R11p_pre'))
item_ok = {}
for item, name, vals, higher in measures:
    vs = [verdict(vals.get('r21'), vals.get('R11p_pre'), higher), verdict(vals.get('SH'), vals.get('R11p_pre'), higher),
          verdict(vals.get('chain_final'), vals.get('chain_pre'), higher), verdict(vals.get('chain_final'), vals.get('R11p_pre'), higher)]
    print('%-14s %-28s' % (item, name) + ''.join('%12s' % fmt(vals.get(c)) for c in COLS) + '   ' + ' '.join('%-17s' % v for v in vs))
    for j, key in enumerate(('r21', 'SH', 'final_own', 'final_R11p')):
        item_ok.setdefault(item, {}).setdefault(key, []).append(vs[j].split()[0])
print('\nper item (HOLDS only when every measure of the item holds; PENDING when any is missing):')
print('%-14s %-12s %-12s %-14s %-14s' % ('item', 'r21', 'SH', 'final vs own', 'final vs R11p'))
for item, d in item_ok.items():
    agg = lambda xs: 'PENDING' if 'PENDING' in xs else ('HOLDS' if all(x == 'HOLDS' for x in xs) else f'NOT ({xs.count("NOT")}/{len(xs)})')
    print('%-14s %-12s %-12s %-14s %-14s' % (item, agg(d['r21']), agg(d['SH']), agg(d['final_own']), agg(d['final_R11p'])))
print('SEVEN_ITEMS_DONE')
