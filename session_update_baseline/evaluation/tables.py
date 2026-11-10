"""Build every table from eval/results and eval/registry.yaml. Nothing is typed in by hand.

usage: tables.py  -> writes eval/tables/T1_real_sessions.md, T2_real_transitions.md,
                     T3_real_memory_ablation.md, S_synthetic.md and ALL_TABLES.md
"""
import csv, json
from pathlib import Path
import yaml

EV = Path('/home/jixian/Desktop/FT/eval'); RES = EV / 'results'; OUT = EV / 'tables'; OUT.mkdir(exist_ok=True)
REG = yaml.safe_load((EV / 'registry.yaml').read_text())
METHOD_LABEL = {'ours': 'Ours (t2)', 'geo21': 'Ours (geo21)', 'ours_clean': 'Ours from-scratch (geo21)', 'plain_v37': 'plain V37 (published)',
                'fix_v37': 'fix V37 (published)', 'khronos': 'Official Khronos', 'panoptic': 'Panoptic Multi-TSDF'}


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, nd=2):
    return '—' if x is None else ('%.*f' % (nd, x))


def prf(t):
    return '—' if not t else ' / '.join(fmt(v) for v in t)


def delta(a, b):
    return '' if a is None or b is None else ' (%+.2f)' % (a - b)


def real_row(m, s):
    base = RES / 'real' / m / s
    o = jload(base / 'objects' / 'SUMMARY.json'); g1 = jload(base / 'geometry' / 'G1.json'); g2 = jload(base / 'geometry' / 'G2.json')
    ref = REG['references']['real'].get(m, {}).get(s, {})
    r = dict(method=m, stage=s)
    if o:
        w = o['object']['duration_weighted_nr']; r['object'] = [w['P'], w['R'], w['F1']]
        r['d2'] = o['change']['D2'].get('published_policy'); r['d3'] = (o['change'].get('D3') or {}).get('published_policy') or (o['change'].get('D3') or {}).get('table2')
        r['ghost'] = (o.get('ghost') or {}).get('old_site_full_map_ghost_pct')
    else:
        r['object'] = ref.get('object_prf'); r['d2'] = {'F1': ref.get('d2_f1')} if ref.get('d2_f1') is not None else None
        r['d3'] = {'P': ref['d3_prf'][0], 'R': ref['d3_prf'][1], 'F1': ref['d3_prf'][2]} if ref.get('d3_prf') else None
        r['ghost'] = ref.get('ghost_pct')
    r['g1'] = g1; r['g2'] = g2
    return r


def t1():
    ref = {s: real_row('plain_v37', s) for s in 'abc'}
    lines = ['# T1 real room-18, per session', '',
             'Object = official Khronos ObjectEvaluator, duration-weighted + NR (Table 1 policy). Change D2 = in-session hidden change (Table 1 policy). '
             'G1 = oriented room-crop two-mesh F-score vs 1 cm same-sensor pseudo-GT (primary). G2 = observed-domain F-score (diagnostic). '
             'Deltas against plain V37 (published).', '',
             '| method | stage | Object P / R / F1 | Change D2 F1 | G1 F1@5 (P / R) | G1 F1@10 | G1 F1@20 | G2 F1@5 | area m² |', '|---|---|---|---|---|---|---|---|---|']
    for m in ['ours', 'geo21', 'ours_clean', 'plain_v37', 'khronos', 'panoptic']:
        for s in 'abc':
            r = real_row(m, s); p = ref[s]
            if not (r.get('object') or r.get('g1')): continue
            f1 = r['object'][2] if r.get('object') else None; pf1 = p['object'][2] if p.get('object') else None
            g1 = r['g1']; pg1 = p['g1']
            lines.append('| %s | %s | %s%s | %s%s | %s%s (%s / %s) | %s | %s | %s%s | %s |' % (
                METHOD_LABEL[m], s.upper(), prf(r.get('object')), delta(f1, pf1) if m != 'plain_v37' else '',
                fmt(r['d2']['F1']) if r.get('d2') else '—', delta(r['d2']['F1'] if r.get('d2') else None, p['d2']['F1'] if p.get('d2') else None) if m != 'plain_v37' else '',
                fmt(g1['F1_05cm'] * 100) if g1 else '—', delta(g1['F1_05cm'] * 100 if g1 else None, pg1['F1_05cm'] * 100 if pg1 else None) if m != 'plain_v37' else '',
                fmt(g1['P_05cm'] * 100) if g1 else '—', fmt(g1['R_05cm'] * 100) if g1 else '—',
                fmt(g1['F1_10cm'] * 100) if g1 else '—', fmt(g1['F1_20cm'] * 100) if g1 else '—',
                fmt(r['g2']['F1_05cm'] * 100) if r.get('g2') else '—', delta(r['g2']['F1_05cm'] * 100 if r.get('g2') else None, p['g2']['F1_05cm'] * 100 if p.get('g2') else None) if m != 'plain_v37' else '',
                fmt(g1['pred_area_m2'], 1) if g1 else '—'))
    return '\n'.join(lines)


def t2():
    lines = ['# T2 real room-18, cross-session transitions', '', 'D3 = previous final -> this final (Table 2 policy, cohort excl. I2). Ghost = old-site full-map residual after the site was confirmed empty (lower is better). Deltas vs plain V37.', '',
             '| method | transition | Change D3 P / R / F1 | old-site ghost % | G1 F1@5 |', '|---|---|---|---|---|']
    for m in ['ours', 'geo21', 'ours_clean', 'plain_v37', 'khronos', 'panoptic']:
        for tr, (_, s) in REG['datasets']['real']['transitions'].items():
            r = real_row(m, s); p = real_row('plain_v37', s)
            if not r.get('d3') and r.get('ghost') is None: continue
            d3 = r['d3']; pd3 = p.get('d3')
            lines.append('| %s | %s | %s%s | %s%s | %s |' % (METHOD_LABEL[m], tr.upper(),
                prf([d3['P'], d3['R'], d3['F1']]) if d3 else '—', delta(d3['F1'] if d3 else None, pd3['F1'] if pd3 else None) if m != 'plain_v37' else '',
                fmt(r.get('ghost')), delta(r.get('ghost'), p.get('ghost')) if m != 'plain_v37' else '',
                fmt(r['g1']['F1_05cm'] * 100) if r.get('g1') else '—'))
    return '\n'.join(lines)


def t3():
    lines = ['# T3 real room-18, memory ablation (same code, chain vs from-scratch)', '', '| stage | Ours chain Object F1 | Ours from-scratch Object F1 | Δ memory | Khronos | Δ vs Khronos | chain G1 F1@5 | from-scratch G1 F1@5 |', '|---|---|---|---|---|---|---|---|']
    for s in 'abc':
        a, b, k = real_row('ours', s), real_row('ours_clean', s), real_row('khronos', s)
        fa = a['object'][2] if a.get('object') else None; fb = b['object'][2] if b.get('object') else None; fk = k['object'][2] if k.get('object') else None
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (s.upper(), fmt(fa), fmt(fb), delta(fa, fb).strip() or '—', fmt(fk), delta(fa, fk).strip() or '—',
                                                          fmt(a['g1']['F1_05cm'] * 100) if a.get('g1') else '—', fmt(b['g1']['F1_05cm'] * 100) if b.get('g1') else '—'))
    return '\n'.join(lines)


def synthetic():
    lines = ['# S synthetic A/B (full table rows from evaluate_ours_full.sh; delta vs published fix V37)', '']
    for m in ['ours', 'geo21', 'ours_clean']:
        f = RES / 'synthetic' / m / 'ab' / 'full' / 'FULL_TABLE.md'
        if not f.exists(): continue
        lines += ['## %s' % METHOD_LABEL[m], '']
        rows = [l for l in f.read_text().splitlines() if l.startswith('|')]
        lines += rows + ['']
    pub = Path(REG['references']['synthetic']['published_csv']) / 'MAIN_RESULTS.csv'
    if pub.exists():
        lines += ['## published main results (fix V37 / Khronos), for reference', '', '| method | stage | Mesh F1@5 | Online Object F1 | Final Object F1 | D2 State F1 | D3 State F1 |', '|---|---|---|---|---|---|---|']
        for r in csv.DictReader(pub.open()):
            g = lambda k: fmt(float(r[k])) if r.get(k) not in (None, '') else '—'
            lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (r['method'], r['session'], g('Mesh_F1_5cm_pct'), g('Online_Object_full_duration_F1_pct'), g('Final_Object_F1_pct'), g('D2_State_F1_pct'), g('D3_State_F1_pct')))
    return '\n'.join(lines)


if __name__ == '__main__':
    parts = {'T1_real_sessions.md': t1(), 'T2_real_transitions.md': t2(), 'T3_real_memory_ablation.md': t3(), 'S_synthetic.md': synthetic()}
    for n, t in parts.items(): (OUT / n).write_text(t + '\n')
    (OUT / 'ALL_TABLES.md').write_text('\n\n'.join(parts.values()) + '\n')
    print('\n\n'.join(parts.values()))
