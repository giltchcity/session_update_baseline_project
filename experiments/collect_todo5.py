"""Every number of the TODO 5 chains next to the user's acceptance lines and the references (CPU, reads the run directories).
原理思想：结果只从运行目录里读（官方评分 CSV/JSON、run.json、工具输出），旁边列用户的线（合成 B post_ref F1 >= 96.43、P >= 96.6，
D1/D2/D3 不变；真实 A/B/C G1 F1 >= 93.66 / 92.98 / 94.82；显存 <= 13.5 GB；时间 <= 1.3x 原生）和参照（R11p pre_ref、r21 post_ref、
原生 GaME 第 3 行的会话时间 syn_row3v3 / real_row3v3）。缺的项写 PENDING，不当作通过。
usage: collect_todo5.py REFINE_COMMIT_SHORT
"""
import csv
import json
import re
import sys
from pathlib import Path

RR = Path('/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006')
RK = sys.argv[1]
S, ES = RR / 'syn_chain_4ca6976/map', RR / f'syn_chain_4ca6976/refine_{RK}'
RL, ER = RR / 'real_chain_4ca6976/map', RR / f'real_chain_4ca6976/refine_{RK}'
FO = RR / f'flat_chain_{RK}'


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def geo(csvp, session='b', scope='all', thr='0.05'):
    try:
        for r in csv.DictReader(open(csvp)):
            if r['method'] == 'ours' and r['session'] == session and r['scope'] == scope and r['threshold_m'] == thr:
                return {k: round(100 * float(r[k]), 2) for k in ('Precision', 'Recall', 'F1')}
    except Exception:
        return None


def states(run_eval):
    p = run_eval / 'ours/state_eval/group_scores/GROUP_STATE_SUMMARY.csv'
    try:
        return {r['kind']: f"TP {r['TP']} FP {r['FP']} FN {r['FN']} F1 {float(r['F1']):.3f}" for r in csv.DictReader(open(p)) if r['session'] == 'ab'}
    except Exception:
        return None


def g1(p):
    d = load(p)
    return None if d is None else round(100 * d['F1_05cm'], 2)


def pend(x):
    return 'PENDING' if x is None else x


print(f'TODO 5 results (mapping tree r21c_base 4ca6976; refinement tree {RK}); every line read from the run directories')
print('\n== synthetic B from the accepted A (official scorer, all scope, 5 cm)')
pre, post = geo(S / 'eval_syn/geometry/FINAL_GEOMETRY.csv'), geo(ES / 'post_ref/eval_syn/geometry/FINAL_GEOMETRY.csv')
print(f'   pre_ref : {pend(pre)}   [R11p B pre_ref]')
ok = post is not None and post['F1'] >= 96.43 and post['Precision'] >= 96.6
print(f'   post_ref: {pend(post)}   lines F1 >= 96.43, P >= 96.6 -> {"PASS" if ok else ("FAIL" if post else "PENDING")}   [r21: P 96.68 R 96.56 F1 96.62]')
for lab, d, ref in (('pre_ref', S / 'eval_syn', RR / 'syn_row4f3e5_r11p/eval_syn'), ('post_ref', ES / 'post_ref/eval_syn', RR / 'syn_row4f3e5_r21/post_ref/eval_syn')):
    a, b = states(d), states(ref)
    print(f'   states {lab} (ab): {pend(a)}')
    print(f'      reference ({ref.parent.name if lab == "post_ref" else ref.parent.name}): {pend(b)}')
rj = load(S / 'session_b/run.json')
if rj:
    print(f'   B mapping: {rj["seconds"]} s for {rj["frames"]} frames = {rj["seconds"] / 4496.8:.2f} x native GaME row 3 (syn_row3v3 B 4496.8 s); '
          f'peak GPU {rj.get("peak_gpu_gb")} GB (line 13.5)')
pr = (load(ES / 'session_b/run.json') or {}).get('post_ref')
if pr:
    print(f'   refinement: {pr.get("completed_iterations")}/{pr.get("requested_iterations")} steps, {pr.get("seconds")} s '
          f'(native row 3 refinement 1590.3 s -> {pr.get("seconds", 0) / 1590.3:.2f} x), peak {pr.get("peak_gpu_gb")} GB')
for tool in ('views15_final', 'bathroom_struct_final', 'compensated_end_final', 'heldout_last_final', 'last_kf_final'):
    p = ES / f'{tool}.out'
    if p.exists():
        lines = [l for l in p.read_text(errors='ignore').splitlines() if l.strip() and 'Warning' not in l and not l.startswith(('/', '[W'))]
        print(f'   -- {tool} (columns: preref = R11p B, r21, final = this chain)')
        for l in lines[-24:]:
            print('      ' + l[:160])
print('\n== real A -> B -> C (G1 F1 at 5 cm; lines 93.66 / 92.98 / 94.82)')
a, b, c = (g1(RL / f'eval_real/geometry/{s}/G1_tsdf.json') for s in 'abc')
cp = g1(ER / 'post_ref/eval_real/geometry/c/G1_tsdf.json')
lines = dict(a=93.66, b=92.98, c=94.82)
for s, v in zip('abc', (a, b, c)):
    print(f'   {s.upper()} pre_ref {pend(v)} -> {"PASS" if v is not None and v >= lines[s] else ("FAIL" if v is not None else "PENDING")} (line {lines[s]})')
print(f'   C post_ref {pend(cp)}')
native = dict(a=749.7, b=1178.9, c=3795.4)
for s in 'bc':
    j = load(RL / f'session_{s}/run.json')
    if j:
        print(f'   {s.upper()} mapping {j["seconds"]} s = {j["seconds"] / native[s]:.2f} x native GaME row 3 (real_row3v3 {native[s]} s); peak GPU {j.get("peak_gpu_gb")} GB')
acc = RR / 'analysis/acceptance_real_chain_4ca6976.txt'
if acc.exists():
    for l in acc.read_text(errors='ignore').splitlines():
        if l.startswith(('VERDICT', '  FAIL', '  PENDING', 'CONDITION')):
            print('   ' + l[:170])
print('\n== Flat (GaME protocol: run1 -> run2, every 10th run2 frame held out)')
for v in ('pre_ref', 'post_ref'):
    j = load(FO / f'render_metrics_{v}.json')
    print(f'   ours {v}: {pend(json.dumps(j)[:600] if j else None)}')
nl = Path('/home/jixian/Desktop/FT/runs/game_flat_official_20261007/run.log')
if nl.exists():
    tail = [l for l in nl.read_text(errors='ignore').replace('\r', '\n').splitlines() if re.search(r'PSNR|SSIM|LPIPS|depth|exit', l)]
    print('   native GaME: ' + (' | '.join(tail[-6:])[:600] if tail else 'PENDING'))
print('\n== code traceability')
print('   results: mapping 4ca6976 (r21c_base), refinement', RK, '; delivered tree chain_clean 2b65d6d (4ca6976 minus dead code and the 10-09 record scripts, comments as')
print('   principle + source); equivalence of 2b65d6d with the result trees on the same inputs (equiv_compare.py):')
for name, old in (('equiv150_clean_2b65d6d', 'speed150_final0638_b9400c9'), ('equiv30_real_bc_clean_2b65d6d', 'smoke30_real_bc_4ca6976'),
                  ('equiv4k_clean_2b65d6d', 'speed3k_r21c_base_4ca6976')):
    d = RR / 'code_mine_20261014/runs_quick3k' / name
    print(f'   {name} vs {old}: {"present" if d.exists() else "PENDING"} ({d})')
print('\n== numbers without a derivation (GameBackend.__init__, chain_clean game.py)')
for k, use, src in (('tolerance 0.05 m', 'depth match of the support test, F3e5 read-through / behind', 'fork ray_verificator.h:124 surface_match_tolerance (same role); official Khronos has no such value -> not derived at the official level'),
                    ('bg_voxel 0.02 m', 'readout de-duplication grid, background', 'not derived (own; = the points / wavemap 2 cm, also own; the TSDF rows use 0.05 / 0.10)'),
                    ('obj_voxel 0.01 m', 'readout de-duplication grid, objects', 'not derived (the TSDF object resolution is 0.02 real / 0.05 synthetic)'),
                    ('min_mask_px_full 50 px', 'smallest instance mask (full resolution, / step^2)', 'Khronos uHumans2_openset.yaml:61 / jackal_openset.yaml:61 InstanceForwarding min_cluster_size (same role), not rescaled to the image size'),
                    ('F3e4 band 0.10 m', 'behind-surface band of the support test', '2DGS mesh extraction sdf_trunc = 5 x voxel with eval/tsdf_export.py\'s 2 cm voxel (the 5 is sourced, the 2 cm is own)'),
                    ('min_alpha 0.5', '-', 'removed (a13bfc2): the chain never read it')):
    print(f'   {k:24s} {use:58s} {src}')
print('COLLECT_DONE')
