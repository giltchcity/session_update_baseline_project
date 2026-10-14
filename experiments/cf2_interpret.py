"""Reading of the bathroom counterfactual (restore_cf2.py; supervisor 13:14): where the B map's photo error over A's map comes from.
CPU, reads the counterfactual's run.log (its 'photo MAE (0-255) per view' table). Per view, the arms are nested, so consecutive
differences split the excess of the B map over the anchor (A's own map at A's end) into parts:
  B-born rows            B_as_is - hide_B_born
  A rows B changed       hide_B_born - hide_B_born+touched_from_A          (parameters of A rows B measured, back to their A values)
  A rows B ended         hide_B_born+touched_from_A - +revive_ended_A      (A rows alive at A's end, stored in B, not alive at B's end)
  A rows B deleted       +revive_ended_A - +revive_deleted_A                (A rows alive at A's end, gone from B's checkpoint)
  not explained          +revive_deleted_A - A_at_A_end                     (what the four parts do not account for)
and the single effects B_as_is - B_as_is+revive_ended_A / B_as_is+revive_deleted_A (each part alone on the B map). A negative
part means that restoring the part made the view worse. The bathroom views are a2700 / a2850 / a2985 (static A -> B); b2625 is the
kitchen wall, b3447 the last keyframe of B (the pot). No threshold is applied; the numbers are the reading.
usage: cf2_interpret.py CF2_RUN_LOG
"""
import re
import sys

lines = open(sys.argv[1], errors='ignore').read().splitlines()
head_i = next(i for i, l in enumerate(lines) if l.startswith('photo MAE'))
cols = lines[head_i + 1].split()[1:]
mae = {}
for l in lines[head_i + 2:]:
    parts = l.split()
    if len(parts) == len(cols) + 1 and re.fullmatch(r'[\d.]+', parts[-1]):
        mae[parts[0]] = dict(zip(cols, map(float, parts[1:])))
info = next((l for l in lines if l.startswith('A rows alive at A end')), '')
print(info)
need = ['A_at_A_end', 'B_as_is', 'hide_B_born', 'hide_B_born+touched_from_A', '+revive_ended_A', '+revive_deleted_A',
        'B_as_is+revive_ended_A', 'B_as_is+revive_deleted_A']
missing = [a for a in need if a not in mae]
if missing:
    print('missing arms:', missing)
    sys.exit(1)
parts = [('B-born rows', 'B_as_is', 'hide_B_born'), ('A rows B changed', 'hide_B_born', 'hide_B_born+touched_from_A'),
         ('A rows B ended', 'hide_B_born+touched_from_A', '+revive_ended_A'), ('A rows B deleted', '+revive_ended_A', '+revive_deleted_A')]
print('\nphoto MAE (0-255): anchor = A_at_A_end, B = B_as_is')
print('%-26s' % 'view' + ''.join('%10s' % c for c in cols))
for arm in need:
    print('%-26s' % arm + ''.join('%10.2f' % mae[arm][c] for c in cols))
print('\nparts of the excess B - anchor (consecutive arms; share of the excess in brackets)')
print('%-26s' % 'part' + ''.join('%18s' % c for c in cols))
excess = {c: mae['B_as_is'][c] - mae['A_at_A_end'][c] for c in cols}
print('%-26s' % 'excess B - anchor' + ''.join('%18.2f' % excess[c] for c in cols))
for name, a, b in parts:
    row = ''
    for c in cols:
        d = mae[a][c] - mae[b][c]
        row += '%18s' % (f'{d:+.2f} ({100 * d / excess[c]:+.0f}%)' if abs(excess[c]) > 1e-9 else f'{d:+.2f}')
    print('%-26s' % name + row)
row = ''
for c in cols:
    d = mae['+revive_deleted_A'][c] - mae['A_at_A_end'][c]
    row += '%18s' % (f'{d:+.2f} ({100 * d / excess[c]:+.0f}%)' if abs(excess[c]) > 1e-9 else f'{d:+.2f}')
print('%-26s' % 'not explained' + row)
print('\nsingle effects on the B map (B_as_is minus the arm)')
for arm in ('B_as_is+revive_ended_A', 'B_as_is+revive_deleted_A'):
    print('%-26s' % arm.replace('B_as_is+', 'alone: ') + ''.join('%18.2f' % (mae['B_as_is'][c] - mae[arm][c]) for c in cols))
print('CF2_INTERPRET_DONE')
