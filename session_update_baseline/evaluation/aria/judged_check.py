"""Was each ADT object judged right by our TSDF maps (user 11-03)? gt_check_adt.py's ground truth (only to check) on the
maps of a run (RUN_DIR, aria/run_tsdf.sh): per ADT object in view, whether our map at the end of A built it at all (a
surface inside its visit-A box higher than half a voxel, 2.5 cm, above its base: at 5 cm a thinner thing is one surface
with what it lies on, nothing to judge), and if so whether that surface is still there at the end of B (inside the
visit-A box, out of the visit-B box, the same height): a moved object should keep none, one that stayed should keep it.
Verdicts: moved and cleared = right, moved and kept = missed, stayed and kept = right, stayed and cleared = wrongly removed,
not built in A = below the map's resolution.
usage: judged_check.py ROOM RUN_DIR"""
import sys
from pathlib import Path
import numpy as np
room, run = sys.argv[1], sys.argv[2]
src = Path('/home/jixian/Desktop/FT/wt_takeover_runs/aria/gt_check_tsdf.py').read_text()
sys.argv = ['gt_check_tsdf.py', room, run]
exec(compile(src[:src.index('\nXA, objA, closedA, idsA')], 'gt_check_tsdf.py', 'exec'))
HALF_VOXEL = 0.025


def final(s):
    path = Path(run) / f'session_{s}/final.4dmap.zpk'; objs, bg = snapshot(path, len(zs.stamps(str(path))) - 1)
    return np.concatenate([bg] + [P for P, present, _ in objs.values() if present and len(P)])


XA, XB = final('a'), final('b')


def raised(X, uid, pose, other=None):
    """Vertices inside the object's box at pose, higher than half a voxel above its base (out of the box at `other`)."""
    lo, hi = BX[uid]; t, q = pose; loc = (X - t) @ rot(q)
    sel = np.all((loc >= lo + [0, HALF_VOXEL, 0]) & (loc <= hi), axis=1)
    if other is not None:
        t2, q2 = other; loc2 = (X - t2) @ rot(q2); sel &= ~np.all((loc2 >= lo) & (loc2 <= hi), axis=1)
    return int(sel.sum())


rows = []
for uid in sorted(set(PA) & set(BX)):
    name = names.get(str(uid), {}).get('instance_name', str(uid))
    if 'ApartmentEnv' in name: continue
    a, b = PA[uid], PB.get(uid)
    status = 'gone' if b is None else ('moved' if float(np.linalg.norm(a[0] - b[0])) > 0.01 else 'stayed')
    if in_view(a[0]) < 1: continue
    lo, hi = BX[uid]; height = float(hi[1] - lo[1]); vol = float(np.prod(hi - lo))
    other = b if status == 'moved' else None
    na, nb = raised(XA, uid, a, other), raised(XB, uid, a, other)
    if na < 5: verdict = 'not built (below 5 cm)'
    elif status in ('moved', 'gone'): verdict = 'right (cleared)' if nb <= 0.1 * na else f'missed (kept {nb / na:.2f})'
    else: verdict = 'right (kept)' if nb >= 0.5 * na else f'WRONGLY REMOVED (kept {nb / na:.2f})'
    rows.append((status, name, height, vol, na, nb, verdict))
for st in ('moved', 'gone', 'stayed'):
    rs = [r for r in rows if r[0] == st]
    if not rs: continue
    counts = {}
    for r in rs: counts[r[6].split(' (')[0]] = counts.get(r[6].split(' (')[0], 0) + 1
    print(f'{st}: {len(rs)} objects: {counts}')
    for r in sorted(rs, key=lambda r: -r[3]):
        if st == 'stayed' and r[6].startswith('right'): continue
        print(f'  {r[1][:28]:28s} height {r[2]:.3f} m vol {r[3]:.4f} m3  A {r[4]:5d}  B {r[5]:5d}  -> {r[6]}')
