"""Ground truth of one visit mapped alone (a single run, as Khronos maps: from an empty map) for the TESSE table's metrics
(user 11-01: background, static objects, dynamic objects, changes, each Pre / Rec / F1 by Khronos's official evaluator),
into eval/khronos4d_gt/KIND_single_S:
  gt_dsg_consolidated.json: the chain's ground truth (eval/khronos4d_gt.py) with the visit's moving things
    (dynamic_gt_eval.py, eval/khronos4d_gt/KIND_dynamic_S), cut to the visit: the states present at some time of the
    visit; a state present at the visit's start is present from the run's start (first observed 0: the run starts
    there, so a change between two visits is not a change of this run); a state that ends after the visit never ends.
    The moving things as in TESSE's ground truth: no class (4294967295), no mesh, observed over their trajectory.
  gt_background.ply: the reference surface G1 scores against: real the vertices of the visit's reference mesh
    (results/abc_eval_v2/geometry/S_reference_1cm.ply), synthetic the causal reference up to the visit's end
    (datasets/synthetic_ab/evaluation_complete_20260917/causal_reference, Isaac).
  gt_changes.csv: the visit's appearing and disappearing states, for the record (the evaluator reads only the graph);
and the config eval/khronos4d_gt_KIND_single_S.yaml: configs/khronos/eval_chain_KIND_local.yaml with these files.
usage: single_run_official.py real|synthetic SESSION"""
import csv, glob, json, os, sys
from pathlib import Path
import numpy as np
FT = Path('/home/jixian/Desktop/FT'); R = FT / 'wt_takeover_runs'
NEVER, NO_CLASS = 2**64 - 1, 4294967295
BANK = FT / 'datasets/synthetic_ab/evaluation_complete_20260917/causal_reference'


def window(kind, session):
    chain = json.loads((FT / f'eval/khronos4d_gt/{kind}/chain.json').read_text())
    i = chain['sessions'].index(session); starts = chain['starts_ns']
    return starts[i], starts[i + 1] if i + 1 < len(starts) else NEVER


def background(kind, session, start, end):
    if kind == 'real':
        raw = (FT / f'results/abc_eval_v2/geometry/{session.upper()}_reference_1cm.ply').read_bytes()
        head = raw[:raw.index(b'end_header\n') + 11].decode()
        n = int(head.split('element vertex ')[1].split()[0])
        assert 'property double x\nproperty double y\nproperty double z\nproperty double nx\nproperty double ny\nproperty double nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face' in head, head
        V = np.frombuffer(raw, np.dtype([('p', '<f8', 3), ('n', '<f8', 3), ('c', 'u1', 3)]), n, len(head))['p']
    else:
        parts = [np.load(f) for f in glob.glob(str(BANK / '*.npz'))]
        V = np.concatenate([z['points'][z['first_stamp_ns'] < end] for z in parts])
    return np.ascontiguousarray(V, np.float32)


def build(kind, session):
    start, end = window(kind, session)
    src = FT / f'eval/khronos4d_gt/{kind}_dynamic_{session}/gt_dsg_consolidated.json'; d = json.loads(src.read_text())
    keep, changes, counts = [], [], dict(static=0, dropped=0, moving=0, appeared=0, disappeared=0)
    for node in d['nodes']:
        if node['layer'] != 2: keep.append(node); continue
        a = node['attributes']
        if a.get('trajectory_timestamps'):
            a['semantic_label'] = NO_CLASS; keep.append(node); counts['moving'] += 1; continue
        first, last = [], []
        for f, l in zip(a['first_observed_ns'], a['last_observed_ns']):
            if f >= end or l <= start: continue                     # not present in this visit
            first.append(0 if f <= start else f); last.append(NEVER if l >= end else l)
        if not first: counts['dropped'] += 1; continue
        a['first_observed_ns'], a['last_observed_ns'] = first, last; keep.append(node)
        counts['static'] += 1; counts['appeared'] += sum(f > 0 for f in first); counts['disappeared'] += sum(l < NEVER for l in last)
        changes += [(node['id'], f, 0 if l == NEVER else l) for f, l in zip(first, last) if f > 0 or l < NEVER]
    ids = {n['id'] for n in keep}
    d['nodes'] = keep; d['edges'] = [e for e in d['edges'] if e['source'] in ids and e['target'] in ids]
    out = FT / f'eval/khronos4d_gt/{kind}_single_{session}'; out.mkdir(parents=True, exist_ok=True)
    (out / 'gt_dsg_consolidated.json').write_text(json.dumps(d))
    with (out / 'gt_changes.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['ObjectSymbol', 'AppearedAt', 'DisappearedAt'])
        w.writerows([(f'O({i})', a, b) for i, a, b in changes])
    V = background(kind, session, start, end)
    with (out / 'gt_background.ply').open('wb') as f:
        f.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\n'
                f'property float z\nend_header\n'.encode()); f.write(V.tobytes())
    base = (FT / f'configs/khronos/eval_chain_{kind}_local.yaml').read_text(); text = base
    for key, path in (('ground_truth_dsg_file', out / 'gt_dsg_consolidated.json'), ('ground_truth_background_file', out / 'gt_background.ply'),
                      ('gt_changes_file', out / 'gt_changes.csv')):
        line = next(l for l in text.splitlines() if l.startswith(key + ':'))
        text = text.replace(line, line.split('"')[0] + f'"{path}"' + line.split('"', 2)[2])
    assert text.count(str(out)) == 3, text[:600]
    (R / f'eval/khronos4d_gt_{kind}_single_{session}.yaml').write_text(f'# {Path(__file__).name}: the visit {session} alone\n' + text)
    print(f'{kind} {session}: visit [{start}, {end}), states {counts}, background points {len(V)}, changes {len(changes)} -> {out}')


if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2])
