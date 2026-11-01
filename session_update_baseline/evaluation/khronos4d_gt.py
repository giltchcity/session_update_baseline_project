#!/usr/bin/env python3
"""Ground truth of a multi-session chain (real A->B->C, synthetic A->B, Flat A->B) for Khronos's official ChangeF1, in
the format eval/threerscan_khronos_gt.py writes for 3RScan (analysis/khronos4d_change_gt_20261029/DRAFT.md):
  gt_dsg_consolidated.json: one object node per state of a thing; first_observed_ns = 0 for a state present from the
    chain's start, else the time it begins; last_observed_ns = the time it ends, else never (uint64 max). A state that
    ends stays in the graph (the official evaluator reads it as disappeared). A thing no change lists is one node over
    the sessions (its points = every session's state). A change between two sessions happens while the robot is away:
    at the later session's start. Node: AABB of the state's points, those points box-local (at most 2000), its class.
  gt_changes.csv: the states that appear or disappear, for the record (the evaluator reads only the graph).
  chain.json: the sessions, their start times (the clock of the method timelines: absolute, sessions in order) and the
    config, for the chain evaluation.
And configs/khronos/eval_chain_DATASET_local.yaml: eval_office_local.yaml with these files.
Sources (the published protocols' own ground truth, nothing annotated anew):
  real: per session the object F1's states (results/current_version_20260921/real_abc/objects/gt_S_latest/
    gt_payload.json: world points, physical id, the in-session times of the D2 / D1 changes), linked across sessions
    by the state protocol's D3 events (results/abc_eval_final_baseline/events.json: moved, added, removed).
  synthetic, flat: see synthetic() and flat().
A listed move makes a new state only beyond the 3RScan gate (displaced()).
usage: khronos4d_gt.py real|synthetic|flat
       khronos4d_gt.py DATASET perfect RUN_EXPORT OUT   (the ground truth as a method: each snapshot of RUN_EXPORT/
       visit_K/snapshots.jsonl, timeline_export format, lists the things present then, one node per thing; the upper
       bound of the chain evaluation)
"""
import copy, json, sys
from pathlib import Path

import numpy as np

FT = Path('/home/jixian/Desktop/FT')
TEMPLATE = FT / 'datasets/khronos/tesse_cd/office/ground_truth/gt_dsg_consolidated.json'
MAP_TEMPLATE = FT / 'wt_takeover_runs/3rscan/p3_ad408c8f/ours_290e110/visit_0/final.4dmap.zpk'   # layer layout only
MAX_POINTS, NEVER, GATE = 2000, 2**64 - 1, 0.10


def displaced(a, b):
    """A listed move makes a new state when the thing moved more than the gate (3RScan: mean point displacement
    under the annotated pose > 0.10 m; without poses, the displacement of the surface's centroid)."""
    return float(np.linalg.norm(np.asarray(a, float).mean(0) - np.asarray(b, float).mean(0))) > GATE


def real():
    """States of the real chain: [thing, semantic, first, last, points]."""
    G = FT / 'results/current_version_20260921/real_abc/objects'
    sessions = 'abc'
    payload = {s: json.load((G / f'gt_{s}_latest/gt_payload.json').open())['objects'] for s in sessions}
    starts = [json.load((G / f'gt_{s}_latest/GT_PROVENANCE.json').open())['base_ns'] for s in sessions]
    events = json.load((FT / 'results/abc_eval_final_baseline/events.json').open())
    d3 = {(e['transition'], e['object_id']): e['change_type'] for e in events if e['scope'] == 'D3'}
    states, open_ = [], {}                              # open_: thing -> its state still present at the session's end
    for k, s in enumerate(sessions):
        t0 = 0 if k == 0 else starts[k]
        here = sorted(payload[s], key=lambda o: o['first_observed_ns'][0])
        things = {o['physical_object_id'] for o in here} | (set(open_) if k else set())
        for i in sorted(things):
            mine = [o for o in here if o['physical_object_id'] == i]
            change = d3.get((sessions[k - 1:k + 1].upper(), i)) if k else None
            at_start = [o for o in mine if o['first_observed_ns'] == [0]]
            prev = open_.pop(i, None)
            if change == 'moved' and prev is not None and at_start and not displaced(prev[4][-1], at_start[0]['world_points']):
                change = None                           # listed, but moved no more than the gate: the same state
            if k and change is None:                    # no change listed: present in both, or in neither
                assert (prev is None) == (not at_start), (s, i, 'present in one session only with no change listed')
            if change == 'added': assert prev is None and at_start, (s, i)
            if change == 'removed': assert prev is not None and not at_start, (s, i)
            if prev is not None and change in ('moved', 'removed'):
                prev[3] = starts[k]; states.append(prev); prev = None
            for o in mine:
                first = t0 if o['first_observed_ns'] == [0] else o['first_observed_ns'][0]
                last = o['last_observed_ns'][0]
                pts = np.asarray(o['world_points'], float)
                if prev is not None and o['first_observed_ns'] == [0]:
                    prev[4].append(pts); cur = prev; prev = None   # the same state carried on
                else:
                    cur = [i, o['semantic'], first, NEVER, [pts]]
                if last == NEVER:
                    open_[i] = cur
                else:
                    cur[3] = last; states.append(cur)
    states += open_.values()
    return 'real', sessions, starts, states


def synthetic():
    """States of the synthetic chain, one thing per unit of the 29-unit scope, from the published harness's own ground
    truth (cvpr_eval_common: per frame each object's pose T_world_object and render_visible; the units' members'
    local surfaces paper_aligned/S/gt_ID.f32, as the harness's object evaluator places them per query): a unit is
    present while a member is visible; a new state begins when, after its members moved, the unit stands more than
    0.10 m (mean point displacement, the 3RScan criterion) from the state's pose, at the first frame it stands still
    again (the frames in motion belong to no state); between the sessions, at B's start."""
    import csv
    sys.path.insert(0, '/mnt/d/3Study/ETH/FT/Synthetic_ABC/scripts')
    from cvpr_eval_common import GT, IX, ORIGIN, OUT, CAT
    sessions = 'ab'
    starts = [ORIGIN[s] for s in sessions]
    units = list(csv.DictReader((OUT / 'paper_aligned/EVAL_SCOPE_UNITS.csv').open()))
    local = {(s, i): np.fromfile(OUT / f'paper_aligned/{s}/gt_{i}.f32', np.float32).reshape(-1, 3).astype(float)
             for s in sessions for u in units for i in map(int, u['core_members'].split(','))}
    def place(s, members, frame):
        T = GT[s]['T_world_object'][frame]
        return np.concatenate([local[s, i] @ T[IX[s][i], :3, :3].T + T[IX[s][i], :3, 3] for i in members])
    states = []
    for n, u in enumerate(units, 1):
        members = [int(i) for i in u['core_members'].split(',')]
        segments = []                                   # standing still and present: [session, first frame, last frame]
        for k, s in enumerate(sessions):
            T = GT[s]['T_world_object'][:, [IX[s][i] for i in members]]
            vis = GT[s]['render_visible'][:, [IX[s][i] for i in members]].any(1)
            still = np.r_[True, np.abs(np.diff(T, axis=0)).max(axis=(1, 2, 3)) <= 1e-6]
            for f in range(len(vis)):
                if vis[f] and segments and segments[-1][0] == k and segments[-1][2] == f - 1 and still[f]:
                    segments[-1][2] = f
                elif vis[f] and (still[f] or f + 1 == len(vis) or not np.abs(T[f + 1] - T[f]).max() > 1e-6):
                    segments.append([k, f, f])
        ns = lambda k, f: int(ORIGIN[sessions[k]] + round(float(GT[sessions[k]]['time_s'][f]) * 1e9))
        def begin(k, f): return 0 if (k, f) == (0, 0) else starts[k] if f == 0 else ns(k, f)
        def end(k, f):                                  # the first time after frame f of session k
            if f + 1 < len(GT[sessions[k]]['time_s']): return ns(k, f + 1)
            return starts[k + 1] if k + 1 < len(sessions) else NEVER
        cur = None
        for k, f0, f1 in segments:
            P = place(sessions[k], members, f0)
            if cur is not None and np.linalg.norm(P - cur[5], axis=1).mean() <= GATE:
                cur[3] = end(k, f1); cur[4].append(P); continue   # the same state (re-observed, moved less than the gate)
            if cur is not None: states.append(cur[:5])
            cur = [n, CAT[members[0]]['semantic_id'], begin(k, f0), end(k, f1), [P], P]
        if cur is not None: states.append(cur[:5])
    return 'synthetic', sessions, starts, states


def flat():
    """States of the Flat chain (run1 -> run2), from the object F1's ground truth (analysis/wm_flat_eval_20261028:
    gt_states_S.npz, each instance's pixels of the session's rendered frames back-projected, 1 cm voxels) and the
    dataset's change table (panoptic_mapping/flat_dataset/changes.txt: moved, new, removed, by instance name;
    groundtruth_labels.csv gives the ids). The structure (ids 0-5: ceiling, floor, walls, windows) is no object. An
    instance no change lists is one state over both sessions, also where one session's frames never show it."""
    import csv
    H = FT / 'runs/update_layer_game_T1_20261006/analysis/wm_flat_eval_20261028'
    L = FT / 'datasets/panoptic_mapping/flat_dataset'
    sessions = 'ab'
    starts = [json.load((FT / 'datasets/flat_ab/session.json').open())[s]['start_ns'] for s in sessions]
    ids = {r['Name']: int(r['InstanceID']) for r in csv.DictReader((L / 'groundtruth_labels.csv').open())}
    kind = {}
    for block in (L / 'changes.txt').read_text().split('#'):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if lines:
            for name in lines[1:]:
                kind[name] = lines[0].split()[0].lower()       # moved, new, removed
    change = {ids[n]: k for n, k in kind.items() if n in ids}     # SM_Office_Chair_seat_a, SM_Laptop_b: no id
    gt = {s: np.load(H / f'gt_states_{s}.npz', allow_pickle=True)['states'].item() for s in sessions}
    states = []
    for i in sorted((set(gt['a']) | set(gt['b'])) - set(range(6))):
        a, b, k = gt['a'].get(i), gt['b'].get(i), change.get(i)
        sem = (a or b)['semantic']
        if k == 'moved' and not displaced(a['points'], b['points']):
            k = None                                    # listed, but moved no more than the gate (SM_Chair_b, SM_Picture_c)
        if k == 'moved':
            assert a is not None and b is not None, i
            states += [[i, sem, 0, starts[1], [a['points']]], [i, sem, starts[1], NEVER, [b['points']]]]
        elif k == 'new':
            assert a is None and b is not None, i
            states.append([i, sem, starts[1], NEVER, [b['points']]])
        elif k == 'removed':
            assert a is not None and b is None, i
            states.append([i, sem, 0, starts[1], [a['points']]])
        else:
            states.append([i, sem, 0, NEVER, [x['points'] for x in (a, b) if x is not None]])
    return 'flat', sessions, starts, states


def write(name, sessions, starts, states):
    out = FT / 'eval/khronos4d_gt' / name; out.mkdir(parents=True, exist_ok=True)
    graph = json.load(TEMPLATE.open())
    template = next(n for n in graph['nodes'] if len(n['attributes']['mesh'].get('points', [])) > 0)
    nodes, changes = [], []
    rng = np.random.default_rng(0)
    for n, (i, sem, first, last, pts) in enumerate(sorted(states, key=lambda x: (x[0], x[2]))):
        pts = np.concatenate(pts)
        if len(pts) > MAX_POINTS: pts = pts[rng.choice(len(pts), MAX_POINTS, replace=False)]
        lo, hi = pts.min(0), pts.max(0); c = (lo + hi) / 2
        node = copy.deepcopy(template); a = node['attributes']; node['id'] = n
        a['name'] = f'O({n})'; a['semantic_label'] = int(sem); a['position'] = c.tolist()
        a['bounding_box'] = dict(dimensions=(hi - lo).tolist(), type='AABB', world_P_center=c.tolist(),
                                 world_R_center=dict(w=1.0, x=0.0, y=0.0, z=0.0))
        a['mesh']['points'] = (pts - c).tolist(); a['mesh']['colors'] = [dict(a=255, b=0, g=0, r=0)] * len(pts)
        a['first_observed_ns'] = [int(first)]; a['last_observed_ns'] = [int(last)]
        a['details'] = {'thing': [int(i)]}
        nodes.append(node)
        if first or last != NEVER: changes.append((f'O({n})', int(first), 0 if last == NEVER else int(last)))
    graph['nodes'] = nodes; graph['edges'] = []
    (out / 'gt_dsg_consolidated.json').write_text(json.dumps(graph))
    (out / 'gt_changes.csv').write_text('ObjectSymbol,AppearedAt,DisappearedAt\n' + ''.join(f'{s},{a},{d}\n' for s, a, d in changes))
    config = (FT / 'configs/khronos/eval_office_local.yaml').read_text()
    for f in ('gt_dsg_consolidated.json', 'gt_changes.csv'):
        config = config.replace(f'/home/jixian/Desktop/FT/datasets/khronos/tesse_cd/office/ground_truth/{f}', str(out / f))
    assert str(out) in config and config.count('tesse_cd/office') == 1   # the background file: mesh evaluation off
    yaml = FT / f'configs/khronos/eval_chain_{name}_local.yaml'
    yaml.write_text(f'# eval_office_local.yaml with the ground truth of the {name} chain (eval/khronos4d_gt.py)\n' + config)
    (out / 'chain.json').write_text(json.dumps(dict(dataset=name, sessions=list(sessions), starts_ns=[int(x) for x in starts],
                                                    config=str(yaml), template=str(MAP_TEMPLATE)), indent=1) + '\n')
    print(name, f'{len(states)} states of {len({x[0] for x in states})} things, {len(changes)} appear/disappear')


def perfect(name, export, dest):
    """The ground truth as a method's timeline: at each of EXPORT's snapshot stamps, the things present then, one node
    per thing, each declaring its own changes as a method does (eval/timeline_complete.py): first_observed_ns = when
    its first state began (0: present from the start), session_mobility_changes = how many of its states came after
    the first, observation_first_stamp_ns = when the current one began."""
    graph = json.load((FT / 'eval/khronos4d_gt' / name / 'gt_dsg_consolidated.json').open())
    things = {}
    for n in graph['nodes']:
        a = n['attributes']
        things.setdefault(a['details']['thing'][0], []).append(
            (a['first_observed_ns'][0], a['last_observed_ns'][0], a['semantic_label'], np.asarray(a['mesh']['points']) + a['position']))
    for v in sorted(Path(export).glob('visit_*')):
        out = Path(dest) / v.name; out.mkdir(parents=True, exist_ok=True)
        with (v / 'snapshots.jsonl').open() as f, (out / 'snapshots.jsonl').open('w') as g:
            for line in f:
                snap = json.loads(line); t = snap['stamp_ns']; rows = []
                for i, sts in things.items():
                    sts = sorted(sts, key=lambda x: x[0])
                    j = next((j for j, (first, last, _, _) in enumerate(sts) if first <= t < last), None)
                    if j is None:
                        continue
                    first, _, sem, P = sts[j]
                    rows.append({'id': str(i), 'semantic': int(sem), 'first_observed_ns': [int(sts[0][0])], 'last_observed_ns': [NEVER],
                                 'present': True, 'mesh_vertices': len(P), 'observation_first_ns': 0, 'observation_last_ns': 0,
                                 'details': {'instance_id': [i], 'session_mobility_changes': [j], 'observation_first_stamp_ns': [int(first)]},
                                 'centroid': P.mean(0).tolist(), 'surface': P.tolist(), 'trajectory_timestamps': [],
                                 'trajectory_positions': [], 'bbox_min': P.min(0).tolist(), 'bbox_max': P.max(0).tolist()})
                g.write(json.dumps({'index': snap['index'], 'stamp_ns': t, 'objects': rows}) + '\n')
        print(name, 'perfect', v.name, flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[2] == 'perfect':
        perfect(sys.argv[1], sys.argv[3], sys.argv[4])
    else:
        write(*{'real': real, 'synthetic': synthetic, 'flat': flat}[sys.argv[1]]())
