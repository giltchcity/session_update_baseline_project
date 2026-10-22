"""The 4D map in hindsight (user 10-23: "如果我们的4DGS 能把这个东西补回去呢 ... 我们能补啊"): once a session is mapped, every
object state of our rounds (identity, state birth) gets the interval it existed in, filled back where our own
observations allow it and left as mapped where they do not.
  points   the state's rows in the round that holds the most of them (its best-observed version), their centres;
  frames   the dataset's frames of the session, every 3rd (the depth and the instance labels the mapping reads): a frame
           sees the state's place when at least 30 and a quarter of its points are in view and not hidden (the depth
           not more than 10 cm short of them); it sees the state THERE when half of those are within 5 cm and labelled
           with its identity, EMPTY when half of them have depth beyond them by 10 cm or are within 5 cm but labelled
           otherwise (a TV taken off a wall leaves the wall's depth);
  start    born in the session: the session's start if no frame saw its place empty before it was first seen there
           (it was there before it was seen: a change between the sessions, or an object seen late), else the first
           frame that saw it there (when it arrived in between is not known: as mapped);
           inherited from the session before: the session's start, unless its place was seen empty and never with it:
           then it is gone from the start (it changed between the sessions);
  end      the first frame that saw its place empty after the last that saw it there (as mapped: when it left in
           between is not known); else, if the rounds drop it for good, the first round without it; else the end.
One object is in one place: of one identity, a state inherited from the session before is superseded from the start
when a state of this session is filled back to the start; the others, in the order they were first seen, start no
earlier than the one before ends. A watched move (d1_4d/out/<session>_<unit>.json, track_d1.py) replaces its identity's
states from its own start on: its model at its first pose from when that pose is filled back to (the same rule, on the
posed model's points), then moving with the track, then held at its last pose. Writes OUT.json: per state its identity, birth, best round, rows, first /
last seen there, the empty sightings around them, start, end (session seconds); per track the start of its first pose.
  hindsight_intervals.py RUN SESSION TRACK_DIR OUT.json"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))                    # the repository's root
from update_layer.backends.game.game import CONFIGS  # noqa: E402
from update_layer.backends.game.snapshots import _replay  # noqa: E402
from update_layer.frames import FlatSession, synthetic_session  # noqa: E402
from update_layer.run import dataset_config  # noqa: E402
import yaml  # noqa: E402

run, s, tracks_dir, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
SCALE = float(yaml.safe_load(CONFIGS[dataset_config("synthetic")[1].name].read_text())["game"].get("scale", 1.0))
spec = synthetic_session(s); fs = FlatSession(spec); rel = np.array(fs.rel_ns) / 1e9
end_s = float(rel[-1])
rounds = run / "snapshots" / f"synthetic_{s}" / "rounds"
index = json.loads((rounds / "index.json").read_text())
stamps = [(r["stamp"] - spec.start_ns) / 1e9 for r in index]

# every object state's rows per round; its points in its best round
states, st = {}, None
for k in range(len(index)):
    st = _replay(rounds, k, st)
    _, _, uid, bits, ints = st
    ident, sb = ints["identity"], ints["state_birth"]
    obj = np.flatnonzero(ident != 0)
    keys = np.stack([ident[obj], sb[obj]], 1)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    keys, obj = keys[order], obj[order]
    cut = np.flatnonzero(np.any(np.diff(keys, axis=0) != 0, axis=1)) + 1
    for rows in np.split(obj, cut):
        key = (int(ident[rows[0]]), int(sb[rows[0]]))
        d = states.setdefault(key, dict(rounds=[], best=None))
        d["rounds"].append((k, len(rows)))
        if d["best"] is None or len(rows) > d["best"][1]:
            P = bits[rows].view(np.float32)[:, :3] / SCALE
            d["best"] = (k, len(rows), P[:: max(1, len(P) // 400)].astype(np.float64))
print(f"session {s}: {len(states)} object states over {len(index)} rounds", flush=True)

# watched moves: the tracked model at its first pose is one more 'state' of its identity
tracks = []
for f in sorted(tracks_dir.glob(f"{s}_*.json")):
    tr = json.loads(f.read_text())
    _, _, uid, bits, _ = _replay(rounds, tr["model_round"] - 1)
    M = bits[np.searchsorted(uid, np.array(tr["model_uid"]))].view(np.float32)[:, :3] / SCALE
    first = next(r for r in tr["track"] if "yaw" in r)
    c, a, tau = np.array(tr["centre"]), first["yaw"], np.array(first["tau"])
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    W = (M - c) @ R.T + c + tau
    tracks.append(dict(unit=tr["unit"], first_t=first["t"], file=f.name))
    states[("track", tr["unit"])] = dict(rounds=[], best=(None, len(W), W[:: max(1, len(W) // 400)]), unit=tr["unit"])

# one pass over the frames: when each state's place is seen with it, or empty
keys = sorted(states, key=lambda k: (isinstance(k[0], str), 0 if isinstance(k[0], str) else k[0], k[1]))
there = {key: [] for key in keys}
empty = {key: [] for key in keys}
for i in range(0, len(rel), 3):
    f = fs.load(i)
    T = np.linalg.inv(f.T_world_cam)
    for key in keys:
        P = states[key]["best"][2]
        ident = key[1] if key[0] == "track" else key[0]
        u, v, z = f.K.project(P @ T[:3, :3].T + T[:3, 3])
        ok = u >= 0
        if ok.sum() < 30:
            continue
        d, z, lab = f.depth[v[ok], u[ok]], z[ok], f.instance[v[ok], u[ok]]
        good = np.isfinite(d)
        d, z, lab = d[good], z[good], lab[good]
        free = d >= z - 0.10
        if free.sum() < max(30, 0.25 * len(P)):
            continue
        d, z, lab = d[free], z[free], lab[free]
        near = np.abs(d - z) < 0.05
        if (near & (lab == ident)).mean() >= 0.5:
            there[key].append(float(rel[i]))
        elif ((d > z + 0.10) | (near & (lab != ident))).mean() >= 0.5:
            empty[key].append(float(rel[i]))

result = []
for key in keys:
    if key[0] == "track":
        continue
    d = states[key]
    ident, birth = key
    born_s = (birth - spec.start_ns) / 1e9
    th, em = there[key], empty[key]
    first_there, last_there = (th[0], th[-1]) if th else (None, None)
    first_round_s = stamps[d["rounds"][0][0]]
    ref = first_there if first_there is not None else first_round_s
    empty_before = max((t for t in em if t < ref), default=None)
    empty_after = min((t for t in em if last_there is not None and t > last_there), default=None)
    last_round = d["rounds"][-1][0]
    if born_s < 0:                                             # inherited from the session before
        start = 0.0 if (th or not em) else None                # seen empty and never with it: gone before the session
    else:
        start = 0.0 if empty_before is None else (first_there if first_there is not None else first_round_s)
    if empty_after is not None:
        end = empty_after
    elif last_round < len(index) - 1:
        end = stamps[last_round + 1]
    else:
        end = end_s
    result.append(dict(identity=ident, birth=birth, born_s=born_s, best_round=d["best"][0] + 1, best_rows=d["best"][1],
                       rounds=[k + 1 for k, _ in d["rounds"]], first_there=first_there, last_there=last_there,
                       empty_before=empty_before, empty_after=empty_after, seen_empty=len(em),
                       start=None if start is None else float(start), end=float(end)))
# one object, one place
by_id = {}
for r in result:
    by_id.setdefault(r["identity"], []).append(r)
for ident, rs in by_id.items():
    if any(r["born_s"] >= 0 and r["start"] == 0.0 for r in rs):
        for r in rs:
            if r["born_s"] < 0 and r["start"] is not None:
                r["start"], r["why"] = None, "superseded from the start by a state of this session"
    seen = sorted((r for r in rs if r["start"] is not None and r["first_there"] is not None), key=lambda r: r["first_there"])
    for prev, r in zip(seen, seen[1:]):
        if r["first_there"] > prev["first_there"] and r["start"] < prev["end"]:   # first seen later: not before it left
            r["start"] = prev["end"]
for tr in tracks:
    em, th = empty[("track", tr["unit"])], there[("track", tr["unit"])]
    before = max((t for t in em if t < tr["first_t"]), default=None)
    tr["start"] = 0.0 if before is None else min([t for t in th if t > before] + [tr["first_t"]])
    tr["empty_before_first_pose"] = before
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(dict(run=str(run), session=s, end_s=end_s, rounds_s=stamps, states=result, tracks=tracks), indent=1))
for r in result:
    if r["best_rows"] >= 1000:
        span = "gone from the start" if r["start"] is None else f"[{r['start']:.1f}, {r['end']:.1f})"
        print(f"  unit {r['identity']:3d} born {r['born_s']:8.1f}  rounds {r['rounds'][0]}-{r['rounds'][-1]}  there "
              f"{r['first_there']}..{r['last_there']}  empty before {r['empty_before']} after {r['empty_after']} (x{r['seen_empty']})  -> {span}", flush=True)
for tr in tracks:
    print(f"  track unit {tr['unit']}: first pose at {tr['first_t']:.1f} s, filled back to {tr['start']:.1f} s (empty seen before: {tr['empty_before_first_pose']})", flush=True)
