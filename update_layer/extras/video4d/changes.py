"""The changes the 4D video marks (user 10-22: "主要要看到变化"), from the ground truth only (states.json, gt_syn/<state>.npy,
the dataset's camera), never from our map:
  move      an object's state ends (it starts to leave) and its next state in the session begins elsewhere (it has
            arrived; it slides for a few seconds between the two): red where it was, green where it is;
  between   in B, a state of A's end that is gone (ended 0.0) and the B state that replaced it: from 0;
  taken     a state that stops before the session's end with no later state: red only.
A mark is drawn from the move's start until the dataset camera has seen the change (the old place seen empty and the
new place seen with the object, from the arrival on) and the 4D map shows our first round after that (the 4D map of t
is our latest round at or before t), and 10 s more. Never seen: to the session's end. A green also ends when its object leaves again. Until the arrival the
outlines are dashed (still moving).
Kept: furniture, per object (not pillows, books, cups, ornaments, vases or tea sets; at least 0.55 m across in one of
its states; an unnamed object only when it stands on the floor, its bottom under 0.15 m). A pair whose floor rectangles
overlap by more than 70 % (IoU) is not a move (dropped); by 40-70 % it turned or shifted in place (the green alone). A
mark the video camera does not see at a moment it stands (under 25 % of its surface points in Isaac's depth of the
video's view) is not drawn: an unseen red drops its pair; an unseen green (a TV on the near wall) is drawn hidden (thin,
dashed) with its arrow. The outline is the convex hull of the state's surface points in the video's view.
Seeing (the dataset's frames, every 3rd): a frame sees a place when at least 30 and at least a quarter of the state's
points are in view and not hidden (the frame's depth not more than 10 cm short of them); empty when half of those have
depth beyond them by 10 cm, with the object when half are within 5 cm. Writes OUT.json: {session: {end_s, rounds_s,
marks, dropped}}.
  changes.py CAMERA.json ISAAC_DIR RUN OUT.json"""
import csv, json, sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))                    # the repository's root
from update_layer.frames import FlatSession, synthetic_session  # noqa: E402

cam = json.load(open(sys.argv[1])); isaac, run, out = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
GT = Path("/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/analysis/scratch_keep_20261022/fourd")
U = "/home/jixian/Desktop/FT/datasets/synthetic_ab/results/cvpr_surface_final_20260916/paper_aligned/EVAL_SCOPE_UNITS.csv"
unit_of = {int(m): int(u["core_members"].split(",")[0]) for u in csv.DictReader(open(U)) for m in u["all_members"].split(",")}
SMALL = ("pillow", "book", "cup", "ornament", "vase", "tea_set")
END = {"a": 139.5, "b": 118.5}                                  # the sessions' last frames (s)
HOLD = 10.0                                                     # s a mark stays after our map can show it
K = np.array(cam["K"], dtype=np.float64); w2c = np.array(cam["T_world_to_camera"], dtype=np.float64)


def video_view(P):
    c = P @ w2c[:3, :3].T + w2c[:3, 3]
    return np.stack([K[0, 0] * c[:, 0] / c[:, 2] + K[0, 2], K[1, 1] * c[:, 1] / c[:, 2] + K[1, 2]], 1), c[:, 2]


def seen_by_video(P, s, t):
    """Share of the points the video camera sees at session time t (Isaac's depth of the nearest rendered time)."""
    t = min(max(round(t * 4) / 4, 0.0), END[s])
    d = np.load(isaac / f"{s}_{t:07.2f}_depth.npz")["depth"]
    uv, z = video_view(P)
    u, v = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
    ok = (z > 0) & (u >= 0) & (u < d.shape[1]) & (v >= 0) & (v < d.shape[0])
    vis = np.zeros(len(P), bool)
    vis[ok] = z[ok] <= d[v[ok], u[ok]] + 0.05 + 0.01 * z[ok]
    return float(vis.mean())


def footprint(P):
    lo, hi = P.min(0), P.max(0)
    return lo, hi


def area(lo, hi):
    return float(np.prod(np.clip(hi[:2] - lo[:2], 0, None)))


def overlap(a, b):
    return area(np.maximum(a[0], b[0]), np.minimum(a[1], b[1]))


def sightings(s, queries):
    """For each query (P, since, want): the first time >= since that the dataset camera (every 3rd frame) sees P's place
    empty (want 'empty') or holding P (want 'there'); None if never. One pass over the frames."""
    fs = FlatSession(synthetic_session(s))
    rel = np.array(fs.rel_ns) / 1e9
    Q = [(P[:: max(1, len(P) // 400)], since, want) for P, since, want in queries]
    found = [None] * len(Q)
    for i in range(0, len(fs.ids), 3):
        todo = [q for q in range(len(Q)) if found[q] is None and Q[q][1] <= rel[i]]
        if not todo:
            continue
        f = fs.load(i)
        T = np.linalg.inv(f.T_world_cam)
        for q in todo:
            P, _, want = Q[q]
            u, v, z = f.K.project(P @ T[:3, :3].T + T[:3, 3])
            ok = u >= 0
            if ok.sum() < 30:
                continue
            d, z = f.depth[v[ok], u[ok]], z[ok]
            good = np.isfinite(d)
            d, z = d[good], z[good]
            free = d >= z - 0.10                                    # not hidden in front of the place
            if free.sum() < max(30, 0.25 * len(P)):
                continue
            if want == "empty" and (d[free] > z[free] + 0.10).mean() >= 0.5 or \
               want == "there" and (np.abs(d[free] - z[free]) < 0.05).mean() >= 0.5:
                found[q] = float(rel[i])
    return found


result = {}
for s in "ab":
    spec = synthetic_session(s)
    stamps = sorted((r["stamp"] - spec.start_ns) / 1e9 for r in json.loads((run / "snapshots" / f"synthetic_{s}" / "rounds" / "index.json").read_text()))
    after = lambda t: next((st for st in stamps if st >= t), None)               # our first round holding time t
    states = json.load(open(GT / "out" / f"syn_{s}" / "states.json"))
    by_obj = {}
    for k, v in states.items():
        by_obj.setdefault(v["object"], []).append((k, v))
    marks, dropped = [], []
    for o, L in sorted(by_obj.items()):
        L = sorted(L, key=lambda kv: (kv[1]["present"][0], kv[0][0] == s.upper()))
        pts = {k: np.load(GT / "gt_syn" / f"{k}.npy") for k, _ in L}
        name = L[0][1]["name"]
        size = max(max(np.ptp(P[:, 0]), np.ptp(P[:, 1])) for P in pts.values())
        if any(w in name for w in SMALL) or size < 0.55 or name.startswith("instance_") and min(P[:, 2].min() for P in pts.values()) > 0.15:
            continue
        for k, v in L:
            if v.get("ended") is not None:
                left = float(v["ended"])                                    # 0.0: between the sessions
            elif k[0] == s.upper() and v["present"][1] < END[s] - 0.5:
                left = float(v["present"][1])                               # taken away
            else:
                continue
            P = pts[k]
            lo, hi = footprint(P)
            nxt = [(k2, v2) for k2, v2 in L if k2[0] == s.upper() and k2 != k and v2["present"][0] >= left - 1e-6
                   and (left == 0 or v2["present"][0] > 0)]
            kind = "between" if left == 0 else ("move" if nxt else "taken")
            red = dict(state=k, P=P, at=END["a"] if k[0] == "A" and s == "b" else left - 0.5, sess="a" if k[0] == "A" and s == "b" else s)
            green = None
            if nxt:
                k2, v2 = nxt[0]
                gone = float(v2["ended"]) if v2.get("ended") else (float(v2["present"][1]) if v2["present"][1] < END[s] - 0.5 else None)
                green = dict(state=k2, P=pts[k2], gone=gone, arrived=float(v2["present"][0]), hidden=False)
            arrived = green["arrived"] if green else left
            info = dict(object=o, identity=unit_of.get(o), name=name, kind=kind, red=k, green=green["state"] if green else None)
            if green is not None:
                lo2, hi2 = footprint(green["P"])
                inter = overlap((lo, hi), (lo2, hi2))
                iou = inter / (area(lo, hi) + area(lo2, hi2) - inter)
                if iou > 0.7:
                    dropped.append(dict(info, why=f"not a move (IoU {iou:.2f})")); continue
                if iou > 0.4:
                    dropped.append(dict(info, why=f"red dropped: turned or shifted in place (IoU {iou:.2f})"))
                    red = None
            if red is not None and seen_by_video(red["P"], red["sess"], red["at"]) < 0.25:
                dropped.append(dict(info, why="red not seen by the video camera; the pair dropped")); continue
            if green is not None and seen_by_video(green["P"], s, min(green["arrived"] + 1.0, (green["gone"] or END[s]) - 0.25)) < 0.25:
                green["hidden"] = True
                if red is None:
                    dropped.append(dict(info, why="green alone and not seen by the video camera")); continue
            marks.append(dict(info, left=left, arrived=arrived, red_=red, green_=green))
    # when the dataset camera has seen each change (from the arrival on), and the 4D map shows our round after that
    queries = [(m[c + "_"]["P"], m["arrived"], w) for m in marks for c, w in (("red", "empty"), ("green", "there")) if m[c + "_"] is not None]
    found = iter(sightings(s, queries))
    for m in marks:
        red, green, left, arrived = m.pop("red_"), m.pop("green_"), m["left"], m["arrived"]
        t_red = next(found) if red is not None else arrived
        t_green = next(found) if green is not None else arrived
        seen = None if t_red is None or t_green is None else max(t_red, t_green)
        start = left
        until = END[s] if seen is None or after(seen) is None else max(arrived, after(seen)) + HOLD
        if green is not None and green["gone"] is not None:
            until = min(until, green["gone"])
        polys = {}
        for col, st in (("red", red), ("green", green)):
            if st is not None:
                uv, _ = video_view(st["P"][:: max(1, len(st["P"]) // 600)])
                polys[col] = cv2.convexHull(np.round(uv).astype(np.int32)).reshape(-1, 2).tolist()
        m.update(start=float(start), until=float(until), seen_red=t_red, seen_green=t_green, polygons=polys,
                 green_hidden=bool(green and green["hidden"]))
    result[s] = dict(end_s=END[s], rounds_s=stamps, marks=marks, dropped=dropped)
    print(f"session {s}: {len(marks)} marks, dropped {len(dropped)}")
    for m in sorted(marks, key=lambda m: m["start"]):
        print(f"   {m['kind']:7s} {m['name']:18s} {m['red'] if 'red' in m['polygons'] else '-':14s} -> "
              f"{(m['green'] or '-') + (' (hidden)' if m['green_hidden'] else ''):23s} start {m['start']:6.1f} left {m['left']:6.1f} "
              f"arrived {m['arrived']:6.1f}  seen {m['seen_red'] if m['seen_red'] is None else round(m['seen_red'], 1)}/"
              f"{m['seen_green'] if m['seen_green'] is None else round(m['seen_green'], 1)}  until {m['until']:6.1f}")
    for d in dropped:
        print(f"   dropped {d['name']:18s} {d['red']:14s} {d['why']}")
out.write_text(json.dumps(result, indent=1))
