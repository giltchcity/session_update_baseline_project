"""Where do GaME row 3 / row 4 lose G1? Final map samples of a stage, inside the G1 crop (eval/crops/real_S.json),
against the 1 cm reference mesh vertices (official G1 uses area samples and exact triangle distances; this is a
diagnostic approximation). False samples (> 5 cm from the reference) are split into: at the old site of an object
that changed into this stage (events.json old surface, 10 cm) / elsewhere. Missed reference (> 5 cm from the map):
at a new site of a changed object / elsewhere."""
import json, sys
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/jixian/Desktop/FT/session_update_baseline_project")
from update_layer.eval.scenelist import load_timeline
FT = "/home/jixian/Desktop/FT/"
stage = sys.argv[1]; runs = dict(a.split("=", 1) for a in sys.argv[2:])
crop = json.load(open(FT + f"eval/crops/real_{stage.upper()}.json"))
def rot(p, deg):
    t = np.radians(deg); R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]); q = p.astype(np.float64).copy(); q[:, [0, 2]] = q[:, [0, 2]] @ R.T; return q
def inside(p):
    q = rot(p, crop["yaw_deg"]); return np.all((q >= crop["lo"]) & (q <= crop["hi"]), axis=1)
ref = np.asarray(o3d.io.read_triangle_mesh(FT + crop["reference"]).vertices); ref = ref[inside(ref)]
events = [e for e in json.load(open(FT + "results/abc_eval_final_baseline/events.json")) if e["session"] == stage.upper()]
old = [np.load(e["old"]["surface"]) for e in events if e.get("old") and e["old"].get("surface")]
new = [np.load(e["new"]["surface"]) for e in events if e.get("new") and e["new"].get("surface")]
old_t = cKDTree(np.concatenate(old)) if old else None; new_t = cKDTree(np.concatenate(new)) if new else None
ref_t = cKDTree(ref)
maps = {}
for name, rel in runs.items():
    tl = load_timeline(FT + "runs/" + rel + f"/session_{stage}/timeline.pkl"); sc = tl.scene(tl.stamps()[-1])
    P = np.concatenate([sc.background] + [o.points for o in sc.objects]); P = P[inside(P)]
    k = np.unique(np.floor(P / 0.01).astype(np.int64), axis=0, return_index=True)[1]; maps[name] = P[k]
print(f"stage {stage.upper()}: reference vertices in crop {len(ref)}, changed events {len(events)}")
for name, P in maps.items():
    d = ref_t.query(P)[0]; false = d > 0.05
    at_old = old_t.query(P[false], distance_upper_bound=0.10)[0] < 0.10 if old_t else np.zeros(false.sum(), bool)
    m = cKDTree(P).query(ref, distance_upper_bound=0.05)[0]; miss = ~(m < 0.05)
    at_new = new_t.query(ref[miss], distance_upper_bound=0.10)[0] < 0.10 if new_t else np.zeros(miss.sum(), bool)
    print(f"  {name:10s} samples {len(P):7d} | P~{100*(1-false.mean()):5.1f}% false {false.sum():6d}: at old sites {at_old.sum():6d} "
          f"elsewhere {(~at_old).sum():6d} | R~{100*(1-miss.mean()):5.1f}% missed {miss.sum():6d}: at new sites {at_new.sum():6d} elsewhere {(~at_new).sum():6d}")
names = list(maps)
if len(names) == 2:
    a, b = maps[names[0]], maps[names[1]]
    for x, y, nx, ny in ((a, b, names[0], names[1]), (b, a, names[1], names[0])):
        only = cKDTree(y).query(x, distance_upper_bound=0.05)[0] >= 0.05
        f = ref_t.query(x[only])[0] > 0.05
        oo = old_t.query(x[only][f], distance_upper_bound=0.10)[0] < 0.10 if old_t else np.zeros(f.sum(), bool)
        print(f"  samples only in {nx} (none of {ny} within 5 cm): {only.sum()}, of them false {f.sum()} (at old sites {oo.sum()})")
