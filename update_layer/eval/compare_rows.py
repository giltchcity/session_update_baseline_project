"""Real A/B/C comparison table of update_layer rows against the TSDF version (L2_FINAL2), as one CSV.

  python -m update_layer.eval.compare_rows [real|synthetic] OUT.csv LABEL=RUN_DIR [LABEL=RUN_DIR ...]

Per session and metric one line; a missing value is written as "missing: <reason>". Row metrics are read from
RUN_DIR/eval_real (eval_row.sh); the TSDF column from the L2_FINAL2 evaluation (same G1 tool, same change and
ghost harness, official object evaluator).
  G1 / G2 @5 cm (P, R, F1)  surfel export (geometry/<s>/G1.json, G2.json) and median-depth TSDF export
                            (G1_tsdf.json, G2_tsdf.json); TSDF version: its mesh (geometry_viewer_G1.json)
  object P/R/F1 (Table 1)   the published Table 1 variant for rows and TSDF alike: duration-weighted, NR-corrected
                            (harness object_weighted_nr.py on objects/<s>/post/native_unique_online_rows.csv ->
                            objects/<s>/weighted_nr/OBJECT_WEIGHTED_NR.json; TSDF: SUMMARY object.duration_weighted_nr);
                            plus the raw native summary (objects/<s>/post/NATIVE_ONLINE_SUMMARY.json; TSDF: SUMMARY
                            object_raw_native) on separate lines. Notes on comparability in the note column.
  D2, D3 TP/FP/FN           changes/<s>/STATE_CHANGE_SUMMARY.csv (object_layer, combined); TSDF: SUMMARY raw protocol_v1
  ghost %                   ghost/<s>/GHOST.json (old-site full map); TSDF: SUMMARY
  retention                 retention_<s>.json (unobserved inherited kept, absent residue, present kept)
  D1 person P/R/F1          d1_person/DYNAMIC.csv; TSDF: published abc_eval_final_baseline (91.9/13.2/22.2, ABC)
Synthetic (A, A->B) from RUN_DIR/eval_syn: geometry/FINAL_GEOMETRY.csv (Mesh P/R/F1 @5/10/20 cm, MAD),
STABLE_ACCUMULATION.csv (retention of A-correct surface), ours/state_eval GROUP_STATE_SUMMARY.csv (D1 visible, D2,
D3, mixed visibility, visibility change), objects/online/ONLINE_OBJECT_SUMMARY.csv, d1/dynamics/
LIVE_DYNAMICS_SUMMARY.csv, retention_b.json; TSDF: full_eval/FULL_RESULTS_ours.csv and full_eval/dynamics/
LIVE_DYNAMICS_SUMMARY.csv.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

TSDF = Path("/home/jixian/Desktop/FT/session_update_baseline/runs/rewrite_goal_L2_FINAL2_20261006")
TSDF_NAME = "rewrite_goal_L2_FINAL2_20261006"


def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _pct(x):
    return None if x is None else round(100.0 * float(x), 2)


def row_metrics(run: Path) -> dict:
    E = run / "eval_real"
    out = {}
    for s in "abc":
        S = s.upper()
        for tag, name in (("surfel", "G1.json"), ("median TSDF", "G1_tsdf.json")):
            d = _json(E / "geometry" / s / name)
            for k in ("P_05cm", "R_05cm", "F1_05cm"):
                out[(S, f"G1@5 {k[0] if k != 'F1_05cm' else 'F1'} ({tag})")] = _pct(d[k]) if d else None
        for tag, name in (("surfel", "G2.json"), ("median TSDF", "G2_tsdf.json")):
            d = _json(E / "geometry" / s / name)
            out[(S, f"G2@5 F1 ({tag})")] = _pct(d["F1_05cm"]) if d else None
        d = _json(E / "objects" / s / "post" / "NATIVE_ONLINE_SUMMARY.json")
        for k, m in (("P", "ObjectPrecision"), ("R", "ObjectRecall"), ("F1", "ObjectF1")):
            out[(S, f"object {k} (raw native, unweighted)")] = _pct(d["summary"][m]["mean"]) if d else None
        d = _json(E / "objects" / s / "weighted_nr" / "OBJECT_WEIGHTED_NR.json")
        for k in ("P", "R", "F1"):
            out[(S, f"object {k} (Table 1: duration-weighted, NR)")] = d["duration_weighted_nr"][k] if d else None
        f = E / "changes" / s / "STATE_CHANGE_SUMMARY.csv"
        rows = list(csv.DictReader(f.open())) if f.exists() else []
        for grp in [f"D2_{S}"] + ([f"D3_{'AB' if s == 'b' else 'BC'}"] if s != "a" else []):
            if True:
                r = next((r for r in rows if r["group"] == grp and r["component"] == "combined"
                          and r["protocol"] == "object_layer"), None)
                out[(S, f"{grp[:2]} TP/FP/FN")] = f"{r['TP']}/{r['FP']}/{r['FN']}" if r else None
        if s != "a":
            g = _json(E / "ghost" / s / "GHOST.json")
            out[(S, "ghost % (old-site full map)")] = (None if g is None else g.get("old_site_full_map_ghost_pct")
                                                       if g.get("denominator") else "n/a: no inherited old-site probes")
            rc = _json(E / f"retention_correct_{s}.json")
            v = next(iter(rc["maps"].values()))["5cm"]["retained_fraction_of_prev"] if rc else None
            out[(S, "retention of the previous session's correct surface @5cm (real)")] = _pct(v) if v is not None else None
            rt = _json(E / f"retention_{s}.json")
            for k in ("unobserved_retention", "absent_residue", "present_kept"):
                out[(S, f"retention {k}")] = _pct(rt[k]) if rt else None
        f = E / "d1_person" / "DYNAMIC.csv"
        rows = list(csv.DictReader(f.open())) if f.exists() else []
        r = next((r for r in rows if r["session"] == S and r["category"] == "person"), None)
        out[(S, "D1 person P/R/F1")] = (f"{_pct(r['Precision'] or 0)}/{_pct(r['Recall'])}/{_pct(r['F1'])}" if r else None)
    f = E / "d1_person" / "DYNAMIC.csv"
    rows = list(csv.DictReader(f.open())) if f.exists() else []
    r = next((r for r in rows if r["session"] == "ABC" and r["category"] == "person"), None)
    out[("ABC", "D1 person P/R/F1")] = f"{_pct(r['Precision'] or 0)}/{_pct(r['Recall'])}/{_pct(r['F1'])}" if r else None
    return out


def tsdf_metrics() -> dict:
    out = {}
    for s in "abc":
        S = s.upper()
        base = TSDF / f"real_{s}_evaluation" / "real_abc"
        g = _json(base / "geometry" / f"{TSDF_NAME}_{s}" / "geometry_viewer_G1.json")
        for k in ("P_05cm", "R_05cm", "F1_05cm"):
            label = k[0] if k != "F1_05cm" else "F1"
            out[(S, f"G1@5 {label} (surfel)")] = _pct(g[k]) if g else None
            out[(S, f"G1@5 {label} (median TSDF)")] = "n/a: the TSDF version's own mesh (row above)"
        sm = _json(base / "summary" / f"{TSDF_NAME}_{s}" / "SUMMARY.json") or {}
        obj = sm.get("object", {})
        raw, nr = sm.get("object_raw_native"), obj.get("duration_weighted_nr", {})
        for j, k in enumerate(("P", "R", "F1")):
            out[(S, f"object {k} (raw native, unweighted)")] = round(raw[j], 2) if raw else None
            out[(S, f"object {k} (Table 1: duration-weighted, NR)")] = nr.get(k)
        ch = sm.get("change", {})
        for grp, key in (("D2", "D2"), ("D3", "D3")):
            c = ch.get(key, {}).get("raw_protocol_v1") or ch.get(key, {}).get("raw_protocol_v1_excl_I2")
            if c:
                out[(S, f"{grp} TP/FP/FN")] = f"{c['TP']}/{c['FP']}/{c['FN']}"
        gh = sm.get("ghost", {})
        if gh:
            out[(S, "ghost % (old-site full map)")] = gh.get("old_site_full_map_ghost_pct", gh.get("pct"))
    for S, pair in (("B", "ab"), ("C", "bc")):
        rc = _json(RETENTION_REF / f"retention_correct_tsdf_{pair}.json")
        out[(S, "retention of the previous session's correct surface @5cm (real)")] = (
            _pct(rc["maps"]["TSDF L2_FINAL2"]["5cm"]["retained_fraction_of_prev"]) if rc else None)
    out[("ABC", "D1 person P/R/F1")] = "91.9/13.2/22.2 (published abc_eval_final_baseline)"
    return out


TSDF_SYN = TSDF / "synthetic" / "full_eval" / "FULL_RESULTS_ours.csv"
# eval/retention_correct.py run unchanged on L2_FINAL2's final meshes (real_<s>_evaluation .../geometry_viewer.ply)
RETENTION_REF = Path("/home/jixian/Desktop/FT/runs/update_layer_game_T1_20261006/analysis")


def _csv(p: Path):
    return list(csv.DictReader(p.open())) if p.exists() else []


def syn_row_metrics(run: Path) -> dict:
    """Synthetic A / A->B (session a / b of the chain) from RUN_DIR/eval_syn (eval_row.sh)."""
    E = run / "eval_syn"
    out = {}
    stage = {"a": "A", "b": "A->B"}
    for r in _csv(E / "geometry" / "FINAL_GEOMETRY.csv"):
        if r["scope"] != "all":
            continue
        t = int(round(float(r["threshold_m"]) * 100))
        st = stage.get(r["session"])
        if st and t in (5, 10, 20):
            out[(st, f"Mesh F1@{t}cm")] = _pct(r["F1"])
            if t == 5:
                out[(st, "Mesh Precision@5cm")] = _pct(r["Precision"])
                out[(st, "Observed GT coverage (Recall)@5cm")] = _pct(r["Recall"])
                out[(st, "MAD (cm)")] = round(100 * float(r["MAD_m"]), 4)
    for r in _csv(E / "geometry" / "STABLE_ACCUMULATION.csv"):
        if abs(float(r["threshold_m"]) - 0.05) < 1e-9:
            out[("A->B", "Retention of A-correct surface in B @5cm")] = _pct(r["retained_fraction_of_A"])
    kinds = {"D1_visible": "D1 visible", "D2_hidden": "D2", "D3": "D3", "mixed_visibility": "mixed visibility",
             "visibility_change": "visibility change"}
    for r in _csv(E / "ours" / "state_eval" / "group_scores" / "GROUP_STATE_SUMMARY.csv"):
        st, k = stage.get(r["session"]), kinds.get(r["kind"])
        if st and k:
            out[(st, f"{k} State TP/FP/FN")] = f"{r['TP']}/{r['FP']}/{r['FN']}"
            out[(st, f"{k} State F1")] = _pct(r["F1"]) if r["F1"] else None
    for r in _csv(E / "objects" / "online" / "ONLINE_OBJECT_SUMMARY.csv"):
        st = stage.get(r["session"])
        if st:
            out[(st, "Online Object F1 full duration")] = _pct(r["F1"])
            out[(st, "Online Object F1 compat (non-empty output)")] = _pct(r["compatibility_nonempty_output_F1"])
            out[(st, "Final Object TP/FP/FN")] = f"{r['final_TP']}/{r['final_FP']}/{r['final_FN']}"
    out.update(_live_dynamics(E / "d1" / "dynamics" / "LIVE_DYNAMICS_SUMMARY.csv"))
    rt = _json(E / "retention_b.json")
    for k in ("unobserved_retention", "absent_residue", "present_kept"):
        out[("A->B", f"retention {k}")] = _pct(rt[k]) if rt else None
    return out


def _live_dynamics(path: Path) -> dict:
    """evaluate_live_dynamics.py LIVE_DYNAMICS_SUMMARY.csv: time P/R/F1 of visible motion, trajectory coverage."""
    out = {}
    stage = {"a": "A", "b": "A->B"}
    for r in _csv(path):
        st = stage.get(r["session"])
        if st and r.get("method", "ours") == "ours":
            out[(st, "D1 live dynamics time P/R/F1")] = "/".join(
                str(_pct(r[k]) if r[k] not in ("", None) else "-") for k in ("time_Precision", "time_Recall", "time_F1"))
            out[(st, "D1 trajectory coverage")] = _pct(r["trajectory_coverage"]) if r["trajectory_coverage"] else None
    return out


def syn_tsdf_metrics() -> dict:
    out = {}
    names = {"Online Object F1, full duration incl. initial empty map (%)": "Online Object F1 full duration",
             "Online Object F1, compat (non-empty output duration) (%)": "Online Object F1 compat (non-empty output)",
             "Final Object TP/FP/FN": "Final Object TP/FP/FN", "Mesh F1@5cm (%)": "Mesh F1@5cm",
             "Mesh Precision@5cm (%)": "Mesh Precision@5cm",
             "Observed GT coverage (Recall)@5cm (%)": "Observed GT coverage (Recall)@5cm", "MAD (cm)": "MAD (cm)",
             "Mesh F1@10cm (%)": "Mesh F1@10cm", "Mesh F1@20cm (%)": "Mesh F1@20cm",
             "Retention of A-correct surface in B @5cm (%)": "Retention of A-correct surface in B @5cm",
             "D2 State TP/FP/FN": "D2 State TP/FP/FN", "D2 State F1 (%)": "D2 State F1",
             "D3 State TP/FP/FN": "D3 State TP/FP/FN", "D3 State F1 (%)": "D3 State F1",
             "D1 visible State TP/FP/FN": "D1 visible State TP/FP/FN", "D1 visible State F1 (%)": "D1 visible State F1",
             "mixed visibility State TP/FP/FN": "mixed visibility State TP/FP/FN",
             "mixed visibility State F1 (%)": "mixed visibility State F1",
             "visibility change State TP/FP/FN": "visibility change State TP/FP/FN",
             "visibility change State F1 (%)": "visibility change State F1"}
    for r in _csv(TSDF_SYN):
        k = names.get(r["metric"])
        st = r["stage"].replace("→", "->")
        if k:
            v = r["new"]
            try:
                v = round(float(v), 2)
            except ValueError:
                pass
            out[(st, k)] = v
    out.update(_live_dynamics(TSDF_SYN.parent / "dynamics" / "LIVE_DYNAMICS_SUMMARY.csv"))
    return out


NOTES = {"real": (
    ("object F1 (Table 1", "TSDF value = L2_FINAL2 SUMMARY object.duration_weighted_nr (the variant of Table 1 and of the "
     "novelty numbers 92.92/98.43/98.55, which are final10_20261005 'Object F1', an earlier TSDF build; its raw native F1 "
     "88.20/83.91/76.83). Checked 2026-10-06 13:03: (1) same variant for rows and TSDF (harness object_weighted_nr.py; "
     "NR = 0 for update_layer exports: export_obj4d emits identities > 0 with points only, no person/empty nodes, "
     "checked on smoke3 A); (2) identity input: both read datasets/local_ab/instance_labels/session_<s> (L2_FINAL2 "
     "control/command.txt instance_dir) and the same ObjectEvaluator associates by geometry; the TSDF objects pass through "
     "Khronos' extraction (volume gates, observations), the GaME objects are the Gaussians' I1 identity: same input, "
     "different object formation; (3) last_observed = UINT64_MAX on every node, but one map per snapshot holding only "
     "the objects shown then: the evaluator loads exactly the snapshot's nodes (smoke3 A: 15/15 queries NumDsgLoaded = "
     "snapshot nodes), so ended objects are not carried into later queries"),
    ("retention of the previous session's correct surface", "real-data definition (eval/retention_correct.py): stable "
     "reference = previous session's G1 reference within 5 cm of the current one; retained = share of what the previous "
     "map covered that the current map still covers; rows 3/4: median-depth TSDF exports, row 1: surfel exports; TSDF: "
     "its final meshes. Not the synthetic 'Retention of A-correct surface' (STABLE_ACCUMULATION, the novelty's 99.75)"),
    ("object F1 (raw native", "raw native counts every present node: the TSDF maps hold person / empty nodes "
     "(NR nodes A 92, B 305, C 382) as hallucinations, update_layer exports none: not like for like; use the Table 1 lines"),
)}


def main():
    args = sys.argv[1:]
    mode = "real"
    if args and args[0] in ("real", "synthetic"):
        mode, args = args[0], args[1:]
    out = Path(args[0])
    cols = [a.split("=", 1) for a in args[1:]]
    if mode == "synthetic":
        data = {label: syn_row_metrics(Path(p)) for label, p in cols}
        tsdf = syn_tsdf_metrics()
    else:
        data = {label: row_metrics(Path(p)) for label, p in cols}
        tsdf = tsdf_metrics()
    keys = []
    for d in list(data.values()) + [tsdf]:
        for k in d:
            if k not in keys:
                keys.append(k)
    keys.sort(key=lambda k: ("ABC" in k[0], k[0], k[1]))
    tsdf_label = "TSDF L2_FINAL2" if mode == "real" else "TSDF L2_FINAL2 (full_eval)"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "metric"] + [label for label, _ in cols] + [tsdf_label, "note"])
        for k in keys:
            vals = []
            for label, _ in cols:
                v = data[label].get(k)
                vals.append("missing: not produced by this run" if v is None else v)
            v = tsdf.get(k)
            vals.append("missing: not in the TSDF evaluation" if v is None else v)
            w.writerow([k[0], k[1]] + vals + [next((n for pre, n in NOTES.get(mode, ()) if k[1].startswith(pre)), "")])
    print(out, len(keys), "lines")


if __name__ == "__main__":
    main()
