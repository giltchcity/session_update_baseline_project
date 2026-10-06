"""Real A/B/C comparison table of update_layer rows against the TSDF version (L2_FINAL2), as one CSV.

  python -m update_layer.eval.compare_rows OUT.csv LABEL=RUN_DIR [LABEL=RUN_DIR ...]

Per session and metric one line; a missing value is written as "missing: <reason>". Row metrics are read from
RUN_DIR/eval_real (eval_row.sh); the TSDF column from the L2_FINAL2 evaluation (same G1 tool, same change and
ghost harness, official object evaluator).
  G1 / G2 @5 cm (P, R, F1)  surfel export (geometry/<s>/G1.json, G2.json) and median-depth TSDF export
                            (G1_tsdf.json, G2_tsdf.json); TSDF version: its mesh (geometry_viewer_G1.json)
  object P/R/F1             official native online, unweighted (objects/<s>/official/NATIVE_ONLINE_SUMMARY.json);
                            TSDF: SUMMARY raw native unweighted and the duration-weighted NR (Table 1)
  D2, D3 TP/FP/FN           changes/<s>/STATE_CHANGE_SUMMARY.csv (object_layer, combined); TSDF: SUMMARY raw protocol_v1
  ghost %                   ghost/<s>/GHOST.json (old-site full map); TSDF: SUMMARY
  retention                 retention_<s>.json (unobserved inherited kept, absent residue, present kept)
  D1 person P/R/F1          d1_person/DYNAMIC.csv; TSDF: published abc_eval_final_baseline (91.9/13.2/22.2, ABC)
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
        d = _json(E / "objects" / s / "official" / "NATIVE_ONLINE_SUMMARY.json")
        for k, m in (("P", "ObjectPrecision"), ("R", "ObjectRecall"), ("F1", "ObjectF1")):
            out[(S, f"object {k} (native online unweighted)")] = _pct(d["summary"][m]["mean"]) if d else None
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
            out[(S, f"object {k} (native online unweighted)")] = (
                f"{round(raw[j], 2)} (raw native; Table 1 weighted NR {nr.get(k)})" if raw else None)
        ch = sm.get("change", {})
        for grp, key in (("D2", "D2"), ("D3", "D3")):
            c = ch.get(key, {}).get("raw_protocol_v1") or ch.get(key, {}).get("raw_protocol_v1_excl_I2")
            if c:
                out[(S, f"{grp} TP/FP/FN")] = f"{c['TP']}/{c['FP']}/{c['FN']}"
        gh = sm.get("ghost", {})
        if gh:
            out[(S, "ghost % (old-site full map)")] = gh.get("old_site_full_map_ghost_pct", gh.get("pct"))
    out[("ABC", "D1 person P/R/F1")] = "91.9/13.2/22.2 (published abc_eval_final_baseline)"
    return out


def main():
    out = Path(sys.argv[1])
    cols = [a.split("=", 1) for a in sys.argv[2:]]
    data = {label: row_metrics(Path(p)) for label, p in cols}
    tsdf = tsdf_metrics()
    keys = []
    for d in list(data.values()) + [tsdf]:
        for k in d:
            if k not in keys:
                keys.append(k)
    keys.sort(key=lambda k: ("ABC" in k[0], k[0], k[1]))
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "metric"] + [label for label, _ in cols] + ["TSDF L2_FINAL2"])
        for k in keys:
            vals = []
            for label, _ in cols:
                v = data[label].get(k)
                vals.append("missing: not produced by this run" if v is None else v)
            v = tsdf.get(k)
            vals.append("missing: not in the TSDF evaluation" if v is None else v)
            w.writerow([k[0], k[1]] + vals)
    print(out, len(keys), "lines")


if __name__ == "__main__":
    main()
