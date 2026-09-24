"""Comparison tables for EvaluationScene runs next to the existing t2 rows.

  python -m eval.scene.summarize synthetic RUNS_ROOT
  python -m eval.scene.summarize real RUNS_ROOT

Synthetic: GROUP_STATE_SUMMARY (D1/D2/D3/mixed), FINAL_GEOMETRY (Mesh F1@5cm), retention json.
Real: CHANGE_ROWS.json (D2 / D3 published policy), GHOST.json, G1.json (P/R/F1@5cm), retention.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

FT = Path("/home/jixian/Desktop/FT")


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def synthetic(root: Path) -> None:
    runs = {"t2 (TSDF)": FT / "datasets/synthetic_ab/eval_t2_20260923"}
    tag = "3DGS " if "game" in root.name else "points "
    for d in sorted(p for p in root.glob("eval_*") if p.is_dir() and not p.name.startswith("eval_real")):
        runs[d.name.replace("eval_", tag)] = d
    kinds = [("a", "D2_hidden"), ("b", "D2_hidden"), ("b", "D3"), ("a", "D1_visible"), ("b", "D1_visible"),
             ("a", "mixed_visibility"), ("b", "mixed_visibility"), ("b", "visibility_change")]
    print("| run | " + " | ".join(f"{k} {s.upper()}" for s, k in kinds) + " | Mesh F1 A | Mesh F1 B | retention B | residue B |")
    print("|---" * (len(kinds) + 5) + "|")
    for name, d in runs.items():
        state = d / "ours/state_eval/group_scores/GROUP_STATE_SUMMARY.csv"
        cells = []
        rows = {(r["session"], r["kind"]): r for r in csv.DictReader(open(state))} if state.exists() else {}
        for s, k in kinds:
            r = rows.get((s, k))
            cells.append(f"{r['TP']}/{r['FP']}/{r['FN']}" if r else "-")
        geo = d / "geometry/FINAL_GEOMETRY.csv"
        g = {}
        if geo.exists():
            for r in csv.DictReader(open(geo)):
                if r["method"] == "ours" and r["scope"] == "all" and r["threshold_m"] == "0.05":
                    g[r["session"]] = 100 * float(r["F1"])
        ret = load(d / "retention_b.json") or {}
        fmt = lambda v: "-" if v is None else f"{100 * v:.1f}"
        cells += [f"{g['a']:.2f}" if "a" in g else "-", f"{g['b']:.2f}" if "b" in g else "-",
                  fmt(ret.get("unobserved_retention")), fmt(ret.get("absent_residue"))]
        print(f"| {name} | " + " | ".join(cells) + " |")


def real(root: Path) -> None:
    ra = FT / "results/current_version_20260921/real_abc"
    rows = {"t2 (TSDF)": dict(ch=lambda s: ra / f"changes/eval_cand_t2_{s}", gh=lambda s: ra / f"ghost/eval_cand_t2_{s}",
                              g1=lambda s: FT / f"eval/results/real/ours/{s}/geometry/G1.json",
                              ret=lambda s: None)}
    tag = "3DGS " if "game" in root.name else "points "
    for d in sorted(p for p in root.glob("eval_*") if p.is_dir() and not p.name.startswith("eval_syn")):
        rows[d.name.replace("eval_", tag)] = dict(
            ch=lambda s, d=d: d / f"changes/{s}", gh=lambda s, d=d: d / f"ghost/{s}",
            g1=lambda s, d=d: d / f"geometry/{s}/G1.json", ret=lambda s, d=d: d / f"retention_{s}.json")
    print("| run | stage | D2 TP/FP/FN (F1) | D3 TP/FP/FN (F1) | ghost % | G1 P/R/F1 @5cm | retention | absent residue |")
    print("|---|---|---|---|---|---|---|---|")
    for name, f in rows.items():
        for s in "abc":
            ch = load(Path(f["ch"](s)) / "CHANGE_ROWS.json")
            gh = load(Path(f["gh"](s)) / "GHOST.json") if s != "a" else None
            g1 = load(f["g1"](s))
            ret = load(f["ret"](s)) if f["ret"](s) else None
            if not (ch or g1):
                continue
            d2 = ch and ch["D2"]["published_policy"]
            d3 = ch and ch.get("D3") and ch["D3"]["published_policy"]
            c = lambda x: f"{x['TP']}/{x['FP']}/{x['FN']} ({x['F1']:.1f})" if x else "-"   # already percent
            geo = f"{100 * g1['P_05cm']:.2f}/{100 * g1['R_05cm']:.2f}/{100 * g1['F1_05cm']:.2f}" if g1 else "-"
            fmt = lambda v: "-" if v is None else f"{100 * v:.1f}"
            ghost = "-" if not gh else ("n/e" if gh["old_site_full_map_ghost_pct"] is None
                                        else f"{gh['old_site_full_map_ghost_pct']:.2f}")
            print(f"| {name} | {s.upper()} | {c(d2)} | {c(d3)} | {ghost} | ", end="")
            print(f"{geo} | {fmt(ret and ret.get('unobserved_retention'))} | {fmt(ret and ret.get('absent_residue'))} |")


if __name__ == "__main__":
    kind, roots = sys.argv[1], [Path(r) for r in sys.argv[2:]]
    for root in roots:
        synthetic(root) if kind == "synthetic" else real(root)
