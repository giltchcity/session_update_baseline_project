"""Online Object P/R/F1 for ONE session: duration-weighted + NR correction (published
Table 1 / OBJECT_ONLINE_FINAL.csv), unweighted NR-corrected, the raw native summary, and
the diagnostic native ChangeF1 (DIAGNOSTIC_OFFICIAL_NATIVE_CHANGE.csv).

Harness copy of nr_sequence / weighted_mean / plain_mean / object_metrics and of the
diagnostic block of tools/abc_final/build_final_tables.py. Formulas are verbatim; the
only change is that the two inputs are explicit paths instead of
results/provenance/official_native_eval_matched/<run>/explanation/native_unique_online_rows.csv
and results/abc_eval_final_{baseline,khronos}/predictions/<s>/snapshots.jsonl.

  TP = NumObjDetected, FN = NumObjMissed (official, unchanged)
  FP = max(0, NumObjHallucinated - NR)
  NR = present nodes of the saved snapshot that are the person class (12, ADE; --person-label) or have 0 mesh vertices
  weight(i) = length of the interval in which saved map i is current (seed map mapped to
              session start, inter-session downtime not weighted, last query weight 0,
              queries with TP == 0 weight 0); F1 undefined -> excluded (weighted) / 0 (unweighted)

usage: object_weighted_nr.py --rows native_unique_online_rows.csv --snapshots snapshots.jsonl
                             [--native-summary NATIVE_ONLINE_SUMMARY.json] [--person-label 12] --out DIR
"""
import argparse
import csv
import json
from pathlib import Path


def nr_sequence(path, person=12):
    """Present object nodes that are not reviewed objects (person or empty)."""
    if path is None:
        return []
    out = []
    with Path(path).open() as handle:
        for line in handle:
            snap = json.loads(line)
            present = [o for o in snap["objects"] if o["present"]]
            out.append(sum(1 for o in present if o["semantic"] == person or o["mesh_vertices"] == 0))
    return out


def weighted_mean(values, weights):
    num = sum(w * v for v, w in zip(values, weights) if v == v)
    den = sum(w for v, w in zip(values, weights) if v == v)
    return 100.0 * num / den if den else float("nan")


def plain_mean(values):
    live = [v for v in values if v == v]
    return 100.0 * sum(live) / len(live) if live else float("nan")


def object_metrics(rows, nr):
    if not nr:
        nr = [0] * len(rows)
    assert len(nr) == len(rows), f"snapshot/query mismatch: {len(nr)} snapshots vs {len(rows)} unique queries"

    stamps = [int(r["Query"]) for r in rows]
    raw = [(s - stamps[0]) / 1e9 for s in stamps]
    n = len(raw)
    gap = raw[2] - raw[1] if n > 2 else 2.13
    t_eff = [0.0] * n
    for i in range(1, n):
        t_eff[i] = gap + (raw[i] - raw[1])
    weights = [(t_eff[i + 1] - t_eff[i]) if i + 1 < n else 0.0 for i in range(n)]

    P, R, F1, TP, per_query = [], [], [], [], []
    for i, row in enumerate(rows):
        tp = int(row["NumObjDetected"])
        fp = max(0, int(row["NumObjHallucinated"]) - nr[i])
        fn = int(row["NumObjMissed"])
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        P.append(p)
        R.append(r)
        F1.append(2 * p * r / (p + r) if (p == p and r == r and p + r > 0) else float("nan"))
        TP.append(tp)
        per_query.append(dict(Name=row["Name"], Query=row["Query"], NumObjDetected=tp,
                              NumObjHallucinated=int(row["NumObjHallucinated"]), NR=nr[i], FP_corrected=fp,
                              NumObjMissed=fn, t_eff_s=t_eff[i], weight_s=weights[i]))

    active = [w if TP[i] > 0 else 0.0 for i, w in enumerate(weights)]
    for q, a, p, r, f in zip(per_query, active, P, R, F1):
        q.update(active_weight_s=a, P=p, R=r, F1=f)

    return {
        "queries": n,
        "nr_total": sum(nr),
        "P_unweighted": plain_mean(P),
        "R_unweighted": plain_mean(R),
        "F1_unweighted": plain_mean([f if f == f else 0.0 for f in F1]),
        "P_weighted": weighted_mean(P, active),
        "R_weighted": weighted_mean(R, active),
        "F1_weighted": weighted_mean(F1, active),
    }, per_query


def native_change_diag(rows):
    values = []
    for row in rows:
        try:
            values.append(float(row["ChangeF1"]))
        except (KeyError, ValueError):
            pass
    live = [v for v in values if v == v]
    return dict(unique_queries=len(rows), ChangeF1_defined_queries=len(live),
                ChangeF1_unweighted=round(100 * sum(live) / len(live), 2) if live else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', type=Path, required=True)
    ap.add_argument('--snapshots', type=Path, required=True)
    ap.add_argument('--native-summary', type=Path, default=None)
    ap.add_argument('--person-label', type=int, default=12)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(args.rows.open()))
    m, per_query = object_metrics(rows, nr_sequence(args.snapshots, args.person_label))
    result = dict(
        rows=str(args.rows), snapshots=str(args.snapshots),
        unique_queries=m["queries"], non_reviewed_nodes_total=m["nr_total"],
        duration_weighted_nr=dict(P=round(m["P_weighted"], 2), R=round(m["R_weighted"], 2), F1=round(m["F1_weighted"], 2)),
        unweighted_nr=dict(P=round(m["P_unweighted"], 2), R=round(m["R_unweighted"], 2), F1=round(m["F1_unweighted"], 2)),
        unrounded=m,
        native_change_diagnostic=native_change_diag(rows),
    )
    if args.native_summary and args.native_summary.exists():
        s = json.loads(args.native_summary.read_text())['summary']
        result['raw_native'] = {k: s[k]['mean'] for k in ('ObjectPrecision', 'ObjectRecall', 'ObjectF1')}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'OBJECT_WEIGHTED_NR.json').write_text(json.dumps(result, indent=2))
    with (args.out / 'OBJECT_WEIGHTED_NR_PER_QUERY.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_query[0]))
        w.writeheader()
        w.writerows(per_query)
    print(json.dumps({k: v for k, v in result.items() if k != 'unrounded'}), flush=True)


if __name__ == '__main__':
    main()
