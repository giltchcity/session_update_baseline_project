"""Table 2 'Old-site full-map ghost' for one transition from a cleanup_evaluate event summary.

build_final_tables.py hard-codes 21.70 / 24.73 (Panoptic 57.07 / 60.13) and describes
them as "the conservative full-map residual of the cleanup probe pipeline". They are
reproduced exactly (to 0.01, for both Ours and Panoptic) by this pooled ratio over all
D3 events of the transition (I2 included) in results/abc_eval_v2/cleanup/event_summary_{b,c}.csv:

    ghost = 100 * sum(final_full_confirmed_wrong_surface)
                / sum(final_full_initial_supported_known_empty)

i.e. of all old-site probes that were represented in the inherited (before) map and are
sensor-confirmed empty at the session end, the share whose nearest current full-map
surface (background + native-present object meshes, within 10 cm) is itself in
sensor-confirmed free space. Identification is by exact numeric match; no script that
computed it survives.

usage: ghost_cohort.py --summary event_summary_b.csv --out GHOST.json [--method baseline]
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--summary', type=Path, required=True)
    ap.add_argument('--method', default='baseline')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(args.summary.open())
            if r['method'] == args.method and r['scope'] == 'D3']
    if not rows:
        raise SystemExit('no D3 rows for method %s in %s' % (args.method, args.summary))
    per_event = []
    num = den = 0
    for r in rows:
        w = r.get('final_full_confirmed_wrong_surface', '')
        k = r.get('final_full_initial_supported_known_empty', '')
        if w in ('', 'None') or k in ('', 'None'):
            per_event.append(dict(event_id=r['event_id'], status=r['status'], included=False))
            continue
        num += int(float(w))
        den += int(float(k))
        per_event.append(dict(event_id=r['event_id'], status=r['status'], included=True,
                              full_confirmed_wrong_surface=int(float(w)),
                              full_initial_supported_known_empty=int(float(k)),
                              full_conservative_residual_fraction=r.get('final_full_conservative_residual_fraction')))
    transition = rows[0]['transition']
    result = dict(transition=transition, method_label=args.method, source=str(args.summary),
                  formula='100*sum(final_full_confirmed_wrong_surface)/sum(final_full_initial_supported_known_empty), D3 events incl. I2',
                  numerator=num, denominator=den,
                  old_site_full_map_ghost_pct=round(100.0 * num / den, 2) if den else None,
                  old_site_full_map_ghost_pct_unrounded=100.0 * num / den if den else None,
                  events_included=sum(1 for e in per_event if e['included']), events=per_event)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'events'}), flush=True)


if __name__ == '__main__':
    main()
