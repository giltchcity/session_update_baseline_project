#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fin:
        return list(csv.DictReader(fin))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_by_name(rows: list[dict[str, str]], name_key: str = "name") -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(row[name_key], []).append(row)
    return out


def official_state(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "Missing"
    states = [row.get("change_state", "") for row in rows]
    return "+".join(sorted(set(states)))


def ours_expected_state(change_type: str, belief: dict[str, str] | None, summary_present: bool) -> str:
    if belief is None:
        return "Missing"
    decision = belief.get("decision", "")
    if change_type == "removed":
        if not summary_present and "absent" in decision:
            return "Absent"
        if not summary_present and "unobserved" in decision:
            return "Unobserved"
        return "Leaked"
    if change_type == "new":
        if summary_present and "new_current" in decision:
            return "New"
        return "MissingOrMisclassified"
    if change_type == "moved":
        if not summary_present:
            return "MissingCurrent"
        if "support_filter" in decision or "supported_prior" in decision or "current_only" in decision:
            return "MovedCurrent"
        if "persistent_union" in decision:
            return "RiskyUnion"
        return "Persistent"
    return "Unknown"


def status(row: dict[str, Any]) -> str:
    change_type = row["change_type"]
    gt_points = int(row["gt_points"])
    ours_state = row["ours_logical_state"]
    support = float(row["prior_support_ratio"] or 0.0)
    free_conflict = float(row["prior_free_space_conflict"] or 0.0)
    if gt_points == 0 and row["ours_belief_present"] == "False":
        return "ignore_no_gt_geometry"
    if change_type == "removed":
        return "ok" if ours_state == "Absent" else "error_removed_leaked"
    if change_type == "new":
        return "ok" if ours_state == "New" else "error_new_missing"
    if change_type == "moved":
        agrees = float(row["current_agrees_with_prior"] or 0.0)
        if support >= 0.98 and agrees >= 0.98:
            return "ok_stable_overlap"
        if ours_state == "MovedCurrent":
            return "ok_moved_filtered"
        if ours_state == "RiskyUnion" and free_conflict < 0.2:
            return "warn_union_low_conflict"
        return "error_moved_stale_risk"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--changes", type=Path, required=True)
    parser.add_argument("--official-states", type=Path, required=True)
    parser.add_argument("--ours-beliefs", type=Path, required=True)
    parser.add_argument("--ours-summary", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    changes = read_rows(args.changes)
    official_by_name = load_by_name(read_rows(args.official_states))
    beliefs_by_name = {row["name"]: row for row in read_rows(args.ours_beliefs)}
    summary_by_name = {row["name"]: row for row in read_rows(args.ours_summary)}

    rows: list[dict[str, Any]] = []
    for change in changes:
        label = change["label"]
        belief = beliefs_by_name.get(label)
        summary = summary_by_name.get(label)
        out = {
            "label": label,
            "change_type": change["change_type"],
            "gt_points": int(change.get("points", 0)),
            "official_raw_state": official_state(official_by_name.get(label, [])),
            "official_raw_rows": len(official_by_name.get(label, [])),
            "ours_belief_present": str(belief is not None),
            "ours_summary_present": str(summary is not None),
            "ours_decision": belief.get("decision", "") if belief else "",
            "ours_logical_state": ours_expected_state(
                change["change_type"],
                belief,
                summary is not None,
            ),
            "prior_support_ratio": belief.get("prior_support_ratio", "") if belief else "",
            "prior_free_space_conflict": belief.get("prior_free_space_conflict", "") if belief else "",
            "current_agrees_with_prior": belief.get("current_agrees_with_prior", "") if belief else "",
            "prior_covered_by_current": belief.get("prior_covered_by_current", "") if belief else "",
        }
        out["status"] = status(out)
        rows.append(out)

    fieldnames = [
        "label",
        "change_type",
        "gt_points",
        "official_raw_state",
        "official_raw_rows",
        "ours_belief_present",
        "ours_summary_present",
        "ours_decision",
        "ours_logical_state",
        "prior_support_ratio",
        "prior_free_space_conflict",
        "current_agrees_with_prior",
        "prior_covered_by_current",
        "status",
    ]
    write_csv(args.out_csv, rows, fieldnames)

    counts = Counter(row["status"] for row in rows)
    by_change = Counter((row["change_type"], row["status"]) for row in rows)
    lines = [
        "# Base1.2 state audit",
        "",
        "## Status Counts",
        "",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## By Change Type", ""]
    for (change_type, key), value in sorted(by_change.items()):
        lines.append(f"- {change_type} / {key}: {value}")
    lines += ["", "## Non-OK Rows", ""]
    for row in rows:
        if row["status"] != "ok" and not row["status"].startswith("ok_"):
            lines.append(
                f"- {row['label']} ({row['change_type']}): {row['status']}; "
                f"official={row['official_raw_state']}; ours={row['ours_decision']}; "
                f"gt_points={row['gt_points']}"
            )
    args.out_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
