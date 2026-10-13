#!/usr/bin/env python3
"""Parse the per-slice object-state ledger from a session mapper log.

The mapper emits structured lines per change-detection round:

  STATE_SLICE    per-object six-class surface evidence of the RGB-D world
  MATERIALIZE    which fragment was materialized into the DSG node, and why
  SESSION_ABSORB candidate absorption decision inside the B-session state
  TOP_ABSORB     candidate absorption decision for the top-level current
  FINALIZE       terminal A/B completion decision (shared surface gate)
  EVIDENCE       ray support/contradiction used by the state machine

This is the ground-truth trace that answers "did the map follow the real
world, and when": each round shows what the current RGB-D measured at the old
site (supported / free-space / replaced-by-other / replaced-by-background /
occluded / unobserved) and what the map did with it.

Usage:
  parse_state_slices.py <logfile> [--inst N] [--out DIR]
      logfile: khronos.log / glog INFO file from a run output
      --inst N: restrict the CSV to one physical instance id
      --out DIR: write <DIR>/state_slices.csv and per-instance CSVs

Without --out the CSV is printed to stdout.
"""
import argparse
import csv
import io
import os
import re
import sys

LINE_RE = re.compile(r"\]\s*(?P<tag>STATE_SLICE|MATERIALIZE|SESSION_ABSORB|"
                     r"TOP_ABSORB|FINALIZE|EVIDENCE|INGEST_DECIDE)\s+"
                     r"(?P<fields>.*)$")
PAIR_RE = re.compile(r"(\w+)=([^\s]+)")


def parse_line(line):
    match = LINE_RE.search(line)
    if not match:
        return None
    fields = {}
    for key, value in PAIR_RE.findall(match.group("fields")):
        try:
            fields[key] = int(value)
        except ValueError:
            fields[key] = value
    return match.group("tag"), fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logfile")
    parser.add_argument("--inst", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = []
    with open(args.logfile, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parsed = parse_line(line)
            if parsed is None:
                continue
            tag, fields = parsed
            instance = fields.get("inst")
            if args.inst is not None and instance != args.inst:
                continue
            row = {"tag": tag}
            row.update(fields)
            rows.append(row)

    if not rows:
        print("no ledger lines found", file=sys.stderr)
        return 1

    tags = sorted({row["tag"] for row in rows})
    keys = ["tag", "inst", "stamp"]
    for tag in tags:
        for row in rows:
            if row["tag"] == tag:
                for key in row:
                    if key not in keys:
                        keys.append(key)
                break

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        target = os.path.join(args.out, "state_slices.csv")
        with open(target, "w", encoding="utf-8") as stream:
            stream.write(output.getvalue())
        print(f"wrote {len(rows)} rows -> {target}")

        # Per-instance split for easy timeline inspection.
        by_inst = {}
        for row in rows:
            by_inst.setdefault(row.get("inst", "?"), []).append(row)
        for instance, inst_rows in sorted(by_inst.items()):
            inst_file = os.path.join(args.out, f"state_slices_inst{instance}.csv")
            with open(inst_file, "w", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=keys,
                                        extrasaction="ignore")
                writer.writeheader()
                for row in inst_rows:
                    writer.writerow(row)
            print(f"  inst {instance}: {len(inst_rows)} rows -> {inst_file}")
    else:
        sys.stdout.write(output.getvalue())
    return 0


if __name__ == "__main__":
    sys.exit(main())
