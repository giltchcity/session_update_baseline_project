#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def f1(p: float, r: float) -> float:
    return 2.0 * p * r / (p + r) if p + r > 0 else 0.0


def read_last_csv(path: Path) -> dict[str, str]:
    with path.open(newline="") as fin:
        rows = list(csv.DictReader(fin))
    if not rows:
        raise SystemExit(f"empty CSV: {path}")
    return rows[-1]


def safe_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "0")
    try:
        return float(value)
    except ValueError:
        return 0.0


def metrics_for_eval_dir(label: str, eval_dir: Path, threshold: str) -> dict[str, object]:
    bg = read_last_csv(eval_dir / "results" / "background_mesh.csv")
    obj = read_last_csv(eval_dir / "results" / "static_objects.csv")

    bg_p = safe_float(bg, f"Accuracy@{threshold}")
    bg_r = safe_float(bg, f"Completeness@{threshold}")
    bg_p05 = safe_float(bg, "Accuracy@0.05")
    bg_r05 = safe_float(bg, "Completeness@0.05")
    bg_p10 = safe_float(bg, "Accuracy@0.1")
    bg_r10 = safe_float(bg, "Completeness@0.1")
    bg_p20 = safe_float(bg, "Accuracy@0.2")
    bg_r20 = safe_float(bg, "Completeness@0.2")

    obj_det = safe_float(obj, "NumObjDetected")
    obj_miss = safe_float(obj, "NumObjMissed")
    obj_hall = safe_float(obj, "NumObjHallucinated")
    obj_p = obj_det / (obj_det + obj_hall) if obj_det + obj_hall > 0 else 0.0
    obj_r = obj_det / (obj_det + obj_miss) if obj_det + obj_miss > 0 else 0.0

    change_tp = (
        safe_float(obj, "AppearedTP")
        + safe_float(obj, "DisappearedTP")
    )
    change_fp = (
        safe_float(obj, "AppearedFP")
        + safe_float(obj, "DisappearedFP")
    )
    change_fn = (
        safe_float(obj, "AppearedFN")
        + safe_float(obj, "DisappearedFN")
    )
    change_p = change_tp / (change_tp + change_fp) if change_tp + change_fp > 0 else 0.0
    change_r = change_tp / (change_tp + change_fn) if change_tp + change_fn > 0 else 0.0

    row: dict[str, object] = {
        "label": label,
        "eval_dir": str(eval_dir),
        "map_name": bg.get("Name", ""),
        "query": obj.get("Query", ""),
        "background_p": bg_p,
        "background_r": bg_r,
        "background_f1": f1(bg_p, bg_r),
        "background_p05": bg_p05,
        "background_r05": bg_r05,
        "background_f1_05": f1(bg_p05, bg_r05),
        "background_p10": bg_p10,
        "background_r10": bg_r10,
        "background_f1_10": f1(bg_p10, bg_r10),
        "background_p20": bg_p20,
        "background_r20": bg_r20,
        "background_f1_20": f1(bg_p20, bg_r20),
        "object_p": obj_p,
        "object_r": obj_r,
        "object_f1": f1(obj_p, obj_r),
        "change_p": change_p,
        "change_r": change_r,
        "change_f1": f1(change_p, change_r),
        "change_tp": change_tp,
        "change_fp": change_fp,
        "change_fn": change_fn,
        "object_detected": obj_det,
        "object_missed": obj_miss,
        "object_hallucinated": obj_hall,
    }

    dyn_path = eval_dir / "results" / "dynamic_objects.csv"
    if dyn_path.exists():
        dyn = read_last_csv(dyn_path)
        dyn_det = safe_float(dyn, "NumObjDetected")
        dyn_miss = safe_float(dyn, "NumObjMissed")
        dyn_hall = safe_float(dyn, "NumObjHallucinated")
        dyn_p = dyn_det / (dyn_det + dyn_hall) if dyn_det + dyn_hall > 0 else 0.0
        dyn_r = dyn_det / (dyn_det + dyn_miss) if dyn_det + dyn_miss > 0 else 0.0
        row.update(
            {
                "dynamic_p": dyn_p,
                "dynamic_r": dyn_r,
                "dynamic_f1": f1(dyn_p, dyn_r),
            }
        )
    else:
        row.update({"dynamic_p": "", "dynamic_r": "", "dynamic_f1": ""})

    return row


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    headers = [
        "label",
        "background_p",
        "background_r",
        "background_f1",
        "background_p05",
        "background_r05",
        "background_f1_05",
        "background_p10",
        "background_r10",
        "background_f1_10",
        "background_p20",
        "background_r20",
        "background_f1_20",
        "object_p",
        "object_r",
        "object_f1",
        "change_p",
        "change_r",
        "change_f1",
        "dynamic_p",
        "dynamic_r",
        "dynamic_f1",
    ]
    with path.open("w") as out:
        out.write("| " + " | ".join(headers) + " |\n")
        out.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            vals = []
            for h in headers:
                v = row.get(h, "")
                vals.append(f"{v:.6f}" if isinstance(v, float) else str(v))
            out.write("| " + " | ".join(vals) + " |\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", default="0.2")
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("items", nargs="+", help="label=eval_dir")
    args = ap.parse_args()

    rows = []
    for item in args.items:
        if "=" not in item:
            raise SystemExit(f"expected label=eval_dir, got {item}")
        label, path = item.split("=", 1)
        rows.append(metrics_for_eval_dir(label, Path(path), args.threshold))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fout:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_markdown(rows, args.out_md)
    print(args.out_csv)
    print(args.out_md)


if __name__ == "__main__":
    main()
