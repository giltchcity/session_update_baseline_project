"""Online object P/R/F1 of one session again under eval_fast's current object configs (configs/{a,b,c}.yaml), when
nothing but the object matching changed: only the official object evaluation runs (each snapshot cut from the .zpk
in /dev/shm, scored alone, EVAL_FAST_JOBS in parallel), then the same postprocess and duration-weighted NR scorer as
run.py. The NR count of a snapshot depends on the map only, so it is read from an earlier full run of the same map
and session (its weighted_nr/OBJECT_WEIGHTED_NR_PER_QUERY.csv) instead of exporting the snapshots again.
Writes OUT/results/static_objects.csv, OUT/explanation/, OUT/weighted_nr/; nothing else.
usage: object_only.py --map MAP.zpk --session a|b|c --moments FIRST:LAST --nr-from EARLIER_EVAL_DIR --out OUT
"""
import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor/pack"))
import map4d_pack as mp  # noqa: E402
import zpk_snapshot as zs  # noqa: E402

PY_METRIC = "/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python"


def score_one(job):
    """The official object table row of one snapshot."""
    i, source, moment, session = job
    with tempfile.TemporaryDirectory(prefix="eval-fast-object-", dir="/dev/shm") as tmp:
        tmp = Path(tmp)
        sliced = tmp / "one.4dmap"
        zs.cut_many(source, [moment], out=str(sliced), work=str(tmp))
        subprocess.run([sys.executable, str(HERE / "vendor/object/official_native_eval_run.py"), "--map", str(sliced),
                        "--config", str(HERE / f"configs/{session}.yaml"), "--output", str(tmp / "official"),
                        "--only-final-argument", "false"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        lines = (tmp / "official/results/static_objects.csv").read_text().splitlines()
    return i, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, type=Path)
    ap.add_argument("--session", required=True, choices="abc")
    ap.add_argument("--moments", required=True)
    ap.add_argument("--nr-from", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    source = str(args.map.resolve())
    first, last = map(int, args.moments.split(":"))
    moments = list(range(first, last + 1))
    stamps = [mp.read_manifest(source)["source"]["stamps"][m] for m in moments]

    out = args.out.resolve()
    (out / "results").mkdir(parents=True, exist_ok=False)
    jobs = [(i, source, m, args.session) for i, m in enumerate(moments)]
    with ProcessPoolExecutor(int(os.environ.get("EVAL_FAST_JOBS", "8"))) as pool:
        done = dict(pool.map(score_one, jobs))
    header = done[0][0]
    with (out / "results/static_objects.csv").open("w") as f:   # as run.py: map i repeated i + 1 times
        f.write(header + "\n")
        rows = []
        for i in range(len(moments)):
            assert done[i][0] == header and done[i][1].split(",", 2)[1] == str(stamps[i]), i
            rows += [str(i) + done[i][1][1:]] * (i + 1)
        f.write("\n".join(rows))

    # The NR of each snapshot, from the earlier run of the same map and session.
    per_query = next(args.nr_from.glob("real_abc/objects/*/weighted_nr/OBJECT_WEIGHTED_NR_PER_QUERY.csv"))
    nr = {int(r["Name"]): int(r["NR"]) for r in csv.DictReader(per_query.open())}
    assert sorted(nr) == list(range(len(moments))), per_query
    with tempfile.TemporaryDirectory(prefix="eval-fast-object-nr-", dir="/dev/shm") as tmp:
        timeline = Path(tmp) / "snapshots.jsonl"     # only what the NR count reads: NR present 0-vertex nodes
        timeline.write_text("".join(json.dumps(dict(index=i, stamp_ns=stamps[i], objects=[
            dict(present=True, semantic=-1, mesh_vertices=0)] * nr[i])) + "\n" for i in range(len(moments))))
        subprocess.run([PY_METRIC, str(HERE / "vendor/object/official_native_eval_postprocess.py"),
                        str(out / "results/static_objects.csv"), "--output", str(out / "explanation")],
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run([PY_METRIC, str(HERE / "vendor/harness/object_weighted_nr.py"),
                        "--rows", str(out / "explanation/native_unique_online_rows.csv"),
                        "--native-summary", str(out / "explanation/NATIVE_ONLINE_SUMMARY.json"),
                        "--snapshots", str(timeline), "--out", str(out / "weighted_nr")], check=True,
                       stdout=subprocess.DEVNULL)
    print(json.dumps(json.loads((out / "weighted_nr/OBJECT_WEIGHTED_NR.json").read_text())["duration_weighted_nr"]))


if __name__ == "__main__":
    main()
