#!/usr/bin/env python3
"""Low-disk real A/B/C evaluation using copied, unchanged scoring programs.

Only the input adapter and orchestration are new. One decoded .4dmap snapshot
lives in /dev/shm at a time; the original multi-gigabyte map is never written.
"""

import argparse
import csv
import hashlib
import json
import os
import signal
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


HERE = Path(__file__).resolve().parent
FT = HERE.parent
PACK = HERE / "vendor/pack"
sys.path.insert(0, str(PACK))
import map4d_pack as mp  # noqa: E402
import zpk_snapshot as zs  # noqa: E402

PY_METRIC = Path("/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python")
GEO = HERE / "vendor/geometry"
HARNESS = HERE / "vendor/harness"
SOURCE_ENV = HERE / "runtime_khronos_env.sh"
MAGIC = b"ABCCLN1\0"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


RESOURCE_PEAK_SWAP = 0


def evaluator_swap_bytes():
    """Read swap charged to this evaluator and its live descendants."""
    pending = [os.getpid()]
    visited = set()
    total = 0
    while pending:
        pid = pending.pop()
        if pid in visited:
            continue
        visited.add(pid)
        proc = Path("/proc") / str(pid)
        try:
            status = (proc / "status").read_text()
            total += next((int(line.split()[1]) * 1024 for line in status.splitlines()
                           if line.startswith("VmSwap:")), 0)
            children = proc / "task" / str(pid) / "children"
            pending.extend(map(int, children.read_text().split()))
        except (FileNotFoundError, ProcessLookupError):
            pass  # The sampled child finished between the two procfs reads.
    return total


def guard_evaluator_swap():
    global RESOURCE_PEAK_SWAP
    swapped = evaluator_swap_bytes()
    RESOURCE_PEAK_SWAP = max(RESOURCE_PEAK_SWAP, swapped)


def run(command, *, canonical=False, temp=None):
    if canonical:
        command = ["bash", "-c", 'source "$1" >/dev/null 2>&1 || exit 90; shift; exec "$@"',
                   "eval-fast", str(SOURCE_ENV), *map(str, command)]
    env = os.environ.copy()
    if temp:
        env["TMPDIR"] = str(temp)
    # Poll the owned process tree while native exporters and scorers run.
    # Unrelated WSL services may page out while this tree stays entirely in RAM.
    with tempfile.TemporaryFile(mode="w+", dir=temp or "/dev/shm") as log:
        process = subprocess.Popen(list(map(str, command)), env=env, text=True,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                guard_evaluator_swap()
                time.sleep(0.25)
            guard_evaluator_swap()
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    process.wait()
            raise
        log.seek(0)
        output = log.read()
    if process.returncode:
        raise RuntimeError(f"command failed ({process.returncode}): {command}\n"
                           + output[-6000:])
    return output


def memory_guard(largest, original_swap_free):
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemAvailable:", "SwapFree:")):
            key, value, _ = line.split()
            values[key.rstrip(":")] = int(value) * 1024
    shm = os.statvfs("/dev/shm")
    shm_free = shm.f_bavail * shm.f_frsize
    guard_evaluator_swap()
    return values["MemAvailable"], shm_free


def only_line(path, index, stamp):
    lines = Path(path).read_text().splitlines()
    if len(lines) != 1:
        raise RuntimeError(f"Expected exactly one JSONL row: {path}")
    row = json.loads(lines[0])
    if row["index"] != 0 or row["stamp_ns"] != stamp:
        raise RuntimeError(f"Snapshot metadata mismatch: {path}")
    needle = '"index":0'
    if lines[0].count(needle) != 1:
        raise RuntimeError(f"Cannot replace exactly one snapshot index: {path}")
    return lines[0].replace(needle, f'"index":{index}', 1) + "\n"



def score_snapshot(i, c):
    """One snapshot scored alone in its own temporary directory (a worker process); returns what the
    main process writes, in index order."""
    stamps = c["stamps"]
    with tempfile.TemporaryDirectory(prefix="eval-fast-", dir=Path("/dev/shm")) as tmp_name:
        tmp = Path(tmp_name)
        sliced = tmp / "one.4dmap"
        zs.cut_many(c["source"], [c["moments"][i]], out=str(sliced), work=str(tmp))
        result = {"i": i, "slice_size": sliced.stat().st_size}
        native = tmp / "official"
        run([sys.executable, HERE / "vendor/object/official_native_eval_run.py", "--map", sliced,
             "--config", HERE / os.environ.get("EVAL_FAST_CONFIGS", "configs") / f"{c['session']}.yaml",
             "--output", native, "--only-final-argument", "false"], temp=tmp)
        lines = (native / "results/static_objects.csv").read_text().splitlines()
        if len(lines) != 2 or not lines[1].startswith("0,"):
            raise RuntimeError(f"Unexpected official CSV at snapshot {i}")
        if (str(i) + lines[1][1:]).split(",", 2)[1] != str(stamps[i]):
            raise RuntimeError(f"Official object timestamp mismatch at {i}")
        result["object_lines"] = lines
        if c["mode"] == "full":
            timeline_dir = tmp / "timeline"
            run([HERE / "vendor/timeline/export_timeline", sliced, timeline_dir], canonical=True, temp=tmp)
            result["timeline"] = only_line(timeline_dir / "snapshots.jsonl", i, stamps[i])
            if c["cleanup"]:
                one_bin = tmp / "cleanup.bin"
                run([HERE / "vendor/cleanup/export_cleanup_binary", sliced, c["probes"], one_bin],
                    canonical=True, temp=tmp)
                data = one_bin.read_bytes()
                if (data[:8] != MAGIC or struct.unpack_from("<I", data, 8)[0] != 1 or
                        struct.unpack_from("<I", data, 12)[0] != 0 or
                        struct.unpack_from("<Q", data, 16)[0] != stamps[i]):
                    raise RuntimeError(f"Cleanup binary header mismatch at {i}")
                result["cleanup"] = struct.pack("<I", i) + data[16:]
                one_ghost = tmp / "ghost.jsonl"
                run([HERE / "vendor/ghost/cleanup_mesh", sliced, c["ghost_probes"], one_ghost],
                    canonical=True, temp=tmp)
                result["ghost"] = only_line(one_ghost, i, stamps[i])
        if i == c["count"] - 1:
            mesh = Path(c["work"]) / "geometry_viewer.ply"
            run([HERE / "vendor/geometry/geometry_export", sliced, mesh, "viewer"], canonical=True, temp=tmp)
            (Path(c["geo"]) / "geometry_viewer_export_meta.json").write_bytes(
                Path(str(mesh) + ".json").read_bytes())
        return result

def selected_indices(spec, count):
    if not spec:
        return sorted({0, count // 2, count - 1})
    numbers = [count - 1 if part == "final" else int(part)
               for part in spec.split(",")]
    indices = sorted(set(numbers))
    if not indices or indices[0] < 0 or indices[-1] >= count:
        raise ValueError("Selected snapshot index is outside this map")
    return indices


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", required=True, type=Path, help=".4dmap.zpk input")
    ap.add_argument("--session", required=True, choices="abc")
    ap.add_argument("--name", required=True, help="Short output label, e.g. candidate1")
    ap.add_argument("--mode", required=True, choices=("quick", "full"))
    ap.add_argument("--out", required=True, type=Path, help="New result directory")
    ap.add_argument("--prior", type=Path, help="Prior A/B map for full B/C D3 provenance")
    ap.add_argument("--indices", help="Quick mode only: 0,31,final; default first/middle/final")
    ap.add_argument("--moments", help="FIRST:LAST, the map's snapshots (inclusive) that are this session's map, "
                    "e.g. one session of a continuous map; default all")
    ap.add_argument("--compare-archive", type=Path,
                    help="Existing real_abc directory; compare exact copied native exports")
    args = ap.parse_args()
    source = args.map.resolve(strict=True)
    if source.suffix != ".zpk":
        ap.error("Pass a packed .4dmap.zpk, not a plain map")
    if args.mode == "full" and args.indices:
        ap.error("--indices is only for quick mode")
    if args.mode == "full" and args.session in "bc" and not args.prior:
        ap.error("Full B/C summary needs --prior to label D3 provenance")
    if not args.name.replace("_", "").replace("-", "").isalnum():
        ap.error("--name must contain only letters, digits, underscores or hyphens")
    if not PY_METRIC.is_file():
        ap.error(f"Missing scoring Python environment: {PY_METRIC}")

    manifest = mp.read_manifest(source)
    first, last = (map(int, args.moments.split(":")) if args.moments
                   else (0, len(manifest["source"]["stamps"]) - 1))
    moments = list(range(first, last + 1))  # this session's snapshots in the map; i below counts within them
    if not moments or first < 0 or last >= len(manifest["source"]["stamps"]):
        ap.error("--moments is outside this map")
    stamps = [manifest["source"]["stamps"][m] for m in moments]
    count = len(stamps)
    indices = list(range(count)) if args.mode == "full" else selected_indices(args.indices, count)
    largest = max(manifest["snapshots"][moments[i]]["decoded_len"] for i in indices)
    swap_free = next(int(line.split()[1]) * 1024 for line in
                     Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("SwapFree:"))
    memory_guard(largest, swap_free)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ra = out / "real_abc"
    tag = f"{args.name}_{args.session}"
    obj = ra / "objects" / tag
    exp = ra / "exports" / tag
    geo = ra / "geometry" / tag
    changes = ra / "changes" / tag
    ghost = ra / "ghost" / tag
    for p in (obj / "results", exp, geo, changes, ghost):
        p.mkdir(parents=True, exist_ok=True)
    raw_path = obj / "results" / ("static_objects.csv" if args.mode == "full"
                                   else "static_objects_selected.csv")
    # The per-snapshot exports only feed the scorers below: they live in memory (/dev/shm) for this run and are
    # dropped at the end; only the scores are written to OUT.
    work = Path(tempfile.mkdtemp(prefix="eval-fast-exports-", dir="/dev/shm"))
    timeline_path = work / "snapshots.jsonl"
    cleanup_path = work / "cleanup_probes.bin"
    ghost_path = work / "cleanup_mesh.jsonl"
    temp_root = Path("/dev/shm")
    started = time.monotonic()
    minimum_free_ram = None
    minimum_free_shm = None
    maximum_slice = 0
    object_header = None
    wrote_object_row = False
    single_runner = HERE / "vendor/object/official_native_eval_run.py"
    snapshot_runner = HERE / "vendor/timeline/export_timeline"
    cleanup_runner = HERE / "vendor/cleanup/export_cleanup_binary"
    ghost_runner = HERE / "vendor/ghost/cleanup_mesh"
    geom_runner = HERE / "vendor/geometry/geometry_export"
    probes = FT / "results/abc_eval_final_baseline/temporal/RESIDUAL_PROBES.json"
    ghost_probes = FT / f"results/abc_eval_v2/cleanup/probes_{args.session}.json"

    with raw_path.open("w") as object_csv:
        timeline_file = timeline_path.open("w") if args.mode == "full" else None
        cleanup_file = cleanup_path.open("wb") if args.mode == "full" and args.session != "a" else None
        ghost_file = ghost_path.open("w") if args.mode == "full" and args.session != "a" else None
        try:
            if cleanup_file:
                cleanup_file.write(MAGIC + struct.pack("<I", count))
            jobs = int(os.environ.get("EVAL_FAST_JOBS", "8"))   # snapshots are independent
            context = dict(source=str(source), moments=moments, stamps=stamps, count=count,
                           mode=args.mode, session=args.session, cleanup=bool(cleanup_file), work=str(work),
                           geo=str(geo), probes=str(probes), ghost_probes=str(ghost_probes))
            done = {}
            next_write = 0
            with ProcessPoolExecutor(jobs) as pool:
                futures = {pool.submit(score_snapshot, i, context): i for i in indices}
                for step, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    done[result["i"]] = result
                    free_ram, free_shm = memory_guard(largest, swap_free)
                    minimum_free_ram = min(minimum_free_ram or free_ram, free_ram)
                    minimum_free_shm = min(minimum_free_shm or free_shm, free_shm)
                    maximum_slice = max(maximum_slice, result["slice_size"])
                    # write every finished snapshot in index order
                    while next_write < len(indices) and indices[next_write] in done:
                        r = done.pop(indices[next_write])
                        i = r["i"]
                        lines = r["object_lines"]
                        if object_header is None:
                            object_header = lines[0]
                            object_csv.write(object_header + "\n")
                        elif lines[0] != object_header:
                            raise RuntimeError("Official object CSV schema changed")
                        row = str(i) + lines[1][1:]
                        repeats = i + 1 if args.mode == "full" else 1
                        for _ in range(repeats):
                            if wrote_object_row:
                                object_csv.write("\n")
                            object_csv.write(row)
                            wrote_object_row = True
                        if timeline_file:
                            timeline_file.write(r["timeline"])
                        if cleanup_file:
                            cleanup_file.write(r["cleanup"])
                            ghost_file.write(r["ghost"])
                        next_write += 1
                    elapsed = time.monotonic() - started
                    print(f"{step}/{len(indices)} snapshot={result['i']} elapsed={elapsed:.1f}s", flush=True)
        finally:
            if timeline_file:
                timeline_file.close()
            if cleanup_file:
                cleanup_file.close()
            if ghost_file:
                ghost_file.close()

    mesh = work / "geometry_viewer.ply"
    if mesh.exists():
        session = args.session.upper()
        g2 = geo / "geometry_viewer_1cm.json"
        run([PY_METRIC, GEO / "score_geometry.py", session, args.name,
             mesh, "--json", g2])
        g1 = geo / "geometry_viewer_G1.json"
        run([PY_METRIC, GEO / "geometry_g1.py", "score", "real", session,
             FT / f"results/abc_eval_v2/geometry/{session}_reference_1cm.ply",
             mesh, FT / f"eval/crops/real_{session}.json", g1])

    if args.mode == "full":
        (exp / "COMPLETE.json").write_text(json.dumps({
            "map": str(source), "snapshots": count,
            "provenance": "lossless-zpk-per-snapshot-copied-exporters"}, indent=2) + "\n")
        explanation = obj / "explanation"
        run([PY_METRIC, HERE / "vendor/object/official_native_eval_postprocess.py",
             raw_path, "--output", explanation])
        run([PY_METRIC, HARNESS / "object_weighted_nr.py", "--rows",
             explanation / "native_unique_online_rows.csv", "--native-summary",
             explanation / "NATIVE_ONLINE_SUMMARY.json", "--snapshots", timeline_path,
             "--out", obj / "weighted_nr"])
        if args.session != "a":
            run([PY_METRIC, HARNESS / "cleanup_eval_h.py", "--session",
                 args.session.upper(), "--snapshots", timeline_path, "--bin", cleanup_path,
                 "--out", changes])
        state = [PY_METRIC, HARNESS / "eval_state_change_h.py", "--session",
                 args.session.upper(), "--snapshots", timeline_path]
        if args.session != "a":
            state += ["--cleanup", changes / f"CLEANUP_PROBES_{args.session.upper()}.csv"]
        run(state + ["--out", changes])
        if args.session != "a":
            run([PY_METRIC, HARNESS / "cleanup_evaluate_h.py", "--session",
                 args.session.upper(), "--mesh-jsonl", ghost_path,
                 "--expected-snapshots", str(count), "--out", ghost])
            run([PY_METRIC, HARNESS / "ghost_cohort.py", "--summary",
                 ghost / f"event_summary_{args.session}.csv", "--out", ghost / "GHOST.json"])
        summary = [PY_METRIC, HARNESS / "summarize_session.py", "--session",
                   args.session, "--name", args.name, "--real-abc", ra,
                   "--map", source, "--geom-policies", "viewer"]
        if args.prior:
            summary += ["--prior", args.prior.resolve(strict=True), "--table2"]
        run(summary)

    comparison = {}
    if args.compare_archive and args.mode == "full":
        reference = args.compare_archive.resolve(strict=True)
        candidates = [
            (obj / "results/static_objects.csv", reference / f"objects/{tag}/results/static_objects.csv"),
            (timeline_path, reference / f"exports/{tag}/snapshots.jsonl"),
        ]
        if args.session != "a":
            candidates.extend([
                (cleanup_path, reference / f"exports/{tag}/cleanup_probes.bin"),
                (ghost_path, reference / f"exports/{tag}/cleanup_mesh.jsonl"),
            ])
        for produced, archived in candidates:
            if archived.is_file():
                comparison[produced.name] = digest(produced) == digest(archived)
        if comparison and not all(comparison.values()):
            raise RuntimeError(f"Archived native export mismatch: {comparison}")

    provenance = {
        "mode": args.mode, "source_zpk": str(source),
        "virtual_original_sha256": manifest["source"]["sha256"],
        "session": args.session, "name": args.name, "moments": [first, last], "indices": indices,
        "snapshot_count": count, "official_binary_sha256": digest(
            HERE / "official/install/lib/khronos_eval/exp_pipeline"),
        "maximum_one_snapshot_mb": round(maximum_slice / 1e6, 1),
        "minimum_mem_available_gib": round(minimum_free_ram / (1 << 30), 2),
        "minimum_shm_free_gib": round(minimum_free_shm / (1 << 30), 2),
        "swap_consumed_during_run": RESOURCE_PEAK_SWAP > 0,
        "swap_monitor_scope": "evaluation_process_and_descendants",
        "swap_monitor_sampling_s": 0.25,
        "maximum_observed_evaluator_swap_bytes": RESOURCE_PEAK_SWAP,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "archive_native_export_parity": comparison,
    }
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    shutil.rmtree(work)
    print(out)


if __name__ == "__main__":
    main()
