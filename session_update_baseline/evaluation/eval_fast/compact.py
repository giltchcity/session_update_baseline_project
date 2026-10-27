#!/usr/bin/env python3
"""Compress finished eval_fast exports after checking their exact decompression hash.

Only files with fixed names under an eval_fast/outputs run are eligible. Scored
JSON/CSV summaries remain plain text. A compact run is no longer directly usable
as an input to the copied derivation scripts until its exports are restored.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
OUTPUTS = (ROOT / "outputs").resolve()
LARGE_NAMES = {"snapshots.jsonl", "cleanup_probes.bin", "cleanup_mesh.jsonl",
               "geometry_viewer.ply"}
ZSTD = shutil.which("zstd") or "/home/jixian/Desktop/miniconda3/bin/zstd"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def decoded_sha(path):
    h = hashlib.sha256()
    proc = subprocess.Popen([ZSTD, "-q", "-d", "-c", str(path)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    for chunk in iter(lambda: proc.stdout.read(1 << 20), b""):
        h.update(chunk)
    error = proc.stderr.read() if proc.stderr else b""
    if proc.wait() != 0:
        raise RuntimeError(f"zstd verification failed: {error.decode(errors='replace')}")
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    run = args.run_dir.resolve(strict=True)
    if not run.is_relative_to(OUTPUTS) or run == OUTPUTS:
        ap.error("Run directory must be inside this eval_fast/outputs")
    if not (run / "PROVENANCE.json").is_file():
        ap.error("Only completed eval_fast runs with PROVENANCE.json may be compacted")
    manifest_path = run / "COMPACTED_EXPORTS.json"
    if manifest_path.exists():
        ap.error("This run has already been compacted")
    exports = run / "real_abc/exports"
    files = sorted(p for p in exports.rglob("*") if p.is_file() and p.name in LARGE_NAMES)
    if not files:
        ap.error("No large exports found")
    records = []
    for source in files:
        if not source.resolve().is_relative_to(run):
            raise RuntimeError(f"Export escapes run directory: {source}")
        packed = source.with_name(source.name + ".zst")
        if packed.exists():
            raise RuntimeError(f"Existing compressed export: {packed}")
        original_hash = sha(source)
        with tempfile.NamedTemporaryFile(prefix=packed.name + ".", suffix=".part",
                                         dir=packed.parent, delete=False) as tmp:
            partial = Path(tmp.name)
        try:
            result = subprocess.run([ZSTD, "-q", "-f", "-T4", "-5", "-o", str(partial),
                                     str(source)], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr)
            if decoded_sha(partial) != original_hash:
                raise RuntimeError(f"Lossless verification failed for {source}")
            partial.replace(packed)
        finally:
            if partial.exists():
                partial.unlink()
        records.append({"original": str(source.relative_to(run)),
                        "original_bytes": source.stat().st_size,
                        "original_sha256": original_hash,
                        "compressed": str(packed.relative_to(run)),
                        "compressed_bytes": packed.stat().st_size,
                        "compressed_sha256": sha(packed)})
        # This is our own already-verified export, never a map or user input.
        source.unlink()
        print(f"{records[-1]['original_bytes']/1e6:.1f} MB -> "
              f"{records[-1]['compressed_bytes']/1e6:.1f} MB {packed}", flush=True)
    manifest_path.write_text(json.dumps({"schema": "eval-fast-compact-v1",
                                         "files": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
