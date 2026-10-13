"""Differential test of classify_measurements (l2/evidence.py) against the original C++ text.

  python -m update_layer.core.l2.diff_classify [WORKDIR]

projectedRelationProbabilities and classifyMeasurement (projected_physical_evidence.cpp @192c1cf lines
20-86, with the endpoint structs of physical_evidence_store.h) are cut verbatim from git and compiled;
both sides classify the same measurements: stored ranges of real frames (datasets/local_ab session A
depth, millimetres as PhysicalEvidenceStore stores them, /1000 in float), query ranges at uniform
offsets and at offsets crowding the decision boundaries (+-tolerance, +-1 mm quantum, 0, in 1e-7 m
steps), every endpoint class, own / other / no identity. Every vote must be equal.
"""
from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from .evidence import classify_measurements

SRC = "192c1cf:session_update_baseline/session_core/src/evidence/projected_physical_evidence.cpp"
HDR = "192c1cf:session_update_baseline/session_core/include/session_core/evidence/physical_evidence_store.h"
FT = Path(__file__).resolve().parents[4]
HEAD = """#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>
#include <Eigen/Dense>
namespace khronos {
"""
MAIN = """
}  // namespace
}  // namespace khronos
using namespace khronos;
static void rd(void* p, size_t n, FILE* f) { if (std::fread(p, 1, n, f) != n) std::exit(2); }
int main(int, char** argv) {
  FILE* in = std::fopen(argv[1], "rb");
  FILE* out = std::fopen(argv[2], "wb");
  uint32_t n; rd(&n, 4, in);
  float tolerance; rd(&tolerance, 4, in);
  for (uint32_t i = 0; i < n; ++i) {
    int32_t type, pid, id; float meas, query;
    rd(&type, 4, in); rd(&pid, 4, in); rd(&id, 4, in); rd(&meas, 4, in); rd(&query, 4, in);
    ProjectedEndpointEvidence p;
    p.endpoint.type = static_cast<EndpointClass>(type);
    p.endpoint.physical_id = pid;
    p.endpoint.measured_depth_m = meas;
    p.query_range_m = query;
    const int8_t v = static_cast<int8_t>(classifyMeasurement(p, static_cast<size_t>(id), tolerance));
    std::fwrite(&v, 1, 1, out);
  }
  return 0;
}
"""


def main():
    work = Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp())
    src = subprocess.run(["git", "show", SRC], capture_output=True, text=True, check=True).stdout.splitlines()
    hdr = subprocess.run(["git", "show", HDR], capture_output=True, text=True, check=True).stdout.splitlines()
    first = next(i for i, l in enumerate(hdr) if l.startswith("enum class EndpointClass"))
    last = next(i for i, l in enumerate(hdr) if l.startswith("ProjectedRelationProbabilities projectedRelation"))
    start = next(i for i, l in enumerate(src) if l.startswith("ProjectedRelationProbabilities projectedRelationProbabilities("))
    end = next(i for i, l in enumerate(src) if l.startswith("Vote classifyMeasurement"))
    end = next(i for i in range(end, len(src)) if src[i] == "}")
    (work / "classify.cpp").write_text(HEAD + "\n".join(hdr[first:last]) + "\n" + "\n".join(src[start:end + 1]) + MAIN)
    subprocess.run(["g++", "-O2", "-std=c++17", "-I/usr/include/eigen3", str(work / "classify.cpp"), "-o",
                    str(work / "classify")], check=True)

    import tifffile
    rgbd = sorted((FT / "datasets/local_ab/rgbd").glob("session_a_*"))[0]
    mm = np.concatenate([np.round(tifffile.imread(rgbd / f"{k:06d}_depth.tiff").astype(np.float64).ravel()
                                  * (1000.0 if tifffile.imread(rgbd / f"{k:06d}_depth.tiff").max() < 100 else 1.0))
                         for k in (0, 300, 900)])
    mm = mm[(mm > 0) & np.isfinite(mm)].astype(np.int64)
    rng = np.random.default_rng(20261006)
    n = 3_000_000
    tolerance = np.float32(0.05)
    meas = (rng.choice(mm, n).astype(np.float32) / np.float32(1000.0)).astype(np.float32)
    edges = np.array([0.0, 0.05, -0.05, 0.001, -0.001, 0.0005, -0.0005, 0.0505, -0.0505, 0.0495, -0.0495])
    off = np.where(rng.random(n) < 0.5, rng.uniform(-0.2, 0.2, n),
                   rng.choice(edges, n) + rng.integers(-30, 31, n) * 1e-7)
    query = (meas.astype(np.float64) - off).astype(np.float32)
    meas[rng.random(n) < 0.01] = np.nan
    etype = rng.integers(0, 5, n).astype(np.int32)
    pid = np.where(etype == 4, rng.integers(0, 4, n), 0).astype(np.int32)
    ident = rng.integers(1, 4, n).astype(np.int32)
    with open(work / "in.bin", "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(struct.pack("<f", float(tolerance)))
        rec = np.zeros(n, dtype=[("t", "<i4"), ("p", "<i4"), ("i", "<i4"), ("m", "<f4"), ("q", "<f4")])
        rec["t"], rec["p"], rec["i"], rec["m"], rec["q"] = etype, pid, ident, meas, query
        f.write(rec.tobytes())
    subprocess.run([str(work / "classify"), str(work / "in.bin"), str(work / "out.bin")], check=True)
    cpp = np.fromfile(work / "out.bin", dtype=np.int8)
    py = np.empty(n, np.int8)
    for i in np.unique(ident):
        m = ident == i
        p = dict(etype=torch.as_tensor(etype[m]).to(torch.int8), pid=torch.as_tensor(pid[m]).to(torch.int64),
                 measured=torch.as_tensor(meas[m]), query=torch.as_tensor(query[m]))
        py[m] = classify_measurements(p, int(i), float(tolerance)).numpy()
    bad = np.nonzero(py != cpp)[0]
    print(f"votes equal {n - len(bad)}/{n}; near-boundary samples {int((np.abs(np.abs(off) - 0.05) < 1e-5).sum() + (np.abs(np.abs(off) - 0.001) < 1e-5).sum())}")
    for b in bad[:10]:
        print("  ", etype[b], pid[b], ident[b], repr(meas[b]), repr(query[b]), "py", py[b], "cpp", cpp[b])
    return 0 if not len(bad) else 1


if __name__ == "__main__":
    sys.exit(main())
