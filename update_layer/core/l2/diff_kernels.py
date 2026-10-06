"""Differential test of state.py's geometric kernels against the original C++ text.

  python -m update_layer.core.l2.diff_kernels [WORKDIR]

sharedSpaceProbability, extentSameSiteProbability (persistent_object_state.cpp @192c1cf lines 79-169),
motionGeometryBayesFactor, kOwnCellFirst and offStateShare (452-545) are cut verbatim from git, compiled
against minimal Mesh/BoundingBox stubs (identity box frame) and run on object-like point sets (box
surfaces, 5 mm - 2 cm spacing, 2 mm noise, shifted 0 - 3 m and rotated, partial, identical, single-point
and empty copies, resolutions 0.05 / 0.1). The port must reproduce every value bit for bit; the Bayes
factor (scipy betainc against boost ibetac) to 1e-13 relative.
"""
from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .state import extent_same_site_probability, motion_geometry_bayes_factor, off_state_share, \
    shared_space_probability

SRC = "192c1cf:session_update_baseline/session_core/src/state/persistent_object_state.cpp"
HEAD = """#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <map>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <boost/math/special_functions/beta.hpp>
using Point = Eigen::Vector3f;
namespace spark_dsg { struct Mesh { std::vector<Point> points; size_t numVertices() const { return points.size(); } }; }
struct BoundingBox { Point pointToWorldFrame(const Point& p) const { return p; } };
struct PersistentObjectState {
  float map_resolution_ = 0.05f;
  static double motionGeometryBayesFactor(double off, double effective_cells);
  double offStateShare(const spark_dsg::Mesh& copy, const BoundingBox& copy_box,
                       const spark_dsg::Mesh& reference, const BoundingBox& reference_box,
                       const float tolerance, double& effective_cells) const;
};
constexpr double kStateTolerance = 0.10;
namespace {
"""
MAIN = """
static void rd(void* p, size_t n, FILE* f) { if (std::fread(p, 1, n, f) != n) std::exit(2); }
int main(int, char** argv) {
  FILE* in = std::fopen(argv[1], "rb");
  FILE* out = std::fopen(argv[2], "w");
  uint32_t cases;
  rd(&cases, 4, in);
  for (uint32_t c = 0; c < cases; ++c) {
    spark_dsg::Mesh a, b;
    for (auto* m : {&a, &b}) {
      uint32_t n; rd(&n, 4, in);
      m->points.resize(n);
      for (auto& p : m->points) rd(p.data(), 12, in);
    }
    float res; double decision;
    rd(&res, 4, in); rd(&decision, 8, in);
    BoundingBox box;
    PersistentObjectState s; s.map_resolution_ = res;
    const double shared = sharedSpaceProbability(a, box, b, box, res, decision);
    const double extent = extentSameSiteProbability(a, box, b, box, res);
    double eff = 0.0;
    const double off = s.offStateShare(b, box, a, box, kStateTolerance, eff);
    const double bf = PersistentObjectState::motionGeometryBayesFactor(off, eff);
    std::fprintf(out, "%.17g %.17g %.17g %.17g %.17g\\n", shared, extent, off, eff, bf);
  }
  return 0;
}
"""


def _cases(rng, count=400):
    def box_surface(size, spacing):
        pts = []
        for ax in range(3):
            for side in (0, 1):
                o = [a for a in range(3) if a != ax]
                U, V = np.meshgrid(np.arange(0, size[o[0]], spacing), np.arange(0, size[o[1]], spacing))
                P = np.zeros((U.size, 3))
                P[:, o[0]], P[:, o[1]], P[:, ax] = U.ravel(), V.ravel(), side * size[ax]
                pts.append(P)
        return np.concatenate(pts)

    def rot(deg):
        a = np.radians(deg) * rng.normal(size=3) / np.sqrt(3)
        th = np.linalg.norm(a)
        if th == 0:
            return np.eye(3)
        k = a / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

    out = []
    for c in range(count):
        A = box_surface(rng.uniform(0.1, 1.0, 3), rng.choice([0.005, 0.01, 0.02])) + rng.normal(0, 0.002, (1, 3))
        A = A[rng.random(len(A)) < rng.uniform(0.3, 1.0)]
        origin = rng.uniform(-5, 5, 3)
        d = rng.choice([0, 0.002, 0.01, 0.03, 0.05, 0.08, 0.11, 0.15, 0.3, 1.0, 3.0])
        u = rng.normal(size=3)
        B = (A - A.mean(0)) @ rot(rng.choice([0, 0, 5, 20])).T + A.mean(0) + d * u / np.linalg.norm(u) + \
            rng.normal(0, 0.002, A.shape)
        B = B[rng.random(len(B)) < rng.uniform(0.2, 1.0)]
        if c % 17 == 0:
            B = A.copy()
        if c % 23 == 0:
            B = B[:1]
        if c % 29 == 0:
            B = B[:0]
        out.append(((A + origin).astype(np.float32), (B + origin).astype(np.float32),
                    np.float32(rng.choice([0.05, 0.1])), float(rng.choice([0.25, 0.5, 0.75, 0.9, 0.975, 0.99]))))
    return out


def main():
    work = Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp())
    src = subprocess.run(["git", "show", SRC], capture_output=True, text=True, check=True).stdout.splitlines()
    body = "\n".join(src[78:169]) + "\n}  // namespace\n" + "\n".join(src[451:545])
    (work / "kernels.cpp").write_text(HEAD + body + MAIN)
    subprocess.run(["g++", "-O2", "-std=c++17", "-I/usr/include/eigen3", str(work / "kernels.cpp"), "-o",
                    str(work / "kernels")], check=True)
    cases = _cases(np.random.default_rng(20261006))
    with open(work / "in.bin", "wb") as f:
        f.write(struct.pack("<I", len(cases)))
        for A, B, res, dec in cases:
            for M in (A, B):
                f.write(struct.pack("<I", len(M)))
                f.write(M.astype("<f4").tobytes())
            f.write(struct.pack("<f", res))
            f.write(struct.pack("<d", dec))
    subprocess.run([str(work / "kernels"), str(work / "in.bin"), str(work / "out.txt")], check=True)
    cpp = np.loadtxt(work / "out.txt")
    py = []
    for A, B, res, dec in cases:
        off, eff = off_state_share(B, A, 0.10, float(res))
        py.append((shared_space_probability(A, B, float(res), dec), extent_same_site_probability(A, B, float(res)),
                   off, eff, motion_geometry_bayes_factor(off, eff)))
    py = np.array(py)
    ok = True
    for j, name in enumerate(["shared_space", "extent_same_site", "off_share", "effective_cells"]):
        exact = int((py[:, j] == cpp[:, j]).sum())
        ok &= exact == len(cases)
        print(f"{name:17s} bit-exact {exact}/{len(cases)}")
    big = np.abs(cpp[:, 4]) > 1e-200
    rel = np.abs(py[:, 4] - cpp[:, 4])[big] / np.abs(cpp[:, 4])[big]
    ok &= rel.max() < 1e-12
    print(f"bayes_factor      max rel {rel.max():.2g} over {int(big.sum())} values > 1e-200")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
