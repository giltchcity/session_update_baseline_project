#include "session_core/surface/triangle_grid.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace khronos {

Eigen::Vector3f closestPointOnTriangle(const Eigen::Vector3f& p,
                                       const Eigen::Vector3f& a,
                                       const Eigen::Vector3f& b,
                                       const Eigen::Vector3f& c) {
  const Eigen::Vector3f ab = b - a, ac = c - a, ap = p - a;
  const float d1 = ab.dot(ap), d2 = ac.dot(ap);
  if (d1 <= 0.f && d2 <= 0.f) return a;
  const Eigen::Vector3f bp = p - b;
  const float d3 = ab.dot(bp), d4 = ac.dot(bp);
  if (d3 >= 0.f && d4 <= d3) return b;
  const float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) return a + ab * (d1 / (d1 - d3));
  const Eigen::Vector3f cp = p - c;
  const float d5 = ab.dot(cp), d6 = ac.dot(cp);
  if (d6 >= 0.f && d5 <= d6) return c;
  const float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) return a + ac * (d2 / (d2 - d6));
  const float va = d3 * d6 - d5 * d4;
  if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
    return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
  }
  const float denom = 1.f / (va + vb + vc);
  return a + ab * (vb * denom) + ac * (vc * denom);
}

namespace {
constexpr int64_t kOffset = int64_t(1) << 20;
}

TriangleGrid::Key TriangleGrid::key(int x, int y, int z) const {
  return (static_cast<uint64_t>(x + kOffset) << 42) | (static_cast<uint64_t>(y + kOffset) << 21) |
         static_cast<uint64_t>(z + kOffset);
}

Eigen::Vector3i TriangleGrid::cellOf(const Eigen::Vector3f& p) const {
  return Eigen::Vector3i(static_cast<int>(std::floor(p.x() / cell_)),
                         static_cast<int>(std::floor(p.y() / cell_)),
                         static_cast<int>(std::floor(p.z() / cell_)));
}

const std::vector<uint32_t>* TriangleGrid::cellFaces(int x, int y, int z) const {
  const auto it = cells_.find(key(x, y, z));
  return it == cells_.end() ? nullptr : &it->second;
}

TriangleGrid::TriangleGrid(const std::vector<Eigen::Vector3f>& vertices,
                           const std::vector<Face>& faces,
                           const std::vector<uint32_t>* subset,
                           float cell)
    : vertices_(vertices), faces_(faces), cell_(cell) {
  bounds_.setEmpty();
  auto add = [&](uint32_t f) {
    const auto& t = faces_[f];
    if (t[0] >= vertices_.size() || t[1] >= vertices_.size() || t[2] >= vertices_.size()) return;
    Eigen::AlignedBox3f box(vertices_[t[0]]);
    box.extend(vertices_[t[1]]);
    box.extend(vertices_[t[2]]);
    if (!box.min().allFinite() || !box.max().allFinite()) return;
    const Eigen::Vector3i lo = cellOf(box.min()), hi = cellOf(box.max());
    for (int x = lo.x(); x <= hi.x(); ++x)
      for (int y = lo.y(); y <= hi.y(); ++y)
        for (int z = lo.z(); z <= hi.z(); ++z) cells_[key(x, y, z)].push_back(f);
    if (num_registered_ == 0) {
      min_cell_ = lo;
      max_cell_ = hi;
    } else {
      min_cell_ = min_cell_.cwiseMin(lo);
      max_cell_ = max_cell_.cwiseMax(hi);
    }
    bounds_.extend(box);
    ++num_registered_;
  };
  if (subset) {
    for (const auto f : *subset) add(f);
  } else {
    for (uint32_t f = 0; f < faces_.size(); ++f) add(f);
  }
}

bool TriangleGrid::closest(const Eigen::Vector3f& p,
                           float r_max,
                           float& distance,
                           Eigen::Vector3f& point,
                           uint32_t& face,
                           const std::function<bool(uint32_t, const Eigen::Vector3f&)>& accept) const {
  if (empty()) return false;
  const Eigen::Vector3i c0 = cellOf(p);
  float best_sq = std::numeric_limits<float>::infinity();
  bool found = false;
  // Rings needed to cover the whole grid from c0.
  const int max_ring = std::max({std::abs(c0.x() - min_cell_.x()), std::abs(c0.x() - max_cell_.x()),
                                 std::abs(c0.y() - min_cell_.y()), std::abs(c0.y() - max_cell_.y()),
                                 std::abs(c0.z() - min_cell_.z()), std::abs(c0.z() - max_cell_.z())});
  auto visit = [&](int x, int y, int z) {
    const auto* list = cellFaces(x, y, z);
    if (!list) return;
    for (const auto f : *list) {
      const auto& t = faces_[f];
      const Eigen::Vector3f q =
          closestPointOnTriangle(p, vertices_[t[0]], vertices_[t[1]], vertices_[t[2]]);
      const float d_sq = (q - p).squaredNorm();
      if (d_sq <= r_max * r_max &&
          (d_sq < best_sq || (found && d_sq == best_sq && f < face)) &&
          (!accept || accept(f, q))) {
        best_sq = d_sq;
        point = q;
        face = f;
        found = true;
      }
    }
  };
  for (int k = 0; k <= max_ring; ++k) {
    // Cells of Chebyshev ring k around c0, clipped to the occupied cell range.
    for (int dx = -k; dx <= k; ++dx) {
      const int x = c0.x() + dx;
      if (x < min_cell_.x() || x > max_cell_.x()) continue;
      for (int dy = -k; dy <= k; ++dy) {
        const int y = c0.y() + dy;
        if (y < min_cell_.y() || y > max_cell_.y()) continue;
        const bool edge_xy = std::abs(dx) == k || std::abs(dy) == k;
        if (edge_xy) {
          for (int dz = -k; dz <= k; ++dz) {
            const int z = c0.z() + dz;
            if (z < min_cell_.z() || z > max_cell_.z()) continue;
            visit(x, y, z);
          }
        } else {
          for (const int dz : {-k, k}) {
            const int z = c0.z() + dz;
            if (z < min_cell_.z() || z > max_cell_.z()) continue;
            visit(x, y, z);
            if (k == 0) break;
          }
        }
      }
    }
    // Every unvisited triangle lies outside the ring-k cube: >= k * cell away.
    const float reach = static_cast<float>(k) * cell_;
    if (found && best_sq <= reach * reach) break;
    if (reach >= r_max) break;
  }
  if (!found) return false;
  distance = std::sqrt(best_sq);
  return distance <= r_max;
}

bool TriangleGrid::firstHit(const Eigen::Vector3f& origin,
                            const Eigen::Vector3f& direction,
                            float& t_out,
                            uint32_t& face_out) const {
  if (empty()) return false;
  // Slab test against the registered bounds.
  double t_enter = 0.0, t_exit = std::numeric_limits<double>::infinity();
  const Eigen::Vector3d o = origin.cast<double>(), d = direction.cast<double>();
  const Eigen::Vector3d lo = bounds_.min().cast<double>(), hi = bounds_.max().cast<double>();
  for (int k = 0; k < 3; ++k) {
    if (d[k] == 0.0) {
      if (o[k] < lo[k] || o[k] > hi[k]) return false;
      continue;
    }
    double t0 = (lo[k] - o[k]) / d[k], t1 = (hi[k] - o[k]) / d[k];
    if (t0 > t1) std::swap(t0, t1);
    t_enter = std::max(t_enter, t0);
    t_exit = std::min(t_exit, t1);
    if (t_enter > t_exit) return false;
  }
  // 3D-DDA from the entry point.
  const double cell = cell_;
  const Eigen::Vector3d start = o + d * t_enter;
  Eigen::Vector3i c(static_cast<int>(std::floor(start.x() / cell)),
                    static_cast<int>(std::floor(start.y() / cell)),
                    static_cast<int>(std::floor(start.z() / cell)));
  c = c.cwiseMax(min_cell_).cwiseMin(max_cell_);
  Eigen::Vector3i step;
  Eigen::Vector3d t_max, t_delta;
  for (int k = 0; k < 3; ++k) {
    if (d[k] > 0) {
      step[k] = 1;
      t_max[k] = ((c[k] + 1) * cell - o[k]) / d[k];
      t_delta[k] = cell / d[k];
    } else if (d[k] < 0) {
      step[k] = -1;
      t_max[k] = (c[k] * cell - o[k]) / d[k];
      t_delta[k] = -cell / d[k];
    } else {
      step[k] = 0;
      t_max[k] = std::numeric_limits<double>::infinity();
      t_delta[k] = std::numeric_limits<double>::infinity();
    }
  }
  double best_t = std::numeric_limits<double>::infinity();
  uint32_t best_face = 0;
  while (true) {
    const auto* list = cellFaces(c.x(), c.y(), c.z());
    if (list) {
      for (const auto f : *list) {
        const auto& tri = faces_[f];
        const Eigen::Vector3d a = vertices_[tri[0]].cast<double>();
        const Eigen::Vector3d e1 = vertices_[tri[1]].cast<double>() - a;
        const Eigen::Vector3d e2 = vertices_[tri[2]].cast<double>() - a;
        const Eigen::Vector3d pvec = d.cross(e2);
        const double det = e1.dot(pvec);
        if (det == 0.0) continue;
        const double inv = 1.0 / det;
        const Eigen::Vector3d tvec = o - a;
        const double u = tvec.dot(pvec) * inv;
        if (u < 0.0 || u > 1.0) continue;
        const Eigen::Vector3d qvec = tvec.cross(e1);
        const double v = d.dot(qvec) * inv;
        if (v < 0.0 || u + v > 1.0) continue;
        const double t = e2.dot(qvec) * inv;
        if (t > 0.0 && (t < best_t || (t == best_t && f < best_face))) {
          best_t = t;
          best_face = f;
        }
      }
    }
    const double cell_exit = std::min({t_max.x(), t_max.y(), t_max.z()});
    // At a cell boundary another cell can contain the same first hit. Visit
    // those cells as well so exact distance ties always select the lower face.
    if (best_t < cell_exit) break;
    // Advance to the next cell.
    int axis = 0;
    if (t_max.y() < t_max[axis]) axis = 1;
    if (t_max.z() < t_max[axis]) axis = 2;
    if (t_max[axis] > t_exit + cell) break;
    c[axis] += step[axis];
    if (c[axis] < min_cell_[axis] || c[axis] > max_cell_[axis]) break;
    t_max[axis] += t_delta[axis];
  }
  if (!std::isfinite(best_t)) return false;
  t_out = static_cast<float>(best_t);
  face_out = best_face;
  return true;
}

}  // namespace khronos
