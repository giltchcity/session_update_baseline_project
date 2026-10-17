#include "session_core/surface/present_tsdf.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <thread>

#include <hydra/reconstruction/marching_cubes.h>

namespace khronos {

namespace {

constexpr int kRes = PresentTsdf::kUnitResolution;
constexpr int kVoxels = kRes * kRes * kRes;

inline int indexOf(int x, int y, int z) { return x * kRes * kRes + y * kRes + z; }

// Corner offsets (Bourke / Open3D order) and, per edge, its lower corner and
// axis (Open3D MarchingCubesConst edge_shift / edge_to_vert).
constexpr int kShift[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                              {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
constexpr int kEdgeShift[12][4] = {{0, 0, 0, 0}, {1, 0, 0, 1}, {0, 1, 0, 0}, {0, 0, 0, 1},
                                   {0, 0, 1, 0}, {1, 0, 1, 1}, {0, 1, 1, 0}, {0, 0, 1, 1},
                                   {0, 0, 0, 2}, {1, 0, 0, 2}, {1, 1, 0, 2}, {0, 1, 0, 2}};
constexpr int kEdgeToVert[12][2] = {{0, 1}, {1, 2}, {3, 2}, {0, 3}, {4, 5}, {5, 6},
                                    {7, 6}, {4, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};

template <typename Fn>
void parallelRange(size_t n, int num_threads, Fn fn) {
  const size_t workers = std::max<size_t>(1, std::min<size_t>(std::max(1, num_threads), n));
  if (workers <= 1) {
    fn(0, n);
    return;
  }
  std::vector<std::thread> threads;
  std::atomic<size_t> next{0};
  constexpr size_t kChunk = 8;
  for (size_t w = 0; w < workers; ++w) {
    threads.emplace_back([&] {
      while (true) {
        const size_t begin = next.fetch_add(kChunk);
        if (begin >= n) break;
        fn(begin, std::min(n, begin + kChunk));
      }
    });
  }
  for (auto& t : threads) t.join();
}

}  // namespace

struct PresentTsdf::Unit {
  Eigen::Vector3i index;
  std::vector<float> tsdf;
  std::vector<float> weight;
  explicit Unit(const Eigen::Vector3i& i) : index(i), tsdf(kVoxels, 0.f), weight(kVoxels, 0.f) {}
};

size_t PresentTsdf::KeyHash::operator()(const Eigen::Vector3i& k) const {
  // Open3D hash_eigen style combination.
  size_t seed = 0;
  for (int i = 0; i < 3; ++i) {
    seed ^= std::hash<int>()(k[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}

PresentTsdf::PresentTsdf(double voxel, double truncation, int num_threads)
    : voxel_(voxel),
      truncation_(truncation),
      unit_length_(voxel * kRes),
      num_threads_(num_threads) {}

PresentTsdf::~PresentTsdf() = default;

PresentTsdf::Unit* PresentTsdf::openUnit(const Eigen::Vector3i& index) {
  auto it = units_.find(index);
  if (it != units_.end()) return it->second.get();
  auto unit = std::make_unique<Unit>(index);
  Unit* ptr = unit.get();
  units_.emplace(index, std::move(unit));
  return ptr;
}

size_t PresentTsdf::numUnits() const { return units_.size(); }

size_t PresentTsdf::numBytes() const {
  return units_.size() * (sizeof(Unit) + 2 * kVoxels * sizeof(float));
}

PresentTsdf::UnitSet PresentTsdf::activeUnits(const Camera& camera,
                                              const Eigen::Isometry3d& world_T_sensor,
                                              const std::vector<float>& depth_z) const {
  // Open3D CreateFromDepthImage(stride = 4) and LocateVolumeUnit: the +-T box
  // of every stride-4 pixel back-projected.
  const int W = static_cast<int>(camera.width), H = static_cast<int>(camera.height);
  UnitSet active;
  const double fxd = camera.fx, fyd = camera.fy, cxd = camera.cx, cyd = camera.cy;
  const Eigen::Matrix4d camera_pose = world_T_sensor.matrix();
  for (int v = 0; v < H; v += 4) {
    for (int u = 0; u < W; u += 4) {
      const float d = depth_z[static_cast<size_t>(v) * W + u];
      if (!(d > 0.f)) continue;
      const double z = d;
      const double x = (u - cxd) * z / fxd;
      const double y = (v - cyd) * z / fyd;
      const Eigen::Vector4d p = camera_pose * Eigen::Vector4d(x, y, z, 1.0);
      Eigen::Vector3i lo, hi;
      for (int k = 0; k < 3; ++k) {
        lo[k] = static_cast<int>(std::floor((p[k] - truncation_) / unit_length_));
        hi[k] = static_cast<int>(std::floor((p[k] + truncation_) / unit_length_));
      }
      for (int a = lo[0]; a <= hi[0]; ++a)
        for (int b = lo[1]; b <= hi[1]; ++b)
          for (int c = lo[2]; c <= hi[2]; ++c) active.insert(Eigen::Vector3i(a, b, c));
    }
  }
  return active;
}

void PresentTsdf::update(const Camera& camera,
                         const Eigen::Isometry3d& world_T_sensor,
                         const std::vector<float>& ray_norm,
                         const Domain& from,
                         const Domain& to) {
  const int W = static_cast<int>(camera.width), H = static_cast<int>(camera.height);
  const float inf = std::numeric_limits<float>::infinity();
  // 1) The units each domain updates: a voxel takes a frame's reading only in a
  // unit the frame's depth activates.
  const UnitSet none;
  const bool same_depth = from.depth_z && to.depth_z && *from.depth_z == *to.depth_z;
  const UnitSet from_units = from.depth_z ? activeUnits(camera, world_T_sensor, *from.depth_z) : none;
  const UnitSet own_to_units =
      to.depth_z && !same_depth ? activeUnits(camera, world_T_sensor, *to.depth_z) : none;
  const UnitSet& to_units = same_depth ? from_units : own_to_units;

  // 2) In a unit both domains activate, readings differ only on the pixels where
  // the domains differ: such a unit is visited only if its projection covers one
  // (integral image of those pixels).
  const Eigen::Matrix4d extrinsic = world_T_sensor.inverse().matrix();
  std::vector<uint32_t> changed;
  if (from.depth_z && to.depth_z) {
    changed.assign(static_cast<size_t>(W + 1) * (H + 1), 0);
    for (int v = 0; v < H; ++v) {
      uint32_t row = 0;
      for (int u = 0; u < W; ++u) {
        const size_t p = static_cast<size_t>(v) * W + u;
        const float limit_from = from.free_limit ? (*from.free_limit)[p] : inf;
        const float limit_to = to.free_limit ? (*to.free_limit)[p] : inf;
        row += (*from.depth_z)[p] != (*to.depth_z)[p] || limit_from != limit_to;
        changed[static_cast<size_t>(v + 1) * (W + 1) + u + 1] =
            changed[static_cast<size_t>(v) * (W + 1) + u + 1] + row;
      }
    }
  }
  auto coversChanged = [&](const Eigen::Vector3i& index) {
    double umin = inf, umax = -inf, vmin = inf, vmax = -inf;
    for (int c = 0; c < 8; ++c) {
      const Eigen::Vector4d corner = extrinsic * Eigen::Vector4d((index.x() + (c & 1)) * unit_length_,
                                                                 (index.y() + ((c >> 1) & 1)) * unit_length_,
                                                                 (index.z() + ((c >> 2) & 1)) * unit_length_,
                                                                 1.0);
      if (!(corner.z() > 0.0)) return true;  // reaches behind the camera
      const double uu = corner.x() * camera.fx / corner.z() + camera.cx + 0.5;
      const double vv = corner.y() * camera.fy / corner.z() + camera.cy + 0.5;
      umin = std::min(umin, uu);
      umax = std::max(umax, uu);
      vmin = std::min(vmin, vv);
      vmax = std::max(vmax, vv);
    }
    // One pixel wider than the voxels' rounded pixels can reach.
    const double u0 = std::max(0.0, std::floor(umin) - 1.0), u1 = std::min(W - 1.0, std::floor(umax) + 1.0);
    const double v0 = std::max(0.0, std::floor(vmin) - 1.0), v1 = std::min(H - 1.0, std::floor(vmax) + 1.0);
    if (u0 > u1 || v0 > v1) return false;
    const auto S = [&](double u, double v) {
      return changed[static_cast<size_t>(v) * (W + 1) + static_cast<size_t>(u)];
    };
    return S(u1 + 1, v1 + 1) + S(u0, v0) != S(u0, v1 + 1) + S(u1 + 1, v0);
  };
  struct Visit {
    Unit* unit;
    bool from, to;
  };
  std::vector<Visit> visits;
  auto consider = [&](const Eigen::Vector3i& index, bool in_from, bool in_to) {
    if (in_from && in_to && !coversChanged(index)) return;
    Unit* unit = nullptr;
    if (in_to) {
      unit = openUnit(index);
    } else {
      const auto it = units_.find(index);
      if (it == units_.end()) return;
      unit = it->second.get();
    }
    visits.push_back({unit, in_from, in_to});
  };
  for (const auto& index : from_units) consider(index, true, to_units.count(index) > 0);
  for (const auto& index : to_units) {
    if (!from_units.count(index)) consider(index, false, true);
  }
  if (visits.empty()) return;

  // 3) Projective reading of every voxel of the visited units under both domains
  // (UniformTSDFVolume::IntegrateWithDepthToCameraDistanceMultiplier), the
  // difference applied.
  const Eigen::Matrix4f extrinsic_f = extrinsic.cast<float>();
  const float voxel_f = static_cast<float>(voxel_);
  const float half_f = voxel_f * 0.5f;
  const float trunc_f = static_cast<float>(truncation_);
  const float trunc_inv_f = 1.0f / trunc_f;
  const Eigen::Matrix4f extrinsic_scaled_f = extrinsic_f * voxel_f;
  const float safe_width_f = static_cast<float>(W) - 0.0001f;
  const float safe_height_f = static_cast<float>(H) - 0.0001f;
  const float fx = camera.fx, fy = camera.fy, cx = camera.cx, cy = camera.cy;
  // The reading f = min(1, sdf / T) of pixel `pix` at depth z if eq. (8) admits it.
  auto reading = [&](const Domain& domain, size_t pix, float z, float& f) {
    const float d = (*domain.depth_z)[pix];
    if (!(d > 0.0f)) return false;
    const float sdf = (d - z) * ray_norm[pix];
    if (!(sdf > -trunc_f)) return false;
    if (domain.free_limit && sdf > trunc_f && z * ray_norm[pix] >= (*domain.free_limit)[pix] - trunc_f) {
      return false;
    }
    f = std::min(1.0f, sdf * trunc_inv_f);
    return true;
  };
  parallelRange(visits.size(), num_threads_, [&](size_t begin, size_t end) {
    for (size_t k = begin; k < end; ++k) {
      const Visit& visit = visits[k];
      Unit& unit = *visit.unit;
      const Eigen::Vector3d origin = unit.index.cast<double>() * unit_length_;
      for (int x = 0; x < kRes; ++x) {
        for (int y = 0; y < kRes; ++y) {
          const int idx_shift = x * kRes * kRes + y * kRes;
          Eigen::Vector4f pt = extrinsic_f * Eigen::Vector4f(half_f + voxel_f * x + (float)origin(0),
                                                             half_f + voxel_f * y + (float)origin(1),
                                                             half_f + (float)origin(2), 1.0f);
          for (int z = 0; z < kRes; ++z, pt(0) += extrinsic_scaled_f(0, 2),
                   pt(1) += extrinsic_scaled_f(1, 2), pt(2) += extrinsic_scaled_f(2, 2)) {
            if (!(pt(2) > 0)) continue;
            const float u_f = pt(0) * fx / pt(2) + cx + 0.5f;
            const float v_f = pt(1) * fy / pt(2) + cy + 0.5f;
            if (!(u_f >= 0.0001f && u_f < safe_width_f && v_f >= 0.0001f && v_f < safe_height_f)) {
              continue;
            }
            const size_t pix = static_cast<size_t>(static_cast<int>(v_f)) * W + static_cast<int>(u_f);
            float f_from = 0.0f, f_to = 0.0f;
            const bool in_from = visit.from && reading(from, pix, pt(2), f_from);
            const bool in_to = visit.to && reading(to, pix, pt(2), f_to);
            if (in_from == in_to && (!in_from || f_from == f_to)) continue;
            float& F = unit.tsdf[idx_shift + z];
            float& Wt = unit.weight[idx_shift + z];
            if (in_from) {
              // Exact inverse of the equal-weight mean below.
              if (Wt > 1.0f) {
                F = (F * Wt - f_from) / (Wt - 1.0f);
                Wt -= 1.0f;
              } else {
                F = 0.0f;
                Wt = 0.0f;
              }
            }
            if (in_to) {
              F = (F * Wt + f_to) / (Wt + 1.0f);
              Wt += 1.0f;
            }
          }
        }
      }
    }
  });
}

std::vector<float> PresentTsdf::rayNorm(const Camera& camera) {
  std::vector<float> mult(static_cast<size_t>(camera.width) * camera.height);
  const float fx_inv = 1.0f / camera.fx, fy_inv = 1.0f / camera.fy;
  for (uint32_t v = 0; v < camera.height; ++v) {
    const float yy = (static_cast<float>(v) - camera.cy) * fy_inv;
    for (uint32_t u = 0; u < camera.width; ++u) {
      const float xx = (static_cast<float>(u) - camera.cx) * fx_inv;
      mult[static_cast<size_t>(v) * camera.width + u] = std::sqrt(xx * xx + yy * yy + 1.0f);
    }
  }
  return mult;
}

void PresentTsdf::depthFromRange(const Camera& camera,
                                 const std::vector<uint16_t>& range_mm,
                                 std::vector<float>& depth_z) {
  const size_t n = static_cast<size_t>(camera.width) * camera.height;
  // The per-pixel norm, computed once per camera (numpy float32 order of operations).
  thread_local std::vector<float> norm;
  thread_local Camera cached{};
  if (norm.size() != n || cached.fx != camera.fx || cached.fy != camera.fy ||
      cached.cx != camera.cx || cached.cy != camera.cy || cached.width != camera.width) {
    norm.resize(n);
    for (uint32_t v = 0; v < camera.height; ++v) {
      const float b = (static_cast<float>(v) - camera.cy) / camera.fy;
      for (uint32_t u = 0; u < camera.width; ++u) {
        const float a = (static_cast<float>(u) - camera.cx) / camera.fx;
        norm[static_cast<size_t>(v) * camera.width + u] = std::sqrt(a * a + b * b + 1.0f);
      }
    }
    cached = camera;
  }
  depth_z.resize(n);
  for (size_t p = 0; p < n; ++p) {
    depth_z[p] = range_mm[p] ? (static_cast<float>(range_mm[p]) * 0.001f) / norm[p] : 0.0f;
  }
}

void PresentTsdf::extractMesh(std::vector<Eigen::Vector3f>& vertices,
                              std::vector<Face>& faces) const {
  vertices.clear();
  faces.clear();
  // Edge mask per cube configuration, from the triangle table.
  static const std::array<uint16_t, 256> kEdgeMask = [] {
    std::array<uint16_t, 256> mask{};
    for (int c = 0; c < 256; ++c) {
      for (int i = 0; i < 16 && hydra::MarchingCubes::kTriangleTable[c][i] != -1; ++i) {
        mask[c] |= static_cast<uint16_t>(1u << hydra::MarchingCubes::kTriangleTable[c][i]);
      }
    }
    return mask;
  }();

  // Deterministic unit order (Open3D iterates its hash map; only the vertex /
  // face sets matter).
  std::vector<const Unit*> order;
  order.reserve(units_.size());
  for (const auto& [index, unit] : units_) order.push_back(unit.get());
  std::sort(order.begin(), order.end(), [](const Unit* a, const Unit* b) {
    return std::lexicographical_compare(a->index.data(), a->index.data() + 3, b->index.data(),
                                        b->index.data() + 3);
  });

  std::unordered_map<uint64_t, uint32_t> edge_vertex;
  edge_vertex.reserve(units_.size() * 64);
  auto edgeKey = [](int64_t gx, int64_t gy, int64_t gz, int axis) {
    constexpr int64_t kOff = int64_t(1) << 19;
    return (static_cast<uint64_t>(gx + kOff) << 42) | (static_cast<uint64_t>(gy + kOff) << 22) |
           (static_cast<uint64_t>(gz + kOff) << 2) | static_cast<uint64_t>(axis);
  };
  const double half = 0.5 * voxel_;

  for (const Unit* unit : order) {
    // Neighbour units (+x, +y, +z combinations) for corners outside the unit.
    const Unit* nb[2][2][2];
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b)
        for (int c = 0; c < 2; ++c) {
          if (!a && !b && !c) {
            nb[a][b][c] = unit;
            continue;
          }
          const auto it = units_.find(unit->index + Eigen::Vector3i(a, b, c));
          nb[a][b][c] = it == units_.end() ? nullptr : it->second.get();
        }
    for (int x = 0; x < kRes; ++x) {
      for (int y = 0; y < kRes; ++y) {
        for (int z = 0; z < kRes; ++z) {
          float f[8];
          int cube = 0;
          bool ok = true;
          for (int i = 0; i < 8; ++i) {
            int xi = x + kShift[i][0], yi = y + kShift[i][1], zi = z + kShift[i][2];
            const int a = xi >= kRes, b = yi >= kRes, c = zi >= kRes;
            const Unit* u = nb[a][b][c];
            if (!u) {
              ok = false;
              break;
            }
            if (a) xi -= kRes;
            if (b) yi -= kRes;
            if (c) zi -= kRes;
            const int vi = indexOf(xi, yi, zi);
            const float w = u->weight[vi];
            if (w == 0.0f) {
              ok = false;
              break;
            }
            f[i] = u->tsdf[vi];
            if (f[i] < 0.0f) cube |= (1 << i);
          }
          if (!ok || cube == 0 || cube == 255) continue;
          uint32_t edge_to_index[12];
          const uint16_t mask = kEdgeMask[cube];
          for (int e = 0; e < 12; ++e) {
            if (!(mask & (1u << e))) continue;
            const int64_t gx = static_cast<int64_t>(unit->index.x()) * kRes + x + kEdgeShift[e][0];
            const int64_t gy = static_cast<int64_t>(unit->index.y()) * kRes + y + kEdgeShift[e][1];
            const int64_t gz = static_cast<int64_t>(unit->index.z()) * kRes + z + kEdgeShift[e][2];
            const int axis = kEdgeShift[e][3];
            const uint64_t key = edgeKey(gx, gy, gz, axis);
            const auto it = edge_vertex.find(key);
            if (it != edge_vertex.end()) {
              edge_to_index[e] = it->second;
              continue;
            }
            Eigen::Vector3d pt(half + voxel_ * gx, half + voxel_ * gy, half + voxel_ * gz);
            const double f0 = std::abs(static_cast<double>(f[kEdgeToVert[e][0]]));
            const double f1 = std::abs(static_cast<double>(f[kEdgeToVert[e][1]]));
            pt(axis) += f0 * voxel_ / (f0 + f1);
            const uint32_t id = static_cast<uint32_t>(vertices.size());
            vertices.push_back(pt.cast<float>());
            edge_vertex.emplace(key, id);
            edge_to_index[e] = id;
          }
          const int* row = hydra::MarchingCubes::kTriangleTable[cube];
          for (int i = 0; i < 16 && row[i] != -1; i += 3) {
            const Face face = {edge_to_index[row[i]], edge_to_index[row[i + 2]],
                               edge_to_index[row[i + 1]]};
            // remove_degenerate_triangles (index-degenerate only).
            if (face[0] == face[1] || face[1] == face[2] || face[2] == face[0]) continue;
            faces.push_back(face);
          }
        }
      }
    }
  }
}

}  // namespace khronos
