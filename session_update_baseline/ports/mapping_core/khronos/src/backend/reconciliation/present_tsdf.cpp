#include "khronos/backend/reconciliation/present_tsdf.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
#include <unordered_set>

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

void PresentTsdf::integrate(const Camera& camera,
                            const Eigen::Isometry3d& world_T_sensor,
                            const std::vector<float>& depth_z,
                            const std::vector<float>& ray_norm) {
  const int W = static_cast<int>(camera.width), H = static_cast<int>(camera.height);
  // 1) Units touched by the stride-4 pixels (+-T box), as Open3D does with
  // CreateFromDepthImage(stride = 4) and LocateVolumeUnit.
  std::unordered_set<Eigen::Vector3i, KeyHash> touched;
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
          for (int c = lo[2]; c <= hi[2]; ++c) touched.insert(Eigen::Vector3i(a, b, c));
    }
  }
  if (touched.empty()) return;
  std::vector<Unit*> units;
  units.reserve(touched.size());
  for (const auto& index : touched) units.push_back(openUnit(index));

  // 2) Projective update of every voxel of the touched units
  // (UniformTSDFVolume::IntegrateWithDepthToCameraDistanceMultiplier).
  const Eigen::Matrix4d extrinsic = world_T_sensor.inverse().matrix();
  const Eigen::Matrix4f extrinsic_f = extrinsic.cast<float>();
  const float voxel_f = static_cast<float>(voxel_);
  const float half_f = voxel_f * 0.5f;
  const float trunc_f = static_cast<float>(truncation_);
  const float trunc_inv_f = 1.0f / trunc_f;
  const Eigen::Matrix4f extrinsic_scaled_f = extrinsic_f * voxel_f;
  const float safe_width_f = static_cast<float>(W) - 0.0001f;
  const float safe_height_f = static_cast<float>(H) - 0.0001f;
  const float fx = camera.fx, fy = camera.fy, cx = camera.cx, cy = camera.cy;
  parallelRange(units.size(), num_threads_, [&](size_t begin, size_t end) {
    for (size_t k = begin; k < end; ++k) {
      Unit& unit = *units[k];
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
            const int u = static_cast<int>(u_f), v = static_cast<int>(v_f);
            const size_t pix = static_cast<size_t>(v) * W + u;
            const float d = depth_z[pix];
            if (!(d > 0.0f)) continue;
            const float sdf = (d - pt(2)) * ray_norm[pix];
            if (sdf > -trunc_f) {
              const int vi = idx_shift + z;
              const float tsdf = std::min(1.0f, sdf * trunc_inv_f);
              float& F = unit.tsdf[vi];
              float& Wt = unit.weight[vi];
              F = (F * Wt + tsdf) / (Wt + 1.0f);
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
