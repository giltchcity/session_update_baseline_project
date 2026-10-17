#include "session_core/surface/present_surface.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <thread>
#include <unordered_map>

#include <glog/logging.h>
#include <hydra/utils/nearest_neighbor_utilities.h>

#include "khronos/active_window/data/frame_data.h"
#include "khronos/backend/update_khronos_objects_functor.h"
#include "khronos/utils/khronos_attribute_utils.h"
#include "session_core/surface/triangle_grid.h"

namespace khronos {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

// Dynamic chunked parallel loop: fn(worker, begin, end) over [0, n), worker < num_threads.
template <typename Fn>
void parallelFor(size_t n, int num_threads, Fn&& fn, size_t chunk = 2048) {
  const size_t workers = std::min<size_t>(std::max(1, num_threads), (n + chunk - 1) / chunk);
  if (workers <= 1) {
    if (n) fn(size_t{0}, size_t{0}, n);
    return;
  }
  std::atomic<size_t> next{0};
  std::vector<std::thread> threads;
  for (size_t w = 0; w < workers; ++w) {
    threads.emplace_back([&, w] {
      while (true) {
        const size_t begin = next.fetch_add(chunk);
        if (begin >= n) break;
        fn(w, begin, std::min(n, begin + chunk));
      }
    });
  }
  for (auto& t : threads) t.join();
}

// Voxel keys, 21 bits per axis.
constexpr int64_t kKeyOffset = int64_t(1) << 20;

uint64_t voxelKey(const Eigen::Vector3f& p, float v) {
  const int64_t x = static_cast<int64_t>(std::floor(p.x() / v)) + kKeyOffset;
  const int64_t y = static_cast<int64_t>(std::floor(p.y() / v)) + kKeyOffset;
  const int64_t z = static_cast<int64_t>(std::floor(p.z() / v)) + kKeyOffset;
  return (static_cast<uint64_t>(x) << 42) | (static_cast<uint64_t>(y) << 21) | static_cast<uint64_t>(z);
}

// Lower corner of a voxel [m].
Eigen::Vector3f voxelCorner(uint64_t key, float v) {
  return Eigen::Vector3f(static_cast<float>(static_cast<int64_t>(key >> 42) - kKeyOffset) * v,
                         static_cast<float>(static_cast<int64_t>((key >> 21) & 0x1FFFFF) - kKeyOffset) * v,
                         static_cast<float>(static_cast<int64_t>(key & 0x1FFFFF) - kKeyOffset) * v);
}

// Edges of a triangle and of a box (corner c: bit 0 x, bit 1 y, bit 2 z).
using Edge = std::array<int, 2>;
constexpr Edge kTriangleEdges[3] = {{{0, 1}}, {{1, 2}}, {{2, 0}}};
constexpr Edge kBoxEdges[12] = {{{0, 1}}, {{2, 3}}, {{4, 5}}, {{6, 7}}, {{0, 2}}, {{1, 3}},
                                {{4, 6}}, {{5, 7}}, {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}}};

struct PixelBox {
  int u0 = 0, u1 = -1, v0 = 0, v1 = -1;
  PixelBox clip(const PixelBox& o) const {
    return {std::max(u0, o.u0), std::min(u1, o.u1), std::max(v0, o.v0), std::min(v1, o.v1)};
  }
  bool empty() const { return u0 > u1 || v0 > v1; }
};

// The pixels whose centre ray can meet the part with positive depth of a convex
// polytope (vertices X in the camera frame): the bounding box of its projection,
// open towards the image border where the polytope crosses the camera plane
// (its projection is unbounded in the direction of the crossing).
PixelBox pixelBox(const PresentTsdf::Camera& K, const Eigen::Vector3f* X, int nx, const Edge* edges, int ne) {
  float umin = kInf, umax = -kInf, vmin = kInf, vmax = -kInf;
  bool front = false;
  auto unbounded = [&](float x, float y) {
    if (x > 0.f) umax = kInf;
    if (x < 0.f) umin = -kInf;
    if (y > 0.f) vmax = kInf;
    if (y < 0.f) vmin = -kInf;
  };
  for (int i = 0; i < nx; ++i) {
    if (X[i].z() > 0.f) {
      front = true;
      const float u = X[i].x() / X[i].z() * K.fx + K.cx, v = X[i].y() / X[i].z() * K.fy + K.cy;
      umin = std::min(umin, u);
      umax = std::max(umax, u);
      vmin = std::min(vmin, v);
      vmax = std::max(vmax, v);
    } else if (X[i].z() == 0.f) {
      unbounded(X[i].x(), X[i].y());
    }
  }
  if (!front) return {};
  for (int e = 0; e < ne; ++e) {
    const Eigen::Vector3f& a = X[edges[e][0]];
    const Eigen::Vector3f& b = X[edges[e][1]];
    if (!((a.z() > 0.f && b.z() < 0.f) || (a.z() < 0.f && b.z() > 0.f))) continue;
    const float s = a.z() / (a.z() - b.z());
    unbounded(a.x() + s * (b.x() - a.x()), a.y() + s * (b.y() - a.y()));
  }
  // One pixel wider than the bounds, which only the per-pixel tests decide.
  const float u0 = std::max(0.f, std::floor(umin) - 1.f), u1 = std::min(K.width - 1.f, std::ceil(umax) + 1.f);
  const float v0 = std::max(0.f, std::floor(vmin) - 1.f), v1 = std::min(K.height - 1.f, std::ceil(vmax) + 1.f);
  if (u0 > u1 || v0 > v1) return {};
  return {static_cast<int>(u0), static_cast<int>(u1), static_cast<int>(v0), static_cast<int>(v1)};
}

// The voxels (size v) whose centre lies within one voxel of a triangle of `triangles` (3 vertices
// per face): where a reading measures that surface, by the same one-voxel correspondence that
// gives present faces their identity.
std::vector<uint64_t> surfaceVoxels(const std::vector<Eigen::Vector3f>& triangles, float v) {
  std::vector<uint64_t> keys;
  for (size_t f = 0; f + 2 < triangles.size(); f += 3) {
    const Eigen::Vector3f& a = triangles[f];
    const Eigen::Vector3f& b = triangles[f + 1];
    const Eigen::Vector3f& c = triangles[f + 2];
    const Eigen::Vector3f lo = a.cwiseMin(b).cwiseMin(c) - Eigen::Vector3f::Constant(v);
    const Eigen::Vector3f hi = a.cwiseMax(b).cwiseMax(c) + Eigen::Vector3f::Constant(v);
    const Eigen::Vector3i i0 = (lo / v).array().floor().cast<int>(), i1 = (hi / v).array().floor().cast<int>();
    for (int x = i0.x(); x <= i1.x(); ++x) {
      for (int y = i0.y(); y <= i1.y(); ++y) {
        for (int z = i0.z(); z <= i1.z(); ++z) {
          const Eigen::Vector3f centre = (Eigen::Vector3f(x, y, z) + Eigen::Vector3f::Constant(0.5f)) * v;
          if ((closestPointOnTriangle(centre, a, b, c) - centre).squaredNorm() <= v * v) {
            keys.push_back(voxelKey(centre, v));
          }
        }
      }
    }
  }
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  return keys;
}

// One mesh of the round's snapshot: the background or one current object mesh.
struct Slot {
  bool background = false;
  NodeId node = 0;
  spark_dsg::Mesh* mesh = nullptr;
  KhronosObjectAttributes* attrs = nullptr;
  uint32_t begin = 0, end = 0;  // global vertex range
  size_t physical = 0;          // node physical id (0: background / none)
};

using Face3 = std::array<uint32_t, 3>;

// Depth noise estimator settings: range bins and residual histogram resolution.
constexpr double kRangeBin = 0.5;
constexpr size_t kNumBins = 16;
constexpr double kHistogramResolution = 0.0005;

struct FrameCam {
  Eigen::Matrix3f R;  // camera_to_world rotation
  Eigen::Vector3f t;  // camera centre (world)
};

FrameCam frameCam(const Eigen::Isometry3d& world_T_sensor) {
  return {world_T_sensor.linear().cast<float>(), world_T_sensor.translation().cast<float>()};
}

// Projection of the offline reference (refusion.py _project): camera-frame
// z > 0.1, rounded pixel floor(x / z * f + c + 0.5), q = |p_C|.
struct Projector {
  float fx, fy, cx, cy;
  int W, H;
  inline bool operator()(const Eigen::Vector3f& p, const FrameCam& c, int& u, int& v,
                         float& q) const {
    const float dx = p.x() - c.t.x(), dy = p.y() - c.t.y(), dz = p.z() - c.t.z();
    const float x = c.R(0, 0) * dx + c.R(1, 0) * dy + c.R(2, 0) * dz;
    const float y = c.R(0, 1) * dx + c.R(1, 1) * dy + c.R(2, 1) * dz;
    const float z = c.R(0, 2) * dx + c.R(1, 2) * dy + c.R(2, 2) * dz;
    if (z <= 0.1f) return false;
    u = static_cast<int>(std::floor(x / z * fx + cx + 0.5f));
    v = static_cast<int>(std::floor(y / z * fy + cy + 0.5f));
    if (u < 0 || v < 0 || u >= W || v >= H) return false;
    q = std::sqrt(x * x + y * y + z * z);
    return true;
  }
};

// Median-based noise scale per range bin (refusion.py noise_table).
// README M1a: log sigma follows a random walk over the range bins with step
// variance delta^2. A bin's median scale estimate is an observation of it
// with the sampling variance of the Gaussian MAD estimator,
// 1 / (16 n phi(z)^2 z^2) with z = Phi^-1(3/4); delta^2 is the moment excess
// of the squared adjacent differences over their sampling variances. The
// Rauch-Tung-Striebel smoother gives each bin its posterior mean: a
// well-sampled bin keeps its own estimate, a thin or empty one borrows from
// its neighbours in proportion to their precision.
std::vector<float> sigmaFromHistogram(const std::vector<std::vector<int64_t>>& hist,
                                      double resolution,
                                      std::vector<int64_t>* counts) {
  const size_t nb = hist.size();
  const double z = 0.6744897501960817;  // Phi^-1(3/4)
  const double phi = std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI);
  std::vector<double> y(nb, std::numeric_limits<double>::quiet_NaN());  // log sigma
  std::vector<double> r(nb, std::numeric_limits<double>::infinity());   // its variance
  if (counts) counts->assign(nb, 0);
  for (size_t b = 0; b < nb; ++b) {
    int64_t cnt = 0;
    for (const auto c : hist[b]) cnt += c;
    if (counts) (*counts)[b] = cnt;
    if (cnt == 0) continue;
    const double half = 0.5 * static_cast<double>(cnt);
    int64_t cum = 0;
    for (size_t k = 0; k < hist[b].size(); ++k) {
      const int64_t prev = cum;
      cum += hist[b][k];
      if (static_cast<double>(cum) >= half) {  // numpy searchsorted(side='left')
        const double sigma = 1.4826 * (static_cast<double>(k) +
                             (half - static_cast<double>(prev)) /
                                 static_cast<double>(std::max<int64_t>(hist[b][k], 1))) *
                             resolution;
        if (sigma > 0.0) {
          y[b] = std::log(sigma);
          r[b] = 1.0 / (16.0 * static_cast<double>(cnt) * phi * phi * z * z);
        }
        break;
      }
    }
  }
  double excess = 0.0;
  size_t pairs = 0;
  for (size_t b = 0; b + 1 < nb; ++b) {
    if (!std::isfinite(y[b]) || !std::isfinite(y[b + 1])) continue;
    excess += (y[b + 1] - y[b]) * (y[b + 1] - y[b]) - r[b] - r[b + 1];
    ++pairs;
  }
  const double delta2 = pairs ? std::max(0.0, excess / static_cast<double>(pairs)) : 0.0;
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> m(nb, 0.0), p(nb, inf), mp(nb, 0.0), pp(nb, inf);
  double mean = 0.0, var = inf;
  for (size_t b = 0; b < nb; ++b) {
    if (b > 0) var += delta2;
    mp[b] = mean;
    pp[b] = var;
    if (std::isfinite(y[b])) {
      if (std::isinf(var)) {
        mean = y[b];
        var = r[b];
      } else {
        const double gain = var / (var + r[b]);
        mean += gain * (y[b] - mean);
        var *= 1.0 - gain;
      }
    }
    m[b] = mean;
    p[b] = var;
  }
  std::vector<float> out(nb, 0.f);
  if (std::isinf(var)) return out;  // no bin measured
  std::vector<double> smooth(m);
  for (size_t b = nb - 1; b-- > 0;) {
    const double gain = std::isinf(p[b]) || !(pp[b + 1] > 0.0) ? 1.0 : p[b] / pp[b + 1];
    smooth[b] = std::isinf(p[b]) ? smooth[b + 1] : m[b] + gain * (smooth[b + 1] - mp[b + 1]);
  }
  for (size_t b = 0; b < nb; ++b) out[b] = static_cast<float>(std::exp(smooth[b]));
  return out;
}

/**
 * The memory principle INSIDE: a memory vertex of an object that the session's
 * frames had in view (projected onto a valid reading) gives way to the present
 * if it lies inside the present surface of the same object -- of 64 fixed
 * directions, more first hit that surface from behind than from the front --
 * and farther than one voxel from any present surface.
 * @param candidates Object memory vertices in view, with their physical ids.
 * @returns The retired vertices (flag per vertex of `pos`).
 */
std::vector<uint8_t> insideMemory(const std::vector<Eigen::Vector3f>& pos,
                                  const std::vector<std::pair<uint32_t, uint32_t>>& candidates,
                                  const std::vector<Eigen::Vector3f>& Vp,
                                  const std::vector<Face3>& Fp,
                                  const std::vector<uint32_t>& face_id,
                                  const std::vector<Eigen::Vector3f>& face_normal,
                                  float voxel,
                                  int threads) {
  constexpr int kRays = 64;
  std::vector<Eigen::Vector3f> dirs(kRays);  // Fibonacci sphere
  for (int i = 0; i < kRays; ++i) {
    const double phi = std::acos(1.0 - 2.0 * (i + 0.5) / kRays);
    const double th = M_PI * (1.0 + std::sqrt(5.0)) * (i + 0.5);
    dirs[i] = Eigen::Vector3f(static_cast<float>(std::cos(th) * std::sin(phi)),
                              static_cast<float>(std::sin(th) * std::sin(phi)),
                              static_cast<float>(std::cos(phi)));
  }
  const TriangleGrid present(Vp, Fp, nullptr, 4.f * voxel);
  std::map<uint32_t, std::vector<uint32_t>> faces_of_label;
  for (uint32_t f = 0; f < Fp.size(); ++f) {
    if (face_id[f] > 0) faces_of_label[face_id[f]].push_back(f);
  }
  std::map<uint32_t, std::unique_ptr<TriangleGrid>> label_grid;
  for (const auto& [label, list] : faces_of_label) {
    label_grid[label] = std::make_unique<TriangleGrid>(Vp, Fp, &list, 4.f * voxel);
  }
  std::vector<uint8_t> retired(pos.size(), 0);
  parallelFor(candidates.size(), threads, [&](size_t, size_t b, size_t e) {
    for (size_t k = b; k < e; ++k) {
      const auto [vertex, label] = candidates[k];
      const Eigen::Vector3f& p = pos[vertex];
      const auto grid = label_grid.find(label);
      if (grid == label_grid.end()) continue;
      float d;
      Eigen::Vector3f closest;
      uint32_t face;
      if (present.closest(p, voxel, d, closest, face)) continue;  // within one voxel
      int behind = 0, front = 0;
      for (const auto& dir : dirs) {
        float t_hit;
        uint32_t hit;
        if (!grid->second->firstHit(p, dir, t_hit, hit)) continue;
        if (face_normal[hit].dot(dir) > 0.f) ++behind;
        else ++front;
      }
      if (behind > front) retired[vertex] = 1;
    }
  }, 1024);
  return retired;
}

}  // namespace

// Per-worker buffers over the pixels of one kept frame.
struct PresentSurface::Scratch {
  Eigen::Matrix3f R;                     // world_R_sensor
  Eigen::Vector3f c;                     // camera centre (world)
  std::vector<float> free_end;           // R - T per pixel, -inf where the reading is invalid
  std::vector<float> mesh_hit, voxel_hit;  // first hits on one state (+inf: none)
  std::vector<Eigen::Vector3f> inv_dir;  // inverse world ray direction per pixel
  Eigen::Vector3f toCamera(const Eigen::Vector3f& w) const { return R.transpose() * (w - c); }
};

bool PresentSurface::State::operator==(const State& other) const {
  return current == other.current && start == other.start && triangles == other.triangles &&
         voxels == other.voxels && end == other.end && ended_voxels == other.ended_voxels;
}

PresentSurface::PresentSurface(double object_voxel,
                               double map_voxel,
                               double map_truncation,
                               int num_threads)
    : voxel_(object_voxel),
      map_voxel_(map_voxel),
      truncation_(map_truncation),
      num_threads_(std::max(1, num_threads)),
      tsdf_(map_voxel, map_truncation, std::max(1, num_threads)) {}

float PresentSurface::copyTolerance(const std::vector<Eigen::Vector3f>& loaded) {
  float extent = 0.f;
  for (const auto& m : loaded) extent = std::max(extent, m.cwiseAbs().maxCoeff());
  return std::sqrt(3.f) * 2.f * 6.f * (0.5f * std::numeric_limits<float>::epsilon()) * 2.f * extent;
}

bool PresentSurface::saveEnded(const std::string& path, const std::map<size_t, Ended>& ended) {
  std::ofstream out(path, std::ios::binary);
  const uint64_t n = ended.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  for (const auto& [id, e] : ended) {
    const uint64_t key = id, end = e.end, points = e.triangles.size();
    out.write(reinterpret_cast<const char*>(&key), sizeof(key));
    out.write(reinterpret_cast<const char*>(&end), sizeof(end));
    out.write(reinterpret_cast<const char*>(&points), sizeof(points));
    for (const auto& p : e.triangles) out.write(reinterpret_cast<const char*>(p.data()), 3 * sizeof(float));
  }
  return static_cast<bool>(out);
}

bool PresentSurface::loadEnded(const std::string& path, std::map<size_t, Ended>& ended) {
  std::ifstream in(path, std::ios::binary);
  uint64_t n = 0;
  if (!in.read(reinterpret_cast<char*>(&n), sizeof(n))) return false;
  ended.clear();
  for (uint64_t i = 0; i < n; ++i) {
    uint64_t key = 0, end = 0, points = 0;
    in.read(reinterpret_cast<char*>(&key), sizeof(key));
    in.read(reinterpret_cast<char*>(&end), sizeof(end));
    in.read(reinterpret_cast<char*>(&points), sizeof(points));
    if (!in) return false;
    Ended& e = ended[key];
    e.end = end;
    e.triangles.resize(points);
    for (auto& p : e.triangles) in.read(reinterpret_cast<char*>(p.data()), 3 * sizeof(float));
  }
  return static_cast<bool>(in);
}

TimeStamp PresentSurface::firstStamp() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return frames_.numFrames() ? frames_.frame(0).stamp : 0;
}

void PresentSurface::prepare(size_t index, const std::vector<uint16_t>& range_mm, Scratch& s) const {
  const auto& pose = frames_.frame(index).world_T_sensor;
  s.R = pose.rotation().cast<float>();
  s.c = pose.translation().cast<float>();
  const size_t n = range_mm.size();
  const float T = static_cast<float>(truncation_);
  s.free_end.resize(n);
  for (size_t p = 0; p < n; ++p) s.free_end[p] = range_mm[p] ? range_mm[p] * 1e-3f - T : -kInf;
  if (s.mesh_hit.size() != n) {
    s.mesh_hit.assign(n, kInf);
    s.voxel_hit.assign(n, kInf);
    s.inv_dir.resize(n);
  }
}

bool PresentSurface::reaches(size_t index, const State& state) const {
  if (state.triangles.empty() && state.voxels.empty() && state.ended_voxels.empty()) return false;
  const auto& pose = frames_.frame(index).world_T_sensor;
  const Eigen::Matrix3f Rt = pose.rotation().cast<float>().transpose();
  const Eigen::Vector3f c = pose.translation().cast<float>();
  Eigen::Vector3f X[8];
  for (int k = 0; k < 8; ++k) X[k] = Rt * (state.box.corner(static_cast<Eigen::AlignedBox3f::CornerType>(k)) - c);
  return !pixelBox(camera_, X, 8, kBoxEdges, 12).empty();
}

void PresentSurface::limitBy(const State& state, Scratch& s, std::vector<float>& limit) const {
  if (state.triangles.empty() && state.voxels.empty()) return;
  const int W = static_cast<int>(camera_.width);
  Eigen::Vector3f X[8];
  for (int k = 0; k < 8; ++k) {
    X[k] = s.toCamera(state.box.corner(static_cast<Eigen::AlignedBox3f::CornerType>(k)));
  }
  const PixelBox region = pixelBox(camera_, X, 8, kBoxEdges, 12);
  if (region.empty()) return;

  // First hit of every pixel's ray on the current object mesh (Moller-Trumbore
  // from the camera centre, double precision as TriangleGrid).
  for (size_t f = 0; f + 2 < state.triangles.size(); f += 3) {
    for (int k = 0; k < 3; ++k) X[k] = s.toCamera(state.triangles[f + k]);
    const PixelBox box = pixelBox(camera_, X, 3, kTriangleEdges, 3).clip(region);
    if (box.empty()) continue;
    const Eigen::Vector3d a = X[0].cast<double>();
    const Eigen::Vector3d e1 = (X[1] - X[0]).cast<double>(), e2 = (X[2] - X[0]).cast<double>();
    const Eigen::Vector3d tvec = -a;
    const Eigen::Vector3d qvec = tvec.cross(e1);
    const double t_num = e2.dot(qvec);
    for (int v = box.v0; v <= box.v1; ++v) {
      for (int u = box.u0; u <= box.u1; ++u) {
        const size_t p = static_cast<size_t>(v) * W + u;
        const Eigen::Vector3d d = ray_dir_[p].cast<double>();
        const Eigen::Vector3d pvec = d.cross(e2);
        const double det = e1.dot(pvec);
        if (std::abs(det) < 1e-12) continue;
        const double inv = 1.0 / det;
        const double bu = tvec.dot(pvec) * inv;
        if (bu < 0.0 || bu > 1.0) continue;
        const double bv = d.dot(qvec) * inv;
        if (bv < 0.0 || bu + bv > 1.0) continue;
        const double t = t_num * inv;
        if (t > 0.0 && t < s.mesh_hit[p]) s.mesh_hit[p] = static_cast<float>(t);
      }
    }
  }

  // Without a qualified mesh hit, the first measured voxel the ray's free part
  // [0, R - T] enters (slab test, entry clipped at the camera centre).
  if (!state.voxels.empty()) {
    for (int v = region.v0; v <= region.v1; ++v) {
      for (int u = region.u0; u <= region.u1; ++u) {
        const size_t p = static_cast<size_t>(v) * W + u;
        s.inv_dir[p] = (s.R * ray_dir_[p]).cwiseInverse();
      }
    }
    const float vox = static_cast<float>(voxel_);
    for (const auto key : state.voxels) {
      const Eigen::Vector3f lo = voxelCorner(key, vox);
      const Eigen::Vector3f hi = lo + Eigen::Vector3f::Constant(vox);
      for (int k = 0; k < 8; ++k) {
        X[k] = s.toCamera(Eigen::Vector3f((k & 1) ? hi.x() : lo.x(), (k & 2) ? hi.y() : lo.y(),
                                          (k & 4) ? hi.z() : lo.z()));
      }
      const PixelBox box = pixelBox(camera_, X, 8, kBoxEdges, 12).clip(region);
      if (box.empty()) continue;
      const Eigen::Vector3f a = lo - s.c, b = hi - s.c;
      for (int v = box.v0; v <= box.v1; ++v) {
        for (int u = box.u0; u <= box.u1; ++u) {
          const size_t p = static_cast<size_t>(v) * W + u;
          if (!(s.free_end[p] > 0.f) || s.mesh_hit[p] < s.free_end[p]) continue;
          const Eigen::Vector3f& id = s.inv_dir[p];
          float t0 = 0.f, t1 = kInf;
          for (int k = 0; k < 3; ++k) {
            float ta = a[k] * id[k], tb = b[k] * id[k];
            if (ta > tb) std::swap(ta, tb);
            t0 = std::max(t0, ta);
            t1 = std::min(t1, tb);
          }
          if (t0 <= t1 && t0 < s.voxel_hit[p]) s.voxel_hit[p] = t0;
        }
      }
    }
  }

  // The state's free limit per pixel; the buffers back to +inf.
  for (int v = region.v0; v <= region.v1; ++v) {
    for (int u = region.u0; u <= region.u1; ++u) {
      const size_t p = static_cast<size_t>(v) * W + u;
      const float end = s.free_end[p];
      float hit = kInf;
      if (s.mesh_hit[p] < end) {
        hit = s.mesh_hit[p];
      } else if (end > 0.f && s.voxel_hit[p] <= end) {
        hit = s.voxel_hit[p];
      }
      limit[p] = std::min(limit[p], hit);
      s.mesh_hit[p] = kInf;
      s.voxel_hit[p] = kInf;
    }
  }
}

void PresentSurface::restrictBy(uint16_t id, const State& state, size_t index,
                                const std::vector<uint16_t>& range, Scratch& scratch, Restriction& r,
                                const bool with_limit) const {
  if (!r.prepared) {
    prepare(index, range, scratch);
    r.prepared = true;
  }
  const TimeStamp t = frames_.frame(index).stamp;
  if (state.current && t < state.start) {
    r.excluded.push_back(id);
    if (with_limit) {
      if (r.limit.empty()) r.limit.assign(range.size(), kInf);
      limitBy(state, scratch, r.limit);
    }
  }
  if (state.end > 0 && t <= state.end && !state.ended_voxels.empty()) r.voided.emplace_back(id, &state);
}

void PresentSurface::collect(const States& states, size_t index, const std::vector<uint16_t>& range,
                             Scratch& scratch, Restriction& r, const bool with_limit) const {
  const TimeStamp t = frames_.frame(index).stamp;
  for (const auto& [id, state] : states) {
    if (state.restricts(t)) restrictBy(id, state, index, range, scratch, r, with_limit);
  }
}

std::vector<float> PresentSurface::apply(const Restriction& r, const Scratch& scratch,
                                         std::vector<uint16_t>& range,
                                         const std::vector<uint16_t>& ids) const {
  std::vector<float> limit = r.limit;
  if (r.excluded.empty() && r.voided.empty()) return limit;
  std::vector<uint16_t> excluded = r.excluded;
  std::sort(excluded.begin(), excluded.end());
  const float vox = static_cast<float>(voxel_);
  bool limited = false;
  for (size_t p = 0; p < range.size(); ++p) {
    if (!range[p]) continue;
    if (ids[p] && std::binary_search(excluded.begin(), excluded.end(), ids[p])) {
      range[p] = 0;
    } else if (!r.voided.empty()) {
      // A pixel of the identity, or unlabelled, that measured the surface of one of its ended states.
      bool computed = false;
      Eigen::Vector3f end;
      uint64_t key = 0;
      for (const auto& [id, state] : r.voided) {
        if (ids[p] && ids[p] != id) continue;
        if (!computed) {
          end = scratch.c + scratch.R * (ray_dir_[p] * (range[p] * 1e-3f));
          key = voxelKey(end, vox);
          computed = true;
        }
        if (state->box.contains(end) &&
            std::binary_search(state->ended_voxels.begin(), state->ended_voxels.end(), key)) {
          range[p] = 0;
          break;
        }
      }
    }
    if (!limit.empty()) {
      if (!range[p]) limit[p] = kInf;
      else if (limit[p] < kInf) limited = true;
    }
  }
  if (!limited) limit.clear();
  return limit;
}

void PresentSurface::domainOf(const std::vector<uint16_t>& range, std::vector<float> limit,
                              Domain& domain) const {
  PresentTsdf::depthFromRange(camera_, range, domain.depth);
  domain.limit = std::move(limit);
}

void PresentSurface::counted(size_t index, std::vector<uint16_t>& range,
                             const std::vector<uint16_t>& ids, Scratch& scratch,
                             std::vector<float>* limit) const {
  Restriction r;
  collect(states_, index, range, scratch, r, limit != nullptr);
  auto counted_limit = apply(r, scratch, range, ids);
  if (limit) *limit = std::move(counted_limit);
}

void PresentSurface::integrate(const FrameData& data) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<uint16_t> range_mm, ids;
  if (!frames_.offer(data, range_mm, ids)) return;
  integrateKept(range_mm, ids);
}

void PresentSurface::integrate(const FrameArchive::Camera& camera, FrameArchive::Frame frame) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<uint16_t> range_mm, ids;
  if (!frame.decode(static_cast<size_t>(camera.width) * camera.height, range_mm, ids)) {
    LOG(ERROR) << "[PresentSurface] archived frame " << frame.stamp << " cannot be decoded; skipped.";
    return;
  }
  if (!frames_.keep(camera, std::move(frame))) return;
  integrateKept(range_mm, ids);
}

void PresentSurface::integrateKept(const std::vector<uint16_t>& range_mm, const std::vector<uint16_t>& ids) {
  if (frames_.numFrames() == 1) {
    const auto& c = frames_.camera();
    camera_ = {c.width, c.height, c.fx, c.fy, c.cx, c.cy};
    ray_norm_ = PresentTsdf::rayNorm(camera_);
    ray_dir_.resize(ray_norm_.size());
    for (uint32_t v = 0; v < camera_.height; ++v) {
      for (uint32_t u = 0; u < camera_.width; ++u) {
        ray_dir_[static_cast<size_t>(v) * camera_.width + u] =
            Eigen::Vector3f((u - camera_.cx) / camera_.fx, (v - camera_.cy) / camera_.fy, 1.f).normalized();
      }
    }
  }
  const size_t index = frames_.numFrames() - 1;
  const auto& frame = frames_.frame(index);
  // The frame's domain under the current states.
  Scratch scratch;
  std::vector<uint16_t> range = range_mm;
  std::vector<float> limit;
  counted(index, range, ids, scratch, &limit);
  Domain domain;
  domainOf(range, std::move(limit), domain);
  tsdf_.update(camera_, frame.world_T_sensor, ray_norm_, {}, domain.view());
}

void PresentSurface::recordEndpoints() {
  const size_t n = frames_.numFrames();
  if (recorded_ >= n) return;
  const size_t num_pixels = static_cast<size_t>(camera_.width) * camera_.height;
  const float vox = static_cast<float>(voxel_);
  std::mutex merge;
  parallelFor(n - recorded_, num_threads_, [&](size_t, size_t b, size_t e) {
    std::vector<uint16_t> range, ids;
    std::unordered_map<uint16_t, std::unordered_map<uint64_t, TimeStamp>> local;
    for (size_t i = recorded_ + b; i < recorded_ + e; ++i) {
      const auto& frame = frames_.frame(i);
      if (!frame.decode(num_pixels, range, ids)) continue;
      const Eigen::Matrix3f R = frame.world_T_sensor.rotation().cast<float>();
      const Eigen::Vector3f c = frame.world_T_sensor.translation().cast<float>();
      uint16_t last_id = 0;
      std::unordered_map<uint64_t, TimeStamp>* keys = nullptr;
      for (size_t p = 0; p < num_pixels; ++p) {
        if (!range[p] || !ids[p]) continue;
        if (ids[p] != last_id) {
          last_id = ids[p];
          keys = &local[last_id];
        }
        (*keys)[voxelKey(c + R * (ray_dir_[p] * (range[p] * 1e-3f)), vox)] = frame.stamp;
      }
    }
    std::lock_guard<std::mutex> lock(merge);
    for (const auto& [id, keys] : local) {
      auto& out = endpoints_[id];
      for (const auto& [key, stamp] : keys) {
        auto& latest = out[key];
        latest = std::max(latest, stamp);
      }
    }
  }, 8);
  recorded_ = n;
}

void PresentSurface::updateStates(States next) {
  const auto start = std::chrono::steady_clock::now();
  // The identities whose state differs from the one the TSDF reflects.
  struct Change {
    uint16_t id = 0;
    const State* before = nullptr;
    const State* after = nullptr;
  };
  std::vector<Change> changes;
  std::vector<std::pair<uint16_t, const State*>> unchanged;
  TimeStamp horizon = 0;
  for (auto b = states_.begin(), a = next.begin(); b != states_.end() || a != next.end();) {
    Change change;
    if (a == next.end() || (b != states_.end() && b->first < a->first)) {
      change = {b->first, &b->second, nullptr};
      ++b;
    } else if (b == states_.end() || a->first < b->first) {
      change = {a->first, nullptr, &a->second};
      ++a;
    } else {
      if (b->second == a->second) {
        unchanged.emplace_back(a->first, &a->second);
        ++a, ++b;
        continue;
      }
      change = {a->first, &b->second, &a->second};
      ++a, ++b;
    }
    // No frame later than a changed state's horizon (before or after) is restricted by it.
    if (change.before) horizon = std::max(horizon, change.before->horizon());
    if (change.after) horizon = std::max(horizon, change.after->horizon());
    changes.push_back(change);
  }
  if (changes.empty()) {
    states_ = std::move(next);
    return;
  }

  // A frame's domain can change only if a changed state restricts it before or
  // after and that restriction appears, disappears, or (restricting both) the
  // surface or past voxels of either reach the frame's image.
  std::vector<size_t> candidates;
  for (size_t i = 0; i < frames_.numFrames(); ++i) {
    const TimeStamp t = frames_.frame(i).stamp;
    if (t > horizon) continue;
    for (const auto& change : changes) {
      const bool before = change.before && change.before->restricts(t);
      const bool after = change.after && change.after->restricts(t);
      if (before != after || (before && (reaches(i, *change.before) || reaches(i, *change.after)))) {
        candidates.push_back(i);
        break;
      }
    }
  }

  // Domains under both states, computed in parallel; the moves applied in order.
  const size_t num_pixels = static_cast<size_t>(camera_.width) * camera_.height;
  struct Work {
    size_t index = 0;
    bool moved = false;
    Domain before, after;
  };
  const size_t batch = static_cast<size_t>(num_threads_);
  std::vector<Work> work(batch);
  std::vector<Scratch> scratch(num_threads_);
  size_t moved = 0, limited = 0;
  for (size_t first = 0; first < candidates.size(); first += batch) {
    const size_t m = std::min(batch, candidates.size() - first);
    parallelFor(m, num_threads_, [&](size_t w, size_t b, size_t e) {
      std::vector<uint16_t> range, ids;
      for (size_t j = b; j < e; ++j) {
        Work& job = work[j];
        job.index = candidates[first + j];
        job.moved = false;
        const auto& frame = frames_.frame(job.index);
        if (!frame.decode(num_pixels, range, ids)) {
          LOG(ERROR) << "[PresentSurface] kept frame " << frame.stamp << " cannot be decoded.";
          continue;
        }
        // The unchanged states restricting the frame restrict both domains alike.
        Restriction shared;
        for (const auto& [id, state] : unchanged) {
          if (state->restricts(frame.stamp)) restrictBy(id, *state, job.index, range, scratch[w], shared, true);
        }
        Restriction before = shared, after = std::move(shared);
        for (const auto& change : changes) {
          if (change.before && change.before->restricts(frame.stamp)) {
            restrictBy(change.id, *change.before, job.index, range, scratch[w], before, true);
          }
          if (change.after && change.after->restricts(frame.stamp)) {
            restrictBy(change.id, *change.after, job.index, range, scratch[w], after, true);
          }
        }
        if (!before.prepared && !after.prepared) continue;  // neither restricts the frame
        std::vector<uint16_t> range_before = range, range_after = range;
        auto limit_before = apply(before, scratch[w], range_before, ids);
        auto limit_after = apply(after, scratch[w], range_after, ids);
        if (range_before == range_after && limit_before == limit_after) continue;
        domainOf(range_before, std::move(limit_before), job.before);
        domainOf(range_after, std::move(limit_after), job.after);
        job.moved = true;
      }
    }, 1);
    for (size_t j = 0; j < m; ++j) {
      if (!work[j].moved) continue;
      tsdf_.update(camera_, frames_.frame(work[j].index).world_T_sensor, ray_norm_,
                   work[j].before.view(), work[j].after.view());
      ++moved;
      for (const float l : work[j].after.limit) limited += l < kInf;
    }
  }
  states_ = std::move(next);
  LOG(INFO) << "[PresentSurface] states " << states_.size() << ": " << changes.size()
            << " changed, " << candidates.size() << " frame(s) examined, " << moved
            << " moved (limited pixels " << limited << ") in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
            << " s";
}

namespace {

void writePly(const std::string& path,
              const std::vector<Eigen::Vector3f>& V,
              const std::vector<Face3>& F,
              const std::vector<int32_t>* face_label) {
  std::ofstream out(path, std::ios::binary);
  out << "ply\nformat binary_little_endian 1.0\nelement vertex " << V.size()
      << "\nproperty float x\nproperty float y\nproperty float z\nelement face " << F.size()
      << "\nproperty list uchar uint vertex_indices\n";
  if (face_label) out << "property int label\n";
  out << "end_header\n";
  for (const auto& p : V) out.write(reinterpret_cast<const char*>(p.data()), 12);
  const uint8_t three = 3;
  for (size_t i = 0; i < F.size(); ++i) {
    out.write(reinterpret_cast<const char*>(&three), 1);
    out.write(reinterpret_cast<const char*>(F[i].data()), 12);
    if (face_label) out.write(reinterpret_cast<const char*>(&(*face_label)[i]), 4);
  }
}

// The session's depth scale (port of the consolidation's estimator): a surface
// point measured by one frame and re-measured by another is read at the same
// position whatever the two ranges when depth and trajectory agree in scale; a
// depth short (long) by a fixed fraction s of the range places the same
// surface at different positions from near and far views. s is the scale that
// makes the frames agree best: every reading scaled by (1 + s), s minimises the
// median disagreement of re-measured points (pixels of one frame projected into
// another, associated within `association`). Frame subset, pixel stride and
// search grid are estimator settings; exact depth gives s = 0.
float depthScale(const FrameArchive& frames, const PresentTsdf::Camera& K,
                 const Projector& project, float association, int threads) {
  constexpr size_t kMaxFrames = 64;
  constexpr int kPixelStride = 16;
  struct View {
    FrameCam cam;
    std::vector<uint16_t> range;
  };
  std::vector<View> views;
  const size_t num_pixels = static_cast<size_t>(K.width) * K.height;
  const size_t step = std::max<size_t>(1, frames.numFrames() / kMaxFrames);
  std::vector<uint16_t> ids;
  for (size_t i = 0; i < frames.numFrames() && views.size() < kMaxFrames; i += step) {
    View view;
    view.cam = frameCam(frames.frame(i).world_T_sensor);
    if (!frames.frame(i).decode(num_pixels, view.range, ids)) view.range.assign(num_pixels, 0);
    views.push_back(std::move(view));
  }
  if (views.size() < 2) return 0.f;
  struct Sample {
    Eigen::Vector3f origin, direction, other;
    float range = 0.f, other_range = 0.f;
  };
  std::vector<std::vector<Sample>> per_view(views.size());
  const int W = static_cast<int>(K.width), H = static_cast<int>(K.height);
  parallelFor(views.size(), threads, [&](size_t, size_t b, size_t e) {
    for (size_t g = b; g < e; ++g) {
      const View& a = views[g];
      for (int v = 0; v < H; v += kPixelStride) {
        for (int u = 0; u < W; u += kPixelStride) {
          const uint16_t d_mm = a.range[static_cast<size_t>(v) * W + u];
          if (!d_mm) continue;
          const Eigen::Vector3f direction =
              a.cam.R * Eigen::Vector3f((u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.f).normalized();
          const float range = 1e-3f * d_mm;
          const Eigen::Vector3f point = a.cam.t + direction * range;
          for (size_t f = 0; f < views.size(); ++f) {
            if (f == g) continue;
            int pu, pv;
            float q;
            if (!project(point, views[f].cam, pu, pv, q)) continue;
            const uint16_t e_mm = views[f].range[static_cast<size_t>(pv) * W + pu];
            if (!e_mm || std::abs(1e-3f * e_mm - q) > association) continue;
            per_view[g].push_back({a.cam.t, direction, views[f].cam.t, range, 1e-3f * e_mm});
          }
        }
      }
    }
  }, 1);
  std::vector<Sample> samples;
  for (auto& list : per_view) samples.insert(samples.end(), list.begin(), list.end());
  if (samples.empty()) return 0.f;
  std::vector<float> residual(samples.size());
  auto disagreement = [&](float scale) {
    for (size_t i = 0; i < samples.size(); ++i) {
      const auto& x = samples[i];
      const Eigen::Vector3f point = x.origin + x.direction * (x.range * (1.f + scale));
      residual[i] = std::abs(x.other_range * (1.f + scale) - (point - x.other).norm());
    }
    auto mid = residual.begin() + residual.size() / 2;
    std::nth_element(residual.begin(), mid, residual.end());
    return *mid;
  };
  // The residuals under scale s are distances of the rescaled geometry, (1 + s) times the measured
  // units; the likelihood of the measured ranges carries the Jacobian of that change of variables
  // (P41), so the scale family is compared on the measured-unit median m_s / (1 + s). Without it a
  // shrinking scale (s < 0) explained the frames better merely by shrinking every residual.
  auto measured = [&](float scale) { return disagreement(scale) / (1.f + scale); };
  float best_s = 0.f, best = measured(0.f);
  for (int k = -50; k <= 50; ++k) {  // coarse: +-10 % in 0.2 % steps
    const float scale = 0.002f * k, m = measured(scale);
    if (m < best) best = m, best_s = scale;
  }
  const float coarse = best_s;
  for (int k = -10; k <= 10; ++k) {  // fine: 0.02 % steps around the coarse optimum
    const float scale = coarse + 0.0002f * k, m = measured(scale);
    if (m < best) best = m, best_s = scale;
  }
  // README M1a: s is one fitted parameter of a scale family of residuals, so
  // its profile log-likelihood gain over the exact-depth explanation is
  // n log(m_0 / m_s) for measured-unit median disagreements m_0 and m_s; the
  // Laplace approximation of the marginal likelihood charges one parameter
  // (1/2) log n. The fit is used only when it explains the frames better.
  const double n = static_cast<double>(samples.size());
  const float exact = measured(0.f);
  const double gain = best > 0.f && exact > 0.f
      ? n * std::log(static_cast<double>(exact) / static_cast<double>(best)) : 0.0;
  const bool fitted = gain > 0.5 * std::log(n);
  LOG(INFO) << "[SessionRefusion] depth_scale_evidence samples=" << samples.size()
            << " median_exact=" << exact << " median_fitted=" << best << " scale=" << best_s
            << " log_gain=" << gain << " penalty=" << 0.5 * std::log(n)
            << " fitted=" << fitted;
  return fitted ? best_s : 0.f;
}

}  // namespace

PresentSurface::Result PresentSurface::compose(spark_dsg::DynamicSceneGraph& dsg,
                                               const TimeStamp stamp,
                                               const std::map<size_t, TimeStamp>& state_starts,
                                               const std::map<size_t, Ended>& ended,
                                               const Inputs& inputs,
                                               const bool terminal) {
  Result result;
  const auto start = std::chrono::steady_clock::now();

  // The round's meshes: the background and every current object mesh (world frame).
  std::vector<Slot> slots;
  std::vector<Eigen::Vector3f> pos;
  std::vector<uint32_t> vslot;
  std::vector<Face3> faces;
  std::vector<uint32_t> fslot;
  auto addSlot = [&](Slot slot) {
    const auto& mesh = *slot.mesh;
    const uint32_t sid = static_cast<uint32_t>(slots.size());
    slot.begin = static_cast<uint32_t>(pos.size());
    for (size_t i = 0; i < mesh.numVertices(); ++i) {
      pos.push_back(slot.attrs ? slot.attrs->bounding_box.pointToWorldFrame(mesh.pos(i))
                               : mesh.pos(i));
      vslot.push_back(sid);
    }
    slot.end = static_cast<uint32_t>(pos.size());
    const size_t n = mesh.numVertices();
    for (const auto& f : mesh.faces) {
      if (f[0] >= n || f[1] >= n || f[2] >= n) continue;
      faces.push_back({slot.begin + static_cast<uint32_t>(f[0]),
                       slot.begin + static_cast<uint32_t>(f[1]),
                       slot.begin + static_cast<uint32_t>(f[2])});
      fslot.push_back(sid);
    }
    slots.push_back(slot);
  };
  if (dsg.hasMesh() && dsg.mesh()) {
    Slot bg;
    bg.background = true;
    bg.mesh = dsg.mesh().get();
    addSlot(bg);
  }
  if (dsg.hasLayer(DsgLayers::OBJECTS)) {
    for (const auto& [id, node] : dsg.getLayer(DsgLayers::OBJECTS).nodes()) {
      auto* attrs = dynamic_cast<KhronosObjectAttributes*>(&node->attributes());
      if (!attrs || !hasCurrentObjectMesh(*attrs)) continue;
      Slot slot;
      slot.node = id;
      slot.mesh = &attrs->mesh;
      slot.attrs = attrs;
      slot.physical = UpdateKhronosObjectsFunctor::physicalInstanceId(*attrs).value_or(0);
      addSlot(slot);
    }
  }

  // The round's states: every identity whose current state began in this
  // session, with its current object meshes, and the surfaces of its states
  // that ended in this session.
  States next;
  const float v_obj = static_cast<float>(voxel_);
  for (const auto& [id, t] : state_starts) {
    if (id == 0 || id > std::numeric_limits<uint16_t>::max()) continue;
    State& state = next[static_cast<uint16_t>(id)];
    state.current = true;
    state.start = t;
  }
  for (const auto& [id, e] : ended) {
    if (id == 0 || id > std::numeric_limits<uint16_t>::max() || e.end == 0) continue;
    State& state = next[static_cast<uint16_t>(id)];
    state.end = e.end;
    state.ended_voxels = surfaceVoxels(e.triangles, v_obj);
  }
  for (uint32_t f = 0; f < faces.size(); ++f) {
    const Slot& slot = slots[fslot[f]];
    if (slot.background || slot.physical == 0 || slot.physical > std::numeric_limits<uint16_t>::max()) {
      continue;
    }
    const auto it = next.find(static_cast<uint16_t>(slot.physical));
    if (it == next.end()) continue;
    for (int k = 0; k < 3; ++k) it->second.triangles.push_back(pos[faces[f][k]]);
  }

  // Frames and states stay fixed until the memory has read them.
  std::unique_lock<std::mutex> lock(mutex_);
  const size_t num_frames = frames_.numFrames();
  if (num_frames == 0) return result;
  std::vector<Eigen::Vector3f> Vp;
  std::vector<Face3> Fp;
  {
    // Measured voxels of each current state: those holding an endpoint of its own
    // pixels from t_l on.
    recordEndpoints();
    const float vox = static_cast<float>(voxel_);
    for (auto& [id, state] : next) {
      const auto it = endpoints_.find(id);
      if (state.current && it != endpoints_.end()) {
        for (const auto& [key, latest] : it->second) {
          if (latest >= state.start) state.voxels.push_back(key);
        }
      }
      std::sort(state.voxels.begin(), state.voxels.end());
      for (const auto& p : state.triangles) state.box.extend(p);
      for (const auto* keys : {&state.voxels, &state.ended_voxels}) {
        for (const auto key : *keys) {
          state.box.extend(voxelCorner(key, vox));
          state.box.extend(voxelCorner(key, vox) + Eigen::Vector3f::Constant(vox));
        }
      }
    }
    updateStates(std::move(next));
    tsdf_.extractMesh(Vp, Fp);
  }
  const bool dump = terminal && !inputs.dump_dir.empty();
  if (dump) {
    std::filesystem::create_directories(inputs.dump_dir);
    FrameArchive::save(inputs.dump_dir + "/archive.kfa", frames_.frames(), frames_.camera());
    saveEnded(inputs.dump_dir + "/ended.bin", ended);
  }
  const size_t num_units = tsdf_.numUnits();
  std::stringstream report;
  report << "{\"stamp\":" << stamp << ",\"frames\":" << num_frames << ",\"voxel\":" << map_voxel_
         << ",\"truncation\":" << truncation_ << ",\"object_voxel\":" << voxel_
         << ",\"present\":{\"units\":" << num_units << ",\"vertices\":" << Vp.size()
         << ",\"faces\":" << Fp.size() << "}";

  // Physical id -> the node slot its surface goes to (the largest if several
  // nodes carry it); the background slot.
  std::map<size_t, uint32_t> slot_of_label;
  int32_t bg_slot = -1;
  for (uint32_t s = 0; s < slots.size(); ++s) {
    if (slots[s].background) bg_slot = static_cast<int32_t>(s);
    if (slots[s].background || slots[s].physical == 0) continue;
    const auto it = slot_of_label.find(slots[s].physical);
    if (it == slot_of_label.end() ||
        slots[s].end - slots[s].begin > slots[it->second].end - slots[it->second].begin) {
      slot_of_label[slots[s].physical] = s;
    }
  }

  // Present face normals and centroids.
  std::vector<Eigen::Vector3f> face_normal(Fp.size()), centroid(Fp.size());
  for (size_t f = 0; f < Fp.size(); ++f) {
    const Eigen::Vector3d a = Vp[Fp[f][0]].cast<double>(), b = Vp[Fp[f][1]].cast<double>(),
                          c = Vp[Fp[f][2]].cast<double>();
    face_normal[f] = (b - a).cross(c - a).cast<float>();
    centroid[f] = ((a + b + c) / 3.0).cast<float>();
  }

  // Identity of every present face: the current object mesh within one object
  // voxel of its centroid (the nearest), or the background.
  const float v_f = static_cast<float>(voxel_);
  std::vector<uint32_t> face_id(Fp.size(), 0);
  {
    std::vector<uint32_t> labelled;
    for (uint32_t f = 0; f < faces.size(); ++f) {
      const Slot& slot = slots[fslot[f]];
      if (!slot.background && slot.physical > 0 && slot_of_label.count(slot.physical)) labelled.push_back(f);
    }
    if (!labelled.empty()) {
      const TriangleGrid objects(pos, faces, &labelled, 4.f * v_f);
      parallelFor(Fp.size(), num_threads_, [&](size_t, size_t b, size_t e) {
        for (size_t f = b; f < e; ++f) {
          float d;
          Eigen::Vector3f closest;
          uint32_t hit;
          if (objects.closest(centroid[f], v_f, d, closest, hit)) {
            face_id[f] = static_cast<uint32_t>(slots[fslot[hit]].physical);
          }
        }
      });
    }
  }

  if (dump) {
    const std::vector<int32_t> labels(face_id.begin(), face_id.end());
    writePly(inputs.dump_dir + "/present.ply", Vp, Fp, &labels);
  }

  // ------------------------------------ memory: the previous final map, where
  // this session's evidence does not contradict it. A shown face of an object
  // stays only while the node of its physical id is current and its state did
  // not begin in this session (the object reasoning decides where objects
  // are). Every other shown face is a surface element known to within
  // tau = max(h, sigma) (h: half a voxel of its layer) and is tested against
  // every frame in the domain the present counts it with: the frame hits it
  // (some pixel of its footprint, radius focal * tau / z, reads within tau),
  // sees through it (every pixel of the footprint is valid and reads beyond it
  // by more than tau, with free space the present counts) or is blocked in
  // front of it (within its layer's truncation: blocked_band). It gives way to
  // the present when the frames see through it more often than they hit it, or
  // when none hits or sees through it and most of its blocked views are blocked
  // within the truncation band (a TSDF holds no second surface that close
  // behind the observed one). Memory no frame observed stays.
  const auto& K = camera_;
  const int W = static_cast<int>(K.width), H = static_cast<int>(K.height);
  const size_t num_pixels = static_cast<size_t>(W) * H;
  const Projector project{K.fx, K.fy, K.cx, K.cy, W, H};
  const float T = static_cast<float>(truncation_);  // the present's truncation (eq. 8)
  const float h_obj = 0.5f * v_f, T_obj = 2.f * v_f;  // object layer: object truncation = 2 voxels
  const float h_bg = static_cast<float>(0.5 * map_voxel_), T_bg = T;
  const bool shown_mode = inputs.shown != nullptr;
  // Without a shown state: memory faces of the snapshot (all three vertices copies of the loaded state).
  std::vector<uint8_t> memory_face(faces.size(), 0);
  size_t num_memory = 0;
  if (!shown_mode && inputs.is_memory) {
    std::vector<uint8_t> is_memory(pos.size(), 0);
    parallelFor(pos.size(), num_threads_, [&](size_t, size_t b, size_t e) {
      for (size_t i = b; i < e; ++i) is_memory[i] = inputs.is_memory(pos[i]) ? 1 : 0;
    });
    num_memory = std::count(is_memory.begin(), is_memory.end(), 1);
    for (size_t f = 0; f < faces.size(); ++f) {
      memory_face[f] = is_memory[faces[f][0]] && is_memory[faces[f][1]] && is_memory[faces[f][2]];
    }
  }
  std::vector<int32_t> shown_slot;  // per shown face: target slot, -1 = not shown
  std::vector<uint32_t> tested;     // shown faces the evidence decides
  std::vector<Eigen::Vector3f> tested_centroid;
  std::vector<float> tested_half, tested_trunc;
  size_t shown_object_state = 0, shown_seen_through = 0, shown_hidden = 0, shown_displaced = 0;
  if (shown_mode) {
    const auto& S = *inputs.shown;
    shown_slot.assign(S.faces.size(), -1);
    for (size_t f = 0; f < S.faces.size(); ++f) {
      const uint32_t p = S.face_physical[f];
      int32_t target = bg_slot;
      if (p > 0) {
        const auto it = slot_of_label.find(p);
        if (it == slot_of_label.end() || state_starts.count(p)) {
          ++shown_object_state;
          continue;
        }
        target = static_cast<int32_t>(it->second);
      }
      if (target < 0) continue;
      shown_slot[f] = target;
      tested.push_back(static_cast<uint32_t>(f));
      tested_centroid.push_back((S.vertices[S.faces[f][0]] + S.vertices[S.faces[f][1]] +
                                 S.vertices[S.faces[f][2]]) / 3.f);
      tested_half.push_back(p > 0 ? h_obj : h_bg);
      tested_trunc.push_back(p > 0 ? T_obj : T_bg);
    }
  }
  struct Evidence {
    uint16_t hit = 0, through = 0, blocked = 0, blocked_band = 0;
    float q_hit = kInf;    // nearest range at which a frame hit it
    float q_reach = kInf;  // nearest range at which a frame reached it
    Eigen::Vector3f cam_reach = Eigen::Vector3f::Zero();  // that frame's centre
  };
  std::vector<Evidence> tested_ev(tested.size());
  // INSIDE candidates: vertices of memory faces on an object with a physical id
  // (index into `inside_pos`: the snapshot's vertices, or the shown map's).
  const std::vector<Eigen::Vector3f>& inside_pos = shown_mode ? inputs.shown->vertices : pos;
  std::vector<std::pair<uint32_t, uint32_t>> memory_objects;
  if (shown_mode) {
    std::vector<uint32_t> label_of(inputs.shown->vertices.size(), 0);
    for (size_t f = 0; f < inputs.shown->faces.size(); ++f) {
      if (shown_slot[f] < 0 || inputs.shown->face_physical[f] == 0) continue;
      for (int k = 0; k < 3; ++k) label_of[inputs.shown->faces[f][k]] = inputs.shown->face_physical[f];
    }
    for (uint32_t i = 0; i < label_of.size(); ++i) {
      if (label_of[i]) memory_objects.emplace_back(i, label_of[i]);
    }
  } else {
    std::vector<uint8_t> in_memory_face(pos.size(), 0);
    for (size_t f = 0; f < faces.size(); ++f) {
      if (memory_face[f]) in_memory_face[faces[f][0]] = in_memory_face[faces[f][1]] = in_memory_face[faces[f][2]] = 1;
    }
    for (uint32_t i = 0; i < pos.size(); ++i) {
      const Slot& slot = slots[vslot[i]];
      if (in_memory_face[i] && !slot.background && slot.physical > 0) {
        memory_objects.emplace_back(i, static_cast<uint32_t>(slot.physical));
      }
    }
  }
  std::vector<uint8_t> in_view(memory_objects.size(), 0);

  // sigma(q): the depth residual per range bin, measured on the present
  // (1.4826 * median |reading - range| of front-facing present vertices within
  // one truncation, every frame as the present counts it). tau(h, q) =
  // max(h, sigma(q)) is how well a surface element of half-voxel h is known
  // along a ray at range q.
  std::vector<float> sigma(kNumBins, 0.f);
  if (!tested.empty()) {
    std::vector<Eigen::Vector3d> vn(Vp.size(), Eigen::Vector3d::Zero());
    for (size_t f = 0; f < Fp.size(); ++f) {
      const Eigen::Vector3d n = face_normal[f].cast<double>();
      for (int k = 0; k < 3; ++k) vn[Fp[f][k]] += n;
    }
    std::vector<Eigen::Vector3f> N(Vp.size());
    for (size_t j = 0; j < Vp.size(); ++j) {
      const double len = vn[j].norm();
      N[j] = len > 0 ? Eigen::Vector3f((vn[j] / len).cast<float>()) : Eigen::Vector3f::Zero();
    }
    const size_t nh = static_cast<size_t>(std::floor(truncation_ / kHistogramResolution + 1e-9)) + 1;
    std::vector<std::vector<int64_t>> hist(kNumBins, std::vector<int64_t>(nh, 0));
    std::mutex hist_mutex;
    parallelFor(num_frames, num_threads_, [&](size_t, size_t b, size_t e) {
      std::vector<int64_t> local(kNumBins * nh, 0);
      std::vector<uint16_t> rng, ids;
      Scratch scratch;
      for (size_t i = b; i < e; ++i) {
        if (!frames_.frame(i).decode(num_pixels, rng, ids)) continue;
        counted(i, rng, ids, scratch, nullptr);
        const FrameCam c = frameCam(frames_.frame(i).world_T_sensor);
        for (size_t j = 0; j < Vp.size(); ++j) {
          int u, v;
          float q;
          if (!project(Vp[j], c, u, v, q)) continue;
          if (N[j].dot(c.t - Vp[j]) <= 0.f) continue;
          const uint16_t d = rng[static_cast<size_t>(v) * W + u];
          if (!d) continue;
          const double r = std::abs(d * 1e-3 - static_cast<double>(q));
          if (r > truncation_) continue;
          const size_t bin = std::min(kNumBins - 1, static_cast<size_t>(q / kRangeBin));
          const size_t cell = std::min(nh - 1, static_cast<size_t>(r / kHistogramResolution));
          ++local[bin * nh + cell];
        }
      }
      std::lock_guard<std::mutex> hist_lock(hist_mutex);
      for (size_t bin = 0; bin < kNumBins; ++bin)
        for (size_t cell = 0; cell < nh; ++cell) hist[bin][cell] += local[bin * nh + cell];
    }, 8);
    std::vector<int64_t> counts;
    sigma = sigmaFromHistogram(hist, kHistogramResolution, &counts);
    if (terminal) {
      // DIAG (P39/P47 same-surface law; no decision reads it): the residual histogram per range bin.
      for (size_t b = 0; b < kNumBins; ++b) {
        size_t last = 0;
        for (size_t k = 0; k < nh; ++k) if (hist[b][k]) last = k + 1;
        if (!last) continue;
        std::string row;
        for (size_t k = 0; k < last; ++k) row += (k ? "," : "") + std::to_string(hist[b][k]);
        LOG(INFO) << "[SessionRefusion] NOISE_HIST_DIAG range_bin=" << b
                  << " range_bin_m=" << kRangeBin << " resolution_m=" << kHistogramResolution
                  << " counts=" << row;
      }
    }
    report << ",\"sigma_cm\":[";
    for (size_t b = 0; b < kNumBins; ++b) report << (b ? "," : "") << sigma[b] * 100.f;
    report << "]";
  }
  auto tauOf = [&](float half, float q) {
    const size_t bin = std::min(kNumBins - 1, static_cast<size_t>(std::max(0.f, q) / kRangeBin));
    return std::max(half, sigma[bin]);
  };

  // This session's depth scale: the memory's displaced-copy window needs it,
  // and the terminal round records it for the next session.
  float depth_scale = 0.f;
  if (!tested.empty() || terminal) {
    depth_scale = depthScale(frames_, K, project, T_bg, num_threads_);
    result.depth_scale = depth_scale;
    report << ",\"depth_scale\":" << depth_scale;
  }

  // One pass over the frames, each as the present counts it: INSIDE candidates
  // in view and the evidence at the tested shown faces.
  if (!tested.empty() || !memory_objects.empty()) {
    const size_t batch = static_cast<size_t>(num_threads_);
    std::vector<std::vector<uint16_t>> rng(batch), ids(batch);
    std::vector<std::vector<float>> limit(batch);
    std::vector<Scratch> scratch(batch);
    for (size_t first = 0; first < num_frames; first += batch) {
      const size_t m = std::min(batch, num_frames - first);
      parallelFor(m, num_threads_, [&](size_t, size_t b, size_t e) {
        for (size_t j = b; j < e; ++j) {
          const size_t i = first + j;
          if (!frames_.frame(i).decode(num_pixels, rng[j], ids[j])) {
            LOG(ERROR) << "[PresentSurface] kept frame " << frames_.frame(i).stamp << " cannot be decoded.";
            rng[j].assign(num_pixels, 0);
            ids[j].assign(num_pixels, 0);
          }
          counted(i, rng[j], ids[j], scratch[j], &limit[j]);
        }
      }, 1);
      for (size_t j = 0; j < m; ++j) {
        const FrameCam c = frameCam(frames_.frame(first + j).world_T_sensor);
        const std::vector<uint16_t>& frame_rng = rng[j];
        const std::vector<float>& free_limit = limit[j];
        const bool limited = !free_limit.empty();
        parallelFor(memory_objects.size(), num_threads_, [&](size_t, size_t b, size_t e) {
          for (size_t k = b; k < e; ++k) {
            int u, v;
            float q;
            if (in_view[k] || !project(inside_pos[memory_objects[k].first], c, u, v, q)) continue;
            if (frame_rng[static_cast<size_t>(v) * W + u]) in_view[k] = 1;
          }
        }, 16384);
        parallelFor(tested.size(), num_threads_, [&](size_t, size_t b, size_t e) {
          for (size_t k = b; k < e; ++k) {
            int u, v;
            float q;
            if (!project(tested_centroid[k], c, u, v, q)) continue;
            const float tau = tauOf(tested_half[k], q);
            auto& ev = tested_ev[k];
            const uint16_t d0 = frame_rng[static_cast<size_t>(v) * W + u];
            if (d0) {
              const float r0 = d0 * 1e-3f - q;
              if (r0 < -tau) {
                ++ev.blocked;
                if (r0 >= -tested_trunc[k]) ++ev.blocked_band;
              }
              if (r0 >= -tested_trunc[k] && q < ev.q_reach) {  // reached: at most T in front
                ev.q_reach = q;
                ev.cam_reach = c.t;
              }
            }
            const float xn = (u - K.cx) / K.fx, yn = (v - K.cy) / K.fy;
            const float rp = K.fx * tau * std::sqrt(xn * xn + yn * yn + 1.f) / q;
            const int R = static_cast<int>(std::floor(rp));
            bool hit = false, all_valid = true, all_beyond = true;
            for (int dv = -R; dv <= R && !hit; ++dv) {
              for (int du = -R; du <= R; ++du) {
                if (du * du + dv * dv > rp * rp && (du || dv)) continue;
                const int x = u + du, y = v + dv;
                if (x < 0 || y < 0 || x >= W || y >= H) {
                  all_valid = false;
                  continue;
                }
                const size_t pix = static_cast<size_t>(y) * W + x;
                const uint16_t d = frame_rng[pix];
                if (!d) {
                  all_valid = false;
                  continue;
                }
                const float r = d * 1e-3f - q;
                if (std::abs(r) <= tau) {
                  // A hit: the reading's point lies inside the element's ball.
                  const Eigen::Vector3f ray((x - K.cx) / K.fx, (y - K.cy) / K.fy, 1.f);
                  const Eigen::Vector3f point = c.t + c.R * (ray.normalized() * (d * 1e-3f));
                  if ((point - tested_centroid[k]).squaredNorm() <= tau * tau) {
                    hit = true;
                    break;
                  }
                }
                if (r <= tau) all_beyond = false;
                else if (limited && q >= free_limit[pix] - T) all_valid = false;
              }
            }
            if (hit) {
              ++ev.hit;
              ev.q_hit = std::min(ev.q_hit, q);
            }
            else if (all_valid && all_beyond) ++ev.through;
          }
        }, 4096);
      }
    }
  }

  // The position error two sessions' measured depth scales explain per metre of
  // range: a reading scaled by (1 + s) is displaced by |s| q along its ray,
  // whatever the sign of s, so the memory element (built in an earlier
  // session, provenance unknown: the largest |s| bounds it) and the present
  // differ by at most (|s_prev| + |s_now|) q (triangle inequality).
  float s_prev = 0.f;
  for (const float scale : inputs.previous_depth_scales) s_prev = std::max(s_prev, std::abs(scale));
  const float error_per_metre = std::abs(depth_scale) + s_prev;
  if (!tested.empty()) {
    const TriangleGrid present_grid(Vp, Fp, nullptr, 4.f * v_f);
    // DIAG (P39/P47 range-dependent same-surface law; no decision reads it): distance of each tested
    // memory element to the nearest present surface within the displaced window, per range bin of
    // its reach and per side (camera side / far side), at the residual histogram resolution.
    std::vector<std::vector<int64_t>> disp_hist[2];
    for (auto& hs : disp_hist) hs.assign(kNumBins, {});
    for (size_t k = 0; k < tested.size(); ++k) {
      const Evidence& ev = tested_ev[k];
      const bool seen_through = ev.through > ev.hit;
      const bool hidden = !ev.hit && !ev.through && 2 * ev.blocked_band > ev.blocked;
      bool displaced = false;
      if (!seen_through && !hidden && error_per_metre > 0.f && std::isfinite(ev.q_reach)) {
        const Eigen::Vector3f& c = tested_centroid[k];
        const float window = tauOf(tested_half[k], ev.q_reach) + error_per_metre * ev.q_reach;
        float d;
        Eigen::Vector3f closest;
        uint32_t face;
        const bool near = present_grid.closest(c, window, d, closest, face);
        const bool camera_side = near && (closest - c).dot(ev.cam_reach - c) > 0.f;
        displaced = near && d > 2.f * tested_half[k] && camera_side;
        if (near && terminal) {
          const size_t rb = std::min(kNumBins - 1, static_cast<size_t>(std::max(0.f, ev.q_reach) / kRangeBin));
          const size_t db = static_cast<size_t>(d / kHistogramResolution);
          auto& h = disp_hist[camera_side ? 0 : 1][rb];
          if (h.size() <= db) h.resize(db + 1, 0);
          ++h[db];
        }
      }
      if (!seen_through && !hidden && !displaced) continue;
      shown_slot[tested[k]] = -1;
      ++(seen_through ? shown_seen_through : hidden ? shown_hidden : shown_displaced);
    }
    for (int side = 0; side < 2; ++side) {
      for (size_t b = 0; b < disp_hist[side].size(); ++b) {
        const auto& h = disp_hist[side][b];
        if (h.empty()) continue;
        std::string row;
        for (size_t k = 0; k < h.size(); ++k) row += (k ? "," : "") + std::to_string(h[k]);
        LOG(INFO) << "[SessionRefusion] DISPLACED_DIAG side=" << (side ? "far" : "camera") << " range_bin=" << b
                  << " range_bin_m=" << kRangeBin << " resolution_m=" << kHistogramResolution
                  << " half_voxel_m=" << tested_half[0] << " error_per_metre=" << error_per_metre
                  << " counts=" << row;
      }
    }
  }
  // INSIDE: an object's memory vertex inside the same object's present surface
  // gives way (a memory face with a retired vertex is dropped).
  std::vector<std::pair<uint32_t, uint32_t>> inside_candidates;
  for (size_t k = 0; k < memory_objects.size(); ++k) {
    if (in_view[k]) inside_candidates.push_back(memory_objects[k]);
  }
  const std::vector<uint8_t> retired =
      inside_candidates.empty()
          ? std::vector<uint8_t>(inside_pos.size(), 0)
          : insideMemory(inside_pos, inside_candidates, Vp, Fp, face_id, face_normal, v_f, num_threads_);
  const size_t num_retired = std::count(retired.begin(), retired.end(), 1);
  lock.unlock();
  report << ",\"memory\":{\"snapshot_memory_vertices\":" << num_memory
         << ",\"object_vertices\":" << memory_objects.size()
         << ",\"in_view\":" << inside_candidates.size() << ",\"inside_retired\":" << num_retired;
  if (shown_mode) {
    report << ",\"shown\":{\"faces\":" << inputs.shown->faces.size()
           << ",\"object_state\":" << shown_object_state << ",\"tested\":" << tested.size()
           << ",\"seen_through\":" << shown_seen_through << ",\"hidden\":" << shown_hidden
           << ",\"displaced\":" << shown_displaced << "}";
  }
  report << "}";

  // ------------------------------------------------------------ compose
  // Present faces: label 0 (or a label without a current node) -> background,
  // label L -> the node with physical id L (the largest if several). New
  // surface = the present, then (memory as the previous final map showed it)
  // the shown memory faces that stay; without a shown state, the snapshot's
  // memory faces that stay keep their own vertices.
  std::vector<Eigen::Vector3f> Vn;
  std::vector<Face3> Fn;
  const std::vector<Eigen::Vector3f>& Vnew = shown_mode ? Vn : Vp;
  const std::vector<Face3>& Fnew = shown_mode ? Fn : Fp;
  if (shown_mode) {
    Vn = Vp;
    Fn = Fp;
    const uint32_t offset = static_cast<uint32_t>(Vp.size());
    Vn.insert(Vn.end(), inputs.shown->vertices.begin(), inputs.shown->vertices.end());
    for (const auto& f : inputs.shown->faces) Fn.push_back({f[0] + offset, f[1] + offset, f[2] + offset});
  }
  std::vector<std::vector<uint32_t>> present_of_slot(slots.size()), kept_of_slot(slots.size());
  size_t present_unassigned = 0;
  for (uint32_t f = 0; f < Fp.size(); ++f) {
    const auto it = face_id[f] ? slot_of_label.find(face_id[f]) : slot_of_label.end();
    if (it != slot_of_label.end()) {
      present_of_slot[it->second].push_back(f);
    } else if (bg_slot >= 0) {
      present_of_slot[bg_slot].push_back(f);
    } else {
      ++present_unassigned;
    }
  }
  size_t memory_faces_total = 0, memory_faces_kept = 0;
  if (shown_mode) {
    for (uint32_t f = 0; f < inputs.shown->faces.size(); ++f) {
      const auto& sf = inputs.shown->faces[f];
      memory_faces_total += 1;
      if (shown_slot[f] < 0 || retired[sf[0]] || retired[sf[1]] || retired[sf[2]]) continue;
      present_of_slot[shown_slot[f]].push_back(static_cast<uint32_t>(Fp.size() + f));
      ++memory_faces_kept;
    }
  } else {
    for (uint32_t f = 0; f < faces.size(); ++f) {
      if (!memory_face[f]) continue;
      memory_faces_total += 1;
      if (retired[faces[f][0]] || retired[faces[f][1]] || retired[faces[f][2]]) continue;
      kept_of_slot[fslot[f]].push_back(f);
      ++memory_faces_kept;
    }
  }
  // Each slot's mesh becomes its kept faces (own vertices and attributes) and
  // its present faces; a present vertex takes its attributes from the nearest
  // vertex of the slot's own mesh, stamps no later than the round. An object
  // whose surface would be removed entirely keeps its mesh.
  size_t guarded = 0, object_faces_out = 0;
  for (uint32_t s = 0; s < slots.size(); ++s) {
    const Slot& slot = slots[s];
    auto& mesh = *slot.mesh;
    const auto& kept = kept_of_slot[s];
    const auto& present = present_of_slot[s];
    if (!slot.background && kept.empty() && present.empty()) {
      ++guarded;
      continue;
    }
    if (!slot.background) object_faces_out += kept.size() + present.size();
    const size_t old_n = mesh.numVertices();
    // New vertex list: vertices of kept faces (original order), then present vertices.
    std::vector<int64_t> kept_new(old_n, -1);
    for (const auto f : kept) {
      for (int k = 0; k < 3; ++k) kept_new[faces[f][k] - slot.begin] = 0;
    }
    std::vector<uint32_t> kept_old;
    for (size_t i = 0; i < old_n; ++i) {
      if (kept_new[i] >= 0) {
        kept_new[i] = static_cast<int64_t>(kept_old.size());
        kept_old.push_back(static_cast<uint32_t>(i));
      }
    }
    std::unordered_map<uint32_t, uint32_t> present_new;
    std::vector<uint32_t> present_old;
    for (const auto f : present) {
      for (int k = 0; k < 3; ++k) {
        const uint32_t pv = Fnew[f][k];
        if (present_new.emplace(pv, static_cast<uint32_t>(kept_old.size() + present_old.size())).second) {
          present_old.push_back(pv);
        }
      }
    }
    const size_t new_n = kept_old.size() + present_old.size();
    std::vector<size_t> nearest(present_old.size(), 0);
    if (old_n > 0 && !present_old.empty()) {
      const std::vector<Eigen::Vector3f> old_world(pos.begin() + slot.begin, pos.begin() + slot.end);
      const hydra::PointNeighborSearch search(old_world);
      parallelFor(present_old.size(), num_threads_, [&](size_t, size_t b, size_t e) {
        for (size_t k = b; k < e; ++k) {
          float d_sq = 0.f;
          size_t idx = 0;
          if (search.search(Vnew[present_old[k]], d_sq, idx)) nearest[k] = idx;
        }
      });
    }
    spark_dsg::Mesh::Positions points(new_n);
    for (size_t k = 0; k < kept_old.size(); ++k) points[k] = mesh.points[kept_old[k]];
    for (size_t k = 0; k < present_old.size(); ++k) {
      const Eigen::Vector3f& p = Vnew[present_old[k]];
      points[kept_old.size() + k] = slot.attrs ? slot.attrs->bounding_box.pointToBoxFrame(p) : p;
    }
    // An attribute the mesh carries: kept vertices keep theirs, a present vertex
    // takes the nearest own vertex's; with no own vertex yet, colors and labels
    // are the defaults and stamps the round.
    auto rebuild = [&](const auto& values, auto& out, const bool carried, const auto fallback,
                       const bool clamp) {
      using V = typename std::decay_t<decltype(values)>::value_type;
      out.clear();
      if (!carried) return;
      out.resize(new_n, fallback);
      if (values.size() != old_n || old_n == 0) return;
      for (size_t k = 0; k < kept_old.size(); ++k) out[k] = values[kept_old[k]];
      for (size_t k = 0; k < present_old.size(); ++k) out[kept_old.size() + k] = values[nearest[k]];
      if constexpr (std::is_integral_v<V>) {
        if (clamp) {
          for (size_t k = kept_old.size(); k < new_n; ++k) out[k] = std::min<V>(out[k], static_cast<V>(stamp));
        }
      }
    };
    spark_dsg::Mesh::Colors colors;
    spark_dsg::Mesh::Timestamps stamps, first_seen_stamps;
    spark_dsg::Mesh::Labels labels;
    rebuild(mesh.colors, colors, mesh.has_colors, spark_dsg::Color(), false);
    rebuild(mesh.stamps, stamps, mesh.has_timestamps, static_cast<spark_dsg::Mesh::Timestamp>(stamp), true);
    rebuild(mesh.first_seen_stamps, first_seen_stamps, mesh.has_first_seen_stamps,
            static_cast<spark_dsg::Mesh::Timestamp>(stamp), true);
    rebuild(mesh.labels, labels, mesh.has_labels, spark_dsg::Mesh::Label(0), false);
    spark_dsg::Mesh::Faces out_faces;
    out_faces.reserve(kept.size() + present.size());
    for (const auto f : kept) {
      out_faces.push_back({static_cast<size_t>(kept_new[faces[f][0] - slot.begin]),
                           static_cast<size_t>(kept_new[faces[f][1] - slot.begin]),
                           static_cast<size_t>(kept_new[faces[f][2] - slot.begin])});
    }
    for (const auto f : present) {
      out_faces.push_back({static_cast<size_t>(present_new[Fnew[f][0]]),
                           static_cast<size_t>(present_new[Fnew[f][1]]),
                           static_cast<size_t>(present_new[Fnew[f][2]])});
    }
    mesh.points.swap(points);
    mesh.colors.swap(colors);
    mesh.stamps.swap(stamps);
    mesh.first_seen_stamps.swap(first_seen_stamps);
    mesh.labels.swap(labels);
    mesh.faces.swap(out_faces);
  }
  const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  report << ",\"compose\":{\"memory_faces\":" << memory_faces_total
         << ",\"memory_faces_kept\":" << memory_faces_kept << ",\"guarded_nodes\":" << guarded
         << ",\"present_unassigned\":" << present_unassigned << "},\"elapsed_s\":" << elapsed << "}";
  result.report_json = report.str();
  if (dump) std::ofstream(inputs.dump_dir + "/refusion_report.json") << result.report_json << "\n";
  LOG(INFO) << "[PresentSurface] round " << stamp << " frames=" << num_frames
            << " units=" << num_units << " present_vertices=" << Vp.size()
            << " present_faces=" << Fp.size() << " object_faces=" << object_faces_out
            << " memory_faces_kept=" << memory_faces_kept << "/" << memory_faces_total
            << " seen_through=" << shown_seen_through << " hidden=" << shown_hidden
            << " displaced=" << shown_displaced << " inside_retired=" << num_retired
            << " depth_scale=" << depth_scale << " guarded_objects=" << guarded
            << " elapsed_s=" << elapsed;
  return result;
}

}  // namespace khronos
