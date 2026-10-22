#include "session_core/surface/present_surface.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
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

// Points grouped by cell (side `size`): each cell's bounds and members. A cell whose bounds no
// pixel's centre ray meets in a frame holds no point that projects into that frame.
struct Cells {
  std::vector<Eigen::AlignedBox3f> bounds;
  std::vector<std::vector<uint32_t>> members;
  Cells(const std::vector<Eigen::Vector3f>& points, float size) {
    std::unordered_map<uint64_t, uint32_t> index;
    for (uint32_t i = 0; i < points.size(); ++i) {
      const auto [it, inserted] = index.emplace(voxelKey(points[i], size), static_cast<uint32_t>(bounds.size()));
      if (inserted) {
        bounds.emplace_back();
        members.emplace_back();
      }
      bounds[it->second].extend(points[i]);
      members[it->second].push_back(i);
    }
  }
  /** fn(point) for the points of the cells whose bounds a pixel's centre ray of frame `c` meets. */
  template <typename Fn>
  void forEachInView(const FrameCam& c, const PresentTsdf::Camera& K, Fn&& fn) const {
    Eigen::Vector3f X[8];
    for (size_t cell = 0; cell < bounds.size(); ++cell) {
      for (int k = 0; k < 8; ++k) {
        X[k] = c.R.transpose() * (bounds[cell].corner(static_cast<Eigen::AlignedBox3f::CornerType>(k)) - c.t);
      }
      if (pixelBox(K, X, 8, kBoxEdges, 12).empty()) continue;
      for (const uint32_t i : members[cell]) fn(i);
    }
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
  // Inverse world ray direction per pixel, computed for the frame when first used (`frame` stamps it).
  std::vector<Eigen::Vector3f> inv_dir;
  std::vector<uint32_t> inv_frame;
  uint32_t frame = 0;
  Eigen::Vector3f toCamera(const Eigen::Vector3f& w) const { return R.transpose() * (w - c); }
};

bool PresentSurface::State::operator==(const State& other) const {
  return current == other.current && start == other.start && triangles == other.triangles &&
         voxels == other.voxels;
}

namespace {
// A reconstruction from every observation of what it covers uses a truncation
// of two voxels: Khronos's MeshObjectExtractor, and the session-end refusion
// this surface moves online.
hydra::VolumetricMap::Config wholeObservation(hydra::VolumetricMap::Config map) {
  map.truncation_distance = 2.0 * map.voxel_size;
  return map;
}
}  // namespace

PresentSurface::PresentSurface(const hydra::VolumetricMap::Config& map,
                               const hydra::ProjectiveIntegrator::Config& integrator,
                               const hydra::MeshIntegratorConfig& mesh)
    : voxel_(map.voxel_size),
      truncation_(wholeObservation(map).truncation_distance),
      num_threads_(integrator.num_threads),
      tsdf_(wholeObservation(map), integrator, mesh) {}

float PresentSurface::copyTolerance(const std::vector<Eigen::Vector3f>& loaded) {
  float extent = 0.f;
  for (const auto& m : loaded) extent = std::max(extent, m.cwiseAbs().maxCoeff());
  return std::sqrt(3.f) * 2.f * 6.f * (0.5f * std::numeric_limits<float>::epsilon()) * 2.f * extent;
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
    s.inv_frame.assign(n, 0);
  }
  if (++s.frame == 0) {  // the stamps wrapped: none is current
    std::fill(s.inv_frame.begin(), s.inv_frame.end(), 0);
    s.frame = 1;
  }
}

PresentSurface::Window PresentSurface::windowOf(size_t index, const Eigen::AlignedBox3f& box) const {
  if (box.isEmpty()) return {};
  const auto& pose = frames_.frame(index).world_T_sensor;
  const Eigen::Matrix3f Rt = pose.rotation().cast<float>().transpose();
  const Eigen::Vector3f c = pose.translation().cast<float>();
  Eigen::Vector3f X[8];
  for (int k = 0; k < 8; ++k) X[k] = Rt * (box.corner(static_cast<Eigen::AlignedBox3f::CornerType>(k)) - c);
  const PixelBox b = pixelBox(camera_, X, 8, kBoxEdges, 12);
  return {b.u0, b.u1, b.v0, b.v1};
}

bool PresentSurface::reaches(size_t index, const Eigen::AlignedBox3f& box, float beyond) const {
  // A ray meets the box no nearer than the box is to the camera centre.
  if (box.isEmpty()) return false;
  const auto& frame = frames_.frame(index);
  const Eigen::Vector3f c = frame.world_T_sensor.translation().cast<float>();
  const float nearest = (c.cwiseMax(box.min()).cwiseMin(box.max()) - c).norm();
  if (nearest > frame.farthest_mm * 1e-3f) return false;
  const Window window = windowOf(index, box);
  return !window.empty() &&
         nearest <= frame.farthestIn(camera_.width, window.u0, window.u1, window.v0, window.v1) * 1e-3f + beyond;
}

PresentSurface::Changed PresentSurface::changedPart(const State& before, const State& after) const {
  // A pixel's first hit changes only where a triangle or measured voxel of one surface is not one
  // of the other: those, and their bounds.
  using Triangle = std::array<float, 9>;
  const auto triangles = [](const State& state) {
    std::vector<Triangle> out(state.triangles.size() / 3);
    for (size_t f = 0; f < out.size(); ++f) {
      for (int k = 0; k < 3; ++k) {
        for (int a = 0; a < 3; ++a) out[f][3 * k + a] = state.triangles[3 * f + k][a];
      }
    }
    std::sort(out.begin(), out.end());
    return out;
  };
  const auto tb = triangles(before), ta = triangles(after);
  std::vector<Triangle> triangle_change;
  std::set_symmetric_difference(tb.begin(), tb.end(), ta.begin(), ta.end(), std::back_inserter(triangle_change));
  Changed changed;
  std::set_symmetric_difference(before.voxels.begin(), before.voxels.end(), after.voxels.begin(),
                                after.voxels.end(), std::back_inserter(changed.voxels));
  for (const auto& t : triangle_change) {
    for (int k = 0; k < 3; ++k) {
      changed.triangles.emplace_back(t[3 * k], t[3 * k + 1], t[3 * k + 2]);
      changed.box.extend(changed.triangles.back());
    }
  }
  const float vox = static_cast<float>(voxel_);
  for (const auto key : changed.voxels) {
    changed.box.extend(voxelCorner(key, vox));
    changed.box.extend(voxelCorner(key, vox) + Eigen::Vector3f::Constant(vox));
  }
  return changed;
}

void PresentSurface::markChanged(size_t index, const Changed& changed, std::vector<uint8_t>& mask,
                                 Window& touched) const {
  const auto& pose = frames_.frame(index).world_T_sensor;
  const Eigen::Matrix3f Rt = pose.rotation().cast<float>().transpose();
  const Eigen::Vector3f c = pose.translation().cast<float>();
  const int W = static_cast<int>(camera_.width);
  const auto fill = [&](const PixelBox& box) {
    if (box.empty()) return;
    for (int v = box.v0; v <= box.v1; ++v) {
      std::fill(mask.begin() + static_cast<size_t>(v) * W + box.u0, mask.begin() + static_cast<size_t>(v) * W + box.u1 + 1, 1);
    }
    touched.extend({box.u0, box.u1, box.v0, box.v1});
  };
  Eigen::Vector3f X[8];
  for (size_t f = 0; f + 2 < changed.triangles.size(); f += 3) {
    for (int k = 0; k < 3; ++k) X[k] = Rt * (changed.triangles[f + k] - c);
    fill(pixelBox(camera_, X, 3, kTriangleEdges, 3));
  }
  const float vox = static_cast<float>(voxel_);
  for (const auto key : changed.voxels) {
    const Eigen::Vector3f lo = voxelCorner(key, vox), hi = lo + Eigen::Vector3f::Constant(vox);
    for (int k = 0; k < 8; ++k) {
      X[k] = Rt * (Eigen::Vector3f((k & 1) ? hi.x() : lo.x(), (k & 2) ? hi.y() : lo.y(), (k & 4) ? hi.z() : lo.z()) - c);
    }
    fill(pixelBox(camera_, X, 8, kBoxEdges, 12));
  }
}

void PresentSurface::limitBy(const State& state, Scratch& s, std::vector<float>& limit,
                             const Window& window, const std::vector<uint8_t>* mask) const {
  if (state.triangles.empty() && state.voxels.empty()) return;
  const int W = static_cast<int>(camera_.width);
  Eigen::Vector3f X[8];
  for (int k = 0; k < 8; ++k) {
    X[k] = s.toCamera(state.box.corner(static_cast<Eigen::AlignedBox3f::CornerType>(k)));
  }
  const PixelBox region =
      pixelBox(camera_, X, 8, kBoxEdges, 12).clip({window.u0, window.u1, window.v0, window.v1});
  if (region.empty()) return;

  // The cells whose bounds the region's pixels can meet.
  std::vector<const State::Cell*> cells;
  for (const auto& cell : state.cells) {
    for (int k = 0; k < 8; ++k) X[k] = s.toCamera(cell.box.corner(static_cast<Eigen::AlignedBox3f::CornerType>(k)));
    if (!pixelBox(camera_, X, 8, kBoxEdges, 12).clip(region).empty()) cells.push_back(&cell);
  }

  // First hit of every pixel's ray on the current object mesh (Moller-Trumbore
  // from the camera centre, double precision as TriangleGrid).
  for (const auto* cell : cells) for (const uint32_t f : cell->triangles) {
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
        if (mask && !(*mask)[p]) continue;
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
    const float vox = static_cast<float>(voxel_);
    for (const auto* cell : cells) for (const uint32_t index : cell->voxels) {
      const Eigen::Vector3f lo = voxelCorner(state.voxels[index], vox);
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
          if ((mask && !(*mask)[p]) || !(s.free_end[p] > 0.f) || s.mesh_hit[p] < s.free_end[p]) continue;
          if (s.inv_frame[p] != s.frame) {
            s.inv_dir[p] = (s.R * ray_dir_[p]).cwiseInverse();
            s.inv_frame[p] = s.frame;
          }
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
      if (mask && !(*mask)[p]) continue;
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

std::vector<float> PresentSurface::apply(const Restriction& r,
                                         std::vector<uint16_t>& range,
                                         const std::vector<FrameArchive::InstanceRun>& runs) const {
  // A free limit is only ever finite on a pixel with a reading.
  std::vector<float> limit = r.limit;
  if (r.excluded.empty()) return limit;
  std::vector<uint16_t> excluded = r.excluded;
  std::sort(excluded.begin(), excluded.end());
  uint32_t begin = 0;
  for (const auto& run : runs) {
    if (run.id && std::binary_search(excluded.begin(), excluded.end(), run.id)) {
      std::fill(range.begin() + begin, range.begin() + run.end, 0);
      if (!limit.empty()) std::fill(limit.begin() + begin, limit.begin() + run.end, kInf);
    }
    begin = run.end;
  }
  if (std::none_of(limit.begin(), limit.end(), [](float l) { return l < kInf; })) limit.clear();
  return limit;
}

void PresentSurface::domainOf(const std::vector<uint16_t>& range, std::vector<float> limit,
                              Domain& domain) const {
  PresentTsdf::depthFromRange(ray_norm_, range, domain.depth);
  domain.limit = std::move(limit);
}

void PresentSurface::measure(const States& states, size_t index, std::vector<uint16_t>& range,
                             const std::vector<FrameArchive::InstanceRun>& runs, Scratch& scratch,
                             std::vector<float>& limit) const {
  Restriction r;
  const TimeStamp t = frames_.frame(index).stamp;
  const Window image{0, static_cast<int>(camera_.width) - 1, 0, static_cast<int>(camera_.height) - 1};
  for (const auto& [id, state] : states) {
    if (!state.restricts(t)) continue;
    if (r.limit.empty()) {
      prepare(index, range, scratch);
      r.limit.assign(range.size(), kInf);
    }
    r.excluded.push_back(id);
    limitBy(state, scratch, r.limit, image, nullptr);
  }
  limit = apply(r, range, runs);
}

bool PresentSurface::counted(size_t index, std::vector<uint16_t>& range, std::vector<float>* limit) const {
  std::vector<uint16_t> excluded;  // the states are ordered by id: sorted
  const auto& frame = frames_.frame(index);
  for (const auto& [id, state] : states_) {
    if (state.restricts(frame.stamp)) excluded.push_back(id);
  }
  if (!frame.decodeExcluding(ray_norm_.size(), excluded, range)) return false;
  if (limit) CHECK(frame.keptLimit(range.size(), *limit));
  return true;
}

void PresentSurface::integrate(const FrameData& data) {
  std::lock_guard<std::mutex> lock(mutex_);
  frames_.offer(data);
}

void PresentSurface::integrate(const FrameArchive::Camera& camera, FrameArchive::Frame frame) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<uint16_t> range_mm;
  std::vector<FrameArchive::InstanceRun> runs;
  if (!frame.decode(static_cast<size_t>(camera.width) * camera.height, range_mm, runs)) {
    LOG(ERROR) << "[PresentSurface] archived frame " << frame.stamp << " cannot be decoded; skipped.";
    return;
  }
  if (!frames_.keep(camera, std::move(frame))) return;
}

void PresentSurface::advanceTo(TimeStamp stamp) {
  if (!frames_.numFrames()) return;
  if (consumed_ == 0) {
    const auto& c = frames_.camera();
    camera_ = {c.width, c.height, c.fx, c.fy, c.cx, c.cy, c.min_range, c.max_range};
    ray_norm_ = PresentTsdf::rayNorm(camera_);
    ray_dir_.resize(ray_norm_.size());
    for (uint32_t v = 0; v < camera_.height; ++v) {
      for (uint32_t u = 0; u < camera_.width; ++u) {
        ray_dir_[static_cast<size_t>(v) * camera_.width + u] =
            Eigen::Vector3f((u - camera_.cx) / camera_.fx, (v - camera_.cy) / camera_.fy, 1.f).normalized();
      }
    }
  }
  // The frames through the stamp, each measured under the reflected states, integrated in order.
  std::vector<size_t> fresh;
  for (size_t i = consumed_; i < frames_.numFrames() && frames_.frame(i).stamp <= stamp; ++i) fresh.push_back(i);
  if (fresh.empty()) return;
  const size_t readers = static_cast<size_t>(std::max(1, num_threads_));
  std::vector<std::vector<uint16_t>> range(readers);
  std::vector<std::vector<FrameArchive::InstanceRun>> runs(readers);
  std::vector<Scratch> scratch(readers);
  std::vector<std::vector<uint8_t>> kept(fresh.size());
  tsdf_.integrate(camera_, ray_norm_, fresh, [&](size_t i) { return frames_.frame(i).world_T_sensor; },
                  [&](int reader, size_t i, std::vector<float>& depth, std::vector<float>& limit) {
    if (!frames_.frame(i).decode(ray_norm_.size(), range[reader], runs[reader])) return false;
    measure(states_, i, range[reader], runs[reader], scratch[reader], limit);
    PresentTsdf::depthFromRange(ray_norm_, range[reader], depth);
    kept[i - fresh.front()] = FrameArchive::packLimit(limit);
    return true;
  });
  for (const size_t i : fresh) frames_.keepLimit(i, std::move(kept[i - fresh.front()]));
  consumed_ = fresh.back() + 1;
}

void PresentSurface::recordEndpoints() {
  const size_t n = consumed_;
  if (recorded_ >= n) return;
  const size_t num_pixels = static_cast<size_t>(camera_.width) * camera_.height;
  const float vox = static_cast<float>(voxel_);
  std::mutex merge;
  parallelFor(n - recorded_, num_threads_, [&](size_t, size_t b, size_t e) {
    std::vector<uint16_t> range;
    std::vector<FrameArchive::InstanceRun> runs;
    std::unordered_map<uint16_t, std::unordered_map<uint64_t, TimeStamp>> local;
    for (size_t i = recorded_ + b; i < recorded_ + e; ++i) {
      const auto& frame = frames_.frame(i);
      if (!frame.decode(num_pixels, range, runs)) continue;
      const Eigen::Matrix3f R = frame.world_T_sensor.rotation().cast<float>();
      const Eigen::Vector3f c = frame.world_T_sensor.translation().cast<float>();
      uint32_t begin = 0;
      for (const auto& run : runs) {
        if (run.id && frame.sees(run.id)) {
          auto& keys = local[run.id];
          for (uint32_t p = begin; p < run.end; ++p) {
            if (range[p]) keys[voxelKey(c + R * (ray_dir_[p] * (range[p] * 1e-3f)), vox)] = frame.stamp;
          }
        }
        begin = run.end;
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

  // A frame's domain changes only through a changed state that restricts it before or after:
  // where the restriction appears or disappears, through the pixels of its identity the frame
  // reads and the free space its surface bounds; where it restricts the frame both before and
  // after, through the part of its surface or past voxels that differs. A surface bounds free
  // space only where the frame reads as far as it.
  std::vector<Changed> changed_part(changes.size());
  // DIAG (no decision reads it): changed / total triangles and voxels of the states that changed in place.
  size_t triangles_changed = 0, triangles_total = 0, voxels_changed = 0, voxels_total = 0;
  for (size_t c = 0; c < changes.size(); ++c) {
    if (!changes[c].before || !changes[c].after) continue;
    changed_part[c] = changedPart(*changes[c].before, *changes[c].after);
    triangles_changed += changed_part[c].triangles.size() / 3;
    triangles_total += (changes[c].before->triangles.size() + changes[c].after->triangles.size()) / 3;
    voxels_changed += changed_part[c].voxels.size();
    voxels_total += changes[c].before->voxels.size() + changes[c].after->voxels.size();
  }
  std::vector<size_t> candidates;
  for (size_t i = 0; i < consumed_; ++i) {
    const auto& frame = frames_.frame(i);
    if (frame.stamp > horizon) continue;
    for (size_t c = 0; c < changes.size(); ++c) {
      const bool before = changes[c].before && changes[c].before->restricts(frame.stamp);
      const bool after = changes[c].after && changes[c].after->restricts(frame.stamp);
      if (!before && !after) continue;
      const bool affects = before && after ? reaches(i, changed_part[c].box, 0.f)
                                           : frame.sees(changes[c].id) ||
                                                 reaches(i, (before ? changes[c].before : changes[c].after)->box, 0.f);
      if (affects) {
        candidates.push_back(i);
        break;
      }
    }
  }

  // Each candidate under both states, in parallel. Before: the domain the TSDF counts (the pixels
  // the reflected states take, the kept free limit). After: the pixels the new states take, and the
  // kept free limit measured anew where it can rise or move -- under the surface of a state that
  // stops restricting the frame, on a returned pixel, where the surface of a state restricting it
  // before and after differs; a state that starts restricting it only lowers it, through its own
  // surface. A moved frame keeps its new limit and marks the voxels its move changes.
  const size_t num_pixels = static_cast<size_t>(camera_.width) * camera_.height;
  const int W = static_cast<int>(camera_.width);
  std::vector<Scratch> scratch(num_threads_);
  std::mutex moving;
  size_t moved = 0;
  // DIAG (no decision reads it): worker seconds decoding, measuring limits anew (of which in
  // limitBy), finding changed voxels.
  std::atomic<int64_t> decode_ns{0}, measure_ns{0}, limits_ns{0}, voxels_ns{0};
  const auto ns = [](std::chrono::steady_clock::time_point from) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - from).count();
  };
  parallelFor(candidates.size(), num_threads_, [&](size_t w, size_t b, size_t e) {
    std::vector<uint16_t> range, range_before, range_after, taken_before, taken_after;
    std::vector<FrameArchive::InstanceRun> runs;
    std::vector<float> limit_before, limit_after;
    std::vector<uint8_t> anew(num_pixels, 0);  // the pixels whose limit is measured anew
    std::vector<std::pair<const State*, Window>> starting;  // with the pixels its surface can limit
    Domain before, after;
    for (size_t j = b; j < e; ++j) {
      const size_t index = candidates[j];
      const auto& frame = frames_.frame(index);
      const TimeStamp t = frame.stamp;
      auto clock = std::chrono::steady_clock::now();
      if (!frame.decode(num_pixels, range, runs)) {
        LOG(ERROR) << "[PresentSurface] kept frame " << t << " cannot be decoded.";
        continue;
      }
      decode_ns += ns(clock);
      clock = std::chrono::steady_clock::now();
      taken_before.clear();
      taken_after.clear();
      for (const auto& [id, state] : states_) if (state.restricts(t)) taken_before.push_back(id);
      for (const auto& [id, state] : next) if (state.restricts(t)) taken_after.push_back(id);
      range_before = range;
      range_after = range;
      Window window;   // bounds of `anew`
      Window differs;  // bounds of the pixels whose reading or limit can differ
      uint32_t begin = 0;
      for (const auto& run : runs) {
        const bool from = run.id && std::binary_search(taken_before.begin(), taken_before.end(), run.id);
        const bool to = run.id && std::binary_search(taken_after.begin(), taken_after.end(), run.id);
        if (from) std::fill(range_before.begin() + begin, range_before.begin() + run.end, 0);
        if (to) std::fill(range_after.begin() + begin, range_after.begin() + run.end, 0);
        if (from != to && run.end > begin) {
          const int v0 = static_cast<int>(begin / W), v1 = static_cast<int>((run.end - 1) / W);
          differs.extend({v0 == v1 ? static_cast<int>(begin % W) : 0, v0 == v1 ? static_cast<int>((run.end - 1) % W) : W - 1,
                          v0, v1});
        }
        if (from && !to) {
          for (uint32_t q = begin; q < run.end; ++q) {
            if (!range[q]) continue;
            anew[q] = 1;
            const int u = static_cast<int>(q % W), v = static_cast<int>(q / W);
            window.extend({u, u, v, v});
          }
        }
        begin = run.end;
      }
      CHECK(frame.keptLimit(num_pixels, limit_before));
      if (limit_before.empty()) {
        limit_after.assign(num_pixels, kInf);
      } else {
        limit_after = limit_before;
      }
      starting.clear();
      for (size_t c = 0; c < changes.size(); ++c) {
        const bool from = changes[c].before && changes[c].before->restricts(t);
        const bool to = changes[c].after && changes[c].after->restricts(t);
        if (from && to) {
          markChanged(index, changed_part[c], anew, window);
        } else if (from) {
          const Window box = windowOf(index, changes[c].before->box);
          if (box.empty()) continue;
          for (int v = box.v0; v <= box.v1; ++v) {
            std::fill(anew.begin() + static_cast<size_t>(v) * W + box.u0, anew.begin() + static_cast<size_t>(v) * W + box.u1 + 1, 1);
          }
          window.extend(box);
        } else if (to) {
          const Window box = windowOf(index, changes[c].after->box);
          if (box.empty()) continue;
          starting.emplace_back(changes[c].after, box);
          differs.extend(box);
        }
      }
      differs.extend(window);
      const auto limits = std::chrono::steady_clock::now();
      if (!window.empty() || !starting.empty()) prepare(index, range, scratch[w]);
      if (!window.empty()) {
        for (int v = window.v0; v <= window.v1; ++v) {
          for (int u = window.u0; u <= window.u1; ++u) {
            const size_t q = static_cast<size_t>(v) * W + u;
            if (anew[q]) limit_after[q] = kInf;
          }
        }
        for (const auto& [id, state] : next) {
          if (state.restricts(t)) limitBy(state, scratch[w], limit_after, window, &anew);
        }
        for (int v = window.v0; v <= window.v1; ++v) {
          std::fill(anew.begin() + static_cast<size_t>(v) * W + window.u0, anew.begin() + static_cast<size_t>(v) * W + window.u1 + 1, 0);
        }
      }
      for (const auto& [state, box] : starting) limitBy(*state, scratch[w], limit_after, box, nullptr);
      limits_ns += ns(limits);
      // As apply(): a taken pixel has no limit; no limited pixel, no limit.
      begin = 0;
      for (const auto& run : runs) {
        if (run.id && std::binary_search(taken_after.begin(), taken_after.end(), run.id)) {
          std::fill(limit_after.begin() + begin, limit_after.begin() + run.end, kInf);
        }
        begin = run.end;
      }
      if (std::none_of(limit_after.begin(), limit_after.end(), [](float l) { return l < kInf; })) limit_after.clear();
      measure_ns += ns(clock);
      if (range_before == range_after && limit_before == limit_after) continue;
      clock = std::chrono::steady_clock::now();
      domainOf(range_before, std::move(limit_before), before);
      domainOf(range_after, std::move(limit_after), after);
      const auto voxels =
          tsdf_.changedVoxels(camera_, frame.world_T_sensor, ray_norm_, before.view(), after.view(), differs);
      auto packed = FrameArchive::packLimit(after.limit);
      voxels_ns += ns(clock);
      std::lock_guard<std::mutex> lock(moving);
      tsdf_.invalidate(voxels);
      frames_.keepLimit(index, std::move(packed));
      ++moved;
    }
  }, 1);
  states_ = std::move(next);
  const auto examined = std::chrono::steady_clock::now();
  // The marked voxels, from every frame that reads as far as one (a voxel the native integrator
  // updates lies no farther than the frame's farthest reading, nor than the band it updates behind
  // the reading the voxel projects onto).
  const auto marked = tsdf_.marked();
  const float behind = tsdf_.behind();
  Eigen::AlignedBox3f all;
  for (const auto& box : marked) all.extend(box);
  std::vector<uint8_t> reads(consumed_, 0);
  parallelFor(consumed_, num_threads_, [&](size_t, size_t b, size_t e) {
    for (size_t i = b; i < e; ++i) {
      reads[i] = reaches(i, all, behind) &&
                 std::any_of(marked.begin(), marked.end(), [&](const auto& box) { return reaches(i, box, behind); });
    }
  }, 64);
  std::vector<size_t> replay;
  for (size_t i = 0; i < consumed_; ++i) if (reads[i]) replay.push_back(i);
  std::vector<std::vector<uint16_t>> replay_range(num_threads_);
  const auto rebuilt = tsdf_.rebuild(camera_, ray_norm_, replay,
                [&](size_t i) { return frames_.frame(i).world_T_sensor; },
                [&](int reader, size_t i, std::vector<float>& depth, std::vector<float>& limit) {
    CHECK(counted(i, replay_range[reader], &limit));
    PresentTsdf::depthFromRange(ray_norm_, replay_range[reader], depth);
    return true;
  });
  const auto now = std::chrono::steady_clock::now();
  LOG(INFO) << "[PresentSurface] states " << states_.size() << ": " << changes.size()
            << " changed, " << candidates.size() << " frame(s) examined, " << moved
            << " moved in " << std::chrono::duration<double>(now - start).count() << " s (examine "
            << std::chrono::duration<double>(examined - start).count() << " s, rebuild "
            << std::chrono::duration<double>(now - examined).count() << " s; worker s decode/measure (limits)/voxels "
            << decode_ns * 1e-9 << "/" << measure_ns * 1e-9 << " (" << limits_ns * 1e-9 << ")/" << voxels_ns * 1e-9
            << "; rebuild frames "
            << rebuilt.frames << " read/integrate s " << rebuilt.read_s << "/" << rebuilt.integrate_s
            << "; changed triangles " << triangles_changed << "/" << triangles_total << " voxels " << voxels_changed
            << "/" << voxels_total << ")";
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

}  // namespace

PresentSurface::Result PresentSurface::compose(spark_dsg::DynamicSceneGraph& dsg,
                                               const TimeStamp stamp,
                                               const std::map<size_t, TimeStamp>& state_starts,
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
  // session, with its current object meshes.
  States next;
  for (const auto& [id, t] : state_starts) {
    if (id == 0 || id > std::numeric_limits<uint16_t>::max()) continue;
    State& state = next[static_cast<uint16_t>(id)];
    state.current = true;
    state.start = t;
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
  // Seconds of the round's phases: integrating its new frames, the states, the mesh, the face
  // identities and tested elements, the evidence, the rest (INSIDE, composing the meshes).
  auto since = [&, mark = std::chrono::steady_clock::now()]() mutable {
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - mark).count();
    mark = now;
    return seconds;
  };
  since();
  advanceTo(stamp);
  const double integrate_s = since();
  const size_t num_frames = consumed_;
  if (num_frames == 0) return result;
  std::vector<Eigen::Vector3f> Vp;
  std::vector<Face3> Fp;
  double states_s = 0.0, mesh_s = 0.0;
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
      for (const auto key : state.voxels) {
        state.box.extend(voxelCorner(key, vox));
        state.box.extend(voxelCorner(key, vox) + Eigen::Vector3f::Constant(vox));
      }
      const float cell_size = tsdf_.map().blockSize();
      std::unordered_map<uint64_t, uint32_t> cell_of;
      const auto cell = [&](const Eigen::Vector3f& at) -> State::Cell& {
        const auto [it, inserted] = cell_of.emplace(voxelKey(at, cell_size), static_cast<uint32_t>(state.cells.size()));
        if (inserted) state.cells.emplace_back();
        return state.cells[it->second];
      };
      for (uint32_t f = 0; f + 2 < state.triangles.size(); f += 3) {
        auto& c = cell(state.triangles[f]);
        for (int k = 0; k < 3; ++k) c.box.extend(state.triangles[f + k]);
        c.triangles.push_back(f);
      }
      for (uint32_t v = 0; v < state.voxels.size(); ++v) {
        auto& c = cell(voxelCorner(state.voxels[v], vox));
        c.box.extend(voxelCorner(state.voxels[v], vox));
        c.box.extend(voxelCorner(state.voxels[v], vox) + Eigen::Vector3f::Constant(vox));
        c.voxels.push_back(v);
      }
    }
    updateStates(std::move(next));
    states_s = since();
    tsdf_.extractMesh(Vp, Fp);
    mesh_s = since();
  }
  const bool dump = terminal && !inputs.dump_dir.empty();
  if (dump) {
    std::filesystem::create_directories(inputs.dump_dir);
    FrameArchive::save(inputs.dump_dir + "/archive.kfa",
                       std::vector<FrameArchive::Frame>(frames_.frames().begin(),
                                                       frames_.frames().begin() + consumed_),
                       frames_.camera());
  }
  const size_t num_units = tsdf_.numUnits();
  std::stringstream report;
  report << "{\"stamp\":" << stamp << ",\"frames\":" << num_frames
         << ",\"observation_end_ns\":" << frames_.frame(num_frames - 1).stamp
         << ",\"voxel\":" << voxel_
         << ",\"truncation\":" << truncation_
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

  // ------------------------------------ memory and fill: surfaces built from
  // earlier observations. Memory is the previous final map as it showed it;
  // fill is this session's own background, the window reconstructions the
  // backend archived (the session-end refusion's measured fill, moved online:
  // the present holds one surface per place, the windows keep what each of
  // them measured). A shown face of an object stays only while the node of its
  // physical id is current and its state did not begin in this session: the
  // object reasoning decides where objects are, as in Khronos, and no per-face
  // vote applies to it. A memory background face on a closed state's old
  // surface follows that state. Every other face is a surface element
  // known to within tau = max(h, sigma) (h: half a voxel), decided as the
  // fusion decides a place: at the element's own pixel, each frame the present
  // counts reads it (within tau, and nearer it than any other present layer
  // along the ray: support) or beyond it (free space the present counts:
  // against); a reading in front of it is no vote. The majority decides.
  // Without one, the volume decides: an element behind another present layer
  // within the truncation, facing the same way, is that layer's place (one
  // volume holds one surface there) and gives way; otherwise a face keeps its
  // status: memory stays shown, fill is not shown.
  const auto& K = camera_;
  const int W = static_cast<int>(K.width), H = static_cast<int>(K.height);
  const Projector project{K.fx, K.fy, K.cx, K.cy, W, H};
  // The present's truncation (eq. 8). A shown face was built by an earlier present or
  // object reconstruction: the same voxel (half of it: h) and the same truncation.
  const float T = static_cast<float>(truncation_);
  const float h = 0.5f * v_f;
  const bool shown_mode = inputs.shown != nullptr;
  // A closed state owns the shown-memory faces that lie on its old surface, the same
  // ownership the live background uses. The per-face evidence test below cannot see
  // the state's death; this does, so the old site follows the state atomically.
  std::unique_ptr<hydra::PointNeighborSearch> closed_memory_search;
  const float closed_memory_radius = v_f;
  const float closed_memory_radius_squared = closed_memory_radius * closed_memory_radius;
  if (shown_mode && !inputs.closed_memory.empty()) {
    closed_memory_search =
        std::make_unique<hydra::PointNeighborSearch>(inputs.closed_memory);
  }
  const size_t num_vertices = pos.size();
  std::vector<uint8_t> memory(num_vertices, 0), memory_face(faces.size(), 0);
  if (inputs.is_memory) {
    parallelFor(num_vertices, num_threads_, [&](size_t, size_t b, size_t e) {
      for (size_t i = b; i < e; ++i) memory[i] = inputs.is_memory(pos[i]) ? 1 : 0;
    });
  }
  const size_t num_memory = std::count(memory.begin(), memory.end(), 1);
  for (size_t f = 0; f < faces.size(); ++f) {
    const int n = memory[faces[f][0]] + memory[faces[f][1]] + memory[faces[f][2]];
    memory_face[f] = n == 3;
  }
  std::vector<int32_t> shown_slot;  // per shown face: target slot, -1 = not shown
  std::vector<uint32_t> tested;     // shown faces the evidence decides
  // Per decided element: centroid and normal (towards the free side, as meshing orients it).
  std::vector<Eigen::Vector3f> tested_centroid, tested_normal;
  auto addTested = [&](const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c) {
    tested_centroid.push_back((a + b + c) / 3.f);
    tested_normal.push_back((b - a).cross(c - a));
  };
  size_t shown_object_state = 0, shown_closed_state = 0, shown_outvoted = 0, shown_covered = 0;
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
      } else if (closed_memory_search) {
        const Eigen::Vector3f centroid = (S.vertices[S.faces[f][0]] + S.vertices[S.faces[f][1]] +
                                          S.vertices[S.faces[f][2]]) / 3.f;
        float distance_squared = 0.f;
        size_t index = 0;
        if (closed_memory_search->search(centroid, distance_squared, index) &&
            distance_squared <= closed_memory_radius_squared) {
          ++shown_closed_state;
          continue;
        }
      }
      if (target < 0) continue;
      shown_slot[f] = target;
      if (p > 0) continue;  // an object follows its state, as in Khronos
      tested.push_back(static_cast<uint32_t>(f));
      addTested(S.vertices[S.faces[f][0]], S.vertices[S.faces[f][1]], S.vertices[S.faces[f][2]]);
    }
  }
  // The snapshot's own background arrives with its closed states' old sites already
  // removed (markClosedObjectBackground: a vertex measured no later than the state's
  // support); a surface measured after a state left is not that state's.
  std::vector<uint32_t> fill;  // snapshot background faces the evidence decides
  size_t fill_outvoted = 0, fill_covered = 0, fill_unreached = 0;
  for (uint32_t f = 0; f < faces.size(); ++f) {
    if (static_cast<int32_t>(fslot[f]) != bg_slot || memory_face[f]) continue;
    fill.push_back(f);
    addTested(pos[faces[f][0]], pos[faces[f][1]], pos[faces[f][2]]);
  }
  struct Evidence {
    uint16_t support = 0, free = 0;
  };
  std::vector<Evidence> tested_ev(tested_centroid.size());
  // An element's rival: the present's other layer near it (farther than one
  // voxel, within the truncation). A reading belongs to the one surface it
  // landed on: a reading nearer the rival's crossing of the ray is the
  // present's, not the element's. And the volume holds one surface per place:
  // an element behind its rival, facing the same way, lies within that layer's
  // truncation band (`covered`), where the present's own surface is.
  std::vector<Eigen::Vector3f> rival_point(tested_centroid.size()), rival_normal(tested_centroid.size());
  std::vector<uint8_t> rival(tested_centroid.size(), 0), covered(tested_centroid.size(), 0);
  if (!tested_centroid.empty() && !Fp.empty()) {
    const TriangleGrid present_grid(Vp, Fp, nullptr, 4.f * v_f);
    parallelFor(tested_centroid.size(), num_threads_, [&](size_t, size_t b, size_t e) {
      for (size_t k = b; k < e; ++k) {
        float d;
        uint32_t face;
        if (!present_grid.closest(tested_centroid[k], T, d, rival_point[k], face) || d <= 2.f * h) continue;
        const float length = face_normal[face].norm();
        if (!(length > 0.f)) continue;
        rival_normal[k] = face_normal[face] / length;
        rival[k] = 1;
        covered[k] = (tested_centroid[k] - rival_point[k]).dot(rival_normal[k]) < 0.f &&
                     tested_normal[k].dot(rival_normal[k]) > 0.f;
      }
    });
  }
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
  const double assign_s = since();  // face identities, memory and fill elements, rivals
  const float cell_size = tsdf_.map().blockSize();  // the cells points are culled in, per frame
  std::vector<uint8_t> in_view(memory_objects.size(), 0);

  // sigma(q): the depth residual per range bin, measured on the present
  // (1.4826 * median |reading - range| of front-facing present vertices within
  // one truncation, every frame as the present counts it). tau(h, q) =
  // max(h, sigma(q)) is how well a surface element of half-voxel h is known
  // along a ray at range q.
  std::vector<float> sigma(kNumBins, 0.f);
  if (!tested_centroid.empty() && !Vp.empty()) {
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
    // A position the mesh stores under several vertices (one per face) projects once; each of
    // them counts with its own normal. Positions by their bits; `members` of position g are
    // vertices at[g] .. at[g + 1].
    std::vector<std::array<uint32_t, 4>> bits(Vp.size());
    for (uint32_t j = 0; j < Vp.size(); ++j) {
      std::memcpy(bits[j].data(), Vp[j].data(), 3 * sizeof(float));
      bits[j][3] = j;
    }
    std::sort(bits.begin(), bits.end());
    std::vector<Eigen::Vector3f> position;
    std::vector<uint32_t> at, members(Vp.size());
    for (size_t k = 0; k < bits.size(); ++k) {
      if (k == 0 || !std::equal(bits[k].begin(), bits[k].begin() + 3, bits[k - 1].begin())) {
        at.push_back(static_cast<uint32_t>(k));
        position.push_back(Vp[bits[k][3]]);
      }
      members[k] = bits[k][3];
    }
    at.push_back(static_cast<uint32_t>(bits.size()));
    const Cells vertex_cells(position, cell_size);
    parallelFor(num_frames, num_threads_, [&](size_t, size_t b, size_t e) {
      std::vector<int64_t> local(kNumBins * nh, 0);
      std::vector<uint16_t> rng;
      for (size_t i = b; i < e; ++i) {
        if (!counted(i, rng, nullptr)) continue;
        const FrameCam c = frameCam(frames_.frame(i).world_T_sensor);
        vertex_cells.forEachInView(c, K, [&](uint32_t g) {
          int u, v;
          float q;
          if (!project(position[g], c, u, v, q)) return;
          const uint16_t d = rng[static_cast<size_t>(v) * W + u];
          if (!d) return;
          const double r = std::abs(d * 1e-3 - static_cast<double>(q));
          if (r > truncation_) return;
          const size_t bin = std::min(kNumBins - 1, static_cast<size_t>(q / kRangeBin));
          const size_t cell = std::min(nh - 1, static_cast<size_t>(r / kHistogramResolution));
          const Eigen::Vector3f toward = c.t - position[g];
          for (uint32_t m = at[g]; m < at[g + 1]; ++m) local[bin * nh + cell] += N[members[m]].dot(toward) > 0.f;
        });
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
        LOG(INFO) << "[PresentSurface] NOISE_HIST_DIAG range_bin=" << b
                  << " range_bin_m=" << kRangeBin << " resolution_m=" << kHistogramResolution
                  << " counts=" << row;
      }
    }
    report << ",\"sigma_cm\":[";
    for (size_t b = 0; b < kNumBins; ++b) report << (b ? "," : "") << sigma[b] * 100.f;
    report << "]";
  }
  const double sigma_s = since();
  auto tauOf = [&](float half, float q) {
    const size_t bin = std::min(kNumBins - 1, static_cast<size_t>(std::max(0.f, q) / kRangeBin));
    return std::max(half, sigma[bin]);
  };

  // One pass over the frames, each as the present counts it: INSIDE candidates
  // in view and the evidence at the tested elements. Each worker counts the
  // frames it reads; the counts add up.
  if (!tested_centroid.empty() || !memory_objects.empty()) {
    std::vector<std::vector<Evidence>> worker_ev(num_threads_, std::vector<Evidence>(tested_centroid.size()));
    std::vector<std::vector<uint8_t>> worker_view(num_threads_, std::vector<uint8_t>(memory_objects.size(), 0));
    std::vector<Eigen::Vector3f> memory_points(memory_objects.size());
    for (size_t k = 0; k < memory_objects.size(); ++k) memory_points[k] = inside_pos[memory_objects[k].first];
    const Cells point_cells(memory_points, cell_size), element_cells(tested_centroid, cell_size);
    parallelFor(num_frames, num_threads_, [&](size_t w, size_t b, size_t e) {
      std::vector<uint16_t> frame_rng;
      std::vector<float> free_limit;
      auto& ev = worker_ev[w];
      auto& view = worker_view[w];
      for (size_t i = b; i < e; ++i) {
        if (!counted(i, frame_rng, &free_limit)) {
          LOG(ERROR) << "[PresentSurface] kept frame " << frames_.frame(i).stamp << " cannot be decoded.";
          continue;
        }
        const FrameCam c = frameCam(frames_.frame(i).world_T_sensor);
        const bool limited = !free_limit.empty();
        point_cells.forEachInView(c, K, [&](uint32_t k) {
          int u, v;
          float q;
          if (view[k] || !project(memory_points[k], c, u, v, q)) return;
          if (frame_rng[static_cast<size_t>(v) * W + u]) view[k] = 1;
        });
        element_cells.forEachInView(c, K, [&](uint32_t k) {
          int u, v;
          float q;
          if (!project(tested_centroid[k], c, u, v, q)) return;
          const float tau = tauOf(h, q);
          const size_t pix = static_cast<size_t>(v) * W + u;
          const uint16_t d = frame_rng[pix];
          if (!d) return;
          const float r = d * 1e-3f - q;
          // Where the other present layer crosses this ray, relative to the element.
          float crossing = kInf;
          if (rival[k]) {
            const float cosine = ((tested_centroid[k] - c.t) / q).dot(rival_normal[k]);
            if (cosine != 0.f) crossing = (rival_point[k] - tested_centroid[k]).dot(rival_normal[k]) / cosine;
          }
          const bool support = std::abs(r) <= tau && std::abs(r) <= std::abs(r - crossing);
          if (support) ++ev[k].support;
          else if (r > 0.f) ev[k].free += !(limited && q >= free_limit[pix] - T);
        });
      }
    }, 4);
    for (size_t w = 0; w < worker_ev.size(); ++w) {
      for (size_t k = 0; k < tested_centroid.size(); ++k) {
        tested_ev[k].support += worker_ev[w][k].support;
        tested_ev[k].free += worker_ev[w][k].free;
      }
      for (size_t k = 0; k < memory_objects.size(); ++k) in_view[k] |= worker_view[w][k];
    }
  }

  const double evidence_s = since();

  // The fusion's verdict on element k: 1 shown, 0 not, -1 no evidence either way.
  auto verdict = [&](size_t k) {
    const Evidence& ev = tested_ev[k];
    if (ev.support != ev.free) return ev.support > ev.free ? 1 : 0;
    return covered[k] ? 0 : -1;
  };
  std::vector<uint8_t> fill_kept(fill.size(), 0);
  for (size_t k = 0; k < tested_centroid.size(); ++k) {
    const int v = verdict(k);
    const bool outvoted = tested_ev[k].free > tested_ev[k].support;
    if (k >= tested.size()) {
      const size_t j = k - tested.size();
      fill_kept[j] = v == 1;
      if (v == 0) ++(outvoted ? fill_outvoted : fill_covered);
      if (v == -1) ++fill_unreached;
      continue;
    }
    if (v != 0) continue;
    shown_slot[tested[k]] = -1;
    ++(outvoted ? shown_outvoted : shown_covered);
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
           << ",\"object_state\":" << shown_object_state
           << ",\"closed_state\":" << shown_closed_state << ",\"tested\":" << tested.size()
           << ",\"outvoted\":" << shown_outvoted << ",\"covered\":" << shown_covered << "}";
  }
  report << ",\"fill\":{\"faces\":" << fill.size() << ",\"outvoted\":" << fill_outvoted
         << ",\"covered\":" << fill_covered << ",\"unreached\":" << fill_unreached
         << ",\"kept\":" << std::count(fill_kept.begin(), fill_kept.end(), 1) << "}";
  report << "}";

  // ------------------------------------------------------------ compose
  // Present faces: label 0 (or a label without a current node) -> background,
  // label L -> the node with physical id L (the largest if several). New
  // surface = the present, then (memory as the previous final map showed it)
  // the shown memory faces that stay; without a shown state, the snapshot's
  // memory faces that stay keep their own vertices, as do the fill faces.
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
  for (size_t j = 0; j < fill.size(); ++j) {
    if (fill_kept[j]) kept_of_slot[bg_slot].push_back(fill[j]);
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
            << " outvoted=" << shown_outvoted << " covered=" << shown_covered
            << " fill_kept=" << std::count(fill_kept.begin(), fill_kept.end(), 1) << "/"
            << fill.size() << " inside_retired=" << num_retired
            << " guarded_objects=" << guarded << " phases_s(integrate/states/mesh/assign/sigma/votes/rest)="
            << integrate_s << "/" << states_s << "/" << mesh_s << "/" << assign_s << "/" << sigma_s << "/"
            << evidence_s << "/" << since() << " elapsed_s=" << elapsed;
  return result;
}

}  // namespace khronos
