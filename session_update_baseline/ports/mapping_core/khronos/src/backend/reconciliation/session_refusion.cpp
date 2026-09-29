#include "khronos/backend/reconciliation/session_refusion.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <glog/logging.h>
#include <hydra/utils/nearest_neighbor_utilities.h>

#include "khronos/backend/reconciliation/present_tsdf.h"
#include "khronos/backend/reconciliation/triangle_grid.h"
#include "khronos/backend/update_khronos_objects_functor.h"
#include "khronos/utils/khronos_attribute_utils.h"

namespace khronos {

namespace {

using Face3 = std::array<uint32_t, 3>;
constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr size_t kNumIds = 65536;

// Dynamic chunked parallel loop: fn(begin, end) over [0, n).
template <typename Fn>
void parallelFor(size_t n, int num_threads, Fn&& fn, size_t chunk = 2048) {
  if (n == 0) return;
  const size_t workers =
      std::max<size_t>(1, std::min<size_t>(std::max(1, num_threads), (n + chunk - 1) / chunk));
  if (workers <= 1) {
    fn(size_t(0), n);
    return;
  }
  std::atomic<size_t> next{0};
  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (size_t w = 0; w < workers; ++w) {
    threads.emplace_back([&] {
      while (true) {
        const size_t begin = next.fetch_add(chunk);
        if (begin >= n) break;
        fn(begin, std::min(n, begin + chunk));
      }
    });
  }
  for (auto& t : threads) t.join();
}

double rssMb() {
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      std::istringstream ss(line.substr(6));
      double kb = 0;
      ss >> kb;
      return kb / 1024.0;
    }
  }
  return 0.0;
}

struct FrameCam {
  Eigen::Matrix3f R;  // camera_to_world rotation
  Eigen::Vector3f t;  // camera centre (world)
};

// Pinhole projection: camera-frame z > 0.1, rounded pixel floor(x / z * f + c + 0.5),
// q = |p_C| (the range along the pixel's ray).
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

/**
 * The session's frames as valid evidence: the archived range and instance ids,
 * with the readings of an object before its current state began voided and, for
 * those earlier frames, the free-space limit where their rays meet where the
 * object is now.
 */
class SessionFrames {
 public:
  SessionFrames(const std::vector<FrameArchive::Frame>& frames,
                const FrameArchive::Camera& camera,
                const std::map<size_t, TimeStamp>& state_starts)
      : frames_(frames),
        num_pixels_(static_cast<size_t>(camera.width) * camera.height),
        start_of_(kNumIds, 0),
        limit_pixels_(frames.size()),
        limit_ranges_(frames.size()) {
    cams_.resize(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
      cams_[i].R = frames[i].world_T_sensor.linear().cast<float>();
      cams_[i].t = frames[i].world_T_sensor.translation().cast<float>();
    }
    for (const auto& [id, t_L] : state_starts) {
      if (id > 0 && id < kNumIds) start_of_[id] = t_L;
    }
  }

  size_t size() const { return frames_.size(); }
  size_t numPixels() const { return num_pixels_; }
  TimeStamp stamp(size_t i) const { return frames_[i].stamp; }
  const FrameCam& cam(size_t i) const { return cams_[i]; }
  const Eigen::Isometry3d& pose(size_t i) const { return frames_[i].world_T_sensor; }

  /** The archived frame (a corrupt frame reads as empty). */
  void decode(size_t i, std::vector<uint16_t>& range, std::vector<uint16_t>& ids) const {
    if (!frames_[i].decode(num_pixels_, range, ids)) {
      LOG(ERROR) << "[SessionRefusion] archived frame " << frames_[i].stamp
                 << " cannot be decoded; it reads as empty.";
      range.assign(num_pixels_, 0);
      ids.assign(num_pixels_, 0);
    }
  }

  /** Void the readings of every object before its current state began; counts per id. */
  void cut(size_t i, std::vector<uint16_t>& range, const std::vector<uint16_t>& ids,
           std::unordered_map<uint16_t, size_t>* removed = nullptr) const {
    const TimeStamp t = frames_[i].stamp;
    for (size_t p = 0; p < num_pixels_; ++p) {
      if (range[p] && t < start_of_[ids[p]]) {
        range[p] = 0;
        if (removed) ++(*removed)[ids[p]];
      }
    }
  }

  /** Frame i as valid evidence (readings of earlier object states voided). */
  void load(size_t i, std::vector<uint16_t>& range, std::vector<uint16_t>& ids) const {
    decode(i, range, ids);
    cut(i, range, ids);
  }

  /** Whether some ray of frame i has its free space limited. */
  bool limited(size_t i) const { return !limit_pixels_[i].empty(); }

  /** Free-space limit per pixel of frame i: +inf, or the range where the ray meets an object now there. */
  void freeLimit(size_t i, std::vector<float>& limit) const {
    limit.assign(num_pixels_, kInf);
    for (size_t k = 0; k < limit_pixels_[i].size(); ++k) limit[limit_pixels_[i][k]] = limit_ranges_[i][k];
  }

  /** Set the limited pixels of frame i (one entry per pixel). */
  void setFreeLimit(size_t i, std::vector<uint32_t> pixels, std::vector<float> ranges) {
    limit_pixels_[i] = std::move(pixels);
    limit_ranges_[i] = std::move(ranges);
  }

  /** fn(i, range, ids) for every valid frame in order; the next frame is loaded meanwhile. */
  template <typename Fn>
  void forEach(Fn&& fn) const {
    if (frames_.empty()) return;
    std::vector<uint16_t> range[2], ids[2];
    load(0, range[0], ids[0]);
    for (size_t i = 0; i < frames_.size(); ++i) {
      const size_t cur = i % 2, next = 1 - cur;
      std::future<void> ahead;
      if (i + 1 < frames_.size()) {
        ahead = std::async(std::launch::async, [&, i, next] { load(i + 1, range[next], ids[next]); });
      }
      fn(i, range[cur], ids[cur]);
      if (ahead.valid()) ahead.get();
    }
  }

 private:
  const std::vector<FrameArchive::Frame>& frames_;
  const size_t num_pixels_;
  std::vector<FrameCam> cams_;
  std::vector<TimeStamp> start_of_;  // physical id -> t_L (0: no state began in this session)
  std::vector<std::vector<uint32_t>> limit_pixels_;
  std::vector<std::vector<float>> limit_ranges_;
};

// One mesh of the final map: the background or one current object mesh.
struct Slot {
  bool background = false;
  NodeId node = 0;
  spark_dsg::Mesh* mesh = nullptr;
  KhronosObjectAttributes* attrs = nullptr;
  uint32_t begin = 0, end = 0;  // global vertex range
  uint32_t physical = 0;        // node physical id (0: background / none)
};

struct StepTimer {
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point last = start;
  std::stringstream timings;
  void step(const std::string& name, const std::string& detail = "") {
    const auto now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(now - last).count();
    const double total = std::chrono::duration<double>(now - start).count();
    last = now;
    timings << (timings.tellp() > 0 ? "," : "") << "\"" << name << "\":" << dt;
    LOG(INFO) << "[SessionRefusion] step=" << name << " elapsed_s=" << dt << " total_s=" << total
              << " rss_mb=" << static_cast<int>(rssMb()) << (detail.empty() ? "" : " ") << detail;
  }
};

// Robust scale per range bin: 1.4826 * median |r|, the median interpolated in
// the histogram cell where the cumulative count reaches half; a bin with too
// few samples takes the nearest estimated bin (lower side first).
std::vector<float> sigmaFromHistogram(const std::vector<std::vector<int64_t>>& hist,
                                      size_t min_samples,
                                      double resolution) {
  const size_t nb = hist.size();
  std::vector<double> sig(nb, std::numeric_limits<double>::quiet_NaN());
  for (size_t b = 0; b < nb; ++b) {
    int64_t cnt = 0;
    for (const auto c : hist[b]) cnt += c;
    if (cnt < static_cast<int64_t>(min_samples)) continue;
    const double half = 0.5 * static_cast<double>(cnt);
    int64_t cum = 0;
    for (size_t k = 0; k < hist[b].size(); ++k) {
      const int64_t prev = cum;
      cum += hist[b][k];
      if (static_cast<double>(cum) >= half) {
        sig[b] = 1.4826 * (static_cast<double>(k) +
                           (half - static_cast<double>(prev)) /
                               static_cast<double>(std::max<int64_t>(hist[b][k], 1))) *
                 resolution;
        break;
      }
    }
  }
  std::vector<double> filled = sig;
  for (size_t b = 0; b < nb; ++b) {
    if (std::isfinite(filled[b])) continue;
    for (size_t off = 1; off < nb; ++off) {
      if (b >= off && std::isfinite(sig[b - off])) {
        filled[b] = sig[b - off];
        break;
      }
      if (b + off < nb && std::isfinite(sig[b + off])) {
        filled[b] = sig[b + off];
        break;
      }
    }
  }
  std::vector<float> out(nb, 0.f);
  for (size_t b = 0; b < nb; ++b) out[b] = std::isfinite(filled[b]) ? static_cast<float>(filled[b]) : 0.f;
  return out;
}

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

// Round a float map scale to the decimal it was configured with (0.02f -> 0.02).
double decimal(float value) { return std::round(static_cast<double>(value) * 1e6) / 1e6; }

/**
 * The session's depth scale s: a surface point measured by one frame and
 * re-measured by another is read at the same position whatever the two ranges
 * when depth and trajectory agree in scale; a depth short (long) by a fixed
 * fraction of the range places the same surface at different positions from
 * near and far views. s makes the frames agree best: every reading scaled by
 * (1 + s), s minimises the median disagreement of re-measured points (pixels of
 * one frame projected into another, associated within `association`). Frame
 * subset, pixel stride and search grid are estimator settings; exact depth
 * gives s = 0. All archived readings take part (a reading measures its own
 * surface whatever the object states).
 */
float depthScale(const SessionFrames& frames, const FrameArchive::Camera& K,
                 const Projector& project, float association, int threads) {
  constexpr size_t kMaxFrames = 64;
  constexpr int kPixelStride = 16;
  struct View {
    FrameCam cam;
    std::vector<uint16_t> range;
  };
  std::vector<View> views;
  const size_t step = std::max<size_t>(1, frames.size() / kMaxFrames);
  std::vector<uint16_t> ids;
  for (size_t i = 0; i < frames.size() && views.size() < kMaxFrames; i += step) {
    View view;
    view.cam = frames.cam(i);
    frames.decode(i, view.range, ids);
    views.push_back(std::move(view));
  }
  if (views.size() < 2) return 0.f;
  struct Sample {
    Eigen::Vector3f origin, direction, other;
    float range = 0.f, other_range = 0.f;
  };
  std::vector<std::vector<Sample>> per_view(views.size());
  const int W = static_cast<int>(K.width), H = static_cast<int>(K.height);
  parallelFor(views.size(), threads, [&](size_t b, size_t e) {
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
  if (samples.size() < 1000) return 0.f;
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
  float best_s = 0.f, best = disagreement(0.f);
  for (int k = -50; k <= 50; ++k) {  // coarse: +-10 % in 0.2 % steps
    const float scale = 0.002f * k, m = disagreement(scale);
    if (m < best) best = m, best_s = scale;
  }
  const float coarse = best_s;
  for (int k = -10; k <= 10; ++k) {  // fine: 0.02 % steps around the coarse optimum
    const float scale = coarse + 0.0002f * k, m = disagreement(scale);
    if (m < best) best = m, best_s = scale;
  }
  return best_s;
}

// Voxel key of the object lattice.
struct VoxelLattice {
  float v;
  static constexpr int64_t kOff = int64_t(1) << 20;
  uint64_t key(int64_t x, int64_t y, int64_t z) const {
    return (static_cast<uint64_t>(x + kOff) << 42) | (static_cast<uint64_t>(y + kOff) << 21) |
           static_cast<uint64_t>(z + kOff);
  }
  uint64_t keyOf(const Eigen::Vector3f& p) const {
    return key(static_cast<int64_t>(std::floor(p.x() / v)), static_cast<int64_t>(std::floor(p.y() / v)),
               static_cast<int64_t>(std::floor(p.z() / v)));
  }
  Eigen::Matrix<int64_t, 3, 1> index(uint64_t key) const {
    return {static_cast<int64_t>(key >> 42) - kOff, static_cast<int64_t>((key >> 21) & 0x1FFFFF) - kOff,
            static_cast<int64_t>(key & 0x1FFFFF) - kOff};
  }
};

/**
 * Where an object whose current state began within this session is now: its
 * online mesh in the final map, and the voxels its own valid readings measured
 * (dense copy over their bounding box for fast lookups).
 */
struct CurrentExtent {
  size_t label = 0;
  TimeStamp t_L = 0;
  std::vector<uint32_t> mesh_faces;
  std::unique_ptr<TriangleGrid> grid;
  std::unordered_set<uint64_t> measured;
  Eigen::AlignedBox3f measured_box;
  Eigen::Matrix<int64_t, 3, 1> dense_lo = Eigen::Matrix<int64_t, 3, 1>::Zero(), dense_dim = dense_lo;
  std::vector<uint8_t> dense;
  std::vector<Eigen::Vector3f> corners;  // of the joint bounding box
  size_t frames_before = 0;
  std::atomic<size_t> cut{0}, limited{0};

  bool isMeasured(const VoxelLattice& lattice, int64_t x, int64_t y, int64_t z) const {
    if (dense.empty()) return measured.count(lattice.key(x, y, z)) > 0;
    x -= dense_lo.x();
    y -= dense_lo.y();
    z -= dense_lo.z();
    if (x < 0 || y < 0 || z < 0 || x >= dense_dim.x() || y >= dense_dim.y() || z >= dense_dim.z()) return false;
    return dense[(x * dense_dim.y() + y) * dense_dim.z() + z] != 0;
  }

  /**
   * Range at which the ray c + t d (t in [0, t_end]) first meets the extent:
   * the online mesh, else the first measured voxel (3D DDA over the part of the
   * ray inside their bounding box). False if it does not.
   */
  bool firstMeet(const VoxelLattice& lattice, const Eigen::Vector3f& c, const Eigen::Vector3f& dir,
                 float t_end, float& t_meet) const {
    uint32_t hit_face = 0;
    if (grid && grid->firstHit(c, dir, t_meet, hit_face) && t_meet < t_end) return true;
    if (measured.empty() || !(t_end > 0.f)) return false;
    const float v = lattice.v;
    float t0 = 0.f, t1 = t_end;
    for (int k = 0; k < 3 && t0 <= t1; ++k) {
      if (std::abs(dir[k]) < 1e-9f) {
        if (c[k] < measured_box.min()[k] || c[k] > measured_box.max()[k]) t0 = t1 + 1.f;
        continue;
      }
      float ta = (measured_box.min()[k] - c[k]) / dir[k];
      float tb = (measured_box.max()[k] - c[k]) / dir[k];
      if (ta > tb) std::swap(ta, tb);
      t0 = std::max(t0, ta);
      t1 = std::min(t1, tb);
    }
    if (t0 > t1) return false;
    const Eigen::Vector3f p0 = c + dir * t0;
    int64_t x = static_cast<int64_t>(std::floor(p0.x() / v));
    int64_t y = static_cast<int64_t>(std::floor(p0.y() / v));
    int64_t z = static_cast<int64_t>(std::floor(p0.z() / v));
    const int sx = dir.x() > 0 ? 1 : -1, sy = dir.y() > 0 ? 1 : -1, sz = dir.z() > 0 ? 1 : -1;
    auto next = [&](int64_t cell, int step, float o, float dk) {
      if (std::abs(dk) < 1e-9f) return kInf;
      const float boundary = static_cast<float>(cell + (step > 0 ? 1 : 0)) * v;
      return (boundary - o) / dk;
    };
    float tx = next(x, sx, c.x(), dir.x()), ty = next(y, sy, c.y(), dir.y()),
          tz = next(z, sz, c.z(), dir.z());
    const float dx = std::abs(dir.x()) < 1e-9f ? kInf : v / std::abs(dir.x());
    const float dy = std::abs(dir.y()) < 1e-9f ? kInf : v / std::abs(dir.y());
    const float dz = std::abs(dir.z()) < 1e-9f ? kInf : v / std::abs(dir.z());
    float t = t0;
    while (t <= t1) {
      if (isMeasured(lattice, x, y, z)) {
        t_meet = t;
        return true;
      }
      if (tx <= ty && tx <= tz) {
        t = tx;
        tx += dx;
        x += sx;
      } else if (ty <= tz) {
        t = ty;
        ty += dy;
        y += sy;
      } else {
        t = tz;
        tz += dz;
        z += sz;
      }
    }
    return false;
  }
};

/**
 * Validity of the free space of earlier readings: for every object whose
 * current state began within this session at t_L, a frame before t_L sees free
 * space only up to one truncation before its ray meets where the object is
 * now. Sets the free-space limits of `frames` and reports the voided readings
 * and limited rays per object.
 */
void limitEarlierFreeSpace(SessionFrames& frames, const FrameArchive::Camera& K,
                           const std::map<size_t, TimeStamp>& state_starts,
                           const std::set<uint32_t>& current_ids,
                           const std::vector<Eigen::Vector3f>& pos, const std::vector<Face3>& faces,
                           const std::vector<uint32_t>& face_object, float voxel, float trunc,
                           int threads, std::stringstream& report, size_t& total_cut,
                           size_t& total_limited) {
  const int W = static_cast<int>(K.width), H = static_cast<int>(K.height);
  const size_t num_pixels = frames.numPixels();
  const VoxelLattice lattice{voxel};
  std::vector<CurrentExtent> extents(state_starts.size());
  std::vector<int32_t> extent_of(kNumIds, -1);
  TimeStamp latest = 0;
  {
    size_t index = 0;
    for (const auto& [label, t_L] : state_starts) {
      CurrentExtent& s = extents[index];
      s.label = label;
      s.t_L = t_L;
      if (label < kNumIds) extent_of[label] = static_cast<int32_t>(index);
      latest = std::max(latest, t_L);
      for (uint32_t f = 0; f < faces.size(); ++f) {
        if (face_object[f] == label) s.mesh_faces.push_back(f);
      }
      ++index;
    }
  }
  // The voxels each such object's own valid readings measured from t_L on.
  {
    std::vector<Eigen::Vector3f> dirs(num_pixels);
    for (int v = 0; v < H; ++v)
      for (int u = 0; u < W; ++u)
        dirs[static_cast<size_t>(v) * W + u] =
            Eigen::Vector3f((u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.f).normalized();
    std::vector<uint8_t> measured_id(kNumIds, 0);
    for (const auto& s : extents) {
      if (s.label < kNumIds && current_ids.count(static_cast<uint32_t>(s.label))) measured_id[s.label] = 1;
    }
    std::mutex merge;
    parallelFor(frames.size(), threads, [&](size_t b, size_t e) {
      std::vector<uint16_t> range, ids;
      std::map<size_t, std::unordered_set<uint64_t>> local;
      for (size_t i = b; i < e; ++i) {
        frames.load(i, range, ids);
        const FrameCam& c = frames.cam(i);
        for (size_t p = 0; p < num_pixels; ++p) {
          if (!range[p] || !measured_id[ids[p]]) continue;
          local[ids[p]].insert(lattice.keyOf(c.t + c.R * (dirs[p] * (range[p] * 1e-3f))));
        }
      }
      std::lock_guard<std::mutex> lock(merge);
      for (auto& [label, keys] : local) {
        auto& target = extents[extent_of[label]].measured;
        target.insert(keys.begin(), keys.end());
      }
    }, 64);
  }
  for (auto& s : extents) {
    if (!s.measured.empty()) {
      for (const auto key : s.measured) {
        const Eigen::Vector3f lo = lattice.index(key).cast<float>() * voxel;
        s.measured_box.extend(lo);
        s.measured_box.extend(lo + Eigen::Vector3f::Constant(voxel));
      }
      Eigen::Matrix<int64_t, 3, 1> hi;
      for (int k = 0; k < 3; ++k) {
        s.dense_lo[k] = static_cast<int64_t>(std::floor(s.measured_box.min()[k] / voxel + 0.5f));
        hi[k] = static_cast<int64_t>(std::floor(s.measured_box.max()[k] / voxel + 0.5f));
        s.dense_dim[k] = hi[k] - s.dense_lo[k] + 1;
      }
      const int64_t cells = s.dense_dim.prod();
      if (cells > 0 && cells <= (int64_t(1) << 28)) {
        s.dense.assign(static_cast<size_t>(cells), 0);
        for (const auto key : s.measured) {
          const Eigen::Matrix<int64_t, 3, 1> i = lattice.index(key) - s.dense_lo;
          s.dense[(i.x() * s.dense_dim.y() + i.y()) * s.dense_dim.z() + i.z()] = 1;
        }
      }
    }
    if (s.mesh_faces.empty() && s.measured.empty()) continue;
    if (!s.mesh_faces.empty()) s.grid = std::make_unique<TriangleGrid>(pos, faces, &s.mesh_faces, 4.f * voxel);
    Eigen::AlignedBox3f box = s.grid ? s.grid->bounds() : s.measured_box;
    if (!s.measured.empty()) box.extend(s.measured_box);
    for (int c = 0; c < 8; ++c) {
      s.corners.emplace_back((c & 1) ? box.max().x() : box.min().x(),
                             (c & 2) ? box.max().y() : box.min().y(),
                             (c & 4) ? box.max().z() : box.min().z());
    }
  }
  std::vector<size_t> before;
  for (size_t i = 0; i < frames.size(); ++i) {
    if (frames.stamp(i) < latest) before.push_back(i);
  }
  for (auto& s : extents) {
    for (const size_t i : before) s.frames_before += frames.stamp(i) < s.t_L;
  }
  parallelFor(before.size(), threads, [&](size_t b, size_t e) {
    std::vector<uint16_t> range, ids;
    std::unordered_map<uint16_t, size_t> removed;
    for (size_t j = b; j < e; ++j) {
      const size_t i = before[j];
      const FrameCam& c = frames.cam(i);
      frames.decode(i, range, ids);
      removed.clear();
      frames.cut(i, range, ids, &removed);
      for (const auto& [id, n] : removed) {
        if (extent_of[id] >= 0) extents[extent_of[id]].cut += n;
      }
      std::vector<uint32_t> pixels;
      std::vector<float> meets;
      for (auto& s : extents) {
        if ((!s.grid && s.measured.empty()) || frames.stamp(i) >= s.t_L) continue;
        int u0 = 0, u1 = W - 1, v0 = 0, v1 = H - 1;
        bool all_front = true;
        float umin = kInf, umax = -kInf, vmin = kInf, vmax = -kInf;
        for (const auto& corner : s.corners) {
          const Eigen::Vector3f pc = c.R.transpose() * (corner - c.t);
          if (!(pc.z() > 0.05f)) {
            all_front = false;
            break;
          }
          const float uu = pc.x() / pc.z() * K.fx + K.cx, vv = pc.y() / pc.z() * K.fy + K.cy;
          umin = std::min(umin, uu);
          umax = std::max(umax, uu);
          vmin = std::min(vmin, vv);
          vmax = std::max(vmax, vv);
        }
        if (all_front) {
          u0 = static_cast<int>(std::max(0.f, std::floor(umin)));
          u1 = static_cast<int>(std::min(static_cast<float>(W - 1), std::ceil(umax)));
          v0 = static_cast<int>(std::max(0.f, std::floor(vmin)));
          v1 = static_cast<int>(std::min(static_cast<float>(H - 1), std::ceil(vmax)));
          if (u0 > u1 || v0 > v1) continue;
        }
        size_t local = 0;
        for (int v = v0; v <= v1; ++v) {
          for (int u = u0; u <= u1; ++u) {
            const size_t p = static_cast<size_t>(v) * W + u;
            const uint16_t d = range[p];
            if (!d) continue;
            const Eigen::Vector3f dir =
                c.R * Eigen::Vector3f((u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.f).normalized();
            float t_meet = 0.f;
            if (s.firstMeet(lattice, c.t, dir, d * 1e-3f - trunc, t_meet)) {
              pixels.push_back(static_cast<uint32_t>(p));
              meets.push_back(t_meet);
              ++local;
            }
          }
        }
        s.limited += local;
      }
      if (pixels.empty()) continue;
      // One entry per pixel: the nearest meeting.
      std::vector<size_t> order(pixels.size());
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return pixels[a] != pixels[b] ? pixels[a] < pixels[b] : meets[a] < meets[b];
      });
      std::vector<uint32_t> px;
      std::vector<float> th;
      for (const auto k : order) {
        if (!px.empty() && px.back() == pixels[k]) continue;
        px.push_back(pixels[k]);
        th.push_back(meets[k]);
      }
      px.shrink_to_fit();
      th.shrink_to_fit();
      frames.setFreeLimit(i, std::move(px), std::move(th));
    }
  }, 1);
  report << ",\"state_starts\":[";
  for (size_t k = 0; k < extents.size(); ++k) {
    const CurrentExtent& s = extents[k];
    total_cut += s.cut;
    total_limited += s.limited;
    LOG(INFO) << "[SessionRefusion] state_start id=" << s.label << " t_L=" << s.t_L
              << " frames_before=" << s.frames_before << " cut_pixels=" << s.cut.load()
              << " limited_rays=" << s.limited.load() << " mesh_faces=" << s.mesh_faces.size()
              << " measured_voxels=" << s.measured.size();
    report << (k ? "," : "") << "{\"id\":" << s.label << ",\"t_L\":" << s.t_L
           << ",\"frames_before\":" << s.frames_before << ",\"cut_pixels\":" << s.cut.load()
           << ",\"limited_rays\":" << s.limited.load() << ",\"mesh_faces\":" << s.mesh_faces.size()
           << ",\"measured_voxels\":" << s.measured.size() << "}";
  }
  report << "]";
}

// Pairs of faces (indices offset by `base`) that share an edge: the edges of
// the listed faces stably sorted by key; consecutive equal keys pair their
// faces (with multiplicity).
void sharedEdgePairs(const std::vector<Face3>& F, const std::vector<uint32_t>& listed, uint64_t num_vertices,
                     uint32_t base, std::vector<std::pair<uint32_t, uint32_t>>& pairs) {
  const uint64_t nV1 = num_vertices + 1;
  std::vector<std::pair<uint64_t, uint32_t>> edges;
  edges.reserve(3 * listed.size());
  for (int e = 0; e < 3; ++e) {
    for (uint32_t k = 0; k < listed.size(); ++k) {
      const Face3& f = F[listed[k]];
      uint64_t a = f[e], b = f[(e + 1) % 3];
      if (a > b) std::swap(a, b);
      edges.emplace_back(a * nV1 + b, base + k);
    }
  }
  std::stable_sort(edges.begin(), edges.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
  for (size_t k = 1; k < edges.size(); ++k) {
    if (edges[k].first == edges[k - 1].first) pairs.emplace_back(edges[k - 1].second, edges[k].second);
  }
}

/**
 * Identity of the session's surface, the present faces and the session's own
 * faces that fill it, from the object reasoning; and the range of the nearest
 * valid view that reached each face (its measurement record).
 * The reasoning states an object's membership twice: by the surface it
 * reconstructed for the object's current state, and by its pixel identities.
 * A face that re-measures a current object's reconstructed surface (within one
 * voxel, `reconstructed`) belongs to that object. Elsewhere the reasoning's
 * voxel membership rule applies to its pixel identities: a frame measures a
 * face when the face is front-facing, its centroid projects onto a valid
 * reading and the reading lies within tau of it; the frame observes object L
 * when it holds a valid reading of L; object L owns a measured face when, among
 * the frames that observe L and measure the face, at least `confidence` read it
 * as L and at least `min_observations` do (the most voted object if several,
 * the smallest id on a tie); a measured face no object owns is the static
 * structure's. A face neither reconstructed nor measured takes the majority
 * identity of its decided faces at the smallest distance along the surface
 * (`adjacency`, with multiplicity; the smallest id on a tie); faces no decided
 * face reaches are the static structure's.
 */
struct SurfaceIdentity {
  std::vector<uint32_t> identity;
  std::vector<float> reach;
  size_t reconstructed = 0, measured = 0, propagated = 0, unreached = 0;
};

SurfaceIdentity identifySurface(const SessionFrames& frames, const FrameArchive::Camera& K,
                                const Projector& project, const std::vector<Eigen::Vector3f>& centroid,
                                const std::vector<Eigen::Vector3f>& face_normal,
                                const std::vector<uint32_t>& reconstructed,
                                const std::vector<std::pair<uint32_t, uint32_t>>& adjacency,
                                const std::function<float(float)>& tau, float reach_trunc,
                                float confidence, int min_observations, int threads) {
  const int W = static_cast<int>(K.width);
  const size_t nF = centroid.size();
  const size_t num_pixels = frames.numPixels();
  // Compact indices of the ids the frames carry (ascending, index 0 = id 0).
  std::vector<uint16_t> ids_of;
  std::vector<uint16_t> compact(kNumIds, 0);
  {
    std::vector<uint8_t> seen(kNumIds, 0);
    seen[0] = 1;
    for (const auto id : reconstructed) {
      if (id < kNumIds) seen[id] = 1;
    }
    std::mutex merge;
    parallelFor(frames.size(), threads, [&](size_t b, size_t e) {
      std::vector<uint16_t> range, ids;
      std::vector<uint8_t> local(kNumIds, 0);
      for (size_t i = b; i < e; ++i) {
        frames.load(i, range, ids);
        for (size_t p = 0; p < num_pixels; ++p) {
          if (range[p]) local[ids[p]] = 1;
        }
      }
      std::lock_guard<std::mutex> lock(merge);
      for (size_t id = 0; id < kNumIds; ++id) seen[id] |= local[id];
    }, 16);
    for (size_t id = 0; id < kNumIds; ++id) {
      if (!seen[id]) continue;
      compact[id] = static_cast<uint16_t>(ids_of.size());
      ids_of.push_back(static_cast<uint16_t>(id));
    }
  }
  const size_t nlab = ids_of.size();
  SurfaceIdentity out;
  out.reach.assign(nF, kInf);
  std::vector<uint16_t> votes(nF * nlab, 0), observed(nF * nlab, 0);
  std::vector<uint8_t> measured(nF, 0);
  frames.forEach([&](size_t i, const std::vector<uint16_t>& range, const std::vector<uint16_t>& ids) {
    const FrameCam& c = frames.cam(i);
    // The objects this frame observes.
    std::vector<uint8_t> in_frame(nlab, 0);
    for (size_t p = 0; p < num_pixels; ++p) {
      if (range[p]) in_frame[compact[ids[p]]] = 1;
    }
    std::vector<uint16_t> observing;
    for (size_t k = 1; k < nlab; ++k) {
      if (in_frame[k]) observing.push_back(static_cast<uint16_t>(k));
    }
    parallelFor(nF, threads, [&](size_t b, size_t e) {
      for (size_t f = b; f < e; ++f) {
        int u, v;
        float q;
        if (!project(centroid[f], c, u, v, q)) continue;
        const size_t pix = static_cast<size_t>(v) * W + u;
        const uint16_t d = range[pix];
        if (!d) continue;
        const float r = d * 1e-3f - q;
        if (r >= -reach_trunc && q < out.reach[f]) out.reach[f] = q;
        if (face_normal[f].dot(c.t - centroid[f]) <= 0.f) continue;
        if (std::abs(r) > tau(q)) continue;
        measured[f] = 1;
        uint16_t* vote = &votes[f * nlab];
        uint16_t* seen = &observed[f * nlab];
        ++vote[compact[ids[pix]]];
        for (const auto k : observing) ++seen[k];
      }
    }, 8192);
  });
  out.identity.assign(nF, 0);
  std::vector<int32_t> lab(nF, -1);  // compact identity; -1: undecided
  for (size_t f = 0; f < nF; ++f) {
    if (reconstructed[f] > 0 && reconstructed[f] < kNumIds) {
      lab[f] = compact[reconstructed[f]];
      ++out.reconstructed;
      continue;
    }
    if (!measured[f]) continue;
    ++out.measured;
    const uint16_t* vote = &votes[f * nlab];
    const uint16_t* seen = &observed[f * nlab];
    int32_t best = 0;
    uint32_t best_votes = 0;
    for (size_t k = 1; k < nlab; ++k) {
      const uint32_t n = vote[k];
      if (n == 0 || n < static_cast<uint32_t>(std::max(0, min_observations))) continue;
      if (static_cast<float>(n) < confidence * static_cast<float>(seen[k])) continue;
      if (n > best_votes) {
        best_votes = n;
        best = static_cast<int32_t>(k);
      }
    }
    lab[f] = best;
  }
  votes.clear();
  votes.shrink_to_fit();
  observed.clear();
  observed.shrink_to_fit();
  std::vector<uint32_t> adj_begin(nF + 1, 0), adj;
  for (const auto& [a, b] : adjacency) {
    ++adj_begin[a + 1];
    ++adj_begin[b + 1];
  }
  for (size_t f = 0; f < nF; ++f) adj_begin[f + 1] += adj_begin[f];
  adj.resize(adj_begin[nF]);
  {
    std::vector<uint32_t> fill(adj_begin.begin(), adj_begin.end() - 1);
    for (const auto& [a, b] : adjacency) {
      adj[fill[a]++] = b;
      adj[fill[b]++] = a;
    }
  }
  // Distance layers from the decided faces; a face of layer d takes the
  // majority identity of its neighbours in layer d - 1.
  std::vector<uint32_t> layer(nF, std::numeric_limits<uint32_t>::max());
  std::vector<uint32_t> frontier, next;
  for (uint32_t f = 0; f < nF; ++f) {
    if (lab[f] >= 0) {
      layer[f] = 0;
      frontier.push_back(f);
    }
  }
  std::vector<uint32_t> count(nlab, 0);
  for (uint32_t d = 1; !frontier.empty(); ++d) {
    next.clear();
    for (const auto f : frontier) {
      for (uint32_t k = adj_begin[f]; k < adj_begin[f + 1]; ++k) {
        const uint32_t g = adj[k];
        if (layer[g] == std::numeric_limits<uint32_t>::max()) {
          layer[g] = d;
          next.push_back(g);
        }
      }
    }
    for (const auto g : next) {
      int32_t best = -1;
      uint32_t best_count = 0;
      for (uint32_t k = adj_begin[g]; k < adj_begin[g + 1]; ++k) {
        const uint32_t h = adj[k];
        if (layer[h] + 1 != d) continue;
        const int32_t l = lab[h];
        const uint32_t n = ++count[l];
        if (n > best_count || (n == best_count && l < best)) {
          best_count = n;
          best = l;
        }
      }
      for (uint32_t k = adj_begin[g]; k < adj_begin[g + 1]; ++k) {
        const uint32_t h = adj[k];
        if (layer[h] + 1 == d) count[lab[h]] = 0;
      }
      lab[g] = best;
      ++out.propagated;
    }
    frontier.swap(next);
  }
  for (size_t f = 0; f < nF; ++f) {
    if (lab[f] < 0) {
      ++out.unreached;
      lab[f] = 0;
    }
    out.identity[f] = ids_of[lab[f]];
  }
  return out;
}

// A surface element the present does not supersede (see the header): a face
// with its corners and centroid x, identity, layer scales, the position error
// of the session that placed it along its line of sight, its prior, and the
// present's nearest point to x (distance +inf beyond the largest position
// error a view can give it).
struct Element {
  std::array<Eigen::Vector3f, 3> corner;
  Eigen::Vector3f x;
  uint32_t identity = 0;
  float half = 0.f, trunc = 0.f;
  float offset = 0.f;  // max(s_f, 0) q_f; 0 for this session's own elements
  uint8_t prior = 0;   // 1: previous surface, 0: own online surface
  Eigen::Vector3f nearest = Eigen::Vector3f::Zero();
  float distance = kInf;
  // The present confirms it: a present surface within one voxel of its layer
  // (the layer cannot tell the two apart).
  bool confirmed() const { return distance <= 2.f * half; }
};

// Whether the present holds a face: it passes through the face's centroid
// within one voxel 2h along the face's normal (a present that ends beside the
// face, at a hole, does not hold it).
bool holds(const TriangleGrid& present, const Element& e) {
  const Eigen::Vector3f normal = (e.corner[1] - e.corner[0]).cross(e.corner[2] - e.corner[0]);
  const float length = normal.norm();
  if (!(length > 0.f)) return false;
  const Eigen::Vector3f n = normal / length;
  const float reach = 2.f * e.half;
  float t;
  uint32_t face;
  return present.firstHit(e.x - n * reach, n, t, face, 2.f * reach);
}

// The views of one element: support, free, remeasured, occluded; and the
// nearest range at which a view reached it (a reading at most one truncation in front).
struct Evidence {
  uint16_t support = 0, free = 0, remeasured = 0, occluded = 0;
  float reach = kInf;
  // The vote p + S - F - max(0, R - O).
  int margin(uint8_t prior) const {
    return static_cast<int>(prior) + static_cast<int>(support) - static_cast<int>(free) -
           std::max(0, static_cast<int>(remeasured) - static_cast<int>(occluded));
  }
};

}  // namespace

SessionRefusion::Result SessionRefusion::apply(DynamicSceneGraph& dsg, const Inputs& in) const {
  Result result;
  StepTimer timer;
  if (!in.frames || in.frames->empty() || !in.camera.valid()) {
    LOG(WARNING) << "[SessionRefusion] no archived frames; skipped.";
    return result;
  }
  const double voxel = decimal(in.scales.object_voxel);
  const double bg_voxel = decimal(in.scales.background_voxel);
  const double bg_trunc = decimal(in.scales.background_truncation);
  if (!(voxel > 0.0) || !(bg_voxel > 0.0) || !(bg_trunc > 0.0)) {
    LOG(ERROR) << "[SessionRefusion] needs absolute object and background scales (> 0); skipped.";
    return result;
  }
  const double trunc = 2.0 * voxel;  // object truncation = 2 voxels
  const float v_f = static_cast<float>(voxel), T_f = static_cast<float>(trunc);
  const float h_obj = 0.5f * v_f;
  const float h_bg = static_cast<float>(0.5 * bg_voxel), T_bg = static_cast<float>(bg_trunc);
  const int threads = std::max(1, config.num_threads);
  const auto& K = in.camera;
  const int W = static_cast<int>(K.width), H = static_cast<int>(K.height);
  const Projector project{K.fx, K.fy, K.cx, K.cy, W, H};
  SessionFrames frames(*in.frames, K, in.state_starts);
  std::stringstream report;
  report << "{\"frames\":" << frames.size() << ",\"voxel\":" << voxel << ",\"truncation\":" << trunc;

  // ------------------------------------------------------------ the final map
  std::vector<Slot> slots;
  std::vector<Eigen::Vector3f> pos;
  std::vector<uint32_t> vslot;
  std::vector<Face3> faces;
  std::vector<uint32_t> fslot;
  auto addSlot = [&](Slot slot, const spark_dsg::Mesh& mesh) {
    slot.begin = static_cast<uint32_t>(pos.size());
    const auto sid = static_cast<uint32_t>(slots.size());
    for (size_t i = 0; i < mesh.numVertices(); ++i) {
      pos.push_back(slot.attrs ? slot.attrs->bounding_box.pointToWorldFrame(mesh.pos(i)) : mesh.pos(i));
      vslot.push_back(sid);
    }
    slot.end = static_cast<uint32_t>(pos.size());
    const size_t n = mesh.numVertices();
    for (const auto& f : mesh.faces) {
      if (f[0] >= n || f[1] >= n || f[2] >= n) continue;
      faces.push_back({slot.begin + static_cast<uint32_t>(f[0]), slot.begin + static_cast<uint32_t>(f[1]),
                       slot.begin + static_cast<uint32_t>(f[2])});
      fslot.push_back(sid);
    }
    slots.push_back(slot);
  };
  if (dsg.hasMesh() && dsg.mesh()) {
    Slot bg;
    bg.background = true;
    bg.mesh = dsg.mesh().get();
    addSlot(bg, *bg.mesh);
  }
  if (dsg.hasLayer(DsgLayers::OBJECTS)) {
    for (const auto& [id, node] : dsg.getLayer(DsgLayers::OBJECTS).nodes()) {
      auto* attrs = dynamic_cast<KhronosObjectAttributes*>(&node->attributes());
      if (!attrs || !hasCurrentObjectMesh(*attrs)) continue;
      Slot slot;
      slot.node = id;
      slot.mesh = &attrs->mesh;
      slot.attrs = attrs;
      slot.physical =
          static_cast<uint32_t>(UpdateKhronosObjectsFunctor::physicalInstanceId(*attrs).value_or(0));
      addSlot(slot, attrs->mesh);
    }
  }
  const size_t num_vertices = pos.size();
  // Own faces: no vertex carried over from the loaded state.
  std::vector<uint8_t> carried(num_vertices, 0);
  if (in.carried) {
    parallelFor(num_vertices, threads, [&](size_t b, size_t e) {
      for (size_t i = b; i < e; ++i) carried[i] = in.carried(pos[i]) ? 1 : 0;
    });
  }
  std::vector<uint8_t> own_face(faces.size(), 0);
  std::vector<uint32_t> face_object(faces.size(), 0);  // physical id of the face's object node
  for (size_t f = 0; f < faces.size(); ++f) {
    own_face[f] = !carried[faces[f][0]] && !carried[faces[f][1]] && !carried[faces[f][2]];
    const Slot& slot = slots[fslot[f]];
    face_object[f] = slot.background ? 0 : slot.physical;
  }
  // Node of every identity with a current node (the largest if several), background slot.
  std::map<uint32_t, uint32_t> slot_of_identity;
  std::set<uint32_t> current_ids;
  int32_t bg_slot = -1;
  for (uint32_t s = 0; s < slots.size(); ++s) {
    if (slots[s].background) bg_slot = static_cast<int32_t>(s);
    if (slots[s].background || slots[s].physical == 0) continue;
    current_ids.insert(slots[s].physical);
    const auto it = slot_of_identity.find(slots[s].physical);
    if (it == slot_of_identity.end() ||
        slots[s].end - slots[s].begin > slots[it->second].end - slots[it->second].begin) {
      slot_of_identity[slots[s].physical] = s;
    }
  }
  const size_t num_carried = std::count(carried.begin(), carried.end(), 1);
  timer.step("gather", "slots=" + std::to_string(slots.size()) + " vertices=" + std::to_string(num_vertices) +
                           " faces=" + std::to_string(faces.size()) + " carried=" + std::to_string(num_carried));
  report << ",\"map\":{\"slots\":" << slots.size() << ",\"vertices\":" << num_vertices
         << ",\"faces\":" << faces.size() << ",\"carried_vertices\":" << num_carried << "}";
  if (!in.dump_dir.empty()) {
    std::filesystem::create_directories(in.dump_dir);
    FrameArchive::save(in.dump_dir + "/archive.kfa", *in.frames, K);
  }

  // ------------------------------------------------ validity: earlier object states
  size_t total_cut = 0, total_limited = 0;
  limitEarlierFreeSpace(frames, K, in.state_starts, current_ids, pos, faces, face_object, v_f, T_f,
                        threads, report, total_cut, total_limited);
  timer.step("validity", "objects=" + std::to_string(in.state_starts.size()) + " cut=" +
                             std::to_string(total_cut) + " limited=" + std::to_string(total_limited));

  // ------------------------------------------------------------ present: TSDF + MC
  auto tsdf = std::make_unique<PresentTsdf>(voxel, trunc, threads);
  uint16_t max_reading_mm = 0;  // the farthest valid reading
  std::vector<Eigen::Vector3f> Vp;
  std::vector<Face3> Fp;
  {
    const PresentTsdf::Camera cam{K.width, K.height, K.fx, K.fy, K.cx, K.cy};
    const std::vector<float> mult = PresentTsdf::rayNorm(cam);
    std::vector<float> depth(frames.numPixels());
    std::vector<float> free_limit;
    size_t pixels = 0;
    frames.forEach([&](size_t i, const std::vector<uint16_t>& range, const std::vector<uint16_t>&) {
      for (const auto d : range) {
        pixels += d != 0;
        max_reading_mm = std::max(max_reading_mm, d);
      }
      PresentTsdf::depthFromRange(cam, range, depth);
      if (frames.limited(i)) {
        frames.freeLimit(i, free_limit);
        tsdf->integrate(cam, frames.pose(i), depth, mult, &free_limit);
      } else {
        tsdf->integrate(cam, frames.pose(i), depth, mult);
      }
      if ((i + 1) % 500 == 0) {
        LOG(INFO) << "[SessionRefusion] tsdf frames=" << (i + 1) << "/" << frames.size()
                  << " units=" << tsdf->numUnits();
      }
    });
    tsdf->extractMesh(Vp, Fp);
    timer.step("present", "pixels=" + std::to_string(pixels) + " units=" + std::to_string(tsdf->numUnits()) +
                              " vertices=" + std::to_string(Vp.size()) + " faces=" + std::to_string(Fp.size()));
    report << ",\"present\":{\"pixels\":" << pixels << ",\"units\":" << tsdf->numUnits()
           << ",\"vertices\":" << Vp.size() << ",\"faces\":" << Fp.size() << "}";
  }
  if (Fp.empty()) {
    report << "}";
    LOG(ERROR) << "[SessionRefusion] empty present surface; the final map is left unchanged.";
    result.report_json = report.str();
    return result;
  }
  std::vector<Eigen::Vector3f> face_normal(Fp.size()), centroid(Fp.size());
  for (size_t f = 0; f < Fp.size(); ++f) {
    const Eigen::Vector3d a = Vp[Fp[f][0]].cast<double>(), b = Vp[Fp[f][1]].cast<double>(),
                          c = Vp[Fp[f][2]].cast<double>();
    face_normal[f] = (b - a).cross(c - a).cast<float>();
    centroid[f] = ((a + b + c) / 3.0).cast<float>();
  }

  // ------------------------------------------------------------ sensor: sigma(q), s
  std::vector<float> sigma;
  {
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
    const size_t nb = config.num_bins;
    const size_t nh = static_cast<size_t>(std::floor(trunc / config.histogram_resolution + 1e-9)) + 1;
    std::vector<std::vector<int64_t>> hist(nb, std::vector<int64_t>(nh, 0));
    std::mutex hist_mutex;
    frames.forEach([&](size_t i, const std::vector<uint16_t>& rng, const std::vector<uint16_t>&) {
      const FrameCam& c = frames.cam(i);
      parallelFor(Vp.size(), threads, [&](size_t b, size_t e) {
        std::vector<int64_t> local(nb * nh, 0);
        bool any = false;
        for (size_t j = b; j < e; ++j) {
          int u, v;
          float q;
          if (!project(Vp[j], c, u, v, q)) continue;
          if (N[j].dot(c.t - Vp[j]) <= 0.f) continue;
          const uint16_t d = rng[static_cast<size_t>(v) * W + u];
          if (!d) continue;
          const double r = std::abs(d * 1e-3 - static_cast<double>(q));
          if (r > trunc) continue;
          const size_t bin = std::min(nb - 1, static_cast<size_t>(q / config.range_bin));
          const size_t cell = std::min(nh - 1, static_cast<size_t>(r / config.histogram_resolution));
          ++local[bin * nh + cell];
          any = true;
        }
        if (!any) return;
        std::lock_guard<std::mutex> lock(hist_mutex);
        for (size_t bin = 0; bin < nb; ++bin)
          for (size_t cell = 0; cell < nh; ++cell) hist[bin][cell] += local[bin * nh + cell];
      }, 16384);
    });
    sigma = sigmaFromHistogram(hist, config.min_bin_samples, config.histogram_resolution);
  }
  auto tauOf = [&](float half, float q) {
    const size_t bin = std::min(config.num_bins - 1, static_cast<size_t>(std::max(0.f, q) / config.range_bin));
    return std::max(half, sigma[bin]);
  };
  const float depth_scale = depthScale(frames, K, project, T_bg, threads);
  const float s_pos = std::max(0.f, depth_scale);
  result.depth_scale = depth_scale;
  {
    std::stringstream ss;
    report << ",\"sensor\":{\"sigma_cm\":[";
    for (size_t b = 0; b < sigma.size(); ++b) {
      ss << (b ? " " : "") << std::round(sigma[b] * 1e4) / 100.0;
      report << (b ? "," : "") << sigma[b] * 100.f;
    }
    report << "],\"depth_scale\":" << depth_scale << "}";
    timer.step("sensor", "sigma_cm=[" + ss.str() + "] s=" + std::to_string(depth_scale));
  }

  // ------------------------------------------------------------ own faces
  // The session's own online faces where the present is undefined (a corner of
  // their centroid cube never integrated) and does not hold them: they fill the
  // present, so they lie at the present's resolution.
  const TriangleGrid present_grid(Vp, Fp, nullptr, 4.f * v_f);
  std::vector<uint32_t> own_faces;
  std::vector<Element> own_elements;
  {
    std::vector<uint32_t> undefined;
    for (uint32_t f = 0; f < faces.size(); ++f) {
      if (!own_face[f]) continue;
      const Eigen::Vector3d c = (pos[faces[f][0]].cast<double>() + pos[faces[f][1]].cast<double>() +
                                 pos[faces[f][2]].cast<double>()) / 3.0;
      Eigen::Vector3i cube;
      for (int k = 0; k < 3; ++k) cube[k] = static_cast<int>(std::floor(c[k] / voxel - 0.5));
      bool defined = true;
      for (int corner = 0; corner < 8 && defined; ++corner) {
        defined = tsdf->integrated(cube + Eigen::Vector3i(corner & 1, (corner >> 1) & 1, corner >> 2));
      }
      if (!defined) undefined.push_back(f);
    }
    tsdf.reset();  // release the volume
    std::vector<Element> candidates(undefined.size());
    std::vector<uint8_t> held(undefined.size(), 0);
    parallelFor(undefined.size(), threads, [&](size_t b, size_t e) {
      for (size_t k = b; k < e; ++k) {
        const Face3& f = faces[undefined[k]];
        Element& el = candidates[k];
        for (int j = 0; j < 3; ++j) el.corner[j] = pos[f[j]];
        el.x = ((pos[f[0]].cast<double>() + pos[f[1]].cast<double>() + pos[f[2]].cast<double>()) / 3.0).cast<float>();
        el.half = h_obj;
        el.trunc = T_f;
        el.offset = 0.f;
        el.prior = 0;
        held[k] = holds(present_grid, el);
      }
    }, 1024);
    for (size_t k = 0; k < undefined.size(); ++k) {
      if (held[k]) continue;  // the same readings' surface at a coarser resolution
      own_faces.push_back(undefined[k]);
      own_elements.push_back(candidates[k]);
    }
  }
  const size_t nP = Fp.size(), nO = own_faces.size();

  // ------------------------------------------------------------ identity of the session's surface
  // The present faces, then the own faces. Along the surface: faces sharing an
  // edge within the present or within the own surface, and across their seam
  // an own face and the present face nearest to it within one voxel.
  SurfaceIdentity surface_identity;
  {
    std::vector<Eigen::Vector3f> x(centroid), n(face_normal);
    x.reserve(nP + nO);
    n.reserve(nP + nO);
    for (const auto& el : own_elements) {
      x.push_back(el.x);
      n.push_back((el.corner[1] - el.corner[0]).cross(el.corner[2] - el.corner[0]));
    }
    // The current object surfaces the reasoning reconstructed, re-measured within one voxel.
    std::vector<uint32_t> reconstructed(nP + nO, 0);
    std::vector<uint32_t> object_faces;
    for (uint32_t f = 0; f < faces.size(); ++f) {
      if (face_object[f] > 0 && slot_of_identity.count(face_object[f])) object_faces.push_back(f);
    }
    if (!object_faces.empty()) {
      const TriangleGrid objects(pos, faces, &object_faces, 4.f * v_f);
      parallelFor(nP + nO, threads, [&](size_t b, size_t e) {
        for (size_t f = b; f < e; ++f) {
          float d;
          Eigen::Vector3f closest;
          uint32_t hit;
          if (objects.closest(x[f], v_f, d, closest, hit)) reconstructed[f] = face_object[hit];
        }
      });
    }
    std::vector<std::pair<uint32_t, uint32_t>> adjacency;
    {
      std::vector<uint32_t> all(nP);
      std::iota(all.begin(), all.end(), 0u);
      sharedEdgePairs(Fp, all, Vp.size(), 0, adjacency);
    }
    sharedEdgePairs(faces, own_faces, pos.size(), static_cast<uint32_t>(nP), adjacency);
    {
      std::vector<int64_t> seam(nO, -1);
      parallelFor(nO, threads, [&](size_t b, size_t e) {
        for (size_t k = b; k < e; ++k) {
          float d;
          Eigen::Vector3f closest;
          uint32_t face;
          if (present_grid.closest(own_elements[k].x, v_f, d, closest, face)) seam[k] = face;
        }
      });
      for (size_t k = 0; k < nO; ++k) {
        if (seam[k] >= 0) adjacency.emplace_back(static_cast<uint32_t>(seam[k]), static_cast<uint32_t>(nP + k));
      }
    }
    surface_identity = identifySurface(frames, K, project, x, n, reconstructed, adjacency,
                                       [&](float q) { return tauOf(h_obj, q); }, T_f,
                                       in.membership_confidence, in.membership_observations, threads);
  }
  const auto& identity = surface_identity.identity;
  {
    std::map<uint32_t, size_t> per_identity, own_per_identity;
    for (size_t f = 0; f < nP; ++f) ++per_identity[identity[f]];
    for (size_t k = 0; k < nO; ++k) ++own_per_identity[identity[nP + k]];
    std::stringstream ss;
    report << ",\"identity\":{\"reconstructed\":" << surface_identity.reconstructed
           << ",\"measured\":" << surface_identity.measured << ",\"propagated\":" << surface_identity.propagated
           << ",\"unreached\":" << surface_identity.unreached << ",\"faces\":{";
    bool first = true;
    for (const auto& [id, n] : per_identity) {
      report << (first ? "" : ",") << "\"" << id << "\":" << n;
      ss << " " << id << ":" << n;
      first = false;
    }
    report << "},\"own_faces\":{";
    ss << " own";
    first = true;
    for (const auto& [id, n] : own_per_identity) {
      report << (first ? "" : ",") << "\"" << id << "\":" << n;
      ss << " " << id << ":" << n;
      first = false;
    }
    report << "}}";
    timer.step("identity", "reconstructed=" + std::to_string(surface_identity.reconstructed) + " measured=" +
                               std::to_string(surface_identity.measured) + " propagated=" +
                               std::to_string(surface_identity.propagated) + " unreached=" +
                               std::to_string(surface_identity.unreached) + " faces=" + ss.str());
  }
  if (!in.dump_dir.empty()) {
    std::vector<int32_t> fl(identity.begin(), identity.begin() + nP);
    writePly(in.dump_dir + "/present.ply", Vp, Fp, &fl);
  }
  // The node of an identity: its current node, else the background.
  auto slotOf = [&](uint32_t id) {
    const auto it = id ? slot_of_identity.find(id) : slot_of_identity.end();
    return it != slot_of_identity.end() ? static_cast<int32_t>(it->second) : bg_slot;
  };

  // ------------------------------------------------------------ elements
  // Previous surface faces whose state is still current (prior 1), and the own
  // faces (prior 0), each in the node of its identity.
  std::vector<Element> elements;
  std::vector<uint32_t> element_face;   // index into previous faces or final-map faces
  std::vector<uint32_t> element_slot;   // target slot
  size_t previous_void = 0, previous_elements = 0;
  if (in.previous && in.previous->consistent()) {
    const auto& S = *in.previous;
    for (uint32_t f = 0; f < S.faces.size(); ++f) {
      const uint32_t id = S.identity[f];
      int32_t target = bg_slot;
      if (id > 0) {
        const auto it = slot_of_identity.find(id);
        if (it == slot_of_identity.end() || in.state_starts.count(id)) {
          ++previous_void;
          continue;
        }
        target = static_cast<int32_t>(it->second);
      }
      if (target < 0) {
        ++previous_void;
        continue;
      }
      Element e;
      for (int k = 0; k < 3; ++k) e.corner[k] = S.vertices[S.faces[f][k]];
      e.x = (e.corner[0] + e.corner[1] + e.corner[2]) / 3.f;
      e.identity = id;
      e.half = id > 0 ? h_obj : h_bg;
      e.trunc = id > 0 ? T_f : T_bg;
      e.offset = std::max(0.f, S.scale[f]) * std::max(0.f, S.range[f]);
      e.prior = 1;
      elements.push_back(e);
      element_face.push_back(f);
      element_slot.push_back(static_cast<uint32_t>(target));
    }
    previous_elements = elements.size();
  }
  const size_t own_begin = elements.size();
  for (size_t k = 0; k < nO; ++k) {
    const int32_t target = slotOf(identity[nP + k]);
    if (target < 0) continue;
    Element e = own_elements[k];
    e.identity = identity[nP + k];
    elements.push_back(e);
    element_face.push_back(own_faces[k]);
    element_slot.push_back(static_cast<uint32_t>(target));
  }
  own_elements.clear();
  // The present's nearest point to every element, within the largest position
  // error a view can give it (a reading ends no farther than the farthest valid
  // reading).
  {
    const float sigma_max = *std::max_element(sigma.begin(), sigma.end());
    const float max_range = 1e-3f * max_reading_mm;
    parallelFor(elements.size(), threads, [&](size_t b, size_t e) {
      for (size_t k = b; k < e; ++k) {
        Element& el = elements[k];
        const float r_max =
            std::max(2.f * el.half, std::max(el.half, sigma_max) + s_pos * max_range + el.offset);
        float d;
        uint32_t face;
        if (present_grid.closest(el.x, r_max, d, el.nearest, face)) el.distance = d;
      }
    }, 1024);
  }
  const size_t own_candidates = elements.size() - own_begin;
  std::vector<Evidence> evidence(elements.size());
  {
    std::vector<float> free_limit;
    frames.forEach([&](size_t i, const std::vector<uint16_t>& rng, const std::vector<uint16_t>& ids) {
      const FrameCam& c = frames.cam(i);
      const bool limited = frames.limited(i);
      if (limited) frames.freeLimit(i, free_limit);
      parallelFor(elements.size(), threads, [&](size_t b, size_t e) {
        for (size_t k = b; k < e; ++k) {
          const Element& el = elements[k];
          int u, v;
          float q;
          if (!project(el.x, c, u, v, q)) continue;
          const float tau = tauOf(el.half, q);
          Evidence& ev = evidence[k];
          const size_t centre = static_cast<size_t>(v) * W + u;
          const uint16_t d0 = rng[centre];
          bool occluded = false;
          if (d0) {
            const float r0 = d0 * 1e-3f - q;
            if (r0 >= -el.trunc && q < ev.reach) ev.reach = q;
            if (r0 < -tau) {
              // Its own line of sight ends in front of it, on a reading at
              // range r. Unless the present confirms it, the view
              // remeasured its surface when the map cannot tell the two apart:
              // the reading lies within the layer's truncation in front of it
              // (its TSDF holds no second surface closer behind along the ray),
              // or the present lies within the sessions' position error eps(r)
              // of it on the view's side (the same surface placed apart by the
              // depth scales), or the reading is on its own object (a solid
              // holds none of its own surface behind its surface). Otherwise the
              // view is occluded.
              if (!el.confirmed()) {
                const float r = d0 * 1e-3f;
                const float eps = tauOf(el.half, r) + s_pos * r + el.offset;
                if (-r0 <= el.trunc ||
                    (el.distance <= eps && (el.nearest - el.x).dot(c.t - el.x) > 0.f) ||
                    (el.identity > 0 && ids[centre] == el.identity)) {
                  ++ev.remeasured;
                  continue;
                }
              }
              occluded = true;
            }
          }
          // The footprint: every pixel whose ray passes within tau of x.
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
              const uint16_t d = rng[pix];
              if (!d) {
                all_valid = false;
                continue;
              }
              const float r = d * 1e-3f - q;
              if (std::abs(r) <= tau) {
                const Eigen::Vector3f ray((x - K.cx) / K.fx, (y - K.cy) / K.fy, 1.f);
                const Eigen::Vector3f point = c.t + c.R * (ray.normalized() * (d * 1e-3f));
                if ((point - el.x).squaredNorm() <= tau * tau) {
                  hit = true;
                  break;
                }
              }
              if (r <= tau) all_beyond = false;
              else if (limited && q >= free_limit[pix] - T_f) all_valid = false;
            }
          }
          if (hit) ++ev.support;
          else if (all_valid && all_beyond) ++ev.free;
          else if (occluded) ++ev.occluded;
        }
      }, 4096);
    });
  }
  std::vector<uint8_t> keep(elements.size(), 0);
  size_t kept_previous = 0, kept_own = 0;
  for (size_t k = 0; k < elements.size(); ++k) {
    keep[k] = evidence[k].margin(elements[k].prior) > 0;
    (elements[k].prior ? kept_previous : kept_own) += keep[k];
  }
  report << ",\"elements\":{\"previous_faces\":" << (in.previous ? in.previous->numFaces() : 0)
         << ",\"previous_void\":" << previous_void << ",\"previous\":" << previous_elements
         << ",\"previous_kept\":" << kept_previous << ",\"own\":" << own_candidates
         << ",\"own_kept\":" << kept_own << "}";
  timer.step("elements", "previous=" + std::to_string(previous_elements) + " void=" +
                             std::to_string(previous_void) + " kept=" + std::to_string(kept_previous) +
                             " own=" + std::to_string(own_candidates) + " kept=" + std::to_string(kept_own));
  if (!in.dump_dir.empty()) {
    std::ofstream out(in.dump_dir + "/elements.bin", std::ios::binary);
    const uint64_t n = elements.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (size_t k = 0; k < elements.size(); ++k) {
      const float rec[5] = {elements[k].x.x(), elements[k].x.y(), elements[k].x.z(), evidence[k].reach,
                            elements[k].offset};
      const uint32_t info[3] = {elements[k].identity, element_face[k],
                                static_cast<uint32_t>(elements[k].prior | (keep[k] << 1))};
      const uint16_t counts[4] = {evidence[k].support, evidence[k].free, evidence[k].remeasured,
                                  evidence[k].occluded};
      out.write(reinterpret_cast<const char*>(rec), sizeof(rec));
      out.write(reinterpret_cast<const char*>(info), sizeof(info));
      out.write(reinterpret_cast<const char*>(counts), sizeof(counts));
    }
  }

  // ------------------------------------------------------------ compose
  // Per slot: kept own faces of the slot's own mesh (input vertices in their
  // original order), then kept own faces of other meshes, present faces and
  // kept previous faces (vertices in first-use order); every face with its
  // record (range, scale).
  struct NewFace {
    uint32_t a, b, c;  // vertex indices into the source (own: final map; present: Vp; previous: S)
    uint8_t source;    // 0 own (this mesh), 1 present, 2 previous, 3 own (another mesh)
    float range, scale;
  };
  std::vector<std::vector<NewFace>> slot_faces(slots.size());
  for (size_t k = 0; k < elements.size(); ++k) {
    if (!keep[k] || elements[k].prior) continue;
    const auto& f = faces[element_face[k]];
    const float q = std::isfinite(evidence[k].reach) ? evidence[k].reach : 0.f;
    const uint8_t source = element_slot[k] == fslot[element_face[k]] ? 0 : 3;
    slot_faces[element_slot[k]].push_back({f[0], f[1], f[2], source, q, depth_scale});
  }
  for (uint32_t f = 0; f < nP; ++f) {
    const int32_t target = slotOf(identity[f]);
    if (target < 0) continue;
    const float q = std::isfinite(surface_identity.reach[f]) ? surface_identity.reach[f] : 0.f;
    slot_faces[target].push_back({Fp[f][0], Fp[f][1], Fp[f][2], 1, q, depth_scale});
  }
  for (size_t k = 0; k < elements.size(); ++k) {
    if (!keep[k] || !elements[k].prior) continue;
    const uint32_t f = element_face[k];
    const auto& sf = in.previous->faces[f];
    slot_faces[element_slot[k]].push_back({sf[0], sf[1], sf[2], 2, in.previous->range[f], in.previous->scale[f]});
  }
  auto sourcePoint = [&](uint8_t source, uint32_t index) -> const Eigen::Vector3f& {
    return source == 1 ? Vp[index] : source == 3 ? pos[index] : in.previous->vertices[index];
  };
  struct NewMesh {
    uint32_t slot = 0;
    spark_dsg::Mesh::Positions points;
    spark_dsg::Mesh::Colors colors;
    spark_dsg::Mesh::Timestamps stamps, first_seen_stamps;
    spark_dsg::Mesh::Labels labels;
    spark_dsg::Mesh::Faces faces;
    std::vector<Eigen::Vector3f> world;
  };
  std::vector<NewMesh> rebuilt;
  std::stringstream slot_report;
  size_t empty_objects = 0;
  const TimeStamp stamp_cap = in.final_stamp;
  for (uint32_t s = 0; s < slots.size(); ++s) {
    const Slot& slot = slots[s];
    const auto& mesh = *slot.mesh;
    const auto& list = slot_faces[s];
    const size_t old_n = mesh.numVertices();
    // Kept own vertices in their original order.
    std::vector<int64_t> own_new(old_n, -1);
    for (const auto& f : list) {
      if (f.source != 0) continue;
      own_new[f.a - slot.begin] = own_new[f.b - slot.begin] = own_new[f.c - slot.begin] = 0;
    }
    std::vector<uint32_t> own_old;
    for (size_t i = 0; i < old_n; ++i) {
      if (own_new[i] < 0) continue;
      own_new[i] = static_cast<int64_t>(own_old.size());
      own_old.push_back(static_cast<uint32_t>(i));
    }
    // Present and previous vertices in first-use order.
    std::unordered_map<uint64_t, uint32_t> added_new;
    std::vector<std::pair<uint8_t, uint32_t>> added;
    auto addedIndex = [&](uint8_t source, uint32_t index) {
      const uint64_t key = (static_cast<uint64_t>(source) << 32) | index;
      const auto [it, inserted] =
          added_new.emplace(key, static_cast<uint32_t>(own_old.size() + added.size()));
      if (inserted) added.emplace_back(source, index);
      return it->second;
    };
    NewMesh out;
    out.slot = s;
    out.faces.reserve(list.size());
    for (const auto& f : list) {
      if (f.source == 0) {
        out.faces.push_back({static_cast<size_t>(own_new[f.a - slot.begin]),
                             static_cast<size_t>(own_new[f.b - slot.begin]),
                             static_cast<size_t>(own_new[f.c - slot.begin])});
      }
    }
    for (const auto& f : list) {
      if (f.source == 0) continue;
      out.faces.push_back({addedIndex(f.source, f.a), addedIndex(f.source, f.b), addedIndex(f.source, f.c)});
    }
    const size_t new_n = own_old.size() + added.size();
    // Attributes of added vertices: the nearest vertex of this mesh before the update.
    const bool need_nearest = (mesh.colors.size() == old_n || mesh.stamps.size() == old_n ||
                               mesh.first_seen_stamps.size() == old_n || mesh.labels.size() == old_n) &&
                              old_n > 0 && !added.empty();
    std::vector<size_t> nearest(added.size(), 0);
    if (need_nearest) {
      const std::vector<Eigen::Vector3f> old_world(pos.begin() + slot.begin, pos.begin() + slot.end);
      const hydra::PointNeighborSearch search(old_world);
      parallelFor(added.size(), threads, [&](size_t b, size_t e) {
        for (size_t k = b; k < e; ++k) {
          float d_sq = 0.f;
          size_t idx = 0;
          if (search.search(sourcePoint(added[k].first, added[k].second), d_sq, idx)) nearest[k] = idx;
        }
      });
    }
    out.points.resize(new_n);
    out.world.resize(new_n);
    for (size_t k = 0; k < own_old.size(); ++k) {
      out.points[k] = mesh.points[own_old[k]];
      out.world[k] = pos[slot.begin + own_old[k]];
    }
    for (size_t k = 0; k < added.size(); ++k) {
      const Eigen::Vector3f& p = sourcePoint(added[k].first, added[k].second);
      out.points[own_old.size() + k] = slot.attrs ? slot.attrs->bounding_box.pointToBoxFrame(p) : p;
      out.world[own_old.size() + k] = p;
    }
    auto rebuild = [&](const auto& values, auto& target, bool clamp_stamp) {
      using T = typename std::decay_t<decltype(values)>::value_type;
      if (values.size() != old_n) {
        target.assign(values.begin(), values.begin() + std::min(values.size(), new_n));
        return;
      }
      target.resize(new_n);
      for (size_t k = 0; k < own_old.size(); ++k) target[k] = values[own_old[k]];
      for (size_t k = 0; k < added.size(); ++k) target[own_old.size() + k] = values[nearest[k]];
      if constexpr (std::is_integral_v<T>) {
        if (clamp_stamp && stamp_cap > 0) {
          for (size_t k = own_old.size(); k < new_n; ++k) target[k] = std::min<T>(target[k], static_cast<T>(stamp_cap));
        }
      }
    };
    rebuild(mesh.colors, out.colors, false);
    rebuild(mesh.stamps, out.stamps, true);
    rebuild(mesh.first_seen_stamps, out.first_seen_stamps, true);
    rebuild(mesh.labels, out.labels, false);
    size_t n_own = 0, n_present = 0, n_previous = 0;
    for (const auto& f : list) (f.source == 1 ? n_present : f.source == 2 ? n_previous : n_own) += 1;
    if (!slot.background && list.empty()) ++empty_objects;
    slot_report << (slot_report.tellp() > 0 ? "," : "") << "{\"node\":" << slot.node
                << ",\"physical\":" << slot.physical << ",\"background\":" << slot.background
                << ",\"old_vertices\":" << old_n << ",\"own_faces\":" << n_own
                << ",\"present_faces\":" << n_present << ",\"previous_faces\":" << n_previous
                << ",\"vertices\":" << new_n << "}";
    rebuilt.push_back(std::move(out));
  }
  // The final map's surface with identities and records.
  SessionSurface& surface = result.surface;
  for (const auto& out : rebuilt) {
    const Slot& slot = slots[out.slot];
    const auto& list = slot_faces[out.slot];
    const auto base = static_cast<uint32_t>(surface.vertices.size());
    surface.vertices.insert(surface.vertices.end(), out.world.begin(), out.world.end());
    const uint32_t id = slot.background ? 0 : slot.physical;
    // out.faces lists own faces first, then the others, in the order of `list`.
    size_t k = 0;
    auto record = [&](const NewFace& f) {
      const auto& nf = out.faces[k++];
      surface.faces.push_back({base + static_cast<uint32_t>(nf[0]), base + static_cast<uint32_t>(nf[1]),
                               base + static_cast<uint32_t>(nf[2])});
      surface.identity.push_back(id);
      surface.range.push_back(f.range);
      surface.scale.push_back(f.scale);
    };
    for (const auto& f : list)
      if (f.source == 0) record(f);
    for (const auto& f : list)
      if (f.source != 0) record(f);
  }
  for (auto& out : rebuilt) {
    auto& mesh = *slots[out.slot].mesh;
    mesh.points.swap(out.points);
    mesh.colors.swap(out.colors);
    mesh.stamps.swap(out.stamps);
    mesh.first_seen_stamps.swap(out.first_seen_stamps);
    mesh.labels.swap(out.labels);
    mesh.faces.swap(out.faces);
  }
  timer.step("compose", "surface_faces=" + std::to_string(surface.numFaces()) + " empty_objects=" +
                            std::to_string(empty_objects));
  report << ",\"compose\":{\"surface_faces\":" << surface.numFaces() << ",\"empty_objects\":" << empty_objects
         << ",\"slots\":[" << slot_report.str() << "]}";
  const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - timer.start).count();
  report << ",\"timings_s\":{" << timer.timings.str() << ",\"total\":" << total << "},\"rss_mb_end\":" << rssMb()
         << "}";
  result.report_json = report.str();
  std::stringstream summary;
  summary << "frames=" << frames.size() << " present_faces=" << Fp.size() << " cut=" << total_cut
          << " limited=" << total_limited << " previous=" << previous_elements << "/void " << previous_void
          << " kept=" << kept_previous << " own=" << own_candidates << " kept=" << kept_own
          << " surface_faces=" << surface.numFaces() << " total_s=" << total;
  result.summary = summary.str();
  result.applied = true;
  if (!in.dump_dir.empty()) std::ofstream(in.dump_dir + "/refusion_report.json") << result.report_json;
  return result;
}

}  // namespace khronos
