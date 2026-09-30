#include "session_core/surface/session_refusion.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <glog/logging.h>
#include <hydra/utils/nearest_neighbor_utilities.h>

#include "session_core/surface/present_tsdf.h"
#include "session_core/surface/frame_endpoint_index.h"
#include "session_core/surface/range_calibration.h"
#include "session_core/surface/triangle_grid.h"
#include "khronos/backend/update_khronos_objects_functor.h"
#include "khronos/utils/khronos_attribute_utils.h"

namespace khronos {

namespace {

using Face3 = std::array<uint32_t, 3>;
using GeometryKey = std::array<std::array<float, 3>, 3>;

// README (14a): the same oriented world triangle is one hypothesis and one
// stored face. Candidate reduction and final assembly share this exact key.
template <typename Vertices>
GeometryKey geometryKey(const Vertices& vertices, const Face3& face) {
  GeometryKey key;
  for (size_t k = 0; k < 3; ++k) {
    const auto& point = vertices.at(face[k]);
    if (!point.allFinite()) throw std::invalid_argument("Non-finite output surface");
    key[k] = {{point.x(), point.y(), point.z()}};
  }
  GeometryKey canonical = key;
  for (size_t shift = 1; shift < 3; ++shift) {
    std::rotate(key.begin(), key.begin() + 1, key.end());
    canonical = std::min(canonical, key);
  }
  return canonical;
}

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

// README (6), (9a), (12): a centre ray exists for positive camera Z.
// The sphere footprint is evaluated separately, including near/behind centres.
struct Projector {
  float fx, fy, cx, cy;
  int W, H;
  inline bool operator()(const Eigen::Vector3f& p, const FrameCam& c, int& u, int& v,
                         float& q) const {
    const float dx = p.x() - c.t.x(), dy = p.y() - c.t.y(), dz = p.z() - c.t.z();
    const float x = c.R(0, 0) * dx + c.R(1, 0) * dy + c.R(2, 0) * dz;
    const float y = c.R(0, 1) * dx + c.R(1, 1) * dy + c.R(2, 1) * dz;
    const float z = c.R(0, 2) * dx + c.R(1, 2) * dy + c.R(2, 2) * dz;
    if (!std::isfinite(z) || z <= 0.f) return false;
    const double pixel_x = std::floor(static_cast<double>(x) / z * fx + cx + 0.5);
    const double pixel_y = std::floor(static_cast<double>(y) / z * fy + cy + 0.5);
    if (!std::isfinite(pixel_x) || !std::isfinite(pixel_y) ||
        pixel_x < 0 || pixel_y < 0 || pixel_x >= W || pixel_y >= H) return false;
    u = static_cast<int>(pixel_x);
    v = static_cast<int>(pixel_y);
    q = std::hypot(x, y, z);
    return std::isfinite(q);
  }
};

/**
 * The session's frames as the present sees them: the archived range with the
 * pixels of step 1 (object L before t_L) and step 1b (`stale`) removed.
 */
class SessionFrames {
 public:
  SessionFrames(const std::vector<FrameArchive::Frame>& frames,
                const FrameArchive::Camera& camera,
                const std::map<size_t, std::optional<TimeStamp>>& state_starts)
      : frames_(frames),
        num_pixels_(static_cast<size_t>(camera.width) * camera.height),
        start_of_(kNumIds, std::numeric_limits<TimeStamp>::max()),
        stale(frames.size()),
        stale_hit(frames.size()) {
    start_of_[0] = 0;  // Background has no object-state time restriction.
    cams_.resize(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
      cams_[i].R = frames[i].world_T_sensor.linear().cast<float>();
      cams_[i].t = frames[i].world_T_sensor.translation().cast<float>();
    }
    for (const auto& [id, t_L] : state_starts) {
      if (id > 0 && id < kNumIds) start_of_[id] = t_L.value_or(std::numeric_limits<TimeStamp>::max());
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
      throw std::runtime_error("Session frame decode failed at " +
                               std::to_string(frames_[i].stamp));
    }
  }

  /** Step 1 on a decoded frame; counts the removed pixels per id in `removed` (if given). */
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

  /** Frame i as the present sees it (steps 1 and 1b applied). */
  void load(size_t i, std::vector<uint16_t>& range, std::vector<uint16_t>& ids) const {
    decode(i, range, ids);
    cut(i, range, ids);
  }

  /** Free-space limit per pixel of frame i (+inf, or the first hit of a stale pixel). */
  void freeLimit(size_t i, std::vector<float>& limit) const {
    limit.assign(num_pixels_, kInf);
    for (size_t k = 0; k < stale[i].size(); ++k) limit[stale[i][k]] = stale_hit[i][k];
  }

  /** fn(i, range, ids) for every frame in order; the next frame is decoded meanwhile. */
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
  std::vector<TimeStamp> start_of_;  // physical id -> t_L (0: not cut)

 public:
  std::vector<std::vector<uint32_t>> stale;  // step-1b pixels per frame
  std::vector<std::vector<float>> stale_hit;  // their first hit on the object [m]
};

// One mesh of the final map: the background or one current object mesh.
struct Slot {
  bool background = false;
  NodeId node = 0;
  spark_dsg::Mesh* mesh = nullptr;
  KhronosObjectAttributes* attrs = nullptr;
  uint32_t begin = 0, end = 0;  // global vertex range
  size_t physical = 0;          // node physical id (0: background / none)
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

// Freeze directed correspondences on the raw archived frames. This intentionally
// uses decode(), not the state-authorized load()/forEach() domain of the residual
// histogram. Sampling and pair order follow the fixed baseline protocol; the
// pure RangeCalibration solver owns the objective and ordered grid search.
std::vector<RangePair> collectScalePairs(const SessionFrames& frames,
                                         const FrameArchive::Camera& K,
                                         const Projector& project,
                                         float association,
                                         int threads) {
  struct View {
    FrameCam cam;
    std::vector<uint16_t> range;
  };
  std::vector<View> views;
  const size_t step = std::max<size_t>(1, frames.size() / ScaleProtocol::kMaxFrames);
  std::vector<uint16_t> ids;
  for (size_t i = 0; i < frames.size() && views.size() < ScaleProtocol::kMaxFrames; i += step) {
    View view;
    view.cam = frames.cam(i);
    frames.decode(i, view.range, ids);
    views.push_back(std::move(view));
  }
  if (views.size() < 2) return {};
  std::vector<std::vector<RangePair>> per_view(views.size());
  const int W = static_cast<int>(K.width), H = static_cast<int>(K.height);
  parallelFor(views.size(), threads, [&](size_t b, size_t e) {
    for (size_t g = b; g < e; ++g) {
      const View& a = views[g];
      for (int v = 0; v < H; v += ScaleProtocol::kPixelStride) {
        for (int u = 0; u < W; u += ScaleProtocol::kPixelStride) {
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
  std::vector<RangePair> samples;
  for (auto& list : per_view) samples.insert(samples.end(), list.begin(), list.end());
  return samples;
}

}  // namespace

SessionRefusion::Result SessionRefusion::apply(DynamicSceneGraph& dsg, const Inputs& in) const {
  Result result;
  StepTimer timer;
  if (!in.frames || in.frames->empty() || !in.camera.valid()) {
    throw std::invalid_argument("Session surface estimator needs archived frames and a valid camera");
  }
  const double voxel = decimal(in.scales.object_voxel);
  if (!std::isfinite(voxel) || voxel <= 0.0 ||
      !std::isfinite(in.scales.background_voxel) || in.scales.background_voxel <= 0.f ||
      !std::isfinite(in.scales.background_truncation) || in.scales.background_truncation <= 0.f ||
      config.num_bins == 0 || !std::isfinite(config.range_bin) || config.range_bin <= 0.0 ||
      !std::isfinite(config.histogram_resolution) || config.histogram_resolution <= 0.0) {
    throw std::invalid_argument("Invalid surface resolution or noise calibration configuration");
  }
  const double trunc = 2.0 * voxel;  // object truncation = 2 voxels
  const float v_f = static_cast<float>(voxel), T_f = static_cast<float>(trunc);
  const float h_obj = 0.5f * v_f;
  const int threads = std::max(1, config.num_threads);
  const auto& K = in.camera;
  const int W = static_cast<int>(K.width), H = static_cast<int>(K.height);
  const Projector project{K.fx, K.fy, K.cx, K.cy, W, H};
  const size_t num_pixels = static_cast<size_t>(K.width) * K.height;
  std::stringstream report;
  report << "{\"frames\":" << in.frames->size() << ",\"voxel\":" << voxel << ",\"truncation\":" << trunc;

  // ------------------------------------------------------------------ final map
  std::vector<Slot> slots;
  std::vector<Eigen::Vector3f> pos;
  std::vector<Face3> faces;
  std::vector<uint32_t> fslot;
  auto addSlot = [&](Slot slot, const spark_dsg::Mesh& mesh) {
    const uint32_t sid = static_cast<uint32_t>(slots.size());
    slot.begin = static_cast<uint32_t>(pos.size());
    for (size_t i = 0; i < mesh.numVertices(); ++i) {
      pos.push_back(slot.attrs ? slot.attrs->bounding_box.pointToWorldFrame(mesh.pos(i))
                               : mesh.pos(i));
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
  // An empty background is a valid output carrier, committed with the other slots.
  auto background = dsg.hasMesh() && dsg.mesh()
      ? dsg.mesh() : std::make_shared<spark_dsg::Mesh>(true, true, true, true);
  {
    Slot bg;
    bg.background = true;
    bg.mesh = background.get();
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
      slot.physical = UpdateKhronosObjectsFunctor::physicalInstanceId(*attrs).value_or(0);
      addSlot(slot, attrs->mesh);
    }
  }
  const size_t num_vertices = pos.size();
  // Physical id -> the node slot its present surface goes to (the largest if
  // several nodes carry it); background slot.
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

  // README (8a): explicit registry domains take precedence. Conditional replay
  // can establish a current domain from its supplied final object geometry;
  // unestablished physical IDs retain an empty domain.
  auto state_domains = in.state_starts;
  for (const auto& [id, slot] : slot_of_label) {
    (void)slot;
    state_domains.try_emplace(id, TimeStamp{0});
  }
  SessionFrames frames(*in.frames, K, state_domains);

  timer.step("gather", "slots=" + std::to_string(slots.size()) + " vertices=" +
                           std::to_string(num_vertices) + " faces=" +
                           std::to_string(faces.size()));
  report << ",\"map\":{\"slots\":" << slots.size() << ",\"vertices\":" << num_vertices
         << ",\"faces\":" << faces.size() << "}";

  if (!in.dump_dir.empty()) {
    std::filesystem::create_directories(in.dump_dir);
    FrameArchive::save(in.dump_dir + "/archive.kfa", *in.frames, K);
  }

  // ------------------------------- measured current-state surface of every object
  // Voxels (object resolution) holding an endpoint of the object's own pixels
  // in its current state (frames from t_L on; every frame when the state began
  // before this session): where this session measured the object as it is now.
  std::map<size_t, std::unordered_set<uint64_t>> measured_of;
  auto voxelKey = [&](const Eigen::Vector3f& p) -> uint64_t {
    constexpr int64_t kOff = int64_t(1) << 20;
    const int64_t x = static_cast<int64_t>(std::floor(p.x() / v_f)) + kOff;
    const int64_t y = static_cast<int64_t>(std::floor(p.y() / v_f)) + kOff;
    const int64_t z = static_cast<int64_t>(std::floor(p.z() / v_f)) + kOff;
    return (static_cast<uint64_t>(x) << 42) | (static_cast<uint64_t>(y) << 21) | static_cast<uint64_t>(z);
  };
  auto keyOf = [](int64_t x, int64_t y, int64_t z) -> uint64_t {
    constexpr int64_t kOff = int64_t(1) << 20;
    return (static_cast<uint64_t>(x + kOff) << 42) | (static_cast<uint64_t>(y + kOff) << 21) |
           static_cast<uint64_t>(z + kOff);
  };
  {
    std::vector<TimeStamp> from(kNumIds, std::numeric_limits<TimeStamp>::max());
    for (const auto& [label, slot] : slot_of_label) {
      if (label >= kNumIds) continue;
      const auto it = in.state_starts.find(label);
      from[label] = it == in.state_starts.end() ? 0 : it->second.value_or(std::numeric_limits<TimeStamp>::max());
    }
    std::vector<Eigen::Vector3f> dirs(num_pixels);
    for (int v = 0; v < H; ++v)
      for (int u = 0; u < W; ++u)
        dirs[static_cast<size_t>(v) * W + u] =
            Eigen::Vector3f((u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.f).normalized();
    std::mutex merge;
    parallelFor(frames.size(), threads, [&](size_t b, size_t e) {
      std::vector<uint16_t> range, ids;
      std::map<size_t, std::unordered_set<uint64_t>> local;
      for (size_t i = b; i < e; ++i) {
        frames.decode(i, range, ids);
        const FrameCam& c = frames.cam(i);
        const TimeStamp t = frames.stamp(i);
        for (size_t p = 0; p < num_pixels; ++p) {
          if (!range[p] || t < from[ids[p]]) continue;
          local[ids[p]].insert(voxelKey(c.t + c.R * (dirs[p] * (range[p] * 1e-3f))));
        }
      }
      std::lock_guard<std::mutex> lock(merge);
      for (auto& [label, keys] : local) measured_of[label].insert(keys.begin(), keys.end());
    }, 64);
    size_t total = 0;
    for (const auto& [label, keys] : measured_of) total += keys.size();
    timer.step("measured", "objects=" + std::to_string(measured_of.size()) + " voxels=" + std::to_string(total));
  }

  // ------------------------------------------------ step 1 / 1b: current-state starts
  struct Start {
    size_t label = 0;
    TimeStamp t_L = 0;
    std::vector<uint32_t> mesh_faces;
    size_t measured_voxels = 0;
    const std::unordered_set<uint64_t>* measured = nullptr;
    Eigen::AlignedBox3f measured_box;
    // Dense copy of `measured` over its box (same content, faster lookups).
    Eigen::Matrix<int64_t, 3, 1> dense_lo = Eigen::Matrix<int64_t, 3, 1>::Zero(), dense_dim = dense_lo;
    std::vector<uint8_t> dense;
    bool isMeasured(int64_t x, int64_t y, int64_t z, uint64_t key) const {
      if (dense.empty()) return measured->count(key) > 0;
      x -= dense_lo.x();
      y -= dense_lo.y();
      z -= dense_lo.z();
      if (x < 0 || y < 0 || z < 0 || x >= dense_dim.x() || y >= dense_dim.y() || z >= dense_dim.z()) return false;
      return dense[(x * dense_dim.y() + y) * dense_dim.z() + z] != 0;
    }
    std::unique_ptr<TriangleGrid> grid;
    std::vector<Eigen::Vector3f> corners;
    size_t frames_before = 0;
    std::atomic<size_t> cut{0}, stale{0};
  };
  std::vector<Start> starts(in.state_starts.size());
  size_t total_cut = 0, total_stale = 0;
  {
    std::vector<int32_t> start_index(kNumIds, -1);
    TimeStamp latest = 0;
    size_t index = 0;
    for (const auto& [label, t_L] : in.state_starts) {
      Start& s = starts[index];
      s.label = label;
      s.t_L = t_L.value_or(std::numeric_limits<TimeStamp>::max());
      if (label < kNumIds) start_index[label] = static_cast<int32_t>(index);
      latest = std::max(latest, s.t_L);
      for (uint32_t f = 0; f < faces.size(); ++f) {
        const Slot& slot = slots[fslot[f]];
        if (!slot.background && slot.physical == label) s.mesh_faces.push_back(f);
      }
      ++index;
    }
    for (auto& s : starts) {
      {
        const auto it = measured_of.find(s.label);
        if (it != measured_of.end() && !it->second.empty()) {
          s.measured = &it->second;
          s.measured_voxels = it->second.size();
          for (const auto key : it->second) {
            constexpr int64_t kOff = int64_t(1) << 20;
            const Eigen::Vector3f lo(static_cast<float>(static_cast<int64_t>(key >> 42) - kOff) * v_f,
                                     static_cast<float>(static_cast<int64_t>((key >> 21) & 0x1FFFFF) - kOff) * v_f,
                                     static_cast<float>(static_cast<int64_t>(key & 0x1FFFFF) - kOff) * v_f);
            s.measured_box.extend(lo);
            s.measured_box.extend(lo + Eigen::Vector3f::Constant(v_f));
          }
          Eigen::Matrix<int64_t, 3, 1> hi;
          for (int k = 0; k < 3; ++k) {
            s.dense_lo[k] = static_cast<int64_t>(std::floor(s.measured_box.min()[k] / v_f + 0.5f));
            hi[k] = static_cast<int64_t>(std::floor(s.measured_box.max()[k] / v_f + 0.5f));
            s.dense_dim[k] = hi[k] - s.dense_lo[k] + 1;
          }
          const int64_t cells = s.dense_dim.prod();
          if (cells > 0 && cells <= (int64_t(1) << 28)) {  // otherwise the hash set is used
            constexpr int64_t kOff = int64_t(1) << 20;
            s.dense.assign(static_cast<size_t>(cells), 0);
            for (const auto key : it->second) {
              const int64_t x = static_cast<int64_t>(key >> 42) - kOff - s.dense_lo.x();
              const int64_t y = static_cast<int64_t>((key >> 21) & 0x1FFFFF) - kOff - s.dense_lo.y();
              const int64_t z = static_cast<int64_t>(key & 0x1FFFFF) - kOff - s.dense_lo.z();
              s.dense[(x * s.dense_dim.y() + y) * s.dense_dim.z() + z] = 1;
            }
          }
        }
      }
      if (s.mesh_faces.empty() && !s.measured) continue;
      if (!s.mesh_faces.empty()) {
        s.grid = std::make_unique<TriangleGrid>(pos, faces, &s.mesh_faces, 4.f * v_f);
      }
      Eigen::AlignedBox3f box = s.grid ? s.grid->bounds() : s.measured_box;
      if (s.measured) box.extend(s.measured_box);
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
    for (auto& s : starts) {
      for (const size_t i : before) s.frames_before += frames.stamp(i) < s.t_L;
    }
    parallelFor(before.size(), threads, [&](size_t b, size_t e) {
      std::vector<uint16_t> range, ids;
      std::unordered_map<uint16_t, size_t> removed;
      for (size_t j = b; j < e; ++j) {
        const size_t i = before[j];
        const FrameCam& c = frames.cam(i);
        frames.decode(i, range, ids);
        // Step 1: the object's own pixels before its current state began.
        removed.clear();
        frames.cut(i, range, ids, &removed);
        for (const auto& [id, n] : removed) {
          // Unestablished IDs have an empty input domain but no registry row.
          const auto state = start_index[id];
          if (state >= 0) starts[state].cut += n;
        }
        // Step 1b: free space seen through the object's current surface before t_L.
        auto& stale = frames.stale[i];
        for (auto& s : starts) {
          if ((!s.grid && !s.measured) || frames.stamp(i) >= s.t_L) continue;
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
              uint16_t& d = range[p];
              if (!d) continue;
              const Eigen::Vector3f ray =
                  Eigen::Vector3f((u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.f).normalized();
              float t_hit = 0.f;
              uint32_t hit_face = 0;
              const float t_end = d * 1e-3f - T_f;
              const Eigen::Vector3f dir = c.R * ray;
              bool hit = s.grid && s.grid->firstHit(c.t, dir, t_hit, hit_face) && t_hit < t_end;
              if (!hit && s.measured && t_end > 0.f) {
                // The ray's free part [0, R - T] through the measured voxels (3D DDA
                // over the part inside their bounding box).
                float t0 = 0.f, t1 = t_end;
                for (int k = 0; k < 3 && t0 <= t1; ++k) {
                  if (std::abs(dir[k]) < 1e-9f) {
                    if (c.t[k] < s.measured_box.min()[k] || c.t[k] > s.measured_box.max()[k]) t0 = t1 + 1.f;
                    continue;
                  }
                  float ta = (s.measured_box.min()[k] - c.t[k]) / dir[k];
                  float tb = (s.measured_box.max()[k] - c.t[k]) / dir[k];
                  if (ta > tb) std::swap(ta, tb);
                  t0 = std::max(t0, ta);
                  t1 = std::min(t1, tb);
                }
                if (t0 <= t1) {
                  const Eigen::Vector3f p0 = c.t + dir * t0;
                  int64_t x = static_cast<int64_t>(std::floor(p0.x() / v_f));
                  int64_t y = static_cast<int64_t>(std::floor(p0.y() / v_f));
                  int64_t z = static_cast<int64_t>(std::floor(p0.z() / v_f));
                  const int sx = dir.x() > 0 ? 1 : -1, sy = dir.y() > 0 ? 1 : -1, sz = dir.z() > 0 ? 1 : -1;
                  auto next = [&](int64_t cell, int step, float o, float dk) {
                    if (std::abs(dk) < 1e-9f) return kInf;
                    const float boundary = static_cast<float>(cell + (step > 0 ? 1 : 0)) * v_f;
                    return (boundary - o) / dk;
                  };
                  float tx = next(x, sx, c.t.x(), dir.x()), ty = next(y, sy, c.t.y(), dir.y()),
                        tz = next(z, sz, c.t.z(), dir.z());
                  const float dx = std::abs(dir.x()) < 1e-9f ? kInf : v_f / std::abs(dir.x());
                  const float dy = std::abs(dir.y()) < 1e-9f ? kInf : v_f / std::abs(dir.y());
                  const float dz = std::abs(dir.z()) < 1e-9f ? kInf : v_f / std::abs(dir.z());
                  float t = t0;
                  while (t <= t1) {
                    if (s.isMeasured(x, y, z, keyOf(x, y, z))) {
                      if (!hit || t < t_hit) t_hit = t;
                      hit = true;
                      break;
                    }
                    if (tx <= ty && tx <= tz) { t = tx; tx += dx; x += sx; }
                    else if (ty <= tz) { t = ty; ty += dy; y += sy; }
                    else { t = tz; tz += dz; z += sz; }
                  }
                }
              }
              if (hit) {
                stale.push_back(static_cast<uint32_t>(p));
                frames.stale_hit[i].push_back(t_hit);
                ++local;
              }
            }
          }
          s.stale += local;
        }
        if (!stale.empty()) {
          // one entry per pixel, the nearest hit
          std::vector<size_t> order(stale.size());
          std::iota(order.begin(), order.end(), 0);
          auto& hits = frames.stale_hit[i];
          std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return stale[a] != stale[b] ? stale[a] < stale[b] : hits[a] < hits[b];
          });
          std::vector<uint32_t> px;
          std::vector<float> th;
          for (const auto k : order) {
            if (!px.empty() && px.back() == stale[k]) continue;
            px.push_back(stale[k]);
            th.push_back(hits[k]);
          }
          stale.swap(px);
          hits.swap(th);
        }
        stale.shrink_to_fit();
        frames.stale_hit[i].shrink_to_fit();
      }
    }, 1);
    report << ",\"state_starts\":[";
    for (size_t k = 0; k < starts.size(); ++k) {
      const Start& s = starts[k];
      total_cut += s.cut;
      total_stale += s.stale;
      LOG(INFO) << "[SessionRefusion] state_start id=" << s.label << " t_L=" << s.t_L
                << " frames_before=" << s.frames_before << " cut_pixels=" << s.cut.load()
                << " stale_pixels=" << s.stale.load() << " mesh_faces=" << s.mesh_faces.size()
                << " measured_voxels=" << s.measured_voxels;
      report << (k ? "," : "") << "{\"id\":" << s.label << ",\"t_L\":" << s.t_L
             << ",\"frames_before\":" << s.frames_before << ",\"cut_pixels\":" << s.cut.load()
             << ",\"stale_pixels\":" << s.stale.load() << ",\"mesh_faces\":" << s.mesh_faces.size()
             << "}";
    }
    report << "]";
  }
  timer.step("cut_and_stale", "objects=" + std::to_string(starts.size()) +
                                  " cut=" + std::to_string(total_cut) +
                                  " stale=" + std::to_string(total_stale));

  // ------------------------------------------------------------ step 2: TSDF + MC
  auto tsdf = std::make_unique<PresentTsdf>(voxel, trunc, threads);
  std::vector<Eigen::Vector3f> Vp;
  std::vector<Face3> Fp;
  {
    const PresentTsdf::Camera cam{K.width, K.height, K.fx, K.fy, K.cx, K.cy};
    const std::vector<float> mult = PresentTsdf::rayNorm(cam);
    std::vector<float> depth(num_pixels);
    std::vector<float> free_limit;
    size_t pixels = 0;
    frames.forEach([&](size_t i, const std::vector<uint16_t>& range, const std::vector<uint16_t>&) {
      for (size_t p = 0; p < num_pixels; ++p) pixels += range[p] != 0;
      PresentTsdf::depthFromRange(cam, range, depth);
      if (!frames.stale[i].empty()) {
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
    timer.step("tsdf_mc", "pixels=" + std::to_string(pixels) + " units=" +
                              std::to_string(tsdf->numUnits()) + " tsdf_mb=" +
                              std::to_string(tsdf->numBytes() >> 20) + " present_vertices=" +
                              std::to_string(Vp.size()) + " present_faces=" +
                              std::to_string(Fp.size()));
    report << ",\"present\":{\"pixels\":" << pixels << ",\"units\":" << tsdf->numUnits()
           << ",\"vertices\":" << Vp.size() << ",\"faces\":" << Fp.size();
  }
  // ------------------------------------------------ present face normals and centroids
  std::vector<Eigen::Vector3f> face_normal(Fp.size()), centroid(Fp.size());
  for (size_t f = 0; f < Fp.size(); ++f) {
    const Eigen::Vector3d a = Vp[Fp[f][0]].cast<double>(), b = Vp[Fp[f][1]].cast<double>(),
                          c = Vp[Fp[f][2]].cast<double>();
    face_normal[f] = (b - a).cross(c - a).cast<float>();
    centroid[f] = ((a + b + c) / 3.0).cast<float>();
  }

  // ------------------------------------------------------------ depth noise
  // sigma(q): the sensor's depth noise per range bin, measured on the present
  // (1.4826 * median |reading - range| of front-facing present vertices within
  // one truncation). tau(h, q) = max(h, sigma(q)) is how well a surface element
  // of half-voxel h is known along a ray at range q.
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
    std::vector<int64_t> counts;
    sigma = RangeCalibration::finishResidualScale(
        hist, config.min_bin_samples, config.histogram_resolution, &counts);
    std::stringstream ss;
    report << ",\"sigma_cm\":[";
    for (size_t b = 0; b < nb; ++b) {
      ss << (b ? " " : "") << std::round(sigma[b] * 1e4) / 100.0;
      report << (b ? "," : "") << sigma[b] * 100.f;
    }
    report << "]}";  // closes "present"
    timer.step("noise", "sigma_cm=[" + ss.str() + "]");
  }
  auto tauOf = [&](float half, float q) {
    const size_t bin = std::min(config.num_bins - 1,
                                static_cast<size_t>(std::max(0.f, q) / config.range_bin));
    return std::max(half, sigma[bin]);
  };

  const float depth_scale = RangeCalibration::fitScale(collectScalePairs(
      frames, K, project, static_cast<float>(decimal(in.scales.background_truncation)), threads));
  result.depth_scale = depth_scale;
  report << ",\"depth_scale\":" << depth_scale;
  timer.step("depth_scale", "s=" + std::to_string(depth_scale));

  // ------------------------------------------------------------ step 4: measured fill
  // Fill candidates: online faces whose centroid cube (voxel-centre lattice) has a
  // corner the present never integrated, i.e. where it extracts no surface.
  std::vector<uint32_t> candidates;
  for (uint32_t f = 0; f < faces.size(); ++f) {
    const Eigen::Vector3d c =
        (pos[faces[f][0]].cast<double>() + pos[faces[f][1]].cast<double>() +
         pos[faces[f][2]].cast<double>()) / 3.0;
    Eigen::Vector3i cube;
    for (int k = 0; k < 3; ++k) cube[k] = static_cast<int>(std::floor(c[k] / voxel - 0.5));
    bool integrated = true;
    for (int corner = 0; corner < 8 && integrated; ++corner) {
      integrated = tsdf->integrated(cube + Eigen::Vector3i(corner & 1, (corner >> 1) & 1, corner >> 2));
    }
    if (!integrated) candidates.push_back(f);
  }
  tsdf.reset();  // release the volume
  // README (9a): identity and support times come from authorized sensor endpoints.
  struct IdentitySupport {
    uint32_t physical = 0;
    uint64_t votes = 0;
    TimeStamp first = 0, last = 0;
  };
  std::vector<std::vector<IdentitySupport>> identity_support(Fp.size());
  std::vector<uint32_t> face_id(Fp.size(), 0);
  std::vector<TimeStamp> face_first(Fp.size(), 0), face_last(Fp.size(), 0);

  // README (10): history is an explicit predecessor surface. An initial
  // session has empty history; online-fill candidates share the same loss.
  const Surface empty_history;
  const Surface& history = in.shown ? *in.shown : empty_history;
  if (history.face_physical.size() != history.faces.size() ||
      (!history.face_error.empty() && history.face_error.size() != history.faces.size())) {
    throw std::invalid_argument("Surface memory has inconsistent face metadata");
  }
  for (const auto& face : history.faces) {
    for (const auto v : face) {
      if (v >= history.vertices.size()) throw std::invalid_argument("Invalid memory face index");
    }
  }

  struct Element {
    Eigen::Vector3f point = Eigen::Vector3f::Zero();
    Eigen::Vector3f normal = Eigen::Vector3f::Zero();
    float half = 0.f, truncation = 0.f, error = 0.f;
    uint32_t face = 0, slot = 0, physical = 0;
    bool historical = false;
  };
  struct Evidence {
    uint64_t support = 0, free = 0, remeasured = 0, occluded = 0;
    float support_range = kInf;
    bool retain(bool historical) const {
      // README (13)--(14): identical loss for every candidate source.
      return support + static_cast<uint64_t>(historical) > free + remeasured;
    }
  };
  std::vector<Element> elements;
  elements.reserve(candidates.size() + history.faces.size());
  for (const uint32_t f : candidates) {
    const auto& face = faces[f];
    Element element;
    element.point = ((pos[face[0]].cast<double>() + pos[face[1]].cast<double>() +
                      pos[face[2]].cast<double>()) / 3.0).cast<float>();
    element.half = h_obj;
    element.truncation = T_f;
    element.face = f;
    element.slot = fslot[f];
    element.physical = static_cast<uint32_t>(slots[element.slot].physical);
    element.normal = (pos[faces[f][1]] - pos[faces[f][0]]).cross(
        pos[faces[f][2]] - pos[faces[f][0]]);
    elements.push_back(element);
  }
  size_t state_retired = 0;
  for (uint32_t f = 0; f < history.faces.size(); ++f) {
    const auto physical = history.face_physical[f];
    auto target = bg_slot;
    if (physical > 0) {
      const auto it = slot_of_label.find(physical);
      if (it == slot_of_label.end() || in.replaced_states.count(physical)) {
        ++state_retired;
        continue;
      }
      target = static_cast<int32_t>(it->second);
    }
    const auto& face = history.faces[f];
    Element element;
    element.point = (history.vertices[face[0]] + history.vertices[face[1]] +
                     history.vertices[face[2]]) / 3.f;
    element.half = physical ? h_obj : static_cast<float>(0.5 * decimal(in.scales.background_voxel));
    element.truncation = physical ? T_f : static_cast<float>(decimal(in.scales.background_truncation));
    element.error = history.face_error.empty() ? 0.f : history.face_error[f];
    if (!std::isfinite(element.error) || element.error < 0.f) {
      throw std::invalid_argument("Invalid surface source error");
    }
    element.face = f;
    element.slot = static_cast<uint32_t>(target);
    element.physical = physical;
    element.historical = true;
    element.normal = (history.vertices[face[1]] - history.vertices[face[0]]).cross(
        history.vertices[face[2]] - history.vertices[face[0]]);
    elements.push_back(element);
  }

  // README (14b): appearance/time reductions are independent of geometry loss.
  struct AttributeObservation {
    uint32_t vertex = 0;
    TimeStamp last = 0, first = 0;
  };
  std::map<std::pair<uint32_t, uint32_t>, AttributeObservation> observed_attributes;
  // Reduce storage copies before evaluating a source-dependent loss. An
  // authorized predecessor owns p, resolution and error of its key;
  // an online copy must not recreate that hypothesis after its history is rejected.
  {
    std::map<std::pair<uint32_t, GeometryKey>, size_t> representative;
    std::vector<Element> unique_elements;
    unique_elements.reserve(elements.size());
    for (const auto& element : elements) {
      const auto key = element.historical
          ? geometryKey(history.vertices, history.faces[element.face])
          : geometryKey(pos, faces[element.face]);
      const auto [found, inserted] = representative.emplace(
          std::make_pair(element.slot, key), unique_elements.size());
      if (inserted) {
        unique_elements.push_back(element);
      } else if (element.historical && !unique_elements[found->second].historical) {
        unique_elements[found->second] = element;
      }
    }
    // Same-face membership establishes the correspondence; proximity alone does
    // not transfer attributes. Attribute observations cover the complete online
    // mesh, independently of the unintegrated-domain geometry candidate filter.
    for (uint32_t f = 0; f < faces.size(); ++f) {
      const auto slot_id = fslot[f];
      const auto found = representative.find({slot_id, geometryKey(pos, faces[f])});
      if (found == representative.end()) continue;
      const auto& owner = unique_elements[found->second];
      if (!owner.historical) continue;
      const auto& slot = slots[slot_id];
      const auto& mesh = *slot.mesh;
      for (const auto old_vertex : history.faces[owner.face]) {
        for (const auto vertex : faces[f]) {
          if (!(history.vertices[old_vertex].array() == pos[vertex].array()).all()) continue;
          const auto local = vertex - slot.begin;
          const TimeStamp last = mesh.stamps.empty() ? 0 : mesh.stamps.at(local);
          const TimeStamp first = mesh.first_seen_stamps.empty() ? 0 : mesh.first_seen_stamps.at(local);
          const auto [entry, inserted] = observed_attributes.emplace(
              std::make_pair(slot_id, old_vertex), AttributeObservation{local, last, first});
          if (inserted) continue;
          auto& observation = entry->second;
          if (last > observation.last || (last == observation.last && local < observation.vertex))
            observation.vertex = local;
          observation.last = std::max(observation.last, last);
          if (first) observation.first = observation.first ? std::min(observation.first, first) : first;
        }
      }
    }
    report << ",\"candidate_storage_duplicates\":" << elements.size() - unique_elements.size();
    elements = std::move(unique_elements);
  }

  std::vector<float> present_range(Fp.size(), kInf);
  frames.forEach([&](size_t i, const std::vector<uint16_t>& ranges,
                     const std::vector<uint16_t>& ids) {
    const FrameCam& camera = frames.cam(i);
    parallelFor(Fp.size(), threads, [&](size_t begin, size_t end) {
      for (size_t f = begin; f < end; ++f) {
        int u, v;
        float q;
        if (!project(centroid[f], camera, u, v, q)) continue;
        const size_t pixel = static_cast<size_t>(v) * W + u;
        const auto range = ranges[pixel];
        if (range && std::abs(range * 1e-3f - q) <= tauOf(h_obj, q)) {
          present_range[f] = std::min(present_range[f], q);
          const uint32_t label = slot_of_label.count(ids[pixel]) ? ids[pixel] : 0;
          auto& votes = identity_support[f];
          auto found = std::find_if(votes.begin(), votes.end(),
              [label](const auto& item) { return item.physical == label; });
          if (found == votes.end()) {
            votes.push_back({label, 1, frames.stamp(i), frames.stamp(i)});
          } else {
            ++found->votes;
            found->first = std::min(found->first, frames.stamp(i));
            found->last = std::max(found->last, frames.stamp(i));
          }
        }
      }
    }, 16384);
  });
  for (size_t f = 0; f < Fp.size(); ++f) {
    const IdentitySupport* best = nullptr;
    for (const auto& vote : identity_support[f]) {
      if (!best || vote.votes > best->votes ||
          (vote.votes == best->votes && (vote.last > best->last ||
           (vote.last == best->last && vote.physical < best->physical)))) best = &vote;
    }
    if (best) {
      face_id[f] = best->physical;
      face_first[f] = best->first;
      face_last[f] = best->last;
    }
  }
  identity_support.clear();
  identity_support.shrink_to_fit();
  const size_t num_object_faces = Fp.size() - std::count(face_id.begin(), face_id.end(), 0u);
  report << ",\"object_faces\":" << num_object_faces;
  timer.step("identity", "object_faces=" + std::to_string(num_object_faces));
  if (!in.dump_dir.empty()) {
    std::vector<int32_t> labels(face_id.begin(), face_id.end());
    writePly(in.dump_dir + "/present.ply", Vp, Fp, &labels);
  }

  const float positive_scale = std::max(0.f, depth_scale);
  // Predict the actual centre pixel with the first hit of the entire present
  // surface. Identity and orientation are checked after that hit, never by
  // searching through an incompatible foreground surface.
  const TriangleGrid present_grid(Vp, Fp, nullptr, 4.f * v_f);
  std::vector<Evidence> evidence(elements.size());
  std::vector<float> free_limit;
  FrameEndpointIndex endpoint_index(W, H, K.fx, K.fy, K.cx, K.cy);
  frames.forEach([&](size_t i, const std::vector<uint16_t>& ranges,
                     const std::vector<uint16_t>& ids) {
    const FrameCam& camera = frames.cam(i);
    const bool limited = !frames.stale[i].empty();
    if (limited) frames.freeLimit(i, free_limit);
    endpoint_index.bind(ranges);
    parallelFor(elements.size(), threads, [&](size_t begin, size_t end) {
      for (size_t k = begin; k < end; ++k) {
        const auto& element = elements[k];
        auto& ev = evidence[k];
        const Eigen::Vector3d camera_point = camera.R.cast<double>().transpose() *
            (element.point.cast<double>() - camera.t.cast<double>());
        int u = 0, v = 0;
        float q = static_cast<float>(camera_point.norm());
        const bool centre_in_view = project(element.point, camera, u, v, q);
        const float tau = tauOf(element.half, q);
        // A ball strictly behind the camera plane has no forward pixel rays (12a).
        if (camera_point.z() + tau < 0.0) continue;
        const size_t centre_pixel = centre_in_view ? static_cast<size_t>(v) * W + u : 0;
        const uint16_t centre_range = centre_in_view ? ranges[centre_pixel] : 0;
        const float measured_range = centre_range * 1e-3f;
        const float residual = measured_range - q;
        const bool occluded = centre_range && residual < -tau;
        if (occluded) {
          const float window = tauOf(element.half, measured_range) +
                               positive_scale * measured_range + element.error;
          const Eigen::Vector3f pixel_ray((u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.f);
          const Eigen::Vector3f direction = (camera.R * pixel_ray).normalized();
          float hit_range;
          uint32_t face;
          if (present_grid.firstHit(camera.t, direction, hit_range, face)) {
            const Eigen::Vector3f point = camera.t + direction * hit_range;
            const Eigen::Vector3f& normal = face_normal[face];
            const float predicted_range = (point - camera.t).norm();
            // README (11)--(12): under the local single-layer model, this
            // supported, co-oriented visible branch explains a shifted old face.
            const bool same_surface = ids[centre_pixel] == element.physical &&
                face_id[face] == element.physical &&
                element.normal.dot(normal) > 0.f &&
                element.normal.dot(camera.t - element.point) > 0.f &&
                normal.dot(camera.t - point) > 0.f &&
                (point - element.point).dot(camera.t - element.point) > 0.f &&
                (point - element.point).norm() <= window &&
                std::abs(measured_range - predicted_range) <= tauOf(h_obj, predicted_range);
            if (same_surface) {
              ++ev.remeasured;
              continue;
            }
          }
        }

        // README (12): the footprint consists of actual pixel rays intersecting
        // B(camera_point, tau). The tangent bounds only enumerate candidates;
        // every ray still has to intersect the ball, including outside the image.
        const double radius = tau, radius_sq = radius * radius;
        const double z = camera_point.z();
        int64_t x_min = 0, x_max = W - 1, y_min = 0, y_max = H - 1;
        bool valid = z > radius;
        if (valid) {
          const double denominator = (z - radius) * (z + radius);
          const auto bounds = [&](double coordinate, double focal, double centre) {
            const double tangent = radius * std::sqrt(coordinate * coordinate + denominator);
            return std::array<double, 2>{
                std::floor(focal * (coordinate * z - tangent) / denominator + centre),
                std::ceil(focal * (coordinate * z + tangent) / denominator + centre)};
          };
          const auto horizontal = bounds(camera_point.x(), K.fx, K.cx);
          const auto vertical = bounds(camera_point.y(), K.fy, K.cy);
          const double index_limit = std::ldexp(1.0, 63);
          for (const auto value : {horizontal[0], horizontal[1], vertical[0], vertical[1]}) {
            if (!std::isfinite(value) || value < -index_limit || value >= index_limit) valid = false;
          }
          if (valid) {
            x_min = static_cast<int64_t>(horizontal[0]);
            x_max = static_cast<int64_t>(horizontal[1]);
            y_min = static_cast<int64_t>(vertical[0]);
            y_max = static_cast<int64_t>(vertical[1]);
          }
        }
        const auto intersection = [&](int64_t x, int64_t y, Eigen::Vector3d& direction,
                                      double& ray_norm, double& far_intersection) {
          const Eigen::Vector3d ray((static_cast<double>(x) - K.cx) / K.fx,
                                     (static_cast<double>(y) - K.cy) / K.fy, 1.0);
          ray_norm = ray.norm();
          direction = ray / ray_norm;
          const double along = camera_point.dot(direction);
          const double perpendicular_sq = (camera_point - along * direction).squaredNorm();
          if (perpendicular_sq > radius_sq) return false;
          far_intersection = along + std::sqrt(radius_sq - perpendicular_sq);
          return far_intersection >= 0.0;
        };
        // README (12): support is existential and takes priority over free space.
        // The image tree rejects only endpoint boxes outside the same sphere;
        // its leaves retain the exact ray-intersection predicate below.
        const FrameEndpointIndex::Rectangle image_area{
            static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(W, x_min))),
            static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(H, y_min))),
            static_cast<int>(std::max<int64_t>(-1, std::min<int64_t>(W - 1, x_max))),
            static_cast<int>(std::max<int64_t>(-1, std::min<int64_t>(H - 1, y_max)))};
        const bool supported = endpoint_index.any(camera_point, radius_sq, image_area,
            centre_in_view ? u : -1, centre_in_view ? v : -1, [&](int x, int y) {
              Eigen::Vector3d direction;
              double ray_norm, far_intersection;
              return intersection(x, y, direction, ray_norm, far_intersection);
            });
        if (supported) {
          ++ev.support;
          ev.support_range = std::min(ev.support_range, q);
          continue;
        }
        // With support exhausted, an incomplete footprint is already undecidable.
        if (!valid) {
          if (occluded) ++ev.occluded;
          continue;
        }
        // A finite ellipse is checked one integer scan line at a time. Solve its
        // interval on the other axis analytically, and verify an outside integer
        // endpoint by ray intersection. This never walks a 2-D outside rectangle.
        if (valid && (x_min < 0 || y_min < 0 || x_max >= W || y_max >= H)) {
          const bool fixed_y = static_cast<long double>(y_max) - y_min <=
                               static_cast<long double>(x_max) - x_min;
          const double variable_coordinate = camera_point[fixed_y ? 0 : 1];
          const double fixed_coordinate = camera_point[fixed_y ? 1 : 0];
          const double variable_focal = fixed_y ? K.fx : K.fy;
          const double variable_centre = fixed_y ? K.cx : K.cy;
          const double fixed_focal = fixed_y ? K.fy : K.fx;
          const double fixed_centre = fixed_y ? K.cy : K.cx;
          const int64_t lower = fixed_y ? y_min : x_min;
          const int64_t upper = fixed_y ? y_max : x_max;
          const double Q = camera_point.squaredNorm() - radius_sq;
          const double A = fixed_coordinate * fixed_coordinate + (z - radius) * (z + radius);
          const auto outside_at = [&](int64_t fixed) {
            const double normalized = (static_cast<double>(fixed) - fixed_centre) / fixed_focal;
            const double h = fixed_coordinate * normalized + z;
            const double discriminant = Q * (h * h - A * (1.0 + normalized * normalized));
            if (discriminant < 0.0) return false;
            const double centre = variable_coordinate * h / A;
            const double half_width = std::sqrt(discriminant) / A;
            const auto lo = static_cast<int64_t>(std::floor(
                variable_focal * (centre - half_width) + variable_centre));
            const auto hi = static_cast<int64_t>(std::ceil(
                variable_focal * (centre + half_width) + variable_centre));
            for (const auto variable : {lo, lo + 1, hi - 1, hi}) {
              const int64_t x = fixed_y ? variable : fixed;
              const int64_t y = fixed_y ? fixed : variable;
              if (x >= 0 && y >= 0 && x < W && y < H) continue;
              Eigen::Vector3d direction;
              double ray_norm, far_intersection;
              if (intersection(x, y, direction, ray_norm, far_intersection)) return true;
            }
            return false;
          };
          const double ellipse_centre = fixed_focal * fixed_coordinate * z /
                                        ((z - radius) * (z + radius)) + fixed_centre;
          const auto middle = std::clamp(static_cast<int64_t>(std::floor(ellipse_centre + 0.5)),
                                         lower, upper);
          valid = !outside_at(middle);
          for (int64_t fixed = lower; fixed <= upper && valid; ++fixed) {
            if (fixed != middle && outside_at(fixed)) valid = false;
          }
        }
        // A ball crossing the camera plane has an unbounded outside footprint.
        // If its bounds exceed the pixel index domain, only support is decidable.
        // The measurement loop itself is always bounded by the actual image.
        x_min = std::max<int64_t>(0, x_min);
        x_max = std::min<int64_t>(W - 1, x_max);
        y_min = std::max<int64_t>(0, y_min);
        y_max = std::min<int64_t>(H - 1, y_max);
        bool beyond = true, has_footprint = false;
        for (int64_t y = y_min; y <= y_max && valid && beyond; ++y) {
          for (int64_t x = x_min; x <= x_max && valid && beyond; ++x) {
            Eigen::Vector3d direction;
            double ray_norm, far_intersection;
            if (!intersection(x, y, direction, ray_norm, far_intersection)) continue;
            has_footprint = true;
            const size_t pixel = static_cast<size_t>(y) * W + static_cast<size_t>(x);
            const auto range = ranges[pixel];
            if (!range) { valid = false; break; }
            const double measured = range * 1e-3;
            if (measured <= far_intersection) { beyond = false; break; }
            // Use exactly the projective query coordinate and legal contribution
            // domain of PresentTsdf, README (8), on this footprint ray.
            const double projected_range = z * ray_norm;
            const double delta = measured - projected_range;
            if (!(delta > -T_f &&
                  (delta <= T_f || !limited || projected_range < free_limit[pixel] - T_f))) {
              valid = false;
            }
          }
        }
        if (has_footprint && valid && beyond) ++ev.free;
        else if (occluded) ++ev.occluded;
      }
    }, 4096);
    if ((i + 1) % 250 == 0 || i + 1 == frames.size())
      LOG(INFO) << "[SessionRefusion] surface frames=" << i + 1 << "/" << frames.size();
  });

  std::vector<uint8_t> fill(faces.size(), 0);
  std::vector<float> fill_error(faces.size(), 0.f), present_error(Fp.size(), 0.f);
  for (size_t f = 0; f < Fp.size(); ++f) {
    if (std::isfinite(present_range[f])) present_error[f] = positive_scale * present_range[f];
  }
  std::vector<int32_t> history_slot(history.faces.size(), -1);
  size_t num_fill = 0, memory_faces_kept = 0;
  uint64_t support_votes = 0, free_votes = 0, remeasured_votes = 0, occluded_votes = 0;
  for (size_t k = 0; k < elements.size(); ++k) {
    const auto& element = elements[k];
    const auto& ev = evidence[k];
    support_votes += ev.support;
    free_votes += ev.free;
    remeasured_votes += ev.remeasured;
    occluded_votes += ev.occluded;
    if (!ev.retain(element.historical)) continue;
    if (element.historical) {
      history_slot[element.face] = static_cast<int32_t>(element.slot);
      ++memory_faces_kept;
    } else {
      fill[element.face] = 1;
      fill_error[element.face] = positive_scale * ev.support_range;
      ++num_fill;
    }
  }
  const size_t memory_faces_total = history.faces.size();
  report << ",\"surface_model\":\"explanation_loss_v1\",\"elements\":{\"count\":" << elements.size()
         << ",\"support\":" << support_votes << ",\"free\":" << free_votes
         << ",\"remeasured\":" << remeasured_votes << ",\"occluded\":" << occluded_votes
         << ",\"state_retired\":" << state_retired << "}"
         << ",\"fill\":{\"candidates\":" << candidates.size() << ",\"kept\":" << num_fill << "}";
  timer.step("surface_loss", "elements=" + std::to_string(elements.size()) +
      " memory_kept=" + std::to_string(memory_faces_kept) + " fill=" + std::to_string(num_fill));

  // README (14): assembly consumes the single decision; it makes no new geometry decisions.
  std::vector<Eigen::Vector3f> Vnew = Vp;
  std::vector<Face3> Fnew = Fp;
  std::vector<float> new_error = present_error;
  const uint32_t offset = static_cast<uint32_t>(Vp.size());
  Vnew.insert(Vnew.end(), history.vertices.begin(), history.vertices.end());
  for (size_t f = 0; f < history.faces.size(); ++f) {
    const auto& face = history.faces[f];
    Fnew.push_back({face[0] + offset, face[1] + offset, face[2] + offset});
    new_error.push_back(history.face_error.empty() ? 0.f : history.face_error[f]);
  }
  std::vector<std::vector<uint32_t>> present_of_slot(slots.size()), kept_of_slot(slots.size());
  for (uint32_t f = 0; f < Fp.size(); ++f) {
    const auto it = slot_of_label.find(face_id[f]);
    const auto target = face_id[f] && it != slot_of_label.end() ? it->second : static_cast<uint32_t>(bg_slot);
    present_of_slot[target].push_back(f);
  }
  for (uint32_t f = 0; f < history.faces.size(); ++f) {
    if (history_slot[f] >= 0) {
      present_of_slot[history_slot[f]].push_back(static_cast<uint32_t>(Fp.size() + f));
    }
  }
  for (uint32_t f = 0; f < faces.size(); ++f) {
    if (fill[f]) kept_of_slot[fslot[f]].push_back(f);
  }
  // README (14a): set union by exact geometry inside each physical output slot.
  // Ordering above gives current faces ownership of duplicate storage and attributes.
  size_t duplicate_faces = 0;
  for (size_t slot = 0; slot < slots.size(); ++slot) {
    std::set<GeometryKey> seen;
    const auto unique = [&](auto& selected, const auto& vertices, const auto& triangles) {
      auto write = selected.begin();
      for (const auto face : selected) {
        if (seen.insert(geometryKey(vertices, triangles[face])).second) *write++ = face;
        else ++duplicate_faces;
      }
      selected.erase(write, selected.end());
    };
    unique(present_of_slot[slot], Vnew, Fnew);
    unique(kept_of_slot[slot], pos, faces);
  }
  report << ",\"duplicate_storage_faces\":" << duplicate_faces;

  // New meshes are built first and swapped in only when all are complete, so
  // a failure leaves the final map untouched.
  struct NewMesh {
    uint32_t slot = 0;
    spark_dsg::Mesh::Positions points;
    spark_dsg::Mesh::Colors colors;
    spark_dsg::Mesh::Timestamps stamps, first_seen_stamps;
    spark_dsg::Mesh::Labels labels;
    spark_dsg::Mesh::Faces faces;
  };
  std::vector<NewMesh> rebuilt;
  std::stringstream slot_report;
  for (uint32_t s = 0; s < slots.size(); ++s) {
    const Slot& slot = slots[s];
    const auto& mesh = *slot.mesh;
    const auto& kept = kept_of_slot[s];
    const auto& present = present_of_slot[s];
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
    // Attributes of present vertices: the nearest vertex of this mesh before the refusion.
    const bool need_nearest = (mesh.colors.size() == old_n || mesh.stamps.size() == old_n ||
                               mesh.first_seen_stamps.size() == old_n ||
                               mesh.labels.size() == old_n) &&
                              old_n > 0 && !present_old.empty();
    std::vector<size_t> nearest(present_old.size(), 0);
    if (need_nearest) {
      std::vector<Eigen::Vector3f> old_world(pos.begin() + slot.begin, pos.begin() + slot.end);
      const hydra::PointNeighborSearch search(old_world);
      parallelFor(present_old.size(), threads, [&](size_t b, size_t e) {
        for (size_t k = b; k < e; ++k) {
          float d_sq = 0.f;
          size_t idx = 0;
          if (search.search(Vnew[present_old[k]], d_sq, idx)) nearest[k] = idx;
        }
      });
    }
    NewMesh out;
    out.slot = s;
    out.points.resize(new_n);
    for (size_t k = 0; k < kept_old.size(); ++k) out.points[k] = mesh.points[kept_old[k]];
    for (size_t k = 0; k < present_old.size(); ++k) {
      const Eigen::Vector3f& p = Vnew[present_old[k]];
      out.points[kept_old.size() + k] = slot.attrs ? slot.attrs->bounding_box.pointToBoxFrame(p) : p;
    }
    auto rebuild = [&](const auto& values, auto& attributes, bool enabled,
                       const auto& historical_values) {
      using T = typename std::decay_t<decltype(values)>::value_type;
      if (!enabled) return;
      if ((!values.empty() && values.size() != old_n) ||
          (!historical_values.empty() && historical_values.size() != history.vertices.size())) {
        throw std::invalid_argument("Mesh attributes do not match source vertices");
      }
      attributes.assign(new_n, T{});
      for (size_t k = 0; k < kept_old.size(); ++k) {
        if (!values.empty()) attributes[k] = values[kept_old[k]];
      }
      for (size_t k = 0; k < present_old.size(); ++k) {
        const auto vertex = present_old[k];
        if (vertex >= offset) {
          const auto old_vertex = vertex - offset;
          if (!historical_values.empty()) attributes[kept_old.size() + k] = historical_values[old_vertex];
          const auto observed = observed_attributes.find({s, old_vertex});
          const TimeStamp historical_last = history.stamps.empty() ? 0 : history.stamps.at(old_vertex);
          if (observed != observed_attributes.end() && !values.empty() &&
              (historical_values.empty() || observed->second.last >= historical_last))
            attributes[kept_old.size() + k] = values[observed->second.vertex];
        } else if (!values.empty()) {
          attributes[kept_old.size() + k] = values[nearest[k]];
        }
      }
    };
    rebuild(mesh.colors, out.colors, true, history.colors);
    rebuild(mesh.labels, out.labels, true, history.labels);
    rebuild(mesh.stamps, out.stamps, true, history.stamps);
    rebuild(mesh.first_seen_stamps, out.first_seen_stamps, true,
            history.first_seen_stamps);
    // README (14b): observation extent is independent of appearance ownership.
    for (size_t k = 0; k < present_old.size(); ++k) {
      if (present_old[k] < offset) continue;
      const auto vertex = present_old[k] - offset;
      const auto observed = observed_attributes.find({s, vertex});
      if (observed == observed_attributes.end()) continue;
      const auto index = kept_old.size() + k;
      const TimeStamp last = history.stamps.empty() ? 0 : history.stamps.at(vertex);
      const TimeStamp first = history.first_seen_stamps.empty() ? 0 : history.first_seen_stamps.at(vertex);
      out.stamps[index] = std::max(last, observed->second.last);
      out.first_seen_stamps[index] = first && observed->second.first
          ? std::min(first, observed->second.first) : std::max(first, observed->second.first);
    }
    // Current surface time is measured at its own assigned faces, README (9a).
    for (size_t k = 0; k < present_old.size(); ++k) {
      if (present_old[k] >= offset) continue;
      out.stamps[kept_old.size() + k] = 0;
      out.first_seen_stamps[kept_old.size() + k] = 0;
    }
    for (const auto f : present) {
      if (f >= Fp.size()) continue;
      for (const auto vertex : Fp[f]) {
        const auto index = present_new.at(vertex);
        out.stamps[index] = std::max(out.stamps[index], face_last[f]);
        if (face_first[f]) {
          auto& first = out.first_seen_stamps[index];
          first = first ? std::min(first, face_first[f]) : face_first[f];
        }
      }
    }
    out.faces.reserve(kept.size() + present.size());
    for (const auto f : kept) {
      result.surface_error.push_back(fill_error[f]);
      out.faces.push_back({static_cast<size_t>(kept_new[faces[f][0] - slot.begin]),
                           static_cast<size_t>(kept_new[faces[f][1] - slot.begin]),
                           static_cast<size_t>(kept_new[faces[f][2] - slot.begin])});
    }
    for (const auto f : present) {
      result.surface_error.push_back(new_error[f]);
      out.faces.push_back({static_cast<size_t>(present_new[Fnew[f][0]]),
                           static_cast<size_t>(present_new[Fnew[f][1]]),
                           static_cast<size_t>(present_new[Fnew[f][2]])});
    }
    slot_report << (slot_report.tellp() > 0 ? "," : "") << "{\"node\":" << slot.node
                << ",\"physical\":" << slot.physical << ",\"background\":" << slot.background
                << ",\"old_vertices\":" << old_n << ",\"kept_faces\":" << kept.size()
                << ",\"present_faces\":" << present.size() << ",\"vertices\":" << new_n << "}";
    rebuilt.push_back(std::move(out));
  }
  for (auto& out : rebuilt) {
    spark_dsg::Mesh mesh(true, true, true, true);
    mesh.points.swap(out.points);
    mesh.colors.swap(out.colors);
    mesh.stamps.swap(out.stamps);
    mesh.first_seen_stamps.swap(out.first_seen_stamps);
    mesh.labels.swap(out.labels);
    mesh.faces.swap(out.faces);
    *slots[out.slot].mesh = std::move(mesh);
  }
  dsg.setMesh(background);
  timer.step("compose", "memory_faces=" + std::to_string(memory_faces_total) + " kept=" +
                            std::to_string(memory_faces_kept) + " fill=" + std::to_string(num_fill));
  report << ",\"compose\":{\"memory_faces\":" << memory_faces_total
         << ",\"memory_faces_kept\":" << memory_faces_kept << ",\"fill_faces\":" << num_fill
         << ",\"slots\":[" << slot_report.str() << "]}";
  const double total =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - timer.start).count();
  report << ",\"timings_s\":{" << timer.timings.str() << ",\"total\":" << total
         << "},\"rss_mb_end\":" << rssMb() << "}";
  result.report_json = report.str();
  std::stringstream summary;
  summary << "frames=" << frames.size() << " present_vertices=" << Vp.size()
          << " present_faces=" << Fp.size() << " cut=" << total_cut << " stale=" << total_stale
          << " fill=" << num_fill << "/" << candidates.size() << " memory_faces_kept="
          << memory_faces_kept << "/" << memory_faces_total << " total_s=" << total;
  result.summary = summary.str();
  result.applied = true;
  if (!in.dump_dir.empty()) {
    std::ofstream(in.dump_dir + "/refusion_report.json") << result.report_json;
  }
  return result;
}

}  // namespace khronos
