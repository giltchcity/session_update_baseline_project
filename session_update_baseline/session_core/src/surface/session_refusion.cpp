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

#include "session_core/evidence/element_round.h"
#include "session_core/evidence/range_encoding.h"
#include "session_core/model/model_math.h"
#include "session_core/model/range_model.h"
#include "session_core/model/sensor_calibrator.h"
#include "session_core/surface/element_posterior.h"
#include "session_core/surface/present_tsdf.h"
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
// The quantisation unit u of the depth code (README table 5.1, class 1): code * u is metres.
constexpr double kUnit = static_cast<double>(measurement::kRangeUnit);
constexpr float kUnitF = measurement::kRangeUnit;

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
 * pixels of step 1 (object L before t_L) and step 1b (`stale`) removed, and the readings without
 * identity judged by principle 5: a reading enters the refusion iff the surface it measured
 * persists to the end of the session with probability at least 1/2.
 */
class SessionFrames {
 public:
  SessionFrames(const std::vector<FrameArchive::Frame>& frames,
                const FrameArchive::Camera& camera,
                const std::map<size_t, std::optional<TimeStamp>>& state_starts,
                TimeStamp final_stamp)
      : frames_(frames),
        num_pixels_(static_cast<size_t>(camera.width) * camera.height),
        final_stamp_(final_stamp),
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
  void decode(size_t i, std::vector<uint16_t>& range, std::vector<uint16_t>& ids,
              std::vector<uint16_t>* classes = nullptr, std::vector<uint8_t>* motion = nullptr) const {
    if (!frames_[i].decode(num_pixels_, range, ids, classes, motion)) {
      throw std::runtime_error("Session frame decode failed at " +
                               std::to_string(frames_[i].stamp));
    }
  }

  /** Step 1 on a decoded frame, and the judgement of the readings without identity; counts the
   * pixels removed by step 1 per id in `removed` (if given). */
  void cut(size_t i, std::vector<uint16_t>& range, const std::vector<uint16_t>& ids,
           const std::vector<uint16_t>& classes, const std::vector<uint8_t>& motion,
           std::unordered_map<uint16_t, size_t>* removed = nullptr) const {
    const TimeStamp t = frames_[i].stamp;
    (void)classes;
    for (size_t p = 0; p < num_pixels_; ++p) {
      if (!range[p]) continue;
      if (ids[p]) {
        if (t < start_of_[ids[p]]) {
          range[p] = 0;
          if (removed) ++(*removed)[ids[p]];
        }
      } else if (motion[p]) {
        // The native integration mask excludes the motion clusters: a reading in motion belongs
        // to a state that ends at once (principle 5).
        range[p] = 0;
      }
    }
  }

  /** Frame i as the present sees it (steps 1 and 1b applied). */
  void load(size_t i, std::vector<uint16_t>& range, std::vector<uint16_t>& ids) const {
    thread_local std::vector<uint16_t> classes;
    thread_local std::vector<uint8_t> motion;
    decode(i, range, ids, &classes, &motion);
    cut(i, range, ids, classes, motion);
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
  const TimeStamp final_stamp_;
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

}  // namespace

SessionRefusion::Result SessionRefusion::apply(DynamicSceneGraph& dsg, const Inputs& in) const {
  Result result;
  StepTimer timer;
  if (!in.frames || in.frames->empty() || !in.camera.valid()) {
    throw std::invalid_argument("Session surface estimator needs archived frames and a valid camera");
  }
  const double voxel = decimal(in.scales.object_voxel);
  const double trunc = decimal(in.scales.object_truncation);
  if (!std::isfinite(voxel) || voxel <= 0.0 || !std::isfinite(trunc) || trunc <= 0.0 ||
      !std::isfinite(in.scales.background_voxel) || in.scales.background_voxel <= 0.f ||
      !std::isfinite(in.scales.background_truncation) || in.scales.background_truncation <= 0.f) {
    throw std::invalid_argument("Invalid surface resolution configuration");
  }
  if (!in.psi.valid()) {
    throw std::invalid_argument("The session surface update needs the effective range error model");
  }
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

  // README (8): the registry gives each identity its domain; an identity without an established
  // current state has the empty domain.
  SessionFrames frames(*in.frames, K, in.state_starts, in.final_stamp);

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
      from[label] = it == in.state_starts.end() ? std::numeric_limits<TimeStamp>::max()
                                                : it->second.value_or(std::numeric_limits<TimeStamp>::max());
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
          local[ids[p]].insert(voxelKey(c.t + c.R * (dirs[p] * (range[p] * kUnitF))));
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
      std::vector<uint16_t> range, ids, classes;
      std::vector<uint8_t> motion;
      std::unordered_map<uint16_t, size_t> removed;
      for (size_t j = b; j < e; ++j) {
        const size_t i = before[j];
        const FrameCam& c = frames.cam(i);
        frames.decode(i, range, ids, &classes, &motion);
        // Step 1: the object's own pixels before its current state began.
        removed.clear();
        frames.cut(i, range, ids, classes, motion, &removed);
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
            if (!(pc.z() > 0.f)) {
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
              const float t_end = d * kUnitF - T_f;
              const Eigen::Vector3f dir = c.R * ray;
              bool hit = s.grid && s.grid->firstHit(c.t, dir, t_hit, hit_face) && t_hit < t_end;
              if (!hit && s.measured && t_end > 0.f) {
                // The ray's free part [0, R - T] through the measured voxels (3D DDA
                // over the part inside their bounding box).
                float t0 = 0.f, t1 = t_end;
                for (int k = 0; k < 3 && t0 <= t1; ++k) {
                  if (dir[k] == 0.f) {
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
                    if (dk == 0.f) return kInf;
                    const float boundary = static_cast<float>(cell + (step > 0 ? 1 : 0)) * v_f;
                    return (boundary - o) / dk;
                  };
                  float tx = next(x, sx, c.t.x(), dir.x()), ty = next(y, sy, c.t.y(), dir.y()),
                        tz = next(z, sz, c.t.z(), dir.z());
                  const float dx = dir.x() == 0.f ? kInf : v_f / std::abs(dir.x());
                  const float dy = dir.y() == 0.f ? kInf : v_f / std::abs(dir.y());
                  const float dz = dir.z() == 0.f ? kInf : v_f / std::abs(dir.z());
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
    // README (9), principle 7: constant weight 1, the special case of the precision weighting for
    // equal sigma_eff within a session (the verified choice; sigma_eff weighting stays optional).
    size_t pixels = 0;
    frames.forEach([&](size_t i, const std::vector<uint16_t>& range, const std::vector<uint16_t>&) {
      for (size_t p = 0; p < num_pixels; ++p) pixels += range[p] != 0;
      PresentTsdf::depthFromRange(cam, range, depth);
      if (!frames.stale[i].empty()) {
        frames.freeLimit(i, free_limit);
        tsdf->integrate(cam, frames.pose(i), depth, mult, &free_limit, nullptr);
      } else {
        tsdf->integrate(cam, frames.pose(i), depth, mult, nullptr, nullptr);
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
  // sigma_table(rho) of (9v) and w_pm of principle 8 on the present: the signed residuals of the
  // readings against the front-facing vertices of the fused present surface, per range bin. Like every
  // quantity of principle 8 they are the posterior of "prior + data": the prior is the model the
  // session holds (the previous session's bins, the neighbouring bins, the sensor curve -- already
  // merged online), counted as one sample, so a bin with few residuals stays near it and no bin
  // switches on at a sample count (the pooled fallback of the first implementation is gone).
  // tau(h, rho) = max(h, sigma_table(rho)) is how well a surface element of half-voxel h is known
  // along a ray at range rho.
  model::SensorCalibrator present_residuals;
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
    constexpr size_t nb = model::SensorCalibrator::kRangeBins;
    // The residuals inside the truncation are binned at the cell of the calibrator; the others are
    // outliers (far: see-through, near: occluded).
    const long half_cells = static_cast<long>(std::ceil(trunc / model::SensorCalibrator::kCell));
    const size_t cells = 2 * static_cast<size_t>(half_cells);
    frames.forEach([&](size_t i, const std::vector<uint16_t>& rng, const std::vector<uint16_t>&) {
      const FrameCam& c = frames.cam(i);
      parallelFor(Vp.size(), threads, [&](size_t b, size_t e) {
        std::vector<int64_t> local(nb * cells, 0), far(nb, 0), near(nb, 0);
        std::vector<uint8_t> touched(nb, 0);
        for (size_t j = b; j < e; ++j) {
          int u, v;
          float q;
          if (!project(Vp[j], c, u, v, q)) continue;
          if (N[j].dot(c.t - Vp[j]) <= 0.f) continue;
          const uint16_t d = rng[static_cast<size_t>(v) * W + u];
          if (!d) continue;
          const double z = d * kUnit - static_cast<double>(q);
          const size_t bin = std::min(nb - 1, static_cast<size_t>(q / model::SensorCalibrator::kRangeBin));
          const long cell = static_cast<long>(std::floor(z / model::SensorCalibrator::kCell));
          touched[bin] = 1;
          if (cell >= half_cells) ++far[bin];
          else if (cell < -half_cells) ++near[bin];
          else ++local[bin * cells + static_cast<size_t>(cell + half_cells)];
        }
        for (size_t bin = 0; bin < nb; ++bin) {
          if (!touched[bin]) continue;
          present_residuals.addCells(
              bin, -half_cells,
              std::vector<int64_t>(local.begin() + bin * cells, local.begin() + (bin + 1) * cells),
              far[bin], near[bin]);
        }
      }, 16384);
    });
  }
  // README (6m), (9v): the session end compares the readings with the fused surface of the session
  // end, the present. The posterior is taken from the model the session holds as prior; the range
  // scale and the alignment residual are those of the session's estimator.
  model::RangeModel psi = present_residuals.estimate(in.psi, {}, K.max_range, trunc);
  psi.zeta = in.psi.zeta;
  psi.sigma_x = in.psi.sigma_x;
  psi.sigma_x_known = in.psi.sigma_x_known;
  if (!psi.valid()) throw std::invalid_argument("The session surface update needs the range model");
  {
    std::stringstream ss;
    report << ",\"sigma_cm\":[";
    result.sigma.assign(psi.sigma_table.size(), 0.f);
    for (size_t b = 0; b < psi.sigma_table.size(); ++b) {
      result.sigma[b] = static_cast<float>(psi.sigma_table[b]);
      ss << (b ? " " : "") << std::round(psi.sigma_table[b] * 1e4) / 100.0;
      report << (b ? "," : "") << psi.sigma_table[b] * 100.0;
    }
    report << "]}";  // closes "present"
    timer.step("noise", "sigma_cm=[" + ss.str() + "] residuals=" +
                            std::to_string(present_residuals.numResiduals()));
  }
  const double max_range = K.max_range, min_range = K.min_range;
  result.depth_scale = static_cast<float>(psi.zeta);
  report << ",\"depth_scale\":" << psi.zeta << ",\"sigma_x\":" << psi.sigma_x
         << ",\"sigma_x_known\":" << psi.sigma_x_known << ",\"w_plus\":" << psi.w_plus
         << ",\"w_minus\":" << psi.w_minus;
  timer.step("depth_scale", "zeta=" + std::to_string(psi.zeta));
  // The residual scale of the fused present surface at a range (quantisation floor, README (9c)).
  const auto sigmaTable = [&](double range) { return psi.sigmaTable(range); };

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
    // README principle 9, (10b): a face of this session's own present surface; `face` indexes Fp.
    bool present = false;
    // README principle 12: the recorded distance rho_e and the scale zeta_e of the session that
    // made a memory element (rho_e 0: no record, the range of the reading stands in for it).
    float record_range = 0.f, record_zeta = 0.f;
    // README principles 4 and 9: a completion candidate is fitted from the readings of its own
    // observation, which cannot confirm it again; its evidence is the readings after its last
    // observation. 0: no restriction (a memory element; a mesh without observation times).
    TimeStamp last_observed = 0;
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
    // The newest observation of the triangle: its own readings cannot confirm it again.
    {
      const Slot& owner = slots[element.slot];
      const auto& mesh = *owner.mesh;
      if (mesh.stamps.size() == mesh.numVertices()) {
        for (const auto vertex : face) {
          element.last_observed = std::max<TimeStamp>(element.last_observed, mesh.stamps[vertex - owner.begin]);
        }
      }
    }
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
    if (history.face_rho.size() == history.faces.size()) element.record_range = history.face_rho[f];
    element.record_zeta = history.face_zeta.size() == history.faces.size()
        ? history.face_zeta[f] : static_cast<float>(in.previous_zeta);
    element.normal = (history.vertices[face[1]] - history.vertices[face[0]]).cross(
        history.vertices[face[2]] - history.vertices[face[0]]);
    elements.push_back(element);
  }

  // README (14a), appendix: appearance/time reductions are independent of geometry loss.
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

  // README (9), (9a): the posterior precision of a present face is the sum of 1/sigma_table^2 of the
  // echoes it explains (the hit class of (6e) at the residual scale of the fused surface); the mean
  // range of those echoes is the distance rho_e of principle 12.
  std::vector<double> face_precision(Fp.size(), 0.0), present_range_sum(Fp.size(), 0.0);
  std::vector<uint32_t> present_range_count(Fp.size(), 0);
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
        if (!range) continue;
        const double sigma_q = sigmaTable(q);
        if (model::classifyRange(psi, range * kUnit, q, sigma_q, min_range, max_range) !=
            model::RangeClass::kHit) {
          continue;
        }
        face_precision[f] += 1.0 / (sigma_q * sigma_q);
        present_range_sum[f] += q;
        ++present_range_count[f];
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
    }, 16384);
  });
  // README (9): the uncertainty of a present face is the posterior standard deviation.
  std::vector<float> present_error(Fp.size(), 0.f);
  for (size_t f = 0; f < Fp.size(); ++f) {
    if (face_precision[f] > 0) present_error[f] = static_cast<float>(1.0 / std::sqrt(face_precision[f]));
  }
  // README principle 10 (a): the identity of a face of the present surface is a representation
  // output (5e). A face less than one voxel h_o from the current reconstruction of an object is the
  // same zero crossing as that surface in this representation and belongs to the object (the
  // nearest one if several); otherwise it belongs to the background.
  {
    std::vector<uint32_t> object_faces;
    for (uint32_t f = 0; f < faces.size(); ++f) {
      const Slot& slot = slots[fslot[f]];
      if (!slot.background && slot.physical != 0) object_faces.push_back(f);
    }
    if (!object_faces.empty()) {
      const TriangleGrid object_grid(pos, faces, &object_faces, 4.f * v_f);
      parallelFor(Fp.size(), threads, [&](size_t begin, size_t end) {
        for (size_t f = begin; f < end; ++f) {
          float distance = 0.f;
          Eigen::Vector3f nearest;
          uint32_t face = 0;
          if (!object_grid.closest(centroid[f], v_f, distance, nearest, face)) continue;
          face_id[f] = static_cast<uint32_t>(slots[fslot[face]].physical);
        }
      }, 4096);
    }
  }
  // Support times come from the authorised sensor endpoints of the face: those of its identity
  // where it has any, else the best supported label.
  for (size_t f = 0; f < Fp.size(); ++f) {
    const IdentitySupport* best = nullptr;
    for (const auto& vote : identity_support[f]) {
      if (face_id[f] != 0 && vote.physical == face_id[f]) {
        best = &vote;
        break;
      }
      if (!best || vote.votes > best->votes ||
          (vote.votes == best->votes && (vote.last > best->last ||
           (vote.last == best->last && vote.physical < best->physical)))) best = &vote;
    }
    if (best) {
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

  const TriangleGrid present_grid(Vp, Fp, nullptr, 4.f * v_f);

  // The nearest face of the present surface of the memory element's identity in front of it (the
  // viewing side of its normal) within `reach`, and the offset d > 0 of that surface in front of it.
  struct FrontSurface {
    bool found = false;
    uint32_t face = 0;
    double d = 0.0;
  };
  const auto frontSurface = [&](const Element& element, double reach) {
    FrontSurface result;
    if (!(reach > 0.0)) return result;
    const auto accept = [&](uint32_t face, const Eigen::Vector3f&) {
      return face_id[face] == element.physical && element.normal.dot(face_normal[face]) > 0.f;
    };
    float distance = 0.f;
    Eigen::Vector3f nearest;
    uint32_t face = 0;
    if (!present_grid.closest(element.point, static_cast<float>(reach), distance, nearest, face, accept)) {
      return result;
    }
    const Eigen::Vector3d n = face_normal[face].cast<double>().normalized();
    result.d = (nearest - element.point).cast<double>().dot(n);
    result.found = true;
    result.face = face;
    return result;
  };

  // ------------------------------------------- principle 10: sigma_x from the first overlap
  // The memory elements are judged across sessions with sigma_x. Where the session's estimator has
  // none yet, it is estimated here before any memory element is judged: from the offsets d_j of the
  // memory elements from the present surface of the same identity along the line of sight, after
  // the predicted scale displacement is removed, of ALL the pairs inside the window of (12d)
  // (a pair taken only inside the hit band would cut the misalignment itself off and bias the
  // estimate low),
  //   sigma_x^2 = (1.4826 median_j |d_j - (zeta_now rho_j - zeta_e rho_{e,j})|)^2 - sigma_ee'^2,
  // sigma_ee'^2 the discretisation terms h^2/12 of the two surfaces, as the posterior of the prior
  // (the previous session's sigma_x, else W / sqrt(3) as one pair) and these pairs; rho_j is the
  // nearest range at which the element is seen (the frames are sampled evenly to the computation
  // budget of the nearest range).
  bool any_history = false;
  for (const auto& element : elements) any_history = any_history || element.historical;
  if (!psi.sigma_x_known && any_history) {
    const size_t frame_step = std::max<size_t>(1, frames.size() / model::SensorCalibrator::kScaleFrames);
    std::vector<double> nearest_range(elements.size(), std::numeric_limits<double>::infinity());
    for (size_t i = 0; i < frames.size(); i += frame_step) {
      const FrameCam& camera = frames.cam(i);
      parallelFor(elements.size(), threads, [&](size_t begin, size_t end) {
        for (size_t k = begin; k < end; ++k) {
          if (!elements[k].historical) continue;
          int u, v;
          float q;
          if (project(elements[k].point, camera, u, v, q)) {
            nearest_range[k] = std::min(nearest_range[k], static_cast<double>(q));
          }
        }
      }, 4096);
    }
    model::SensorCalibrator alignment;
    size_t pairs = 0;
    std::mutex pair_mutex;
    parallelFor(elements.size(), threads, [&](size_t begin, size_t end) {
      std::vector<model::BandPair> local;
      for (size_t k = begin; k < end; ++k) {
        const auto& element = elements[k];
        if (!element.historical || !std::isfinite(nearest_range[k])) continue;
        const double rho = nearest_range[k];
        const double rho_e = element.record_range > 0.f ? element.record_range : rho;
        // W of (12d): tau + |zeta_now| rho + |zeta_e| rho_e, tau = max(h, sigma_table(rho)).
        const double tau = std::max(static_cast<double>(element.half), sigmaTable(rho));
        const double window = tau + std::abs(psi.zeta) * rho +
                              std::abs(static_cast<double>(element.record_zeta)) * rho_e;
        const auto front = frontSurface(element, window);
        if (!front.found || !(std::abs(front.d) <= window)) continue;
        // The present surface lies d in front of the memory element; the readings put it behind
        // by the predicted scale displacement b.
        const double residual = -front.d - psi.bias(rho, rho_e, element.record_zeta);
        const double edge = 2.0 * element.half;
        local.push_back({residual, (edge * edge + voxel * voxel) / 12.0, window});
      }
      std::lock_guard<std::mutex> lock(pair_mutex);
      for (const auto& pair : local) alignment.addBandPair(pair);
      pairs += local.size();
    }, 4096);
    report << ",\"sigma_x_pairs\":" << pairs;
    double sigma_x = 0.0;
    if (alignment.estimateSigmaX(in.prior_sigma_x, in.prior_sigma_x_known, sigma_x)) {
      psi.sigma_x = sigma_x;
      psi.sigma_x_known = true;
    }
    report << ",\"sigma_x\":" << psi.sigma_x << ",\"sigma_x_known\":" << psi.sigma_x_known;
  }

  // ---------------------------------------------------- principle 9: the verdicts of the frames
  // README principle 9, assumption (2): a frame has four relations to an element, with
  // tau = max(h, sigma_table(rho)), h the half voxel of the layer of the element and the footprint
  // the projection of the ball of radius tau (radius f tau / z in the image):
  //   hit          a reading in the footprint has its 3D point inside the ball of radius tau;
  //   see-through  every reading in the footprint is valid, and all are behind the element by more
  //                than tau, in free space that counts;
  //   blocked in band   a reading in front of it by more than tau and within the truncation T;
  //   blocked outside   a reading farther in front.
  // The frames are the rounds of (7) for the deletion of a record and the votes of (10b) for the
  // display. The first-return classification (6e) of principle 4 is not used here.
  // README principle 9, (10b): the faces of the present surface are elements too. The readings that
  // built a face are its hits in this session; a person who stood and left is seen through by the
  // later readings, and the face is not displayed when it is seen through more often than hit.
  const size_t present_base = elements.size();
  elements.reserve(present_base + Fp.size());
  for (size_t f = 0; f < Fp.size(); ++f) {
    const bool object = face_id[f] != 0;
    Element element;
    element.point = centroid[f];
    element.normal = face_normal[f];
    element.half = object ? h_obj : static_cast<float>(0.5 * decimal(in.scales.background_voxel));
    element.truncation = object ? T_f : static_cast<float>(decimal(in.scales.background_truncation));
    element.face = static_cast<uint32_t>(f);
    element.physical = face_id[f];
    element.present = true;
    elements.push_back(element);
  }
  std::vector<surface::ElementTally> tallies(elements.size());
  std::vector<float> free_limit;
  // Performance only: per-pixel unit ray of README (12a), so each ray is computed once.
  std::vector<Eigen::Vector3d> pixel_direction(num_pixels);
  std::vector<double> pixel_ray_norm(num_pixels);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const Eigen::Vector3d ray((static_cast<double>(x) - K.cx) / K.fx,
                                (static_cast<double>(y) - K.cy) / K.fy, 1.0);
      const size_t pixel = static_cast<size_t>(y) * W + x;
      pixel_ray_norm[pixel] = ray.norm();
      pixel_direction[pixel] = ray / pixel_ray_norm[pixel];
    }
  }
  frames.forEach([&](size_t i, const std::vector<uint16_t>& ranges,
                     const std::vector<uint16_t>&) {
    const FrameCam& camera = frames.cam(i);
    const bool limited = !frames.stale[i].empty();
    if (limited) frames.freeLimit(i, free_limit);
    const TimeStamp frame_stamp = frames.stamp(i);
    parallelFor(elements.size(), threads, [&](size_t begin, size_t end) {
      for (size_t k = begin; k < end; ++k) {
        const auto& element = elements[k];
        if (frame_stamp <= element.last_observed) continue;  // the candidate's own observation
        auto& tally = tallies[k];
        const Eigen::Vector3d camera_point = camera.R.cast<double>().transpose() *
            (element.point.cast<double>() - camera.t.cast<double>());
        const double z = camera_point.z();
        const double nominal = camera_point.norm();
        // tau = max(h, sigma_table(rho)); the footprint is the projection of the ball of radius tau.
        const double tau = std::max(static_cast<double>(element.half), sigmaTable(nominal));
        const double radius = tau;
        // README (12a): a finite footprint has pixel rays only in front of the camera plane.
        if (!(z > radius)) continue;
        // README (12a): exact pixel bounding box of the projected ball. Pixels outside the view
        // carry no information (README s4: unknown data).
        const double denominator = (z - radius) * (z + radius);
        const auto bounds = [&](double coordinate, double focal, double centre) {
          const double tangent = radius * std::sqrt(coordinate * coordinate + denominator);
          return std::array<double, 2>{
              std::floor(focal * (coordinate * z - tangent) / denominator + centre),
              std::ceil(focal * (coordinate * z + tangent) / denominator + centre)};
        };
        const auto horizontal = bounds(camera_point.x(), K.fx, K.cx);
        const auto vertical = bounds(camera_point.y(), K.fy, K.cy);
        const int x_min = static_cast<int>(std::max(0.0, horizontal[0]));
        const int x_max = static_cast<int>(std::min(static_cast<double>(W - 1), horizontal[1]));
        const int y_min = static_cast<int>(std::max(0.0, vertical[0]));
        const int y_max = static_cast<int>(std::min(static_cast<double>(H - 1), vertical[1]));
        if (x_min > x_max || y_min > y_max) continue;
        // sigma_eff of (6s) at the element's own range and incidence: the weight of its supporting
        // echoes in (9) and the false see-through rate m_0 of the frame.
        const double normal_length = element.normal.cast<double>().norm();
        const double cosine = normal_length > 0.0
            ? std::abs(element.normal.cast<double>().dot(camera_point)) / (normal_length * nominal)
            : 1.0;
        const double incidence = std::acos(std::min(1.0, cosine));
        const double sigma = psi.sigmaEff(nominal, incidence, 2.0 * element.half, element.historical);
        const double radius_sq = radius * radius;
        const double band = static_cast<double>(element.truncation);
        bool hit = false, any_reading = false, all_valid = true, all_behind = true;
        for (int y = y_min; y <= y_max; ++y) {
          for (int x = x_min; x <= x_max; ++x) {
            const size_t pixel = static_cast<size_t>(y) * W + x;
            const double along = camera_point.dot(pixel_direction[pixel]);
            if ((camera_point - along * pixel_direction[pixel]).squaredNorm() > radius_sq) continue;
            const uint16_t code = ranges[pixel];
            const double reading = code * kUnit;
            if (!code || reading < min_range || reading > max_range) {
              all_valid = false;  // no valid echo: no information, and not a see-through frame
              continue;
            }
            any_reading = true;
            // The 3D point of the reading against the ball of radius tau.
            if ((reading * pixel_direction[pixel] - camera_point).squaredNorm() <= radius_sq) {
              hit = true;
              tally.precision += 1.0 / (sigma * sigma);
            }
            // Principle 11: a reading in front of the element by more than tau and within the
            // truncation band shares the zero crossing of the surface that cut it off.
            tally.addReading(along - reading > tau && along - reading <= band);
            // README (14f): a valid reading within tau of the range of the element puts the surface on it.
            const double projective = z * pixel_ray_norm[pixel];  // projective range, as in (8)
            if (std::abs(reading - projective) <= tau) tally.near_reading = true;
            // See-through needs every reading behind it by more than tau, in free space that counts
            // (README (8): free space through the object's own surface before its state began is
            // not a measurement of the new state).
            if (!(reading - along > tau) || (limited && !(projective < free_limit[pixel] - T_f))) {
              all_behind = false;
            }
          }
        }
        const double predicted = psi.predictedSeeThrough(nominal, sigma, max_range);
        if (hit) tally.addFrame(false, nominal, predicted);
        else if (any_reading && all_valid && all_behind) tally.addFrame(true, nominal, predicted);
      }
    }, 4096);
    if ((i + 1) % 250 == 0 || i + 1 == frames.size())
      LOG(INFO) << "[SessionRefusion] surface frames=" << i + 1 << "/" << frames.size();
  });

  // README principle 10 (b), eq. (12d): a memory element that no frame saw through (10b) and that
  // principle 11 does not hide, with a present surface of the same identity in front of it on the
  // line of sight at a distance d with 2h < d <= tau + |zeta_now| rho + |zeta_e| rho_e, is the
  // same surface estimated twice: the present surface represents it and the memory element is not
  // output. rho is the range of the nearest reading that reaches it; h is the half voxel of its
  // layer, so 2h is one voxel: within it the two are one zero crossing (principle 11).
  std::vector<uint8_t> shadow(elements.size(), 0);
  std::vector<uint32_t> shadow_face(elements.size(), 0);
  parallelFor(elements.size(), threads, [&](size_t begin, size_t end) {
    for (size_t k = begin; k < end; ++k) {
      const auto& element = elements[k];
      const auto& tally = tallies[k];
      if (!element.historical || (tally.range_count == 0 && tally.arrived == 0)) continue;
      if (tally.through > tally.hits) continue;
      if (tally.frames() == 0 && tally.blockedMajority()) continue;
      const double rho = tally.nearestRange() > 0.0 ? tally.nearestRange() : tally.meanRange();
      const double rho_e = element.record_range > 0.f ? element.record_range : rho;
      const double tau = std::max(static_cast<double>(element.half), sigmaTable(rho));
      const double reach = tau + std::abs(psi.zeta) * rho +
                           std::abs(static_cast<double>(element.record_zeta)) * rho_e;
      const auto front = frontSurface(element, reach);
      shadow[k] = front.found && front.d > 2.0 * element.half && front.d <= reach;
      if (shadow[k]) shadow_face[k] = front.face;
    }
  }, 4096);
  // README principle 7, (9): the surface of the previous session that is the same surface enters the
  // present surface with its own precision 1/error^2 (W_0 of (9)), so the face is known better.
  for (size_t k = 0; k < elements.size(); ++k) {
    if (shadow[k] && elements[k].error > 0.f) {
      face_precision[shadow_face[k]] += 1.0 / (static_cast<double>(elements[k].error) * elements[k].error);
    }
  }
  for (size_t f = 0; f < Fp.size(); ++f) {
    if (face_precision[f] > 0) present_error[f] = static_cast<float>(1.0 / std::sqrt(face_precision[f]));
  }

  // README principle 10 (c): an object's memory element that is seen in this session, lies more
  // than a voxel from the present surface of the object and has, among the fixed directions in
  // which a ray from it first meets that surface, more back faces than front faces, lies inside
  // the solid and is not an outer surface: it is not output (the record stays, principle 9). The
  // directions are a computation budget of 64 (a Fibonacci lattice); directions without a hit give
  // no information.
  std::vector<uint8_t> inside(elements.size(), 0);
  {
    constexpr int kDirections = 64;
    std::vector<Eigen::Vector3f> directions(kDirections);
    const double golden_angle = M_PI * (3.0 - std::sqrt(5.0));
    for (int d = 0; d < kDirections; ++d) {
      const double z = 1.0 - (2.0 * d + 1.0) / kDirections;
      const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
      directions[d] = Eigen::Vector3f(static_cast<float>(r * std::cos(golden_angle * d)),
                                      static_cast<float>(r * std::sin(golden_angle * d)),
                                      static_cast<float>(z));
    }
    std::map<uint32_t, std::vector<uint32_t>> faces_of;
    for (uint32_t f = 0; f < Fp.size(); ++f) {
      if (face_id[f] != 0) faces_of[face_id[f]].push_back(f);
    }
    std::map<uint32_t, std::unique_ptr<TriangleGrid>> grids;
    for (const auto& [identity, subset] : faces_of) {
      grids[identity] = std::make_unique<TriangleGrid>(Vp, Fp, &subset, 4.f * v_f);
    }
    parallelFor(elements.size(), threads, [&](size_t begin, size_t end) {
      for (size_t k = begin; k < end; ++k) {
        const auto& element = elements[k];
        const auto& tally = tallies[k];
        if (!element.historical || element.physical == 0 || shadow[k]) continue;
        if (tally.range_count == 0 && tally.arrived == 0) continue;  // not seen in this session
        const auto grid = grids.find(element.physical);
        if (grid == grids.end()) continue;
        float distance = 0.f;
        Eigen::Vector3f nearest;
        uint32_t nearest_face = 0;
        if (grid->second->closest(element.point, v_f, distance, nearest, nearest_face)) continue;
        int back = 0, front = 0;
        for (const auto& direction : directions) {
          float t = 0.f;
          uint32_t face = 0;
          if (!grid->second->firstHit(element.point, direction, t, face)) continue;
          if (direction.dot(face_normal[face]) > 0.f) ++back;
          else ++front;
        }
        inside[k] = back > front;
      }
    }, 1024);
  }

  // ---------------------------------------------- principle 9: the display and the records
  // README (5e), principles 9, 10: the display of a memory element is a representation output
  // (maximum a posteriori); deleting its record is a commitment. A completion candidate is a
  // representation output that needs no prior.
  //
  // A memory element is not shown iff the frames that reach it see through it more often than
  // they hit it (10b); with no hit and no see-through it is not shown iff the majority of the
  // readings that reach it are cut off by a present surface within the band (principle 11), and
  // otherwise it stays. Its record is deleted iff the evidence of an ended surface exceeds
  // ln((1 - alpha) / alpha): the frames of the session as one look (7), F = see-through frames of
  // n = hit and see-through frames, against the in-place distribution of its class.
  std::vector<uint8_t> fill(faces.size(), 0);
  std::vector<float> fill_error(faces.size(), 0.f);
  std::vector<std::array<float, 4>> fill_record(faces.size(), std::array<float, 4>{0.f, 0.f, 0.f, 0.f});
  std::vector<int32_t> history_slot(history.faces.size(), -1);
  size_t num_fill = 0, memory_faces_kept = 0, num_shadow = 0, num_inside = 0, num_blocked = 0,
         num_gone = 0, num_deleted = 0, num_hidden = 0;
  uint64_t total_sources = 0;
  auto& hidden = result.hidden;
  // The record of a kept memory face: h_e, v_e of this session's frames, rho_e, zeta_e.
  std::vector<std::array<float, 4>> history_record(history.faces.size(), std::array<float, 4>{0.f, 0.f, 0.f, 0.f});
  const bool labelled = history.labels.size() == history.vertices.size();
  const model::RoundModel neutral_rounds;
  const model::RoundModel& rounds = in.rounds ? *in.rounds : neutral_rounds;
  // The faces of the present surface: displayed unless seen through more often than hit (10b); a
  // face that is not displayed has its record deleted by the commitment of (7) over the frames as one
  // look and otherwise travels with the evidence state like a hidden memory element.
  std::vector<uint8_t> present_shown(Fp.size(), 1);
  std::vector<std::array<float, 4>> present_records(Fp.size(), std::array<float, 4>{0.f, 0.f, 0.f, 0.f});
  size_t num_present_hidden = 0, num_present_deleted = 0;
  // A hidden record's vertices carry their attributes densely: the times and, where the memory
  // has them, colours and labels (a present face has no colour or label of its own).
  const auto hideRecord = [&](const std::array<Eigen::Vector3f, 3>& corners,
                              const std::array<TimeStamp, 3>& last, const std::array<TimeStamp, 3>& first,
                              const std::array<spark_dsg::Color, 3>& colors, const std::array<uint32_t, 3>& labels,
                              uint32_t physical, float error, const std::array<float, 4>& record) {
    const uint32_t base = static_cast<uint32_t>(hidden.vertices.size());
    for (size_t k = 0; k < 3; ++k) {
      hidden.vertices.push_back(corners[k]);
      hidden.stamps.push_back(last[k]);
      hidden.first_seen_stamps.push_back(first[k]);
      if (!history.colors.empty()) hidden.colors.push_back(colors[k]);
      if (labelled) hidden.labels.push_back(labels[k]);
    }
    hidden.faces.push_back({base, base + 1, base + 2});
    hidden.face_physical.push_back(physical);
    hidden.face_error.push_back(error);
    hidden.face_hits.push_back(record[0]);
    hidden.face_through.push_back(record[1]);
    hidden.face_rho.push_back(record[2]);
    hidden.face_zeta.push_back(record[3]);
  };
  for (size_t k = 0; k < elements.size(); ++k) {
    const auto& element = elements[k];
    const auto& tally = tallies[k];
    total_sources += tally.frames();
    bool keep = false;
    if (element.present) {
      const size_t f = element.face;
      const bool seen_through = tally.through > tally.hits;
      bool deleted = false;
      if (seen_through) {
        int cls = -1;
        if (element.physical > 0) {
          const auto found = in.identity_class.find(element.physical);
          if (found != in.identity_class.end()) cls = found->second;
        }
        const auto judgement = model::RoundModel::judge(
            rounds.classInPlace(cls, tally.meanPredicted()), tally.frames(), tally.through);
        deleted = judgement.commitment == model::Commitment::kCommitH;
      }
      const float range = present_range_count[f]
          ? static_cast<float>(present_range_sum[f] / present_range_count[f]) : static_cast<float>(tally.meanRange());
      present_records[f] = {static_cast<float>(tally.hits), static_cast<float>(tally.through), range,
                            static_cast<float>(psi.zeta)};
      present_shown[f] = !seen_through;
      num_gone += seen_through;
      num_present_deleted += deleted;
      if (seen_through && !deleted) {
        ++num_present_hidden;
        const auto& face = Fp[f];
        hideRecord({Vp[face[0]], Vp[face[1]], Vp[face[2]]}, {face_last[f], face_last[f], face_last[f]},
                   {face_first[f], face_first[f], face_first[f]}, {}, {}, element.physical,
                   present_error[f], present_records[f]);
      }
      continue;
    }
    if (element.historical) {
      const bool seen_through = tally.through > tally.hits;
      const bool blocked = tally.frames() == 0 && tally.blockedMajority();
      bool deleted = false;
      if (seen_through) {
        int cls = -1;
        if (element.physical > 0) {
          const auto found = in.identity_class.find(element.physical);
          if (found != in.identity_class.end()) cls = found->second;
        } else if (labelled) {
          cls = static_cast<int>(history.labels[history.faces[element.face][0]]);
        }
        const auto judgement = model::RoundModel::judge(
            rounds.classInPlace(cls, tally.meanPredicted()), tally.frames(), tally.through);
        deleted = judgement.commitment == model::Commitment::kCommitH;
      }
      num_gone += seen_through;
      num_deleted += deleted;
      num_shadow += shadow[k];
      num_inside += inside[k];
      num_blocked += blocked;
      keep = !seen_through && !shadow[k] && !inside[k] && !blocked;
      if (!deleted) {
        // The record (15b): the frames of this session that hit it and saw through it, the distance
        // and the scale of the session that made it. A kept element keeps its record; one that is
        // not shown travels in the evidence state.
        const std::array<float, 4> record{static_cast<float>(tally.hits), static_cast<float>(tally.through),
                                          element.record_range, element.record_zeta};
        history_record[element.face] = record;
        if (!keep) {
          const auto& face = history.faces[element.face];
          std::array<Eigen::Vector3f, 3> corners;
          std::array<TimeStamp, 3> last{}, first{};
          std::array<spark_dsg::Color, 3> colors{};
          std::array<uint32_t, 3> labels{};
          for (size_t c = 0; c < 3; ++c) {
            corners[c] = history.vertices[face[c]];
            if (!history.stamps.empty()) last[c] = history.stamps[face[c]];
            if (!history.first_seen_stamps.empty()) first[c] = history.first_seen_stamps[face[c]];
            if (!history.colors.empty()) colors[c] = history.colors[face[c]];
            if (labelled) labels[c] = history.labels[face[c]];
          }
          hideRecord(corners, last, first, colors, labels, element.physical, element.error, record);
          ++num_hidden;
        }
      } else {
        keep = false;
      }
    } else {
      // README (14f): a face of the online map the present does not draw is kept iff a valid
      // reading puts the surface on it, within tau = max(h_o / 2, sigma_table).
      keep = tally.near_reading;
      fill_record[element.face] = {0.f, 0.f, static_cast<float>(tally.meanRange()),
                                   static_cast<float>(psi.zeta)};
    }
    if (!keep) continue;
    if (element.historical) {
      history_slot[element.face] = static_cast<int32_t>(element.slot);
      ++memory_faces_kept;
    } else {
      fill[element.face] = 1;
      fill_error[element.face] = tally.precision > 0 ? static_cast<float>(1.0 / std::sqrt(tally.precision)) : 0.f;
      ++num_fill;
    }
  }
  const size_t memory_faces_total = history.faces.size();
  report << ",\"surface_model\":\"unified_posterior_v3\",\"elements\":{\"count\":" << elements.size()
         << ",\"echoes\":" << total_sources << ",\"seen_through\":" << num_gone << ",\"deleted\":"
         << num_deleted << ",\"hidden_records\":" << num_hidden << ",\"present_hidden\":" << num_present_hidden
         << ",\"present_deleted\":" << num_present_deleted
         << ",\"shown_as_present\":" << num_shadow << ",\"inside\":" << num_inside
         << ",\"blocked_in_band\":" << num_blocked << ",\"state_retired\":" << state_retired << "}"
         << ",\"fill\":{\"candidates\":" << candidates.size() << ",\"kept\":" << num_fill << "}";
  timer.step("surface_loss", "elements=" + std::to_string(elements.size()) +
      " memory_kept=" + std::to_string(memory_faces_kept) + " fill=" + std::to_string(num_fill));

  // -------------------------------------------------------------- V_free, README (15b)
  // The free space this session observed, at the map resolution: the voxels a ray passes before its
  // first return, stopping one hit band in front of it (the band of (6e) at the largest registration
  // drift of the session, so that drift cannot paint an occupied voxel free).
  {
    const double voxel_size = in.scales.background_voxel;
    FreeSpaceRecords records(static_cast<float>(voxel_size));
    records.beginSession();
    // Computation budget: at most 256 evenly spaced frames, every 8th pixel of a frame.
    constexpr size_t kFreeSpaceFrames = 256;
    constexpr int kFreeSpaceStride = 8;
    const size_t frame_step = std::max<size_t>(1, frames.size() / kFreeSpaceFrames);
    std::vector<size_t> chosen;
    for (size_t i = 0; i < frames.size(); i += frame_step) chosen.push_back(i);
    std::mutex records_mutex;
    parallelFor(chosen.size(), threads, [&](size_t begin, size_t end) {
      FreeSpaceRecords local(static_cast<float>(voxel_size));
      local.beginSession();
      std::vector<uint16_t> range, ids;
      for (size_t c = begin; c < end; ++c) {
        const size_t i = chosen[c];
        frames.decode(i, range, ids);
        const FrameCam& camera = frames.cam(i);
        const TimeStamp stamp = frames.stamp(i);
        for (int y = kFreeSpaceStride / 2; y < H; y += kFreeSpaceStride) {
          for (int x = kFreeSpaceStride / 2; x < W; x += kFreeSpaceStride) {
            const size_t pixel = static_cast<size_t>(y) * W + x;
            if (!range[pixel]) continue;
            const double reading = range[pixel] * kUnit;
            const double margin = psi.bounds(reading, psi.sigmaEff(reading, 0.0, voxel_size, false),
                                             max_range).plus;
            const double length = reading - (std::isfinite(margin) ? margin : reading);
            if (!(length > 0.0)) continue;
            const Eigen::Vector3d origin = camera.t.cast<double>();
            const Eigen::Vector3d direction = camera.R.cast<double>() * pixel_direction[pixel];
            // Voxel traversal along the ray (Amanatides-Woo).
            std::array<int64_t, 3> cell;
            std::array<int, 3> step;
            std::array<double, 3> t_max, t_delta;
            for (size_t a = 0; a < 3; ++a) {
              cell[a] = static_cast<int64_t>(std::floor(origin[a] / voxel_size));
              step[a] = direction[a] > 0 ? 1 : -1;
              if (direction[a] == 0.0) {
                t_max[a] = std::numeric_limits<double>::infinity();
                t_delta[a] = std::numeric_limits<double>::infinity();
              } else {
                const double boundary = (static_cast<double>(cell[a]) + (step[a] > 0 ? 1.0 : 0.0)) * voxel_size;
                t_max[a] = (boundary - origin[a]) / direction[a];
                t_delta[a] = voxel_size / std::abs(direction[a]);
              }
            }
            while (true) {
              local.markFree(packElementCell(cell), stamp);
              const size_t a = t_max[0] <= t_max[1] ? (t_max[0] <= t_max[2] ? 0 : 2)
                                                    : (t_max[1] <= t_max[2] ? 1 : 2);
              if (t_max[a] > length) break;
              cell[a] += step[a];
              t_max[a] += t_delta[a];
            }
          }
        }
      }
      std::lock_guard<std::mutex> lock(records_mutex);
      records.merge(local);
    }, 1);
    report << ",\"free_space\":{\"voxels\":" << records.size() << ",\"frames\":" << chosen.size() << "}";
    timer.step("free_space", "voxels=" + std::to_string(records.size()));
    result.free_space = std::move(records);
  }

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
    if (!present_shown[f]) continue;  // (10b): seen through more often than hit
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
    // README (14a), appendix: observation extent is independent of appearance ownership.
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
    const auto push_record = [&result](const std::array<float, 4>& record) {
      result.surface_records.hits.push_back(record[0]);
      result.surface_records.through.push_back(record[1]);
      result.surface_records.rho.push_back(record[2]);
      result.surface_records.zeta.push_back(record[3]);
    };
    for (const auto f : kept) {
      result.surface_error.push_back(fill_error[f]);
      push_record(fill_record[f]);
      out.faces.push_back({static_cast<size_t>(kept_new[faces[f][0] - slot.begin]),
                           static_cast<size_t>(kept_new[faces[f][1] - slot.begin]),
                           static_cast<size_t>(kept_new[faces[f][2] - slot.begin])});
    }
    for (const auto f : present) {
      result.surface_error.push_back(new_error[f]);
      push_record(f >= Fp.size() ? history_record[f - Fp.size()] : present_records[f]);
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
  result.psi = psi;
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
