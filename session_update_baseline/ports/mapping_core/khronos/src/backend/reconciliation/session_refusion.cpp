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
#include <numeric>
#include <sstream>
#include <thread>
#include <unordered_map>

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

/**
 * The session's frames as the present sees them: the archived range with the
 * pixels of step 1 (object L before t_L) and step 1b (`stale`) removed.
 */
class SessionFrames {
 public:
  SessionFrames(const std::vector<FrameArchive::Frame>& frames,
                const FrameArchive::Camera& camera,
                const std::map<size_t, TimeStamp>& state_starts)
      : frames_(frames),
        num_pixels_(static_cast<size_t>(camera.width) * camera.height),
        start_of_(kNumIds, 0),
        stale(frames.size()) {
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
                 << " cannot be decoded; it is skipped.";
      range.assign(num_pixels_, 0);
      ids.assign(num_pixels_, 0);
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
    for (const uint32_t p : stale[i]) range[p] = 0;
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

// Median-based noise scale per range bin (refusion.py noise_table).
std::vector<float> sigmaFromHistogram(const std::vector<std::vector<int64_t>>& hist,
                                      size_t min_samples,
                                      double resolution,
                                      std::vector<int64_t>* counts) {
  const size_t nb = hist.size();
  std::vector<double> sig(nb, std::numeric_limits<double>::quiet_NaN());
  if (counts) counts->assign(nb, 0);
  for (size_t b = 0; b < nb; ++b) {
    int64_t cnt = 0;
    for (const auto c : hist[b]) cnt += c;
    if (counts) (*counts)[b] = cnt;
    if (cnt < static_cast<int64_t>(min_samples)) continue;
    const double half = 0.5 * static_cast<double>(cnt);
    int64_t cum = 0;
    for (size_t k = 0; k < hist[b].size(); ++k) {
      const int64_t prev = cum;
      cum += hist[b][k];
      if (static_cast<double>(cum) >= half) {  // numpy searchsorted(side='left')
        sig[b] = 1.4826 * (static_cast<double>(k) +
                           (half - static_cast<double>(prev)) /
                               static_cast<double>(std::max<int64_t>(hist[b][k], 1))) *
                 resolution;
        break;
      }
    }
  }
  std::vector<double> filled = sig;
  for (size_t b = 0; b < nb; ++b) {  // nearest populated bin, lower side first
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
  parallelFor(candidates.size(), threads, [&](size_t b, size_t e) {
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

SessionRefusion::Result SessionRefusion::apply(DynamicSceneGraph& dsg, const Inputs& in) const {
  Result result;
  StepTimer timer;
  if (!in.frames || in.frames->empty() || !in.camera.valid()) {
    LOG(WARNING) << "[SessionRefusion] no archived frames; skipped.";
    return result;
  }
  const double voxel = decimal(in.scales.object_voxel);
  if (!(voxel > 0.0)) {
    LOG(ERROR) << "[SessionRefusion] needs an absolute object voxel (> 0); skipped.";
    return result;
  }
  const double trunc = 2.0 * voxel;  // object truncation = 2 voxels
  const float v_f = static_cast<float>(voxel), T_f = static_cast<float>(trunc);
  const float h_obj = 0.5f * v_f;
  const int threads = std::max(1, config.num_threads);
  const auto& K = in.camera;
  const int W = static_cast<int>(K.width), H = static_cast<int>(K.height);
  const Projector project{K.fx, K.fy, K.cx, K.cy, W, H};
  SessionFrames frames(*in.frames, K, in.state_starts);
  const size_t num_pixels = frames.numPixels();
  std::stringstream report;
  report << "{\"frames\":" << frames.size() << ",\"voxel\":" << voxel << ",\"truncation\":" << trunc;

  // ------------------------------------------------------------------ final map
  std::vector<Slot> slots;
  std::vector<Eigen::Vector3f> pos;
  std::vector<uint32_t> vslot;
  std::vector<Face3> faces;
  std::vector<uint32_t> fslot;
  auto addSlot = [&](Slot slot, const spark_dsg::Mesh& mesh) {
    slot.begin = static_cast<uint32_t>(pos.size());
    const uint32_t sid = static_cast<uint32_t>(slots.size());
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
  std::vector<uint8_t> memory(num_vertices, 0);
  if (in.is_memory) {
    parallelFor(num_vertices, threads, [&](size_t b, size_t e) {
      for (size_t i = b; i < e; ++i) memory[i] = in.is_memory(pos[i]) ? 1 : 0;
    });
  }
  // Memory faces: all three vertices are memory; own faces: none is.
  std::vector<uint8_t> memory_face(faces.size(), 0), own_face(faces.size(), 0);
  for (size_t f = 0; f < faces.size(); ++f) {
    const int m = memory[faces[f][0]] + memory[faces[f][1]] + memory[faces[f][2]];
    memory_face[f] = m == 3;
    own_face[f] = m == 0;
  }
  const size_t num_memory = std::count(memory.begin(), memory.end(), 1);
  timer.step("gather", "slots=" + std::to_string(slots.size()) + " vertices=" +
                           std::to_string(num_vertices) + " faces=" +
                           std::to_string(faces.size()) + " memory=" + std::to_string(num_memory));
  report << ",\"map\":{\"slots\":" << slots.size() << ",\"vertices\":" << num_vertices
         << ",\"faces\":" << faces.size() << ",\"memory_vertices\":" << num_memory << "}";

  if (!in.dump_dir.empty()) {
    std::filesystem::create_directories(in.dump_dir);
    FrameArchive::save(in.dump_dir + "/archive.kfa", *in.frames, K);
  }

  // ------------------------------------------------ step 1 / 1b: current-state starts
  struct Start {
    size_t label = 0;
    TimeStamp t_L = 0;
    std::vector<uint32_t> mesh_faces;
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
      s.t_L = t_L;
      if (label < kNumIds) start_index[label] = static_cast<int32_t>(index);
      latest = std::max(latest, t_L);
      for (uint32_t f = 0; f < faces.size(); ++f) {
        const Slot& slot = slots[fslot[f]];
        if (!slot.background && slot.physical == label) s.mesh_faces.push_back(f);
      }
      if (!s.mesh_faces.empty()) {
        s.grid = std::make_unique<TriangleGrid>(pos, faces, &s.mesh_faces, 4.f * v_f);
        const Eigen::AlignedBox3f box = s.grid->bounds();
        for (int c = 0; c < 8; ++c) {
          s.corners.emplace_back((c & 1) ? box.max().x() : box.min().x(),
                                 (c & 2) ? box.max().y() : box.min().y(),
                                 (c & 4) ? box.max().z() : box.min().z());
        }
      }
      ++index;
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
        for (const auto& [id, n] : removed) starts[start_index[id]].cut += n;
        // Step 1b: free space seen through the object's current mesh before t_L.
        auto& stale = frames.stale[i];
        for (auto& s : starts) {
          if (!s.grid || frames.stamp(i) >= s.t_L) continue;
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
              if (s.grid->firstHit(c.t, c.R * ray, t_hit, hit_face) && t_hit < d * 1e-3f - T_f) {
                d = 0;
                stale.push_back(static_cast<uint32_t>(p));
                ++local;
              }
            }
          }
          s.stale += local;
        }
        stale.shrink_to_fit();
      }
    }, 1);
    report << ",\"state_starts\":[";
    for (size_t k = 0; k < starts.size(); ++k) {
      const Start& s = starts[k];
      total_cut += s.cut;
      total_stale += s.stale;
      LOG(INFO) << "[SessionRefusion] state_start id=" << s.label << " t_L=" << s.t_L
                << " frames_before=" << s.frames_before << " cut_pixels=" << s.cut.load()
                << " stale_pixels=" << s.stale.load() << " mesh_faces=" << s.mesh_faces.size();
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
  std::vector<uint8_t> seen(kNumIds, 0);  // physical ids present in the frames
  seen[0] = 1;
  {
    const PresentTsdf::Camera cam{K.width, K.height, K.fx, K.fy, K.cx, K.cy};
    const std::vector<float> mult = PresentTsdf::rayNorm(cam);
    std::vector<float> depth(num_pixels);
    size_t pixels = 0;
    frames.forEach([&](size_t i, const std::vector<uint16_t>& range,
                       const std::vector<uint16_t>& ids) {
      for (size_t p = 0; p < num_pixels; ++p) {
        seen[ids[p]] = 1;
        pixels += range[p] != 0;
      }
      PresentTsdf::depthFromRange(cam, range, depth);
      tsdf->integrate(cam, frames.pose(i), depth, mult);
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
  if (Fp.empty()) {
    report << "}}";
    LOG(ERROR) << "[SessionRefusion] empty present mesh; the final map is left unchanged.";
    result.report_json = report.str();
    return result;
  }

  // ------------------------------------------------------------ step 3: noise
  std::vector<Eigen::Vector3f> face_normal(Fp.size()), centroid(Fp.size());
  std::vector<float> sigma;
  {
    std::vector<Eigen::Vector3d> vn(Vp.size(), Eigen::Vector3d::Zero());
    for (size_t f = 0; f < Fp.size(); ++f) {
      const Eigen::Vector3d a = Vp[Fp[f][0]].cast<double>(), b = Vp[Fp[f][1]].cast<double>(),
                            c = Vp[Fp[f][2]].cast<double>();
      const Eigen::Vector3d n = (b - a).cross(c - a);
      vn[Fp[f][0]] += n;
      vn[Fp[f][1]] += n;
      vn[Fp[f][2]] += n;
      face_normal[f] = n.cast<float>();
      centroid[f] = ((a + b + c) / 3.0).cast<float>();
    }
    // Every present vertex (independent of the vertex order).
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
    sigma = sigmaFromHistogram(hist, config.min_bin_samples, config.histogram_resolution, &counts);
    std::stringstream ss;
    report << ",\"sigma_cm\":[";
    for (size_t b = 0; b < nb; ++b) {
      ss << (b ? " " : "") << std::round(sigma[b] * 1e4) / 100.0;
      report << (b ? "," : "") << sigma[b] * 100.f;
    }
    report << "],\"sigma_samples\":[";
    for (size_t b = 0; b < nb; ++b) report << (b ? "," : "") << counts[b];
    report << "]}";
    timer.step("noise", "vertices=" + std::to_string(Vp.size()) + " sigma_cm=[" + ss.str() + "]");
  }
  auto tauOf = [&](float q) {
    const size_t bin = std::min(config.num_bins - 1,
                                static_cast<size_t>(std::max(0.f, q) / config.range_bin));
    return std::max(h_obj, sigma[bin]);
  };

  // ------------------------------- steps 4 and 5: face labels and measured fill
  // Fill candidates: own faces whose centroid cube (voxel-centre lattice) has a
  // corner the present never integrated, i.e. where it extracts no surface.
  std::vector<uint32_t> candidates;
  for (uint32_t f = 0; f < faces.size(); ++f) {
    if (!own_face[f]) continue;
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
  std::vector<Eigen::Vector3f> candidate_centroid(candidates.size());
  for (size_t k = 0; k < candidates.size(); ++k) {
    const auto& f = faces[candidates[k]];
    candidate_centroid[k] = ((pos[f[0]].cast<double>() + pos[f[1]].cast<double>() +
                              pos[f[2]].cast<double>()) / 3.0).cast<float>();
  }
  std::vector<uint8_t> measured(candidates.size(), 0);
  // INSIDE candidates: vertices of memory faces on an object with a physical id.
  std::vector<std::pair<uint32_t, uint32_t>> memory_objects;
  {
    std::vector<uint8_t> in_memory_face(num_vertices, 0);
    for (size_t f = 0; f < faces.size(); ++f) {
      if (memory_face[f]) in_memory_face[faces[f][0]] = in_memory_face[faces[f][1]] = in_memory_face[faces[f][2]] = 1;
    }
    for (uint32_t i = 0; i < num_vertices; ++i) {
      const Slot& slot = slots[vslot[i]];
      if (in_memory_face[i] && !slot.background && slot.physical > 0) {
        memory_objects.emplace_back(i, static_cast<uint32_t>(slot.physical));
      }
    }
  }
  std::vector<uint8_t> in_view(memory_objects.size(), 0);

  std::vector<uint16_t> ids;  // compact index -> physical id (ascending, ids[0] = 0)
  std::vector<uint16_t> compact(kNumIds, 0);
  for (size_t id = 0; id < kNumIds; ++id) {
    if (seen[id]) {
      compact[id] = static_cast<uint16_t>(ids.size());
      ids.push_back(static_cast<uint16_t>(id));
    }
  }
  const size_t nlab = ids.size();
  std::vector<int32_t> lab(Fp.size(), -1);  // compact label index
  {
    std::vector<uint16_t> votes(Fp.size() * nlab, 0);
    frames.forEach([&](size_t i, const std::vector<uint16_t>& rng,
                       const std::vector<uint16_t>& pixel_id) {
      const FrameCam& c = frames.cam(i);
      parallelFor(Fp.size(), threads, [&](size_t b, size_t e) {
        for (size_t j = b; j < e; ++j) {
          int u, v;
          float q;
          if (!project(centroid[j], c, u, v, q)) continue;
          if (face_normal[j].dot(c.t - centroid[j]) <= 0.f) continue;
          const size_t pix = static_cast<size_t>(v) * W + u;
          const uint16_t d = rng[pix];
          if (!d) continue;
          if (std::abs(d * 1e-3f - q) <= tauOf(q)) ++votes[j * nlab + compact[pixel_id[pix]]];
        }
      }, 16384);
      parallelFor(candidates.size(), threads, [&](size_t b, size_t e) {
        for (size_t k = b; k < e; ++k) {
          int u, v;
          float q;
          if (measured[k] || !project(candidate_centroid[k], c, u, v, q)) continue;
          const uint16_t d = rng[static_cast<size_t>(v) * W + u];
          if (d && std::abs(d * 1e-3 - static_cast<double>(q)) <= tauOf(q)) measured[k] = 1;
        }
      }, 16384);
      parallelFor(memory_objects.size(), threads, [&](size_t b, size_t e) {
        for (size_t k = b; k < e; ++k) {
          int u, v;
          float q;
          if (in_view[k] || !project(pos[memory_objects[k].first], c, u, v, q)) continue;
          if (rng[static_cast<size_t>(v) * W + u]) in_view[k] = 1;
        }
      }, 16384);
    });
    size_t num_measured = 0;
    double purity = 0.0, mean_votes = 0.0;
    for (size_t f = 0; f < Fp.size(); ++f) {
      const uint16_t* row = &votes[f * nlab];
      uint32_t sum = 0, best = 0;
      int32_t arg = -1;
      for (size_t k = 0; k < nlab; ++k) {
        sum += row[k];
        if (row[k] > best) {
          best = row[k];
          arg = static_cast<int32_t>(k);
        }
      }
      if (sum > 0) {
        lab[f] = arg;
        ++num_measured;
        purity += static_cast<double>(best) / sum;
        mean_votes += sum;
      }
    }
    // Propagation over shared edges (refusion.py face_labels): edges of all
    // faces (0,1), (1,2), (2,0) concatenated, stable-sorted by key; consecutive
    // equal keys are paired (non-manifold edges pair consecutive faces).
    const size_t nF = Fp.size();
    const uint64_t nV1 = Vp.size() + 1;
    std::vector<std::pair<uint64_t, uint32_t>> edges;
    edges.reserve(3 * nF);
    for (int e = 0; e < 3; ++e) {
      for (size_t f = 0; f < nF; ++f) {
        uint64_t a = Fp[f][e], b = Fp[f][(e + 1) % 3];
        if (a > b) std::swap(a, b);
        edges.emplace_back(a * nV1 + b, static_cast<uint32_t>(f));
      }
    }
    std::stable_sort(edges.begin(), edges.end(),
                     [](const auto& x, const auto& y) { return x.first < y.first; });
    // Neighbour lists (CSR, with multiplicity) of the paired faces.
    std::vector<uint32_t> adj_begin(nF + 1, 0), adj;
    {
      std::vector<std::pair<uint32_t, uint32_t>> pairs;
      for (size_t k = 1; k < edges.size(); ++k) {
        if (edges[k].first == edges[k - 1].first) {
          pairs.emplace_back(edges[k - 1].second, edges[k].second);
        }
      }
      edges.clear();
      edges.shrink_to_fit();
      for (const auto& [a, b] : pairs) {
        ++adj_begin[a + 1];
        ++adj_begin[b + 1];
      }
      for (size_t f = 0; f < nF; ++f) adj_begin[f + 1] += adj_begin[f];
      adj.resize(adj_begin[nF]);
      std::vector<uint32_t> fill(adj_begin.begin(), adj_begin.end() - 1);
      for (const auto& [a, b] : pairs) {
        adj[fill[a]++] = b;
        adj[fill[b]++] = a;
      }
    }
    // Rounds: every unlabelled face with a labelled neighbour takes the
    // majority label of its labelled neighbours (ties -> smallest id),
    // simultaneously, until nothing changes.
    size_t rounds = 0;
    std::vector<uint32_t> cnt(nlab, 0);
    std::vector<std::pair<uint32_t, int32_t>> updates;
    while (true) {
      updates.clear();
      for (uint32_t f = 0; f < nF; ++f) {
        if (lab[f] >= 0) continue;
        bool any = false;
        for (uint32_t k = adj_begin[f]; k < adj_begin[f + 1]; ++k) {
          const int32_t l = lab[adj[k]];
          if (l >= 0) {
            ++cnt[l];
            any = true;
          }
        }
        if (!any) continue;
        uint32_t best = 0;
        int32_t arg = -1;
        for (uint32_t k = adj_begin[f]; k < adj_begin[f + 1]; ++k) {
          const int32_t l = lab[adj[k]];
          if (l < 0) continue;
          if (cnt[l] > best || (cnt[l] == best && l < arg)) {
            best = cnt[l];
            arg = l;
          }
        }
        for (uint32_t k = adj_begin[f]; k < adj_begin[f + 1]; ++k) {
          const int32_t l = lab[adj[k]];
          if (l >= 0) cnt[l] = 0;
        }
        updates.emplace_back(f, arg);
      }
      if (updates.empty()) break;
      for (const auto& [f, l] : updates) lab[f] = l;
      ++rounds;
    }
    size_t unreached = 0;
    for (auto& l : lab) {
      if (l < 0) {
        ++unreached;
        l = 0;
      }
    }
    const size_t propagated = nF - num_measured - unreached;
    std::map<uint16_t, size_t> per_label;
    for (const auto l : lab) ++per_label[ids[l]];
    std::stringstream ss;
    report << ",\"labels\":{\"faces\":" << nF << ",\"measured\":" << num_measured
           << ",\"propagated\":" << propagated << ",\"unreached\":" << unreached
           << ",\"rounds\":" << rounds
           << ",\"mean_votes\":" << (num_measured ? mean_votes / num_measured : 0)
           << ",\"purity\":" << (num_measured ? purity / num_measured : 0) << ",\"per_label\":{";
    bool first = true;
    for (const auto& [id, n] : per_label) {
      report << (first ? "" : ",") << "\"" << id << "\":" << n;
      ss << " " << id << ":" << n;
      first = false;
    }
    report << "}}";
    timer.step("labels", "measured=" + std::to_string(num_measured) + " propagated=" +
                             std::to_string(propagated) + " unreached=" +
                             std::to_string(unreached) + " rounds=" + std::to_string(rounds) +
                             " per_label=" + ss.str());
  }
  std::vector<uint8_t> fill(faces.size(), 0);
  size_t num_fill = 0;
  for (size_t k = 0; k < candidates.size(); ++k) {
    if (measured[k]) {
      fill[candidates[k]] = 1;
      ++num_fill;
    }
  }
  report << ",\"fill\":{\"candidates\":" << candidates.size() << ",\"kept\":" << num_fill << "}";
  timer.step("fill", "candidates=" + std::to_string(candidates.size()) +
                         " kept=" + std::to_string(num_fill));
  // Physical id per present face.
  std::vector<uint32_t> face_id(Fp.size());
  for (size_t f = 0; f < Fp.size(); ++f) face_id[f] = ids[lab[f]];
  if (!in.dump_dir.empty()) {
    std::vector<int32_t> fl(face_id.begin(), face_id.end());
    writePly(in.dump_dir + "/present.ply", Vp, Fp, &fl);
  }

  // ------------------------------------------------------------ step 6: memory
  // Memory stays as the online consolidation left it, except where INSIDE
  // retires a vertex (a memory face with a retired vertex is dropped).
  std::vector<std::pair<uint32_t, uint32_t>> inside_candidates;
  for (size_t k = 0; k < memory_objects.size(); ++k) {
    if (in_view[k]) inside_candidates.push_back(memory_objects[k]);
  }
  const std::vector<uint8_t> retired =
      insideMemory(pos, inside_candidates, Vp, Fp, face_id, face_normal, v_f, threads);
  const size_t num_retired = std::count(retired.begin(), retired.end(), 1);
  report << ",\"memory\":{\"object_vertices\":" << memory_objects.size()
         << ",\"in_view\":" << inside_candidates.size() << ",\"inside_retired\":" << num_retired
         << "}";
  timer.step("memory_inside", "object_memory_vertices=" + std::to_string(memory_objects.size()) +
                                  " in_view=" + std::to_string(inside_candidates.size()) +
                                  " retired=" + std::to_string(num_retired));

  // ------------------------------------------------------------ step 7: compose
  // Present faces: label 0 (or a label without a current node) -> background,
  // label L -> the node with physical id L (the largest if several).
  std::map<size_t, uint32_t> slot_of_label;
  for (uint32_t s = 0; s < slots.size(); ++s) {
    if (slots[s].background || slots[s].physical == 0) continue;
    const auto it = slot_of_label.find(slots[s].physical);
    if (it == slot_of_label.end() ||
        slots[s].end - slots[s].begin > slots[it->second].end - slots[it->second].begin) {
      slot_of_label[slots[s].physical] = s;
    }
  }
  int32_t bg_slot = -1;
  for (uint32_t s = 0; s < slots.size(); ++s) {
    if (slots[s].background) bg_slot = static_cast<int32_t>(s);
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
  for (uint32_t f = 0; f < faces.size(); ++f) {
    const bool kept_memory = memory_face[f] && !retired[faces[f][0]] && !retired[faces[f][1]] &&
                             !retired[faces[f][2]];
    memory_faces_total += memory_face[f];
    memory_faces_kept += kept_memory;
    if (kept_memory || fill[f]) kept_of_slot[fslot[f]].push_back(f);
  }
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
  size_t guarded = 0;
  std::stringstream slot_report;
  const TimeStamp stamp_cap = in.final_stamp;
  for (uint32_t s = 0; s < slots.size(); ++s) {
    const Slot& slot = slots[s];
    const auto& mesh = *slot.mesh;
    const auto& kept = kept_of_slot[s];
    const auto& present = present_of_slot[s];
    if (!slot.background && kept.empty() && present.empty()) {
      ++guarded;  // an object's surface is never removed entirely
      slot_report << (slot_report.tellp() > 0 ? "," : "") << "{\"node\":" << slot.node
                  << ",\"physical\":" << slot.physical << ",\"guard\":true,\"faces\":"
                  << mesh.numFaces() << "}";
      continue;
    }
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
        const uint32_t pv = Fp[f][k];
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
          if (search.search(Vp[present_old[k]], d_sq, idx)) nearest[k] = idx;
        }
      });
    }
    NewMesh out;
    out.slot = s;
    out.points.resize(new_n);
    for (size_t k = 0; k < kept_old.size(); ++k) out.points[k] = mesh.points[kept_old[k]];
    for (size_t k = 0; k < present_old.size(); ++k) {
      const Eigen::Vector3f& p = Vp[present_old[k]];
      out.points[kept_old.size() + k] = slot.attrs ? slot.attrs->bounding_box.pointToBoxFrame(p) : p;
    }
    auto rebuild = [&](const auto& values, auto& result, bool clamp_stamp) {
      using T = typename std::decay_t<decltype(values)>::value_type;
      if (values.size() != old_n) {
        result.assign(values.begin(), values.begin() + std::min(values.size(), new_n));
        return;
      }
      result.resize(new_n);
      for (size_t k = 0; k < kept_old.size(); ++k) result[k] = values[kept_old[k]];
      for (size_t k = 0; k < present_old.size(); ++k) {
        result[kept_old.size() + k] = values[nearest[k]];
      }
      if constexpr (std::is_integral_v<T>) {
        if (clamp_stamp && stamp_cap > 0) {
          for (size_t k = kept_old.size(); k < new_n; ++k) {
            result[k] = std::min<T>(result[k], static_cast<T>(stamp_cap));
          }
        }
      }
    };
    rebuild(mesh.colors, out.colors, false);
    rebuild(mesh.stamps, out.stamps, true);
    rebuild(mesh.first_seen_stamps, out.first_seen_stamps, true);
    rebuild(mesh.labels, out.labels, false);
    out.faces.reserve(kept.size() + present.size());
    for (const auto f : kept) {
      out.faces.push_back({static_cast<size_t>(kept_new[faces[f][0] - slot.begin]),
                           static_cast<size_t>(kept_new[faces[f][1] - slot.begin]),
                           static_cast<size_t>(kept_new[faces[f][2] - slot.begin])});
    }
    for (const auto f : present) {
      out.faces.push_back({static_cast<size_t>(present_new[Fp[f][0]]),
                           static_cast<size_t>(present_new[Fp[f][1]]),
                           static_cast<size_t>(present_new[Fp[f][2]])});
    }
    slot_report << (slot_report.tellp() > 0 ? "," : "") << "{\"node\":" << slot.node
                << ",\"physical\":" << slot.physical << ",\"background\":" << slot.background
                << ",\"old_vertices\":" << old_n << ",\"kept_faces\":" << kept.size()
                << ",\"present_faces\":" << present.size() << ",\"vertices\":" << new_n << "}";
    rebuilt.push_back(std::move(out));
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
  timer.step("compose", "memory_faces=" + std::to_string(memory_faces_total) + " kept=" +
                            std::to_string(memory_faces_kept) + " fill=" +
                            std::to_string(num_fill) + " guarded_nodes=" +
                            std::to_string(guarded) + " present_unassigned=" +
                            std::to_string(present_unassigned));
  report << ",\"compose\":{\"memory_faces\":" << memory_faces_total
         << ",\"memory_faces_kept\":" << memory_faces_kept << ",\"fill_faces\":" << num_fill
         << ",\"guarded_nodes\":" << guarded << ",\"slots\":[" << slot_report.str() << "]}";
  const double total =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - timer.start).count();
  report << ",\"timings_s\":{" << timer.timings.str() << ",\"total\":" << total
         << "},\"rss_mb_end\":" << rssMb() << "}";
  result.report_json = report.str();
  std::stringstream summary;
  summary << "frames=" << frames.size() << " present_vertices=" << Vp.size()
          << " present_faces=" << Fp.size() << " cut=" << total_cut << " stale=" << total_stale
          << " fill=" << num_fill << "/" << candidates.size() << " inside_retired="
          << num_retired << " memory_faces_kept="
          << memory_faces_kept << "/" << memory_faces_total << " guarded_nodes=" << guarded
          << " total_s=" << total;
  result.summary = summary.str();
  result.applied = true;
  if (!in.dump_dir.empty()) {
    std::ofstream(in.dump_dir + "/refusion_report.json") << result.report_json;
  }
  return result;
}

}  // namespace khronos
