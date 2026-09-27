#include "khronos/backend/reconciliation/session_refusion.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
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

// One mesh of the final map: the background or one current object mesh.
struct Slot {
  bool background = false;
  NodeId node = 0;
  spark_dsg::Mesh* mesh = nullptr;
  KhronosObjectAttributes* attrs = nullptr;
  uint32_t begin = 0, end = 0;  // global vertex range
  size_t physical = 0;          // node physical id (0: background / none)
  float half = 0.f, trunc = 0.f;
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

}  // namespace

SessionRefusion::Result SessionRefusion::apply(DynamicSceneGraph& dsg, const Inputs& in) const {
  Result result;
  StepTimer timer;
  if (!in.frames || in.frames->empty() || !in.camera.valid()) {
    LOG(WARNING) << "[SessionRefusion] no archived frames; skipped.";
    return result;
  }
  const double voxel = decimal(in.scales.object_voxel);
  if (!(voxel > 0.0) || !(in.scales.background_voxel > 0.f)) {
    LOG(ERROR) << "[SessionRefusion] needs an absolute object voxel (> 0) and background scales; "
                  "skipped.";
    return result;
  }
  auto& frames = *in.frames;
  const double trunc = 2.0 * voxel;  // object truncation = 2 voxels
  const float v_f = static_cast<float>(voxel), T_f = static_cast<float>(trunc);
  const float h_obj = 0.5f * v_f, T_obj = T_f;
  const float h_bg = static_cast<float>(0.5 * decimal(in.scales.background_voxel));
  const float T_bg = static_cast<float>(decimal(in.scales.background_truncation));
  const float s_session = std::max(0.f, in.depth_scale);
  const int threads = std::max(1, config.num_threads);
  const auto& K = in.camera;
  const int W = static_cast<int>(K.width), H = static_cast<int>(K.height);
  const size_t num_pixels = static_cast<size_t>(W) * H;
  const Projector project{K.fx, K.fy, K.cx, K.cy, W, H};
  std::vector<FrameCam> cams(frames.size());
  for (size_t i = 0; i < frames.size(); ++i) {
    cams[i].R = frames[i].world_T_sensor.linear().cast<float>();
    cams[i].t = frames[i].world_T_sensor.translation().cast<float>();
  }
  std::stringstream report;
  report << "{\"frames\":" << frames.size() << ",\"voxel\":" << voxel << ",\"truncation\":" << trunc
         << ",\"depth_scale\":" << in.depth_scale;

  // ------------------------------------------------------------------ final map
  std::vector<Slot> slots;
  std::vector<Eigen::Vector3f> pos;
  std::vector<uint8_t> memory;
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
    bg.half = h_bg;
    bg.trunc = T_bg;
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
      slot.half = h_obj;
      slot.trunc = T_obj;
      addSlot(slot, attrs->mesh);
    }
  }
  const size_t num_vertices = pos.size();
  memory.assign(num_vertices, 0);
  if (in.is_memory) {
    parallelFor(num_vertices, threads, [&](size_t b, size_t e) {
      for (size_t i = b; i < e; ++i) memory[i] = in.is_memory(pos[i]) ? 1 : 0;
    });
  }
  const size_t num_memory = std::count(memory.begin(), memory.end(), 1);
  timer.step("gather", "slots=" + std::to_string(slots.size()) + " vertices=" +
                           std::to_string(num_vertices) + " faces=" +
                           std::to_string(faces.size()) + " memory=" + std::to_string(num_memory));
  report << ",\"map\":{\"slots\":" << slots.size() << ",\"vertices\":" << num_vertices
         << ",\"faces\":" << faces.size() << ",\"memory_vertices\":" << num_memory << "}";

  if (!in.dump_dir.empty()) {
    std::filesystem::create_directories(in.dump_dir);
    FrameArchive::save(in.dump_dir + "/archive_uncut.kfa", frames, K);
  }

  // Nearest reaching view of points over the frames (refusion.py k_reach):
  // inview = frames projecting p onto a valid reading; reached = reading not
  // more than one truncation in front of p; q = min range over reached frames.
  auto reach = [&](const std::vector<Eigen::Vector3f>& P, const std::vector<float>& trunc_of,
                   std::vector<uint32_t>* inview, std::vector<float>& qmin,
                   std::vector<Eigen::Vector3f>* cam) {
    const size_t n = P.size();
    qmin.assign(n, kInf);
    if (inview) inview->assign(n, 0);
    if (cam) cam->assign(n, Eigen::Vector3f::Zero());
    for (size_t i = 0; i < frames.size(); ++i) {
      const auto& rng = frames[i].range_mm;
      const FrameCam& c = cams[i];
      parallelFor(n, threads, [&](size_t b, size_t e) {
        for (size_t j = b; j < e; ++j) {
          int u, v;
          float q;
          if (!project(P[j], c, u, v, q)) continue;
          const uint16_t d = rng[static_cast<size_t>(v) * W + u];
          if (!d) continue;
          if (inview) ++(*inview)[j];
          const float r = d * 1e-3f - q;
          if (r >= -trunc_of[j] && q < qmin[j]) {
            qmin[j] = q;
            if (cam) (*cam)[j] = c.t;
          }
        }
      }, 8192);
    }
  };

  // ---------------------------------------------- memory attributes for the next session
  {
    result.next_session.resize(num_vertices);
    std::vector<Eigen::Vector3f> own_pos;
    std::vector<float> own_trunc;
    std::vector<uint32_t> own_index;
    size_t copied = 0;
    for (size_t i = 0; i < num_vertices; ++i) {
      const Slot& slot = slots[vslot[i]];
      auto& r = result.next_session[i];
      r.x = pos[i].x();
      r.y = pos[i].y();
      r.z = pos[i].z();
      r.label = static_cast<uint32_t>(slot.physical);
      r.layer = slot.background ? 0 : 1;
      if (memory[i]) {
        const auto* prev =
            in.previous ? in.previous->find(pos[i], config.memory_match_distance) : nullptr;
        r.q = prev ? prev->q : kInf;
        r.s = prev ? prev->s : 0.f;
        copied += prev != nullptr;
      } else {
        own_pos.push_back(pos[i]);
        own_trunc.push_back(slot.trunc);
        own_index.push_back(static_cast<uint32_t>(i));
        r.s = in.depth_scale;
      }
    }
    std::vector<float> q_own;
    reach(own_pos, own_trunc, nullptr, q_own, nullptr);
    size_t reached = 0;
    for (size_t k = 0; k < own_index.size(); ++k) {
      result.next_session[own_index[k]].q = q_own[k];
      reached += std::isfinite(q_own[k]);
    }
    timer.step("next_session_attributes", "own=" + std::to_string(own_index.size()) +
                                              " own_reached=" + std::to_string(reached) +
                                              " memory_copied=" + std::to_string(copied));
    report << ",\"next_session\":{\"records\":" << num_vertices << ",\"own\":" << own_index.size()
           << ",\"own_reached\":" << reached << ",\"memory_copied\":" << copied << "}";
  }

  // ------------------------------------------------ step 1 / 1b: current-state starts
  report << ",\"state_starts\":[";
  bool first_entry = true;
  size_t total_cut = 0, total_stale = 0;
  for (const auto& [label, t_L] : in.state_starts) {
    std::vector<size_t> before;
    for (size_t i = 0; i < frames.size(); ++i) {
      if (frames[i].stamp < t_L) before.push_back(i);
    }
    std::atomic<size_t> cut{0}, stale{0};
    // Step 1: the object's own pixels before its current state began.
    parallelFor(before.size(), threads, [&](size_t b, size_t e) {
      for (size_t k = b; k < e; ++k) {
        auto& f = frames[before[k]];
        size_t start = 0, local = 0;
        for (const auto& run : f.instances) {
          const size_t end = std::min<size_t>(run.end, num_pixels);
          if (run.id == label) {
            for (size_t p = start; p < end; ++p) {
              if (f.range_mm[p]) {
                f.range_mm[p] = 0;
                ++local;
              }
            }
          }
          start = std::max(start, end);
        }
        cut += local;
      }
    }, 1);
    // Step 1b: free space seen through the object's current mesh before t_L.
    std::vector<uint32_t> subset;
    size_t mesh_vertices = 0;
    for (uint32_t f = 0; f < faces.size(); ++f) {
      const Slot& s = slots[fslot[f]];
      if (!s.background && s.physical == label) subset.push_back(f);
    }
    for (const auto& s : slots) {
      if (!s.background && s.physical == label) mesh_vertices += s.end - s.begin;
    }
    if (!subset.empty() && !before.empty()) {
      const TriangleGrid grid(pos, faces, &subset, 4.f * v_f);
      const Eigen::AlignedBox3f box = grid.bounds();
      std::vector<Eigen::Vector3f> corners;
      for (int k = 0; k < 8; ++k) {
        corners.emplace_back((k & 1) ? box.max().x() : box.min().x(),
                             (k & 2) ? box.max().y() : box.min().y(),
                             (k & 4) ? box.max().z() : box.min().z());
      }
      parallelFor(before.size(), threads, [&](size_t b, size_t e) {
        for (size_t k = b; k < e; ++k) {
          const size_t i = before[k];
          auto& f = frames[i];
          const FrameCam& c = cams[i];
          int u0 = 0, u1 = W - 1, v0 = 0, v1 = H - 1;
          bool all_front = true;
          float umin = kInf, umax = -kInf, vmin = kInf, vmax = -kInf;
          for (const auto& corner : corners) {
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
              uint16_t& d = f.range_mm[static_cast<size_t>(v) * W + u];
              if (!d) continue;
              const Eigen::Vector3f ray =
                  Eigen::Vector3f((u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.f).normalized();
              float t_hit = 0.f;
              uint32_t hit_face = 0;
              if (grid.firstHit(c.t, c.R * ray, t_hit, hit_face) && t_hit < d * 1e-3f - T_f) {
                d = 0;
                ++local;
              }
            }
          }
          stale += local;
        }
      }, 1);
    }
    total_cut += cut;
    total_stale += stale;
    LOG(INFO) << "[SessionRefusion] state_start id=" << label << " t_L=" << t_L
              << " frames_before=" << before.size() << " cut_pixels=" << cut.load()
              << " stale_pixels=" << stale.load() << " mesh_faces=" << subset.size()
              << " mesh_vertices=" << mesh_vertices;
    report << (first_entry ? "" : ",") << "{\"id\":" << label << ",\"t_L\":" << t_L
           << ",\"frames_before\":" << before.size() << ",\"cut_pixels\":" << cut.load()
           << ",\"stale_pixels\":" << stale.load() << ",\"mesh_faces\":" << subset.size() << "}";
    first_entry = false;
  }
  report << "]";
  timer.step("cut_and_stale", "objects=" + std::to_string(in.state_starts.size()) +
                                  " cut=" + std::to_string(total_cut) +
                                  " stale=" + std::to_string(total_stale));

  // ------------------------------------------------------------ step 2: TSDF + MC
  std::vector<Eigen::Vector3f> Vp;
  std::vector<Face3> Fp;
  {
    PresentTsdf tsdf(voxel, trunc, threads);
    const PresentTsdf::Camera cam{K.width, K.height, K.fx, K.fy, K.cx, K.cy};
    const std::vector<float> mult = PresentTsdf::rayNorm(cam);
    std::vector<float> depth(num_pixels);
    for (size_t i = 0; i < frames.size(); ++i) {
      PresentTsdf::depthFromRange(cam, frames[i].range_mm, depth);
      tsdf.integrate(cam, frames[i].world_T_sensor, depth, mult);
      if ((i + 1) % 200 == 0) {
        LOG(INFO) << "[SessionRefusion] tsdf frames=" << (i + 1) << "/" << frames.size()
                  << " units=" << tsdf.numUnits();
      }
    }
    const size_t units = tsdf.numUnits(), bytes = tsdf.numBytes();
    tsdf.extractMesh(Vp, Fp);
    timer.step("tsdf_mc", "units=" + std::to_string(units) + " tsdf_mb=" +
                              std::to_string(bytes >> 20) + " present_vertices=" +
                              std::to_string(Vp.size()) + " present_faces=" +
                              std::to_string(Fp.size()));
    report << ",\"present\":{\"units\":" << units << ",\"vertices\":" << Vp.size()
           << ",\"faces\":" << Fp.size();
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
    std::vector<Eigen::Vector3f> P, N;
    for (size_t j = 0; j < Vp.size(); j += config.noise_vertex_stride) {
      const double len = vn[j].norm();
      P.push_back(Vp[j]);
      N.push_back(len > 0 ? Eigen::Vector3f((vn[j] / len).cast<float>()) : Eigen::Vector3f::Zero());
    }
    const size_t nb = config.num_bins;
    const size_t nh = static_cast<size_t>(std::floor(trunc / config.histogram_resolution + 1e-9)) + 1;
    std::vector<std::vector<int64_t>> hist(nb, std::vector<int64_t>(nh, 0));
    std::mutex hist_mutex;
    for (size_t i = 0; i < frames.size(); ++i) {
      const auto& rng = frames[i].range_mm;
      const FrameCam& c = cams[i];
      parallelFor(P.size(), threads, [&](size_t b, size_t e) {
        std::vector<int64_t> local(nb * nh, 0);
        bool any = false;
        for (size_t j = b; j < e; ++j) {
          int u, v;
          float q;
          if (!project(P[j], c, u, v, q)) continue;
          if (N[j].dot(c.t - P[j]) <= 0.f) continue;
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
      }, 8192);
    }
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
    report << "]";
    timer.step("noise", "sigma_cm=[" + ss.str() + "]");
  }
  auto tauOf = [&](float half, float q) {
    const size_t bin = std::min(config.num_bins - 1,
                                static_cast<size_t>(std::max(0.f, q) / config.range_bin));
    return std::max(half, sigma[bin]);
  };

  // ------------------------------------------------------------ step 4: face labels
  std::vector<uint16_t> ids;  // compact index -> physical id (ascending, ids[0] = 0)
  std::vector<uint16_t> compact(65536, 0);
  {
    std::vector<uint8_t> seen(65536, 0);
    seen[0] = 1;
    for (const auto& f : frames)
      for (const auto& run : f.instances) seen[run.id] = 1;
    for (size_t id = 0; id < seen.size(); ++id) {
      if (seen[id]) {
        compact[id] = static_cast<uint16_t>(ids.size());
        ids.push_back(static_cast<uint16_t>(id));
      }
    }
  }
  const size_t nlab = ids.size();
  std::vector<int32_t> lab(Fp.size(), -1);  // compact label index
  {
    std::vector<uint16_t> votes(Fp.size() * nlab, 0);
    std::vector<uint16_t> pixel_label(num_pixels);
    for (size_t i = 0; i < frames.size(); ++i) {
      // Decode this frame's physical ids to compact indices.
      {
        size_t start = 0;
        for (const auto& run : frames[i].instances) {
          const size_t end = std::min<size_t>(run.end, num_pixels);
          if (end > start) std::fill(pixel_label.begin() + start, pixel_label.begin() + end, compact[run.id]);
          start = std::max(start, end);
        }
        if (start < num_pixels) std::fill(pixel_label.begin() + start, pixel_label.end(), 0);
      }
      const auto& rng = frames[i].range_mm;
      const FrameCam& c = cams[i];
      parallelFor(Fp.size(), threads, [&](size_t b, size_t e) {
        for (size_t j = b; j < e; ++j) {
          int u, v;
          float q;
          if (!project(centroid[j], c, u, v, q)) continue;
          if (face_normal[j].dot(c.t - centroid[j]) <= 0.f) continue;
          const size_t pix = static_cast<size_t>(v) * W + u;
          const uint16_t d = rng[pix];
          if (!d) continue;
          if (std::abs(d * 1e-3f - q) <= tauOf(h_obj, q)) ++votes[j * nlab + pixel_label[pix]];
        }
      }, 8192);
    }
    size_t measured = 0;
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
        ++measured;
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
    size_t unreached = 0, propagated = 0;
    for (auto& l : lab) {
      if (l < 0) {
        ++unreached;
        l = 0;
      }
    }
    propagated = nF - measured - unreached;
    std::map<uint16_t, size_t> per_label;
    for (const auto l : lab) ++per_label[ids[l]];
    std::stringstream ss;
    report << ",\"labels\":{\"faces\":" << nF << ",\"measured\":" << measured
           << ",\"propagated\":" << propagated << ",\"unreached\":" << unreached
           << ",\"rounds\":" << rounds << ",\"mean_votes\":" << (measured ? mean_votes / measured : 0)
           << ",\"purity\":" << (measured ? purity / measured : 0) << ",\"per_label\":{";
    bool first = true;
    for (const auto& [id, n] : per_label) {
      report << (first ? "" : ",") << "\"" << id << "\":" << n;
      ss << " " << id << ":" << n;
      first = false;
    }
    report << "}}}";
    timer.step("labels", "measured=" + std::to_string(measured) + " propagated=" +
                             std::to_string(propagated) + " unreached=" +
                             std::to_string(unreached) + " rounds=" + std::to_string(rounds) +
                             " per_label=" + ss.str());
  }
  // Physical id per present face.
  std::vector<uint32_t> face_id(Fp.size());
  for (size_t f = 0; f < Fp.size(); ++f) face_id[f] = ids[lab[f]];
  if (!in.dump_dir.empty()) {
    std::vector<int32_t> fl(face_id.begin(), face_id.end());
    writePly(in.dump_dir + "/present.ply", Vp, Fp, &fl);
  }

  // ------------------------------------------------------------ step 6: memory rules
  std::vector<uint8_t> memory_face(faces.size(), 0);
  for (size_t f = 0; f < faces.size(); ++f) {
    memory_face[f] = memory[faces[f][0]] && memory[faces[f][1]] && memory[faces[f][2]];
  }
  std::vector<uint8_t> retire(num_vertices, 0);
  size_t retired_a = 0, retired_b = 0, retired_a_bg = 0, retired_b_bg = 0;
  size_t mem_rule_vertices = 0, mem_reached = 0, mem_inview = 0;
  {
    std::vector<uint8_t> in_mface(num_vertices, 0);
    for (size_t f = 0; f < faces.size(); ++f) {
      if (!memory_face[f]) continue;
      in_mface[faces[f][0]] = in_mface[faces[f][1]] = in_mface[faces[f][2]] = 1;
    }
    std::vector<uint32_t> mv;
    for (uint32_t i = 0; i < num_vertices; ++i) {
      if (in_mface[i]) mv.push_back(i);
    }
    mem_rule_vertices = mv.size();
    std::vector<Eigen::Vector3f> P(mv.size());
    std::vector<float> half(mv.size()), T_l(mv.size()), qprev(mv.size(), kInf), sprev(mv.size(), 0.f);
    std::vector<uint32_t> mlabel(mv.size(), 0);
    size_t with_prev = 0;
    for (size_t k = 0; k < mv.size(); ++k) {
      const Slot& s = slots[vslot[mv[k]]];
      P[k] = pos[mv[k]];
      half[k] = s.half;
      T_l[k] = s.trunc;
      mlabel[k] = static_cast<uint32_t>(s.physical);
      if (in.previous) {
        if (const auto* prev = in.previous->find(P[k], config.memory_match_distance)) {
          qprev[k] = prev->q;
          sprev[k] = std::max(0.f, prev->s);
          ++with_prev;
        }
      }
    }
    std::vector<uint32_t> inview;
    std::vector<float> qB;
    std::vector<Eigen::Vector3f> camB;
    reach(P, T_l, &inview, qB, &camB);
    for (size_t k = 0; k < mv.size(); ++k) {
      mem_reached += std::isfinite(qB[k]);
      mem_inview += inview[k] > 0;
    }
    timer.step("memory_reach", "memory_vertices=" + std::to_string(mv.size()) +
                                   " reached=" + std::to_string(mem_reached) + " inview=" +
                                   std::to_string(mem_inview) + " with_previous_attributes=" +
                                   std::to_string(with_prev));

    const TriangleGrid present_grid(Vp, Fp, nullptr, 4.f * v_f);
    std::map<uint32_t, std::vector<uint32_t>> faces_of_label;
    for (uint32_t f = 0; f < Fp.size(); ++f) {
      if (face_id[f] > 0) faces_of_label[face_id[f]].push_back(f);
    }
    std::map<uint32_t, std::unique_ptr<TriangleGrid>> label_grid;
    for (const auto& [L, list] : faces_of_label) {
      label_grid[L] = std::make_unique<TriangleGrid>(Vp, Fp, &list, 4.f * v_f);
    }
    // Unit face normals (sign only) of the present.
    std::vector<Eigen::Vector3f> dirs(config.inside_rays);
    for (int i = 0; i < config.inside_rays; ++i) {
      const double phi = std::acos(1.0 - 2.0 * (i + 0.5) / config.inside_rays);
      const double th = M_PI * (1.0 + std::sqrt(5.0)) * (i + 0.5);
      dirs[i] = Eigen::Vector3f(static_cast<float>(std::cos(th) * std::sin(phi)),
                                static_cast<float>(std::sin(th) * std::sin(phi)),
                                static_cast<float>(std::cos(phi)));
    }
    std::vector<uint8_t> rule_a(mv.size(), 0), rule_b(mv.size(), 0);
    parallelFor(mv.size(), threads, [&](size_t b, size_t e) {
      for (size_t k = b; k < e; ++k) {
        const bool reached = std::isfinite(qB[k]);
        const float q = reached ? qB[k] : 0.f;
        const float window = tauOf(half[k], q) + s_session * q +
                             (std::isfinite(qprev[k]) ? sprev[k] * qprev[k] : 0.f);
        const float two_h = 2.f * half[k];
        const float r_max = reached ? std::max(window, two_h) : two_h;
        float d = kInf;
        Eigen::Vector3f cp;
        uint32_t face = 0;
        if (!present_grid.closest(P[k], r_max, d, cp, face)) d = kInf;
        // (a) displaced copy of the present's surface.
        if (reached && d > two_h && d <= window && (cp - P[k]).dot(camB[k] - P[k]) > 0.f) {
          rule_a[k] = 1;
        }
        // (b) buried inside the present surface of its own label.
        if (mlabel[k] > 0 && inview[k] > 0 && d > two_h) {
          const auto it = label_grid.find(mlabel[k]);
          if (it != label_grid.end()) {
            int vin = 0, vout = 0;
            for (const auto& dir : dirs) {
              float t_hit;
              uint32_t hit;
              if (!it->second->firstHit(P[k], dir, t_hit, hit)) continue;
              if (face_normal[hit].dot(dir) > 0.f) ++vin;
              else ++vout;
            }
            if (vin > vout) rule_b[k] = 1;
          }
        }
      }
    }, 1024);
    for (size_t k = 0; k < mv.size(); ++k) {
      const bool bg = slots[vslot[mv[k]]].background;
      if (rule_a[k]) {
        ++retired_a;
        retired_a_bg += bg;
      }
      if (rule_b[k]) {
        ++retired_b;
        retired_b_bg += bg;
      }
      if (rule_a[k] || rule_b[k]) retire[mv[k]] = 1;
    }
    timer.step("memory_rules", "a=" + std::to_string(retired_a) + " (bg " +
                                   std::to_string(retired_a_bg) + ") b=" +
                                   std::to_string(retired_b) + " (bg " +
                                   std::to_string(retired_b_bg) + ")");
    report << ",\"memory\":{\"rule_vertices\":" << mv.size() << ",\"reached\":" << mem_reached
           << ",\"inview\":" << mem_inview << ",\"with_previous_attributes\":" << with_prev
           << ",\"retired_a\":" << retired_a << ",\"retired_a_bg\":" << retired_a_bg
           << ",\"retired_b\":" << retired_b << ",\"retired_b_bg\":" << retired_b_bg << "}";
    if (!in.dump_dir.empty()) {
      std::ofstream out(in.dump_dir + "/memory_rules.bin", std::ios::binary);
      const uint64_t n = mv.size();
      out.write(reinterpret_cast<const char*>(&n), sizeof(n));
      for (size_t k = 0; k < mv.size(); ++k) {
        const float rec[6] = {P[k].x(), P[k].y(), P[k].z(), qB[k], qprev[k],
                              static_cast<float>(inview[k])};
        const uint32_t flags[2] = {mlabel[k], static_cast<uint32_t>(rule_a[k] | (rule_b[k] << 1) |
                                                                   (slots[vslot[mv[k]]].background ? 4 : 0))};
        out.write(reinterpret_cast<const char*>(rec), sizeof(rec));
        out.write(reinterpret_cast<const char*>(flags), sizeof(flags));
      }
    }
  }

  // ------------------------------------------------------------ step 5: compose
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
  for (uint32_t f = 0; f < faces.size(); ++f) {
    if (!memory_face[f]) continue;
    if (retire[faces[f][0]] || retire[faces[f][1]] || retire[faces[f][2]]) continue;
    kept_of_slot[fslot[f]].push_back(f);
  }
  size_t guarded = 0, memory_faces_kept = 0, memory_faces_total = 0;
  for (const auto m : memory_face) memory_faces_total += m;
  std::stringstream slot_report;
  const TimeStamp stamp_cap = in.final_stamp;
  for (uint32_t s = 0; s < slots.size(); ++s) {
    Slot& slot = slots[s];
    auto& mesh = *slot.mesh;
    const auto& kept = kept_of_slot[s];
    const auto& present = present_of_slot[s];
    memory_faces_kept += kept.size();
    if (!slot.background && kept.empty() && present.empty()) {
      ++guarded;  // an object's surface is never removed entirely
      slot_report << (slot_report.tellp() > 0 ? "," : "") << "{\"node\":" << slot.node
                  << ",\"physical\":" << slot.physical << ",\"guard\":true,\"faces\":"
                  << mesh.numFaces() << "}";
      continue;
    }
    const size_t old_n = mesh.numVertices();
    // New vertex list: kept memory vertices (original order), then present vertices.
    std::vector<int64_t> mem_new(old_n, -1);
    for (const auto f : kept) {
      for (int k = 0; k < 3; ++k) mem_new[faces[f][k] - slot.begin] = 0;
    }
    std::vector<uint32_t> mem_old;
    for (size_t i = 0; i < old_n; ++i) {
      if (mem_new[i] >= 0) {
        mem_new[i] = static_cast<int64_t>(mem_old.size());
        mem_old.push_back(static_cast<uint32_t>(i));
      }
    }
    std::unordered_map<uint32_t, uint32_t> present_new;
    std::vector<uint32_t> present_old;
    for (const auto f : present) {
      for (int k = 0; k < 3; ++k) {
        const uint32_t pv = Fp[f][k];
        if (present_new.emplace(pv, static_cast<uint32_t>(mem_old.size() + present_old.size())).second) {
          present_old.push_back(pv);
        }
      }
    }
    const size_t new_n = mem_old.size() + present_old.size();
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
    spark_dsg::Mesh::Positions points(new_n);
    for (size_t k = 0; k < mem_old.size(); ++k) points[k] = mesh.points[mem_old[k]];
    for (size_t k = 0; k < present_old.size(); ++k) {
      const Eigen::Vector3f& p = Vp[present_old[k]];
      points[mem_old.size() + k] = slot.attrs ? slot.attrs->bounding_box.pointToBoxFrame(p) : p;
    }
    auto rebuild = [&](auto& values, bool clamp_stamp) {
      using T = typename std::decay_t<decltype(values)>::value_type;
      if (values.size() != old_n) {
        if (values.size() > new_n) values.resize(new_n);
        return;
      }
      std::vector<T> out(new_n);
      for (size_t k = 0; k < mem_old.size(); ++k) out[k] = values[mem_old[k]];
      for (size_t k = 0; k < present_old.size(); ++k) out[mem_old.size() + k] = values[nearest[k]];
      if constexpr (std::is_integral_v<T>) {
        if (clamp_stamp && stamp_cap > 0) {
          for (size_t k = mem_old.size(); k < new_n; ++k) {
            out[k] = std::min<T>(out[k], static_cast<T>(stamp_cap));
          }
        }
      }
      values = std::move(out);
    };
    rebuild(mesh.colors, false);
    rebuild(mesh.stamps, true);
    rebuild(mesh.first_seen_stamps, true);
    rebuild(mesh.labels, false);
    mesh.points = std::move(points);
    spark_dsg::Mesh::Faces new_faces;
    new_faces.reserve(kept.size() + present.size());
    for (const auto f : kept) {
      new_faces.push_back({static_cast<size_t>(mem_new[faces[f][0] - slot.begin]),
                           static_cast<size_t>(mem_new[faces[f][1] - slot.begin]),
                           static_cast<size_t>(mem_new[faces[f][2] - slot.begin])});
    }
    for (const auto f : present) {
      new_faces.push_back({static_cast<size_t>(present_new[Fp[f][0]]),
                           static_cast<size_t>(present_new[Fp[f][1]]),
                           static_cast<size_t>(present_new[Fp[f][2]])});
    }
    mesh.faces = std::move(new_faces);
    slot_report << (slot_report.tellp() > 0 ? "," : "") << "{\"node\":" << slot.node
                << ",\"physical\":" << slot.physical << ",\"background\":" << slot.background
                << ",\"old_vertices\":" << old_n << ",\"memory_faces_kept\":" << kept.size()
                << ",\"present_faces\":" << present.size() << ",\"vertices\":" << new_n << "}";
  }
  timer.step("compose", "memory_faces=" + std::to_string(memory_faces_total) + " kept=" +
                            std::to_string(memory_faces_kept) + " guarded_nodes=" +
                            std::to_string(guarded) + " present_unassigned=" +
                            std::to_string(present_unassigned));
  report << ",\"compose\":{\"memory_faces\":" << memory_faces_total
         << ",\"memory_faces_kept\":" << memory_faces_kept << ",\"guarded_nodes\":" << guarded
         << ",\"slots\":[" << slot_report.str() << "]}";
  const double total =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - timer.start).count();
  report << ",\"timings_s\":{" << timer.timings.str() << ",\"total\":" << total
         << "},\"rss_mb_end\":" << rssMb() << "}";
  result.report_json = report.str();
  std::stringstream summary;
  summary << "frames=" << frames.size() << " present_vertices=" << Vp.size()
          << " present_faces=" << Fp.size() << " cut=" << total_cut << " stale=" << total_stale
          << " memory_rule_vertices=" << mem_rule_vertices << " retired_a=" << retired_a
          << " retired_b=" << retired_b << " memory_faces_kept=" << memory_faces_kept << "/"
          << memory_faces_total << " guarded_nodes=" << guarded << " total_s=" << total;
  result.summary = summary.str();
  result.applied = true;
  if (!in.dump_dir.empty()) {
    std::ofstream(in.dump_dir + "/refusion_report.json") << result.report_json;
  }
  return result;
}

}  // namespace khronos
