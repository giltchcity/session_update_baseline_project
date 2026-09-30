// Offline replay of the session-end re-integration (SessionRefusion) without
// the mapper, for checks against the offline reference.
//
//   refusion_replay mesh ARCHIVE.kfa OUT.ply [VOXEL] [THREADS]
//       TSDF + marching cubes of all archived frames (no cut), as PLY.
//   refusion_replay map FINAL.4dmap.zpk MEMORY.4dmap.zpk ARCHIVE.kfa OUT.4dmap.zpk
//       [--tl id:stamp_ns|closed,...] [--scales BGV,BGT,OBJV] [--threads N] [--dump DIR]
//       [--shown PREV_FINAL.4dmap.zpk]
//       Runs the session-end update on the final snapshot of FINAL (the session's
//       chain_state: the object reasoning's final state) with the explicit
//       predecessor surface from MEMORY, or --shown when supplied. MEMORY '-' is
//       an initial session. The common surface importer retains identity, color,
//       time and source-error attributes; writes a one-snapshot map and the report
//       OUT.json.
//   refusion_replay export MAP.4dmap.zpk OUT.ply
//       The latest snapshot's current surfaces with per-face physical id and slot.
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <glog/logging.h>

#include <session_core/surface/frame_archive.h>
#include <session_core/surface/present_tsdf.h>
#include <session_core/surface/session_refusion.h>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <session_update_baseline/runtime/session_state.h>
#include <khronos/active_window/object_extraction/mesh_object_extractor.h>
#include <khronos/backend/update_khronos_objects_functor.h>
#include <khronos/utils/khronos_attribute_utils.h>

using namespace khronos;

namespace {

int meshMode(int argc, char** argv) {
  if (argc < 4) return 2;
  const double voxel = argc > 4 ? std::stod(argv[4]) : 0.02;
  const int threads = argc > 5 ? std::stoi(argv[5]) : 4;
  std::vector<FrameArchive::Frame> frames;
  FrameArchive::Camera K;
  if (!FrameArchive::load(argv[2], frames, K)) {
    std::cerr << "cannot load archive " << argv[2] << '\n';
    return 1;
  }
  const auto t0 = std::chrono::steady_clock::now();
  PresentTsdf tsdf(voxel, 2 * voxel, threads);
  const PresentTsdf::Camera cam{K.width, K.height, K.fx, K.fy, K.cx, K.cy};
  const std::vector<float> mult = PresentTsdf::rayNorm(cam);
  std::vector<float> depth;
  std::vector<uint16_t> range, ids;
  for (const auto& f : frames) {
    if (!f.decode(static_cast<size_t>(K.width) * K.height, range, ids)) return 1;
    PresentTsdf::depthFromRange(cam, range, depth);
    tsdf.integrate(cam, f.world_T_sensor, depth, mult);
  }
  const double t_fuse = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::vector<Eigen::Vector3f> V;
  std::vector<PresentTsdf::Face> F;
  tsdf.extractMesh(V, F);
  const double t_all = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "frames=" << frames.size() << " units=" << tsdf.numUnits() << " vertices=" << V.size()
            << " faces=" << F.size() << " fuse_s=" << t_fuse << " total_s=" << t_all << '\n';
  std::ofstream out(argv[3], std::ios::binary);
  out << "ply\nformat binary_little_endian 1.0\nelement vertex " << V.size()
      << "\nproperty float x\nproperty float y\nproperty float z\nelement face " << F.size()
      << "\nproperty list uchar uint vertex_indices\nend_header\n";
  for (const auto& p : V) out.write(reinterpret_cast<const char*>(p.data()), 12);
  const uint8_t three = 3;
  for (const auto& f : F) {
    out.write(reinterpret_cast<const char*>(&three), 1);
    out.write(reinterpret_cast<const char*>(f.data()), 12);
  }
  return 0;
}

// Use the same predecessor representation as the online continuation path.
bool loadSurface(const std::string& path, SessionRefusion::Surface& out) {
  auto surface_path = std::filesystem::path(path);
  const auto directory = surface_path.parent_path();
  if (surface_path.filename() == "chain_state.4dmap.zpk" &&
      std::filesystem::exists(directory / "shown_state.4dmap.zpk"))
    surface_path = directory / "shown_state.4dmap.zpk";
  const auto map = session_update::runtime::loadSessionMap(surface_path.string());
  if (!map || !map->numTimeSteps()) return false;
  out = SessionRefusion::fromDsg(*map->rawDsg(map->numTimeSteps() - 1));
  const auto replay_error = std::filesystem::path(surface_path.string() + ".surface_error.bin");
  const auto error_path = std::filesystem::exists(replay_error)
      ? replay_error : directory / "surface_error.bin";
  if (std::filesystem::exists(error_path))
    SessionRefusion::loadSurfaceError(error_path.string(), out);
  return true;
}

int mapMode(int argc, char** argv) {
  if (argc < 6) return 2;
  const std::string final_path = argv[2], memory_path = argv[3], archive_path = argv[4],
                    out_path = argv[5];
  SessionRefusion::Inputs inputs;
  SessionRefusion::Scales scales;
  scales.background_voxel = 0.05f;
  scales.background_truncation = 0.15f;
  scales.object_voxel = 0.02f;
  int threads = 4;
  std::string shown_path;
  for (int i = 6; i + 1 < argc; i += 2) {
    const std::string key = argv[i], value = argv[i + 1];
    if (key == "--tl") {
      std::stringstream ss(value);
      std::string item;
      while (std::getline(ss, item, ',')) {
        const auto colon = item.find(':');
        if (colon == std::string::npos || colon == 0 || colon + 1 == item.size())
          throw std::invalid_argument("State domain requires id:stamp or id:closed");
        const auto id = std::stoul(item.substr(0, colon));
        const auto state = item.substr(colon + 1);
        inputs.state_starts[id] = state == "closed" ? std::nullopt
            : std::optional<TimeStamp>(std::stoull(state));
        // --tl supplies an explicit changed/closed state in conditional replay.
        inputs.replaced_states.insert(id);
      }
    } else if (key == "--scales") {
      std::sscanf(value.c_str(), "%f,%f,%f", &scales.background_voxel,
                  &scales.background_truncation, &scales.object_voxel);
    } else if (key == "--threads") {
      threads = std::stoi(value);
    } else if (key == "--dump") {
      inputs.dump_dir = value;
    } else if (key == "--shown") {
      shown_path = value;
    } else if (key == "--prev-scales") {
      std::stringstream ss(value);
      std::string item;
      while (std::getline(ss, item, ',')) inputs.previous_depth_scales.push_back(std::stof(item));
    } else {
      std::cerr << "unknown option " << key << '\n';
      return 2;
    }
  }
  const auto final_map = session_update::runtime::loadSessionMap(final_path);
  if (!final_map || !final_map->numTimeSteps()) {
    std::cerr << "cannot load " << final_path << '\n';
    return 1;
  }
  const size_t last = final_map->numTimeSteps() - 1;
  const TimeStamp stamp = final_map->stamps()[last];
  auto edited = final_map->rawDsg(last)->clone();
  SessionRefusion::Surface shown;
  const std::string history_path = shown_path.empty() ? memory_path : shown_path;
  if (history_path != "-") {
    if (!loadSurface(history_path, shown)) {
      std::cerr << "cannot load " << history_path << '\n';
      return 1;
    }
    inputs.shown = &shown;
  }
  std::cout << "final snapshot " << last << " stamp " << stamp
            << ", history faces " << shown.faces.size() << '\n';
  std::vector<FrameArchive::Frame> frames;
  if (!FrameArchive::load(archive_path, frames, inputs.camera)) {
    std::cerr << "cannot load archive " << archive_path << '\n';
    return 1;
  }
  inputs.frames = &frames;
  scales.object_truncation = MeshObjectExtractor::objectTruncationDistance(scales.object_voxel);
  inputs.scales = scales;
  inputs.final_stamp = stamp;
  // README (8): the registry supplies each identity's domain. This replay has no registry, so
  // an object with current geometry in the supplied final map, and no --tl entry, is given the
  // whole session here, explicitly; the surface update itself treats omitted IDs as empty.
  if (edited->hasLayer(DsgLayers::OBJECTS)) {
    for (const auto& [node_id, node] : edited->getLayer(DsgLayers::OBJECTS).nodes()) {
      (void)node_id;
      const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
      if (!attrs || !hasCurrentObjectMesh(*attrs)) continue;
      if (const auto id = UpdateKhronosObjectsFunctor::physicalInstanceId(*attrs))
        inputs.state_starts.try_emplace(*id, TimeStamp{0});
    }
  }
  SessionRefusion::Config config;
  config.num_threads = threads;
  const SessionRefusion refusion(config);
  auto result = refusion.apply(*edited, inputs);
  std::cout << "applied=" << result.applied << " depth_scale=" << result.depth_scale << " " << result.summary << '\n';
  if (!result.applied) return 1;
  SpatioTemporalMap out(final_map->config);
  out.update(edited, stamp);
  const auto parent = std::filesystem::path(out_path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  if (!out.save(out_path)) {
    std::cerr << "cannot save " << out_path << '\n';
    return 1;
  }
  auto surface = SessionRefusion::fromDsg(*edited);
  surface.face_error = result.surface_error;
  SessionRefusion::saveSurfaceError(out_path + ".surface_error.bin", surface);
  std::ofstream report(out_path + ".json");
  report.exceptions(std::ios::badbit | std::ios::failbit);
  report << result.report_json << '\n';
  report.close();
  return 0;
}

}  // namespace

// export MAP OUT.ply: the latest snapshot's current surfaces in world coordinates, one
// PLY with per-face int properties "physical" (0 = background) and "slot" (0 =
// background, k = k-th current object node).
int exportMode(int argc, char** argv) {
  if (argc < 4) return 2;
  const auto map = session_update::runtime::loadSessionMap(argv[2]);
  if (!map || !map->numTimeSteps()) return 1;
  const auto dsg = map->rawDsg(map->numTimeSteps() - 1);
  std::vector<Eigen::Vector3f> V;
  std::vector<std::array<uint32_t, 3>> F;
  std::vector<int32_t> phys, slot;
  auto add = [&](const spark_dsg::Mesh& mesh, const KhronosObjectAttributes* attrs, int32_t p, int32_t s) {
    const uint32_t base = static_cast<uint32_t>(V.size());
    for (size_t i = 0; i < mesh.numVertices(); ++i) {
      V.push_back(attrs ? attrs->bounding_box.pointToWorldFrame(mesh.pos(i)) : mesh.pos(i));
    }
    const size_t n = mesh.numVertices();
    for (const auto& f : mesh.faces) {
      if (f[0] >= n || f[1] >= n || f[2] >= n) continue;
      F.push_back({base + static_cast<uint32_t>(f[0]), base + static_cast<uint32_t>(f[1]),
                   base + static_cast<uint32_t>(f[2])});
      phys.push_back(p);
      slot.push_back(s);
    }
  };
  if (dsg->hasMesh() && dsg->mesh()) add(*dsg->mesh(), nullptr, 0, 0);
  int32_t k = 0;
  if (dsg->hasLayer(DsgLayers::OBJECTS)) {
    for (const auto& [id, node] : dsg->getLayer(DsgLayers::OBJECTS).nodes()) {
      const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
      if (!attrs || !hasCurrentObjectMesh(*attrs)) continue;
      ++k;
      add(attrs->mesh, attrs,
          static_cast<int32_t>(UpdateKhronosObjectsFunctor::physicalInstanceId(*attrs).value_or(0)), k);
    }
  }
  std::ofstream out(argv[3], std::ios::binary);
  out << "ply\nformat binary_little_endian 1.0\nelement vertex " << V.size()
      << "\nproperty float x\nproperty float y\nproperty float z\nelement face " << F.size()
      << "\nproperty list uchar uint vertex_indices\nproperty int physical\nproperty int slot\nend_header\n";
  for (const auto& p : V) out.write(reinterpret_cast<const char*>(p.data()), 12);
  const uint8_t three = 3;
  for (size_t i = 0; i < F.size(); ++i) {
    out.write(reinterpret_cast<const char*>(&three), 1);
    out.write(reinterpret_cast<const char*>(F[i].data()), 12);
    out.write(reinterpret_cast<const char*>(&phys[i]), 4);
    out.write(reinterpret_cast<const char*>(&slot[i]), 4);
  }
  std::cout << "exported " << V.size() << " vertices, " << F.size() << " faces, " << k << " object slots\n";
  return 0;
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;
  const std::string mode = argc > 1 ? argv[1] : "";
  int status = 2;
  if (mode == "mesh") status = meshMode(argc, argv);
  if (mode == "map") status = mapMode(argc, argv);
  if (mode == "export") status = exportMode(argc, argv);
  if (status == 2) {
    std::cerr << "usage: refusion_replay mesh ARCHIVE OUT.ply [VOXEL] [THREADS]\n"
                 "       refusion_replay map FINAL MEMORY|- ARCHIVE OUT [--tl id:ns|closed,...] "
                 "[--scales bgv,bgt,objv] [--threads N] [--dump DIR] [--shown PREV_FINAL]\n";
  }
  return status;
}
