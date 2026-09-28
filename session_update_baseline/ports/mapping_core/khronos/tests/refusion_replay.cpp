// Offline replay of the session-end re-integration (SessionRefusion) without
// the mapper, for checks against the offline reference.
//
//   refusion_replay mesh ARCHIVE.kfa OUT.ply [VOXEL] [THREADS]
//       TSDF + marching cubes of all archived frames (no cut), as PLY.
//   refusion_replay map FINAL.4dmap.zpk MEMORY.4dmap.zpk ARCHIVE.kfa OUT.4dmap.zpk
//       [--tl id:stamp_ns,...] [--scales BGV,BGT,OBJV] [--threads N] [--dump DIR]
//       [--retired RETIRED.xyz]
//       Re-integrates the final snapshot of FINAL (a consolidated session map; or the
//       session's chain_state with --retired = its consolidation_retired.xyz, whose
//       positions are erased first, which reproduces the consolidated map) with
//       memory = every surface point of MEMORY's latest snapshot (the state that
//       session loaded), and writes a one-snapshot map and the report OUT.json.
//       [--shown PREV_FINAL.4dmap.zpk --chain PREV_RETIRED.xyz]: memory as the
//       previous session's final map showed it (SessionRefusion::Inputs::shown),
//       with the previous session's consolidation retirements.
//   refusion_replay export MAP.4dmap.zpk OUT.ply
//       The latest snapshot's current surfaces with per-face physical id and slot.
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include <glog/logging.h>

#include <khronos/backend/reconciliation/frame_archive.h>
#include <khronos/backend/reconciliation/present_tsdf.h>
#include <khronos/backend/reconciliation/session_consolidation.h>
#include <khronos/backend/reconciliation/session_refusion.h>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <hydra/utils/nearest_neighbor_utilities.h>
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

std::vector<Eigen::Vector3f> surfacePoints(const DynamicSceneGraph& dsg,
                                           std::vector<uint32_t>* physical = nullptr) {
  std::vector<Eigen::Vector3f> points;
  if (dsg.hasMesh() && dsg.mesh()) {
    for (size_t i = 0; i < dsg.mesh()->numVertices(); ++i) points.push_back(dsg.mesh()->pos(i));
  }
  if (physical) physical->assign(points.size(), 0);
  if (dsg.hasLayer(DsgLayers::OBJECTS)) {
    for (const auto& [id, node] : dsg.getLayer(DsgLayers::OBJECTS).nodes()) {
      const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
      if (!attrs) continue;
      const uint32_t p = static_cast<uint32_t>(UpdateKhronosObjectsFunctor::physicalInstanceId(*attrs).value_or(0));
      for (size_t i = 0; i < attrs->mesh.numVertices(); ++i) {
        points.push_back(attrs->bounding_box.pointToWorldFrame(attrs->mesh.pos(i)));
        if (physical) physical->push_back(p);
      }
    }
  }
  return points;
}

// The current surfaces of a map's latest snapshot with per-face physical ids.
bool loadSurface(const std::string& path, SessionRefusion::Surface& out) {
  const auto map = SpatioTemporalMap::load(path);
  if (!map || !map->numTimeSteps()) return false;
  const auto dsg = map->rawDsg(map->numTimeSteps() - 1);
  auto add = [&](const spark_dsg::Mesh& mesh, const KhronosObjectAttributes* attrs, uint32_t p) {
    const uint32_t base = static_cast<uint32_t>(out.vertices.size());
    for (size_t i = 0; i < mesh.numVertices(); ++i) {
      out.vertices.push_back(attrs ? attrs->bounding_box.pointToWorldFrame(mesh.pos(i)) : mesh.pos(i));
    }
    const size_t n = mesh.numVertices();
    for (const auto& f : mesh.faces) {
      if (f[0] >= n || f[1] >= n || f[2] >= n) continue;
      out.faces.push_back({base + static_cast<uint32_t>(f[0]), base + static_cast<uint32_t>(f[1]),
                           base + static_cast<uint32_t>(f[2])});
      out.face_physical.push_back(p);
    }
  };
  if (dsg->hasMesh() && dsg->mesh()) add(*dsg->mesh(), nullptr, 0);
  if (dsg->hasLayer(DsgLayers::OBJECTS)) {
    for (const auto& [id, node] : dsg->getLayer(DsgLayers::OBJECTS).nodes()) {
      const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
      if (!attrs || !hasCurrentObjectMesh(*attrs)) continue;
      add(attrs->mesh, attrs,
          static_cast<uint32_t>(UpdateKhronosObjectsFunctor::physicalInstanceId(*attrs).value_or(0)));
    }
  }
  return true;
}

std::vector<Eigen::Vector3f> readXyz(const std::string& path) {
  std::vector<Eigen::Vector3f> points;
  std::ifstream in(path, std::ios::binary);
  uint64_t n = 0;
  if (in.read(reinterpret_cast<char*>(&n), sizeof(n))) {
    float xyz[3];
    for (uint64_t i = 0; i < n && in.read(reinterpret_cast<char*>(xyz), sizeof(xyz)); ++i) {
      points.emplace_back(xyz[0], xyz[1], xyz[2]);
    }
  }
  return points;
}

// Erase every vertex within 3 mm (SessionConsolidation::memory_match_distance) of a
// retired position, and the faces that use it: the consolidated final map from
// the unconsolidated chain state (the consolidation erases exactly these).
size_t eraseRetired(DynamicSceneGraph& dsg, const std::string& path) {
  std::vector<Eigen::Vector3f> retired;
  std::ifstream in(path, std::ios::binary);
  uint64_t n = 0;
  if (in.read(reinterpret_cast<char*>(&n), sizeof(n))) {
    float xyz[3];
    for (uint64_t i = 0; i < n && in.read(reinterpret_cast<char*>(xyz), sizeof(xyz)); ++i) {
      retired.emplace_back(xyz[0], xyz[1], xyz[2]);
    }
  }
  if (retired.empty()) return 0;
  const hydra::PointNeighborSearch search(retired);
  size_t erased = 0;
  auto erase = [&](spark_dsg::Mesh& mesh, const KhronosObjectAttributes* attrs) {
    const size_t nv = mesh.numVertices();
    std::vector<int64_t> remap(nv, -1);
    size_t kept = 0;
    std::vector<uint8_t> gone(nv, 0);
    for (size_t i = 0; i < nv; ++i) {
      const Eigen::Vector3f p = attrs ? attrs->bounding_box.pointToWorldFrame(mesh.pos(i)) : mesh.pos(i);
      float d_sq = 0.f;
      size_t idx = 0;
      gone[i] = search.search(p, d_sq, idx) && d_sq <= 0.003f * 0.003f;
      if (!gone[i]) remap[i] = static_cast<int64_t>(kept++);
    }
    if (kept == nv) return;
    erased += nv - kept;
    auto compact = [&](auto& values) {
      if (values.size() != nv) return;
      size_t j = 0;
      for (size_t i = 0; i < nv; ++i) if (!gone[i]) values[j++] = values[i];
      values.resize(j);
    };
    compact(mesh.points);
    compact(mesh.colors);
    compact(mesh.stamps);
    compact(mesh.labels);
    compact(mesh.first_seen_stamps);
    spark_dsg::Mesh::Faces faces;
    for (const auto& f : mesh.faces) {
      if (f[0] >= nv || f[1] >= nv || f[2] >= nv) continue;
      const int64_t a = remap[f[0]], b = remap[f[1]], c = remap[f[2]];
      if (a < 0 || b < 0 || c < 0) continue;
      faces.push_back({{static_cast<size_t>(a), static_cast<size_t>(b), static_cast<size_t>(c)}});
    }
    mesh.faces = std::move(faces);
  };
  if (dsg.hasMesh() && dsg.mesh()) erase(*dsg.mesh(), nullptr);
  if (dsg.hasLayer(DsgLayers::OBJECTS)) {
    for (const auto& [id, node] : dsg.getLayer(DsgLayers::OBJECTS).nodes()) {
      auto* attrs = dynamic_cast<KhronosObjectAttributes*>(&node->attributes());
      if (!attrs || !hasCurrentObjectMesh(*attrs)) continue;
      erase(attrs->mesh, attrs);
    }
  }
  return erased;
}

int mapMode(int argc, char** argv) {
  if (argc < 6) return 2;
  const std::string final_path = argv[2], memory_path = argv[3], archive_path = argv[4],
                    out_path = argv[5];
  SessionRefusion::Inputs inputs;
  SessionConsolidation::Scales scales;
  scales.background_voxel = 0.05f;
  scales.background_truncation = 0.15f;
  scales.object_voxel = 0.02f;
  int threads = 4;
  std::string retired_path, shown_path, chain_path;
  for (int i = 6; i + 1 < argc; i += 2) {
    const std::string key = argv[i], value = argv[i + 1];
    if (key == "--tl") {
      std::stringstream ss(value);
      std::string item;
      while (std::getline(ss, item, ',')) {
        const auto colon = item.find(':');
        inputs.state_starts[std::stoul(item.substr(0, colon))] = std::stoull(item.substr(colon + 1));
      }
    } else if (key == "--scales") {
      std::sscanf(value.c_str(), "%f,%f,%f", &scales.background_voxel,
                  &scales.background_truncation, &scales.object_voxel);
    } else if (key == "--threads") {
      threads = std::stoi(value);
    } else if (key == "--dump") {
      inputs.dump_dir = value;
    } else if (key == "--retired") {
      retired_path = value;
    } else if (key == "--shown") {
      shown_path = value;
    } else if (key == "--chain") {
      chain_path = value;



    } else {
      std::cerr << "unknown option " << key << '\n';
      return 2;
    }
  }
  const auto final_map = SpatioTemporalMap::load(final_path);
  if (!final_map || !final_map->numTimeSteps()) {
    std::cerr << "cannot load " << final_path << '\n';
    return 1;
  }
  const size_t last = final_map->numTimeSteps() - 1;
  const TimeStamp stamp = final_map->stamps()[last];
  auto edited = final_map->rawDsg(last)->clone();
  if (!retired_path.empty()) {
    std::cout << "erased " << eraseRetired(*edited, retired_path) << " retired vertices\n";
  }
  std::vector<Eigen::Vector3f> memory;
  std::vector<uint32_t> memory_physical;
  if (memory_path != "-") {
    const auto memory_map = SpatioTemporalMap::load(memory_path);
    if (!memory_map || !memory_map->numTimeSteps()) {
      std::cerr << "cannot load " << memory_path << '\n';
      return 1;
    }
    memory = surfacePoints(*memory_map->rawDsg(memory_map->numTimeSteps() - 1), &memory_physical);
  }
  SessionRefusion::Surface shown;
  if (!shown_path.empty()) {
    if (!loadSurface(shown_path, shown)) {
      std::cerr << "cannot load " << shown_path << '\n';
      return 1;
    }
    inputs.shown = &shown;
    inputs.inherited = &memory;
    inputs.inherited_physical = &memory_physical;
    std::cout << "shown memory: " << shown.vertices.size() << " vertices, " << shown.faces.size() << " faces\n";
  }
  std::unique_ptr<hydra::PointNeighborSearch> chain_search;
  std::vector<Eigen::Vector3f> chain_points;
  if (!chain_path.empty()) {
    chain_points = readXyz(chain_path);
    if (!chain_points.empty()) chain_search = std::make_unique<hydra::PointNeighborSearch>(chain_points);
    inputs.chain_retired = [&](const Eigen::Vector3f& p) {
      float d_sq = 0.f;
      size_t idx = 0;
      return chain_search && chain_search->search(p, d_sq, idx) && d_sq <= 0.003f * 0.003f;
    };
  }
  std::cout << "final snapshot " << last << " stamp " << stamp << ", memory points " << memory.size()
            << '\n';
  SessionConsolidation consolidation(SessionConsolidation::Config{});
  consolidation.setScales(scales);
  consolidation.setMemory(memory);
  std::vector<FrameArchive::Frame> frames;
  if (!FrameArchive::load(archive_path, frames, inputs.camera)) {
    std::cerr << "cannot load archive " << archive_path << '\n';
    return 1;
  }
  inputs.frames = &frames;
  inputs.scales = scales;
  inputs.final_stamp = stamp;
  inputs.is_memory = [&](const Eigen::Vector3f& p) { return consolidation.isMemory(p); };
  SessionRefusion::Config config;
  config.num_threads = threads;
  const SessionRefusion refusion(config);
  auto result = refusion.apply(*edited, inputs);
  std::cout << "applied=" << result.applied << " " << result.summary << '\n';
  if (!result.applied) return 1;
  SpatioTemporalMap out(final_map->config);
  out.update(edited, stamp);
  const auto parent = std::filesystem::path(out_path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  if (!out.save(out_path)) {
    std::cerr << "cannot save " << out_path << '\n';
    return 1;
  }
  std::ofstream(out_path + ".json") << result.report_json << '\n';
  return 0;
}

}  // namespace

// export MAP OUT.ply: the latest snapshot's current surfaces in world coordinates, one
// PLY with per-face int properties "physical" (0 = background) and "slot" (0 =
// background, k = k-th current object node).
int exportMode(int argc, char** argv) {
  if (argc < 4) return 2;
  const auto map = SpatioTemporalMap::load(argv[2]);
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
                 "       refusion_replay map FINAL MEMORY|- ARCHIVE OUT [--tl id:ns,...] "
                 "[--scales bgv,bgt,objv] [--threads N] [--dump DIR]\n";
  }
  return status;
}
