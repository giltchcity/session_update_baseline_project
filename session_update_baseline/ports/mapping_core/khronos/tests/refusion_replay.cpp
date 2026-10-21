// Offline replay of a session's surface (PresentSurface) without the mapper,
// for quick checks and against the offline reference.
//
//   refusion_replay mesh ARCHIVE.kfa OUT.ply [VOXEL] [THREADS] [TRUNCATION]
//       TSDF + marching cubes of all archived frames (no cut), as PLY
//       (truncation 2 * VOXEL unless given).
//   refusion_replay map FINAL.4dmap.zpk MEMORY.4dmap.zpk ARCHIVE.kfa OUT.4dmap.zpk
//       [--tl id:stamp_ns,...] [--scales BGV,BGT,OBJV] [--threads N]
//       [--dump DIR] [--shown PREV_FINAL.4dmap.zpk]
//       Integrates the archived frames (the mapper's KHRONOS_REFUSION_DUMP) and
//       composes the terminal round (--tl: current states begun in the session, as
//       the mapper logs them in that round) on the final snapshot of FINAL (the session's
//       chain_state: the object reasoning's final state) with memory = every
//       surface point of MEMORY's latest snapshot (the state that session loaded)
//       and, with --shown, memory as the previous session's final map showed it
//       (PresentSurface::Inputs::shown); writes a one-snapshot map and the report
//       OUT.json. BGV, BGT: the background map's voxel and truncation (the
//       background memory layer), OBJV: the object reconstruction voxel (the
//       present's TSDF, truncation 2 OBJV).
//   refusion_replay export MAP.4dmap.zpk OUT.ply
//       The latest snapshot's current surfaces with per-face physical id and slot.
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include <glog/logging.h>

#include <session_core/surface/frame_archive.h>
#include <session_core/surface/present_tsdf.h>
#include <session_core/surface/present_surface.h>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <hydra/utils/nearest_neighbor_utilities.h>
#include <khronos/backend/update_khronos_objects_functor.h>
#include <khronos/utils/khronos_attribute_utils.h>

using namespace khronos;

namespace {

int meshMode(int argc, char** argv) {
  if (argc < 4) return 2;
  const double voxel = argc > 4 ? std::stod(argv[4]) : 0.05;
  const int threads = argc > 5 ? std::stoi(argv[5]) : 4;
  const double truncation = argc > 6 ? std::stod(argv[6]) : 2 * voxel;
  std::vector<FrameArchive::Frame> frames;
  FrameArchive::Camera K;
  if (!FrameArchive::load(argv[2], frames, K)) {
    std::cerr << "cannot load archive " << argv[2] << '\n';
    return 1;
  }
  const auto t0 = std::chrono::steady_clock::now();
  hydra::VolumetricMap::Config mc;mc.voxel_size=voxel;mc.truncation_distance=truncation;
  hydra::ProjectiveIntegrator::Config ic;ic.num_threads=threads;
  hydra::MeshIntegratorConfig mesh;mesh.integrator_threads=threads;
  PresentTsdf tsdf(mc,ic,mesh);
  const PresentTsdf::Camera cam{K.width, K.height, K.fx, K.fy, K.cx, K.cy, K.min_range, K.max_range};
  const std::vector<float> mult = PresentTsdf::rayNorm(cam);
  std::vector<float> depth;
  std::vector<uint16_t> range, ids;
  for (const auto& f : frames) {
    if (!f.decode(static_cast<size_t>(K.width) * K.height, range, ids)) return 1;
    PresentTsdf::depthFromRange(cam, range, depth);
    tsdf.integrate(cam, f.world_T_sensor, mult, {&depth, nullptr});
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

std::vector<Eigen::Vector3f> surfacePoints(const DynamicSceneGraph& dsg) {
  std::vector<Eigen::Vector3f> points;
  if (dsg.hasMesh() && dsg.mesh()) {
    for (size_t i = 0; i < dsg.mesh()->numVertices(); ++i) points.push_back(dsg.mesh()->pos(i));
  }
  if (dsg.hasLayer(DsgLayers::OBJECTS)) {
    for (const auto& [id, node] : dsg.getLayer(DsgLayers::OBJECTS).nodes()) {
      const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
      if (!attrs) continue;
      for (size_t i = 0; i < attrs->mesh.numVertices(); ++i) {
        points.push_back(attrs->bounding_box.pointToWorldFrame(attrs->mesh.pos(i)));
      }
    }
  }
  return points;
}

// The current surfaces of a map's latest snapshot with per-face physical ids.
bool loadSurface(const std::string& path, PresentSurface::Surface& out) {
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

int mapMode(int argc, char** argv) {
  if (argc < 6) return 2;
  const std::string final_path = argv[2], memory_path = argv[3], archive_path = argv[4],
                    out_path = argv[5];
  PresentSurface::Inputs inputs;
  std::map<size_t, TimeStamp> state_starts;
  float background_voxel = 0.05f, background_truncation = 0.15f, object_voxel = 0.05f;
  int threads = 4;
  std::string shown_path;
  for (int i = 6; i + 1 < argc; i += 2) {
    const std::string key = argv[i], value = argv[i + 1];
    if (key == "--tl") {
      std::stringstream ss(value);
      std::string item;
      while (std::getline(ss, item, ',')) {
        const auto colon = item.find(':');
        state_starts[std::stoul(item.substr(0, colon))] = std::stoull(item.substr(colon + 1));
      }
    } else if (key == "--scales") {
      std::sscanf(value.c_str(), "%f,%f,%f", &background_voxel, &background_truncation, &object_voxel);
    } else if (key == "--threads") {
      threads = std::stoi(value);
    } else if (key == "--dump") {
      inputs.dump_dir = value;
    } else if (key == "--shown") {
      shown_path = value;
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
  std::vector<Eigen::Vector3f> memory;
  if (memory_path != "-") {
    const auto memory_map = SpatioTemporalMap::load(memory_path);
    if (!memory_map || !memory_map->numTimeSteps()) {
      std::cerr << "cannot load " << memory_path << '\n';
      return 1;
    }
    memory = surfacePoints(*memory_map->rawDsg(memory_map->numTimeSteps() - 1));
  }
  PresentSurface::Surface shown;
  if (!shown_path.empty()) {
    if (!loadSurface(shown_path, shown)) {
      std::cerr << "cannot load " << shown_path << '\n';
      return 1;
    }
    inputs.shown = &shown;
    std::cout << "shown memory: " << shown.vertices.size() << " vertices, " << shown.faces.size() << " faces\n";
  }
  std::cout << "final snapshot " << last << " stamp " << stamp << ", memory points " << memory.size()
            << '\n';
  std::unique_ptr<hydra::PointNeighborSearch> memory_search;
  if (!memory.empty()) memory_search = std::make_unique<hydra::PointNeighborSearch>(memory);
  // A copy of the loaded state, as the mapper decides it (P53).
  const float copy_tolerance = PresentSurface::copyTolerance(memory);
  if (memory_search) {
    inputs.is_memory = [&](const Eigen::Vector3f& p) {
      float d_sq = 0.f;
      size_t idx = 0;
      return memory_search->search(p, d_sq, idx) && d_sq <= copy_tolerance * copy_tolerance;
    };
  }
  std::vector<FrameArchive::Frame> frames;
  FrameArchive::Camera camera;
  if (!FrameArchive::load(archive_path, frames, camera)) {
    std::cerr << "cannot load archive " << archive_path << '\n';
    return 1;
  }
  if (object_voxel != background_voxel) {
    std::cerr << "native replay requires a shared object/background voxel\n";return 1;
  }
  hydra::VolumetricMap::Config mc;mc.voxel_size=background_voxel;mc.truncation_distance=background_truncation;
  hydra::ProjectiveIntegrator::Config ic;ic.num_threads=threads;
  hydra::MeshIntegratorConfig mesh;mesh.integrator_threads=threads;
  PresentSurface present(mc,ic,mesh);
  const size_t num_frames = frames.size();
  for (auto& frame : frames) present.integrate(camera, std::move(frame));
  std::cout << "integrated " << num_frames << " archived frames\n";
  const auto result =
      present.compose(*edited, stamp, state_starts, inputs, /*terminal=*/true);
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
  hydra::PipelineConfig pipeline; hydra::GlobalInfo::init(pipeline);
  FLAGS_logtostderr = true;
  const std::string mode = argc > 1 ? argv[1] : "";
  int status = 2;
  if (mode == "mesh") status = meshMode(argc, argv);
  if (mode == "map") status = mapMode(argc, argv);
  if (mode == "export") status = exportMode(argc, argv);
  if (status == 2) {
    std::cerr << "usage: refusion_replay mesh ARCHIVE OUT.ply [VOXEL] [THREADS] [TRUNCATION]\n"
                 "       refusion_replay map FINAL MEMORY|- ARCHIVE OUT [--tl id:ns,...] "
                 "[--scales bgv,bgt,objv] [--threads N] [--dump DIR] [--shown PREV_FINAL]\n"
                 "       refusion_replay export MAP OUT.ply\n";
  }
  return status;
}
