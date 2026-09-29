// Offline replay of the session-end update (SessionRefusion, the production
// code path) from a session's saved state, without the mapper.
//
//   refusion_replay map CHAIN.4dmap.zpk CARRIED.4dmap.zpk|- ARCHIVE.kfa OUT.4dmap.zpk
//       [--tl id:stamp_ns,...] [--scales BGV,BGT,OBJV] [--threads N] [--dump DIR]
//       [--previous SESSION_SURFACE.bin | --previous-map MAP.4dmap.zpk] [--membership C,N]
//       Runs the update on the final snapshot of CHAIN (the session's chain
//       state: the object reasoning's final state). CARRIED is the state that
//       session loaded (the geometry its reasoning carried over; "-" for a first
//       session), the previous surface is the predecessor's SessionSurface or,
//       for a map this algorithm did not produce, that map's surface. Writes
//       the one-snapshot map OUT, its report OUT.json and the session surface
//       (session_surface.bin) next to OUT, so that replays chain like sessions.
//   refusion_replay mesh ARCHIVE.kfa OUT.ply [VOXEL] [THREADS]
//       TSDF + marching cubes of all archived frames, as PLY.
//   refusion_replay export MAP.4dmap.zpk OUT.ply
//       The latest snapshot's current surfaces with per-face physical id and slot.
//   refusion_replay surface SESSION_SURFACE.bin OUT.ply
//       A session surface with per-face identity, range and scale.
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include <glog/logging.h>
#include <hydra/utils/nearest_neighbor_utilities.h>

#include <khronos/backend/reconciliation/frame_archive.h>
#include <khronos/backend/reconciliation/present_tsdf.h>
#include <khronos/backend/reconciliation/session_refusion.h>
#include <khronos/backend/reconciliation/session_surface.h>
#include <khronos/backend/update_khronos_objects_functor.h>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <khronos/utils/khronos_attribute_utils.h>

using namespace khronos;

namespace {

void writePlyHeader(std::ofstream& out, size_t vertices, size_t faces,
                    const std::vector<std::string>& face_properties) {
  out << "ply\nformat binary_little_endian 1.0\nelement vertex " << vertices
      << "\nproperty float x\nproperty float y\nproperty float z\nelement face " << faces
      << "\nproperty list uchar uint vertex_indices\n";
  for (const auto& p : face_properties) out << "property " << p << "\n";
  out << "end_header\n";
}

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
  std::vector<Eigen::Vector3f> V;
  std::vector<PresentTsdf::Face> F;
  tsdf.extractMesh(V, F);
  const double t_all = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "frames=" << frames.size() << " units=" << tsdf.numUnits() << " vertices=" << V.size()
            << " faces=" << F.size() << " total_s=" << t_all << '\n';
  std::ofstream out(argv[3], std::ios::binary);
  writePlyHeader(out, V.size(), F.size(), {});
  for (const auto& p : V) out.write(reinterpret_cast<const char*>(p.data()), 12);
  const uint8_t three = 3;
  for (const auto& f : F) {
    out.write(reinterpret_cast<const char*>(&three), 1);
    out.write(reinterpret_cast<const char*>(f.data()), 12);
  }
  return 0;
}

std::vector<Eigen::Vector3f> vertexPositions(const DynamicSceneGraph& dsg) {
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

int mapMode(int argc, char** argv) {
  if (argc < 6) return 2;
  const std::string chain_path = argv[2], carried_path = argv[3], archive_path = argv[4],
                    out_path = argv[5];
  SessionRefusion::Inputs inputs;
  SessionRefusion::Scales scales;
  scales.background_voxel = 0.05f;
  scales.background_truncation = 0.15f;
  scales.object_voxel = 0.02f;
  int threads = 4;
  std::string previous_path, previous_map_path;
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
      std::sscanf(value.c_str(), "%f,%f,%f", &scales.background_voxel, &scales.background_truncation,
                  &scales.object_voxel);
    } else if (key == "--threads") {
      threads = std::stoi(value);
    } else if (key == "--dump") {
      inputs.dump_dir = value;
    } else if (key == "--previous") {
      previous_path = value;
    } else if (key == "--previous-map") {
      previous_map_path = value;
    } else if (key == "--membership") {  // the object extractor's confidence,min_observations
      std::sscanf(value.c_str(), "%f,%d", &inputs.membership_confidence, &inputs.membership_observations);
    } else {
      std::cerr << "unknown option " << key << '\n';
      return 2;
    }
  }
  const auto chain = SpatioTemporalMap::load(chain_path);
  if (!chain || !chain->numTimeSteps()) {
    std::cerr << "cannot load " << chain_path << '\n';
    return 1;
  }
  const size_t last = chain->numTimeSteps() - 1;
  const TimeStamp stamp = chain->stamps()[last];
  auto edited = chain->rawDsg(last)->clone();
  std::vector<Eigen::Vector3f> carried;
  if (carried_path != "-") {
    const auto carried_map = SpatioTemporalMap::load(carried_path);
    if (!carried_map || !carried_map->numTimeSteps()) {
      std::cerr << "cannot load " << carried_path << '\n';
      return 1;
    }
    carried = vertexPositions(*carried_map->rawDsg(carried_map->numTimeSteps() - 1));
  }
  SessionSurface previous;
  if (!previous_path.empty()) {
    if (!SessionSurface::load(previous_path, previous)) {
      std::cerr << "cannot load " << previous_path << '\n';
      return 1;
    }
    inputs.previous = &previous;
  } else if (!previous_map_path.empty()) {
    const auto map = SpatioTemporalMap::load(previous_map_path);
    if (!map || !map->numTimeSteps()) {
      std::cerr << "cannot load " << previous_map_path << '\n';
      return 1;
    }
    previous = SessionSurface::fromDsg(*map->rawDsg(map->numTimeSteps() - 1));
    inputs.previous = &previous;
  }
  std::cout << "chain snapshot " << last << " stamp " << stamp << ", carried vertices " << carried.size()
            << ", previous surface faces " << previous.numFaces() << '\n';
  std::unique_ptr<hydra::PointNeighborSearch> carried_search;
  if (!carried.empty()) carried_search = std::make_unique<hydra::PointNeighborSearch>(carried);
  std::vector<FrameArchive::Frame> frames;
  if (!FrameArchive::load(archive_path, frames, inputs.camera)) {
    std::cerr << "cannot load archive " << archive_path << '\n';
    return 1;
  }
  inputs.frames = &frames;
  inputs.scales = scales;
  inputs.final_stamp = stamp;
  inputs.carried = [&](const Eigen::Vector3f& p) {
    float d_sq = 0.f;
    size_t idx = 0;
    return carried_search && carried_search->search(p, d_sq, idx) && d_sq <= 0.003f * 0.003f;
  };
  SessionRefusion::Config config;
  config.num_threads = threads;
  const SessionRefusion refusion(config);
  auto result = refusion.apply(*edited, inputs);
  std::cout << "applied=" << result.applied << " depth_scale=" << result.depth_scale << " " << result.summary
            << '\n';
  if (!result.applied) return 1;
  SpatioTemporalMap out(chain->config);
  out.update(edited, stamp);
  const auto parent = std::filesystem::path(out_path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  if (!out.save(out_path)) {
    std::cerr << "cannot save " << out_path << '\n';
    return 1;
  }
  std::ofstream(out_path + ".json") << result.report_json << '\n';
  const auto surface_path = (parent.empty() ? std::filesystem::path(".") : parent) / SessionSurface::kFileName;
  if (!result.surface.save(surface_path.string())) {
    std::cerr << "cannot save " << surface_path << '\n';
    return 1;
  }
  return 0;
}

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
    const auto base = static_cast<uint32_t>(V.size());
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
  writePlyHeader(out, V.size(), F.size(), {"int physical", "int slot"});
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

int surfaceMode(int argc, char** argv) {
  if (argc < 4) return 2;
  SessionSurface s;
  if (!SessionSurface::load(argv[2], s)) {
    std::cerr << "cannot load " << argv[2] << '\n';
    return 1;
  }
  std::ofstream out(argv[3], std::ios::binary);
  writePlyHeader(out, s.vertices.size(), s.faces.size(), {"int identity", "float range", "float scale"});
  for (const auto& p : s.vertices) out.write(reinterpret_cast<const char*>(p.data()), 12);
  const uint8_t three = 3;
  for (size_t i = 0; i < s.faces.size(); ++i) {
    out.write(reinterpret_cast<const char*>(&three), 1);
    out.write(reinterpret_cast<const char*>(s.faces[i].data()), 12);
    out.write(reinterpret_cast<const char*>(&s.identity[i]), 4);
    out.write(reinterpret_cast<const char*>(&s.range[i]), 4);
    out.write(reinterpret_cast<const char*>(&s.scale[i]), 4);
  }
  std::cout << "surface " << s.vertices.size() << " vertices, " << s.faces.size() << " faces\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;
  const std::string mode = argc > 1 ? argv[1] : "";
  int status = 2;
  if (mode == "mesh") status = meshMode(argc, argv);
  if (mode == "map") status = mapMode(argc, argv);
  if (mode == "export") status = exportMode(argc, argv);
  if (mode == "surface") status = surfaceMode(argc, argv);
  if (status == 2) {
    std::cerr << "usage: refusion_replay map CHAIN CARRIED|- ARCHIVE OUT [--tl id:ns,...] "
                 "[--scales bgv,bgt,objv] [--threads N] [--dump DIR] [--previous SURFACE.bin | "
                 "--previous-map MAP] [--membership c,n]\n"
                 "       refusion_replay mesh ARCHIVE OUT.ply [VOXEL] [THREADS]\n"
                 "       refusion_replay export MAP OUT.ply\n"
                 "       refusion_replay surface SURFACE.bin OUT.ply\n";
  }
  return status;
}
