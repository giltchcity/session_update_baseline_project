// Offline replay of the session-end re-integration (SessionRefusion) without
// the mapper, for checks against the offline reference.
//
//   refusion_replay mesh ARCHIVE.kfa OUT.ply [VOXEL] [THREADS]
//       TSDF + marching cubes of all archived frames (no cut), as PLY.
//   refusion_replay map FINAL.4dmap.zpk MEMORY.4dmap.zpk ARCHIVE.kfa OUT.4dmap.zpk
//       [--tl id:stamp_ns,...] [--scales BGV,BGT,OBJV] [--threads N] [--dump DIR]
//       Re-integrates the final snapshot of FINAL (a consolidated session map) with
//       memory = every surface point of MEMORY's latest snapshot (the state that
//       session loaded), and writes a one-snapshot map and the report OUT.json.
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

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;
  const std::string mode = argc > 1 ? argv[1] : "";
  int status = 2;
  if (mode == "mesh") status = meshMode(argc, argv);
  if (mode == "map") status = mapMode(argc, argv);
  if (status == 2) {
    std::cerr << "usage: refusion_replay mesh ARCHIVE OUT.ply [VOXEL] [THREADS]\n"
                 "       refusion_replay map FINAL MEMORY|- ARCHIVE OUT [--tl id:ns,...] "
                 "[--scales bgv,bgt,objv] [--threads N] [--dump DIR]\n";
  }
  return status;
}
