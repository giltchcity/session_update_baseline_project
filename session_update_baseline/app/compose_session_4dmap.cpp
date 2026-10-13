#include <cstdint>
#include <iostream>
#include <string>

#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <spark_dsg/dynamic_scene_graph.h>

#include "session_update_baseline/runtime/session_state.h"

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: compose_session_4dmap PRIOR_MAP CURRENT_DSG STAMP_NS OUTPUT\n";
    return 2;
  }

  auto prior_map = khronos::SpatioTemporalMap::load(argv[1]);
  if (!prior_map || prior_map->numTimeSteps() == 0) {
    std::cerr << "failed to load prior map\n";
    return 3;
  }
  auto dsg = spark_dsg::DynamicSceneGraph::load(argv[2]);
  if (!dsg || !dsg->hasMesh() || dsg->mesh()->empty()) {
    std::cerr << "failed to load current DSG with mesh\n";
    return 4;
  }

  const auto seed = session_update::runtime::latestSessionSeed(*prior_map);
  const auto prior_stamp = seed.stamp;
  const auto stamp = static_cast<uint64_t>(std::stoull(argv[3]));
  if (stamp <= prior_stamp) {
    std::cerr << "current stamp must be newer than prior latest stamp\n";
    return 5;
  }

  // Recursive session contract: one prior-latest seed plus this session's
  // states. Never copy the prior map's complete time axis into its successor.
  khronos::SpatioTemporalMap map(khronos::SpatioTemporalMap::Config{});
  session_update::runtime::initializeSessionTimeline(map, seed);
  map.update(dsg, stamp);
  if (!map.save(argv[4])) {
    std::cerr << "failed to save composed map\n";
    return 7;
  }

  auto check = khronos::SpatioTemporalMap::load(argv[4]);
  if (!check || check->numTimeSteps() != 2 || check->stamps().front() != prior_stamp ||
      check->stamps().back() != stamp) {
    std::cerr << "composed map validation failed\n";
    return 8;
  }
  std::cout << "validated_time_steps=" << check->numTimeSteps() << "\n";
  std::cout << "current_mesh_vertices=" << dsg->mesh()->numVertices() << "\n";
  std::cout << "current_mesh_faces=" << dsg->mesh()->numFaces() << "\n";
  return 0;
}
