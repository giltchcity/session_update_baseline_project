#include <cstdint>
#include <iostream>
#include <string>

#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <spark_dsg/dynamic_scene_graph.h>

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: compose_session_4dmap PRIOR_MAP CURRENT_DSG STAMP_NS OUTPUT\n";
    return 2;
  }

  auto map = khronos::SpatioTemporalMap::load(argv[1]);
  if (!map || map->numTimeSteps() == 0) {
    std::cerr << "failed to load prior map\n";
    return 3;
  }
  auto dsg = spark_dsg::DynamicSceneGraph::load(argv[2]);
  if (!dsg || !dsg->hasMesh() || dsg->mesh()->empty()) {
    std::cerr << "failed to load current DSG with mesh\n";
    return 4;
  }

  const auto stamp = static_cast<uint64_t>(std::stoull(argv[3]));
  map->update(dsg, stamp);
  if (!map->save(argv[4])) {
    std::cerr << "failed to save composed map\n";
    return 5;
  }

  auto check = khronos::SpatioTemporalMap::load(argv[4]);
  if (!check || check->numTimeSteps() != map->numTimeSteps()) {
    std::cerr << "composed map validation failed\n";
    return 6;
  }
  std::cout << "validated_time_steps=" << check->numTimeSteps() << "\n";
  std::cout << "current_mesh_vertices=" << dsg->mesh()->numVertices() << "\n";
  std::cout << "current_mesh_faces=" << dsg->mesh()->numFaces() << "\n";
  return 0;
}
