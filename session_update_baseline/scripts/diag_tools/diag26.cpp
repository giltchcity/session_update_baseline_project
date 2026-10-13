#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_attributes.h>
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 2) { std::cerr << "usage: diag26 <map.4dmap>\n"; return 2; }
  auto map = khronos::SpatioTemporalMap::load(argv[1]);
  if (!map) { std::cerr << "load failed\n"; return 2; }
  const auto stamps = map->stamps();
  const auto dsg = map->getDsgPtr(stamps.back());
  if (!dsg || !dsg->hasLayer(spark_dsg::DsgLayers::OBJECTS)) { std::cerr << "no objects\n"; return 2; }
  for (const auto& [id, node] : dsg->getLayer(spark_dsg::DsgLayers::OBJECTS).nodes()) {
    auto attrs = node->tryAttributes<spark_dsg::KhronosObjectAttributes>();
    if (!attrs) continue;
    long inst = -1;
    auto it = attrs->details.find("instance_id");
    if (it != attrs->details.end() && !it->second.empty()) inst = (long)it->second[0];
    const bool has_traj = attrs->trajectory_positions.size() > 0;
    if (inst != 12 && !has_traj) continue;
    std::cout << "node=" << id << " sem=" << attrs->semantic_label
              << " inst=" << inst
              << " verts=" << attrs->mesh.numVertices()
              << " first=" << (attrs->first_observed_ns.empty()?0:attrs->first_observed_ns.back()/1000000000ULL)
              << "s last=" << (attrs->last_observed_ns.empty()?0:attrs->last_observed_ns.back()/1000000000ULL) << "s"
              << " traj_len=" << attrs->trajectory_positions.size()
              << " traj_ts=" << attrs->trajectory_timestamps.size()
              << " dyn_samples=" << attrs->dynamic_object_points.size()
              << "\n";
    if (attrs->trajectory_positions.size()) {
      std::cout << "  traj_ts[0..min(4)]: ";
      for (size_t i=0;i<attrs->trajectory_timestamps.size() && i<5;++i)
        std::cout << attrs->trajectory_timestamps[i]/1000000000ULL << "s ";
      std::cout << " ... last: ";
      for (size_t i=attrs->trajectory_timestamps.size()>5?attrs->trajectory_timestamps.size()-3:0;
           i<attrs->trajectory_timestamps.size();++i)
        std::cout << attrs->trajectory_timestamps[i]/1000000000ULL << "s ";
      std::cout << "\n";
      std::cout << "  traj_pos[0]: " << attrs->trajectory_positions[0].transpose()
                << "  last: " << attrs->trajectory_positions.back().transpose() << "\n";
    }
    for (size_t i=0;i<attrs->dynamic_object_points.size() && i<6;++i)
      std::cout << "  dyn sample " << i << ": " << attrs->dynamic_object_points[i].size() << " pts\n";
    if (attrs->dynamic_object_points.size()>6)
      std::cout << "  ... last sample: " << attrs->dynamic_object_points.back().size() << " pts\n";
  }
  return 0;
}
