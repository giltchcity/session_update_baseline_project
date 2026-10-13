#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_attributes.h>
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: diag34 <map.4dmap> <stamp_idx> [stamp_idx2 ...]\n";
    return 2;
  }
  auto map = khronos::SpatioTemporalMap::load(argv[1]);
  if (!map) {
    std::cerr << "load failed\n";
    return 2;
  }
  const auto stamps = map->stamps();
  for (int a = 2; a < argc; ++a) {
    const size_t idx = std::atoi(argv[a]);
    if (idx >= stamps.size()) {
      std::cerr << "idx " << idx << " out of range\n";
      continue;
    }
    const auto dsg = map->rawDsg(idx);
    std::cout << "=== f" << idx << " t=" << (stamps[idx] / 1000000000ULL) << "s RAW ===\n";
    if (!dsg || !dsg->hasLayer(spark_dsg::DsgLayers::OBJECTS)) {
      std::cout << "no objects layer\n";
      continue;
    }
    for (const auto& [id, node] : dsg->getLayer(spark_dsg::DsgLayers::OBJECTS).nodes()) {
      auto attrs = node->tryAttributes<spark_dsg::KhronosObjectAttributes>();
      if (!attrs) continue;
      long inst = -1;
      auto it = attrs->details.find("instance_id");
      if (it != attrs->details.end() && !it->second.empty()) inst = (long)it->second[0];
      const long first = attrs->first_observed_ns.empty()
                             ? -1
                             : (long)(attrs->first_observed_ns.front() / 1000000000ULL);
      const long last = attrs->last_observed_ns.empty()
                            ? -1
                            : (long)(attrs->last_observed_ns.back() / 1000000000ULL);
      std::cout << "  node=" << id << " sem=" << attrs->semantic_label
                << " inst=" << inst << " verts=" << attrs->mesh.numVertices()
                << " first=" << first << "s last=" << last << "s\n";
    }
  }
  return 0;
}
