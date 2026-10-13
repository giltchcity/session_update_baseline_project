#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_attributes.h>
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: diag32 <map.4dmap> <inst> <start> <end>\n";
    return 2;
  }
  auto map = khronos::SpatioTemporalMap::load(argv[1]);
  if (!map) {
    std::cerr << "load failed\n";
    return 2;
  }
  const long inst = std::atol(argv[2]);
  const size_t start = std::atoi(argv[3]), end = std::atoi(argv[4]);
  const auto stamps = map->stamps();
  for (size_t idx = start; idx < stamps.size() && idx <= end; ++idx) {
    const auto dsg = map->getDsgPtr(stamps[idx]);
    bool node_found = false;
    size_t nv = 0, recon = 0, dynamic = 0, traj = 0;
    long first_s = 0, last_s = 0;
    if (dsg && dsg->hasLayer(spark_dsg::DsgLayers::OBJECTS)) {
      for (const auto& [id, node] : dsg->getLayer(spark_dsg::DsgLayers::OBJECTS).nodes()) {
        auto attrs = node->tryAttributes<spark_dsg::KhronosObjectAttributes>();
        if (!attrs) continue;
        long i2 = -1;
        auto it = attrs->details.find("instance_id");
        if (it != attrs->details.end() && !it->second.empty()) i2 = (long)it->second[0];
        if (i2 != inst) continue;
        node_found = true;
        nv = attrs->mesh.numVertices();
        traj = attrs->trajectory_positions.size();
        auto rit = attrs->details.find("reconstruction_frames");
        if (rit != attrs->details.end() && !rit->second.empty()) recon = rit->second[0];
        auto dit = attrs->details.find("has_dynamic_history");
        if (dit != attrs->details.end() && !dit->second.empty()) dynamic = dit->second[0];
        if (!attrs->first_observed_ns.empty()) first_s = attrs->first_observed_ns.back() / 1000000000ULL;
        if (!attrs->last_observed_ns.empty()) last_s = attrs->last_observed_ns.back() / 1000000000ULL;
      }
    }
    std::cout << "f" << idx << " t=" << (stamps[idx] / 1000000000ULL) << "s"
              << " node=" << (node_found ? "yes" : "NO")
              << " verts=" << nv
              << " recon=" << recon
              << " dyn=" << dynamic
              << " traj=" << traj
              << " first=" << first_s << "s last=" << last_s << "s\n";
  }
  return 0;
}
