#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_attributes.h>
#include <iostream>
#include <map>

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: diag36 <map.4dmap> <inst> <stamp_idx>\n";
    return 2;
  }
  auto map = khronos::SpatioTemporalMap::load(argv[1]);
  if (!map) {
    std::cerr << "load failed\n";
    return 2;
  }
  const long inst = std::atol(argv[2]);
  const size_t idx = std::atoi(argv[3]);
  const auto stamps = map->stamps();
  if (idx >= stamps.size()) {
    std::cerr << "idx out of range\n";
    return 2;
  }
  const auto dsg = map->getDsgPtr(stamps[idx]);
  if (!dsg || !dsg->hasLayer(spark_dsg::DsgLayers::OBJECTS)) {
    std::cerr << "no objects\n";
    return 2;
  }
  for (const auto& [id, node] : dsg->getLayer(spark_dsg::DsgLayers::OBJECTS).nodes()) {
    auto attrs = node->tryAttributes<spark_dsg::KhronosObjectAttributes>();
    if (!attrs) continue;
    long i2 = -1;
    auto it = attrs->details.find("instance_id");
    if (it != attrs->details.end() && !it->second.empty()) i2 = (long)it->second[0];
    if (i2 != inst) continue;
    const auto& mesh = attrs->mesh;
    std::map<long, size_t> hist;
    size_t finite = 0, open = 0;
    for (size_t i = 0; i < mesh.numVertices() && i < mesh.first_seen_stamps.size(); ++i) {
      const auto s = mesh.first_seen_stamps[i];
      if (s == 0 || s == std::numeric_limits<uint64_t>::max()) {
        ++open;
      } else {
        ++finite;
        hist[(long)(s / 1000000000ULL) / 10 * 10]++;
      }
    }
    std::cout << "node=" << id << " inst=" << inst
              << " verts=" << mesh.numVertices()
              << " first_seen: finite=" << finite << " zero/open=" << open << "\n";
    std::cout << "first_seen 10s-bucket histogram:\n";
    for (const auto& [bucket, n] : hist) {
      std::cout << "  t~" << bucket << "s: " << n << "\n";
    }
  }
  return 0;
}
