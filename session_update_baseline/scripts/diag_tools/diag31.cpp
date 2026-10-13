#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_attributes.h>
#include <iostream>
#include <limits>

int main(int argc, char** argv) {
  if (argc < 6) { std::cerr << "usage: diag31 <map> <inst> <start> <end> <step>\n"; return 2; }
  auto map = khronos::SpatioTemporalMap::load(argv[1]);
  if (!map) { std::cerr << "load failed\n"; return 2; }
  const long inst = std::atol(argv[2]);
  const size_t start = std::atoi(argv[3]), end = std::atoi(argv[4]);
  const size_t step = std::max(1, std::atoi(argv[5]));
  const auto stamps = map->stamps();
  std::cout << "stamps=" << stamps.size() << "\n";
  for (size_t idx = start; idx < stamps.size() && idx <= end; idx += step) {
    const auto dsg = map->getDsgPtr(stamps[idx]);
    size_t nv = 0;
    Eigen::Vector3f mn(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    Eigen::Vector3f mx(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
    if (dsg && dsg->hasLayer(spark_dsg::DsgLayers::OBJECTS)) {
      for (const auto& [id, node] : dsg->getLayer(spark_dsg::DsgLayers::OBJECTS).nodes()) {
        auto attrs = node->tryAttributes<spark_dsg::KhronosObjectAttributes>();
        if (!attrs) continue;
        long i2 = -1;
        auto it = attrs->details.find("instance_id");
        if (it != attrs->details.end() && !it->second.empty()) i2 = (long)it->second[0];
        if (i2 != inst) continue;
        nv = attrs->mesh.numVertices();
        for (size_t i = 0; i < nv; ++i) {
          const auto p = attrs->bounding_box.pointToWorldFrame(attrs->mesh.pos(i));
          mn = mn.cwiseMin(p); mx = mx.cwiseMax(p);
        }
      }
    }
    if (nv == 0) {
      std::cout << "f" << idx << " t=" << (stamps[idx]/1000000000ULL) << "s verts=0\n";
    } else {
      std::cout << "f" << idx << " t=" << (stamps[idx]/1000000000ULL) << "s verts=" << nv
                << " center=(" << (mn.x()+mx.x())/2 << "," << (mn.y()+mx.y())/2 << "," << (mn.z()+mx.z())/2 << ")"
                << " span=(" << mx.x()-mn.x() << "," << mx.y()-mn.y() << "," << mx.z()-mn.z() << ")\n";
    }
  }
  return 0;
}
