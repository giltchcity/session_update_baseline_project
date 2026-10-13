#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_attributes.h>
#include <iostream>
#include <limits>

int main(int argc, char** argv) {
  if (argc < 2) { std::cerr << "usage: diag30 <map.4dmap>\n"; return 2; }
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
    const size_t nv = attrs->mesh.numVertices();
    if (nv == 0) { std::cout << "node=" << id << " inst=" << inst << " verts=0\n"; continue; }
    Eigen::Vector3f mn(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    Eigen::Vector3f mx(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
    for (size_t i = 0; i < nv; ++i) {
      const auto p = attrs->bounding_box.pointToWorldFrame(attrs->mesh.pos(i));
      mn = mn.cwiseMin(p); mx = mx.cwiseMax(p);
    }
    std::cout << "node=" << id << " inst=" << inst << " verts=" << nv
              << " center=(" << (mn.x()+mx.x())/2 << "," << (mn.y()+mx.y())/2 << "," << (mn.z()+mx.z())/2 << ")"
              << " x[" << mn.x() << "," << mx.x() << "] z[" << mn.z() << "," << mx.z() << "]\n";
  }
  return 0;
}
