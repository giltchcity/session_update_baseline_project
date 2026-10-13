#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_attributes.h>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 3) { std::cerr << "usage: diag29 <map.4dmap> <stamp_idx> <out.csv>\n"; return 2; }
  auto map = khronos::SpatioTemporalMap::load(argv[1]);
  if (!map) { std::cerr << "load failed\n"; return 2; }
  const size_t idx = std::atoi(argv[2]);
  const auto stamps = map->stamps();
  if (idx >= stamps.size()) { std::cerr << "idx out of range\n"; return 2; }
  const auto dsg = map->getDsgPtr(stamps[idx]);
  if (!dsg || !dsg->hasLayer(spark_dsg::DsgLayers::OBJECTS)) { std::cerr << "no objects\n"; return 2; }
  std::ofstream out(argv[3]);
  out << "inst,x,y,z\n";
  for (const auto& [id, node] : dsg->getLayer(spark_dsg::DsgLayers::OBJECTS).nodes()) {
    auto attrs = node->tryAttributes<spark_dsg::KhronosObjectAttributes>();
    if (!attrs) continue;
    long inst = -1;
    auto it = attrs->details.find("instance_id");
    if (it != attrs->details.end() && !it->second.empty()) inst = (long)it->second[0];
    const size_t nv = attrs->mesh.numVertices();
    if (nv == 0) { std::cerr << "inst=" << inst << " verts=0\n"; continue; }
    for (size_t i = 0; i < nv; ++i) {
      const auto p = attrs->bounding_box.pointToWorldFrame(attrs->mesh.pos(i));
      out << inst << "," << p.x() << "," << p.y() << "," << p.z() << "\n";
    }
    std::cerr << "inst=" << inst << " verts=" << nv << "\n";
  }
  return 0;
}
