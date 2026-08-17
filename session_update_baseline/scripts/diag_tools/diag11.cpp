#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 3) { std::cerr << "usage: diag11 <map.4dmap> <stamp_idx> [out.csv]\n"; return 2; }
  auto map = khronos::SpatioTemporalMap::load(argv[1]);
  if (!map) { std::cerr << "load failed\n"; return 2; }
  const size_t idx = std::atoi(argv[2]);
  const auto stamps = map->stamps();
  if (idx >= stamps.size()) { std::cerr << "idx out of range\n"; return 2; }
  const auto dsg = map->getDsgPtr(stamps[idx]);
  if (!dsg || !dsg->hasMesh() || !dsg->mesh()) { std::cerr << "no mesh\n"; return 2; }
  const auto& m = *dsg->mesh();
  std::ostream* out = &std::cout;
  std::ofstream file;
  if (argc > 3) { file.open(argv[3]); out = &file; }
  for (size_t i = 0; i < m.numVertices(); ++i) {
    const auto p = m.pos(i);
    const uint64_t st = (m.has_timestamps && i < m.stamps.size()) ? m.stamps[i] : 0;
    *out << p.x() << "," << p.y() << "," << p.z() << "," << st << "\n";
  }
  std::cerr << "stamp_idx=" << idx << " verts=" << m.numVertices() << "\n";
  return 0;
}
