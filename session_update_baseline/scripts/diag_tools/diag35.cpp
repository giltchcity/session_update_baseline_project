#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <khronos/utils/khronos_attribute_utils.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: diag35 <map.4dmap> <stamp_idx> [stamp_idx2 ...]\n";
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
    const auto dsg = map->getDsgPtr(stamps[idx]);
    const auto mesh = khronos::composeCurrentSceneMesh(*dsg);
    size_t in_computer_region = 0;
    size_t in_table_region = 0;
    for (size_t i = 0; i < mesh->numVertices(); ++i) {
      const auto& p = mesh->pos(i);
      if (p.x() >= 2.4f && p.x() <= 4.5f && p.z() >= 0.8f && p.z() <= 2.3f &&
          p.y() >= -1.0f && p.y() <= 1.5f) {
        ++in_computer_region;
      }
      if (p.x() >= 1.7f && p.x() <= 4.2f && p.z() >= -0.3f && p.z() <= 2.5f &&
          p.y() >= -0.5f && p.y() <= 1.2f) {
        ++in_table_region;
      }
    }
    std::cout << "f" << idx << " t=" << (stamps[idx] / 1000000000ULL) << "s"
              << " scene_verts=" << mesh->numVertices()
              << " computer_region=" << in_computer_region
              << " table_region=" << in_table_region << "\n";
  }
  return 0;
}
