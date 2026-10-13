#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <khronos/utils/khronos_attribute_utils.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <iostream>

struct Region {
  const char* name;
  float x0, x1, y0, y1, z0, z1;
};

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: diag37 <map.4dmap> <stamp_idx> [stamp_idx2 ...]\n";
    return 2;
  }
  auto map = khronos::SpatioTemporalMap::load(argv[1]);
  if (!map) {
    std::cerr << "load failed\n";
    return 2;
  }
  const Region regions[] = {
      {"A-cabinet", 0.0f, 2.0f, 0.0f, 1.2f, 1.8f, 3.4f},
      {"B-cabinet", -1.6f, 0.6f, -0.6f, 1.0f, -1.2f, 1.2f},
      {"desk-area", 1.8f, 4.7f, -1.0f, 1.0f, -0.4f, 2.6f},
  };
  const auto stamps = map->stamps();
  for (int a = 2; a < argc; ++a) {
    const size_t idx = std::atoi(argv[a]);
    if (idx >= stamps.size()) continue;
    const auto dsg = map->getDsgPtr(stamps[idx]);
    const auto mesh = khronos::composeCurrentSceneMesh(*dsg);
    const auto& bg = *dsg->mesh();
    std::cout << "f" << idx << " t=" << (stamps[idx] / 1000000000ULL)
              << "s scene_verts=" << mesh->numVertices()
              << " bg_verts=" << bg.numVertices();
    for (const auto& r : regions) {
      size_t n = 0;
      for (size_t i = 0; i < mesh->numVertices(); ++i) {
        const auto& p = mesh->pos(i);
        if (p.x() >= r.x0 && p.x() <= r.x1 && p.y() >= r.y0 && p.y() <= r.y1 &&
            p.z() >= r.z0 && p.z() <= r.z1) {
          ++n;
        }
      }
      std::cout << "  " << r.name << "=" << n;
    }
    for (const auto& r : regions) {
      size_t n = 0;
      for (size_t i = 0; i < bg.numVertices(); ++i) {
        const auto& p = bg.pos(i);
        if (p.x() >= r.x0 && p.x() <= r.x1 && p.y() >= r.y0 && p.y() <= r.y1 &&
            p.z() >= r.z0 && p.z() <= r.z1) {
          ++n;
        }
      }
      std::cout << "  bg_" << r.name << "=" << n;
    }
    std::cout << "\n";
  }
  return 0;
}
