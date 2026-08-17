#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_attributes.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <array>

static void writePly(const std::string& path,
                     const std::vector<Eigen::Vector3f>& verts,
                     const std::vector<std::array<uint8_t,3>>& colors,
                     const std::vector<std::array<size_t,3>>& faces) {
  std::ofstream f(path);
  f << "ply\nformat ascii 1.0\n";
  f << "element vertex " << verts.size() << "\n";
  f << "property float x\nproperty float y\nproperty float z\n";
  f << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
  if (!faces.empty()) {
    f << "element face " << faces.size() << "\n";
    f << "property list uchar int vertex_indices\n";
  }
  f << "end_header\n";
  for (size_t i = 0; i < verts.size(); ++i) {
    const auto& p = verts[i];
    const auto& c = colors[i];
    f << p.x() << " " << p.y() << " " << p.z() << " "
      << int(c[0]) << " " << int(c[1]) << " " << int(c[2]) << "\n";
  }
  for (const auto& fc : faces) {
    f << "3 " << fc[0] << " " << fc[1] << " " << fc[2] << "\n";
  }
}

int main(int argc, char** argv) {
  if (argc < 3) { std::cerr << "usage: diag21 <map.4dmap> <out_prefix>\n"; return 2; }
  auto map = khronos::SpatioTemporalMap::load(argv[1]);
  if (!map) { std::cerr << "load failed\n"; return 2; }
  const auto stamps = map->stamps();
  const auto dsg = map->getDsgPtr(stamps.back());
  if (!dsg) { std::cerr << "no dsg\n"; return 2; }
  const std::string pre = argv[2];

  if (dsg->hasMesh() && dsg->mesh()) {
    const auto& m = *dsg->mesh();
    std::vector<Eigen::Vector3f> v; v.reserve(m.numVertices());
    std::vector<std::array<uint8_t,3>> c; c.reserve(m.numVertices());
    for (size_t i = 0; i < m.numVertices(); ++i) {
      v.push_back(m.pos(i));
      if (m.has_colors && i < m.colors.size()) {
        const auto& col = m.colors[i];
        c.push_back(std::array<uint8_t,3>{{col.r, col.g, col.b}});
      } else {
        c.push_back(std::array<uint8_t,3>{{200,200,200}});
      }
    }
    std::vector<std::array<size_t,3>> f;
    for (size_t i = 0; i < m.numFaces(); ++i) {
      const auto& face = m.face(i);
      f.push_back({face[0], face[1], face[2]});
    }
    writePly(pre + "_bg.ply", v, c, f);
    std::cerr << "bg verts=" << v.size() << " faces=" << f.size() << "\n";
  }

  if (dsg->hasLayer(spark_dsg::DsgLayers::OBJECTS)) {
    std::vector<Eigen::Vector3f> v;
    std::vector<std::array<uint8_t,3>> c;
    std::vector<std::array<size_t,3>> f;
    static const std::array<std::array<uint8_t,3>,20> palette = {{
      std::array<uint8_t,3>{{255,0,0}}, std::array<uint8_t,3>{{0,255,0}},
      std::array<uint8_t,3>{{0,0,255}}, std::array<uint8_t,3>{{255,255,0}},
      std::array<uint8_t,3>{{255,0,255}}, std::array<uint8_t,3>{{0,255,255}},
      std::array<uint8_t,3>{{255,128,0}}, std::array<uint8_t,3>{{128,0,255}},
      std::array<uint8_t,3>{{0,128,255}}, std::array<uint8_t,3>{{255,0,128}},
      std::array<uint8_t,3>{{128,255,0}}, std::array<uint8_t,3>{{200,200,0}},
      std::array<uint8_t,3>{{0,200,200}}, std::array<uint8_t,3>{{200,0,200}},
      std::array<uint8_t,3>{{128,128,255}}, std::array<uint8_t,3>{{255,128,128}},
      std::array<uint8_t,3>{{128,255,128}}, std::array<uint8_t,3>{{80,80,255}},
      std::array<uint8_t,3>{{255,80,80}}, std::array<uint8_t,3>{{80,255,80}}
    }};
    size_t idx = 0;
    for (const auto& [id, node] : dsg->getLayer(spark_dsg::DsgLayers::OBJECTS).nodes()) {
      auto attrs = node->tryAttributes<spark_dsg::KhronosObjectAttributes>();
      if (!attrs) continue;
      long inst = -1;
      auto it = attrs->details.find("instance_id");
      if (it != attrs->details.end() && !it->second.empty()) inst = (long)it->second[0];
      const auto& col = palette[(inst>=0&&inst<20)?inst:11];
      const size_t nv = attrs->mesh.numVertices();
      if (nv == 0) continue;
      for (size_t i = 0; i < nv; ++i) {
        const auto p = attrs->bounding_box.pointToWorldFrame(attrs->mesh.pos(i));
        v.push_back(p);
        c.push_back(std::array<uint8_t,3>{{col[0],col[1],col[2]}});
      }
      for (size_t i = 0; i < attrs->mesh.numFaces(); ++i) {
        const auto& face = attrs->mesh.face(i);
        f.push_back({face[0]+idx, face[1]+idx, face[2]+idx});
      }
      std::cerr << "object node=" << id << " inst=" << inst << " verts=" << nv << "\n";
      idx += nv;
    }
    writePly(pre + "_objects.ply", v, c, f);
    std::cerr << "objects verts=" << v.size() << " faces=" << f.size() << "\n";
  }
  return 0;
}
