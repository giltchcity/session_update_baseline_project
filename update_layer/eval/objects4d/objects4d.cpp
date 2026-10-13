// objects4d: Khronos 4D maps <-> plain per-snapshot object lists, so that any representation's
// timeline can be scored by the unchanged official Khronos ObjectEvaluator (Object F1).
//
//   objects4d dump  IN.4dmap OUT.obj4d                 every saved stamp: the objects of getDsgPtr(stamp)
//   objects4d build TEMPLATE.4dmap IN.obj4d OUT.4dmap  one DSG per snapshot (layers copied from TEMPLATE,
//                                                      all nodes removed, empty background mesh), one
//                                                      KhronosObjectAttributes node per object
//
// OBJ4D file (little endian): "OBJ4D002", u32 snapshots; per snapshot: u64 stamp, u32 objects; per
// object: u64 node id, i32 semantic, f32[3] box centre, f32[3] box dimensions, f32[4] box rotation
// (w, x, y, z), u32 n, u64[n] first_observed_ns, u32 m, u64[m] last_observed_ns, u32 k, f32[3k]
// surface points in world coordinates.
// A built object gets that bounding box and its points as mesh vertices in the box frame -- what
// ObjectEvaluator reads (semantic_label, bounding_box, mesh, first/last_observed_ns via isPresent).
// Built maps have an empty background mesh, so an object needs first_observed_ns > 0 to appear:
// dump replaces a static object's 0 by the time Khronos derives for it from the background mesh
// (SpatioTemporalMap::getObjectEffectiveTime: earliest first-seen stamp of a vertex within 0.5 m of
// the box), which is when it appears in the map.
// Object P/R/F1 survive the round trip exactly (validate_roundtrip.sh; official Khronos real A:
// 1.0 / 0.7689 / 0.8908 both ways). The official change metrics (Appeared*/Disappeared*) read
// first_observed_ns (0 = static) and are therefore NOT preserved; do not report them from built maps.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <khronos/spatio_temporal_map/spatio_temporal_map.h>

using khronos::KhronosObjectAttributes;
using khronos::SpatioTemporalMap;
using spark_dsg::DsgLayers;
using spark_dsg::DynamicSceneGraph;

namespace {

struct Object {
  uint64_t id = 0;
  int32_t semantic = 0;
  float box[10] = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0};  // centre, dimensions, rotation w x y z
  std::vector<uint64_t> first, last;
  std::vector<float> xyz;  // 3k
};
struct Snapshot {
  uint64_t stamp = 0;
  std::vector<Object> objects;
};

template <typename T> void put(std::ostream& o, const T& v) { o.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
template <typename T> T get(std::istream& i) { T v; i.read(reinterpret_cast<char*>(&v), sizeof(T)); if (!i) throw std::runtime_error("short OBJ4D file"); return v; }
template <typename T> void put_vec(std::ostream& o, const std::vector<T>& v, uint32_t n) { if (n) o.write(reinterpret_cast<const char*>(v.data()), sizeof(T) * v.size()); }
template <typename T> std::vector<T> get_vec(std::istream& i, size_t n) {
  std::vector<T> v(n);
  if (n) { i.read(reinterpret_cast<char*>(v.data()), sizeof(T) * n); if (!i) throw std::runtime_error("short OBJ4D file"); }
  return v;
}

void write_obj4d(const std::string& path, const std::vector<Snapshot>& snaps) {
  std::ofstream o(path, std::ios::binary);
  o.write("OBJ4D002", 8);
  put<uint32_t>(o, snaps.size());
  for (const auto& s : snaps) {
    put<uint64_t>(o, s.stamp);
    put<uint32_t>(o, s.objects.size());
    for (const auto& ob : s.objects) {
      put<uint64_t>(o, ob.id);
      put<int32_t>(o, ob.semantic);
      o.write(reinterpret_cast<const char*>(ob.box), sizeof(ob.box));
      put<uint32_t>(o, ob.first.size()); put_vec(o, ob.first, ob.first.size());
      put<uint32_t>(o, ob.last.size()); put_vec(o, ob.last, ob.last.size());
      put<uint32_t>(o, ob.xyz.size() / 3); put_vec(o, ob.xyz, ob.xyz.size());
    }
  }
}

std::vector<Snapshot> read_obj4d(const std::string& path) {
  std::ifstream i(path, std::ios::binary);
  char magic[8];
  i.read(magic, 8);
  if (!i || std::string(magic, 8) != "OBJ4D002") throw std::runtime_error("not an OBJ4D002 file: " + path);
  std::vector<Snapshot> snaps(get<uint32_t>(i));
  for (auto& s : snaps) {
    s.stamp = get<uint64_t>(i);
    s.objects.resize(get<uint32_t>(i));
    for (auto& ob : s.objects) {
      ob.id = get<uint64_t>(i);
      ob.semantic = get<int32_t>(i);
      i.read(reinterpret_cast<char*>(ob.box), sizeof(ob.box));
      ob.first = get_vec<uint64_t>(i, get<uint32_t>(i));
      ob.last = get_vec<uint64_t>(i, get<uint32_t>(i));
      ob.xyz = get_vec<float>(i, 3 * size_t(get<uint32_t>(i)));
    }
  }
  return snaps;
}

// SpatioTemporalMap::getObjectEffectiveTime for an object without an explicit time.
uint64_t mesh_time(const KhronosObjectAttributes& a, const spark_dsg::Mesh& mesh) {
  uint64_t t = std::numeric_limits<uint64_t>::max();
  if (!mesh.has_first_seen_stamps) return t;
  const Eigen::Vector3f lo = a.bounding_box.world_P_center - a.bounding_box.dimensions * 0.5f;
  const Eigen::Vector3f hi = a.bounding_box.world_P_center + a.bounding_box.dimensions * 0.5f;
  const float m = 0.5f;
  for (size_t v = 0; v < mesh.numVertices() && v < mesh.first_seen_stamps.size(); ++v) {
    const auto& p = mesh.pos(v);
    if (p.x() >= lo.x() - m && p.x() <= hi.x() + m && p.y() >= lo.y() - m && p.y() <= hi.y() + m &&
        p.z() >= lo.z() - m && p.z() <= hi.z() + m && mesh.first_seen_stamps[v] > 0)
      t = std::min<uint64_t>(t, mesh.first_seen_stamps[v]);
  }
  return t;
}

int dump(const std::string& in, const std::string& out) {
  auto map = SpatioTemporalMap::load(in);
  if (!map) throw std::runtime_error("cannot load " + in);
  std::vector<Snapshot> snaps;
  for (const auto stamp : map->stamps()) {
    const auto g = map->getDsgPtr(stamp);
    Snapshot s;
    s.stamp = stamp;
    if (g && g->hasLayer(DsgLayers::OBJECTS)) {
      for (const auto& [id, node] : g->getLayer(DsgLayers::OBJECTS).nodes()) {
        const auto* a = node->tryAttributes<KhronosObjectAttributes>();
        if (!a) throw std::runtime_error("object node without KhronosObjectAttributes");
        Object ob;
        ob.id = id;
        ob.semantic = int32_t(a->semantic_label);
        ob.first = a->first_observed_ns;
        ob.last = a->last_observed_ns;
        if (ob.first.empty() || ob.first.front() == 0) {
          const uint64_t t = g->hasMesh() ? mesh_time(*a, *g->mesh()) : std::numeric_limits<uint64_t>::max();
          if (ob.first.empty()) ob.first.push_back(t); else ob.first.front() = t;
        }
        const auto& bb = a->bounding_box;
        const Eigen::Quaternionf q(bb.world_R_center);
        const float box[10] = {bb.world_P_center.x(), bb.world_P_center.y(), bb.world_P_center.z(),
                               bb.dimensions.x(), bb.dimensions.y(), bb.dimensions.z(), q.w(), q.x(), q.y(), q.z()};
        std::copy(box, box + 10, ob.box);
        ob.xyz.reserve(3 * a->mesh.numVertices());
        for (size_t v = 0; v < a->mesh.numVertices(); ++v) {
          const Eigen::Vector3f p = a->bounding_box.pointToWorldFrame(a->mesh.pos(v));
          ob.xyz.insert(ob.xyz.end(), {p.x(), p.y(), p.z()});
        }
        s.objects.push_back(std::move(ob));
      }
    }
    snaps.push_back(std::move(s));
  }
  write_obj4d(out, snaps);
  size_t n = 0;
  for (const auto& s : snaps) n += s.objects.size();
  std::cout << "dumped " << snaps.size() << " snapshots, " << n << " object nodes" << std::endl;
  return 0;
}

int build(const std::string& tmpl_path, const std::string& in, const std::string& out) {
  auto tmpl = SpatioTemporalMap::load(tmpl_path);
  if (!tmpl || tmpl->stamps().empty()) throw std::runtime_error("cannot load template " + tmpl_path);
  auto base = tmpl->getDsgPtr(tmpl->stamps().front())->clone();
  std::vector<spark_dsg::NodeId> all;
  for (const auto& [layer_id, layer] : base->layers())
    for (const auto& [id, node] : layer->nodes()) all.push_back(id);
  for (const auto& [layer_id, partitions] : base->layer_partitions())
    for (const auto& [pid, layer] : partitions)
      for (const auto& [id, node] : layer->nodes()) all.push_back(id);
  for (const auto id : all) base->removeNode(id);

  const auto snaps = read_obj4d(in);
  SpatioTemporalMap map(SpatioTemporalMap::Config{});
  size_t n = 0;
  for (const auto& s : snaps) {
    auto dsg = base->clone();
    dsg->setMesh(std::make_shared<spark_dsg::Mesh>(true, true, true, true));
    for (const auto& ob : s.objects) {
      const size_t k = ob.xyz.size() / 3;
      const Eigen::Vector3f centre(ob.box[0], ob.box[1], ob.box[2]);
      const Eigen::Vector3f dims(ob.box[3], ob.box[4], ob.box[5]);
      const Eigen::Quaternionf rot(ob.box[6], ob.box[7], ob.box[8], ob.box[9]);
      auto a = std::make_unique<KhronosObjectAttributes>();
      a->semantic_label = ob.semantic;
      a->bounding_box = spark_dsg::BoundingBox(dims, centre, rot);
      a->position = centre.cast<double>();
      a->first_observed_ns = ob.first;
      a->last_observed_ns = ob.last;
      a->mesh.resizeVertices(k);
      for (size_t v = 0; v < k; ++v)
        a->mesh.setPos(v, a->bounding_box.pointToBoxFrame(
                              Eigen::Vector3f(ob.xyz[3 * v], ob.xyz[3 * v + 1], ob.xyz[3 * v + 2])));
      if (!dsg->emplaceNode(DsgLayers::OBJECTS, ob.id, std::move(a)))
        throw std::runtime_error("cannot add object node " + std::to_string(ob.id));
      ++n;
    }
    map.update(dsg, s.stamp);
  }
  map.finalize();
  if (!map.save(out)) throw std::runtime_error("cannot save " + out);
  std::cout << "built " << snaps.size() << " snapshots, " << n << " object nodes -> " << out << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "dump" && argc == 4) return dump(argv[2], argv[3]);
    if (mode == "build" && argc == 5) return build(argv[2], argv[3], argv[4]);
    std::cerr << "usage: objects4d dump IN.4dmap OUT.obj4d | objects4d build TEMPLATE.4dmap IN.obj4d OUT.4dmap\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "objects4d: " << e.what() << std::endl;
    return 1;
  }
}
