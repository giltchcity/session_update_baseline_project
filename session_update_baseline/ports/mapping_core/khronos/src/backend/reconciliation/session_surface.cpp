#include "khronos/backend/reconciliation/session_surface.h"

#include <cstring>
#include <fstream>

#include "khronos/backend/update_khronos_objects_functor.h"
#include "khronos/utils/khronos_attribute_utils.h"

namespace khronos {

namespace {

constexpr char kMagic[4] = {'K', 'S', 'S', 'F'};
constexpr uint32_t kVersion = 1;

template <typename T>
void writeArray(std::ofstream& out, const std::vector<T>& values) {
  const uint64_t n = values.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(n * sizeof(T)));
}

template <typename T>
bool readArray(std::ifstream& in, std::vector<T>& values, uint64_t max_count) {
  uint64_t n = 0;
  if (!in.read(reinterpret_cast<char*>(&n), sizeof(n)) || n > max_count) return false;
  values.resize(n);
  return static_cast<bool>(
      in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(n * sizeof(T))));
}

}  // namespace

bool SessionSurface::consistent() const {
  const size_t n = faces.size();
  if (identity.size() != n || range.size() != n || scale.size() != n) return false;
  for (const auto& f : faces) {
    if (f[0] >= vertices.size() || f[1] >= vertices.size() || f[2] >= vertices.size()) return false;
  }
  return true;
}

SessionSurface SessionSurface::fromDsg(const DynamicSceneGraph& dsg) {
  SessionSurface surface;
  auto add = [&](const spark_dsg::Mesh& mesh, const KhronosObjectAttributes* attrs, uint32_t id) {
    const auto base = static_cast<uint32_t>(surface.vertices.size());
    const size_t n = mesh.numVertices();
    for (size_t i = 0; i < n; ++i) {
      surface.vertices.push_back(attrs ? attrs->bounding_box.pointToWorldFrame(mesh.pos(i))
                                       : mesh.pos(i));
    }
    for (const auto& f : mesh.faces) {
      if (f[0] >= n || f[1] >= n || f[2] >= n) continue;
      surface.faces.push_back({base + static_cast<uint32_t>(f[0]), base + static_cast<uint32_t>(f[1]),
                               base + static_cast<uint32_t>(f[2])});
      surface.identity.push_back(id);
      surface.range.push_back(0.f);
      surface.scale.push_back(0.f);
    }
  };
  if (dsg.hasMesh() && dsg.mesh()) add(*dsg.mesh(), nullptr, 0);
  if (dsg.hasLayer(DsgLayers::OBJECTS)) {
    for (const auto& [id, node] : dsg.getLayer(DsgLayers::OBJECTS).nodes()) {
      const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
      if (!attrs || !hasCurrentObjectMesh(*attrs)) continue;
      add(attrs->mesh, attrs,
          static_cast<uint32_t>(UpdateKhronosObjectsFunctor::physicalInstanceId(*attrs).value_or(0)));
    }
  }
  return surface;
}

bool SessionSurface::save(const std::string& path) const {
  if (!consistent()) return false;
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out.write(kMagic, 4);
  out.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
  writeArray(out, vertices);
  writeArray(out, faces);
  writeArray(out, identity);
  writeArray(out, range);
  writeArray(out, scale);
  return static_cast<bool>(out);
}

bool SessionSurface::load(const std::string& path, SessionSurface& surface) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  char magic[4];
  uint32_t version = 0;
  if (!in.read(magic, 4) || std::memcmp(magic, kMagic, 4) != 0 ||
      !in.read(reinterpret_cast<char*>(&version), sizeof(version)) || version != kVersion) {
    return false;
  }
  constexpr uint64_t kMax = uint64_t(1) << 32;
  SessionSurface s;
  if (!readArray(in, s.vertices, kMax) || !readArray(in, s.faces, kMax) ||
      !readArray(in, s.identity, kMax) || !readArray(in, s.range, kMax) ||
      !readArray(in, s.scale, kMax) || !s.consistent()) {
    return false;
  }
  surface = std::move(s);
  return true;
}

}  // namespace khronos
