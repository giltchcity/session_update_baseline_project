#include "khronos/backend/reconciliation/session_surface.h"

#include <cstring>
#include <fstream>

#include "khronos/backend/update_khronos_objects_functor.h"
#include "khronos/utils/khronos_attribute_utils.h"

namespace khronos {

namespace {
constexpr char kMagic[4] = {'K', 'S', 'E', 'R'};
constexpr uint32_t kVersion = 1;
}  // namespace

bool SessionSurface::consistent() const {
  if (identity.size() != faces.size() || error.size() != faces.size()) return false;
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
      surface.error.push_back(0.f);
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

bool SessionSurface::saveErrors(const std::string& path, const std::vector<float>& error) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  const uint64_t n = error.size();
  out.write(kMagic, 4);
  out.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(reinterpret_cast<const char*>(error.data()), static_cast<std::streamsize>(n * sizeof(float)));
  return static_cast<bool>(out);
}

bool SessionSurface::loadErrors(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  char magic[4];
  uint32_t version = 0;
  uint64_t n = 0;
  if (!in.read(magic, 4) || std::memcmp(magic, kMagic, 4) != 0 ||
      !in.read(reinterpret_cast<char*>(&version), sizeof(version)) || version != kVersion ||
      !in.read(reinterpret_cast<char*>(&n), sizeof(n)) || n != faces.size()) {
    return false;
  }
  std::vector<float> values(n);
  if (!in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(n * sizeof(float)))) {
    return false;
  }
  error = std::move(values);
  return true;
}

}  // namespace khronos
