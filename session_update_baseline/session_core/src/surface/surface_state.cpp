#include "session_core/surface/session_refusion.h"

#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "khronos/backend/update_khronos_objects_functor.h"
#include "khronos/utils/khronos_attribute_utils.h"

namespace khronos {
namespace {

// README appendix: file continuation is checked by the record count, never by a file hash.
constexpr std::array<char, 8> kMagic{{'S', 'E', 'P', 'S', '0', '0', '0', '2'}};
constexpr std::array<char, 8> kLegacyMagic{{'S', 'E', 'P', 'S', '0', '0', '0', '1'}};

void put32(unsigned char* out, uint32_t value) {
  for (size_t i = 0; i < 4; ++i) out[i] = static_cast<unsigned char>(value >> (8 * i));
}
uint32_t get32(const unsigned char* in) {
  uint32_t value = 0;
  for (size_t i = 0; i < 4; ++i) value |= static_cast<uint32_t>(in[i]) << (8 * i);
  return value;
}
uint32_t floatBits(float value) {
  static_assert(sizeof(float) == sizeof(uint32_t) && std::numeric_limits<float>::is_iec559);
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::array<unsigned char, 4> encodeError(float error) {
  if (!std::isfinite(error) || error < 0.f) throw std::invalid_argument("Invalid surface error");
  std::array<unsigned char, 4> encoded;
  put32(encoded.data(), floatBits(error));
  return encoded;
}

}  // namespace

SessionRefusion::Surface SessionRefusion::fromDsg(const DynamicSceneGraph& dsg) {
  Surface result;
  auto append = [&](const spark_dsg::Mesh& mesh, const KhronosObjectAttributes* attrs,
                    uint32_t physical) {
    if (result.vertices.size() + mesh.numVertices() > std::numeric_limits<uint32_t>::max()) {
      throw std::length_error("Surface exceeds 32-bit vertex indexing");
    }
    const auto offset = static_cast<uint32_t>(result.vertices.size());
    const auto append_field = [&](const auto& source, auto& target) {
      if (!source.empty() && source.size() != mesh.numVertices()) {
        throw std::invalid_argument("Surface source attribute count mismatch");
      }
      if (source.empty()) target.resize(target.size() + mesh.numVertices());
      else target.insert(target.end(), source.begin(), source.end());
    };
    for (const auto& point : mesh.points) {
      result.vertices.push_back(attrs ? attrs->bounding_box.pointToWorldFrame(point) : point);
    }
    append_field(mesh.stamps, result.stamps);
    append_field(mesh.first_seen_stamps, result.first_seen_stamps);
    append_field(mesh.colors, result.colors);
    append_field(mesh.labels, result.labels);
    for (const auto& face : mesh.faces) {
      for (const auto vertex : face) {
        if (vertex >= mesh.numVertices()) throw std::invalid_argument("Invalid DSG face");
      }
      result.faces.push_back({offset + static_cast<uint32_t>(face[0]),
                              offset + static_cast<uint32_t>(face[1]),
                              offset + static_cast<uint32_t>(face[2])});
      result.face_physical.push_back(physical);
    }
  };
  if (dsg.hasMesh() && dsg.mesh()) append(*dsg.mesh(), nullptr, 0);
  if (dsg.hasLayer(DsgLayers::OBJECTS)) {
    for (const auto& [id, node] : dsg.getLayer(DsgLayers::OBJECTS).nodes()) {
      (void)id;
      const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
      if (attrs && hasCurrentObjectMesh(*attrs)) {
        append(attrs->mesh, attrs, static_cast<uint32_t>(
            UpdateKhronosObjectsFunctor::physicalInstanceId(*attrs).value_or(0)));
      }
    }
  }
  return result;
}

// README (10), s7: the uncertainty of every output face, in the order of fromDsg; the file is
// bound to its surface by the face count.
void SessionRefusion::saveSurfaceError(const std::string& path, const Surface& surface) {
  if (surface.face_error.size() != surface.faces.size()) {
    throw std::invalid_argument("Surface error count mismatch");
  }
  const auto temporary = path + ".tmp";
  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  out.exceptions(std::ios::failbit | std::ios::badbit);
  out.write(kMagic.data(), kMagic.size());
  std::array<unsigned char, 8> count;
  const auto n = static_cast<uint64_t>(surface.faces.size());
  for (size_t i = 0; i < count.size(); ++i) count[i] = static_cast<unsigned char>(n >> (8 * i));
  out.write(reinterpret_cast<const char*>(count.data()), count.size());
  for (const float error : surface.face_error) {
    const auto bytes = encodeError(error);
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
  out.close();
  std::filesystem::rename(temporary, path);
}

void SessionRefusion::loadSurfaceError(const std::string& path, Surface& surface) {
  if (!std::filesystem::exists(path)) {
    surface.face_error.assign(surface.faces.size(), 0.f);
    return;
  }
  std::ifstream in(path, std::ios::binary);
  in.exceptions(std::ios::failbit | std::ios::badbit);
  std::array<char, 8> magic;
  std::array<unsigned char, 8> count;
  in.read(magic.data(), magic.size());
  in.read(reinterpret_cast<char*>(count.data()), count.size());
  uint64_t n = 0;
  for (size_t i = 0; i < count.size(); ++i) n |= static_cast<uint64_t>(count[i]) << (8 * i);
  const bool legacy = magic == kLegacyMagic;
  if ((magic != kMagic && !legacy) || n != surface.faces.size()) {
    throw std::runtime_error("Surface error file does not match the loaded map: " + path);
  }
  if (legacy) {
    // README s7.1: an older file is read as it was written; its two digests are skipped.
    std::array<char, 64> digests;
    in.read(digests.data(), digests.size());
  }
  std::vector<float> errors(surface.faces.size());
  for (float& error : errors) {
    std::array<unsigned char, 4> bytes;
    in.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    const uint32_t bits = get32(bytes.data());
    std::memcpy(&error, &bits, sizeof(error));
    if (!std::isfinite(error) || error < 0.f) throw std::runtime_error("Invalid surface error payload");
  }
  // peek sets eofbit at the expected end; exceptions concern failed reads and I/O errors.
  if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Trailing surface error data");
  surface.face_error = std::move(errors);
}

}  // namespace khronos
