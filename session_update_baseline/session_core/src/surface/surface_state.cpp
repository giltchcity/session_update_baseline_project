#include "session_core/surface/session_refusion.h"

#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "khronos/backend/update_khronos_objects_functor.h"
#include "khronos/utils/khronos_attribute_utils.h"

namespace khronos {
namespace {

// README appendix: file continuation is checked by the record count, never by a file hash.
constexpr std::array<char, 8> kMagic{{'S', 'E', 'P', 'S', '0', '0', '0', '4'}};
constexpr std::array<char, 8> kPendingMagic{{'S', 'E', 'P', 'S', '0', '0', '0', '3'}};
constexpr std::array<char, 8> kErrorOnlyMagic{{'S', 'E', 'P', 'S', '0', '0', '0', '2'}};
constexpr std::array<char, 8> kOldFormatMagic{{'S', 'E', 'P', 'S', '0', '0', '0', '1'}};

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

namespace {

// Raw little-endian bytes of a trivially copyable array, as a CBOR/JSON binary blob.
template <typename T>
std::vector<uint8_t> bytesOf(const std::vector<T>& values) {
  static_assert(std::is_trivially_copyable_v<T>);
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  if (!values.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

template <typename T>
std::vector<T> valuesOf(const nlohmann::json& value, size_t count) {
  std::vector<T> values;
  if (value.is_null()) return values;
  const auto& bytes = value.get_binary();
  if (bytes.size() != count * sizeof(T)) throw std::invalid_argument("Invalid hidden record array");
  values.resize(count);
  if (count) std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

}  // namespace

// Join an optional per-vertex or per-face array of `other` to that of `target`: both empty stay
// empty, otherwise both are filled to their full size first.
template <typename T>
static void joinArray(std::vector<T>& target, size_t target_size, const std::vector<T>& source,
                      size_t source_size) {
  if (target.empty() && source.empty()) return;
  target.resize(target_size);
  if (source.empty()) target.resize(target_size + source_size);
  else target.insert(target.end(), source.begin(), source.end());
}

void SessionRefusion::Surface::append(const Surface& other) {
  if (other.faces.empty()) return;
  if (vertices.size() + other.vertices.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::length_error("Surface exceeds 32-bit vertex indexing");
  }
  const auto offset = static_cast<uint32_t>(vertices.size());
  const size_t vertex_count = vertices.size(), face_count = faces.size();
  joinArray(stamps, vertex_count, other.stamps, other.vertices.size());
  joinArray(first_seen_stamps, vertex_count, other.first_seen_stamps, other.vertices.size());
  joinArray(colors, vertex_count, other.colors, other.vertices.size());
  joinArray(labels, vertex_count, other.labels, other.vertices.size());
  joinArray(face_error, face_count, other.face_error, other.faces.size());
  joinArray(face_hits, face_count, other.face_hits, other.faces.size());
  joinArray(face_through, face_count, other.face_through, other.faces.size());
  joinArray(face_rho, face_count, other.face_rho, other.faces.size());
  joinArray(face_zeta, face_count, other.face_zeta, other.faces.size());
  vertices.insert(vertices.end(), other.vertices.begin(), other.vertices.end());
  for (const auto& face : other.faces) {
    faces.push_back({face[0] + offset, face[1] + offset, face[2] + offset});
  }
  face_physical.insert(face_physical.end(), other.face_physical.begin(), other.face_physical.end());
}

nlohmann::json SessionRefusion::Surface::toJson() const {
  std::vector<float> positions;
  positions.reserve(vertices.size() * 3);
  for (const auto& v : vertices) positions.insert(positions.end(), {v.x(), v.y(), v.z()});
  std::vector<uint32_t> indices;
  indices.reserve(faces.size() * 3);
  for (const auto& f : faces) indices.insert(indices.end(), {f[0], f[1], f[2]});
  std::vector<uint8_t> rgba;
  rgba.reserve(colors.size() * 4);
  for (const auto& c : colors) rgba.insert(rgba.end(), {c.r, c.g, c.b, c.a});
  const auto binary = [](std::vector<uint8_t> bytes) { return nlohmann::json::binary(std::move(bytes)); };
  nlohmann::json value{{"vertices", vertices.size()}, {"faces", faces.size()}};
  value["positions"] = binary(bytesOf(positions));
  value["indices"] = binary(bytesOf(indices));
  value["physical"] = binary(bytesOf(face_physical));
  if (!face_error.empty()) value["error"] = binary(bytesOf(face_error));
  if (!stamps.empty()) value["stamps"] = binary(bytesOf(stamps));
  if (!first_seen_stamps.empty()) value["first_seen"] = binary(bytesOf(first_seen_stamps));
  if (!colors.empty()) value["colors"] = binary(bytesOf(rgba));
  if (!labels.empty()) value["labels"] = binary(bytesOf(labels));
  if (!face_hits.empty()) value["hits"] = binary(bytesOf(face_hits));
  if (!face_through.empty()) value["through"] = binary(bytesOf(face_through));
  if (!face_rho.empty()) value["rho"] = binary(bytesOf(face_rho));
  if (!face_zeta.empty()) value["zeta"] = binary(bytesOf(face_zeta));
  return value;
}

SessionRefusion::Surface SessionRefusion::Surface::fromJson(const nlohmann::json& value) {
  Surface surface;
  const size_t num_vertices = value.at("vertices").get<size_t>(), num_faces = value.at("faces").get<size_t>();
  const auto positions = valuesOf<float>(value.at("positions"), num_vertices * 3);
  surface.vertices.reserve(num_vertices);
  for (size_t i = 0; i < num_vertices; ++i) {
    surface.vertices.emplace_back(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]);
  }
  const auto indices = valuesOf<uint32_t>(value.at("indices"), num_faces * 3);
  for (size_t i = 0; i < num_faces; ++i) {
    const std::array<uint32_t, 3> face{indices[3 * i], indices[3 * i + 1], indices[3 * i + 2]};
    for (const auto vertex : face) {
      if (vertex >= num_vertices) throw std::invalid_argument("Invalid hidden record face");
    }
    surface.faces.push_back(face);
  }
  const auto get = [&](const char* key) -> const nlohmann::json& {
    static const nlohmann::json null;
    const auto found = value.find(key);
    return found == value.end() ? null : *found;
  };
  surface.face_physical = valuesOf<uint32_t>(value.at("physical"), num_faces);
  surface.face_error = valuesOf<float>(get("error"), num_faces);
  surface.stamps = valuesOf<spark_dsg::Mesh::Timestamps::value_type>(get("stamps"), num_vertices);
  surface.first_seen_stamps =
      valuesOf<spark_dsg::Mesh::Timestamps::value_type>(get("first_seen"), num_vertices);
  surface.labels = valuesOf<spark_dsg::Mesh::Labels::value_type>(get("labels"), num_vertices);
  const auto rgba = valuesOf<uint8_t>(get("colors"), get("colors").is_null() ? 0 : num_vertices * 4);
  for (size_t i = 0; 4 * i < rgba.size(); ++i) {
    surface.colors.emplace_back(rgba[4 * i], rgba[4 * i + 1], rgba[4 * i + 2], rgba[4 * i + 3]);
  }
  surface.face_hits = valuesOf<float>(get("hits"), num_faces);
  surface.face_through = valuesOf<float>(get("through"), num_faces);
  surface.face_rho = valuesOf<float>(get("rho"), num_faces);
  surface.face_zeta = valuesOf<float>(get("zeta"), num_faces);
  return surface;
}

// README (10), s7: the uncertainty of every output face, in the order of fromDsg; the file is
// bound to its surface by the face count.
void SessionRefusion::saveSurfaceError(const std::string& path, const Surface& surface) {
  if (surface.face_error.size() != surface.faces.size()) {
    throw std::invalid_argument("Surface error count mismatch");
  }
  // README (15b): with the element records of the faces (version 4) or without (version 2).
  const bool records = surface.face_hits.size() == surface.faces.size() &&
                       surface.face_through.size() == surface.faces.size() &&
                       surface.face_rho.size() == surface.faces.size() &&
                       surface.face_zeta.size() == surface.faces.size();
  const auto temporary = path + ".tmp";
  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  out.exceptions(std::ios::failbit | std::ios::badbit);
  const auto& magic = records ? kMagic : kErrorOnlyMagic;
  out.write(magic.data(), magic.size());
  std::array<unsigned char, 8> count;
  const auto n = static_cast<uint64_t>(surface.faces.size());
  for (size_t i = 0; i < count.size(); ++i) count[i] = static_cast<unsigned char>(n >> (8 * i));
  out.write(reinterpret_cast<const char*>(count.data()), count.size());
  const auto write_values = [&out](const std::vector<float>& values) {
    for (const float value : values) {
      const auto bytes = encodeError(value);
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  };
  write_values(surface.face_error);
  if (records) {
    write_values(surface.face_hits);
    write_values(surface.face_through);
    write_values(surface.face_rho);
    // zeta_e is signed (the depth scale of the session that made the face).
    for (const float value : surface.face_zeta) {
      if (!std::isfinite(value)) throw std::invalid_argument("Invalid surface scale");
      std::array<unsigned char, 4> bytes;
      put32(bytes.data(), floatBits(value));
      out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
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
  const bool old_format = magic == kOldFormatMagic;
  const bool with_records = magic == kMagic;
  const bool pending_records = magic == kPendingMagic;
  if ((magic != kMagic && magic != kPendingMagic && magic != kErrorOnlyMagic && !old_format) ||
      n != surface.faces.size()) {
    throw std::runtime_error("Surface error file does not match the loaded map: " + path);
  }
  if (old_format) {
    // README s7.1: an older file is read as it was written; its two digests are skipped.
    std::array<char, 64> digests;
    in.read(digests.data(), digests.size());
  }
  const auto read_values = [&](std::vector<float>& values) {
    values.assign(surface.faces.size(), 0.f);
    for (float& value : values) {
      std::array<unsigned char, 4> bytes;
      in.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
      const uint32_t bits = get32(bytes.data());
      std::memcpy(&value, &bits, sizeof(value));
      if (!std::isfinite(value) || value < 0.f) throw std::runtime_error("Invalid surface error payload");
    }
  };
  std::vector<float> errors;
  read_values(errors);
  if (with_records) {
    read_values(surface.face_hits);
    read_values(surface.face_through);
    read_values(surface.face_rho);
    surface.face_zeta.assign(surface.faces.size(), 0.f);
    for (float& value : surface.face_zeta) {
      std::array<unsigned char, 4> bytes;
      in.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
      const uint32_t bits = get32(bytes.data());
      std::memcpy(&value, &bits, sizeof(value));
      if (!std::isfinite(value)) throw std::runtime_error("Invalid surface scale payload");
    }
  } else if (pending_records) {
    // The records of the previous format: hits and see-throughs are kept, the frames that were
    // pending are not evidence of a later session.
    std::vector<float> ignored;
    read_values(surface.face_hits);
    read_values(surface.face_through);
    read_values(ignored);
    read_values(ignored);
  }
  // peek sets eofbit at the expected end; exceptions concern failed reads and I/O errors.
  if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Trailing surface error data");
  surface.face_error = std::move(errors);
}

}  // namespace khronos
