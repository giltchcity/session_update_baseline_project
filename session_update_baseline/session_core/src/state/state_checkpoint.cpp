#include "session_core/state/persistent_object_state.h"
#include "session_core/runtime/session_bundle.h"
#include "session_core/surface/closed_object_background.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <unordered_map>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <spark_dsg/serialization/json_conversions.h>
#include "khronos/backend/update_khronos_objects_functor.h"

namespace khronos {
namespace {
using Json = nlohmann::json;

// README (7s), (15b): the element histories {k_e, j_e} as one binary blob of (cell key, hits,
// see-throughs) records.
Json encodeElements(const std::unordered_map<uint64_t, std::pair<float, float>>& elements) {
  std::vector<std::pair<uint64_t, std::pair<float, float>>> sorted(elements.begin(), elements.end());
  std::sort(sorted.begin(), sorted.end());
  std::vector<uint8_t> bytes;
  bytes.reserve(sorted.size() * (sizeof(uint64_t) + 2 * sizeof(float)));
  for (const auto& [key, history] : sorted) {
    const auto put = [&bytes](const auto& value) {
      const auto* raw = reinterpret_cast<const uint8_t*>(&value);
      bytes.insert(bytes.end(), raw, raw + sizeof(value));
    };
    put(key);
    put(history.first);
    put(history.second);
  }
  return Json::binary(std::move(bytes));
}

std::unordered_map<uint64_t, std::pair<float, float>> decodeElements(const Json& value) {
  if (!value.is_binary()) throw std::invalid_argument("Element histories are not binary");
  const auto& bytes = value.get_binary();
  constexpr size_t kRecord = sizeof(uint64_t) + 2 * sizeof(float);
  if (bytes.size() % kRecord != 0) throw std::invalid_argument("Truncated element histories");
  std::unordered_map<uint64_t, std::pair<float, float>> result;
  for (size_t offset = 0; offset < bytes.size(); offset += kRecord) {
    uint64_t key;
    float hits, through;
    std::memcpy(&key, bytes.data() + offset, sizeof(key));
    std::memcpy(&hits, bytes.data() + offset + sizeof(key), sizeof(hits));
    std::memcpy(&through, bytes.data() + offset + sizeof(key) + sizeof(hits), sizeof(through));
    if (!(hits >= 0.f) || !(through >= 0.f) || !std::isfinite(hits) || !std::isfinite(through)) {
      throw std::invalid_argument("Invalid element history");
    }
    result[key] = {hits, through};
  }
  return result;
}

void validateGeometry(const spark_dsg::Mesh& mesh) {
  const size_t n = mesh.numVertices();
  const auto field = [n](const auto& values) {
    if (!values.empty() && values.size() != n)
      throw std::invalid_argument("Checkpoint mesh attribute count mismatch");
  };
  field(mesh.colors); field(mesh.stamps); field(mesh.first_seen_stamps); field(mesh.labels);
  for (const auto& point : mesh.points)
    if (!point.allFinite()) throw std::invalid_argument("Non-finite checkpoint vertex");
  for (const auto& face : mesh.faces)
    for (const auto vertex : face)
      if (vertex >= n) throw std::invalid_argument("Invalid checkpoint face");
}
}  // namespace

void PersistentObjectState::saveCheckpoint(const std::string& path,
    const std::string& chain_path, TimeStamp boundary, const DynamicSceneGraph& chain) const {
  const auto encode = [](const Fragment& f, bool with_geometry) {
    if (!f.geometry_revision) throw std::invalid_argument("Invalid fragment geometry revision");
    std::unordered_map<uint64_t, std::pair<float, float>> elements;
    for (const auto& [key, history] : f.elements) elements[key] = {history.hits, history.through};
    Json item{{"key",f.evidence_key},{"geometry_revision",f.geometry_revision},
        {"birth",f.birth_time},{"presence_begin",f.presence_begin},{"support",f.last_support_time},
        {"input_boundary",f.input_boundary},{"track_first",f.track_first_seen},
        {"confirmed",f.last_confirmed_support},{"semantic",f.semantic_label},
        {"reconstruction_frames",f.reconstruction_frames},
        {"frame_keys",f.frame_keys},{"elements",encodeElements(elements)}};
    if (with_geometry) {
      validateGeometry(f.geometry);
      item["geometry"] = f.geometry;
      item["bbox"] = f.bbox;
      item["position"] = f.position;
    }
    return item;
  };
  Json records = Json::array();
  for (const auto& [id, state] : states_) {
    Json item{{"physical",id},{"transitioned",state.has_dynamic_history},
              {"succession_floor",state.succession_floor},{"closed_through",state.closed_through},
              {"motion_consumed",state.last_motion_consumed},
              {"sources",state.ingested_sources},{"pending",Json::array()},{"current",nullptr}};
    if (state.current) item["current"] = encode(state.fragments.at(*state.current), false);
    for (const auto& f : state.observed_new) item["pending"].push_back(encode(f, true));
    records.push_back(std::move(item));
  }
  Json obligations = Json::array();
  if (chain.hasMesh() && chain.mesh()) {
    for (const auto& item : closedObjectBackgroundObligations(*chain.mesh(),*this,map_resolution_,boundary))
      obligations.push_back(Json{{"point",item.point},{"reconstructed",item.reconstructed},
                                 {"supported",item.supported},{"odds",item.odds},
                                 {"distance",item.distance}});
  }
  const Json packet{{"background_obligations",std::move(obligations)},
                    {"schema",5},{"boundary",boundary},{"resolution",map_resolution_},
                    {"prior",prior_.toJson()},{"rounds",rounds_.toJson()},
                    {"chain_bytes",std::filesystem::file_size(chain_path)},{"objects",std::move(records)}};
  const auto bytes = Json::to_cbor(packet);
  const std::string temporary = path + ".tmp";
  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  out.exceptions(std::ios::badbit | std::ios::failbit);
  out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  out.close();
  std::filesystem::rename(temporary, path);
}

void PersistentObjectState::loadCheckpoint(const std::string& path,
    const std::string& chain_path, const DynamicSceneGraph& chain, TimeStamp boundary) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open registry checkpoint: " + path);
  const auto packet = Json::from_cbor(input);
  const auto schema = packet.at("schema").get<unsigned>();
  if ((schema < 1 || schema > 5) ||
      packet.at("boundary").get<TimeStamp>() != boundary ||
      (packet.contains("chain_bytes") &&
       packet.at("chain_bytes").get<uintmax_t>() != std::filesystem::file_size(chain_path))) {
    throw std::invalid_argument("Registry checkpoint does not match its chain map");
  }
  std::map<size_t,const KhronosObjectAttributes*> geometry;
  if (chain.hasLayer(DsgLayers::OBJECTS)) {
    for (const auto& [node_id,node] : chain.getLayer(DsgLayers::OBJECTS).nodes()) {
      const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
      if (!attrs) continue;
      const auto id = UpdateKhronosObjectsFunctor::physicalInstanceId(*attrs);
      if (id && !geometry.emplace(*id,attrs).second)
        throw std::invalid_argument("Duplicate physical identity in checkpoint chain");
    }
  }
  // README s7.1: an older record is initialised with the cold-start statistics (Jeffreys prior,
  // neutral round model) of principles 2 and 6.
  model::PersistencePrior prior;
  model::RoundModel rounds;
  if (schema >= 5) {
    prior = model::PersistencePrior::fromJson(packet.at("prior"));
    rounds = model::RoundModel::fromJson(packet.at("rounds"));
  }
  std::set<uint64_t> keys;
  uint64_t maximum_key = 0;
  const auto decode = [&](const Json& item, const KhronosObjectAttributes* attrs) {
    Fragment f;
    f.evidence_key = item.at("key").get<uint64_t>();
    // Schema 1 did not retain geometry versions or relation measurements.
    // Its restored geometry starts a new revision domain with empty evidence caches.
    f.geometry_revision = schema >= 2 ? item.at("geometry_revision").get<uint64_t>() : 1;
    if (!f.geometry_revision) throw std::invalid_argument("Invalid fragment geometry revision");
    if (!f.evidence_key || !keys.insert(f.evidence_key).second)
      throw std::invalid_argument("Invalid or duplicate fragment evidence key");
    maximum_key = std::max(maximum_key,f.evidence_key);
    f.birth_time = item.at("birth").get<TimeStamp>();
    f.presence_begin = schema >= 5 ? item.at("presence_begin").get<TimeStamp>() : f.birth_time;
    f.last_support_time = item.at("support").get<TimeStamp>();
    f.input_boundary = boundary;
    f.track_first_seen = item.at("track_first").get<TimeStamp>();
    f.last_confirmed_support = schema >= 3 ? item.at("confirmed").get<TimeStamp>() : 0;
    f.semantic_label = item.at("semantic").get<int>();
    f.reconstruction_frames = item.at("reconstruction_frames").get<size_t>();
    if (schema >= 4) {
      f.frame_keys = item.value("frame_keys", std::vector<TimeStamp>{});
      if (!std::is_sorted(f.frame_keys.begin(), f.frame_keys.end()))
        throw std::invalid_argument("Unsorted frame keys");
    }
    if (schema >= 5) {
      for (const auto& [key, history] : decodeElements(item.at("elements"))) {
        f.elements[key] = ElementHistory{history.first, history.second};
      }
    }
    // README (5r), (12): a restored placement starts this session at Lambda_0 = q^g / (1 - q^g).
    f.inherited = true;
    f.gap_pending = true;
    f.filter_time = 0;
    f.exposure_clock = 0;
    if (f.last_support_time < f.birth_time || f.last_support_time > boundary || f.last_confirmed_support > boundary ||
        f.track_first_seen > f.birth_time)
      throw std::invalid_argument("Checkpoint fragment time lies outside its input domain");
    if (attrs) {
      f.geometry = attrs->mesh; f.bbox = attrs->bounding_box; f.position = attrs->position;
    } else {
      f.geometry = item.at("geometry").get<spark_dsg::Mesh>();
      f.bbox = item.at("bbox").get<BoundingBox>();
      f.position = item.at("position").get<Eigen::Vector3d>();
    }
    validateGeometry(f.geometry);
    if (!f.position.allFinite() || !f.bbox.isValid() || f.geometry.numVertices() == 0)
      throw std::invalid_argument("Checkpoint live fragment has invalid geometry");
    return f;
  };
  // The gap probability of a restored placement comes from the restored prior.
  std::swap(prior_, prior);
  std::swap(rounds_, rounds);
  decltype(states_) restored;
  try {
    for (const auto& item : packet.at("objects")) {
      const auto id = item.at("physical").get<size_t>();
      if (!id || restored.count(id)) throw std::invalid_argument("Duplicate registry physical ID");
      PhysicalState state;
      if (schema >= 3 || item.contains("sources"))
        state.ingested_sources = item.at("sources").get<std::set<std::string>>();
      state.has_dynamic_history = item.at("transitioned").get<bool>();
      state.succession_floor = item.at("succession_floor").get<TimeStamp>();
      state.closed_through = schema >= 5 ? item.at("closed_through").get<TimeStamp>()
                                         : state.succession_floor;
      state.last_motion_consumed = item.at("motion_consumed").get<TimeStamp>();
      if (state.last_motion_consumed > boundary) throw std::invalid_argument("Invalid motion watermark");
      if (state.succession_floor > boundary || state.closed_through > boundary)
        throw std::invalid_argument("Invalid succession floor");
      if (!item.at("current").is_null()) {
        const auto found = geometry.find(id);
        if (found == geometry.end()) throw std::invalid_argument("Checkpoint current has no chain geometry");
        auto fragment = decode(item.at("current"),found->second);
        fragment.filter = newFilter(id, fragment);
        state.fragments.push_back(std::move(fragment));
        state.current = 0;
      }
      for (const auto& pending : item.at("pending")) {
        auto fragment = decode(pending,nullptr);
        fragment.filter = newFilter(id, fragment);
        state.observed_new.push_back(std::move(fragment));
      }
      restored.emplace(id,std::move(state));
    }
  } catch (...) {
    std::swap(prior_, prior);
    std::swap(rounds_, rounds);
    throw;
  }
  const float resolution = packet.at("resolution").get<float>();
  if (!std::isfinite(resolution) || resolution <= 0) {
    std::swap(prior_, prior);
    std::swap(rounds_, rounds);
    throw std::invalid_argument("Invalid checkpoint map resolution");
  }
  std::vector<BackgroundObligation> restored_obligations;
  for (const auto& item : packet.at("background_obligations")) {
    BackgroundObligation value{item.at("point").get<Point>(),
        item.at("reconstructed").get<TimeStamp>(),item.at("supported").get<TimeStamp>(),
        schema >= 5 ? item.at("odds").get<double>() : 0.0,
        schema >= 5 ? item.at("distance").get<float>() : 0.f};
    if (!value.point.allFinite() || !value.reconstructed ||
        value.supported < value.reconstructed || value.supported > boundary) {
      std::swap(prior_, prior);
      std::swap(rounds_, rounds);
      throw std::invalid_argument("Invalid background obligation");
    }
    restored_obligations.push_back(std::move(value));
  }
  reserveEvidenceKeys(maximum_key);
  states_ = std::move(restored);
  background_obligations_ = std::move(restored_obligations);
  map_resolution_ = resolution;
}
}  // namespace khronos
