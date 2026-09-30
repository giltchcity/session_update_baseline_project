#include "session_core/state/persistent_object_state.h"
#include "session_core/runtime/session_bundle.h"
#include "session_core/surface/closed_object_background.h"

#include <algorithm>
#include <array>
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
    Json item{{"key",f.evidence_key},{"geometry_revision",f.geometry_revision},
        {"birth",f.birth_time},{"support",f.last_support_time},
        {"input_boundary",f.input_boundary},{"track_first",f.track_first_seen},
        {"confirmed",f.last_confirmed_support},{"semantic",f.semantic_label},
        {"reconstruction_frames",f.reconstruction_frames},
        {"alpha",f.alpha},{"beta",f.beta}};
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
    if (state.b_session) throw std::logic_error("Checkpoint requires completed terminal drain");
    Json item{{"physical",id},{"transitioned",state.has_dynamic_history},
              {"succession_floor",state.succession_floor},{"motion_consumed",state.last_motion_consumed},
              {"sources",state.ingested_sources},{"pending",Json::array()},{"current",nullptr}};
    if (state.current) item["current"] = encode(state.fragments.at(*state.current), false);
    for (const auto& f : state.observed_new) item["pending"].push_back(encode(f, true));
    records.push_back(std::move(item));
  }
  Json obligations = Json::array();
  if (chain.hasMesh() && chain.mesh()) {
    for (const auto& item : closedObjectBackgroundObligations(*chain.mesh(),*this,map_resolution_,boundary))
      obligations.push_back(Json{{"point",item.point},{"reconstructed",item.reconstructed},
                                 {"supported",item.supported}});
  }
  const Json packet{{"background_obligations",std::move(obligations)},
                    {"schema",4},{"boundary",boundary},{"resolution",map_resolution_},
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
  if ((schema < 1 || schema > 4) ||
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
    f.last_support_time = item.at("support").get<TimeStamp>();
    f.input_boundary = boundary;
    f.track_first_seen = item.at("track_first").get<TimeStamp>();
    f.last_confirmed_support = schema >= 3 ? item.at("confirmed").get<TimeStamp>() : 0;
    f.semantic_label = item.at("semantic").get<int>();
    f.reconstruction_frames = item.at("reconstruction_frames").get<size_t>();
    // README (7.1): stationarity of the placement is restored; an older record starts from the
    // declared initial prior.
    if (schema >= 4) {
      f.alpha = item.at("alpha").get<double>();
      f.beta = item.at("beta").get<double>();
      if (!(std::isfinite(f.alpha) && std::isfinite(f.beta) && f.alpha > 0 && f.beta > 0))
        throw std::invalid_argument("Invalid placement stationarity");
    }
    // Earlier schemas also stored empty_look/counter_look. They no longer
    // authorize state association; unknown legacy fields are intentionally ignored.
    f.requires_current_session_support = true;
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
  decltype(states_) restored;
  for (const auto& item : packet.at("objects")) {
    const auto id = item.at("physical").get<size_t>();
    if (!id || restored.count(id)) throw std::invalid_argument("Duplicate registry physical ID");
    PhysicalState state;
    if (schema >= 3 || item.contains("sources"))
      state.ingested_sources = item.at("sources").get<std::set<std::string>>();
    state.has_dynamic_history = item.at("transitioned").get<bool>();
    state.succession_floor = item.at("succession_floor").get<TimeStamp>();
    state.last_motion_consumed = item.at("motion_consumed").get<TimeStamp>();
    if (state.last_motion_consumed > boundary) throw std::invalid_argument("Invalid motion watermark");
    if (state.succession_floor > boundary) throw std::invalid_argument("Invalid succession floor");
    if (!item.at("current").is_null()) {
      const auto found = geometry.find(id);
      if (found == geometry.end()) throw std::invalid_argument("Checkpoint current has no chain geometry");
      state.fragments.push_back(decode(item.at("current"),found->second));
      state.current = 0;
    }
    for (const auto& pending : item.at("pending")) state.observed_new.push_back(decode(pending,nullptr));
    restored.emplace(id,std::move(state));
  }
  const float resolution = packet.at("resolution").get<float>();
  if (!std::isfinite(resolution) || resolution <= 0)
    throw std::invalid_argument("Invalid checkpoint map resolution");
  std::vector<BackgroundObligation> restored_obligations;
  for (const auto& item : packet.at("background_obligations")) {
    BackgroundObligation value{item.at("point").get<Point>(),
        item.at("reconstructed").get<TimeStamp>(),item.at("supported").get<TimeStamp>()};
    if (!value.point.allFinite() || !value.reconstructed ||
        value.supported < value.reconstructed || value.supported > boundary)
      throw std::invalid_argument("Invalid background obligation");
    restored_obligations.push_back(std::move(value));
  }
  reserveEvidenceKeys(maximum_key);
  states_ = std::move(restored);
  background_obligations_ = std::move(restored_obligations);
  map_resolution_ = resolution;
}
}  // namespace khronos
