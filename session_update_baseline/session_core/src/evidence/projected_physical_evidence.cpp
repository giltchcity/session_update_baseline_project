#include "session_core/evidence/observed_absence.h"
#include "session_core/evidence/predictive_surface.h"
#include "session_core/surface/surface_sampling.h"
#include "khronos/backend/change_detection/ray_verificator.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_set>

#include <glog/logging.h>
#include <nlohmann/json.hpp>

namespace khronos {

using measurement::SurfaceVote;

// README (7): Beta moment inversion, valid for 0 < mean < 1 and 0 < variance < mean (1 - mean).
bool fitNormalRoundBeta(const NormalRoundStats& stats, double& a, double& b) {
  if (!(stats.count > 0)) return false;
  const double mean = stats.sum / stats.count;
  const double variance = stats.sum_sq / stats.count - mean * mean;
  if (!(mean > 0 && mean < 1 && variance > 0 && variance < mean * (1 - mean))) return false;
  const double nu = mean * (1 - mean) / variance - 1;
  a = mean * nu;
  b = (1 - mean) * nu;
  return std::isfinite(a) && std::isfinite(b) && a > 0 && b > 0;
}

namespace {
using Json = nlohmann::json;

struct PlacementCalibration {
  uint64_t processed = 0;
  NormalRoundStats local;
  bool has_initial = false;
  double initial_a = 1, initial_b = 1;
};

Json encodeStats(const NormalRoundStats& value) {
  return Json::array({value.count, value.sum, value.sum_sq});
}

NormalRoundStats decodeStats(const Json& value) {
  if (!value.is_array() || value.size() != 3) throw std::invalid_argument("Invalid round statistics");
  NormalRoundStats result{value.at(0).get<double>(), value.at(1).get<double>(),
                          value.at(2).get<double>()};
  if (!std::isfinite(result.count) || !std::isfinite(result.sum) || !std::isfinite(result.sum_sq) ||
      result.count < 0 || result.sum < 0 || result.sum_sq < 0 || result.sum > result.count ||
      result.sum_sq > result.sum) {
    throw std::invalid_argument("Inconsistent round statistics");
  }
  return result;
}
}  // namespace

struct ObservedAbsenceModel::Impl {
  mutable std::mutex mutex;
  measurement::ErrorModel psi;
  NormalRoundStats population;
  double identity_conflicts = 0, identity_total = 0;
  std::map<StateKey, PlacementCalibration> states;
};

ObservedAbsenceModel::ObservedAbsenceModel() : impl_(std::make_unique<Impl>()) {}
ObservedAbsenceModel::~ObservedAbsenceModel() = default;

void ObservedAbsenceModel::setErrorModel(measurement::ErrorModel psi) {
  if (!psi.valid()) throw std::invalid_argument("Invalid effective range error model");
  for (const double value : psi.sigma) {
    if (!std::isfinite(value) || value < 0) throw std::invalid_argument("Invalid range error");
  }
  if (!std::isfinite(psi.zeta) || psi.zeta <= -1) throw std::invalid_argument("Invalid depth scale");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->psi = std::move(psi);
}

measurement::ErrorModel ObservedAbsenceModel::errorModel() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->psi;
}

bool ObservedAbsenceModel::hasErrorModel() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->psi.valid();
}

std::pair<double, double> ObservedAbsenceModel::normalModel(const StateKey& key) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto& state = impl_->states[key];
  if (!state.has_initial) {
    double a = 1, b = 1;
    if (!fitNormalRoundBeta(impl_->population, a, b)) a = b = 1;
    state.initial_a = a;
    state.initial_b = b;
    state.has_initial = true;
  }
  double a, b;
  if (fitNormalRoundBeta(state.local, a, b)) return {a, b};
  return {state.initial_a, state.initial_b};
}

double ObservedAbsenceModel::populationNormalMean() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  double a = 1, b = 1;
  if (!fitNormalRoundBeta(impl_->population, a, b)) return 0.5;
  return a / (a + b);
}

uint64_t ObservedAbsenceModel::processed(const StateKey& key) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto found = impl_->states.find(key);
  return found == impl_->states.end() ? 0 : found->second.processed;
}

void ObservedAbsenceModel::setProcessed(const StateKey& key, uint64_t stamp) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto& state = impl_->states[key];
  state.processed = std::max(state.processed, stamp);
}

void ObservedAbsenceModel::recordNormalRound(const StateKey& key, double fraction) {
  if (!(fraction >= 0 && fraction <= 1)) throw std::invalid_argument("Invalid round fraction");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->states[key].local.add(fraction);
  impl_->population.add(fraction);
}

void ObservedAbsenceModel::recordIdentityRound(size_t conflicts, size_t total) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->identity_conflicts += static_cast<double>(conflicts);
  impl_->identity_total += static_cast<double>(total);
}

void ObservedAbsenceModel::retain(const LiveStates& live) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  for (auto it = impl_->states.begin(); it != impl_->states.end();) {
    if (live.count(it->first)) ++it;
    else it = impl_->states.erase(it);
  }
  // A retired placement keeps its contribution to the population; its own statistics are released.
}

bool ObservedAbsenceModel::saveSensorStatistics(const std::string& path) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto temp = path + ".tmp";
  {
    std::ofstream out(temp);
    out << std::setprecision(17) << "NORMAL_ROUND_FRACTION_V1 " << impl_->population.count << ' '
        << impl_->population.sum << ' ' << impl_->population.sum_sq << '\n';
    out << "IDENTITY_CONFLICT_RAYS " << impl_->identity_conflicts << ' ' << impl_->identity_total
        << '\n';
    out << "RANGE_ERROR_MODEL " << impl_->psi.range_bin << ' ' << impl_->psi.zeta;
    for (const double value : impl_->psi.sigma) out << ' ' << value;
    out << '\n';
    out.close();
    if (!out) return false;
  }
  std::error_code error;
  std::filesystem::rename(temp, path, error);
  return !error;
}

bool ObservedAbsenceModel::loadSensorStatistics(const std::string& path) {
  std::ifstream in(path);
  std::string tag;
  if (!(in >> tag)) return false;
  if (tag != "NORMAL_ROUND_FRACTION_V1") {
    // README s7.1: an older format is initialised with the declared defaults.
    LOG(WARNING) << "Sensor statistics '" << path << "' use an older format; starting empty";
    return true;
  }
  NormalRoundStats loaded;
  if (!(in >> loaded.count >> loaded.sum >> loaded.sum_sq)) return false;
  try {
    loaded = decodeStats(encodeStats(loaded));
  } catch (const std::exception&) {
    return false;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->population = loaded;
  impl_->states.clear();
  return true;
}

void ObservedAbsenceModel::save(const std::string& path, uint64_t boundary,
                                const LiveStates& live) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  Json records = Json::array();
  for (const auto& [key, state] : impl_->states) {
    if (!live.count(key)) continue;
    if (state.processed > boundary) throw std::logic_error("Evidence exceeds saved boundary");
    records.push_back(Json{{"physical", key.first}, {"key", key.second},
                           {"processed", state.processed}, {"local", encodeStats(state.local)},
                           {"initial", Json::array({state.has_initial, state.initial_a,
                                                    state.initial_b})}});
  }
  const Json psi{{"sigma", impl_->psi.sigma}, {"range_bin", impl_->psi.range_bin},
                 {"zeta", impl_->psi.zeta}};
  const Json packet{{"schema", 3}, {"boundary", boundary}, {"psi", psi},
                    {"population", encodeStats(impl_->population)},
                    {"identity", Json::array({impl_->identity_conflicts, impl_->identity_total})},
                    {"states", std::move(records)}};
  const auto bytes = Json::to_cbor(packet);
  const auto temporary = path + ".tmp";
  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  out.exceptions(std::ios::badbit | std::ios::failbit);
  out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  out.close();
  std::filesystem::rename(temporary, path);
}

void ObservedAbsenceModel::load(const std::string& path, uint64_t boundary,
                                const LiveStates& live) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open evidence checkpoint: " + path);
  const auto packet = Json::from_cbor(in);
  if (packet.at("schema").get<unsigned>() != 3) {
    // README s7.1: an older format is initialised with the declared defaults.
    LOG(WARNING) << "Evidence checkpoint '" << path << "' uses an older format; starting empty";
    return;
  }
  if (packet.at("boundary").get<uint64_t>() != boundary)
    throw std::invalid_argument("Evidence checkpoint boundary mismatch");
  measurement::ErrorModel psi;
  psi.sigma = packet.at("psi").at("sigma").get<std::vector<double>>();
  psi.range_bin = packet.at("psi").at("range_bin").get<double>();
  psi.zeta = packet.at("psi").at("zeta").get<double>();
  const auto population = decodeStats(packet.at("population"));
  std::map<StateKey, PlacementCalibration> staged;
  for (const auto& item : packet.at("states")) {
    const StateKey key{item.at("physical").get<size_t>(), item.at("key").get<uint64_t>()};
    if (!key.second || !live.count(key) || staged.count(key))
      throw std::invalid_argument("Evidence checkpoint state ownership mismatch");
    PlacementCalibration state;
    state.processed = item.at("processed").get<uint64_t>();
    if (state.processed > boundary) throw std::invalid_argument("Invalid evidence watermark");
    state.local = decodeStats(item.at("local"));
    const auto& initial = item.at("initial");
    state.has_initial = initial.at(0).get<bool>();
    state.initial_a = initial.at(1).get<double>();
    state.initial_b = initial.at(2).get<double>();
    if (!(state.initial_a > 0 && state.initial_b > 0))
      throw std::invalid_argument("Invalid initial normal model");
    staged.emplace(key, state);
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (psi.valid()) impl_->psi = std::move(psi);
  impl_->population = population;
  impl_->identity_conflicts = packet.at("identity").at(0).get<double>();
  impl_->identity_total = packet.at("identity").at(1).get<double>();
  impl_->states = std::move(staged);
}

RayVerificator::CheckResult RayVerificator::checkProjectedPhysical(
    const Point& point, const size_t physical_id,
    const PhysicalEvidenceSnapshot& evidence_snapshot,
    const uint64_t earliest, const uint64_t latest, CheckDetails* details) const {
  CheckResult result;
  if (!point.allFinite()) { ++result.reasons.invalid; return result; }
  if (!evidence_snapshot) return result;
  const auto psi = absence_model_->errorModel();
  for (const auto stamp : evidence_snapshot->timestamps(earliest, latest)) {
    const auto p = evidence_snapshot->project(stamp, point);
    const auto vote = measurement::classifySurfaceMeasurement(p, psi);
    if (vote == SurfaceVote::Unavailable) continue;
    auto decision = CheckDetails::Result::kOccludded;
    switch (vote) {
      case SurfaceVote::Supported: {
        // README s4: presence is geometric; the identity of a coincident echo is only recorded.
        result.present.push_back(stamp);
        decision = CheckDetails::Result::kMatch;
        const auto& e = p.endpoint;
        if (e.type == EndpointClass::kPhysical && static_cast<size_t>(e.physical_id) == physical_id)
          ++result.reasons.same_id;
        else if (e.type == EndpointClass::kPhysical) ++result.reasons.different_id;
        else if (e.type == EndpointClass::kBackground) ++result.reasons.background_replacement;
        else ++result.reasons.unidentified_object;
        break;
      }
      case SurfaceVote::Free:
        result.absent.push_back(stamp);
        ++result.reasons.free_space;
        decision = CheckDetails::Result::kAbsent;
        break;
      case SurfaceVote::Occluded:
        result.inconclusive.push_back(stamp);
        ++result.reasons.geometric_occlusion;
        break;
      case SurfaceVote::Invalid:
        result.inconclusive.push_back(stamp);
        ++result.reasons.invalid;
        break;
      case SurfaceVote::Unavailable: break;
    }
    if (details) {
      const float q = std::isfinite(p.query_range_m) && p.query_range_m > 0 ? p.query_range_m : 0.f;
      const float depth = std::isfinite(p.endpoint.measured_depth_m) &&
          p.endpoint.measured_depth_m > 0 ? p.endpoint.measured_depth_m : 0.f;
      const Point source = point - q * p.view_direction_world;
      details->start.push_back(source);
      details->end.push_back(source + depth * p.view_direction_world);
      details->range.push_back(depth);
      details->result.push_back(decision);
    }
  }
  return result;
}

RayVerificator::SurfaceEvidenceCounts RayVerificator::countProjectedPhysicalSurface(
    const size_t physical_id, const spark_dsg::Mesh& mesh, const BoundingBox& bbox,
    const PhysicalEvidenceSnapshot& evidence_snapshot, const float cell_size,
    const uint64_t earliest, const uint64_t latest) const {
  SurfaceEvidenceCounts result;
  if (!evidence_snapshot) return result;
  const auto psi = absence_model_->errorModel();
  Points samples;
  for (const auto& sample : sampleSurface(mesh, bbox, cell_size,
                                          std::numeric_limits<size_t>::max()))
    samples.push_back(sample.point);
  result.surface_samples = samples.size();
  const auto stamps = evidence_snapshot->timestamps(earliest, latest);
  std::unordered_set<size_t> support, penetration;
  for (const auto& point : samples) {
    bool covered = false;
    for (size_t frame = 0; frame < stamps.size(); ++frame) {
      const auto p = evidence_snapshot->project(stamps[frame], point);
      const auto vote = measurement::classifySurfaceMeasurement(p, psi);
      if (vote == SurfaceVote::Unavailable) continue;
      covered = true;
      // One actual (frame,pixel) is one measurement, however many samples project onto it.
      const size_t key = (frame << 32) | static_cast<size_t>(p.pixel_index);
      switch (vote) {
        case SurfaceVote::Supported: {
          result.latest_support_stamp = std::max(result.latest_support_stamp, stamps[frame]);
          support.insert(key);
          ++result.supported_votes;
          const auto& e = p.endpoint;
          if (!(e.type == EndpointClass::kPhysical &&
                static_cast<size_t>(e.physical_id) == physical_id))
            ++result.identity_conflict_rays;
          break;
        }
        case SurfaceVote::Free:
          penetration.insert(key);
          ++result.free_space_votes;
          if (result.first_penetration_stamp == 0 || stamps[frame] < result.first_penetration_stamp)
            result.first_penetration_stamp = stamps[frame];
          break;
        case SurfaceVote::Occluded: ++result.occluded_votes; break;
        default: break;
      }
    }
    if (!covered) ++result.unobserved_samples;
  }
  result.support_rays = support.size();
  result.contradiction_rays = penetration.size();
  return result;
}

RayVerificator::SurfaceEvidenceCounts RayVerificator::countCurrentPhysicalSurface(
    const size_t physical_id, const spark_dsg::Mesh& mesh, const BoundingBox& bbox,
    const PhysicalEvidenceSnapshot& snapshot, const uint64_t latest, const float cell_size,
    const uint64_t state_birth, const uint64_t evidence_key, bool* projected) const {
  if (projected) *projected = snapshot && snapshot->numFrames() > 0;
  SurfaceEvidenceCounts counts;
  if (!snapshot || snapshot->numFrames() == 0) return counts;
  if (!evidence_key) throw std::invalid_argument("A placement's evidence needs its state key");
  const ObservedAbsenceModel::StateKey key{physical_id, evidence_key};
  auto& model = *absence_model_;
  // README (7a), P1: the new sources of this placement lie after its watermark and its birth.
  const uint64_t first = std::max<uint64_t>(model.processed(key) + 1, state_birth);
  if (latest >= first) {
    counts = countProjectedPhysicalSurface(physical_id, mesh, bbox, snapshot, cell_size, first,
                                           latest);
    model.recordIdentityRound(counts.identity_conflict_rays, counts.supported_votes);
  }
  model.setProcessed(key, latest);
  const size_t n = counts.support_rays + counts.contradiction_rays;
  if (n == 0) return counts;  // README (7u): no valid source, the unit factor
  // README (7u): the model held before this round predicts it.
  const auto [a, b] = model.normalModel(key);
  counts.informative = true;
  counts.normal_a = a;
  counts.normal_b = b;
  counts.fraction = static_cast<double>(counts.contradiction_rays) / static_cast<double>(n);
  const double half = 0.5 / static_cast<double>(n);
  const double x0 = std::max(0.0, counts.fraction - half), x1 = std::min(1.0, counts.fraction + half);
  counts.l_in = measurement::betaMass(a, b, x0, x1);
  counts.l_out = x1 - x0;
  return counts;
}

}  // namespace khronos
