#include "session_core/evidence/observed_absence.h"
#include "session_core/evidence/predictive_surface.h"
#include "session_core/surface/surface_sampling.h"
#include "khronos/backend/change_detection/ray_verificator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <stdexcept>
#include <set>
#include <tuple>
#include <nlohmann/json.hpp>

namespace khronos {
namespace {

using Vote = measurement::SurfaceVote;
using measurement::classifySurfaceMeasurement;

// README (7)--(7d): a finite set of latest surface measurements and two
// normalized predictive hypotheses. Calibration and state evidence are separate.
struct AbsenceSample {
  uint16_t identity_hits = 0;
  TimeStamp last_on_surface = 0;
  TimeStamp last_seen_through = 0;
  TimeStamp reliable_since = 0;
  uint32_t surface_pixel = 0, absence_pixel = 0;
};
using AbsenceCell = std::tuple<int64_t, int64_t, int64_t>;

// Fixed observation protocol; README section 4.1.
constexpr uint16_t kMinIdentityHits = 3;
constexpr size_t kMaxAbsenceSamples = 1500;

// Moments of a mixture of Beta(k+1,n-k+1) calibration views, README (7b).
struct Calibration {
  double count = 0;
  double first = 0;
  double second = 0;

  void add(size_t free, size_t measured) {
    const double a = static_cast<double>(free) + 1;
    const double c = static_cast<double>(measured) + 2;
    count += 1;
    first += a / c;
    second += a * (a + 1) / (c * (c + 1));
  }

  std::pair<long double, long double> shape() const {
    if (count == 0) return {1, 1};
    const long double m = static_cast<long double>(first) / count;
    const long double empirical = static_cast<long double>(second) / count - m * m;
    const long double v = std::max(empirical, m * (1 - m) / (kMaxAbsenceSamples + 3.L));
    const long double c = m * (1 - m) / v - 1;
    if (!(m > 0 && m < 1 && empirical >= 0 && v > 0 && c > 0 && std::isfinite(c))) {
      throw std::runtime_error("Invalid absence calibration moments");
    }
    return {m * c, (1 - m) * c};
  }
};

struct CalibrationView { TimeStamp stamp; size_t free, measured; };

struct ObjectAbsenceState {
  uint64_t evidence_key = 0;
  TimeStamp processed = 0, calibration_processed = 0;
  bool inherited = false;
  TimeStamp sampled_birth = 0;
  float sampled_cell_size = 0, sampled_min_cos = 0;
  std::vector<std::array<float, 7>> sampled_geometry;
  Calibration calibration;
  std::deque<CalibrationView> pending_calibration;
  std::map<AbsenceCell, AbsenceSample> samples, calibration_samples;
};


struct MosaicCount {
  size_t groups = 0, positive = 0, unknown = 0;
  measurement::CompletedEvidence completion() const {
    return measurement::retainingCompletion(positive,groups,unknown);
  }
};

// README (7): one event per real (frame,pixel), then the minimum over
// unknown-source completions. The same reducer serves calibration and scoring.
MosaicCount countMosaic(const std::map<AbsenceCell,AbsenceSample>& mosaic,
                       const std::set<AbsenceCell>& reliable, TimeStamp earliest) {
  std::map<std::pair<TimeStamp,uint32_t>,bool> groups;
  MosaicCount result;
  for (const auto& cell : reliable) {
    const auto found = mosaic.find(cell);
    if (found == mosaic.end()) { ++result.unknown; continue; }
    const auto& sample = found->second;
    const auto stamp = std::max(sample.last_on_surface,sample.last_seen_through);
    if (!stamp || stamp < earliest) { ++result.unknown; continue; }
    const bool absent = sample.last_seen_through > sample.last_on_surface;
    const auto pixel = absent ? sample.absence_pixel : sample.surface_pixel;
    const auto [it,inserted] = groups.emplace(std::make_pair(stamp,pixel),absent);
    if (!inserted) it->second = it->second && absent;
  }
  result.groups = groups.size();
  for (const auto& [source,absent] : groups) {
    (void)source;
    result.positive += absent;
  }
  return result;
}



long double absenceLogOdds(measurement::CompletedEvidence evidence,
                           const Calibration& local, const Calibration& sensor) {
  const auto reference = [](const Calibration& calibration) {
    const auto [a,b] = calibration.shape();
    return measurement::BetaReference{a,b};
  };
  // README (7b): both models present -> equal-prior mixture; population only -> population;
  // otherwise the initial Beta(1,1), which an empty population yields.
  if (local.count > 0 && sensor.count > 0)
    return measurement::surfaceLogOdds(evidence,0.5L,reference(local),reference(sensor));
  return measurement::surfaceLogOdds(evidence,0.5L,reference(sensor));
}

// Give each calibrated state one contribution to the population. Updating its
// local mean replaces that contribution, independent of how often it is seen.
void calibrate(Calibration& sensor_calibration, ObjectAbsenceState& state, size_t free, size_t measured) {
  auto& c = state.calibration;
  if (c.count > 0) {
    sensor_calibration.first -= c.first / c.count;
    sensor_calibration.second -= c.second / c.count;
  } else {
    sensor_calibration.count += 1;
  }
  c.add(free, measured);
  sensor_calibration.first += c.first / c.count;
  sensor_calibration.second += c.second / c.count;
}

}  // namespace

struct ObservedAbsenceModel::Impl {
  mutable std::recursive_mutex mutex;
  Calibration sensor;
  const Calibration* frozen = nullptr;
  std::map<StateKey, std::shared_ptr<ObjectAbsenceState>> states;
};

ObservedAbsenceModel::ObservedAbsenceModel() : impl_(std::make_unique<Impl>()) {}
ObservedAbsenceModel::~ObservedAbsenceModel() = default;

struct ObservedAbsenceBatch::Impl {
  ObservedAbsenceModel::Impl& model;
  std::unique_lock<std::recursive_mutex> lock;
  const Calibration* previous;
  Calibration frozen;
  explicit Impl(ObservedAbsenceModel::Impl& value)
      : model(value), lock(value.mutex), previous(value.frozen),
        frozen(previous ? *previous : value.sensor) {
    model.frozen = &frozen;
  }
  ~Impl() { model.frozen = previous; }
};
ObservedAbsenceBatch::ObservedAbsenceBatch(const ObservedAbsenceModel& model)
    : impl_(std::make_unique<Impl>(*model.impl_)) {}
ObservedAbsenceBatch::~ObservedAbsenceBatch() = default;

bool ObservedAbsenceModel::saveSensorStatistics(const std::string& path) const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  const auto& sensor_calibration = impl_->sensor;
  const auto temp = path + ".tmp";
  {
    std::ofstream out(temp);
    out << std::setprecision(17) << "ABSENCE_PREDICTIVE_V1 "
        << sensor_calibration.count << ' ' << sensor_calibration.first << ' '
        << sensor_calibration.second << '\n';
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
  Calibration loaded;
  if (tag == "ABSENCE_PREDICTIVE_V1") {
    if (!(in >> loaded.count >> loaded.first >> loaded.second)) return false;
  } else {
    // Legacy mean and variance represent calibrated objects. One uniform
    // component supplies the finite-sample prior in README (7b).
    in.clear();
    in.seekg(0);
    double count, sum, variance, label_count, label_sum, label_variance;
    if (!(in >> count >> sum >> variance >> label_count >> label_sum >> label_variance) ||
        !std::isfinite(count) || !std::isfinite(sum) || !std::isfinite(variance) ||
        count < 0 || sum < 0 || sum > count) return false;
    if (count > 0) {
      const double m = sum / count;
      // Old writers use -1 when no scatter estimate exists; that is no
      // calibration distribution, so the unit uniform prior remains active.
      if (variance >= 0) {
        const double feasible_variance = std::min(variance, m * (1 - m));
        loaded = {count + 1, sum + 0.5, count * (feasible_variance + m * m) + 1. / 3};
      }
    }
  }
  in >> std::ws;
  if (!in.eof() || !std::isfinite(loaded.count) || !std::isfinite(loaded.first) ||
      !std::isfinite(loaded.second) || loaded.count < 0 ||
      (loaded.count == 0 && (loaded.first != 0 || loaded.second != 0))) return false;
  try { loaded.shape(); } catch (const std::exception&) { return false; }
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  if (impl_->frozen) throw std::logic_error("Load during an absence evidence pass");
  impl_->sensor = loaded;
  impl_->states.clear();
  return true;
}

namespace {
using Json = nlohmann::json;
Json encodeCalibration(const Calibration& value) {
  return Json::array({value.count,value.first,value.second});
}
Calibration decodeCalibration(const Json& value) {
  if (!value.is_array() || value.size() != 3)
    throw std::invalid_argument("Invalid calibration tuple");
  Calibration result{value.at(0).get<double>(),value.at(1).get<double>(),value.at(2).get<double>()};
  if (!std::isfinite(result.count) || !std::isfinite(result.first) ||
      !std::isfinite(result.second) || result.count < 0 ||
      (result.count == 0 && (result.first != 0 || result.second != 0)))
    throw std::invalid_argument("Invalid calibration moments");
  result.shape();
  return result;
}
Json encodeSamples(const std::map<AbsenceCell,AbsenceSample>& samples) {
  Json encoded = Json::array();
  for (const auto& [cell,value] : samples) {
    const auto [x,y,z] = cell;
    encoded.push_back(Json::array({x,y,z,value.identity_hits,value.last_on_surface,
        value.last_seen_through,value.reliable_since,value.surface_pixel,value.absence_pixel}));
  }
  return encoded;
}
std::map<AbsenceCell,AbsenceSample> decodeSamples(const Json& encoded, TimeStamp through) {
  std::map<AbsenceCell,AbsenceSample> samples;
  for (const auto& item : encoded) {
    if (!item.is_array() || item.size() != 9)
      throw std::invalid_argument("Invalid absence checkpoint cell");
    const AbsenceCell cell{item.at(0).get<int64_t>(),item.at(1).get<int64_t>(),item.at(2).get<int64_t>()};
    const auto hits = item.at(3).get<uint64_t>();
    const auto support = item.at(4).get<uint64_t>();
    const auto absent = item.at(5).get<uint64_t>();
    const auto established = item.at(6).get<uint64_t>();
    const auto support_pixel = item.at(7).get<uint64_t>();
    const auto absent_pixel = item.at(8).get<uint64_t>();
    if (hits > kMinIdentityHits || support > through || absent > through ||
        established > support || (hits == kMinIdentityHits) != (established > 0) ||
        support_pixel >= UINT32_MAX || absent_pixel >= UINT32_MAX ||
        !samples.emplace(cell,AbsenceSample{static_cast<uint16_t>(hits),support,absent,
            established,static_cast<uint32_t>(support_pixel),static_cast<uint32_t>(absent_pixel)}).second)
      throw std::invalid_argument("Invalid absence checkpoint evidence");
  }
  return samples;
}

}  // namespace

void ObservedAbsenceModel::retain(const LiveStates& live) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  for (auto it = impl_->states.begin(); it != impl_->states.end();) {
    if (live.count(it->first)) ++it;
    else it = impl_->states.erase(it);
  }
  // Retired states retain their population contribution; detailed samples are released.
}

void ObservedAbsenceModel::save(const std::string& path, uint64_t boundary,
                                const LiveStates& live) const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  Json records = Json::array();
  for (const auto& [key,ptr] : impl_->states) {
    if (!live.count(key)) continue;
    const auto& state = *ptr;
    if (state.processed > boundary) throw std::logic_error("Evidence exceeds saved boundary");
    if (state.calibration_processed > state.processed)
      throw std::logic_error("Calibration exceeds evidence boundary");
    Json pending = Json::array();
    for (const auto& view : state.pending_calibration)
      pending.push_back(Json::array({view.stamp,view.free,view.measured}));
    records.push_back(Json{{"physical",key.first},{"key",key.second},
        {"processed",state.processed},{"calibration_processed",state.calibration_processed},
        {"birth",state.sampled_birth},
        {"inherited",state.inherited},{"cell_size",state.sampled_cell_size},
        {"min_cos",state.sampled_min_cos},{"geometry",state.sampled_geometry},
        {"calibration",encodeCalibration(state.calibration)},
        {"samples",encodeSamples(state.samples)},
        {"calibration_samples",encodeSamples(state.calibration_samples)},
        {"pending_calibration",std::move(pending)}});
  }
  const Json packet{{"schema",2},{"boundary",boundary},
      {"population",encodeCalibration(impl_->sensor)},{"states",std::move(records)}};
  const auto bytes = Json::to_cbor(packet);
  const auto temporary = path + ".tmp";
  std::ofstream out(temporary,std::ios::binary | std::ios::trunc);
  out.exceptions(std::ios::badbit | std::ios::failbit);
  out.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
  out.close();
  std::filesystem::rename(temporary,path);
}

void ObservedAbsenceModel::load(const std::string& path, uint64_t boundary,
                                const LiveStates& live) {
  std::ifstream in(path,std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open absence checkpoint: " + path);
  const auto packet = Json::from_cbor(in);
  if (packet.at("schema").get<unsigned>() != 2 ||
      packet.at("boundary").get<uint64_t>() != boundary)
    throw std::invalid_argument("Absence checkpoint boundary mismatch");
  const auto sensor = decodeCalibration(packet.at("population"));
  std::map<StateKey,std::shared_ptr<ObjectAbsenceState>> staged;
  size_t calibrated = 0;
  for (const auto& item : packet.at("states")) {
    const StateKey key{item.at("physical").get<size_t>(),item.at("key").get<uint64_t>()};
    if (!key.second || !live.count(key) || staged.count(key))
      throw std::invalid_argument("Absence checkpoint state ownership mismatch");
    auto state = std::make_shared<ObjectAbsenceState>();
    state->evidence_key = key.second;
    state->processed = item.at("processed").get<uint64_t>();
    state->calibration_processed = item.at("calibration_processed").get<uint64_t>();
    state->sampled_birth = item.at("birth").get<uint64_t>();
    state->inherited = item.at("inherited").get<bool>();
    state->sampled_cell_size = item.at("cell_size").get<float>();
    state->sampled_min_cos = item.at("min_cos").get<float>();
    state->sampled_geometry = item.at("geometry").get<std::vector<std::array<float,7>>>();
    state->calibration = decodeCalibration(item.at("calibration"));
    calibrated += state->calibration.count > 0;
    if (state->processed > boundary || state->calibration_processed > state->processed ||
        state->sampled_birth > state->processed ||
        !(std::isfinite(state->sampled_cell_size) && state->sampled_cell_size > 0) ||
        !(std::isfinite(state->sampled_min_cos) && std::abs(state->sampled_min_cos) <= 1) ||
        state->sampled_geometry.size() > kMaxAbsenceSamples)
      throw std::invalid_argument("Invalid absence checkpoint query");
    for (const auto& point : state->sampled_geometry) {
      for (const auto value : point)
        if (!std::isfinite(value)) throw std::invalid_argument("Non-finite absence query");
      if (point[6] != 0 && point[6] != 1)
        throw std::invalid_argument("Invalid absence query normal flag");
    }
    state->samples = decodeSamples(item.at("samples"),state->processed);
    state->calibration_samples = decodeSamples(item.at("calibration_samples"),state->processed);
    for (const auto& [cell,value] : state->calibration_samples) {
      if (!state->samples.count(cell) || value.identity_hits || value.reliable_since)
        throw std::invalid_argument("Calibration sample ownership mismatch");
    }
    TimeStamp pending_after = state->calibration_processed;
    for (const auto& view : item.at("pending_calibration")) {
      if (!view.is_array() || view.size() != 3)
        throw std::invalid_argument("Invalid pending calibration view");
      CalibrationView value{view.at(0).get<TimeStamp>(),view.at(1).get<size_t>(),view.at(2).get<size_t>()};
      if (value.stamp <= pending_after || value.stamp > state->processed ||
          !value.measured || value.measured > kMaxAbsenceSamples || value.free > value.measured)
        throw std::invalid_argument("Invalid pending calibration interval");
      pending_after = value.stamp;
      state->pending_calibration.push_back(value);
    }
    if (state->samples.size() != state->sampled_geometry.size())
      throw std::invalid_argument("Absence checkpoint sample count mismatch");
    staged.emplace(key,std::move(state));
  }
  if (sensor.count < calibrated)
    throw std::invalid_argument("Absence population omits live state contributions");
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  if (impl_->frozen) throw std::logic_error("Load during an absence evidence pass");
  impl_->sensor = sensor;
  impl_->states = std::move(staged);
}

RayVerificator::CheckResult RayVerificator::checkProjectedPhysical(
    const Point& point, const size_t physical_id,
    const PhysicalEvidenceSnapshot& evidence_snapshot,
    const uint64_t earliest, const uint64_t latest, CheckDetails* details) const {
  CheckResult result;
  if (!point.allFinite()) { ++result.reasons.invalid; return result; }
  if (!evidence_snapshot) return result;
  for (const auto stamp : evidence_snapshot->timestamps(earliest,latest)) {
    const auto p = evidence_snapshot->project(stamp,point);
    const auto vote = classifySurfaceMeasurement(p,physical_id,config.depth_tolerance);
    if (vote == Vote::Unavailable) continue;
    auto decision = CheckDetails::Result::kOccludded;
    const auto present = [&](size_t& reason) {
      result.present.push_back(stamp); ++reason; decision = CheckDetails::Result::kMatch;
    };
    const auto absent = [&](size_t& reason) {
      result.absent.push_back(stamp); ++reason; decision = CheckDetails::Result::kAbsent;
    };
    const auto inconclusive = [&](size_t& reason) {
      result.inconclusive.push_back(stamp); ++reason;
    };
    switch (vote) {
      case Vote::Supported: present(result.reasons.same_id); break;
      case Vote::Free: absent(result.reasons.free_space); break;
      case Vote::Background: absent(result.reasons.background_replacement); break;
      case Vote::Other: absent(result.reasons.different_id); break;
      case Vote::Occluded: inconclusive(result.reasons.geometric_occlusion); break;
      case Vote::Unidentified: inconclusive(result.reasons.unidentified_object); break;
      case Vote::Invalid: inconclusive(result.reasons.invalid); break;
      case Vote::Unavailable: break;
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
    const PhysicalEvidenceSnapshot& evidence_snapshot,
    const uint64_t earliest, const uint64_t latest) const {
  SurfaceEvidenceCounts result;
  if (!evidence_snapshot) return result;
  Points samples;
  for (const auto& sample : sampleSurface(mesh,bbox,0.5f * config.surface_match_tolerance,
                                         kMaxAbsenceSamples))
    samples.push_back(sample.point);
  result.surface_samples = samples.size();
  const auto stamps = evidence_snapshot->timestamps(earliest, latest);
  for (const auto& point : samples) {
    bool coverage = false;
    TimeStamp sample_support = 0, sample_absence = 0;
    for (size_t frame = 0; frame < stamps.size(); ++frame) {
      const auto p = evidence_snapshot->project(stamps[frame], point);
      const auto vote = classifySurfaceMeasurement(p, physical_id, config.surface_match_tolerance);
      if (vote == Vote::Unavailable) continue;
      coverage = true;
      // Exact (frame, pixel), not (sample, frame): a pixel is one measurement
      // even when several surface triangles project onto it.
      const size_t key = (frame << 32) | static_cast<size_t>(p.pixel_index);
      switch (vote) {
        case Vote::Supported:
          sample_support = stamps[frame];
          result.latest_support_stamp = std::max(result.latest_support_stamp, stamps[frame]);
          result.support_indices.insert(key); ++result.supported_votes; break;
        case Vote::Free:
          sample_absence = stamps[frame];
          result.contradiction_indices.insert(key); ++result.free_space_votes; break;
        case Vote::Background:
          sample_absence = stamps[frame];
          result.contradiction_indices.insert(key); ++result.replaced_by_background_votes; break;
        case Vote::Other:
          sample_absence = stamps[frame];
          result.contradiction_indices.insert(key); ++result.replaced_by_other_votes; break;
        case Vote::Occluded:
          ++result.occluded_votes; break;
        default:
          break;
      }
    }
    if (!coverage) ++result.unobserved_samples;
    if (sample_absence > sample_support) ++result.contradicted_surface_samples;
  }
  result.support_rays = result.support_indices.size();
  result.contradiction_rays = result.contradiction_indices.size();
  return result;
}


void RayVerificator::applyObservedAbsence(
    const size_t physical_id, const spark_dsg::Mesh& mesh, const BoundingBox& bbox,
    const PhysicalEvidenceSnapshot& evidence_snapshot, const uint64_t earliest,
    const uint64_t latest, SurfaceEvidenceCounts& counts, const int state_slot,
    const uint64_t state_birth, const uint64_t evidence_key, const bool inherited,
    const uint64_t direct_support) const {
  counts.absence_coverage_sufficient = false;
  if (!evidence_snapshot) return;
  const ObservedAbsenceBatch batch(*absence_model_);
  auto& model = *absence_model_->impl_;
  auto& sensor_calibration = model.sensor;
  const float tolerance = config.surface_match_tolerance;
  const float incidence = config.max_absence_incidence_deg;
  if (!std::isfinite(incidence) || incidence < 0.f || incidence > 90.f)
    throw std::invalid_argument("Absence incidence angle must lie in [0,90] degrees");
  const float min_cos = std::cos(incidence * static_cast<float>(M_PI) / 180.f);

  if (!(std::isfinite(tolerance) && tolerance > 0)) {
    throw std::invalid_argument("Surface matching tolerance must be positive");
  }
  const float cell_size = 0.5f * tolerance;
  std::vector<std::pair<AbsenceCell,SurfaceSample>> queries;
  for (const auto& query : sampleSurface(mesh,bbox,cell_size,kMaxAbsenceSamples)) {
    queries.emplace_back(AbsenceCell{query.cell[0],query.cell[1],query.cell[2]},query);
  }

  const uint64_t key = evidence_key ? evidence_key : state_birth;
  std::shared_ptr<ObjectAbsenceState> state;
  {
    auto& slot = model.states[{physical_id, key}];
    if (!slot) slot = std::make_shared<ObjectAbsenceState>();
    state = slot;
  }
  std::vector<std::array<float, 7>> geometry;
  geometry.reserve(queries.size());
  for (const auto& [cell, q] : queries) {
    (void)cell;
    geometry.push_back({q.point.x(),q.point.y(),q.point.z(),
        q.normal.x(),q.normal.y(),q.normal.z(),q.has_normal ? 1.f : 0.f});
  }
  if (state->evidence_key != key || latest < state->processed ||
      state->sampled_birth != state_birth || state->sampled_geometry != geometry ||
      state->sampled_cell_size != cell_size || state->sampled_min_cos != min_cos) {
    // README (7b): replace this state's contribution when its query geometry changes.
    {
      const auto& old = state->calibration;
      if (old.count > 0) {
        sensor_calibration.count -= 1;
        sensor_calibration.first -= old.first / old.count;
        sensor_calibration.second -= old.second / old.count;
        if (sensor_calibration.count == 0) sensor_calibration = Calibration{};
      }
    }
    *state = ObjectAbsenceState{};
    state->evidence_key = key;
    state->sampled_birth = state_birth;
    state->sampled_cell_size = cell_size;
    state->sampled_min_cos = min_cos;
    state->sampled_geometry = std::move(geometry);
  }
  state->inherited = inherited;
  for (const auto& [cell,query] : queries) {
    (void)query;
    state->samples.try_emplace(cell);
  }
  const auto observe = [&](AbsenceSample& sample, const SurfaceSample& query,
                           TimeStamp stamp, bool learn_identity) {
    const auto projected = evidence_snapshot->project(stamp,query.point);
    const auto vote = classifySurfaceMeasurement(projected,physical_id,tolerance);
    if (vote == Vote::Supported) {
      sample.last_on_surface = stamp;
      sample.surface_pixel = projected.pixel_index;
      if (learn_identity && sample.identity_hits < kMinIdentityHits) {
        ++sample.identity_hits;
        if (sample.identity_hits == kMinIdentityHits) sample.reliable_since = stamp;
      }
      return true;
    }
    if (vote == Vote::Other || vote == Vote::Background ||
        (vote == Vote::Free && (!query.has_normal ||
         std::abs(query.normal.dot(projected.view_direction_world)) >= min_cos))) {
      sample.last_seen_through = stamp;
      sample.absence_pixel = projected.pixel_index;
      return true;
    }
    return false;
  };
  if (state->processed < latest) {
    const auto first = std::max(state->processed + 1,state_birth);
    for (const auto stamp : evidence_snapshot->timestamps(first,latest)) {
      std::set<AbsenceCell> reliable_before;
      bool calibration_measured = false;
      for (const auto& [cell,query] : queries) {
        auto& sample = state->samples.at(cell);
        if (inherited || (sample.reliable_since && sample.reliable_since < stamp)) {
          reliable_before.insert(cell);
          calibration_measured = observe(state->calibration_samples[cell],query,stamp,false)
              || calibration_measured;
        }
        observe(sample,query,stamp,true);
      }
      if (calibration_measured) {
        const auto count = countMosaic(state->calibration_samples,reliable_before,state_birth);
        state->pending_calibration.push_back({stamp,count.positive,count.groups});
      }
    }
  }
  state->processed = std::max<TimeStamp>(state->processed,latest);

  // The candidate mosaic advances with every real frame. Only sufficient
  // statistics await direct-support authorization; save/load retains this tail.
  const auto calibration_end = std::min<TimeStamp>(direct_support,latest);
  if (calibration_end < state->calibration_processed)
    throw std::logic_error("Direct reconstruction support moved backwards");
  while (!state->pending_calibration.empty() &&
         state->pending_calibration.front().stamp <= calibration_end) {
    const auto& view = state->pending_calibration.front();
    calibrate(sensor_calibration,*state,view.free,view.measured);
    state->pending_calibration.pop_front();
  }
  state->calibration_processed = calibration_end;

  counts.reliable_samples = counts.reliable_in_view = counts.reliable_seen_through = 0;
  counts.reliable_points.clear();
  std::set<AbsenceCell> reliable;
  for (const auto& [cell,query] : queries) {
    const auto& sample = state->samples.at(cell);
    if (!inherited && sample.identity_hits < kMinIdentityHits) continue;
    reliable.insert(cell);
    ++counts.reliable_samples;
    counts.reliable_points.push_back(query.point);
    const auto last = std::max(sample.last_on_surface,sample.last_seen_through);
    if (last < earliest || !last) continue;
    ++counts.reliable_in_view;
    counts.reliable_seen_through += sample.last_seen_through > sample.last_on_surface;
  }
  const auto completed = countMosaic(state->samples,reliable,earliest);
  const Calibration& sensor = *model.frozen;
  const auto odds = counts.reliable_samples > 0
      ? absenceLogOdds(completed.completion(),state->calibration,sensor) : 0.L;
  if (!std::isfinite(odds)) throw std::runtime_error("Non-finite absence predictive odds");
  counts.absence_llr = static_cast<float>(odds);
  counts.absence_coverage_sufficient = measurement::favorsExit(odds);  // README (7d): loss ratio.
  VLOG(1) << "OBSERVED_ABSENCE inst=" << physical_id << " slot=" << state_slot
          << " reliable=" << counts.reliable_samples << " judged=" << counts.reliable_in_view
          << " free=" << counts.reliable_seen_through << " log_odds=" << odds;

}

RayVerificator::SurfaceEvidenceCounts RayVerificator::countCurrentPhysicalSurface(
    size_t physical_id, const spark_dsg::Mesh& mesh, const BoundingBox& bbox,
    const PhysicalEvidenceSnapshot& snapshot,
    uint64_t last_support, uint64_t latest, bool* projected, const int state_slot,
    const uint64_t state_birth, const uint64_t evidence_key, const bool inherited,
    const uint64_t direct_support) const {
  if (projected) *projected = snapshot && snapshot->numFrames() > 0;
  SurfaceEvidenceCounts counts;
  if (!snapshot || snapshot->numFrames() == 0) return counts;
  if (last_support >= latest) {
    applyObservedAbsence(physical_id,mesh,bbox,snapshot,latest,latest,counts,
                         state_slot,state_birth,evidence_key,inherited,direct_support);
    counts.absence_coverage_sufficient = false;
    return counts;
  }
  const uint64_t earliest = last_support + 1;
  counts = countProjectedPhysicalSurface(physical_id,mesh,bbox,snapshot,earliest,latest);
  applyObservedAbsence(physical_id,mesh,bbox,snapshot,earliest,latest,counts,
                       state_slot,state_birth,evidence_key,inherited,direct_support);
  return counts;

}

}  // namespace khronos
