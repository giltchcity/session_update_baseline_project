#include "session_core/evidence/observed_absence.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <stdexcept>

#include <glog/logging.h>

namespace khronos {

struct ObservedAbsenceModel::Impl {
  mutable std::mutex mutex;
  model::RangeModel psi;
  std::shared_ptr<SensorStatistics> statistics = std::make_shared<SensorStatistics>();
  FreeSpaceRecords free_space;
  ElementOutcomes outcomes;
  nlohmann::json motion;
  nlohmann::json hidden;
  std::map<StateKey, uint64_t> processed;
};

ObservedAbsenceModel::ObservedAbsenceModel() : impl_(std::make_unique<Impl>()) {}
ObservedAbsenceModel::~ObservedAbsenceModel() = default;

void ObservedAbsenceModel::setInitialRangeModel(model::RangeModel psi) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->statistics->zeta.store(psi.zeta);
  impl_->psi = std::move(psi);
}

bool ObservedAbsenceModel::hasRangeModel() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return std::any_of(impl_->psi.sigma_s.begin(), impl_->psi.sigma_s.end(),
                     [](double s) { return s > 0.0; });
}

model::RangeModel ObservedAbsenceModel::rangeModel() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->psi;
}

void ObservedAbsenceModel::refreshRangeModel() {
  model::RangeModel previous;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    previous = impl_->psi;
  }
  const double max_range = impl_->statistics->max_range.load();
  if (!(max_range > 0.0)) return;  // no frame yet
  // (9v), (12d): sigma_s, sigma_reg, w_pm and, from the band pairs, Delta_s, sigma_x, pi_dup.
  auto next = impl_->statistics->calibrator.estimate(previous, max_range);
  // (9b): the scale, from the previous session's value (0 at cold start) to the fixed point of the
  // alternation between mutual-hit correspondences and the median-residual minimum.
  double zeta;
  if (impl_->statistics->calibrator.estimateScale(next, max_range, zeta)) next.zeta = zeta;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->statistics->zeta.store(next.zeta);
  impl_->psi = std::move(next);
}

void ObservedAbsenceModel::setSessionModel(model::RangeModel psi) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->statistics->zeta.store(psi.zeta);
  impl_->psi = std::move(psi);
}

std::shared_ptr<SensorStatistics> ObservedAbsenceModel::statistics() const {
  return impl_->statistics;
}

TimeStamp ObservedAbsenceModel::sessionStart() const {
  return impl_->statistics->session_start.load();
}

const FreeSpaceRecords& ObservedAbsenceModel::freeSpace() const { return impl_->free_space; }

void ObservedAbsenceModel::setFreeSpace(FreeSpaceRecords records) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->free_space = std::move(records);
}

ObservedAbsenceModel::ElementOutcomes ObservedAbsenceModel::elementOutcomes() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->outcomes;
}

void ObservedAbsenceModel::addElementOutcomes(const ElementOutcomes& outcomes) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->outcomes.fill_confirmed += outcomes.fill_confirmed;
  impl_->outcomes.fill_total += outcomes.fill_total;
}

nlohmann::json ObservedAbsenceModel::hiddenRecords() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->hidden;
}

void ObservedAbsenceModel::setHiddenRecords(nlohmann::json records) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->hidden = std::move(records);
}

nlohmann::json ObservedAbsenceModel::motionState() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->motion;
}

void ObservedAbsenceModel::setMotionState(nlohmann::json state) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->motion = std::move(state);
}

uint64_t ObservedAbsenceModel::processed(const StateKey& key) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto found = impl_->processed.find(key);
  return found == impl_->processed.end() ? 0 : found->second;
}

void ObservedAbsenceModel::setProcessed(const StateKey& key, uint64_t stamp) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto& value = impl_->processed[key];
  value = std::max(value, stamp);
}

void ObservedAbsenceModel::retain(const LiveStates& live) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  for (auto it = impl_->processed.begin(); it != impl_->processed.end();) {
    if (live.count(it->first)) ++it;
    else it = impl_->processed.erase(it);
  }
}

bool ObservedAbsenceModel::saveSensorStatistics(const std::string& path) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto temp = path + ".tmp";
  {
    std::ofstream out(temp);
    const auto& psi = impl_->psi;
    out << std::setprecision(17) << "RANGE_MODEL_V2 zeta " << psi.zeta << " delta_s " << psi.delta_s
        << " sigma_x " << psi.sigma_x << " pi_dup " << psi.pi_dup << " zeta_memory " << psi.zeta_memory
        << " pair_gamma0 " << psi.pair_gamma0 << " w_plus " << psi.w_plus << " w_minus "
        << psi.w_minus << "\n";
    out << "SIGMA_REG";
    for (size_t i = 0; i < psi.reg_dt.size(); ++i) {
      out << ' ' << psi.reg_dt[i] << ':' << psi.reg_variance[i];
    }
    out << "\nSIGMA_S";
    for (const double s : psi.sigma_s) out << ' ' << s;
    out << "\nPAIRS " << impl_->statistics->calibrator.numPairs() << "\n";
    out.close();
    if (!out) return false;
  }
  std::error_code error;
  std::filesystem::rename(temp, path, error);
  return !error;
}

void ObservedAbsenceModel::save(const std::string& path, uint64_t boundary,
                                const LiveStates& live) const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  nlohmann::json records = nlohmann::json::array();
  for (const auto& [key, processed] : impl_->processed) {
    if (!live.count(key)) continue;
    if (processed > boundary) throw std::logic_error("Evidence exceeds saved boundary");
    records.push_back(nlohmann::json{{"physical", key.first}, {"key", key.second},
                                     {"processed", processed}});
  }
  const nlohmann::json packet{{"schema", 5},
                              {"boundary", boundary},
                              {"psi", impl_->psi.toJson()},
                              {"states", std::move(records)},
                              {"free_space", impl_->free_space.toJson()},
                              {"motion", impl_->motion},
                              {"hidden", impl_->hidden},
                              {"outcomes", nlohmann::json::array({impl_->outcomes.fill_confirmed,
                                                                  impl_->outcomes.fill_total})}};
  const auto bytes = nlohmann::json::to_cbor(packet);
  const auto temporary = path + ".tmp";
  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  out.exceptions(std::ios::badbit | std::ios::failbit);
  out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  out.close();
  std::filesystem::rename(temporary, path);
}

void ObservedAbsenceModel::load(const std::string& path, uint64_t boundary, const LiveStates& live) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open evidence checkpoint: " + path);
  const auto packet = nlohmann::json::from_cbor(in);
  if (packet.at("schema").get<unsigned>() != 5) {
    // README s7.1: an older format is initialised from the default model of the appendix.
    LOG(WARNING) << "Evidence checkpoint '" << path << "' uses an older format; starting empty";
    return;
  }
  if (packet.at("boundary").get<uint64_t>() != boundary) {
    throw std::invalid_argument("Evidence checkpoint boundary mismatch");
  }
  auto psi = model::RangeModel::fromJson(packet.at("psi"));
  std::map<StateKey, uint64_t> staged;
  for (const auto& item : packet.at("states")) {
    const StateKey key{item.at("physical").get<size_t>(), item.at("key").get<uint64_t>()};
    const auto processed = item.at("processed").get<uint64_t>();
    if (!key.second || !live.count(key) || staged.count(key)) {
      throw std::invalid_argument("Evidence checkpoint state ownership mismatch");
    }
    if (processed > boundary) throw std::invalid_argument("Invalid evidence watermark");
    staged.emplace(key, processed);
  }
  auto free_space = FreeSpaceRecords::fromJson(packet.at("free_space"));
  ElementOutcomes outcomes;
  {
    const auto& o = packet.at("outcomes");
    outcomes = {o.at(0).get<double>(), o.at(1).get<double>()};
    if (!(outcomes.fill_confirmed >= 0) || outcomes.fill_total < outcomes.fill_confirmed) {
      throw std::invalid_argument("Invalid element outcome statistics");
    }
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  // The loaded session made the map's memory: its scale joins the bound |zeta_b| of the comparison
  // band (12d) and is the starting value of this session's scale (9b). Delta_s is a property of the
  // pair of sessions and starts from the cold-start value.
  psi.zeta_memory = std::max(psi.zeta_memory, std::abs(psi.zeta));
  psi.delta_s = 0.0;
  impl_->statistics->zeta.store(psi.zeta);
  impl_->psi = std::move(psi);
  impl_->processed = std::move(staged);
  impl_->free_space = std::move(free_space);
  impl_->outcomes = outcomes;
  impl_->motion = packet.contains("motion") ? packet.at("motion") : nlohmann::json();
  impl_->hidden = packet.contains("hidden") ? packet.at("hidden") : nlohmann::json();
}

}  // namespace khronos
