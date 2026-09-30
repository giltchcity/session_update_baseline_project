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
  double reference_zeta = 0.0;
  std::shared_ptr<SensorStatistics> statistics = std::make_shared<SensorStatistics>();
  FreeSpaceRecords free_space;
  ElementOutcomes outcomes;
  nlohmann::json motion;
  std::map<StateKey, uint64_t> processed;
};

ObservedAbsenceModel::ObservedAbsenceModel() : impl_(std::make_unique<Impl>()) {}
ObservedAbsenceModel::~ObservedAbsenceModel() = default;

void ObservedAbsenceModel::setInitialRangeModel(model::RangeModel psi, double reference_zeta) {
  if (!std::isfinite(reference_zeta) || reference_zeta <= -1.0) {
    throw std::invalid_argument("Invalid reference depth scale");
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->psi = std::move(psi);
  impl_->reference_zeta = reference_zeta;
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

void ObservedAbsenceModel::refreshRangeModel(double association_gate) {
  model::RangeModel previous;
  double reference;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    previous = impl_->psi;
    reference = impl_->reference_zeta;
  }
  const double max_range = impl_->statistics->max_range.load();
  if (!(max_range > 0.0)) return;  // no frame yet
  auto next = impl_->statistics->calibrator.estimate(previous, max_range);
  double zeta;
  if (impl_->statistics->calibrator.estimateScale(association_gate, zeta)) {
    next.zeta = zeta;
    next.delta_s = zeta - reference;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->psi = std::move(next);
}

void ObservedAbsenceModel::setSessionModel(model::RangeModel psi) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
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
  impl_->outcomes.dup += outcomes.dup;
  impl_->outcomes.sep += outcomes.sep;
  impl_->outcomes.fill_confirmed += outcomes.fill_confirmed;
  impl_->outcomes.fill_total += outcomes.fill_total;
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
        << " sigma_x " << psi.sigma_x << " w_plus " << psi.w_plus << " w_minus " << psi.w_minus
        << "\n";
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
  const nlohmann::json packet{{"schema", 4},
                              {"boundary", boundary},
                              {"psi", impl_->psi.toJson()},
                              {"states", std::move(records)},
                              {"free_space", impl_->free_space.toJson()},
                              {"motion", impl_->motion},
                              {"outcomes", nlohmann::json::array({impl_->outcomes.dup, impl_->outcomes.sep,
                                                                  impl_->outcomes.fill_confirmed,
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
  if (packet.at("schema").get<unsigned>() != 4) {
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
    outcomes = {o.at(0).get<double>(), o.at(1).get<double>(), o.at(2).get<double>(), o.at(3).get<double>()};
    if (!(outcomes.dup >= 0) || !(outcomes.sep >= 0) || !(outcomes.fill_confirmed >= 0) ||
        outcomes.fill_total < outcomes.fill_confirmed) {
      throw std::invalid_argument("Invalid element outcome statistics");
    }
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  // The loaded session's scale is the reference of the memory; this session starts from it.
  impl_->reference_zeta = psi.zeta;
  psi.delta_s = 0.0;
  impl_->psi = std::move(psi);
  impl_->processed = std::move(staged);
  impl_->free_space = std::move(free_space);
  impl_->outcomes = outcomes;
  impl_->motion = packet.contains("motion") ? packet.at("motion") : nlohmann::json();
}

}  // namespace khronos
