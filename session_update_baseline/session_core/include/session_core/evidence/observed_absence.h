#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include "session_core/evidence/free_space_records.h"
#include "session_core/model/range_model.h"
#include "session_core/model/sensor_calibrator.h"

namespace khronos {

/** What the evidence store's ingest path and the model share: the session's re-measurement
 * statistics and its first frame. */
struct SensorStatistics {
  model::SensorCalibrator calibrator;
  std::atomic<TimeStamp> session_start{std::numeric_limits<TimeStamp>::max()};
  std::atomic<double> max_range{0.0};  // R of the device, from the first frame
  std::atomic<double> zeta{0.0};       // the session's current range scale estimate (9b)
  std::atomic<double> truncation{0.0};  // T of the fused surface the residuals of (9v) are taken on
  /** The earliest frame seen is the start of the session. */
  void noteFrame(TimeStamp stamp, double device_max_range) {
    auto current = session_start.load();
    while (stamp < current && !session_start.compare_exchange_weak(current, stamp)) {}
    max_range.store(device_max_range);
  }
};

/**
 * README principles 4, 6, 8, 12: the evidence-side state of the unified model. It owns psi (the
 * parameters of the first-return model, README (6m): the estimate held before a round predicts
 * that round), the online calibrator behind it, the processed-input watermark of every placement
 * (principle 1, each source once) and V_free, the free space of earlier sessions.
 */
class ObservedAbsenceModel {
 public:
  using StateKey = std::pair<size_t, uint64_t>;
  using LiveStates = std::set<StateKey>;

  ObservedAbsenceModel();
  ~ObservedAbsenceModel();
  ObservedAbsenceModel(const ObservedAbsenceModel&) = delete;
  ObservedAbsenceModel& operator=(const ObservedAbsenceModel&) = delete;

  /** The model the session starts from: the previous session's posterior psi, which is the prior of
   * this session. */
  void setInitialRangeModel(model::RangeModel psi);
  /** The sensor calibration (class 1): the curve of sigma_table per range bin. It is the prior
   * centre of a first session (README table 5.1), after the previous session's posterior and the
   * neighbouring bins of this session; it gives nothing to w_pm and zeta. */
  void setSensorCalibration(std::vector<double> sigma_curve);
  /** psi as held for the current round. */
  model::RangeModel rangeModel() const;
  /** The model the session started from: the previous session's posterior (the prior of every
   * quantity this session estimates). */
  model::RangeModel priorRangeModel() const;
  /** README (6m), (9b): the posterior of the prior and the statistics gathered so far; it predicts
   * the next round. */
  void refreshRangeModel();
  /** The authoritative estimate of the session end: psi with the scale and alignment residual
   * measured on the full archive. */
  void setSessionModel(model::RangeModel psi);

  std::shared_ptr<SensorStatistics> statistics() const;
  TimeStamp sessionStart() const;

  /** V_free of the earlier sessions (read-only during the session). */
  const FreeSpaceRecords& freeSpace() const;
  void setFreeSpace(FreeSpaceRecords records);

  /** zeta_e of the elements a previous session made: the scale of the session that was loaded (0
   * for a first session). */
  double previousZeta() const;

  /** README (15b), principle 9: the records of the memory elements the last session's map did not
   * show but did not delete (a SessionRefusion::Surface as JSON); null where there are none. */
  nlohmann::json hiddenRecords() const;
  void setHiddenRecords(nlohmann::json records);

  /** README (15b): the visible-motion statistics of principle 5 travel with the evidence state. */
  nlohmann::json motionState() const;
  void setMotionState(nlohmann::json state);

  // README (1) principle 1: only inputs after the watermark are new evidence for a placement.
  uint64_t processed(const StateKey& key) const;
  void setProcessed(const StateKey& key, uint64_t stamp);

  bool saveSensorStatistics(const std::string& path) const;
  void save(const std::string& path, uint64_t boundary, const LiveStates& live) const;
  void load(const std::string& path, uint64_t boundary, const LiveStates& live);
  void retain(const LiveStates& live);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace khronos
