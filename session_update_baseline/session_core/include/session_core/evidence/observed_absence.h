#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include "session_core/evidence/error_model.h"

namespace khronos {

// README (7): sample count, sum and sum of squares of the whole-round penetration fraction of
// rounds judged in place.
struct NormalRoundStats {
  double count = 0, sum = 0, sum_sq = 0;
  void add(double fraction) {
    count += 1;
    sum += fraction;
    sum_sq += fraction * fraction;
  }
};

// README (7): Beta(a, b) with the sample mean and variance of `stats`, when they identify one
// (0 < mean < 1 and 0 < variance < mean (1 - mean)); otherwise false.
bool fitNormalRoundBeta(const NormalRoundStats& stats, double& a, double& b);

/**
 * README (6e), (7), (15b): the calibration state of one mapping instance. It owns the effective
 * range error model psi, the population statistics of normal rounds, and per placement its own
 * normal-round statistics, its initial model and its processed-input watermark.
 */
class ObservedAbsenceModel {
 public:
  using StateKey = std::pair<size_t, uint64_t>;
  using LiveStates = std::set<StateKey>;

  ObservedAbsenceModel();
  ~ObservedAbsenceModel();
  ObservedAbsenceModel(const ObservedAbsenceModel&) = delete;
  ObservedAbsenceModel& operator=(const ObservedAbsenceModel&) = delete;

  // README (6e), (9c): psi. Loaded from the previous session, or from the default model.
  void setErrorModel(measurement::ErrorModel psi);
  measurement::ErrorModel errorModel() const;
  bool hasErrorModel() const;

  // README (7): the Beta(a_e, b_e) of a placement's normal whole-round fraction. A new placement
  // is initialised from the population model, or Beta(1,1) when that is not identified; its own
  // rounds replace it once they identify a Beta.
  std::pair<double, double> normalModel(const StateKey& key);

  // README (7): mean of the population's normal-round Beta, or 1/2, the mean of the neutral
  // Beta(1,1), when the population does not identify one.
  double populationNormalMean() const;

  // README (7), P1: only inputs after the watermark are new evidence for a placement.
  uint64_t processed(const StateKey& key) const;
  void setProcessed(const StateKey& key, uint64_t stamp);

  // README (7): a round judged in place enters the placement's and the population statistics.
  void recordNormalRound(const StateKey& key, double fraction);

  // README s4: identity conflicts of coincident echoes are accumulated, never used for a decision.
  void recordIdentityRound(size_t conflicts, size_t total);

  bool saveSensorStatistics(const std::string& path) const;
  bool loadSensorStatistics(const std::string& path);
  void save(const std::string& path, uint64_t boundary, const LiveStates& live) const;
  void load(const std::string& path, uint64_t boundary, const LiveStates& live);
  void retain(const LiveStates& live);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace khronos
