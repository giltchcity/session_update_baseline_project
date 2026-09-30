#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>

namespace khronos {
class RayVerificator;
class ObservedAbsenceBatch;

// README (7b), (15b): one model owned by one mapping instance.
class ObservedAbsenceModel {
 public:
  using StateKey = std::pair<size_t, uint64_t>;
  using LiveStates = std::set<StateKey>;
  ObservedAbsenceModel();
  ~ObservedAbsenceModel();
  ObservedAbsenceModel(const ObservedAbsenceModel&) = delete;
  ObservedAbsenceModel& operator=(const ObservedAbsenceModel&) = delete;
  bool saveSensorStatistics(const std::string& path) const;
  bool loadSensorStatistics(const std::string& path);
  void save(const std::string& path, uint64_t boundary, const LiveStates& live) const;
  void load(const std::string& path, uint64_t boundary, const LiveStates& live);
  void retain(const LiveStates& live);
 private:
  friend class RayVerificator;
  friend class ObservedAbsenceBatch;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// One immutable population model for a reconciliation pass, including nested queries.
class ObservedAbsenceBatch {
 public:
  explicit ObservedAbsenceBatch(const ObservedAbsenceModel& model);
  ~ObservedAbsenceBatch();
  ObservedAbsenceBatch(const ObservedAbsenceBatch&) = delete;
  ObservedAbsenceBatch& operator=(const ObservedAbsenceBatch&) = delete;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace khronos
