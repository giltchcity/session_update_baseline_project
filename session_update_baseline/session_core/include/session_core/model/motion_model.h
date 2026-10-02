#pragma once

#include <cstddef>
#include <mutex>

#include <nlohmann/json.hpp>

namespace khronos::model {

/**
 * README principle 5: the association of a motion cluster without an identity to a dynamic track.
 * The native motion mask only marks readings; it commits no state, so no motion statistics of
 * placements are kept. A cluster is associated to the dynamic track whose predicted position it is
 * nearest (the maximum likelihood candidate of (12a)), inside the (1 - alpha) chi-square gate
 * sqrt(chi2_{3,1-alpha}) sqrt((v dt)^2 + 2 sigma_c^2) once the centroid jitter sigma_c and the speed
 * v of the associated tracks have been estimated; before that there is no gate.
 */
class MotionModel {
 public:
  /** The deviation of an associated centroid from its predicted position. */
  void addCentroidOffset(double offset);
  /** The speed of an associated track between two frames. */
  void addSpeed(double speed);
  /** The gate over `seconds`; infinity while either statistic is unknown. */
  double associationGate(double seconds) const;

  nlohmann::json toJson() const;
  void fromJson(const nlohmann::json& value);

 private:
  mutable std::mutex mutex_;
  double offset_sum_sq_ = 0.0, offset_count_ = 0.0;
  double speed_sum_sq_ = 0.0, speed_count_ = 0.0;
};

}  // namespace khronos::model
