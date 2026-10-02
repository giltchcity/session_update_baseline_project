#pragma once

#include <cstddef>
#include <mutex>

#include <nlohmann/json.hpp>

#include "session_core/model/round_model.h"

namespace khronos::model {

/**
 * README principle 5: the statistics of visible motion, learned online from the frames that showed
 * a placement static.
 *
 *  - The share k/n of an object's pixels that lie in motion clusters is, while the object is static,
 *    beta-binomial under the in-place distribution of principle 6 (the object's own normal share
 *    shrunk to the population); while it moves the share carries no information (Beta(1, 1)). The
 *    frame likelihood ratio moving : static is the look ratio (7); it feeds a CUSUM (5r) without
 *    prior, both for the start of motion (D1) and for its end (settling).
 *  - A motion cluster without an identity is associated to a dynamic track inside the (1 - alpha)
 *    chi-square gate sqrt(chi2_{3,1-alpha}) sqrt((v dt)^2 + 2 sigma_c^2) from the centroid jitter
 *    sigma_c of clusters on static objects and the speed v of committed motion.
 */
class MotionModel {
 public:
  /** ln [L(moving) / L(static)] of one frame of `object` with k of n pixels covered by motion.
   * Neutral (0) while no static frame has been learned. */
  double logMovingRatio(size_t object, double n, double k) const;
  /** A frame that showed the placement static (its CUSUM stayed at 0) enters the object's normal
   * share; learned after the frame was judged. */
  void addStaticFrame(size_t object, double n, double k);

  void addCentroidOffset(double offset);
  void addSpeed(double speed);
  /** The gate over `seconds`; infinity while either statistic is unknown. */
  double associationGate(double seconds) const;

  nlohmann::json toJson() const;
  void fromJson(const nlohmann::json& value);

 private:
  mutable std::mutex mutex_;
  RoundModel static_;
  double share_pixels_ = 0.0, share_covered_ = 0.0;  // pooled over all learned static frames
  double offset_sum_sq_ = 0.0, offset_count_ = 0.0;
  double speed_sum_sq_ = 0.0, speed_count_ = 0.0;
};

}  // namespace khronos::model
