#pragma once

#include <cstddef>
#include <mutex>

#include <nlohmann/json.hpp>

#include "session_core/model/hierarchical_beta.h"

namespace khronos::model {

/**
 * README principle 5: the statistics of visible motion, learned online from committed static
 * frames and committed motion segments.
 *
 *  - The share c of an object's pixels that lie in motion clusters is, while the object is static,
 *    beta-binomial under the same hierarchical family as principle 6 (a normal share per object,
 *    estimated from the frames committed static); while it moves the share carries no information
 *    (the uniform Beta(1, 1)). The frame likelihood ratio moving : static feeds the recursion (5r)
 *    with the persistence prior of the object.
 *  - Motion ends with the hazard lambda_stop, events over exposure of the committed motion
 *    segments, Jeffreys posterior mean.
 *  - A motion cluster without an identity is associated to a dynamic track inside the (1 - alpha)
 *    chi-square gate sqrt(chi2_{3,1-alpha}) sqrt((v dt)^2 + 2 sigma_c^2) from the centroid jitter
 *    sigma_c of clusters on static objects and the speed v of committed motion.
 */
class MotionModel {
 public:
  /** ln [L(moving) / L(static)] of one frame of `object` with k of n pixels covered by motion.
   * `exponent` is the autocorrelation weight. Neutral (0) while no static frame has been
   * committed. */
  double logMovingRatio(size_t object, double n, double k) const;
  /** The exponent w = (1 - rho_1) / (1 + rho_1) of the frame sequence; 1 while rho_1 is unknown. */
  double frameExponent() const;
  void addStaticFrame(size_t object, double n, double k);

  /** q_stop over `seconds`, from lambda_stop = (stops + 1/2) / (committed motion time + the motion
   * time of the ongoing segment). */
  double stopProbability(double seconds, double ongoing_motion_seconds) const;
  void addMotionSegment(double seconds);

  void addCentroidOffset(double offset);
  void addSpeed(double speed);
  /** The gate over `seconds`; infinity while either statistic is unknown. */
  double associationGate(double seconds) const;

  nlohmann::json toJson() const;
  void fromJson(const nlohmann::json& value);

 private:
  mutable std::mutex mutex_;
  HierarchicalBeta static_;
  mutable HierarchicalBeta::Fit fit_;
  mutable bool fit_dirty_ = true;
  mutable size_t added_since_fit_ = 0;
  double stops_ = 0.0, motion_seconds_ = 0.0;
  double offset_sum_sq_ = 0.0, offset_count_ = 0.0;
  double speed_sum_sq_ = 0.0, speed_count_ = 0.0;
};

}  // namespace khronos::model
