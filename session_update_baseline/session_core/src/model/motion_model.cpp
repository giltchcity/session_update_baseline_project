#include "session_core/model/motion_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <boost/math/distributions/chi_squared.hpp>

#include "session_core/model/model_math.h"

namespace khronos::model {

double MotionModel::logMovingRatio(size_t object, double n, double k) const {
  if (!(n > 0.0) || k < 0.0 || k > n) return 0.0;
  std::lock_guard<std::mutex> lock(mutex_);
  if (!(share_pixels_ > 0.0)) return 0.0;
  // The share the learned static frames show (Jeffreys smoothing) is the cold mean of (7h).
  const double cold = (share_covered_ + 0.5) / (share_pixels_ + 1.0);
  const auto in_place = static_.inPlace(object, cold);
  // The look ratio (7) is ended : in place; moving : static is its inverse sign convention of the
  // look: L(moving) = BB(k | n; 1, 1) over L(static) = BB(max(k, round(m n)) | n; m c, (1 - m) c).
  return RoundModel::logLikelihoodRatio(in_place, n, k);
}

void MotionModel::addStaticFrame(size_t object, double n, double k) {
  if (!(n > 0.0) || k < 0.0 || k > n) return;
  std::lock_guard<std::mutex> lock(mutex_);
  static_.addInPlaceLook(object, -1, k / n);
  share_pixels_ += n;
  share_covered_ += k;
}

void MotionModel::addCentroidOffset(double offset) {
  if (!(offset >= 0.0) || !std::isfinite(offset)) return;
  std::lock_guard<std::mutex> lock(mutex_);
  offset_sum_sq_ += offset * offset;
  offset_count_ += 1.0;
}

void MotionModel::addSpeed(double speed) {
  if (!(speed >= 0.0) || !std::isfinite(speed)) return;
  std::lock_guard<std::mutex> lock(mutex_);
  speed_sum_sq_ += speed * speed;
  speed_count_ += 1.0;
}

double MotionModel::associationGate(double seconds) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!(offset_count_ > 0.0) || !(speed_count_ > 0.0)) return std::numeric_limits<double>::infinity();
  // sigma_c: the isotropic per-axis jitter (3 degrees of freedom) of the cluster centroid.
  const double jitter_sq = offset_sum_sq_ / offset_count_ / 3.0;
  const double speed_sq = speed_sum_sq_ / speed_count_;
  const double quantile = boost::math::quantile(boost::math::chi_squared(3.0), 1.0 - kAlpha);
  return std::sqrt(quantile) * std::sqrt(speed_sq * seconds * seconds + 2.0 * jitter_sq);
}

nlohmann::json MotionModel::toJson() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return nlohmann::json{{"static", static_.toJson()},
                        {"share", nlohmann::json::array({share_pixels_, share_covered_})},
                        {"offset", nlohmann::json::array({offset_sum_sq_, offset_count_})},
                        {"speed", nlohmann::json::array({speed_sum_sq_, speed_count_})}};
}

void MotionModel::fromJson(const nlohmann::json& value) {
  std::lock_guard<std::mutex> lock(mutex_);
  static_ = RoundModel::fromJson(value.at("static"));
  share_pixels_ = value.at("share").at(0).get<double>();
  share_covered_ = value.at("share").at(1).get<double>();
  offset_sum_sq_ = value.at("offset").at(0).get<double>();
  offset_count_ = value.at("offset").at(1).get<double>();
  speed_sum_sq_ = value.at("speed").at(0).get<double>();
  speed_count_ = value.at("speed").at(1).get<double>();
  if (!(share_pixels_ >= 0) || !(share_covered_ >= 0) || share_covered_ > share_pixels_ ||
      !(offset_sum_sq_ >= 0) || !(offset_count_ >= 0) || !(speed_sum_sq_ >= 0) || !(speed_count_ >= 0)) {
    throw std::invalid_argument("Invalid motion statistics");
  }
}

}  // namespace khronos::model
