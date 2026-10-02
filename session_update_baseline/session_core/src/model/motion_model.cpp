#include "session_core/model/motion_model.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include <boost/math/distributions/chi_squared.hpp>

#include "session_core/model/model_math.h"

namespace khronos::model {
namespace {
// The centroid has three coordinates (degrees of freedom of the gate, README principle 5).
constexpr double kCoordinates = 3.0;
// The difference of two centroids carries the jitter twice (the gate formula of principle 5).
constexpr double kJitterTerms = 2.0;
}  // namespace

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
  // sigma_c: the isotropic per-axis jitter of the cluster centroid.
  const double jitter_sq = offset_sum_sq_ / offset_count_ / kCoordinates;
  const double speed_sq = speed_sum_sq_ / speed_count_;
  const double quantile = boost::math::quantile(boost::math::chi_squared(kCoordinates), 1.0 - kAlpha);
  return std::sqrt(quantile) * std::sqrt(speed_sq * seconds * seconds + kJitterTerms * jitter_sq);
}

nlohmann::json MotionModel::toJson() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return nlohmann::json{{"offset", nlohmann::json::array({offset_sum_sq_, offset_count_})},
                        {"speed", nlohmann::json::array({speed_sum_sq_, speed_count_})}};
}

void MotionModel::fromJson(const nlohmann::json& value) {
  std::lock_guard<std::mutex> lock(mutex_);
  offset_sum_sq_ = value.at("offset").at(0).get<double>();
  offset_count_ = value.at("offset").at(1).get<double>();
  speed_sum_sq_ = value.at("speed").at(0).get<double>();
  speed_count_ = value.at("speed").at(1).get<double>();
  if (!(offset_sum_sq_ >= 0) || !(offset_count_ >= 0) || !(speed_sum_sq_ >= 0) || !(speed_count_ >= 0)) {
    throw std::invalid_argument("Invalid motion statistics");
  }
}

}  // namespace khronos::model
