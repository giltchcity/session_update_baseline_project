#include "session_core/model/motion_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <boost/math/distributions/chi_squared.hpp>

#include "session_core/model/model_math.h"
#include "session_core/model/round_model.h"

namespace khronos::model {
namespace {
// Computation budget of the frame statistics: the newest frames of an object and the number of
// added frames between two refits of the moment estimates.
constexpr size_t kFramesPerObject = 512;
constexpr size_t kRefitInterval = 64;
constexpr double kCorrelationBound = 1.0e-6;
}  // namespace

double MotionModel::logMovingRatio(size_t object, double n, double k) const {
  if (!(n > 0.0) || k < 0.0 || k > n) return 0.0;
  std::lock_guard<std::mutex> lock(mutex_);
  if (fit_dirty_ || added_since_fit_ >= kRefitInterval) {
    fit_ = static_.fit();
    fit_dirty_ = false;
    added_since_fit_ = 0;
  }
  double a, b;
  if (!static_.predict(fit_, object, a, b)) return 0.0;
  return logBetaBinomial(k, n, 1.0, 1.0) - logBetaBinomial(k, n, a, b);
}

double MotionModel::frameExponent() const {
  std::lock_guard<std::mutex> lock(mutex_);
  double rho;
  if (!static_.lagOneCorrelation(rho)) return 1.0;
  return RoundModel::autocorrelationExponent(rho);  // the projection of README (7h)
}

void MotionModel::addStaticFrame(size_t object, double n, double k) {
  if (!(n > 0.0) || k < 0.0 || k > n) return;
  std::lock_guard<std::mutex> lock(mutex_);
  static_.add(object, n, k, kFramesPerObject);
  ++added_since_fit_;
}

double MotionModel::stopProbability(double seconds, double ongoing_motion_seconds) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const double exposure = motion_seconds_ + ongoing_motion_seconds;
  if (!(exposure > 0.0) || !(seconds >= 0.0)) return 0.0;
  const double rate = (stops_ + 0.5) / exposure;
  return -std::expm1(-rate * seconds);
}

void MotionModel::addMotionSegment(double seconds) {
  if (!(seconds >= 0.0)) return;
  std::lock_guard<std::mutex> lock(mutex_);
  stops_ += 1.0;
  motion_seconds_ += seconds;
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
                        {"stops", stops_},
                        {"motion_seconds", motion_seconds_},
                        {"offset", nlohmann::json::array({offset_sum_sq_, offset_count_})},
                        {"speed", nlohmann::json::array({speed_sum_sq_, speed_count_})}};
}

void MotionModel::fromJson(const nlohmann::json& value) {
  std::lock_guard<std::mutex> lock(mutex_);
  static_.fromJson(value.at("static"));
  stops_ = value.at("stops").get<double>();
  motion_seconds_ = value.at("motion_seconds").get<double>();
  offset_sum_sq_ = value.at("offset").at(0).get<double>();
  offset_count_ = value.at("offset").at(1).get<double>();
  speed_sum_sq_ = value.at("speed").at(0).get<double>();
  speed_count_ = value.at("speed").at(1).get<double>();
  if (!(stops_ >= 0) || !(motion_seconds_ >= 0) || !(offset_sum_sq_ >= 0) || !(offset_count_ >= 0) ||
      !(speed_sum_sq_ >= 0) || !(speed_count_ >= 0)) {
    throw std::invalid_argument("Invalid motion statistics");
  }
  fit_dirty_ = true;
  added_since_fit_ = 0;
}

}  // namespace khronos::model
