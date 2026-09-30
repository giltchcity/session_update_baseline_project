#include "session_core/evidence/predictive_surface.h"
#include "session_core/evidence/physical_evidence_store.h"
#include "session_core/evidence/range_encoding.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <boost/math/policies/policy.hpp>
#include <boost/math/special_functions/beta.hpp>

namespace khronos::measurement {
namespace {

constexpr double kInvSqrt2 = 0.70710678118654752440;
constexpr double kLogSqrt2Pi = 0.91893853320467274178;

double normalCdf(double z) { return 0.5 * std::erfc(-z * kInvSqrt2); }

// log of the normalisation of the Gaussian on the device range, README (6e).
double logNormalisation(double query, double sigma, double r_min, double r_max) {
  return std::log(normalCdf((r_max - query) / sigma) - normalCdf((r_min - query) / sigma));
}

void requireModel(const ErrorModel& psi) {
  if (!psi.valid()) throw std::logic_error("The effective range error model is not available");
}

using BetaPolicy = boost::math::policies::policy<
    boost::math::policies::underflow_error<boost::math::policies::ignore_error>,
    boost::math::policies::denorm_error<boost::math::policies::ignore_error>>;

bool usable(const ProjectedEndpointEvidence& p) {
  const auto& e = p.endpoint;
  return e.type != EndpointClass::kUnavailable && e.type != EndpointClass::kInvalid &&
         std::isfinite(e.measured_depth_m) && std::isfinite(p.query_range_m) &&
         e.measured_depth_m > 0 && p.query_range_m > 0;
}

}  // namespace

SupportInterval supportInterval(double query, double r_min, double r_max, const ErrorModel& psi) {
  requireModel(psi);
  const double sigma = psi.sigmaAt(query);
  // log of the Gaussian density at zero residual minus log of a uniform density 1/length; the
  // surface explains an echo where the Gaussian density is at least the uniform one.
  const double peak = -kLogSqrt2Pi - std::log(sigma) - logNormalisation(query, sigma, r_min, r_max);
  SupportInterval interval{query, query};
  if (r_max > query) {
    const double margin = peak + std::log(r_max - query);
    if (margin > 0) interval.upper = query + sigma * std::sqrt(2.0 * margin);
  }
  if (query > r_min) {
    const double margin = peak + std::log(query - r_min);
    if (margin > 0) interval.lower = query - sigma * std::sqrt(2.0 * margin);
  }
  return interval;
}

SurfaceVote classifySurfaceMeasurement(const ProjectedEndpointEvidence& p, const ErrorModel& psi) {
  if (p.endpoint.type == EndpointClass::kUnavailable) return SurfaceVote::Unavailable;
  if (!usable(p)) return SurfaceVote::Invalid;
  return classifyRange(p.endpoint.measured_depth_m,
                       supportInterval(p.query_range_m, p.sensor_min_range, p.sensor_max_range, psi));
}

KeepModel keepModel(double query, double r_min, double r_max, const ErrorModel& psi) {
  requireModel(psi);
  KeepModel model;
  if (!(query > 0) || !(r_max > query)) return model;
  model.informative = true;
  model.sigma = psi.sigmaAt(query);
  model.kappa = -kLogSqrt2Pi - std::log(model.sigma) -
                logNormalisation(query, model.sigma, r_min, r_max) + std::log(r_max - query);
  const auto interval = supportInterval(query, r_min, r_max, psi);
  model.margin = interval.lower - query;
  model.upper = interval.upper - query;
  return model;
}

double keepLogRatio(double e, const KeepModel& model, double outlier) {
  if (!model.informative || e < model.margin) return 0.0;
  const double gaussian = model.kappa - 0.5 * e * e / (model.sigma * model.sigma);
  if (!(outlier > 0)) return gaussian;
  return std::log(outlier + (1.0 - outlier) * std::exp(gaussian));
}

double betaMass(double a, double b, double x0, double x1) {
  if (!(a > 0) || !(b > 0) || !(x0 >= 0) || !(x1 <= 1)) {
    throw std::invalid_argument("Invalid Beta interval");
  }
  if (!(x1 > x0)) return 0.0;
  const double mass = x0 >= 0.5
      ? boost::math::ibetac(a, b, x0, BetaPolicy{}) - boost::math::ibetac(a, b, x1, BetaPolicy{})
      : boost::math::ibeta(a, b, x1, BetaPolicy{}) - boost::math::ibeta(a, b, x0, BetaPolicy{});
  return std::max(0.0, mass);
}

}  // namespace khronos::measurement
