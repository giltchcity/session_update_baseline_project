#include "session_core/evidence/predictive_surface.h"
#include "session_core/evidence/physical_evidence_store.h"
#include "session_core/evidence/range_encoding.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include <boost/math/policies/policy.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/tools/fraction.hpp>

namespace khronos::measurement {
SurfaceVote classifySurfaceMeasurement(const ProjectedEndpointEvidence& p, size_t id, float tolerance) {
  if (!std::isfinite(tolerance) || tolerance < 0.f)
    throw std::invalid_argument("Invalid surface measurement tolerance");
  const auto& e = p.endpoint;
  if (e.type == EndpointClass::kUnavailable) return SurfaceVote::Unavailable;
  if (e.type == EndpointClass::kInvalid || !std::isfinite(e.measured_depth_m) ||
      !std::isfinite(p.query_range_m) || e.measured_depth_m <= 0 || p.query_range_m <= 0) {
    return SurfaceVote::Invalid;
  }
  const float delta = e.measured_depth_m - p.query_range_m;
  if (delta < -tolerance) return SurfaceVote::Occluded;
  const bool same_identity = e.type == EndpointClass::kPhysical &&
      e.physical_id > 0 && static_cast<size_t>(e.physical_id) == id;
  // README (6e): only the measured endpoint is quantized. A different
  // identity entirely in front of the continuous query provides occlusion.
  if (!same_identity && delta < -0.5f * kRangeUnit) return SurfaceVote::Occluded;
  if (delta > tolerance) return SurfaceVote::Free;
  if (e.type == EndpointClass::kBackground) return SurfaceVote::Background;
  if (e.type == EndpointClass::kUnidentifiedObject) return SurfaceVote::Unidentified;
  if (e.type == EndpointClass::kPhysical) {
    return same_identity ? SurfaceVote::Supported : SurfaceVote::Other;
  }
  return SurfaceVote::Invalid;
}


namespace {
using BetaPolicy = boost::math::policies::policy<
    boost::math::policies::underflow_error<boost::math::policies::ignore_error>,
    boost::math::policies::denorm_error<boost::math::policies::ignore_error>>;

void validateReference(BetaReference reference) {
  if (!(std::isfinite(reference.a) && std::isfinite(reference.b) &&
        reference.a > 0 && reference.b > 0)) {
    throw std::invalid_argument("Invalid predictive beta reference");
  }
}

// Production reference concentration is at most 1502 and the completed count
// at most 1500, so the posterior shapes sum to at most 3002. The ordinary
// long-double lgamma calculation covers this domain without a custom asymptotic.
long double logBeta(long double a, long double b) {
  return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
}

// The incomplete-beta continued fraction used by Boost beta.hpp
// (ibeta_fraction2). Only the public generic continued-fraction evaluator is
// reused here. The first numerator is zero, including at a=1; forming 0/0
// from the unreduced expression would otherwise corrupt this valid case.
struct LowerBetaFraction {
  using result_type = std::pair<long double,long double>;
  long double a, b, x, y;
  unsigned long index = 0;

  result_type operator()() {
    const long double m = index++;
    long double numerator = 0;
    long double middle = 0;
    if (m != 0) {
      const long double denominator = a + 2 * m - 1;
      numerator = ((a + m - 1) / denominator) *
          ((a + b + m - 1) / denominator) * m * (b - m) * x * x;
      middle = (m / denominator) * (b - m) * x;
    }
    const long double denominator = m + middle + ((a + m) / (a + 2 * m + 1)) *
        (a * y - b * x + 1 + m * (2 - x));
    return {numerator,denominator};
  }
};

// Integral over [0,alpha], or [alpha,1] when upper=true. Regularized Boost
// evaluation covers the production alpha=0.5/0.2 domains. Configurable alpha
// can make a tail subnormal/zero; only that tail uses the same integral's
// log B_x = a*log(x) + b*log(1-x) - log(fraction), without forming exp(log B_x).
long double logBetaIntegral(long double a, long double b, long double alpha, bool upper) {
  if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(a + b))
    throw std::overflow_error("Predictive beta shape sum exceeds numeric range");
  const long double probability = upper ? boost::math::ibetac(a,b,alpha,BetaPolicy{})
                                        : boost::math::ibeta(a,b,alpha,BetaPolicy{});
  if (std::isnan(probability) || probability < 0 || probability > 1)
    throw std::runtime_error("Invalid incomplete-beta probability");
  if (probability >= std::numeric_limits<long double>::min())
    return logBeta(a,b) + std::log(probability);

  // Preserve log(1-alpha) with log1p even when 1-alpha rounds to one.
  const long double log_alpha = std::log(alpha), log_complement = std::log1p(-alpha);
  const long double x = upper ? 1 - alpha : alpha;
  const long double y = upper ? alpha : 1 - alpha;
  if (upper) std::swap(a,b);
  LowerBetaFraction fraction{a,b,x,y};
  std::uintmax_t terms = boost::math::policies::get_max_series_iterations<BetaPolicy>();
  const long double denominator = boost::math::tools::continued_fraction_b(
      fraction,std::numeric_limits<long double>::epsilon(),terms);
  if (terms >= boost::math::policies::get_max_series_iterations<BetaPolicy>())
    throw std::runtime_error("Predictive beta continued fraction did not converge");
  if (!(std::isfinite(denominator) && denominator > 0))
    throw std::runtime_error("Invalid predictive beta continued fraction");
  return a * (upper ? log_complement : log_alpha) +
      b * (upper ? log_alpha : log_complement) - std::log(denominator);
}

long double logRetaining(CompletedEvidence evidence, long double alpha,
                         BetaReference reference) {
  const auto [a,b] = reference;
  const long double k = evidence.absent, retained = evidence.total - evidence.absent;
  return logBetaIntegral(a + k,b + retained,alpha,false) -
      logBetaIntegral(a,b,alpha,false);
}

long double logEqualMixture(long double first, long double second) {
  if (std::isnan(first) || std::isnan(second))
    throw std::runtime_error("Invalid predictive mixture component");
  const long double high = std::max(first,second), low = std::min(first,second);
  // Both -infinity represents zero mass; a single -infinity contributes zero.
  // Resolve infinities before subtracting high, which would otherwise give NaN.
  if (std::isinf(high)) return high;
  if (low == -std::numeric_limits<long double>::infinity()) return high - std::log(2.L);
  return high + std::log1p(std::exp(low - high)) - std::log(2.L);
}
}  // namespace

CompletedEvidence retainingCompletion(size_t absent, size_t measured, size_t unknown) {
  if (absent > measured)
    throw std::invalid_argument("Invalid grouped surface evidence counts");
  const size_t extra = unknown > absent ? unknown - absent : 0;
  if (extra > std::numeric_limits<size_t>::max() - measured)
    throw std::overflow_error("Completed surface evidence count exceeds numeric range");
  return {absent > unknown ? absent - unknown : 0,measured + extra};
}

long double surfaceLogOdds(CompletedEvidence evidence, long double alpha,
                           BetaReference reference, std::optional<BetaReference> second) {
  if (evidence.absent > evidence.total || !std::isfinite(alpha) || alpha < 0 || alpha >= 1)
    throw std::invalid_argument("Invalid predictive surface domain");
  validateReference(reference);
  if (second) validateReference(*second);
  // Both normalized hypotheses assign probability one to an empty sequence.
  if (evidence.total == 0) return 0;
  const long double k = evidence.absent, retained = evidence.total - evidence.absent;
  if (alpha == 0)
    return evidence.absent ? std::numeric_limits<long double>::infinity()
                           : -std::log(static_cast<long double>(evidence.total) + 1);
  const long double absent = logBetaIntegral(k + 1,retained + 1,alpha,true) -
      std::log1p(-alpha);
  auto present = logRetaining(evidence,alpha,reference);
  if (second) present = logEqualMixture(present,logRetaining(evidence,alpha,*second));
  const auto odds = absent - present;
  if (std::isnan(odds)) throw std::runtime_error("Indeterminate surface predictive odds");
  return odds;
}

bool favorsExit(long double log_odds) {
  if (std::isnan(log_odds)) throw std::invalid_argument("Invalid surface predictive odds");
  return log_odds > std::log(99.L);
}

}  // namespace khronos::measurement
