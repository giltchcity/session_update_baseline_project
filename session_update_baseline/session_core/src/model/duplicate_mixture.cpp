#include "session_core/model/duplicate_mixture.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace khronos::model {
namespace {

constexpr double kSqrt2Pi = 2.50662827463100050242;

// ln N(d; mean, variance).
double logGaussian(double d, double mean, double variance) {
  const double delta = d - mean;
  return -0.5 * delta * delta / variance - 0.5 * std::log(variance) - std::log(kSqrt2Pi);
}

// ln of the same-surface term and of the different-surface term of (12d), without the weights.
void logTerms(const DuplicateFit& fit, const BandPair& pair, double& log_dup, double& log_sep) {
  const double variance = std::max(pair.base_variance + fit.sigma_x * fit.sigma_x,
                                   std::numeric_limits<double>::min());
  log_dup = logGaussian(pair.d, fit.delta_s * pair.rho, variance);
  log_sep = -std::log(2.0 * pair.band);
}

}  // namespace

double duplicateLogOdds(const DuplicateFit& fit, const BandPair& pair) {
  if (!(pair.band > 0.0) || std::abs(pair.d) > pair.band) return -std::numeric_limits<double>::infinity();
  double log_dup, log_sep;
  logTerms(fit, pair, log_dup, log_sep);
  return std::log(fit.pi_dup / (1.0 - fit.pi_dup)) + log_dup - log_sep;
}

DuplicateFit fitDuplicateMixture(const std::vector<BandPair>& pairs, const DuplicateFit& start,
                                 bool fit_bias, int max_iterations, double tolerance) {
  DuplicateFit fit = start;
  fit.pi_dup = std::clamp(start.pi_dup, 1e-6, 1.0 - 1e-6);
  if (!fit_bias) {
    fit.delta_s = 0.0;
    fit.sigma_x = 0.0;
  }
  std::vector<const BandPair*> inside;
  inside.reserve(pairs.size());
  for (const auto& pair : pairs) {
    if (pair.band > 0.0 && std::abs(pair.d) <= pair.band && std::isfinite(pair.d) &&
        pair.base_variance >= 0.0) {
      inside.push_back(&pair);
    }
  }
  fit.pairs = inside.size();
  if (inside.empty()) return fit;

  std::vector<double> responsibility(inside.size());
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    // E-step: the share of the same-surface term.
    double sum = 0.0;
    for (size_t i = 0; i < inside.size(); ++i) {
      double log_dup, log_sep;
      logTerms(fit, *inside[i], log_dup, log_sep);
      const double a = std::log(fit.pi_dup) + log_dup, b = std::log1p(-fit.pi_dup) + log_sep;
      const double m = std::max(a, b);
      responsibility[i] = std::exp(a - m) / (std::exp(a - m) + std::exp(b - m));
      sum += responsibility[i];
    }
    // M-step.
    DuplicateFit next = fit;
    next.pi_dup = (sum + 0.5) / (static_cast<double>(inside.size()) + 1.0);
    if (fit_bias) {
      // Weighted least squares of d on rho with the weights r_i / sigma_dup,i^2 (the maximum
      // likelihood of the Gaussian term for a given sigma_x).
      double numerator = 0.0, denominator = 0.0;
      for (size_t i = 0; i < inside.size(); ++i) {
        const double weight = responsibility[i] /
            std::max(inside[i]->base_variance + fit.sigma_x * fit.sigma_x, std::numeric_limits<double>::min());
        numerator += weight * inside[i]->rho * inside[i]->d;
        denominator += weight * inside[i]->rho * inside[i]->rho;
      }
      if (denominator > 0.0) next.delta_s = numerator / denominator;
      double excess = 0.0;
      for (size_t i = 0; i < inside.size(); ++i) {
        const double residual = inside[i]->d - next.delta_s * inside[i]->rho;
        excess += responsibility[i] * (residual * residual - inside[i]->base_variance);
      }
      next.sigma_x = sum > 0.0 ? std::sqrt(std::max(0.0, excess / sum)) : 0.0;
    }
    const double change = std::abs(next.pi_dup - fit.pi_dup) + std::abs(next.delta_s - fit.delta_s) +
                          std::abs(next.sigma_x - fit.sigma_x);
    fit = next;
    fit.pairs = inside.size();
    if (change < tolerance) break;
  }
  fit.fitted = true;
  return fit;
}

}  // namespace khronos::model
