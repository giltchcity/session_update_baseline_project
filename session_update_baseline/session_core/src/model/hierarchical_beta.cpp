#include "session_core/model/hierarchical_beta.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace khronos::model {
namespace {

// Numerical bounds of the concentrations: complete pooling (binomial) above, the smallest
// identifiable concentration below; iterations of the bisection (computation budget).
constexpr double kMaxConcentration = 1.0e6;
constexpr double kMinConcentration = 1.0e-2;
constexpr int kBisections = 80;

}  // namespace

double effectiveRoundSize(double n, double phi) { return n * (phi + 1.0) / (n + phi); }

double smoothedProportion(double k, double n) { return (k + 0.5) / (n + 1.0); }

// The concentration s of Beta(m s, (1 - m) s) for which the pooled squared deviation of the
// fractions equals its beta-binomial expectation,
//   sum c (x - m)^2 = sum c m (1 - m) (N + s) / (N (s + 1)),
// whose right side decreases in s from the Bernoulli variance to the binomial one.
double concentrationFromMoments(const std::vector<std::array<double, 3>>& entries, double m) {
  double left = 0.0;
  for (const auto& e : entries) left += e[2] * (e[0] - m) * (e[0] - m);
  const auto right = [&](double s) {
    double sum = 0.0;
    for (const auto& e : entries) sum += e[2] * m * (1.0 - m) * (e[1] + s) / (e[1] * (s + 1.0));
    return sum;
  };
  if (right(kMinConcentration) <= left) return kMinConcentration;
  if (right(kMaxConcentration) >= left) return kMaxConcentration;
  double lo = std::log(kMinConcentration), hi = std::log(kMaxConcentration);
  for (int i = 0; i < kBisections; ++i) {
    const double mid = 0.5 * (lo + hi);
    (right(std::exp(mid)) > left ? lo : hi) = mid;
  }
  return std::exp(0.5 * (lo + hi));
}

HierarchicalBeta::Fit HierarchicalBeta::fit() const {
  Fit result;
  // The round-to-round variance needs an object with two rounds.
  bool repeated = false;
  for (const auto& [id, observations] : rounds_) {
    (void)id;
    repeated = repeated || observations.size() >= 2;
  }
  if (!repeated) return result;

  // phi: the within-object weighted scatter of the round fractions equals its expectation
  // (R - 1) m (1 - m) when each round is weighted by its information n (phi + 1) / (n + phi).
  const auto within = [&](double phi) {
    double scatter = 0.0, expectation = 0.0;
    for (const auto& [id, observations] : rounds_) {
      (void)id;
      if (observations.size() < 2) continue;
      double total_n = 0.0, total_k = 0.0, weight = 0.0, weighted = 0.0;
      for (const auto& [n, k] : observations) {
        total_n += n;
        total_k += k;
        const double w = effectiveRoundSize(n, phi);
        weight += w;
        weighted += w * (k / n);
      }
      const double mean = weighted / weight;
      for (const auto& [n, k] : observations) {
        scatter += effectiveRoundSize(n, phi) * (k / n - mean) * (k / n - mean);
      }
      const double m = smoothedProportion(total_k, total_n);
      expectation += static_cast<double>(observations.size() - 1) * m * (1.0 - m);
    }
    return scatter - expectation;
  };
  double phi;
  if (within(kMinConcentration) >= 0.0) {
    phi = kMinConcentration;
  } else if (within(kMaxConcentration) <= 0.0) {
    phi = kMaxConcentration;
  } else {
    double lo = std::log(kMinConcentration), hi = std::log(kMaxConcentration);
    for (int i = 0; i < kBisections; ++i) {
      const double mid = 0.5 * (lo + hi);
      (within(std::exp(mid)) < 0.0 ? lo : hi) = mid;
    }
    phi = std::exp(0.5 * (lo + hi));
  }

  // mu: the information-weighted mean of all rounds (Jeffreys-smoothed).
  double weight = 0.0, weighted = 0.0;
  for (const auto& [id, observations] : rounds_) {
    (void)id;
    for (const auto& [n, k] : observations) {
      const double w = effectiveRoundSize(n, phi);
      weight += w;
      weighted += w * (k / n);
    }
  }
  const double mu = (weighted + 0.5) / (weight + 1.0);

  // kappa: the between-object variance of the object means, less their sampling variance.
  double scatter = 0.0, sampling = 0.0;
  size_t objects = 0;
  for (const auto& [id, observations] : rounds_) {
    (void)id;
    double w_sum = 0.0, wf = 0.0;
    for (const auto& [n, k] : observations) {
      const double w = effectiveRoundSize(n, phi);
      w_sum += w;
      wf += w * (k / n);
    }
    if (!(w_sum > 0.0)) continue;
    const double mean = wf / w_sum;
    scatter += (mean - mu) * (mean - mu);
    sampling += mu * (1.0 - mu) / w_sum;
    ++objects;
  }
  double kappa = kMaxConcentration;
  if (objects >= 2) {
    const double tau2 = (scatter - sampling) / static_cast<double>(objects);
    if (tau2 > 0.0) {
      kappa = std::clamp(mu * (1.0 - mu) / tau2 - 1.0, kMinConcentration, kMaxConcentration);
    }
  }
  result.available = true;
  result.mu = mu;
  result.phi = phi;
  result.kappa = kappa;
  return result;
}

bool HierarchicalBeta::predict(const Fit& fit, size_t object, double& a, double& b) const {
  if (!fit.available) return false;
  double weight = 0.0, weighted = 0.0;
  const auto found = rounds_.find(object);
  if (found != rounds_.end()) {
    for (const auto& [n, k] : found->second) {
      const double w = effectiveRoundSize(n, fit.phi);
      weight += w;
      weighted += w * (k / n);
    }
  }
  // (7h): empirical-Bayes partial pooling of the object mean.
  const double m = (fit.kappa * fit.mu + weighted) / (fit.kappa + weight);
  const double spread = m * (1.0 - m);
  // The predictive fraction of a new round: Var f = E[m(1-m)]/(phi+1) + Var m, matched to a beta.
  const double variance_m = spread / (fit.kappa + weight + 1.0);
  const double variance_f = (spread - variance_m) / (fit.phi + 1.0) + variance_m;
  const double s = spread / variance_f - 1.0;
  if (!(s > 0.0) || !std::isfinite(s)) return false;
  a = m * s;
  b = (1.0 - m) * s;
  return a > 0.0 && b > 0.0;
}

bool HierarchicalBeta::lagOneCorrelation(double& rho) const {
  double numerator = 0.0, denominator = 0.0;
  size_t pairs = 0;
  for (const auto& [id, observations] : rounds_) {
    (void)id;
    if (observations.size() < 2) continue;
    std::vector<double> x;
    for (const auto& [n, k] : observations) {
      const double p = smoothedProportion(k, n);
      x.push_back(std::log(p / (1.0 - p)));
    }
    double mean = 0.0;
    for (const double v : x) mean += v / static_cast<double>(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      denominator += (x[i] - mean) * (x[i] - mean);
      if (i + 1 < x.size()) {
        numerator += (x[i] - mean) * (x[i + 1] - mean);
        ++pairs;
      }
    }
  }
  if (pairs < 2 || !(denominator > 0.0)) return false;
  rho = numerator / denominator;
  return true;
}

nlohmann::json HierarchicalBeta::toJson() const {
  nlohmann::json objects = nlohmann::json::array();
  for (const auto& [id, observations] : rounds_) {
    nlohmann::json list = nlohmann::json::array();
    for (const auto& [n, k] : observations) list.push_back(nlohmann::json::array({n, k}));
    objects.push_back(nlohmann::json::array({id, std::move(list)}));
  }
  return objects;
}

void HierarchicalBeta::fromJson(const nlohmann::json& value) {
  rounds_.clear();
  for (const auto& item : value) {
    auto& list = rounds_[item.at(0).get<size_t>()];
    for (const auto& o : item.at(1)) {
      const double n = o.at(0).get<double>(), k = o.at(1).get<double>();
      if (!(n > 0.0) || !(k >= 0.0) || k > n) throw std::invalid_argument("Invalid round record");
      list.push_back({n, k});
    }
  }
}

}  // namespace khronos::model
