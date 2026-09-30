#include "session_core/model/persistence_prior.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "session_core/model/model_math.h"

namespace khronos::model {
namespace {

// Computation budget of the marginal-likelihood integrals and the simplex search. The hazard grid
// spans 1e-9 .. 1 per second (mean lifetimes of one second to decades), the probability grid
// logit -20 .. 20; the concentration bound is complete pooling.
constexpr size_t kHazardNodes = 300;
constexpr size_t kLogitNodes = 401;
constexpr double kMinLogRate = -20.7232658369464;  // ln 1e-9
constexpr double kLogitBound = 20.0;
constexpr double kMaxConcentration = 1.0e6;
constexpr double kMinConcentration = 1.0e-2;
constexpr int kSearchIterations = 400;
constexpr double kSearchTolerance = 1.0e-9;
constexpr double kTinyRelative = 1.0e-12;

// ln of the negative-binomial probability of k events when lambda_l ~ Gamma(kappa, kappa / lambda):
// the marginal of a Poisson count with mean lambda_l E.
double logNegativeBinomial(double k, double kappa, double lambda, double exposure) {
  const double mean = lambda * exposure;
  return std::lgamma(k + kappa) - std::lgamma(kappa) - std::lgamma(k + 1.0) -
         kappa * std::log1p(mean / kappa) + k * (std::log(mean) - std::log(kappa + mean));
}

double sigmoid(double t) { return 1.0 / (1.0 + std::exp(-t)); }

using CountList = std::vector<std::pair<double, double>>;  // (k, E) or (g, n)

double gammaLogMarginal(const std::map<int, CountList>& classes, const Grid& grid, double a,
                        double b, double kappa) {
  const double log_norm = a * std::log(b) - std::lgamma(a);
  double total = 0.0;
  std::vector<double> terms(grid.u.size());
  for (const auto& [cls, objects] : classes) {
    (void)cls;
    for (size_t i = 0; i < grid.u.size(); ++i) {
      const double u = grid.u[i], lambda = std::exp(u);
      double value = log_norm + (a - 1.0) * u - b * lambda + u;  // density in u: Jacobian lambda
      for (const auto& [k, e] : objects) value += logNegativeBinomial(k, kappa, lambda, e);
      terms[i] = value;
    }
    total += logSumExp(terms) + std::log(grid.step);
  }
  return total;
}

double betaLogMarginal(const std::map<int, CountList>& classes, const Grid& grid, double a,
                       double b, double kappa) {
  const double log_norm = -logBeta(a, b);
  double total = 0.0;
  std::vector<double> terms(grid.u.size());
  for (const auto& [cls, objects] : classes) {
    (void)cls;
    for (size_t i = 0; i < grid.u.size(); ++i) {
      const double q = sigmoid(grid.u[i]);
      const double log_q = -std::log1p(std::exp(-grid.u[i]));
      const double log_p = -std::log1p(std::exp(grid.u[i]));
      // density in the logit: Jacobian q (1 - q) = exp(log_q + log_p)
      double value = log_norm + a * log_q + b * log_p;
      for (const auto& [g, n] : objects) {
        value += logBetaBinomial(g, n, kappa * q, kappa * (1.0 - q));
      }
      terms[i] = value;
    }
    total += logSumExp(terms) + std::log(grid.step);
  }
  return total;
}

std::vector<double> softmax(const std::vector<double>& values) {
  const double normaliser = logSumExp(values);
  std::vector<double> weights(values.size());
  for (size_t i = 0; i < values.size(); ++i) weights[i] = std::exp(values[i] - normaliser);
  return weights;
}

}  // namespace

PersistencePrior::ObjectStats& PersistencePrior::stats(size_t object, int cls) {
  auto& value = objects_[object];
  value.cls = cls;
  return value;
}

void PersistencePrior::addExposure(size_t object, int cls, double seconds) {
  if (!(seconds >= 0.0) || !std::isfinite(seconds)) throw std::invalid_argument("Invalid exposure");
  stats(object, cls).exposure += seconds;
}

void PersistencePrior::addEvent(size_t object, int cls, double at_risk_seconds) {
  if (!(at_risk_seconds >= 0.0) || !std::isfinite(at_risk_seconds)) {
    throw std::invalid_argument("Invalid time at risk");
  }
  auto& value = stats(object, cls);
  value.events += 1.0;
  value.exposure += at_risk_seconds;
  // The hyperparameters are refitted when the committed outcomes change; exposure accrues
  // continuously and enters the posteriors directly.
  dirty_ = true;
}

void PersistencePrior::addGapOutcome(size_t object, int cls, bool changed) {
  auto& value = stats(object, cls);
  value.gap_judged += 1.0;
  if (changed) value.gap_changed += 1.0;
  dirty_ = true;
}

void PersistencePrior::fit() const {
  if (!dirty_) return;
  dirty_ = false;
  gamma_ = GammaFit{false, 0.5, 0.0, kMaxConcentration};
  beta_ = BetaFit{false, 0.5, 0.5, kMaxConcentration};

  // Rate hierarchy (5a): objects with exposure carry information.
  std::map<int, CountList> classes;
  double events = 0.0, exposure = 0.0;
  for (const auto& [id, s] : objects_) {
    (void)id;
    if (!(s.exposure > 0.0)) continue;
    classes[s.cls].push_back({s.events, s.exposure});
    events += s.events;
    exposure += s.exposure;
  }
  // Without a committed event the rate is not identified (the likelihood only pushes it to 0):
  // the Jeffreys start stays.
  if (events >= 1.0) {
    const auto grid = makeGrid(kMinLogRate, 0.0, kHazardNodes);
    const auto unpack = [](const std::vector<double>& x, double& mean, double& a, double& kappa) {
      mean = std::clamp(std::exp(x[0]), std::exp(kMinLogRate), 1.0);
      a = std::clamp(std::exp(x[1]), kMinConcentration, 1.0e4);
      kappa = std::clamp(std::exp(x[2]), kMinConcentration, kMaxConcentration);
    };
    const auto objective = [&](const std::vector<double>& x) {
      double mean, a, kappa;
      unpack(x, mean, a, kappa);
      const double value = gammaLogMarginal(classes, grid, a, a / mean, kappa);
      return std::isfinite(value) ? -value : std::numeric_limits<double>::max();
    };
    const auto best = nelderMead(objective, {std::log((events + 0.5) / exposure), 0.0, 0.0}, 1.0,
                                 kSearchIterations, kSearchTolerance);
    double mean, a, kappa;
    unpack(best.x, mean, a, kappa);
    gamma_ = GammaFit{true, a, a / mean, kappa};
  }

  // Gap hierarchy (5a): judged outcomes.
  std::map<int, CountList> gap_classes;
  double changed = 0.0, judged = 0.0;
  for (const auto& [id, s] : objects_) {
    (void)id;
    if (!(s.gap_judged > 0.0)) continue;
    gap_classes[s.cls].push_back({s.gap_changed, s.gap_judged});
    changed += s.gap_changed;
    judged += s.gap_judged;
  }
  // The probability is identified only when both outcomes occurred.
  if (changed >= 1.0 && judged - changed >= 1.0) {
    const auto grid = makeGrid(-kLogitBound, kLogitBound, kLogitNodes);
    const auto unpack = [](const std::vector<double>& x, double& mean, double& s, double& kappa) {
      mean = std::clamp(sigmoid(x[0]), 1.0e-6, 1.0 - 1.0e-6);
      s = std::clamp(std::exp(x[1]), kMinConcentration, 1.0e4);
      kappa = std::clamp(std::exp(x[2]), kMinConcentration, kMaxConcentration);
    };
    const auto objective = [&](const std::vector<double>& x) {
      double mean, s, kappa;
      unpack(x, mean, s, kappa);
      const double value = betaLogMarginal(gap_classes, grid, mean * s, (1.0 - mean) * s, kappa);
      return std::isfinite(value) ? -value : std::numeric_limits<double>::max();
    };
    const double start = (changed + 0.5) / (judged + 1.0);
    const auto best = nelderMead(objective, {std::log(start / (1.0 - start)), 0.0, 0.0}, 1.0,
                                 kSearchIterations, kSearchTolerance);
    double mean, s, kappa;
    unpack(best.x, mean, s, kappa);
    beta_ = BetaFit{true, mean * s, (1.0 - mean) * s, kappa};
  }
}

PersistencePrior::GammaFit PersistencePrior::hazardFit() const {
  fit();
  return gamma_;
}

PersistencePrior::BetaFit PersistencePrior::gapFit() const {
  fit();
  return beta_;
}

PersistencePrior::Hazard PersistencePrior::hazard(size_t object, int cls,
                                                  double extra_exposure) const {
  fit();
  const bool identified = gamma_.identified;
  CountList group;
  double own_k = 0.0, own_e = 0.0;
  bool any_exposure = extra_exposure > 0.0;
  bool own_seen = false;
  for (const auto& [id, s] : objects_) {
    const double exposure = s.exposure + (id == object ? std::max(0.0, extra_exposure) : 0.0);
    if (!(exposure > 0.0)) continue;
    any_exposure = true;
    if (id == object) {
      own_k = s.events;
      own_e = exposure;
      own_seen = true;
    }
    // Class-level pooling; where the rate is not identified nothing separates the classes.
    if (!identified || s.cls == cls) group.push_back({s.events, exposure});
  }
  if (!own_seen && extra_exposure > 0.0) {
    own_e = extra_exposure;
    group.push_back({0.0, extra_exposure});
  }
  if (!identified && !any_exposure) return {};
  const double kappa = gamma_.kappa;
  const auto grid = makeGrid(kMinLogRate, 0.0, kHazardNodes);
  std::vector<double> terms(grid.u.size()), mean(grid.u.size()), variance(grid.u.size());
  for (size_t i = 0; i < grid.u.size(); ++i) {
    const double u = grid.u[i], lambda = std::exp(u);
    // Prior density in u: Gamma(a, b) or the Jeffreys lambda^{-1/2}, each with the Jacobian lambda.
    double value = identified
        ? gamma_.a * std::log(gamma_.b) - std::lgamma(gamma_.a) + (gamma_.a - 1.0) * u -
              gamma_.b * lambda + u
        : 0.5 * u;
    for (const auto& [k, e] : group) value += logNegativeBinomial(k, kappa, lambda, e);
    terms[i] = value;
    const double rate = kappa / lambda + own_e;
    mean[i] = (kappa + own_k) / rate;
    variance[i] = (kappa + own_k) / (rate * rate);
  }
  const auto weights = softmax(terms);
  double m = 0.0, second = 0.0;
  for (size_t i = 0; i < weights.size(); ++i) {
    m += weights[i] * mean[i];
    second += weights[i] * (variance[i] + mean[i] * mean[i]);
  }
  const double v = std::max(second - m * m, kTinyRelative * m * m);
  if (!(m > 0.0) || !std::isfinite(v)) return {};
  return {m * m / v, m / v};
}

double PersistencePrior::changeProbability(size_t object, int cls, double seconds,
                                           double extra_exposure) const {
  if (!(seconds >= 0.0) || !std::isfinite(seconds)) throw std::invalid_argument("Invalid duration");
  const auto h = hazard(object, cls, extra_exposure);
  if (!h.valid()) return 0.0;
  return -std::expm1(-h.shape * std::log1p(seconds / h.rate));
}

double PersistencePrior::gapChangeProbability(size_t object, int cls) const {
  fit();
  const bool identified = beta_.identified;
  CountList group;
  double own_g = 0.0, own_n = 0.0;
  for (const auto& [id, s] : objects_) {
    if (!(s.gap_judged > 0.0)) continue;
    if (id == object) {
      own_g = s.gap_changed;
      own_n = s.gap_judged;
    }
    if (!identified || s.cls == cls) group.push_back({s.gap_changed, s.gap_judged});
  }
  const double kappa = beta_.kappa;
  const auto grid = makeGrid(-kLogitBound, kLogitBound, kLogitNodes);
  std::vector<double> terms(grid.u.size()), mean(grid.u.size());
  for (size_t i = 0; i < grid.u.size(); ++i) {
    const double q = sigmoid(grid.u[i]);
    const double log_q = -std::log1p(std::exp(-grid.u[i]));
    const double log_p = -std::log1p(std::exp(grid.u[i]));
    double value = -logBeta(beta_.a, beta_.b) + beta_.a * log_q + beta_.b * log_p;
    for (const auto& [g, n] : group) value += logBetaBinomial(g, n, kappa * q, kappa * (1.0 - q));
    terms[i] = value;
    mean[i] = (kappa * q + own_g) / (kappa + own_n);
  }
  const auto weights = softmax(terms);
  double result = 0.0;
  for (size_t i = 0; i < weights.size(); ++i) result += weights[i] * mean[i];
  return result;
}

nlohmann::json PersistencePrior::toJson() const {
  nlohmann::json records = nlohmann::json::array();
  for (const auto& [id, s] : objects_) {
    records.push_back(nlohmann::json::array({id, s.cls, s.events, s.exposure, s.gap_changed,
                                             s.gap_judged}));
  }
  return nlohmann::json{{"objects", std::move(records)}};
}

PersistencePrior PersistencePrior::fromJson(const nlohmann::json& value) {
  PersistencePrior prior;
  for (const auto& record : value.at("objects")) {
    if (!record.is_array() || record.size() != 6) {
      throw std::invalid_argument("Invalid persistence statistics record");
    }
    ObjectStats s;
    s.cls = record.at(1).get<int>();
    s.events = record.at(2).get<double>();
    s.exposure = record.at(3).get<double>();
    s.gap_changed = record.at(4).get<double>();
    s.gap_judged = record.at(5).get<double>();
    if (!(s.events >= 0) || !(s.exposure >= 0) || !(s.gap_changed >= 0) ||
        s.gap_changed > s.gap_judged || !std::isfinite(s.exposure) ||
        !std::isfinite(s.events) || !std::isfinite(s.gap_judged)) {
      throw std::invalid_argument("Inconsistent persistence statistics");
    }
    prior.objects_[record.at(0).get<size_t>()] = s;
  }
  return prior;
}

}  // namespace khronos::model
