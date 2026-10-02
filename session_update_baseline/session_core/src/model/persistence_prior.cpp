#include "session_core/model/persistence_prior.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "session_core/model/model_math.h"

namespace khronos::model {
namespace {

// Computation budget of the marginal-likelihood integral and the simplex search: the probability
// grid spans logit -20 .. 20; the concentration bound is complete pooling.
constexpr size_t kLogitNodes = 401;
constexpr double kLogitBound = 20.0;
constexpr double kMaxConcentration = 1.0e6;
constexpr double kMinConcentration = 1.0e-2;
constexpr int kSearchIterations = 400;
constexpr double kSearchTolerance = 1.0e-9;

double sigmoid(double t) { return 1.0 / (1.0 + std::exp(-t)); }

using CountList = std::vector<std::pair<double, double>>;  // (g changed, n judged)

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

void PersistencePrior::addOutcome(size_t object, int cls, Gap gap, bool changed) {
  auto& value = objects_[object];
  value.cls = cls;
  const size_t g = static_cast<size_t>(gap);
  value.judged[g] += 1.0;
  if (changed) value.changed[g] += 1.0;
  dirty_ = true;
}

void PersistencePrior::refit() const {
  if (!dirty_) return;
  dirty_ = false;
  for (size_t g = 0; g < kNumGaps; ++g) {
    fits_[g] = BetaFit{false, 0.5, 0.5, kMaxConcentration};
    // The decided gaps of type g, by class (5a).
    std::map<int, CountList> classes;
    double changed = 0.0, judged = 0.0;
    for (const auto& [id, s] : objects_) {
      (void)id;
      if (!(s.judged[g] > 0.0)) continue;
      classes[s.cls].push_back({s.changed[g], s.judged[g]});
      changed += s.changed[g];
      judged += s.judged[g];
    }
    // The probability is identified only when both outcomes occurred.
    if (!(changed >= 1.0 && judged - changed >= 1.0)) continue;
    const auto grid = makeGrid(-kLogitBound, kLogitBound, kLogitNodes);
    const auto unpack = [](const std::vector<double>& x, double& mean, double& s, double& kappa) {
      mean = std::clamp(sigmoid(x[0]), 1.0e-6, 1.0 - 1.0e-6);
      s = std::clamp(std::exp(x[1]), kMinConcentration, 1.0e4);
      kappa = std::clamp(std::exp(x[2]), kMinConcentration, kMaxConcentration);
    };
    const auto objective = [&](const std::vector<double>& x) {
      double mean, s, kappa;
      unpack(x, mean, s, kappa);
      const double value = betaLogMarginal(classes, grid, mean * s, (1.0 - mean) * s, kappa);
      return std::isfinite(value) ? -value : std::numeric_limits<double>::max();
    };
    const double start = (changed + 0.5) / (judged + 1.0);
    const auto best = nelderMead(objective, {std::log(start / (1.0 - start)), 0.0, 0.0}, 1.0,
                                 kSearchIterations, kSearchTolerance);
    double mean, s, kappa;
    unpack(best.x, mean, s, kappa);
    fits_[g] = BetaFit{true, mean * s, (1.0 - mean) * s, kappa};
  }
}

PersistencePrior::BetaFit PersistencePrior::fit(Gap gap) const {
  refit();
  return fits_[static_cast<size_t>(gap)];
}

double PersistencePrior::changeProbability(size_t object, int cls, Gap gap) const {
  refit();
  const size_t g = static_cast<size_t>(gap);
  const auto& beta = fits_[g];
  CountList group;
  double own_changed = 0.0, own_judged = 0.0;
  for (const auto& [id, s] : objects_) {
    if (!(s.judged[g] > 0.0)) continue;
    if (id == object) {
      own_changed = s.changed[g];
      own_judged = s.judged[g];
    }
    // Where the hierarchy is not identified nothing separates the classes.
    if (!beta.identified || s.cls == cls) group.push_back({s.changed[g], s.judged[g]});
  }
  const auto grid = makeGrid(-kLogitBound, kLogitBound, kLogitNodes);
  std::vector<double> terms(grid.u.size()), mean(grid.u.size());
  for (size_t i = 0; i < grid.u.size(); ++i) {
    const double q = sigmoid(grid.u[i]);
    const double log_q = -std::log1p(std::exp(-grid.u[i]));
    const double log_p = -std::log1p(std::exp(grid.u[i]));
    double value = -logBeta(beta.a, beta.b) + beta.a * log_q + beta.b * log_p;
    for (const auto& [changed, judged] : group) {
      value += logBetaBinomial(changed, judged, beta.kappa * q, beta.kappa * (1.0 - q));
    }
    terms[i] = value;
    mean[i] = (beta.kappa * q + own_changed) / (beta.kappa + own_judged);
  }
  const auto weights = softmax(terms);
  double result = 0.0;
  for (size_t i = 0; i < weights.size(); ++i) result += weights[i] * mean[i];
  return result;
}

nlohmann::json PersistencePrior::toJson() const {
  nlohmann::json records = nlohmann::json::array();
  for (const auto& [id, s] : objects_) {
    records.push_back(nlohmann::json::array(
        {id, s.cls, s.changed[0], s.judged[0], s.changed[1], s.judged[1], s.changed[2], s.judged[2]}));
  }
  return nlohmann::json{{"objects", std::move(records)}};
}

PersistencePrior PersistencePrior::fromJson(const nlohmann::json& value) {
  PersistencePrior prior;
  for (const auto& record : value.at("objects")) {
    if (!record.is_array() || record.size() != 8) {
      throw std::invalid_argument("Invalid persistence statistics record");
    }
    ObjectStats s;
    s.cls = record.at(1).get<int>();
    for (size_t g = 0; g < kNumGaps; ++g) {
      s.changed[g] = record.at(2 + 2 * g).get<double>();
      s.judged[g] = record.at(3 + 2 * g).get<double>();
      if (!(s.changed[g] >= 0) || s.changed[g] > s.judged[g] || !std::isfinite(s.judged[g])) {
        throw std::invalid_argument("Inconsistent persistence statistics");
      }
    }
    prior.objects_[record.at(0).get<size_t>()] = s;
  }
  return prior;
}

}  // namespace khronos::model
