#include "session_core/model/round_model.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

#include "session_core/model/model_math.h"

namespace khronos::model {
namespace {

constexpr double kNormalConsistency = 1.4826;  // 1 / Phi^{-1}(3/4), README table 5.1
// (7h): "three pseudo-looks" and the concentration floor 2 of Beta(1, 1).
constexpr double kPseudoLooks = 3.0;
constexpr double kMinConcentration = 2.0;
// A look must have three own looks before it counts (7h); a population needs the same of each object.
constexpr size_t kMinLooks = 3;
// Numerical guards only: a mean strictly inside (0, 1) and a finite concentration.
constexpr double kMeanGuard = 1.0e-9;
constexpr double kMaxConcentration = 1.0e6;
// Computation budget of the decisive-sample search.
constexpr size_t kMaxDecisiveSamples = 100000;

double median(std::vector<double> values) {
  std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
  double upper = values[values.size() / 2];
  if (values.size() % 2 == 1) return upper;
  const double lower = *std::max_element(values.begin(), values.begin() + values.size() / 2);
  return 0.5 * (lower + upper);
}

// (1.4826 MAD)^2 of the values: the robust variance.
double robustVariance(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  const double centre = median(values);
  std::vector<double> deviations;
  deviations.reserve(values.size());
  for (const double v : values) deviations.push_back(std::abs(v - centre));
  const double spread = kNormalConsistency * median(std::move(deviations));
  return spread * spread;
}

double mean(const std::vector<double>& values) {
  double sum = 0.0;
  for (const double v : values) sum += v;
  return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
}

}  // namespace

RoundModel::Population RoundModel::population() const {
  Population result;
  std::vector<double> means;
  for (const auto& [object, looks] : looks_) {
    (void)object;
    if (looks.size() >= kMinLooks) means.push_back(mean(looks));
  }
  result.objects = means.size();
  // A robust scatter between objects needs at least two of them.
  if (means.size() < 2) return result;
  result.available = true;
  result.mu = mean(means);
  result.scatter = robustVariance(means);
  return result;
}

RoundModel::InPlace RoundModel::shrunk(const std::vector<double>& looks, double cold_mean) const {
  const auto pop = population();
  const double mu = pop.available ? pop.mu : cold_mean;
  // Cold start: concentration 2, v_0 = m (1 - m) / (c + 1).
  const double v0 = pop.available ? pop.scatter
                                  : cold_mean * (1.0 - cold_mean) / (kMinConcentration + 1.0);
  // (7h): fewer than three own looks leave the population alone.
  const double n = looks.size() >= kMinLooks ? static_cast<double>(looks.size()) : 0.0;
  const double fbar = n > 0.0 ? mean(looks) : 0.0;
  const double vhat = n > 0.0 ? robustVariance(looks) : 0.0;
  const double m = (n * fbar + kPseudoLooks * mu) / (n + kPseudoLooks);
  const double v = std::max(v0, (n * (vhat + (fbar - m) * (fbar - m)) +
                                 kPseudoLooks * (v0 + (mu - m) * (mu - m))) / (n + kPseudoLooks));
  InPlace result;
  result.m = std::clamp(m, kMeanGuard, 1.0 - kMeanGuard);
  const double concentration = v > 0.0 ? result.m * (1.0 - result.m) / v - 1.0 : kMaxConcentration;
  result.c = std::clamp(concentration, kMinConcentration, kMaxConcentration);
  return result;
}

RoundModel::InPlace RoundModel::inPlace(size_t object, double cold_mean) const {
  const auto found = looks_.find(object);
  static const std::vector<double> none;
  return shrunk(found == looks_.end() ? none : found->second, cold_mean);
}

RoundModel::InPlace RoundModel::classInPlace(int cls, double cold_mean) const {
  std::vector<double> pooled;
  for (const auto& [object, looks] : looks_) {
    const auto c = class_of_.find(object);
    if (c != class_of_.end() && c->second == cls) pooled.insert(pooled.end(), looks.begin(), looks.end());
  }
  return shrunk(pooled, cold_mean);
}

double RoundModel::logLikelihoodRatio(const InPlace& in_place, double n, double f) {
  if (!(n > 0.0) || f < 0.0 || f > n) return 0.0;
  const double f_in_place = std::min(n, std::max(f, std::floor(in_place.m * n + 0.5)));
  const double log_l1 = logBetaBinomial(f_in_place, n, in_place.m * in_place.c,
                                        (1.0 - in_place.m) * in_place.c);
  return -std::log(n + 1.0) - log_l1;
}

size_t RoundModel::decisiveSamples(const InPlace& in_place) {
  const double threshold = std::log(closeOdds());
  for (size_t n = 1; n <= kMaxDecisiveSamples; ++n) {
    if (logLikelihoodRatio(in_place, static_cast<double>(n), static_cast<double>(n)) >= threshold) return n;
  }
  return kMaxDecisiveSamples;
}

void RoundModel::addInPlaceLook(size_t object, int cls, double fraction) {
  if (!(fraction >= 0.0) || fraction > 1.0) throw std::invalid_argument("Invalid in-place look");
  looks_[object].push_back(fraction);
  class_of_[object] = cls;
}

void RoundModel::addLabels(double foreign, double labelled) {
  if (!(foreign >= 0.0) || foreign > labelled) throw std::invalid_argument("Invalid label counts");
  foreign_ += foreign;
  labelled_ += labelled;
}

double RoundModel::foreignShare() const { return (foreign_ + 0.5) / (labelled_ + 1.0); }

size_t RoundModel::minHits() const {
  const double eps = foreignShare();
  size_t k = 1;
  double power = eps;
  while (power > kAlpha && k < 1024) {
    power *= eps;
    ++k;
  }
  return k;
}

size_t RoundModel::numInPlaceLooks(size_t object) const {
  const auto found = looks_.find(object);
  return found == looks_.end() ? 0 : found->second.size();
}

nlohmann::json RoundModel::toJson() const {
  nlohmann::json looks = nlohmann::json::array();
  for (const auto& [object, fractions] : looks_) {
    const auto c = class_of_.find(object);
    looks.push_back(nlohmann::json::array({object, c == class_of_.end() ? -1 : c->second, fractions}));
  }
  return nlohmann::json{{"looks", std::move(looks)}, {"foreign", foreign_}, {"labelled", labelled_}};
}

RoundModel RoundModel::fromJson(const nlohmann::json& value) {
  RoundModel model;
  for (const auto& record : value.at("looks")) {
    if (!record.is_array() || record.size() != 3) throw std::invalid_argument("Invalid look record");
    const auto object = record.at(0).get<size_t>();
    model.class_of_[object] = record.at(1).get<int>();
    auto& fractions = model.looks_[object];
    fractions = record.at(2).get<std::vector<double>>();
    for (const double f : fractions) {
      if (!(f >= 0.0) || f > 1.0) throw std::invalid_argument("Invalid look fraction");
    }
  }
  model.foreign_ = value.at("foreign").get<double>();
  model.labelled_ = value.at("labelled").get<double>();
  if (!(model.foreign_ >= 0.0) || model.foreign_ > model.labelled_) {
    throw std::invalid_argument("Invalid label statistics");
  }
  return model;
}

}  // namespace khronos::model
