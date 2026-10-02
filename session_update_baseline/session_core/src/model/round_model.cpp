#include "session_core/model/round_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "session_core/model/model_math.h"

namespace khronos::model {
namespace {

// README table 5.1, mathematical constants: c_0 = kappa_0 = 5, so that the predictive distribution
// at the cold start, 1/(c_0+1) + 1/(kappa_0+1) = 1/3, has the pseudo-sample count 2 of Beta(1, 1).
constexpr double kPriorConcentration = 5.0;
// The prior counts with the information of one look (unit-information prior, Kass-Wasserman).
constexpr double kPriorInformation = 1.0;

// (7h): the independent-sample equivalent of a look of n trials under the look-to-look
// concentration c: a look is worth at most c + 1 independent samples.
double omegaOf(double n, double within) { return n / (1.0 + (n - 1.0) / (within + 1.0)); }

}  // namespace

RoundModel::InPlace RoundModel::predictive(const Looks& own, double cold_mean) const {
  // ---- c: the look-to-look concentration, pooled over all objects (Kleinman's moment estimator
  // with the binomial sampling variance removed). Objects whose looks are all 0 carry no
  // information about it (their variance m(1-m) is 0), so zero-inflated data stays defined.
  double df = 0.0, ratio_sum = 0.0, extra = 0.0;
  size_t n_looks = 0;
  double n_total = 0.0;
  for (const auto& [object, looks] : looks_) {
    (void)object;
    n_looks += looks.size();
    double total = 0.0, failures = 0.0, squares = 0.0;
    for (const auto& [n, f] : looks) {
      total += n;
      failures += f;
      squares += n * n;
    }
    n_total += total;
    if (looks.size() < 2 || !(total > 0.0)) continue;
    const double bar = failures / total, s = bar * (1.0 - bar);
    if (!(s > 0.0)) continue;
    double q = 0.0;
    for (const auto& [n, f] : looks) q += n * (f / n - bar) * (f / n - bar);
    const double j = static_cast<double>(looks.size());
    df += j - 1.0;
    ratio_sum += q / s;
    extra += total - squares / total - (j - 1.0);
  }
  const double rho_prior = 1.0 / (kPriorConcentration + 1.0);
  const double rho_data = extra > 0.0 ? std::clamp((ratio_sum - df) / extra, 0.0, 1.0) : 0.0;
  const double information = extra > 0.0 ? df : 0.0;
  const double rho = (information * rho_data + kPriorInformation * rho_prior) /
                     (information + kPriorInformation);
  const double within = 1.0 / rho - 1.0;

  // ---- mu: the total mean of the looks, weighted by their independent-sample equivalents, with the
  // prior centre m_0 counted as one look of the average size.
  double weighted = 0.0, weights = 0.0;
  for (const auto& [object, looks] : looks_) {
    (void)object;
    for (const auto& [n, f] : looks) {
      const double w = omegaOf(n, within);
      weighted += w * f / n;
      weights += w;
    }
  }
  const double mean_size = n_looks > 0 ? n_total / static_cast<double>(n_looks) : 0.0;
  const double prior_weight = n_looks > 0 ? kPriorInformation * omegaOf(mean_size, within) : 0.0;
  const double mu = (prior_weight + weights) > 0.0
      ? (prior_weight * cold_mean + weighted) / (prior_weight + weights) : cold_mean;
  const double mu_final = mu;

  // ---- kappa: the scatter of the object means beyond their sampling variance.
  double objects = 0.0, between = 0.0, sampling = 0.0;
  for (const auto& [object, looks] : looks_) {
    (void)object;
    double w_sum = 0.0, f_sum = 0.0, n_sum = 0.0;
    for (const auto& [n, f] : looks) {
      w_sum += omegaOf(n, within);
      f_sum += f;
      n_sum += n;
    }
    if (!(n_sum > 0.0)) continue;
    objects += 1.0;
    const double bar = f_sum / n_sum;
    between += (bar - mu_final) * (bar - mu_final);
    sampling += mu_final * (1.0 - mu_final) / w_sum;
  }
  const double spread = mu_final * (1.0 - mu_final);
  const double tau_prior = 1.0 / (kPriorConcentration + 1.0);
  double tau_data = 0.0, tau_information = 0.0;
  if (objects >= 2.0 && spread > 0.0) {
    tau_data = std::clamp((between / (objects - 1.0) - sampling / objects) / spread, 0.0, 1.0);
    tau_information = objects - 1.0;
  }
  const double tau = (tau_information * tau_data + kPriorInformation * tau_prior) /
                     (tau_information + kPriorInformation);
  const double kappa = 1.0 / tau - 1.0;

  // ---- the object's own looks: posterior mean of its normal share and the predictive concentration.
  double own_weighted = 0.0, own_weights = 0.0;
  for (const auto& [n, f] : own) {
    const double w = omegaOf(n, within);
    own_weighted += w * f / n;
    own_weights += w;
  }
  InPlace result;
  result.mu = mu_final;
  result.kappa = kappa;
  result.within = within;
  result.omega_sum = own_weights;
  result.looks = own.size();
  result.m = (kappa * mu_final + own_weighted) / (kappa + own_weights);
  // Guard of the beta-binomial domain: a share strictly inside (0, 1).
  const double tiny = std::numeric_limits<double>::epsilon();
  result.m = std::clamp(result.m, tiny, 1.0 - tiny);
  result.c = 1.0 / (1.0 / (within + 1.0) + 1.0 / (kappa + own_weights + 1.0)) - 1.0;
  return result;
}

RoundModel::InPlace RoundModel::inPlace(size_t object, double cold_mean) const {
  static const Looks none;
  const auto found = looks_.find(object);
  return predictive(found == looks_.end() ? none : found->second, cold_mean);
}

RoundModel::InPlace RoundModel::classInPlace(int cls, double cold_mean) const {
  Looks pooled;
  for (const auto& [object, looks] : looks_) {
    const auto c = class_of_.find(object);
    if (c != class_of_.end() && c->second == cls) pooled.insert(pooled.end(), looks.begin(), looks.end());
  }
  return predictive(pooled, cold_mean);
}

double RoundModel::logLikelihoodRatio(const InPlace& in_place, double n, double f) {
  if (!(n > 0.0) || f < 0.0 || f > n) return 0.0;
  const double f_in_place = std::min(n, std::max(f, std::floor(in_place.m * n + 0.5)));
  const double log_l1 = logBetaBinomial(f_in_place, n, in_place.m * in_place.c,
                                        (1.0 - in_place.m) * in_place.c);
  return -std::log(n + 1.0) - log_l1;
}

RoundModel::Judgement RoundModel::judge(const InPlace& in_place, double n, double f) {
  Judgement judgement;
  judgement.in_place = in_place;
  judgement.ln_lr = logLikelihoodRatio(in_place, n, f);
  judgement.commitment = decideLog(judgement.ln_lr);
  return judgement;
}

size_t RoundModel::decisiveSamples(const InPlace& in_place) {
  const double threshold = std::log(closeOdds());
  for (size_t n = 1; n <= kElementBudget; ++n) {
    if (logLikelihoodRatio(in_place, static_cast<double>(n), static_cast<double>(n)) >= threshold) return n;
  }
  return kElementBudget + 1;
}

void RoundModel::addInPlaceLook(size_t object, int cls, double n, double f) {
  if (!(n > 0.0) || !(f >= 0.0) || f > n) throw std::invalid_argument("Invalid in-place look");
  looks_[object].emplace_back(n, f);
  class_of_[object] = cls;
}

void RoundModel::addLabels(double foreign, double labelled) {
  if (!(foreign >= 0.0) || foreign > labelled) throw std::invalid_argument("Invalid label counts");
  foreign_ += foreign;
  labelled_ += labelled;
}

// Jeffreys smoothing: (foreign + 1/2) / (labelled + 1).
double RoundModel::foreignShare() const { return (foreign_ + 0.5) / (labelled_ + 1.0); }

double RoundModel::logForeignShare() const { return std::log(foreignShare()); }

size_t RoundModel::minHits() const {
  const double eps = foreignShare();
  size_t k = 1;
  double power = eps;
  // eps < 1 always, so the loop ends; k is the least count with eps^k <= alpha.
  while (power > kAlpha) {
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
  for (const auto& [object, list] : looks_) {
    const auto c = class_of_.find(object);
    nlohmann::json pairs = nlohmann::json::array();
    for (const auto& [n, f] : list) pairs.push_back(nlohmann::json::array({n, f}));
    looks.push_back(nlohmann::json::array({object, c == class_of_.end() ? -1 : c->second, pairs}));
  }
  return nlohmann::json{{"looks", std::move(looks)}, {"foreign", foreign_}, {"labelled", labelled_}};
}

RoundModel RoundModel::fromJson(const nlohmann::json& value) {
  RoundModel model;
  for (const auto& record : value.at("looks")) {
    if (!record.is_array() || record.size() != 3) throw std::invalid_argument("Invalid look record");
    const auto object = record.at(0).get<size_t>();
    model.class_of_[object] = record.at(1).get<int>();
    auto& list = model.looks_[object];
    for (const auto& pair : record.at(2)) {
      const double n = pair.at(0).get<double>(), f = pair.at(1).get<double>();
      if (!(n > 0.0) || !(f >= 0.0) || f > n) throw std::invalid_argument("Invalid look");
      list.emplace_back(n, f);
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
