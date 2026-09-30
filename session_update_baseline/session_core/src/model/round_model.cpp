#include "session_core/model/round_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "session_core/model/model_math.h"

namespace khronos::model {
namespace {

constexpr double kCorrelationBound = 1.0e-6;
using Entry = std::array<double, 3>;  // fraction x, trials N, multiplicity c

}  // namespace

const RoundModel::Estimates& RoundModel::estimates() const {
  if (!dirty_) return estimates_;
  dirty_ = false;
  estimates_ = Estimates{};
  estimates_.see = see_.fit();
  estimates_.foreign = foreign_.fit();

  // rho_1: the lag-1 correlation of the adjacent-round logit fractions within objects.
  {
    double rho;
    if (see_.lagOneCorrelation(rho)) {
      estimates_.rho_available = true;
      estimates_.rho1 = std::clamp(rho, -1.0 + kCorrelationBound, 1.0 - kCorrelationBound);
    }
  }

  // psi: the concentration of the element see-through probabilities from the element histories.
  {
    double elements = 0.0, see_through = 0.0, trials = 0.0;
    std::vector<Entry> entries;
    for (const auto& [history, count] : elements_) {
      if (!(history.second > 0.0)) continue;
      elements += count;
      see_through += count * history.first;
      trials += count * history.second;
      entries.push_back({history.first / history.second, history.second, count});
    }
    if (elements >= 2.0) {
      estimates_.psi_available = true;
      estimates_.psi = concentrationFromMoments(entries, smoothedProportion(see_through, trials));
    }
  }

  // Ended distributions: the maximum-entropy Beta(1, 1) until two ended rounds exist.
  const auto ended = [](const std::vector<Observation>& rounds, double& a, double& b) {
    std::vector<Entry> entries;
    double total_k = 0.0, total_n = 0.0;
    for (const auto& [n, k] : rounds) {
      if (!(n > 0.0)) continue;
      entries.push_back({k / n, n, 1.0});
      total_k += k;
      total_n += n;
    }
    if (entries.size() < 2) return;
    const double m = smoothedProportion(total_k, total_n);
    const double s = concentrationFromMoments(entries, m);
    a = m * s;
    b = (1.0 - m) * s;
  };
  ended(ended_.see_through, estimates_.a0, estimates_.b0);
  ended(ended_.foreign, estimates_.c0, estimates_.d0);
  return estimates_;
}

double RoundModel::logLikelihoodRatio(size_t object, const Counts& round) const {
  if (!(round.n >= 0.0) || round.f < 0.0 || round.f > round.n || round.g < 0.0 ||
      round.g > round.n_labeled) {
    throw std::invalid_argument("Invalid round counts");
  }
  const auto& e = estimates();
  double log_in_place = 0.0, log_ended = 0.0;
  double a, b;
  if (round.n > 0.0 && see_.predict(e.see, object, a, b)) {
    log_in_place += logBetaBinomial(round.f, round.n, a, b);
    log_ended += logBetaBinomial(round.f, round.n, e.a0, e.b0);
  }
  if (round.n_labeled > 0.0 && foreign_.predict(e.foreign, object, a, b)) {
    log_in_place += logBetaBinomial(round.g, round.n_labeled, a, b);
    log_ended += logBetaBinomial(round.g, round.n_labeled, e.c0, e.d0);
  }
  // (7): the exponent discounts the autocorrelation of adjacent rounds.
  const double w = e.rho_available ? autocorrelationExponent(e.rho1) : 1.0;
  return w * (log_ended - log_in_place);
}

void RoundModel::addInPlaceRound(size_t object, const Counts& round) {
  if (round.n > 0.0) see_.add(object, round.n, round.f);
  if (round.n_labeled > 0.0) foreign_.add(object, round.n_labeled, round.g);
  dirty_ = true;
}

void RoundModel::addEndedRound(const Counts& round) {
  if (round.n > 0.0) ended_.see_through.push_back({round.n, round.f});
  if (round.n_labeled > 0.0) ended_.foreign.push_back({round.n_labeled, round.g});
  dirty_ = true;
}

void RoundModel::moveElementHistory(double j_before, double n_before, double j_after,
                                    double n_after) {
  if (n_before > 0.0) {
    // A history restored from another record may not be registered; it then leaves nothing to move.
    const auto found = elements_.find({j_before, n_before});
    if (found != elements_.end()) {
      if (found->second <= 1.0) elements_.erase(found);
      else found->second -= 1.0;
    }
  }
  if (n_after > 0.0) elements_[{j_after, n_after}] += 1.0;
  dirty_ = true;
}

RoundModel::ElementPrior RoundModel::elementPrior(size_t object) const {
  const auto& e = estimates();
  double a, b;
  if (!e.psi_available || !see_.predict(e.see, object, a, b)) return {};
  const double m = a / (a + b);
  return {m * e.psi, (1.0 - m) * e.psi};
}

bool RoundModel::elementStable(const ElementPrior& prior, double k0, double k, double j) {
  return betaCdfHalf(prior.a + j, prior.b + k0 + k) >= 1.0 - kAlpha;
}

double RoundModel::autocorrelationExponent(double rho) {
  const double bounded = std::clamp(rho, -1.0 + kCorrelationBound, 1.0 - kCorrelationBound);
  // The first-order effective sample size of correlated observations; an estimated negative
  // correlation never counts a round for more than one independent observation.
  return std::min(1.0, (1.0 - bounded) / (1.0 + bounded));
}

double RoundModel::elementLogRatio(size_t object, double n, double f, double k0, double k, double j,
                                   double w) const {
  if (!(n > 0.0) || f < 0.0 || f > n) return 0.0;
  const auto& e = estimates();
  if (!e.see.available) return 0.0;
  const auto prior = elementPrior(object);
  const double log_present = logBetaBinomial(f, n, prior.a + j, prior.b + k0 + k);
  const double log_ended = logBetaBinomial(f, n, e.a0, e.b0);
  return w * (log_ended - log_present);
}

RoundModel::Parameters RoundModel::parameters() const {
  const auto& e = estimates();
  Parameters p;
  p.see_through_available = e.see.available;
  p.foreign_available = e.foreign.available;
  p.psi_available = e.psi_available;
  p.mu = e.see.mu;
  p.phi = e.see.phi;
  p.kappa = e.see.kappa;
  p.psi = e.psi;
  p.rho1 = e.rho1;
  p.c_mu = e.foreign.mu;
  p.c_phi = e.foreign.phi;
  p.c_kappa = e.foreign.kappa;
  p.a0 = e.a0;
  p.b0 = e.b0;
  p.c0 = e.c0;
  p.d0 = e.d0;
  return p;
}

nlohmann::json RoundModel::toJson() const {
  const auto pairs = [](const std::vector<Observation>& rounds) {
    nlohmann::json list = nlohmann::json::array();
    for (const auto& [n, k] : rounds) list.push_back(nlohmann::json::array({n, k}));
    return list;
  };
  nlohmann::json elements = nlohmann::json::array();
  for (const auto& [history, count] : elements_) {
    elements.push_back(nlohmann::json::array({history.first, history.second, count}));
  }
  return nlohmann::json{{"see", see_.toJson()},
                        {"foreign", foreign_.toJson()},
                        {"ended_see", pairs(ended_.see_through)},
                        {"ended_foreign", pairs(ended_.foreign)},
                        {"elements", std::move(elements)}};
}

RoundModel RoundModel::fromJson(const nlohmann::json& value) {
  RoundModel model;
  model.see_.fromJson(value.at("see"));
  model.foreign_.fromJson(value.at("foreign"));
  const auto read = [](const nlohmann::json& list, std::vector<Observation>& out) {
    for (const auto& o : list) {
      const double n = o.at(0).get<double>(), k = o.at(1).get<double>();
      if (!(n >= 0.0) || !(k >= 0.0) || k > n) throw std::invalid_argument("Invalid ended round");
      out.push_back({n, k});
    }
  };
  read(value.at("ended_see"), model.ended_.see_through);
  read(value.at("ended_foreign"), model.ended_.foreign);
  for (const auto& item : value.at("elements")) {
    const double j = item.at(0).get<double>(), n = item.at(1).get<double>(),
                 count = item.at(2).get<double>();
    if (!(j >= 0.0) || !(n > 0.0) || j > n || !(count >= 1.0)) {
      throw std::invalid_argument("Invalid element history record");
    }
    model.elements_[{j, n}] = count;
  }
  return model;
}

}  // namespace khronos::model
