#pragma once

#include <array>
#include <cstddef>
#include <map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace khronos::model {

/**
 * README principle 6, eqs. (7), (7h): the hierarchical beta-binomial aggregation of one binary
 * verdict type. Each round of an object has n trials of which k succeed; the round fraction of an
 * object follows Beta(m phi, (1-m) phi), the object means Beta(mu kappa, (1-mu) kappa). The
 * parameters are estimated from the stored rounds by the moment equations of the README
 * (within-object scatter for phi, between-object scatter for kappa).
 */
class HierarchicalBeta {
 public:
  using Observation = std::pair<double, double>;  // (n trials, k successes)

  struct Fit {
    bool available = false;
    double mu = 0.5, phi = 0.0, kappa = 0.0;
  };

  /** Add a round; `capacity` > 0 keeps only the newest rounds of the object (computation budget
   * for streams of frames). */
  void add(size_t object, double n, double k, size_t capacity = 0) {
    if (!(n > 0.0)) return;
    auto& list = rounds_[object];
    list.push_back({n, k});
    if (capacity > 0 && list.size() > capacity) list.erase(list.begin());
  }
  const std::map<size_t, std::vector<Observation>>& rounds() const { return rounds_; }

  /** The moment estimates; unavailable until an object has two rounds. */
  Fit fit() const;
  /** (a, b) of a new round of `object`: empirical-Bayes partial pooling of its mean (7h) and the
   * moment match of the predictive variance. False where the fit is unavailable. */
  bool predict(const Fit& fit, size_t object, double& a, double& b) const;

  /** rho_1: the lag-1 correlation of the adjacent-round logit fractions within objects (7);
   * false while fewer than two adjacent pairs exist. */
  bool lagOneCorrelation(double& rho) const;

  nlohmann::json toJson() const;
  void fromJson(const nlohmann::json& value);

 private:
  std::map<size_t, std::vector<Observation>> rounds_;
};

/** (7h): the information one round of n trials gives about the object mean. */
double effectiveRoundSize(double n, double phi);
/** Jeffreys posterior mean of a proportion k of n. */
double smoothedProportion(double k, double n);
/** The concentration s of Beta(m s, (1 - m) s) for which the pooled squared deviation of the
 * fractions equals its beta-binomial expectation; entries are (fraction, trials, multiplicity). */
double concentrationFromMoments(const std::vector<std::array<double, 3>>& entries, double m);

}  // namespace khronos::model
