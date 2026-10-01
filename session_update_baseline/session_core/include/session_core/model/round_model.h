#pragma once

#include <cstddef>
#include <map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "session_core/model/hierarchical_beta.h"

namespace khronos::model {

/**
 * README principle 6, eqs. (7), (7h), (7s): the round likelihood ratio of one placement as the
 * hierarchical beta-binomial aggregation of the element verdicts.
 *
 * Elements of a placement in place are see-through in a round with probability theta_e ~
 * Beta(m psi, (1-m) psi); the stable elements of a round show a see-through fraction f ~
 * Beta(m phi, (1-m) phi); the object mean m ~ Beta(mu kappa, (1-mu) kappa). The share of foreign
 * labels among labelled hits follows the same family (c, d), and a placement that has ended shows
 * see-through and foreign fractions from the ended distributions (a0, b0), (c0, d0). The
 * parameters (mu, phi, kappa, psi, rho_1, c, d, a0, b0, c0, d0) are estimated online from the
 * rounds committed in place (and from rounds after a committed end) by the moment equations of the
 * README; only committed rounds enter, each source once.
 */
class RoundModel {
 public:
  /** One round of the stable elements of a placement: n verdicts of which f see through; among
   * the n_labeled hits that carry a physical label, g are foreign. */
  struct Counts {
    double n = 0.0, f = 0.0, n_labeled = 0.0, g = 0.0;
  };

  /** ln LR of (7) for one round of `object`: w (ln L0 - ln L1). Neutral (0) where no in-place
   * statistic exists for a channel ("L1 := L0"). */
  double logLikelihoodRatio(size_t object, const Counts& round) const;

  /** A round committed in place (5e) enters the object's in-place statistics. */
  void addInPlaceRound(size_t object, const Counts& round);
  /** A round observed after a committed end enters the ended distributions. */
  void addEndedRound(const Counts& round);
  /** Element histories: an element changes from `before` (see-through j, committed rounds N;
   * N = 0 when new) to `after`. Only the pooled histogram (j, N) is kept, for psi. */
  void moveElementHistory(double j_before, double n_before, double j_after, double n_after);

  /** (7s): the prior Beta(m psi, (1-m) psi) of theta_e of an element of `object`; without an
   * estimate of (m, psi) the Jeffreys Beta(1/2, 1/2). */
  struct ElementPrior {
    double a = 0.5, b = 0.5;
  };
  ElementPrior elementPrior(size_t object) const;
  /** (7s): an element with construction hits k0, committed-round hits k and see-throughs j is
   * stable when Pr(theta_e < 1/2 | history) = I_{1/2}(a + j, b + k0 + k) >= 1 - alpha. */
  static bool elementStable(const ElementPrior& prior, double k0, double k, double j);

  /** README principle 9, eq. (7) with frames as rounds: ln(L_ended / L_present) of n frames, f of
   * them see-through, of an element of `object` whose history is k0 construction hits, k committed
   * hits and j committed see-throughs, discounted by the autocorrelation exponent `w`. Neutral (0)
   * where no in-place statistic exists. */
  double elementLogRatio(size_t object, double n, double f, double k0, double k, double j,
                         double w) const;
  /** README principle 9, (7s): ln BB(f | n; a + j, b + k0 + k), the frames of an element in place
   * under its own theta_e posterior. False (value untouched) where no in-place statistic exists. */
  bool elementLogPresent(size_t object, double n, double f, double k0, double k, double j,
                         double& value) const;
  /** The ended distribution (a0, b0) of the rounds after a committed end; Beta(1,1) at cold start. */
  void endedDistribution(double& a0, double& b0) const;
  /** The exponent w = (1 - rho~)/(1 + rho~) of (7) for the first-order correlation rho projected on
   * [0, 1): rho~ = max(0, rho), so w lies in (0, 1]. */
  static double autocorrelationExponent(double rho);

  /** Estimated parameters, for the record (README table 5.1). */
  struct Parameters {
    bool see_through_available = false, foreign_available = false, psi_available = false;
    double mu = 0.0, phi = 0.0, kappa = 0.0, psi = 0.0, rho1 = 0.0;
    double c_mu = 0.0, c_phi = 0.0, c_kappa = 0.0;
    double a0 = 1.0, b0 = 1.0, c0 = 1.0, d0 = 1.0;
  };
  Parameters parameters() const;

  nlohmann::json toJson() const;
  static RoundModel fromJson(const nlohmann::json& value);

 private:
  using Observation = HierarchicalBeta::Observation;

  struct Ended {
    std::vector<Observation> see_through, foreign;
  };

  // Estimates are functions of the stored statistics, refreshed after a change.
  struct Estimates {
    HierarchicalBeta::Fit see, foreign;
    bool psi_available = false, rho_available = false;
    double psi = 0.0, rho1 = 0.0;
    double a0 = 1.0, b0 = 1.0, c0 = 1.0, d0 = 1.0;
  };
  const Estimates& estimates() const;

  HierarchicalBeta see_, foreign_;
  Ended ended_;
  std::map<std::pair<double, double>, double> elements_;  // (j, N) -> number of elements
  mutable bool dirty_ = true;
  mutable Estimates estimates_;
};

}  // namespace khronos::model
