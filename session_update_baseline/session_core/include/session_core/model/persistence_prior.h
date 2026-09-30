#pragma once

#include <cstddef>
#include <map>

#include <nlohmann/json.hpp>

namespace khronos::model {

/**
 * README principle 2, eqs. (5), (5a), (5c): the persistence prior Pi. A placement has one survival
 * time T_h; its only quantity is the state-change probability q, either over a duration inside a
 * session, q_h(Delta) = 1 - exp(-lambda_l Delta), or across one session gap, q^g_l. The semantic
 * class c is the grouping key of the hierarchy and nothing else.
 *
 *   lambda_l | lambda_c ~ Gamma(kappa_lambda, kappa_lambda / lambda_c),  lambda_c ~ Gamma(a_lambda, b_lambda)
 *   q^g_l | q^g_c ~ Beta(kappa_g q^g_c, kappa_g (1 - q^g_c)),             q^g_c ~ Beta(a_g, b_g)
 *
 * with the hyperparameters fitted by marginal likelihood (empirical Bayes) from the committed
 * events and exposures, and the Jeffreys cold start (lambda^{-1/2}; Beta(1/2, 1/2)) where the data
 * do not identify them. The predictive survival is the Lomax form (5c).
 */
class PersistencePrior {
 public:
  // Gamma(shape A, rate B) posterior of the hazard of one object: S(Delta) = (B / (B + Delta))^A.
  struct Hazard {
    double shape = 0.0, rate = 0.0;
    bool valid() const { return shape > 0.0 && rate > 0.0; }
  };

  /** The committed history of one object (README (5), assumption 5): events k, exposure E in
   * seconds, and the judged outcomes of session gaps (g changed of n judged). */
  struct ObjectStats {
    int cls = 0;
    double events = 0.0, exposure = 0.0, gap_changed = 0.0, gap_judged = 0.0;
  };

  /** Committed-in-place time of an object (seconds), counted once per source by the caller. */
  void addExposure(size_t object, int cls, double seconds);
  /** A committed change: one event, with the time at risk up to it as exposure. */
  void addEvent(size_t object, int cls, double at_risk_seconds);
  /** The judged outcome of one session gap for an inherited placement. */
  void addGapOutcome(size_t object, int cls, bool changed);

  /** (5c): the predictive hazard of the object; invalid while no exposure exists (E = 0).
   * `extra_exposure` is time the object has been observed in place but not yet credited (a
   * placement born in this session is in place from its first observation: Lambda_0 = 0), so the
   * Jeffreys posterior is normal as soon as a placement has been seen for any time. */
  Hazard hazard(size_t object, int cls, double extra_exposure = 0.0) const;
  /** (5): q = 1 - S(delta) over delta seconds inside a session. While the predictive is not
   * normalisable (no exposure anywhere, improper Jeffreys start) the round carries no prior
   * change probability and 0 is returned. */
  double changeProbability(size_t object, int cls, double seconds,
                           double extra_exposure = 0.0) const;
  /** (5): q^g, the posterior mean of the gap change probability of the object. */
  double gapChangeProbability(size_t object, int cls) const;

  // Fitted hyperparameters, for the record (README (5a)).
  struct GammaFit {
    bool identified = false;  // false: Jeffreys start, objects pooled
    double a = 0.5, b = 0.0, kappa = 0.0;
  };
  struct BetaFit {
    bool identified = false;
    double a = 0.5, b = 0.5, kappa = 0.0;
  };
  GammaFit hazardFit() const;
  BetaFit gapFit() const;

  const std::map<size_t, ObjectStats>& objects() const { return objects_; }

  nlohmann::json toJson() const;
  static PersistencePrior fromJson(const nlohmann::json& value);

 private:
  ObjectStats& stats(size_t object, int cls);
  void fit() const;

  std::map<size_t, ObjectStats> objects_;
  mutable bool dirty_ = true;
  mutable GammaFit gamma_;
  mutable BetaFit beta_;
};

}  // namespace khronos::model
