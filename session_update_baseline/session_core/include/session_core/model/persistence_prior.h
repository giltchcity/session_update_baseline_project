#pragma once

#include <array>
#include <cstddef>
#include <map>
#include <vector>

#include <nlohmann/json.hpp>

namespace khronos::model {

/** README principle 2, assumption (2): the way a gap between two looks was observed. */
enum class Gap : int {
  kContinuous = 0,     // adjacent rounds both saw the placement
  kWithinSession = 1,  // rounds without a look lie between
  kSession = 2,        // the gap crosses a session boundary
};
constexpr size_t kNumGaps = 3;

/**
 * README principle 2, eqs. (5), (5a): the persistence prior Pi. Nothing changes while nothing is
 * seen; the only quantity is q_{c,g}, the probability that a placement has changed after a gap of
 * type g when it is next looked at. It is used only where the observation is silent (principle 13);
 * the semantic class c is the grouping key of the hierarchy and nothing else.
 *
 *   q_{l,g} | q_{c,g} ~ Beta(kappa_g q_{c,g}, kappa_g (1 - q_{c,g})),   q_{c,g} ~ Beta(a_g, b_g),
 *   q_hat_{l,g} = (kappa_g q_hat_{c,g} + sum_j pi_j) / (kappa_g + n_{l,g}),
 *
 * with pi_j in {0, 1} the outcome of the j-th gap of type g of object l, decided by the observation
 * after it (seen in place: not changed; ended with the end interval (5t) in the gap: changed), and
 * the hyperparameters (a_g, b_g, kappa_g) fitted by marginal likelihood (empirical Bayes). The cold
 * start is the Jeffreys Beta(1/2, 1/2).
 */
class PersistencePrior {
 public:
  /** The decided gaps of one object, per type. */
  struct ObjectStats {
    int cls = 0;
    std::array<double, kNumGaps> changed{}, judged{};
  };

  /** The outcome of one gap of type `gap` of `object`, decided by the observation after it. */
  void addOutcome(size_t object, int cls, Gap gap, bool changed);

  /** (5): q_hat_{l,g}, the posterior mean change probability of the object. */
  double changeProbability(size_t object, int cls, Gap gap) const;

  /** The prior as the second block reads it: q_{l,g}, the class-level q_{c,g} it shrinks towards
   * and the decided gaps of the object (changed, judged); `identified` is false while the hierarchy
   * is still the Jeffreys start. */
  struct Report {
    double q_object = 0.0, q_class = 0.0;
    double changed = 0.0, judged = 0.0;
    bool identified = false;
  };
  Report report(size_t object, int cls, Gap gap) const;

  struct BetaFit {
    bool identified = false;  // false: Jeffreys start, objects pooled
    double a = 0.5, b = 0.5, kappa = 0.0;
  };
  BetaFit fit(Gap gap) const;

  const std::map<size_t, ObjectStats>& objects() const { return objects_; }

  nlohmann::json toJson() const;
  static PersistencePrior fromJson(const nlohmann::json& value);

 private:
  void refit() const;

  std::map<size_t, ObjectStats> objects_;
  mutable bool dirty_ = true;
  mutable std::array<BetaFit, kNumGaps> fits_;
};

}  // namespace khronos::model
