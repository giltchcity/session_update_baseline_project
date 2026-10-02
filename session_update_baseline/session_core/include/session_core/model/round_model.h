#pragma once

#include <cstddef>
#include <map>
#include <vector>

#include <nlohmann/json.hpp>

namespace khronos::model {

/**
 * README principle 6, eqs. (7), (7h), (7s): the likelihood ratio of one look and the statistics it
 * is judged against.
 *
 * A look is one round of a placement: of the n reliable samples with a verdict, F see through. A
 * placement in place shows a see-through fraction around its own normal share (the in-place
 * distribution (m, c), a beta distribution learned only from the looks that directly saw the
 * placement in place); a placement that has ended shows a fraction with no preference, Beta(1, 1).
 * The in-place distribution of an object is its own mean and robust scatter shrunk to the
 * population of the objects (three pseudo-looks, scatter not narrower than between objects,
 * concentration not below 2); with no population it is the share the measurement model predicts
 * (m_0) with concentration 2.
 *
 * The same statistics give k_min, the number of identity hits that makes "all from label errors"
 * improbable (eps^k <= alpha), with eps the foreign-label share on directly seen placements.
 */
class RoundModel {
 public:
  /** (m, c): the beta distribution Beta(m c, (1 - m) c) of the see-through fraction in place. */
  struct InPlace {
    double m = 0.0, c = 2.0;
  };

  /** (7h) for one object: its own in-place looks shrunk to the population; `cold_mean` is m_0 of
   * the measurement model, used where there is no population yet. */
  InPlace inPlace(size_t object, double cold_mean) const;
  /** (7h) for a semantic class: the pooled in-place looks of the objects of the class. */
  InPlace classInPlace(int cls, double cold_mean) const;

  /** (7): ln LR = ln L0 - ln L1 of a look of n judged samples of which f see through, with
   * L1 = BB(max(f, round(m n)) | n; m c, (1 - m) c) (one-sided) and L0 = BB(f | n; 1, 1) = 1/(n+1). */
  static double logLikelihoodRatio(const InPlace& in_place, double n, double f);
  /** The smallest n whose all-see-through look has ln LR >= ln((1 - alpha)/alpha): a decisive look
   * (principle 13 (g)). */
  static size_t decisiveSamples(const InPlace& in_place);

  /** A look that directly saw the placement in place, with see-through fraction f = F / n,
   * enters the in-place statistics of the object (learned after it was judged). */
  void addInPlaceLook(size_t object, int cls, double fraction);
  /** The identity labels of a look that directly saw the placement in place: `foreign` hits that
   * carry another identity of `labelled` hits that carry any. */
  void addLabels(double foreign, double labelled);

  /** The foreign-label share eps with Jeffreys smoothing, and k_min, the least k >= 1 with
   * eps^k <= alpha. */
  double foreignShare() const;
  size_t minHits() const;

  /** Statistics of the population, for the record (README table 5.1). */
  struct Population {
    bool available = false;
    double mu = 0.0, scatter = 0.0;  // between-object mean and robust variance of the normal share
    size_t objects = 0;
  };
  Population population() const;
  size_t numInPlaceLooks(size_t object) const;

  nlohmann::json toJson() const;
  static RoundModel fromJson(const nlohmann::json& value);

 private:
  InPlace shrunk(const std::vector<double>& looks, double cold_mean) const;

  std::map<size_t, std::vector<double>> looks_;  // object -> see-through fractions in place
  std::map<size_t, int> class_of_;
  double foreign_ = 0.0, labelled_ = 0.0;
};

}  // namespace khronos::model
