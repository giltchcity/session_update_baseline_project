#pragma once

#include <cstddef>
#include <map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "session_core/model/model_math.h"

namespace khronos::model {

/**
 * README principle 6, eqs. (7), (7h), (7s): the likelihood ratio of one look and the statistics it
 * is judged against.
 *
 * A look is one round of a placement: of the n reliable samples with a verdict, F see through. A
 * placement in place shows a see-through fraction around its own normal share m_l, with a spread
 * from look to look of concentration c (the drift of the registration); the normal shares of the
 * objects scatter around mu with concentration kappa (hierarchical beta-binomial, assumptions
 * (4) and (6)). The in-place distribution of an object is the posterior predictive of that model
 * (7h); a placement that has ended shows a fraction with no preference, Beta(1, 1).
 *
 * The hyperparameters (mu, kappa, c) are posteriors of "prior + the looks seen so far": the prior
 * is centred on the false see-through rate m_0 the measurement model predicts and on c_0 = kappa_0
 * (the predictive distribution then has the same two pseudo-samples as Beta(1, 1)), with the
 * information of one look; it never switches off and never becomes a fixed default.
 *
 * The same statistics give k_min, the number of looks whose identity readings cannot all be label
 * errors (eps^k <= alpha), with eps the foreign-label share on directly seen looks.
 */
class RoundModel {
 public:
  /** The in-place distribution Beta(m c, (1 - m) c) of one look of an object or class, with the
   * hyperparameters it came from (for the record of the decision). */
  struct InPlace {
    double m = 0.0, c = 0.0;         // (7h): m_l and the predictive concentration c_l
    double mu = 0.0, kappa = 0.0;    // population mean and between-object concentration
    double within = 0.0;             // c: the look-to-look concentration
    double omega_sum = 0.0;          // sum of the independent-sample equivalents of its own looks
    size_t looks = 0;                // its own directly seen looks
  };

  /** (7h) for one object, with `cold_mean` = m_0 the centre of the prior. */
  InPlace inPlace(size_t object, double cold_mean) const;
  /** (7h) for a semantic class: the pooled looks of its objects as one object. */
  InPlace classInPlace(int cls, double cold_mean) const;

  /** The intermediates of a commitment on one look (frames of a session as one look, principle 9):
   * the in-place distribution that judged it, its ln LR and the commitment of (5e). */
  struct Judgement {
    InPlace in_place;
    double ln_lr = 0.0;
    Commitment commitment = Commitment::kDefer;
  };
  /** (7) and (5e): judge a look of n samples of which f see through against `in_place`. */
  static Judgement judge(const InPlace& in_place, double n, double f);

  /** (7): ln LR = ln L0 - ln L1 of a look of n judged samples of which f see through, with
   * L1 = BB(max(f, round(m n)) | n; m c, (1 - m) c) (one-sided) and L0 = BB(f | n; 1, 1) = 1/(n+1). */
  static double logLikelihoodRatio(const InPlace& in_place, double n, double f);
  /** The smallest n whose all-see-through look has ln LR >= ln((1 - alpha)/alpha), searched up to
   * the sample budget of a placement; one more than the budget if there is none (principle 13 (g)). */
  static size_t decisiveSamples(const InPlace& in_place);

  /** A look that directly saw the placement in place, n judged samples of which f see through,
   * enters the statistics of the object (learned after it was judged). */
  void addInPlaceLook(size_t object, int cls, double n, double f);
  /** The identity labels of a look that directly saw the placement in place: `foreign` samples
   * hit by another identity of `labelled` samples hit by any identity. */
  void addLabels(double foreign, double labelled);

  /** The foreign-label share eps (Jeffreys smoothed), ln eps (the identity channel of (5r)) and
   * k_min, the least k >= 1 with eps^k <= alpha. */
  double foreignShare() const;
  double logForeignShare() const;
  size_t minHits() const;
  /** The labelled samples counted so far: 0 means eps-hat is still its prior. */
  double labelledSamples() const { return labelled_; }

  size_t numInPlaceLooks(size_t object) const;

  nlohmann::json toJson() const;
  static RoundModel fromJson(const nlohmann::json& value);

 private:
  using Looks = std::vector<std::pair<double, double>>;  // (n, F)

  /** (7h) on a set of looks of one object. */
  InPlace predictive(const Looks& looks, double cold_mean) const;

  std::map<size_t, Looks> looks_;
  std::map<size_t, int> class_of_;
  double foreign_ = 0.0, labelled_ = 0.0;
};

}  // namespace khronos::model
