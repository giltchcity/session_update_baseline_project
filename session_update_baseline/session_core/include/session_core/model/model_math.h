#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

namespace khronos::model {

/** README (1) table, class 3: the only decision constant, the largest acceptable posterior
 * probability of a wrong commitment. All binary commitments of the model use this one value. */
constexpr double kAlpha = 0.01;

/** README table 5.1, computation budget: the most elements sampled from one placement (a proportion
 * of 1500 samples has a standard error of at most 1.3%). */
constexpr size_t kElementBudget = 1500;

/** README (5r): odds (1-alpha)/alpha of "changed" at which a placement is closed. */
constexpr double closeOdds() { return (1.0 - kAlpha) / kAlpha; }
/** README (5r): odds alpha/(1-alpha) of "changed" at which a placement is confirmed in place. */
constexpr double confirmOdds() { return kAlpha / (1.0 - kAlpha); }

/** README (5e): the three actions on the posterior odds of a binary hypothesis H. */
enum class Commitment { kCommitH, kCommitNotH, kDefer };
inline Commitment decide(double odds_of_h) {
  if (odds_of_h >= closeOdds()) return Commitment::kCommitH;
  if (odds_of_h <= confirmOdds()) return Commitment::kCommitNotH;
  return Commitment::kDefer;
}

/** README (5e), representation type: an output that is recomputed at every publication has no
 * deferral, so the same loss gives the maximum a posteriori choice; for a binary hypothesis H the
 * boundary is posterior odds 1 (probability 1/2), derived from the loss and not a new constant. */
inline bool representationHolds(double log_odds_of_h) { return log_odds_of_h >= 0.0; }

/** The same action on the log-odds; comparing logs avoids overflow of extreme odds. */
inline Commitment decideLog(double log_odds_of_h) {
  if (log_odds_of_h >= std::log(closeOdds())) return Commitment::kCommitH;
  if (log_odds_of_h <= std::log(confirmOdds())) return Commitment::kCommitNotH;
  return Commitment::kDefer;
}

/** ln B(a, b). */
double logBeta(double a, double b);

/** README (7): ln BB(F | n; a, b), the beta-binomial probability of F of n (a, b > 0, 0 <= F <= n);
 * F and n may be non-integral only through the gamma function continuation. */
double logBetaBinomial(double f, double n, double a, double b);

/** README (7s): I_{1/2}(a, b), the regularised incomplete beta function, i.e. Pr(theta < 1/2) of
 * theta ~ Beta(a, b). */
double betaCdfHalf(double a, double b);

/** ln sum exp(v). */
double logSumExp(const std::vector<double>& values);

/** README (9v): weighted isotonic regression to a non-decreasing sequence (pool adjacent
 * violators). Weights must be positive. */
std::vector<double> isotonicNonDecreasing(const std::vector<double>& y,
                                          const std::vector<double>& w);

/** Simplex minimisation (Nelder-Mead) of f over R^n, used to maximise the marginal likelihood of
 * README (5a). The search stops when the objective of the simplex varies by at most `tolerance`
 * (README section 5.1: a log-likelihood within alpha^2/2 puts a parameter within alpha standard
 * errors); `converged` is false when max_iterations was reached first, which the caller records. */
struct Minimum {
  std::vector<double> x;
  double value = 0.0;
  bool converged = false;
};
Minimum nelderMead(const std::function<double(const std::vector<double>&)>& f,
                   std::vector<double> x0, double step, int max_iterations, double tolerance);

/** Logit grid u_i = lo + i * step of n nodes on which integrals over a probability or a rate are
 * evaluated as sums (computation budget; the integrand carries the Jacobian). */
struct Grid {
  std::vector<double> u;
  double step = 0.0;
};
Grid makeGrid(double lo, double hi, size_t n);

}  // namespace khronos::model
