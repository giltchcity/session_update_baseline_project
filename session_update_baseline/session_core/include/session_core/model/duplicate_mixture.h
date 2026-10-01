#pragma once

#include <vector>

namespace khronos::model {

/**
 * README principle 10, eq. (12d): the offset d between two same-identity surface elements inside
 * the comparison band |d| <= B follows a two-component mixture,
 *
 *   p(d) = pi_dup N(d; Delta_s rho, sigma_dup^2) + (1 - pi_dup) / (2B),
 *   B = T + (|zeta_a| + |zeta_b|) rho,   sigma_dup^2 = sigma_ee'^2 + sigma_x^2,
 *
 * the first term being the same surface (H_dup: scale-difference bias Delta_s rho, alignment and
 * discretisation error), the second a different surface with no preferred offset (H_sep). The
 * parameters (pi_dup, Delta_s, sigma_x) are fitted by EM on all band pairs of a session pair; the
 * posterior of a pair, Pr(H_dup | d), is the share of the first term.
 */
struct BandPair {
  double d = 0.0;              // signed normal offset [m]
  double rho = 0.0;            // mean range of the readings that reach the pair [m]
  double base_variance = 0.0;  // sigma_ee'^2: discretisation and measurement variance, without sigma_x
  double band = 0.0;           // B [m]
};

struct DuplicateFit {
  double pi_dup = 0.5;   // cold start: 1/2
  double delta_s = 0.0;  // cold start: 0
  double sigma_x = 0.0;  // cold start: 0
  size_t pairs = 0;      // pairs in the fit
  bool fitted = false;
};

/** EM of (12d). `fit_bias` false is the same-session case, Delta_s = sigma_x = 0 (only pi_dup is
 * fitted). pi_dup carries the Jeffreys smoothing (r + 1/2)/(N + 1) so that it never reaches 0 or 1.
 * max_iterations and tolerance are the computation budget. */
DuplicateFit fitDuplicateMixture(const std::vector<BandPair>& pairs, const DuplicateFit& start,
                                 bool fit_bias, int max_iterations = 200, double tolerance = 1e-9);

/** ln [Pr(H_dup | d) / Pr(H_sep | d)] of one pair under `fit`. Pairs outside the band are H_sep. */
double duplicateLogOdds(const DuplicateFit& fit, const BandPair& pair);

}  // namespace khronos::model
