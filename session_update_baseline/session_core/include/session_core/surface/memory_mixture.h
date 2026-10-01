#pragma once

#include <cstddef>
#include <vector>

namespace khronos::surface {

/**
 * README principle 9: the prior probability that a memory element has changed (q_e) and the
 * distribution of its see-through fraction once it has, both estimated at the end of the session
 * from the frames of all memory elements of a group as the two-component mixture
 *
 *   p(F_e | n_e) = q BB(F_e | n_e; a_0, b_0) + (1 - q) L_e^{in place},
 *
 * where L_e^{in place} is the element's beta-binomial under its own theta_e posterior (7s) and
 * the first component is the ended distribution. The estimation is empirical Bayes by EM with the
 * Jeffreys smoothing of q; groups are the classes of principle 2 (background by class, objects
 * by class under a placement committed in place).
 */
struct MemoryDatum {
  double frames = 0.0;       // n_e: frames with a verdict (this session and pending ones)
  double see_through = 0.0;  // F_e: of them see-through
  double in_place_log = 0.0; // ln BB(F_e | n_e; theta_e posterior) (7s), fixed
  size_t group = 0;
};

struct MemoryMixture {
  bool identified = false;  // false: the mixture has no information (no in-place statistic or no data)
  double q = 0.5;           // prior probability of "changed"; 1/2 where unidentified
  double a0 = 1.0, b0 = 1.0;  // ended distribution of the see-through fraction; Beta(1,1) start
  size_t elements = 0;
};

/** One mixture per group. `start_a0`/`start_b0` are the initial ended distribution (the rounds'
 * ended distribution of principle 6, Beta(1,1) at cold start). A group with no informative
 * element takes the fit of the pooled data of all groups. `max_iterations` is the computation
 * budget of the EM. */
std::vector<MemoryMixture> fitMemoryMixtures(const std::vector<MemoryDatum>& data,
                                             size_t num_groups, double start_a0, double start_b0,
                                             int max_iterations = 100);

}  // namespace khronos::surface
