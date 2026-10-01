#include "session_core/surface/memory_mixture.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <utility>

#include "session_core/model/model_math.h"

namespace khronos::surface {
namespace {

constexpr double kMinConcentration = 1.0e-2;  // numerical bounds of the ended distribution
constexpr double kMaxConcentration = 1.0e6;
constexpr double kMeanMargin = 1.0e-6;

// The EM of one group over the informative elements `members` of `data`.
MemoryMixture fitOne(const std::vector<MemoryDatum>& data, const std::vector<size_t>& members,
                     double start_a0, double start_b0, int max_iterations) {
  MemoryMixture mixture;
  mixture.elements = members.size();
  if (members.size() < 2) return mixture;

  // Distinct (n, F) pairs: the ended component is a function of them only.
  std::map<std::pair<long long, long long>, size_t> index_of;
  std::vector<std::pair<double, double>> pairs;
  std::vector<size_t> pair_of(members.size());
  for (size_t m = 0; m < members.size(); ++m) {
    const auto& d = data[members[m]];
    const std::pair<long long, long long> key{std::llround(d.frames), std::llround(d.see_through)};
    const auto [found, inserted] = index_of.emplace(key, pairs.size());
    if (inserted) pairs.push_back({std::max(0.0, d.frames), std::min(std::max(0.0, d.see_through), d.frames)});
    pair_of[m] = found->second;
  }

  double q = 0.5, a0 = start_a0, b0 = start_b0;
  std::vector<double> table(pairs.size()), weight(pairs.size());
  const double count = static_cast<double>(members.size());
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    for (size_t i = 0; i < pairs.size(); ++i) {
      table[i] = model::logBetaBinomial(pairs[i].second, pairs[i].first, a0, b0);
    }
    // E-step: the share of the ended component.
    std::fill(weight.begin(), weight.end(), 0.0);
    double sum = 0.0;
    const double log_prior = std::log(q) - std::log1p(-q);
    for (size_t m = 0; m < members.size(); ++m) {
      const double log_ratio = log_prior + table[pair_of[m]] - data[members[m]].in_place_log;
      const double r = log_ratio >= 0.0 ? 1.0 / (1.0 + std::exp(-log_ratio))
                                        : std::exp(log_ratio) / (1.0 + std::exp(log_ratio));
      weight[pair_of[m]] += r;
      sum += r;
    }
    // M-step: q with the Jeffreys smoothing; the ended distribution by the weighted beta-binomial
    // maximum likelihood over the distinct pairs.
    const double q_next = (sum + 0.5) / (count + 1.0);
    double a_next = a0, b_next = b0;
    if (sum > 1e-9) {
      // Only the pairs that carry weight enter the maximisation (computation budget: a pair whose
      // total responsibility is below 1e-9 of the ended mass changes the optimum by less than the
      // tolerance of the search).
      std::vector<size_t> active;
      for (size_t i = 0; i < pairs.size(); ++i) {
        if (weight[i] > 1e-9 * sum) active.push_back(i);
      }
      const auto objective = [&](const std::vector<double>& x) {
        const double mean = std::clamp(1.0 / (1.0 + std::exp(-x[0])), kMeanMargin, 1.0 - kMeanMargin);
        const double concentration = std::clamp(std::exp(x[1]), kMinConcentration, kMaxConcentration);
        double total = 0.0;
        for (const size_t i : active) {
          total += weight[i] * model::logBetaBinomial(pairs[i].second, pairs[i].first,
                                                      mean * concentration,
                                                      (1.0 - mean) * concentration);
        }
        return std::isfinite(total) ? -total : std::numeric_limits<double>::max();
      };
      const double mean0 = std::clamp(a0 / (a0 + b0), kMeanMargin, 1.0 - kMeanMargin);
      const auto best = model::nelderMead(
          objective, {std::log(mean0 / (1.0 - mean0)), std::log(a0 + b0)}, 0.5, 100, 1e-9);
      const double mean = std::clamp(1.0 / (1.0 + std::exp(-best.x[0])), kMeanMargin, 1.0 - kMeanMargin);
      const double concentration = std::clamp(std::exp(best.x[1]), kMinConcentration, kMaxConcentration);
      a_next = mean * concentration;
      b_next = (1.0 - mean) * concentration;
    }
    const double change = std::abs(q_next - q) + std::abs(a_next - a0) / (a0 + 1.0) +
                          std::abs(b_next - b0) / (b0 + 1.0);
    q = q_next;
    a0 = a_next;
    b0 = b_next;
    if (change < 1e-7) break;
  }
  mixture.identified = true;
  mixture.q = q;
  mixture.a0 = a0;
  mixture.b0 = b0;
  return mixture;
}

}  // namespace

std::vector<MemoryMixture> fitMemoryMixtures(const std::vector<MemoryDatum>& data,
                                             size_t num_groups, double start_a0, double start_b0,
                                             int max_iterations) {
  std::vector<std::vector<size_t>> members(num_groups);
  std::vector<size_t> all;
  for (size_t i = 0; i < data.size(); ++i) {
    const auto& d = data[i];
    // An element without a verdict, or without an in-place statistic (NaN), carries no information.
    if (!(d.frames > 0.0) || !std::isfinite(d.in_place_log) || d.group >= num_groups) continue;
    members[d.group].push_back(i);
    all.push_back(i);
  }
  // The pooled fit is only needed by a group that cannot be fitted on its own.
  std::optional<MemoryMixture> pooled;
  std::vector<MemoryMixture> result(num_groups);
  for (size_t g = 0; g < num_groups; ++g) {
    if (members[g].size() >= 2) {
      result[g] = fitOne(data, members[g], start_a0, start_b0, max_iterations);
    } else {
      if (!pooled) pooled = fitOne(data, all, start_a0, start_b0, max_iterations);
      result[g] = *pooled;
    }
    result[g].elements = members[g].size();
  }
  return result;
}

}  // namespace khronos::surface
