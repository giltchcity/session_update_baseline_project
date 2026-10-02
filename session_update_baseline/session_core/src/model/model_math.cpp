#include "session_core/model/model_math.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <boost/math/policies/policy.hpp>
#include <boost/math/special_functions/beta.hpp>

namespace khronos::model {

double logBeta(double a, double b) { return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b); }

double logBetaBinomial(double f, double n, double a, double b) {
  if (!(a > 0) || !(b > 0) || !(n >= 0) || !(f >= 0) || f > n) {
    throw std::invalid_argument("Invalid beta-binomial argument");
  }
  const double log_choose = std::lgamma(n + 1.0) - std::lgamma(f + 1.0) - std::lgamma(n - f + 1.0);
  return log_choose + logBeta(f + a, n - f + b) - logBeta(a, b);
}

double betaCdfHalf(double a, double b) {
  if (!(a > 0) || !(b > 0)) throw std::invalid_argument("Invalid Beta parameters");
  using Policy = boost::math::policies::policy<
      boost::math::policies::underflow_error<boost::math::policies::ignore_error>,
      boost::math::policies::denorm_error<boost::math::policies::ignore_error>>;
  return boost::math::ibeta(a, b, 0.5, Policy{});
}

double logSumExp(const std::vector<double>& values) {
  double maximum = -std::numeric_limits<double>::infinity();
  for (const double v : values) maximum = std::max(maximum, v);
  if (!std::isfinite(maximum)) return maximum;
  double sum = 0.0;
  for (const double v : values) sum += std::exp(v - maximum);
  return maximum + std::log(sum);
}

std::vector<double> isotonicNonDecreasing(const std::vector<double>& y,
                                          const std::vector<double>& w) {
  if (y.size() != w.size()) throw std::invalid_argument("Isotonic regression size mismatch");
  struct Block {
    double mean, weight;
    size_t count;
  };
  std::vector<Block> blocks;
  for (size_t i = 0; i < y.size(); ++i) {
    if (!(w[i] > 0)) throw std::invalid_argument("Isotonic regression needs positive weights");
    blocks.push_back({y[i], w[i], 1});
    while (blocks.size() > 1 && blocks[blocks.size() - 2].mean > blocks.back().mean) {
      const Block top = blocks.back();
      blocks.pop_back();
      Block& below = blocks.back();
      const double weight = below.weight + top.weight;
      below.mean = (below.mean * below.weight + top.mean * top.weight) / weight;
      below.weight = weight;
      below.count += top.count;
    }
  }
  std::vector<double> result;
  result.reserve(y.size());
  for (const auto& block : blocks) result.insert(result.end(), block.count, block.mean);
  return result;
}

Minimum nelderMead(const std::function<double(const std::vector<double>&)>& f,
                   std::vector<double> x0, double step, int max_iterations, double tolerance) {
  const size_t n = x0.size();
  if (n == 0) throw std::invalid_argument("Empty simplex");
  std::vector<std::vector<double>> simplex(n + 1, x0);
  for (size_t i = 0; i < n; ++i) simplex[i + 1][i] += step;
  std::vector<double> value(n + 1);
  for (size_t i = 0; i <= n; ++i) value[i] = f(simplex[i]);
  std::vector<size_t> order(n + 1);
  bool converged = false;
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return value[a] < value[b]; });
    const size_t best = order.front(), worst = order.back(), second_worst = order[n - 1];
    if (std::abs(value[worst] - value[best]) <= tolerance) {
      converged = true;
      break;
    }
    std::vector<double> centroid(n, 0.0);
    for (size_t k = 0; k < n; ++k) {
      for (size_t j = 0; j < n; ++j) centroid[j] += simplex[order[k]][j] / static_cast<double>(n);
    }
    const auto along = [&](double t) {
      std::vector<double> x(n);
      for (size_t j = 0; j < n; ++j) x[j] = centroid[j] + t * (simplex[worst][j] - centroid[j]);
      return x;
    };
    const auto reflected = along(-1.0);
    const double reflected_value = f(reflected);
    if (reflected_value < value[best]) {
      const auto expanded = along(-2.0);
      const double expanded_value = f(expanded);
      if (expanded_value < reflected_value) {
        simplex[worst] = expanded;
        value[worst] = expanded_value;
      } else {
        simplex[worst] = reflected;
        value[worst] = reflected_value;
      }
    } else if (reflected_value < value[second_worst]) {
      simplex[worst] = reflected;
      value[worst] = reflected_value;
    } else {
      const auto contracted = along(reflected_value < value[worst] ? -0.5 : 0.5);
      const double contracted_value = f(contracted);
      if (contracted_value < std::min(reflected_value, value[worst])) {
        simplex[worst] = contracted;
        value[worst] = contracted_value;
      } else {
        for (size_t k = 0; k <= n; ++k) {
          if (k == best) continue;
          for (size_t j = 0; j < n; ++j) {
            simplex[k][j] = simplex[best][j] + 0.5 * (simplex[k][j] - simplex[best][j]);
          }
          value[k] = f(simplex[k]);
        }
      }
    }
  }
  const size_t best = static_cast<size_t>(std::min_element(value.begin(), value.end()) - value.begin());
  return {simplex[best], value[best], converged};
}

Grid makeGrid(double lo, double hi, size_t n) {
  if (n < 2 || !(hi > lo)) throw std::invalid_argument("Invalid grid");
  Grid grid;
  grid.step = (hi - lo) / static_cast<double>(n - 1);
  grid.u.resize(n);
  for (size_t i = 0; i < n; ++i) grid.u[i] = lo + grid.step * static_cast<double>(i);
  return grid;
}

}  // namespace khronos::model
