#pragma once

#include <cmath>
#include <algorithm>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "session_core/model/model_math.h"

namespace khronos::model {

/**
 * README principle 3, eq. (5r): the CUSUM statistic of "this placement has ended",
 *
 *   C_n = max(0, C_{n-1} + w_n l^g_n + l^id_n),  C_0 = 0,   the evidence is sufficient  <=>  C_n > ln((1-alpha)/alpha),
 *
 * with l^g_n the geometric channel (the log likelihood ratio ended : in place of look n, principle
 * 6 eq. (7), weighted by the share w_n of the reliable samples judged for the first time in the
 * pass) and l^id_n the identity channel (ln eps-hat when the look directly saw the placement in
 * place, else 0). There is no prior on the time of the change (nothing seen, nothing changes), so
 * rounds without a look do not update the statistic. C returning to 0 is a new accumulation.
 */
class Cusum {
 public:
  /** The quantities of one step, for the record of the decision. */
  struct Step {
    double before = 0.0, weight = 0.0, geometric = 0.0, identity = 0.0, after = 0.0;
  };

  /** One look; returns the step. */
  Step update(double weight, double geometric, double identity) {
    Step step;
    step.before = c_;
    step.weight = weight;
    step.geometric = geometric;
    step.identity = identity;
    c_ = std::max(0.0, c_ + weight * geometric + identity);
    step.after = c_;
    return step;
  }
  double value() const { return c_; }
  /** (5r): the evidence that the placement ended exceeds the Wald boundary ln((1-alpha)/alpha). */
  bool exceeded() const { return c_ > std::log(closeOdds()); }
  void reset() { c_ = 0.0; }

  nlohmann::json toJson() const { return nlohmann::json{{"c", c_}}; }
  static Cusum fromJson(const nlohmann::json& value) {
    Cusum cusum;
    cusum.c_ = value.at("c").get<double>();
    if (!(cusum.c_ >= 0.0) || !std::isfinite(cusum.c_)) throw std::invalid_argument("Invalid CUSUM statistic");
    return cusum;
  }

 private:
  double c_ = 0.0;
};

}  // namespace khronos::model
