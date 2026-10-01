#include "session_core/model/range_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "session_core/evidence/range_encoding.h"

namespace khronos::model {
namespace {

constexpr double kPi = 3.14159265358979323846;
// Numerical guard of tan(theta) at grazing incidence (the broadening then dominates sigma_eff).
constexpr double kGrazingGuard = 1.0e-6;

}  // namespace

bool RangeModel::valid() const {
  if (!(w_plus > 0.0) || !(w_minus > 0.0) || !(w_plus + w_minus < 1.0)) return false;
  return std::any_of(sigma_s.begin(), sigma_s.end(), [](double s) { return s > 0.0; });
}

double RangeModel::sigmaS(double range, double incidence) const {
  // README (9c): a cell without an estimate carries the quantisation variance u^2/12.
  const double floor_value = static_cast<double>(measurement::kRangeUnit) / std::sqrt(12.0);
  if (sigma_s.empty() || !(range_bin > 0.0) || !(incidence_bin > 0.0)) return floor_value;
  const size_t r = std::min(num_range_bins - 1, static_cast<size_t>(std::max(0.0, range) / range_bin));
  const size_t t =
      std::min(num_incidence_bins - 1, static_cast<size_t>(std::max(0.0, incidence) / incidence_bin));
  const double s = sigma_s[r * num_incidence_bins + t];
  return s > 0.0 ? s : floor_value;
}

double RangeModel::sigmaRegVariance(double dt) const {
  if (reg_dt.size() < 2 || reg_dt.size() != reg_variance.size()) return 0.0;
  if (dt <= reg_dt.front()) return reg_variance.front();
  if (dt >= reg_dt.back()) {
    const size_t n = reg_dt.size();
    const double slope = (reg_variance[n - 1] - reg_variance[n - 2]) / (reg_dt[n - 1] - reg_dt[n - 2]);
    return reg_variance.back() + slope * (dt - reg_dt.back());
  }
  const size_t hi = static_cast<size_t>(std::upper_bound(reg_dt.begin(), reg_dt.end(), dt) - reg_dt.begin());
  const size_t lo = hi - 1;
  const double t = (dt - reg_dt[lo]) / (reg_dt[hi] - reg_dt[lo]);
  return reg_variance[lo] + t * (reg_variance[hi] - reg_variance[lo]);
}

double RangeModel::sigmaBase(double range, double incidence, double h) const {
  const double s = sigmaS(range, incidence);
  const double broadening = h * std::tan(std::min(incidence, 0.5 * kPi - kGrazingGuard));
  return std::sqrt(s * s + broadening * broadening / 12.0);
}

double RangeModel::sigmaEff(double range, double incidence, double dt, double h,
                            bool cross_session) const {
  const double base = sigmaBase(range, incidence, h);
  // (6s) / principle 4 (4): across sessions sigma_reg = 0 and sigma_x carries the alignment.
  const double variance = base * base + (cross_session ? sigma_x * sigma_x : sigmaRegVariance(dt));
  return std::sqrt(variance);
}

double RangeModel::pairSigma(double dt) const {
  if (!(pair_gamma0 > 0.0)) return 0.0;
  return std::sqrt(pair_gamma0 + sigmaRegVariance(dt));
}

double RangeModel::objectResolution(double dt) const {
  return std::sqrt(12.0) * pairSigma(dt);
}

double RangeModel::bias(double range, bool cross_session) const {
  return cross_session ? delta_s * range : 0.0;
}

RangeModel::Bounds RangeModel::bounds(double range, double sigma_eff, double max_range) const {
  const double w_hit = 1.0 - w_plus - w_minus;
  const auto boundary = [&](double length, double weight) {
    if (!(length > 0.0)) return std::numeric_limits<double>::infinity();
    const double argument = w_hit * length / (weight * sigma_eff * std::sqrt(2.0 * kPi));
    return argument > 1.0 ? sigma_eff * std::sqrt(2.0 * std::log(argument)) : 0.0;
  };
  return {boundary(max_range - range, w_plus), boundary(range, w_minus)};
}

RangeClass classifyRange(const RangeModel& psi, double reading, double predicted, double sigma_eff,
                         double min_range, double max_range, bool cross_session) {
  if (!std::isfinite(reading) || !std::isfinite(predicted) || !(reading > 0.0) ||
      reading < min_range || reading > max_range || !(predicted > 0.0) || predicted >= max_range) {
    return RangeClass::kInvalid;
  }
  const double z = reading - predicted - psi.bias(predicted, cross_session);
  const auto bounds = psi.bounds(predicted, sigma_eff, max_range);
  if (z > bounds.plus) return RangeClass::kThrough;
  if (z < -bounds.minus) return RangeClass::kOccluded;
  return RangeClass::kHit;
}

nlohmann::json RangeModel::toJson() const {
  return nlohmann::json{{"range_bin", range_bin},
                        {"incidence_bin", incidence_bin},
                        {"num_range_bins", num_range_bins},
                        {"num_incidence_bins", num_incidence_bins},
                        {"sigma_s", sigma_s},
                        {"reg_dt", reg_dt},
                        {"reg_variance", reg_variance},
                        {"w_plus", w_plus},
                        {"w_minus", w_minus},
                        {"zeta", zeta},
                        {"delta_s", delta_s},
                        {"sigma_x", sigma_x},
                        {"pi_dup", pi_dup},
                        {"zeta_memory", zeta_memory},
                        {"pair_gamma0", pair_gamma0}};
}

RangeModel RangeModel::fromJson(const nlohmann::json& value) {
  RangeModel psi;
  psi.range_bin = value.at("range_bin").get<double>();
  psi.incidence_bin = value.at("incidence_bin").get<double>();
  psi.num_range_bins = value.at("num_range_bins").get<size_t>();
  psi.num_incidence_bins = value.at("num_incidence_bins").get<size_t>();
  psi.sigma_s = value.at("sigma_s").get<std::vector<double>>();
  psi.reg_dt = value.at("reg_dt").get<std::vector<double>>();
  psi.reg_variance = value.at("reg_variance").get<std::vector<double>>();
  psi.w_plus = value.at("w_plus").get<double>();
  psi.w_minus = value.at("w_minus").get<double>();
  psi.zeta = value.at("zeta").get<double>();
  psi.delta_s = value.at("delta_s").get<double>();
  psi.sigma_x = value.at("sigma_x").get<double>();
  psi.pi_dup = value.value("pi_dup", 0.5);
  psi.zeta_memory = value.value("zeta_memory", 0.0);
  psi.pair_gamma0 = value.value("pair_gamma0", 0.0);
  const bool sized = psi.sigma_s.size() == psi.num_range_bins * psi.num_incidence_bins &&
                     psi.reg_dt.size() == psi.reg_variance.size();
  bool finite = std::isfinite(psi.zeta) && std::isfinite(psi.delta_s) && std::isfinite(psi.sigma_x) &&
                psi.sigma_x >= 0.0 && psi.zeta > -1.0 &&
                std::isfinite(psi.pi_dup) && psi.pi_dup > 0.0 && psi.pi_dup < 1.0 &&
                std::isfinite(psi.zeta_memory) && psi.zeta_memory >= 0.0 &&
                std::isfinite(psi.pair_gamma0) && psi.pair_gamma0 >= 0.0;
  for (const double s : psi.sigma_s) finite = finite && std::isfinite(s) && s >= 0.0;
  for (size_t i = 0; i < psi.reg_dt.size(); ++i) {
    finite = finite && std::isfinite(psi.reg_dt[i]) && std::isfinite(psi.reg_variance[i]) &&
             psi.reg_variance[i] >= 0.0 && (i == 0 || (psi.reg_dt[i] > psi.reg_dt[i - 1] &&
                                                        psi.reg_variance[i] >= psi.reg_variance[i - 1]));
  }
  if (!sized || !finite || !(psi.w_plus >= 0.0) || !(psi.w_minus >= 0.0)) {
    throw std::invalid_argument("Invalid range model");
  }
  return psi;
}

}  // namespace khronos::model
