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

size_t binOf(const RangeModel& psi, double range) {
  return std::min(psi.num_range_bins - 1,
                  static_cast<size_t>(std::max(0.0, range) / psi.range_bin));
}

}  // namespace

bool RangeModel::valid() const {
  if (!(w_plus > 0.0) || !(w_minus > 0.0) || !(w_plus + w_minus < 1.0)) return false;
  return std::any_of(sigma_table.begin(), sigma_table.end(), [](double s) { return s > 0.0; });
}

double RangeModel::sigmaTable(double range) const {
  // README (9c): a bin without an estimate carries the quantisation scale u / sqrt(12).
  const double floor_value = static_cast<double>(measurement::kRangeUnit) / std::sqrt(12.0);
  if (sigma_table.empty() || !(range_bin > 0.0) || num_range_bins == 0) return floor_value;
  const double s = sigma_table[binOf(*this, range)];
  return s > 0.0 ? s : floor_value;
}

double RangeModel::wPlus(double range) const {
  if (w_plus_bin.size() != num_range_bins || num_range_bins == 0 || !(range_bin > 0.0)) return w_plus;
  const double w = w_plus_bin[binOf(*this, range)];
  return w > 0.0 ? w : w_plus;
}

double RangeModel::wMinus(double range) const {
  if (w_minus_bin.size() != num_range_bins || num_range_bins == 0 || !(range_bin > 0.0)) return w_minus;
  const double w = w_minus_bin[binOf(*this, range)];
  return w > 0.0 ? w : w_minus;
}

double RangeModel::sigmaBase(double range, double incidence, double h) const {
  const double s = sigmaTable(range);
  const double broadening = h * std::tan(std::min(incidence, 0.5 * kPi - kGrazingGuard));
  return std::sqrt(s * s + broadening * broadening / 12.0);
}

double RangeModel::sigmaEff(double range, double incidence, double h, bool cross_session) const {
  const double base = sigmaBase(range, incidence, h);
  // (6s): sigma_x is nonzero only across sessions; sigma_x = 0 until it has been estimated.
  const double variance = base * base + (cross_session && sigma_x_known ? sigma_x * sigma_x : 0.0);
  return std::sqrt(variance);
}

double RangeModel::bias(double range, double element_range, double element_zeta) const {
  return zeta * range - element_zeta * element_range;
}

RangeModel::Bounds RangeModel::bounds(double range, double sigma_eff, double max_range) const {
  const double wp = wPlus(range), wm = wMinus(range);
  const double w_hit = 1.0 - wp - wm;
  const auto boundary = [&](double length, double weight) {
    if (!(length > 0.0)) return std::numeric_limits<double>::infinity();
    const double argument = w_hit * length / (weight * sigma_eff * std::sqrt(2.0 * kPi));
    return argument > 1.0 ? sigma_eff * std::sqrt(2.0 * std::log(argument)) : 0.0;
  };
  return {boundary(max_range - range, wp), boundary(range, wm)};
}

double RangeModel::predictedSeeThrough(double range, double sigma_eff, double max_range) const {
  const double wp = wPlus(range), wm = wMinus(range);
  const double bound = bounds(range, sigma_eff, max_range).plus;
  // Upper tail of the hit component beyond delta_+: 1 - Phi(delta_+ / sigma_eff).
  const double tail = std::isfinite(bound) ? 0.5 * std::erfc(bound / (sigma_eff * std::sqrt(2.0))) : 0.0;
  return wp + (1.0 - wp - wm) * tail;
}

RangeClass classifyRange(const RangeModel& psi, double reading, double predicted, double sigma_eff,
                         double min_range, double max_range, double bias) {
  if (!std::isfinite(reading) || !std::isfinite(predicted) || !(reading > 0.0) ||
      reading < min_range || reading > max_range || !(predicted > 0.0) || predicted >= max_range) {
    return RangeClass::kInvalid;
  }
  const double z = reading - predicted - bias;
  const auto bounds = psi.bounds(predicted, sigma_eff, max_range);
  if (z > bounds.plus) return RangeClass::kThrough;
  if (z < -bounds.minus) return RangeClass::kOccluded;
  return RangeClass::kHit;
}

nlohmann::json RangeModel::toJson() const {
  return nlohmann::json{{"range_bin", range_bin},
                        {"num_range_bins", num_range_bins},
                        {"sigma_table", sigma_table},
                        {"w_plus_bin", w_plus_bin},
                        {"w_minus_bin", w_minus_bin},
                        {"w_plus", w_plus},
                        {"w_minus", w_minus},
                        {"zeta", zeta},
                        {"sigma_x", sigma_x},
                        {"sigma_x_known", sigma_x_known}};
}

RangeModel RangeModel::fromJson(const nlohmann::json& value) {
  RangeModel psi;
  psi.range_bin = value.at("range_bin").get<double>();
  psi.num_range_bins = value.at("num_range_bins").get<size_t>();
  psi.sigma_table = value.at("sigma_table").get<std::vector<double>>();
  psi.w_plus_bin = value.at("w_plus_bin").get<std::vector<double>>();
  psi.w_minus_bin = value.at("w_minus_bin").get<std::vector<double>>();
  psi.w_plus = value.at("w_plus").get<double>();
  psi.w_minus = value.at("w_minus").get<double>();
  psi.zeta = value.at("zeta").get<double>();
  psi.sigma_x = value.at("sigma_x").get<double>();
  psi.sigma_x_known = value.at("sigma_x_known").get<bool>();
  const bool sized = psi.sigma_table.size() == psi.num_range_bins &&
                     (psi.w_plus_bin.empty() || psi.w_plus_bin.size() == psi.num_range_bins) &&
                     (psi.w_minus_bin.empty() || psi.w_minus_bin.size() == psi.num_range_bins);
  bool finite = std::isfinite(psi.zeta) && std::isfinite(psi.sigma_x) && psi.sigma_x >= 0.0 &&
                psi.zeta > -1.0;
  for (const double s : psi.sigma_table) finite = finite && std::isfinite(s) && s >= 0.0;
  for (const double w : psi.w_plus_bin) finite = finite && std::isfinite(w) && w >= 0.0;
  for (const double w : psi.w_minus_bin) finite = finite && std::isfinite(w) && w >= 0.0;
  if (!sized || !finite || !(psi.w_plus >= 0.0) || !(psi.w_minus >= 0.0)) {
    throw std::invalid_argument("Invalid range model");
  }
  return psi;
}

}  // namespace khronos::model
