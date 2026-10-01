#pragma once

#include <cstddef>
#include <vector>

#include <nlohmann/json.hpp>

namespace khronos::model {

/**
 * README principle 4, eqs. (6s), (6e), (9c): the first-return model of a range reading against a
 * surface element and its parameters psi = (sigma_s, sigma_reg, w_pm, Delta s, sigma_x).
 *
 * For an element in place at predicted range rho the residual z = r - rho - b is a hit,
 * N(0, sigma_eff^2), a farther outlier (density w_+ / (R - rho)) or a nearer occluder/outlier
 * (density w_- / rho), with
 *
 *   sigma_eff^2 = sigma_s^2(rho, theta) + sigma_reg^2(|t_kappa - t_e|) + (h tan(theta))^2 / 12 + sigma_x^2,
 *   b = Delta s * rho   (same session sigma_x = b = 0; across sessions the registration term is not
 *   extrapolated over the gap between sessions, sigma_reg = 0, and sigma_x carries both sessions'
 *   registration error).
 *
 * All quantities are estimated online from static re-measurements (SensorCalibrator).
 */
struct RangeModel {
  // sigma_s(rho, theta): the pair-residual scale at vanishing time difference, metres, per range
  // bin x incidence bin; 0 is "no estimate" and carries only the quantisation variance u^2/12.
  double range_bin = 0.0;      // metres per range bin
  double incidence_bin = 0.0;  // radians per incidence bin
  size_t num_range_bins = 0, num_incidence_bins = 0;
  std::vector<double> sigma_s;  // [range bin * num_incidence_bins + incidence bin]

  // sigma_reg^2(dt): non-decreasing variogram excess over its nugget, knots in seconds.
  std::vector<double> reg_dt, reg_variance;

  double w_plus = 0.0, w_minus = 0.0;  // outlier weights (README table 5.1)
  double zeta = 0.0;                   // range scale of the session that estimated this model
  double delta_s = 0.0;                // Delta s of README (6s): scale difference to the map's sessions
  double sigma_x = 0.0;                // cross-session alignment residual
  // README (12d): weight of the same-surface component of the band-pair mixture, 1/2 at cold start.
  double pi_dup = 0.5;
  // README (9b): the largest |zeta| of the sessions that made the map's memory, |zeta_b| of the
  // comparison band B = T + (|zeta_a| + |zeta_b|) rho.
  double zeta_memory = 0.0;
  // README (9v): gamma(0+), the pooled pair variance at vanishing time difference (0: no estimate);
  // with sigma_reg^2 it gives the single-frame measurement scale sigma_eff(dt) of principle 7.
  double pair_gamma0 = 0.0;

  /** The model can classify readings once the outlier weights and some sigma_s are estimated. */
  bool valid() const;

  /** sigma_s(rho, theta) (metres); a cell without estimate carries the quantisation variance. */
  double sigmaS(double range, double incidence) const;
  /** sigma_reg^2(dt); linear extrapolation with the slope of the last interval beyond the knots. */
  double sigmaRegVariance(double dt_seconds) const;

  /** (6s). `h` is the element edge (metres), `incidence` the angle between the ray and the normal. */
  double sigmaEff(double range, double incidence, double dt_seconds, double h,
                  bool cross_session) const;
  /** (6s) without the registration and cross-session terms: sqrt(sigma_s^2 + (h tan(theta))^2/12). */
  double sigmaBase(double range, double incidence, double h) const;
  /** README (9v), principle 7: the pooled pair scale sqrt(gamma(0+) + sigma_reg^2(dt)) of two
   * readings dt apart; 0 while gamma(0+) is not estimated. */
  double pairSigma(double dt_seconds) const;
  /** README principle 7: the resolution of object reconstruction and of the session-end refusion,
   * h_o = sqrt(12) sigma_eff(dt_f) with dt_f the interval between adjacent frames (the discretisation
   * variance h^2/12 equals the variance of one measurement); 0 while the pair scale is not
   * estimated. The truncation is T = 2 h_o. */
  double objectResolution(double frame_interval_seconds) const;
  /** b of (6s). */
  double bias(double range, bool cross_session) const;

  /** (6e): the Bayes boundaries (delta_plus, delta_minus) of hit against outlier. `max_range` is R. */
  struct Bounds {
    double plus = 0.0, minus = 0.0;
  };
  Bounds bounds(double range, double sigma_eff, double max_range) const;

  nlohmann::json toJson() const;
  static RangeModel fromJson(const nlohmann::json& value);
};

/** The four explanations of one reading of a query at predicted range rho, README (6e). */
enum class RangeClass {
  kInvalid,   // no reading or outside the device range: I
  kThrough,   // z > delta_plus: T, the first return is behind the element
  kHit,       // H, the reading is explained by the element
  kOccluded,  // z < -delta_minus: O, a nearer first return
};

/** README (6e): classify the reading `r` of an element at `predicted` range with sigma_eff. */
RangeClass classifyRange(const RangeModel& psi, double reading, double predicted, double sigma_eff,
                         double min_range, double max_range, bool cross_session);

}  // namespace khronos::model
