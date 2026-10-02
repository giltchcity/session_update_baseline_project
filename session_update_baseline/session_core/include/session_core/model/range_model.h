#pragma once

#include <cstddef>
#include <vector>

#include <nlohmann/json.hpp>

namespace khronos::model {

/**
 * README principle 4, eqs. (6s), (6e), (9c): the first-return model of a range reading against a
 * surface element and its parameters psi = (sigma_table, w_pm, zeta, sigma_x).
 *
 * For an element in place at predicted range rho the residual z = r - rho - b is a hit,
 * N(0, sigma_eff^2), a farther outlier (density w_+ / (R - rho)) or a nearer occluder/outlier
 * (density w_- / rho), with
 *
 *   sigma_eff^2 = sigma_table^2(rho) + (h tan(theta))^2 / 12 + sigma_x^2,
 *   b = zeta_now rho - zeta_e rho_e          (same session: sigma_x = b = 0).
 *
 * sigma_table is the robust residual scale of the readings against the fused surface of the session
 * (principle 8), binned by range; it does not widen with the time difference of two measurements.
 * All quantities are estimated online (SensorCalibrator) or taken from the previous session.
 */
struct RangeModel {
  // sigma_table(rho): metres per range bin; 0 is "no estimate" (the bin takes the nearest estimated
  // one when the model is estimated, else the quantisation scale u / sqrt(12)).
  double range_bin = 0.0;  // metres per range bin
  size_t num_range_bins = 0;
  std::vector<double> sigma_table;
  // Outlier weights per range bin (pooled first, then shrunk per bin), README table 5.1.
  std::vector<double> w_plus_bin, w_minus_bin;
  double w_plus = 0.0, w_minus = 0.0;  // pooled over all bins

  double zeta = 0.0;           // range scale of the session that estimated this model, (9b)
  double sigma_x = 0.0;        // cross-session alignment residual, valid iff sigma_x_known
  bool sigma_x_known = false;  // false until the first overlap with a previous session's surface

  /** The model can classify readings once the outlier weights and some sigma_table are estimated. */
  bool valid() const;

  /** sigma_table(rho) (metres); a bin without estimate carries the quantisation scale. */
  double sigmaTable(double range) const;
  /** The outlier weights at `range` (the pooled weights where the bin holds none). */
  double wPlus(double range) const;
  double wMinus(double range) const;

  /** (6s). `h` is the element edge (metres), `incidence` the angle between the ray and the normal;
   * `cross_session` adds sigma_x^2. */
  double sigmaEff(double range, double incidence, double h, bool cross_session) const;
  /** (6s) without the cross-session term: sqrt(sigma_table^2 + (h tan(theta))^2/12). */
  double sigmaBase(double range, double incidence, double h) const;
  /** b of (6s): zeta_now rho - zeta_e rho_e, the scale displacement between the measurement that
   * made the element (range rho_e in a session of scale zeta_e) and the reading. */
  double bias(double range, double element_range, double element_zeta) const;

  /** (6e): the Bayes boundaries (delta_plus, delta_minus) of hit against outlier. `max_range` is R. */
  struct Bounds {
    double plus = 0.0, minus = 0.0;
  };
  Bounds bounds(double range, double sigma_eff, double max_range) const;

  /** Principle 6: the false see-through rate the measurement model itself predicts for an element in
   * place, m_0 = w_+ + w_H (1 - Phi(delta_+ / sigma_eff)). */
  double predictedSeeThrough(double range, double sigma_eff, double max_range) const;

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

/** README (6e): classify the reading `r` of an element at `predicted` range with sigma_eff and the
 * scale displacement `bias` (b of (6s); 0 within one session). */
RangeClass classifyRange(const RangeModel& psi, double reading, double predicted, double sigma_eff,
                         double min_range, double max_range, double bias = 0.0);

}  // namespace khronos::model
