#pragma once

#include <cstddef>

#include "session_core/evidence/error_model.h"

namespace khronos { struct ProjectedEndpointEvidence; }

namespace khronos::measurement {

// README (4) P4: how one actual endpoint relates to a queried surface point.
enum class SurfaceVote {
  Unavailable,  // outside the view: no information
  Invalid,      // no valid depth: no information
  Occluded,     // a nearer first hit hides the point
  Supported,    // the echo lies where the surface predicts it
  Free          // the echo passes the point: the surface is not there
};

// README (6e), (6), P3: three explanations of one valid echo r of a query at range q, each with a
// density normalised on the device range [r_min, r_max] (midpoint density of the coded bin, the
// O(u^2) approximation of README (12)):
//   support  : Gaussian around q with the effective standard deviation of psi,
//   free     : a first hit farther than q, uniform on (q, r_max],
//   occluded : a first hit nearer than q, uniform on [r_min, q).
// The class is the maximum-likelihood explanation (symmetric unit loss, equal priors, ties go to
// support). Its boundary in r is the closed interval below, computed once per query.
struct SupportInterval {
  double lower = 0, upper = 0;  // echoes in [lower, upper] are explained by the surface
};
SupportInterval supportInterval(double query, double r_min, double r_max, const ErrorModel& psi);

inline SurfaceVote classifyRange(double measured, const SupportInterval& interval) {
  if (measured > interval.upper) return SurfaceVote::Free;
  if (measured < interval.lower) return SurfaceVote::Occluded;
  return SurfaceVote::Supported;
}

// The identity carried by the echo is not used (README s4: geometry only by default).
SurfaceVote classifySurfaceMeasurement(const ProjectedEndpointEvidence& projected,
                                       const ErrorModel& psi);

// README (11), (12): log p(r | keep the surface) - log p(r | remove it) for the echoes of one
// query, with the constants computed once. Keeping predicts the Gaussian of support mixed with
// the fraction `outlier` of first hits farther than the query that a surface in place normally
// shows (README (7): the normal penetration fraction); removing predicts a first hit farther than
// the query, uniform on (q, r_max]. An echo nearer than the support interval is occluded: both
// explanations predict it alike, giving 0.
struct KeepModel {
  bool informative = false;  // false when the query lies at or beyond the device range
  double sigma = 0;
  double kappa = 0;   // log of (Gaussian peak density) / (uniform first-hit density)
  double margin = 0;  // offset of the lower support bound from the query (<= 0)
  double upper = 0;   // offset of the upper support bound from the query (>= 0)
};
KeepModel keepModel(double query, double r_min, double r_max, const ErrorModel& psi);
// `e` is the echo minus the predicted range of the echo's own ray.
double keepLogRatio(double e, const KeepModel& model, double outlier);

// README (7u): integral of Beta(a, b) over [x0, x1], for 0 <= x0 <= x1 <= 1, a, b > 0.
double betaMass(double a, double b, double x0, double x1);

}  // namespace khronos::measurement
