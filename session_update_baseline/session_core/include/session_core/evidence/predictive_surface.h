#pragma once

#include <cstddef>
#include <optional>

namespace khronos { struct ProjectedEndpointEvidence; }

namespace khronos::measurement {

// README (6), (6a): shared range and identity meaning for every surface query.
// Matching tolerance must be finite and nonnegative.
enum class SurfaceVote { Unavailable, Invalid, Occluded, Supported, Free,
                         Background, Other, Unidentified };
SurfaceVote classifySurfaceMeasurement(const ProjectedEndpointEvidence& projected,
                                       size_t expected_id, float tolerance);

struct BetaReference { long double a = 1, b = 1; };
struct CompletedEvidence { size_t absent = 0, total = 0; };

// README (7): minimum predictive odds over permitted unknown-source completions.
// absent <= measured; the completed total, rather than measured+unknown, must fit size_t.
CompletedEvidence retainingCompletion(size_t absent, size_t measured, size_t unknown);

// README (7c): two normalized hypotheses on [0,alpha] and (alpha,1].
// alpha=0 is the retaining point mass at zero; impossible data has +infinite odds.
// A second reference is an equal-prior mixture, as in the state calibration model.
// Counts obey absent <= total, alpha is finite in [0,1), and every supplied beta
// shape is finite and positive, including on the alpha=0 and empty-data paths.
// Integrals are evaluated in log space when tail probabilities underflow; shape
// sums outside the numeric range and convergence failures remain explicit errors.
long double surfaceLogOdds(CompletedEvidence evidence, long double alpha,
                           BetaReference reference = {},
                           std::optional<BetaReference> second = std::nullopt);

// README (7d): equal hypothesis priors and fixed false-exit/false-retain cost ratio.
// Signed infinite odds are meaningful; NaN is invalid.
bool favorsExit(long double log_odds);

}  // namespace khronos::measurement
