#pragma once

#include <cmath>

#include "session_core/evidence/observed_absence.h"
#include "session_core/model/persistence_prior.h"
#include "session_core/model/range_model.h"
#include "session_core/model/round_model.h"

/** Synthetic model parameters for the checks of the unified model (not used by the system). */
namespace khronos::testing {

/** A range model with one sigma_s for every cell, equal outlier weights and the given scale. */
inline model::RangeModel fixedRangeModel(double sigma = 0.02, double outlier = 0.01,
                                         double zeta = 0.0) {
  model::RangeModel psi;
  psi.range_bin = 0.5;
  psi.num_range_bins = 16;
  psi.incidence_bin = 0.5 * 3.14159265358979323846 / 6.0;
  psi.num_incidence_bins = 6;
  psi.sigma_s.assign(psi.num_range_bins * psi.num_incidence_bins, sigma);
  psi.w_plus = psi.w_minus = outlier;
  psi.zeta = zeta;
  return psi;
}

/** The evidence model of a session that started at `session_start` with the fixed range model. */
inline void primeEvidence(ObservedAbsenceModel& model, TimeStamp session_start,
                          double max_range = 5.0, double sigma = 0.02) {
  model.setInitialRangeModel(fixedRangeModel(sigma));
  model.statistics()->noteFrame(session_start, max_range);
}

/** Statistics of a long, quiet history: `seconds` of committed exposure per object and rounds with
 * a small see-through share, so that a placement in place is confirmed by a round of hits. */
inline void trainedStatistics(model::PersistencePrior& prior, model::RoundModel& rounds,
                              size_t first_object, size_t num_objects, double seconds = 1.0e5) {
  for (size_t i = 0; i < num_objects; ++i) {
    prior.addExposure(first_object + i, 0, seconds);
    for (int r = 0; r < 4; ++r) {
      rounds.addInPlaceRound(first_object + i, {50.0, (r % 2) ? 1.0 : 0.0, 0.0, 0.0});
    }
  }
  for (int e = 0; e < 8; ++e) rounds.moveElementHistory(0, 0, (e % 4 == 0) ? 1.0 : 0.0, 4.0);
}

}  // namespace khronos::testing
