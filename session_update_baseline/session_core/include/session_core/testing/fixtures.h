#pragma once

#include <cmath>

#include "session_core/evidence/observed_absence.h"
#include "session_core/model/persistence_prior.h"
#include "session_core/model/range_model.h"
#include "session_core/model/round_model.h"

/** Synthetic model parameters for the checks of the unified model (not used by the system). */
namespace khronos::testing {

/** A range model with one sigma_table for every range bin, equal outlier weights and the given
 * scale. */
inline model::RangeModel fixedRangeModel(double sigma = 0.02, double outlier = 0.01,
                                         double zeta = 0.0) {
  model::RangeModel psi;
  psi.range_bin = 0.5;
  psi.num_range_bins = 16;
  psi.sigma_table.assign(psi.num_range_bins, sigma);
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

/** Statistics of a quiet history: looks that directly saw each object in place with a small
 * see-through share (so that the in-place distribution is learned and one look of hits confirms
 * a placement), the identity labels of those looks, and session gaps decided "not changed" (so that a
 * silent new view of the placement can be merged). */
inline void trainedStatistics(model::PersistencePrior& prior, model::RoundModel& rounds,
                              size_t first_object, size_t num_objects) {
  for (size_t i = 0; i < num_objects; ++i) {
    // The normal shares differ between the objects (0%, 2%, 4%, ...): a population with a spread; a
    // look judges 100 samples.
    for (int r = 0; r < 4; ++r) {
      const double samples = 100.0;
      const double through = 2.0 * static_cast<double>(i % 3) + ((r % 2) ? 1.0 : 0.0);
      rounds.addInPlaceLook(first_object + i, 0, samples, through);
    }
    // 20 gaps of each type decided "not changed" per object: the pooled q = (0 + 1/2)/(60 + 1) < alpha.
    for (int g = 0; g < 20; ++g) prior.addOutcome(first_object + i, 0, model::Gap::kSession, false);
    for (int g = 0; g < 20; ++g) prior.addOutcome(first_object + i, 0, model::Gap::kWithinSession, false);
    for (int g = 0; g < 20; ++g) prior.addOutcome(first_object + i, 0, model::Gap::kContinuous, false);
  }
  rounds.addLabels(0.0, 1000.0);
}

}  // namespace khronos::testing
