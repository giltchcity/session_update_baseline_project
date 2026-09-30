#pragma once

#include "session_core/evidence/observed_absence.h"
#include "session_core/evidence/physical_evidence_store.h"
#include "session_core/state/persistent_object_state.h"

namespace khronos {

/**
 * README (6d), (6m), (5e), principle 13: one decision round of the registry on a frozen evidence
 * snapshot. Every tracked identity gets the verdicts of its placement's elements in the inputs
 * after the placement's watermark, the exclusion evidence of its pending candidates, and is
 * resolved; the rounds after a committed end teach the ended distribution. Returns the number of
 * placements committed changed. A round that cannot classify (no snapshot, psi not yet estimated,
 * no frame) consumes no source.
 */
size_t runEvidenceRound(PersistentObjectState& registry, ObservedAbsenceModel& calibration,
                        const PhysicalEvidenceStore::Snapshot* evidence, TimeStamp stamp,
                        float element_size);

}  // namespace khronos
