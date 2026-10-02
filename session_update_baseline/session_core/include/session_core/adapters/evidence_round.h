#pragma once

#include "session_core/evidence/observed_absence.h"
#include "session_core/evidence/physical_evidence_store.h"
#include "session_core/state/persistent_object_state.h"

namespace khronos {

/**
 * README (6d), (6m), (5e), principles 6 and 13: one decision round of the registry on a frozen
 * evidence snapshot. Every tracked identity gets the verdicts of its placement's elements in the
 * inputs after the placement's watermark (one look), the evidence of its pending candidates in
 * both directions, and is resolved. Returns the number of placements committed ended. A round that
 * cannot classify (no snapshot, psi not yet estimated, no frame) consumes no source, and a
 * placement of a previous session is not judged before sigma_x has been estimated from the first
 * overlap (its frames stay unconsumed). `object_half_voxel` is h = h_o / 2 of the object layer,
 * which with the scales forms the window of (12d) in which the offsets of sigma_x are taken.
 */
size_t runEvidenceRound(PersistentObjectState& registry, ObservedAbsenceModel& calibration,
                        const PhysicalEvidenceStore::Snapshot* evidence, TimeStamp stamp,
                        float element_size, double object_half_voxel);

}  // namespace khronos
