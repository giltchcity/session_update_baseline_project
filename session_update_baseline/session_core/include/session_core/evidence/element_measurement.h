#pragma once

#include <cstddef>
#include <unordered_map>

#include "session_core/evidence/element_round.h"
#include "session_core/evidence/physical_evidence_store.h"
#include "session_core/model/range_model.h"
#include "session_core/model/sensor_calibrator.h"

namespace khronos {

/** What the measurement of one placement's elements needs to know about the placement. */
struct ElementSource {
  float element_size = 0.f;      // h of (6s): the map resolution
  // First frame of this session: an element resting on a measurement before it is an element of a
  // previous session (sigma_x and b of (6s) apply), aligned to this session at its start.
  TimeStamp session_start = 0;
  TimeStamp fallback_time = 0;   // t_e where the mesh has no per-vertex acquisition time
  // T of the element's layer [m]: with (|zeta_now| + |zeta_e|) rho it is the comparison band of the
  // offsets that estimate sigma_x (principle 10). 0 where unknown: no band pair is recorded.
  double truncation = 0.0;
  // zeta_e: the depth scale of the session that made the elements of a previous session, and
  // their recorded distances rho_e by element cell (a missing record means the reading's own
  // range, b = (zeta_now - zeta_e) rho).
  double element_zeta = 0.0;
  const std::unordered_map<uint64_t, float>* element_range = nullptr;
  // The rounds of a placement itself (principles 3 and 6): the frames that directly saw it in
  // place with k_min identity hits, the learning counts of those frames and the rays. Off for the
  // exclusion rounds of candidates.
  bool placement_round = false;
  size_t k_min = 0;
};

/**
 * README principle 4, eqs. (6), (6s), (6e): every element of the surface (one per cell of the map
 * resolution, at most the sampling budget) is looked up in the stored frames of [earliest, latest];
 * each reading is classified T / H / O / I by the Bayes boundaries of psi, and the latest T or H
 * verdict of every element is kept (one look). A reading of an element of a previous session inside
 * the comparison band feeds sigma_x; before sigma_x is estimated such elements are not judged
 * (principle 4: it is not replaced by 0). `physical_id` is the identity whose label counts as own.
 */
ElementRound measureElements(const PhysicalEvidenceStore::Snapshot& evidence, size_t physical_id,
                             const spark_dsg::Mesh& mesh, const BoundingBox& bbox,
                             const model::RangeModel& psi, const ElementSource& source,
                             TimeStamp earliest, TimeStamp latest,
                             model::SensorCalibrator* calibrator = nullptr);

/** The elements' cells of a mesh (for the registry's element bookkeeping). */
std::vector<uint64_t> elementKeys(const spark_dsg::Mesh& mesh, const BoundingBox& bbox,
                                  float element_size);

/** The sampling budget of the elements of a placement (README table 5.1: 1500 elements give a
 * proportion a standard error of at most 1.3%). */
constexpr size_t kElementBudget = 1500;

}  // namespace khronos
