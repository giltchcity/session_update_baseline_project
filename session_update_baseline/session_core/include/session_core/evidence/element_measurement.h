#pragma once

#include <cstddef>
#include <unordered_map>

#include "session_core/evidence/element_round.h"
#include "session_core/evidence/physical_evidence_store.h"
#include "session_core/model/model_math.h"
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
  // h of the element's layer, the half voxel [m]: tau = max(h, sigma_table(rho)) with
  // (|zeta_now| rho + |zeta_e| rho_e) is the window of (12d) in which the offsets that estimate
  // sigma_x are taken (principle 10). 0 where unknown: no band pair is recorded.
  double half_voxel = 0.0;
  // The previous session's sigma_x, the prior centre of this session's (principle 10); when unknown
  // the prior is the standard deviation W / sqrt(3) of the uniform distribution on the window.
  double prior_sigma_x = 0.0;
  bool prior_sigma_x_known = false;
  // zeta_e: the depth scale of the session that made the elements of a previous session, and
  // their recorded distances rho_e by element cell (a missing record means the reading's own
  // range, b = (zeta_now - zeta_e) rho).
  double element_zeta = 0.0;
  const std::unordered_map<uint64_t, float>* element_range = nullptr;
  // The looks of a placement itself (principles 3 and 6): whether the look directly saw it in place
  // with k_min identity samples and the learning records of such a look. Off for the evidence of a
  // candidate against a placement, whose look is judged by its own measurement.
  bool placement_round = false;
  size_t k_min = 0;
};

/**
 * README principle 4, eqs. (6), (6s), (6e), and the first return of the entity (assumption (6)):
 * every element of the surface (one per cell of the map
 * resolution, at most the sampling budget) is looked up in the stored frames of [earliest, latest];
 * each reading is classified T / H / O / I by the Bayes boundaries of psi, and the latest T or H
 * verdict of every element is kept (one look). A reading of an element of a previous session inside
 * the window feeds sigma_x; before sigma_x is estimated such elements are not judged (principle 4:
 * it is not replaced by 0). A reading beyond a sample that lands on another part of the same
 * entity's stored surface (within delta_*) or inside the entity's extent (the box of the mesh
 * expanded by delta_*) with its own identity says the entity is still there: it is O for the
 * sample, and a hit of the entity's identity. `physical_id` is the identity whose label counts as own.
 */
ElementRound measureElements(const PhysicalEvidenceStore::Snapshot& evidence, size_t physical_id,
                             const spark_dsg::Mesh& mesh, const BoundingBox& bbox,
                             const model::RangeModel& psi, const ElementSource& source,
                             TimeStamp earliest, TimeStamp latest,
                             model::SensorCalibrator* calibrator = nullptr);

/** The elements' cells of a mesh (for the registry's element bookkeeping). */
std::vector<uint64_t> elementKeys(const spark_dsg::Mesh& mesh, const BoundingBox& bbox,
                                  float element_size);

/** README principle 4: the world-frame point lies inside the extent of an entity -- the box of its
 * valid observations expanded by `margin` (delta_*). */
bool pointInExtent(const BoundingBox& bbox, const Eigen::Vector3f& world_point, float margin);

}  // namespace khronos
