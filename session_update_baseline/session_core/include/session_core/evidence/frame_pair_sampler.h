#pragma once

#include <vector>

#include "session_core/evidence/physical_evidence_store.h"
#include "session_core/model/sensor_calibrator.h"

namespace khronos {

/**
 * README principle 8, eq. (9b): static re-measurements of the session itself for the range scale.
 * The newest stored frame is paired with earlier frames at geometrically growing offsets (1, 2, 4,
 * ... frames); a sampled pixel of the newest frame is carried to the earlier frame and compared
 * with the range read there. The residual is e_j(zeta) of (9b) at the current scale estimate
 * `zeta`. The pixel stride is the computational sampling of the pairs.
 */
void accumulateFramePairs(const PhysicalEvidenceStore::Snapshot& evidence, TimeStamp stamp,
                          model::SensorCalibrator& calibrator, double zeta,
                          double* device_max_range = nullptr);

/**
 * README eq. (9v): the residuals of the readings of the stored frame `stamp` against the fused
 * surface of the session (the online map `mesh`): for every vertex that faces the sensor and is
 * projected into a valid reading, r - rho. The estimator keeps those inside the truncation band
 * (not occluded) for sigma_table and all of them for w_pm. The vertices are subsampled evenly to
 * the computation budget.
 */
void accumulateFusedResiduals(const PhysicalEvidenceStore::Snapshot& evidence, TimeStamp stamp,
                              const spark_dsg::Mesh& mesh, model::SensorCalibrator& calibrator);

}  // namespace khronos
