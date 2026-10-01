#pragma once

#include "session_core/evidence/physical_evidence_store.h"
#include "session_core/model/sensor_calibrator.h"

namespace khronos {

/**
 * README principle 8, eqs. (9b), (9v): static re-measurements of the session itself. The newest
 * stored frame is paired with earlier frames at geometrically growing offsets (1, 2, 4, ... frames,
 * one time-difference bin each); a sampled pixel of the newest frame is carried to the earlier
 * frame and compared with the range read there. The residuals feed the variogram, sigma_s(rho,
 * theta) and w_pm, and the pairs themselves the range scale. The residual is e_j(zeta) of (9b) at
 * the current scale estimate `zeta`. The pixel stride is the computational sampling of the pairs.
 */
void accumulateFramePairs(const PhysicalEvidenceStore::Snapshot& evidence, TimeStamp stamp,
                          model::SensorCalibrator& calibrator, double zeta,
                          double* device_max_range = nullptr);

}  // namespace khronos
