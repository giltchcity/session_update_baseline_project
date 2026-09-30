/** -----------------------------------------------------------------------------
 * Copyright (c) 2024 Massachusetts Institute of Technology.
 * All Rights Reserved.
 *
 * AUTHORS:      Lukas Schmid <lschmid@mit.edu>, Marcus Abate <mabate@mit.edu>,
 *               Yun Chang <yunchang@mit.edu>, Luca Carlone <lcarlone@mit.edu>
 * AFFILIATION:  MIT SPARK Lab, Massachusetts Institute of Technology
 * YEAR:         2024
 * SOURCE:       https://github.com/MIT-SPARK/Khronos
 * LICENSE:      BSD 3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * -------------------------------------------------------------------------- */

#include "khronos/backend/change_detection/ray_verificator.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <glog/logging.h>
namespace khronos {
void RayVerificator::setPhysicalEvidenceStore(PhysicalEvidenceStore::Ptr store) {
  std::atomic_store(&physical_evidence_store_, std::move(store));
}

void RayVerificator::setPhysicalEvidenceCutoff(TimeStamp stamp) {
  physical_evidence_cutoff_ = stamp;
}

RayVerificator::PhysicalEvidenceSnapshot
RayVerificator::physicalEvidenceSnapshot() const {
  const auto store = std::atomic_load(&physical_evidence_store_);
  if (!store) {
    return std::nullopt;
  }
  return store->snapshot(physical_evidence_cutoff_);
}

RayVerificator::CheckResult RayVerificator::checkPhysical(
    const Point& point,
    const size_t physical_id,
    const uint64_t earliest,
    const uint64_t latest,
    CheckDetails* details) const {
  return checkPhysical(
      point, physical_id, physicalEvidenceSnapshot(), earliest, latest, details);
}

RayVerificator::CheckResult RayVerificator::checkPhysical(
    const Point& point,
    const size_t physical_id,
    const PhysicalEvidenceSnapshot& evidence_snapshot,
    const uint64_t earliest,
    const uint64_t latest,
    CheckDetails* details) const {
  return checkProjectedPhysical(point,physical_id,evidence_snapshot,earliest,latest,details);
}

RayVerificator::CheckResult RayVerificator::checkPhysicalObserved(
    const Point& point, const size_t physical_id,
    const PhysicalEvidenceSnapshot& evidence_snapshot,
    const uint64_t earliest, const uint64_t latest) const {
  return checkProjectedPhysical(point,physical_id,evidence_snapshot,earliest,latest);
}

RayVerificator::CheckResult RayVerificator::checkPhysicalReplacement(
    const Point& point, const size_t physical_id,
    const PhysicalEvidenceSnapshot& evidence_snapshot,
    const uint64_t earliest, const uint64_t latest) const {
  return checkProjectedPhysical(point,physical_id,evidence_snapshot,earliest,latest);
}

}  // namespace khronos
