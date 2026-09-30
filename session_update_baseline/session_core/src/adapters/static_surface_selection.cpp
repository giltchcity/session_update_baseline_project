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

#include "khronos/active_window/object_extraction/mesh_object_extractor.h"

#include <cmath>

namespace khronos {

// README (4.0) P5, eq. (6b): a static reconstruction fuses the observations attributed to one
// placement. The registry publishes, per identity, the latest actual support of its closed
// placements (5b); frames acquired at or before it belong to a closed placement and are not fused
// into a successor. Observations not yet attributed stay in the reconstruction and become pending
// candidates in the registry until the round evidence of P2/P6 decides (eq. (6b)).
std::vector<std::pair<FrameData::Ptr, int>> MeshObjectExtractor::selectStaticFrames(
    const Track& track, const FrameDataBuffer& frame_data,
    std::optional<TimeStamp> after_stamp) const {
  if (attribution_ && track.physical_instance_id) {
    const TimeStamp closed = attribution_->closedThrough(*track.physical_instance_id);
    if (closed > 0 && (!after_stamp || *after_stamp < closed)) after_stamp = closed;
  }
  return collectSemanticFrames(track, frame_data, after_stamp);
}

std::optional<Track> MeshObjectExtractor::preparePhysicalTrack(
    const Track& track, const FrameDataBuffer& frame_data) const {
  std::optional<Track> static_fallback;
  if (track.physical_instance_id &&
      (track.is_dynamic || track.has_dynamic_history) &&
      computeDynamicDisplacement(track, frame_data) < config.min_dynamic_displacement) {
    static_fallback = track;
    static_fallback->is_dynamic = false;
    static_fallback->has_dynamic_history = false;
    static_fallback->last_motion_seen = 0;
  }
  return static_fallback;
}
}  // namespace khronos
