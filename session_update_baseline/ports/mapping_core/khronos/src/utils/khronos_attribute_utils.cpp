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

#include "khronos/utils/khronos_attribute_utils.h"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>

#include "khronos/utils/geometry_utils.h"

namespace khronos {

Point computeSurfaceCentroid(const KhronosObjectAttributes& attrs) {
  return utils::computeCentroid(attrs.mesh.points);
}

spark_dsg::Mesh::Ptr composeCurrentSceneMesh(
    const DynamicSceneGraph& dsg,
    std::optional<TimeStamp> query_time) {
  auto output = dsg.hasMesh() && dsg.mesh()
                    ? dsg.mesh()->clone()
                    : std::make_shared<spark_dsg::Mesh>(true, true, true, false);
  if (!dsg.hasLayer(DsgLayers::OBJECTS)) {
    return output;
  }

  for (const auto& [unused_id, node] :
       dsg.getLayer(DsgLayers::OBJECTS).nodes()) {
    (void)unused_id;
    const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
    if (!attrs || !hasCurrentObjectMesh(*attrs) ||
        (query_time && !isPresent(*attrs, *query_time))) {
      continue;
    }

    spark_dsg::Mesh object_world(output->has_colors,
                                 output->has_timestamps,
                                 output->has_labels,
                                 output->has_first_seen_stamps);
    object_world.resizeVertices(attrs->mesh.numVertices());
    for (size_t i = 0; i < attrs->mesh.numVertices(); ++i) {
      object_world.setPos(
          i, attrs->bounding_box.pointToWorldFrame(attrs->mesh.pos(i)));
      if (object_world.has_colors) {
        object_world.setColor(
            i,
            attrs->mesh.has_colors && i < attrs->mesh.colors.size()
                ? attrs->mesh.colors[i]
                : spark_dsg::Color(180, 180, 180));
      }
      if (object_world.has_timestamps) {
        object_world.setTimestamp(i, 0);
      }
      if (object_world.has_first_seen_stamps) {
        object_world.setFirstSeenTimestamp(i, 0);
      }
      if (object_world.has_labels) {
        object_world.setLabel(
            i, static_cast<spark_dsg::Mesh::Label>(attrs->semantic_label));
      }
    }

    std::vector<spark_dsg::Mesh::Face> valid_faces;
    valid_faces.reserve(attrs->mesh.numFaces());
    for (size_t i = 0; i < attrs->mesh.numFaces(); ++i) {
      const auto& face = attrs->mesh.face(i);
      if (face[0] < attrs->mesh.numVertices() &&
          face[1] < attrs->mesh.numVertices() &&
          face[2] < attrs->mesh.numVertices()) {
        valid_faces.push_back(face);
      }
    }
    object_world.resizeFaces(valid_faces.size());
    for (size_t i = 0; i < valid_faces.size(); ++i) {
      object_world.face(i) = valid_faces[i];
    }

    if (!output->append(object_world)) {
      throw std::runtime_error(
          "Failed to compose a current private object mesh into the scene mesh");
    }
  }
  return output;
}

bool hasCurrentObjectMesh(const KhronosObjectAttributes& attrs) {
  return attrs.mesh.numVertices() > 0;
}

bool hasTrajectoryHistory(const KhronosObjectAttributes& attrs) {
  return trajectoryHistorySize(attrs) > 0;
}

size_t trajectoryHistorySize(const KhronosObjectAttributes& attrs) {
  return std::min(attrs.trajectory_timestamps.size(), attrs.trajectory_positions.size());
}

namespace {

std::optional<TimeStamp> detailStamp(const KhronosObjectAttributes& attrs,
                                     const char* key) {
  const auto iter = attrs.details.find(key);
  if (iter == attrs.details.end() || iter->second.size() != 1) {
    return std::nullopt;
  }
  return static_cast<TimeStamp>(iter->second.front());
}

template <typename Container>
std::optional<TimeStamp> finiteMin(const Container& values) {
  std::optional<TimeStamp> result;
  for (const auto value : values) {
    const auto stamp = static_cast<TimeStamp>(value);
    if (stamp == 0 || stamp == std::numeric_limits<TimeStamp>::max()) {
      continue;
    }
    result = result ? std::min(*result, stamp) : stamp;
  }
  return result;
}

template <typename Container>
std::optional<TimeStamp> finiteMax(const Container& values) {
  std::optional<TimeStamp> result;
  for (const auto value : values) {
    const auto stamp = static_cast<TimeStamp>(value);
    if (stamp == 0 || stamp == std::numeric_limits<TimeStamp>::max()) {
      continue;
    }
    result = result ? std::max(*result, stamp) : stamp;
  }
  return result;
}

}  // namespace

TimeStamp observationFirstStamp(const KhronosObjectAttributes& attrs) {
  if (const auto stamp = detailStamp(attrs, kObservationFirstStampDetail)) {
    return *stamp;
  }
  // This fallback makes states written before the explicit observation fields
  // usable: private object meshes retain their real per-vertex timestamps even
  // when the estimated presence interval is open-ended.
  if (const auto stamp = finiteMin(attrs.mesh.first_seen_stamps)) {
    return *stamp;
  }
  return attrs.first_observed_ns.empty() ? 0 : attrs.first_observed_ns.front();
}

TimeStamp observationLastStamp(const KhronosObjectAttributes& attrs) {
  if (const auto stamp = detailStamp(attrs, kObservationLastStampDetail)) {
    return *stamp;
  }
  if (const auto stamp = finiteMax(attrs.mesh.stamps)) {
    return *stamp;
  }
  return attrs.last_observed_ns.empty() ? 0 : attrs.last_observed_ns.back();
}

void setObservationBounds(KhronosObjectAttributes& attrs,
                          TimeStamp first,
                          TimeStamp last) {
  attrs.details[kObservationFirstStampDetail] = {static_cast<size_t>(first)};
  attrs.details[kObservationLastStampDetail] = {static_cast<size_t>(last)};
}

void persistObservationBounds(KhronosObjectAttributes& attrs) {
  const auto first = observationFirstStamp(attrs);
  const auto last = observationLastStamp(attrs);
  if (first != 0 || last != 0) {
    setObservationBounds(attrs, first, last);
  }
}

void mergeTrajectoryHistory(
    const std::vector<const KhronosObjectAttributes*>& sources,
    KhronosObjectAttributes& output) {
  struct Sample {
    Point position = Point::Zero();
    Points points;
  };
  std::map<TimeStamp, Sample> samples;
  bool any_point_frames = false;
  for (const auto* source : sources) {
    if (!source) {
      continue;
    }
    const auto count = trajectoryHistorySize(*source);
    for (size_t i = 0; i < count; ++i) {
      auto [iter, inserted] = samples.try_emplace(
          source->trajectory_timestamps[i], Sample{source->trajectory_positions[i], {}});
      if (inserted) {
        iter->second.position = source->trajectory_positions[i];
      }
      if (i < source->dynamic_object_points.size() &&
          !source->dynamic_object_points[i].empty()) {
        any_point_frames = true;
        if (iter->second.points.size() < source->dynamic_object_points[i].size()) {
          iter->second.points = source->dynamic_object_points[i];
        }
      }
    }
  }

  output.trajectory_timestamps.clear();
  output.trajectory_positions.clear();
  output.dynamic_object_points.clear();
  output.trajectory_timestamps.reserve(samples.size());
  output.trajectory_positions.reserve(samples.size());
  if (any_point_frames) {
    output.dynamic_object_points.reserve(samples.size());
  }
  for (auto& [stamp, sample] : samples) {
    output.trajectory_timestamps.push_back(stamp);
    output.trajectory_positions.push_back(sample.position);
    if (any_point_frames) {
      output.dynamic_object_points.emplace_back(std::move(sample.points));
    }
  }
}

std::optional<TimeStamp> lastAppearedBefore(const KhronosObjectAttributes& attrs,
                                            const TimeStamp query_time) {
  TimeStamp last_appeared = 0;
  bool found = false;
  for (const auto& time : attrs.first_observed_ns) {
    if (time > query_time) {
      break;
    }
    last_appeared = time;
    found = true;
  }
  if (found) {
    return last_appeared;
  } else {
    return std::nullopt;
  }
}

std::optional<TimeStamp> lastDisappearedBefore(const KhronosObjectAttributes& attrs,
                                               const TimeStamp query_time) {
  TimeStamp last_disappeared = 0;
  bool found = false;
  for (const auto& time : attrs.last_observed_ns) {
    if (time > query_time) {
      break;
    }
    last_disappeared = time;
    found = true;
  }
  if (found) {
    return last_disappeared;
  } else {
    return std::nullopt;
  }
}

bool isPresent(const KhronosObjectAttributes& attrs, const TimeStamp query_time) {
  const auto last_appeared = lastAppearedBefore(attrs, query_time);
  if (!last_appeared) {
    return false;
  }
  const auto last_disappeared = lastDisappearedBefore(attrs, query_time);
  if (!last_disappeared) {
    return true;
  }
  return last_appeared.value() > last_disappeared.value();
}

bool hasAppeared(const KhronosObjectAttributes& attrs, const TimeStamp query_time) {
  const auto last_appeared = lastAppearedBefore(attrs, query_time);
  if (!last_appeared) {
    return false;
  }
  const auto last_disappeared = lastDisappearedBefore(attrs, query_time);
  if (!last_disappeared) {
    return last_appeared.value() > 0;
  }
  return last_appeared.value() > 0 && last_appeared.value() > last_disappeared.value();
}

bool hasDisappeared(const KhronosObjectAttributes& attrs, const TimeStamp query_time) {
  const auto last_disappeared = lastDisappearedBefore(attrs, query_time);
  if (!last_disappeared) {
    return false;
  }
  const auto last_appeared = lastAppearedBefore(attrs, query_time);
  if (!last_appeared) {
    return true;
  }
  return last_disappeared.value() > last_appeared.value();
}

void addPresenceDuration(KhronosObjectAttributes& attrs,
                         const TimeStamp t_start,
                         const TimeStamp t_end) {
  if (t_start > t_end) {
    throw std::invalid_argument(
        "Cannot add a presence interval whose start exceeds its end");
  }
  if (attrs.first_observed_ns.size() != attrs.last_observed_ns.size()) {
    throw std::invalid_argument(
        "Cannot add a presence interval to unpaired presence vectors");
  }

  using Interval = std::pair<TimeStamp, TimeStamp>;
  std::vector<Interval> intervals;
  intervals.reserve(attrs.first_observed_ns.size() + 1);
  for (size_t i = 0; i < attrs.first_observed_ns.size(); ++i) {
    if (attrs.first_observed_ns[i] > attrs.last_observed_ns[i]) {
      throw std::invalid_argument(
          "Cannot normalize an existing inverted presence interval");
    }
    intervals.emplace_back(attrs.first_observed_ns[i],
                           attrs.last_observed_ns[i]);
  }
  intervals.emplace_back(t_start, t_end);
  std::sort(intervals.begin(), intervals.end());

  std::vector<Interval> reduced;
  reduced.reserve(intervals.size());
  for (const auto& interval : intervals) {
    if (reduced.empty() || interval.first > reduced.back().second) {
      reduced.push_back(interval);
    } else {
      // Overlapping and exactly touching intervals are one continuous physical
      // presence duration.
      reduced.back().second = std::max(reduced.back().second, interval.second);
    }
  }

  attrs.first_observed_ns.clear();
  attrs.last_observed_ns.clear();
  attrs.first_observed_ns.reserve(reduced.size());
  attrs.last_observed_ns.reserve(reduced.size());
  for (const auto& interval : reduced) {
    attrs.first_observed_ns.push_back(interval.first);
    attrs.last_observed_ns.push_back(interval.second);
  }
}

}  // namespace khronos
