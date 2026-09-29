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

RayVerificator::PhysicalEvidenceSnapshot
RayVerificator::physicalEvidenceSnapshot() const {
  const auto store = std::atomic_load(&physical_evidence_store_);
  if (!store) {
    return std::nullopt;
  }
  return store->snapshot();
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
  return checkPhysicalImpl(point, physical_id, evidence_snapshot,
                           earliest, latest, details, false);
}

RayVerificator::CheckResult RayVerificator::checkPhysicalObserved(
    const Point& point, const size_t physical_id,
    const PhysicalEvidenceSnapshot& evidence_snapshot,
    const uint64_t earliest, const uint64_t latest) const {
  return checkPhysicalImpl(point, physical_id, evidence_snapshot,
                           earliest, latest, nullptr, false, true);
}

RayVerificator::CheckResult RayVerificator::checkPhysicalReplacement(
    const Point& point, const size_t physical_id,
    const PhysicalEvidenceSnapshot& evidence_snapshot,
    const uint64_t earliest, const uint64_t latest) const {
  return checkPhysicalImpl(point, physical_id, evidence_snapshot,
                           earliest, latest, nullptr, true);
}

RayVerificator::CheckResult RayVerificator::checkPhysicalImpl(
    const Point& point, const size_t physical_id,
    const PhysicalEvidenceSnapshot& evidence_snapshot,
    const uint64_t earliest, const uint64_t latest,
    CheckDetails* details, const bool measured_replacement, const bool require_observed) const {
  CheckResult result;

  if (!point.array().isFinite().all()) {
    // There is no meaningful block or ray projection for this query, and no
    // timestamp that could be attached to an inconclusive coverage vote.
    ++result.reasons.invalid;
    return result;
  }

  if (rays_.empty()) {
    CLOG(6) << "Physical point unobserved: no measurements.";
    return result;
  }

  const auto it = block_seen_by_rays_.find(grid_.toIndex(point));
  if (it == block_seen_by_rays_.end()) {
    CLOG(6) << "Physical point unobserved: no candidate measurements.";
    return result;
  }

  const RayLookup lookup(*dsg_, config);
  for (size_t ray_index : it->second) {
    const Ray& ray = rays_.at(ray_index);
    // A frozen store may also contain asynchronously ingested future frames.
    // Apply the query bounds before consulting endpoint identity.
    if (ray.timestamp < earliest || ray.timestamp > latest) {
      continue;
    }

    const Point source = lookup.getSource(ray);
    const Point vertex = lookup.getTarget(ray);
    const float depth = (point - source).norm();

    if (details) {
      details->start.emplace_back(source);
      details->end.emplace_back(vertex);
    }
    const auto record_invalid_geometry = [&]() {
      result.inconclusive.emplace_back(ray.timestamp);
      ++result.reasons.invalid;
      if (details) {
        details->range.emplace_back(0.f);
        details->result.emplace_back(CheckDetails::Result::kOccludded);
      }
    };
    if (!source.array().isFinite().all() ||
        !vertex.array().isFinite().all() || !std::isfinite(depth) ||
        depth <= std::numeric_limits<float>::epsilon()) {
      record_invalid_geometry();
      continue;
    }

    const Point direction = (point - source) / depth;
    const float radial_distance =
        (point - source).cross(source - vertex).norm() / depth;

    if (!std::isfinite(radial_distance)) {
      record_invalid_geometry();
      continue;
    }
    if (radial_distance > config.radial_tolerance) {
      ++result.reasons.no_overlap;
      if (details) {
        details->range.emplace_back(0.f);
        details->result.emplace_back(CheckDetails::Result::kNoOverlap);
      }
      continue;
    }

    const float depth_distance = (vertex - source).dot(direction);
    if (!std::isfinite(depth_distance)) {
      record_invalid_geometry();
      continue;
    }
    if (details) {
      details->range.emplace_back(depth_distance);
    }
    if (depth - depth_distance > config.depth_tolerance) {
      // A surface in front prevents observation of the old physical surface.
      // It is coverage, but never absence evidence for the hidden object.
      result.inconclusive.emplace_back(ray.timestamp);
      ++result.reasons.geometric_occlusion;
      if (details) {
        details->result.emplace_back(CheckDetails::Result::kOccludded);
      }
      continue;
    }

    // Project the old physical surface into the source frame. Object-labelled
    // pixels are not background-mesh endpoints, so classifying `vertex` here
    // would make same-ID support unreachable. The vertex depth still supplies
    // the independent near/through/occluded geometry above. Missing
    // session-local evidence is represented explicitly as kUnavailable.
    EndpointEvidence endpoint;
    if (evidence_snapshot) {
      endpoint = evidence_snapshot->classify(ray.timestamp, point);
    }
    // The mesh ray can end at a stale/nearby reconstructed surface even
    // when the exact source pixel now sees an occluder. Physical identity
    // checks must respect that measured depth, just as countPhysicalSurface
    // does, before a background pixel can vote for replacement.
    const bool measured_same_identity = endpoint.type == EndpointClass::kPhysical &&
        endpoint.physical_id > 0 && static_cast<size_t>(endpoint.physical_id) == physical_id;
    if ((measured_replacement || require_observed) && std::isfinite(endpoint.measured_depth_m) &&
        endpoint.measured_depth_m <
            depth - (measured_same_identity ? config.depth_tolerance : 1e-3f)) {
      result.inconclusive.emplace_back(ray.timestamp);
      ++result.reasons.geometric_occlusion;
      if (details) {
        details->result.emplace_back(CheckDetails::Result::kOccludded);
      }
      continue;
    }
    const bool ray_through =
        depth_distance - depth > config.depth_tolerance;

    const auto record_present = [&](size_t& reason) {
      result.present.emplace_back(ray.timestamp);
      ++reason;
      if (details) {
        details->result.emplace_back(CheckDetails::Result::kMatch);
      }
    };
    const auto record_absent = [&](size_t& reason) {
      result.absent.emplace_back(ray.timestamp);
      ++reason;
      if (details) {
        details->result.emplace_back(CheckDetails::Result::kAbsent);
      }
    };
    const auto record_inconclusive = [&](size_t& reason) {
      result.inconclusive.emplace_back(ray.timestamp);
      ++reason;
      if (details) {
        details->result.emplace_back(CheckDetails::Result::kOccludded);
      }
    };

    switch (endpoint.type) {
      case EndpointClass::kPhysical:
        if (endpoint.physical_id > 0 &&
            static_cast<size_t>(endpoint.physical_id) == physical_id) {
          if ((measured_replacement || require_observed) && std::isfinite(endpoint.measured_depth_m) &&
              endpoint.measured_depth_m > depth + config.depth_tolerance) {
            record_absent(result.reasons.free_space);
          } else {
            record_present(result.reasons.same_id);
          }
        } else if (measured_replacement &&
                   std::isfinite(endpoint.measured_depth_m)) {
          // The depth gate above has already excluded a nearer occluder.
          record_absent(result.reasons.different_id);
        } else {
          record_inconclusive(result.reasons.different_id);
        }
        break;
      case EndpointClass::kUnidentifiedObject:
        record_inconclusive(result.reasons.unidentified_object);
        break;
      case EndpointClass::kInvalid:
        record_inconclusive(result.reasons.invalid);
        break;
      case EndpointClass::kUnavailable:
        if (ray_through && !measured_replacement && !require_observed) {
          // Preserve the ordinary geometric free-space verdict when typed
          // endpoint data was not captured for an otherwise valid old ray.
          record_absent(result.reasons.unavailable);
        } else {
          record_inconclusive(result.reasons.unavailable);
        }
        break;
      case EndpointClass::kBackground:
        if (ray_through) {
          record_absent(result.reasons.free_space);
        } else {
          record_absent(result.reasons.background_replacement);
        }
        break;
    }
  }

  return result;
}


RayVerificator::SurfaceEvidenceCounts RayVerificator::countPhysicalSurface(
    const size_t physical_id,
    const spark_dsg::Mesh& mesh,
    const BoundingBox& bbox,
    const PhysicalEvidenceSnapshot& evidence_snapshot,
    const uint64_t earliest,
    const uint64_t latest) const {
  SurfaceEvidenceCounts result;
  const auto classify_one = [&](const Point& point) {
    ++result.surface_samples;
    if (!dsg_) {
      ++result.unobserved_samples;
      return;
    }
    const auto it = block_seen_by_rays_.find(grid_.toIndex(point));
    if (it == block_seen_by_rays_.end()) {
      ++result.unobserved_samples;
      return;
    }
    bool had_eligible_ray = false;
    const RayLookup lookup(*dsg_, config);
    for (const size_t ray_index : it->second) {
      const Ray& ray = rays_.at(ray_index);
      if (ray.timestamp < earliest || ray.timestamp > latest) {
        continue;
      }
      had_eligible_ray = true;
      const Point source = lookup.getSource(ray);
      const Point vertex = lookup.getTarget(ray);
      const float depth = (point - source).norm();
      if (!source.array().isFinite().all() ||
          !vertex.array().isFinite().all() || !std::isfinite(depth) ||
          depth <= std::numeric_limits<float>::epsilon()) {
        continue;
      }
      const Point direction = (point - source) / depth;
      const float radial_distance =
          (point - source).cross(source - vertex).norm() / depth;
      if (!std::isfinite(radial_distance) ||
          radial_distance > config.radial_tolerance) {
        continue;
      }
      const float depth_distance = (vertex - source).dot(direction);
      if (!std::isfinite(depth_distance)) {
        continue;
      }
      if (depth - depth_distance > config.depth_tolerance) {
        ++result.occluded_votes;
        continue;  // occluded by a nearer surface along the ray
      }

      EndpointEvidence endpoint;
      if (evidence_snapshot) {
        endpoint = evidence_snapshot->classify(ray.timestamp, point);
      }

      const auto add_support = [&]() {
        result.latest_support_stamp = std::max(result.latest_support_stamp, ray.timestamp);
        result.support_rays +=
            result.support_indices.insert(ray_index).second ? 1 : 0;
      };
      const auto add_contradiction = [&]() {
        result.contradiction_rays +=
            result.contradiction_indices.insert(ray_index).second ? 1 : 0;
      };

      const bool have_measured = std::isfinite(endpoint.measured_depth_m);
      // Measured endpoint is clearly in front of the old surface: occlusion,
      // not evidence of absence.
      const bool same_identity = endpoint.type == EndpointClass::kPhysical &&
          endpoint.physical_id > 0 && static_cast<size_t>(endpoint.physical_id) == physical_id;
      const bool occluded_by_depth = have_measured &&
          endpoint.measured_depth_m < depth - (same_identity ? config.depth_tolerance : 1e-3f);
      // Measured endpoint is at or behind the old surface: the old surface is
      // not there; either it was replaced by another object/background or the
      // ray passed through it.
      const bool absent_by_depth =
          have_measured &&
          endpoint.measured_depth_m >= depth - 1e-3f;

      switch (endpoint.type) {
        case EndpointClass::kPhysical:
          if (endpoint.physical_id > 0 &&
              static_cast<size_t>(endpoint.physical_id) == physical_id) {
            if (occluded_by_depth) {
              ++result.occluded_votes;
            } else if (have_measured &&
                       endpoint.measured_depth_m > depth + config.depth_tolerance) {
              // Seeing the same identity farther down the ray supports its
              // new position, not this vacated surface.
              add_contradiction();
              ++result.free_space_votes;
            } else if (have_measured) {
              add_support();
              ++result.supported_votes;
            }
          } else if (absent_by_depth) {
            // A different physical object occupies this old surface point.
            add_contradiction();
            ++result.replaced_by_other_votes;
          } else if (occluded_by_depth) {
            ++result.occluded_votes;
          }
          break;
        case EndpointClass::kUnidentifiedObject:
          // An unidentified object at the old surface depth could be the same
          // physical object whose tracking identity was lost; it is not
          // reliable replacement evidence and never votes for absence.
          if (occluded_by_depth) {
            ++result.occluded_votes;
          }
          break;
        case EndpointClass::kUnavailable:
          // A mesh ray is only a spatial candidate. Without an actual pixel
          // at this timestamp (outside FOV / missing frame), its geometry
          // cannot establish physical absence.
          break;
        case EndpointClass::kBackground:
          if (absent_by_depth) {
            add_contradiction();
            if (have_measured &&
                endpoint.measured_depth_m > depth + config.depth_tolerance) {
              ++result.free_space_votes;
            } else {
              ++result.replaced_by_background_votes;
            }
          } else if (occluded_by_depth) {
            ++result.occluded_votes;
          }
          break;
        default:
          break;
      }
    }
    if (!had_eligible_ray) {
      ++result.unobserved_samples;
    }
  };

  if (mesh.faces.empty()) {
    for (size_t i = 0; i < mesh.numVertices(); ++i) {
      classify_one(bbox.pointToWorldFrame(mesh.pos(i)));
    }
    result.absence_coverage_sufficient = result.contradiction_rays > result.support_rays;
    return result;
  }
  for (const auto& face : mesh.faces) {
    if (face[0] >= mesh.points.size() || face[1] >= mesh.points.size() ||
        face[2] >= mesh.points.size()) {
      continue;
    }
    const Point p0 = bbox.pointToWorldFrame(mesh.pos(face[0]));
    const Point p1 = bbox.pointToWorldFrame(mesh.pos(face[1]));
    const Point p2 = bbox.pointToWorldFrame(mesh.pos(face[2]));
    classify_one((p0 + p1 + p2) / 3.0f);
  }
  // Mesh-ray proxy alone (no observed-absence test ran): the counts decide, as
  // before. countCurrentPhysicalSurface replaces this with the observed-absence test.
  result.absence_coverage_sufficient = result.contradiction_rays > result.support_rays;
  return result;
}

RayVerificator::CheckResult RayVerificator::checkPhysicalSurface(
    const size_t physical_id,
    const spark_dsg::Mesh& mesh,
    const BoundingBox& bbox,
    const PhysicalEvidenceSnapshot& evidence_snapshot,
    const uint64_t earliest,
    const uint64_t latest,
    CheckDetails* details) const {
  CheckResult merged;
  const auto check_one = [&](const Point& point) {
    merged.merge(checkPhysical(point,
                               physical_id,
                               evidence_snapshot,
                               earliest,
                               latest,
                               details));
  };

  if (mesh.faces.empty()) {
    for (size_t i = 0; i < mesh.numVertices(); ++i) {
      check_one(bbox.pointToWorldFrame(mesh.pos(i)));
    }
    return merged;
  }

  // Sample the actual surface at every triangle centroid instead of at the
  // sparse vertex set. A reconstructed mesh is a sampling of a continuous
  // surface; vertices are the least representative points on it. The centroid
  // lies inside the observed face, so rays that pass through holes between
  // vertices no longer count as free-space evidence, while rays that cross a
  // reconstructed face are queried at a point that belongs to that face.
  for (const auto& face : mesh.faces) {
    if (face[0] >= mesh.points.size() || face[1] >= mesh.points.size() ||
        face[2] >= mesh.points.size()) {
      continue;
    }
    const Point p0 = bbox.pointToWorldFrame(mesh.pos(face[0]));
    const Point p1 = bbox.pointToWorldFrame(mesh.pos(face[1]));
    const Point p2 = bbox.pointToWorldFrame(mesh.pos(face[2]));
    check_one((p0 + p1 + p2) / 3.0f);
  }
  return merged;
}


}  // namespace khronos
