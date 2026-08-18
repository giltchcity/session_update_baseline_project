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

#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <tuple>

#include <glog/logging.h>

#include "khronos/utils/khronos_attribute_utils.h"

namespace khronos {

void declare_config(RayVerificator::Config& config) {
  using namespace config;
  name("RayVerificator");
  field(config.verbosity, "verbosity");
  field(config.block_size, "block_size", "m");
  field(config.radial_tolerance, "radial_tolerance", "m");
  field(config.depth_tolerance, "depth_tolerance", "m");
  enum_field(config.ray_policy,
             "ray_policy",
             {"First", "Last", "FirstAndLast", "Middle", "All", "Random", "Random3"});
  field(config.active_window_duration, "active_window_duration", "s");
  field(config.prefix, "prefix");

  check(config.block_size, GT, 0.f, "block_size");
  check(config.radial_tolerance, GT, 0.f, "radial_tolerance");
  check(config.depth_tolerance, GT, 0.f, "depth_tolerance");
}

RayVerificator::RayVerificator(const Config& config)
    : config(config::checkValid(config)), grid_(config.block_size), seed_(0) {}

RayVerificator::CheckResult RayVerificator::check(const Point& point,
                                                  const uint64_t earliest,
                                                  const uint64_t latest,
                                                  CheckDetails* details) const {
  CheckResult result;

  if (rays_.empty()) {
    CLOG(6) << "Point unobserved: no measurements.";
    return result;
  }

  // Lookup all candidate views that could have observed the point.
  const auto it = block_seen_by_rays_.find(grid_.toIndex(point));
  if (it == block_seen_by_rays_.end()) {
    CLOG(6) << "Point unobserved: no measurements_.";
    return result;
  }

  // TODO(lschmid): This is a bit inefficient, we could cache the lookup.
  const RayLookup lookup(*dsg_, config);

  // Check all candidate views.
  for (size_t ray_index : it->second) {
    const Ray& ray = rays_.at(ray_index);
    if (ray.timestamp < earliest || ray.timestamp > latest) {
      // This measurement is out of the temporal range to check.
      continue;
    }

    const Point source = lookup.getSource(ray);
    const Point direction = (point - source).normalized();
    const float depth = (point - source).norm();

    // Check all vertices representd as rays.
    const Point vertex = lookup.getTarget(ray);
    const float radial_distance = (point - source).cross(source - vertex).norm() / depth;

    if (details) {
      details->start.emplace_back(source);
      details->end.emplace_back(vertex);
    }
    if (radial_distance > config.radial_tolerance) {
      // No overlap on the ray.
      if (details) {
        details->range.emplace_back(0.f);
        details->result.emplace_back(CheckDetails::Result::kNoOverlap);
      }
      continue;
    }

    const float depth_distance = (vertex - source).dot(direction);
    if (details) {
      details->range.emplace_back(depth_distance);
    }
    if (depth - depth_distance > config.depth_tolerance) {
      // This is an occlusion, the point has not been observed.
      if (details) {
        details->result.emplace_back(CheckDetails::Result::kOccludded);
      }
      continue;
    }

    if (depth_distance - depth > config.depth_tolerance) {
      // This is a ray through our point, so it's evidence of absence.
      result.absent.emplace_back(ray.timestamp);
      if (details) {
        details->result.emplace_back(CheckDetails::Result::kAbsent);
      }
      continue;
    }

    // This means the point is within the tolerance, it's a match.
    result.present.emplace_back(ray.timestamp);
    if (details) {
      details->result.emplace_back(CheckDetails::Result::kMatch);
    }
  }

  return result;
}

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
          record_present(result.reasons.same_id);
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
        if (ray_through) {
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
      const bool ray_through =
          depth_distance - depth > config.depth_tolerance;

      const auto add_support = [&]() {
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
      const bool occluded_by_depth =
          have_measured &&
          endpoint.measured_depth_m < depth - config.depth_tolerance;
      // Measured endpoint is at or behind the old surface: the old surface is
      // not there; either it was replaced by another object/background or the
      // ray passed through it.
      const bool absent_by_depth =
          have_measured &&
          endpoint.measured_depth_m >= depth - config.depth_tolerance;

      switch (endpoint.type) {
        case EndpointClass::kPhysical:
          if (endpoint.physical_id > 0 &&
              static_cast<size_t>(endpoint.physical_id) == physical_id) {
            if (occluded_by_depth) {
              ++result.occluded_votes;
            } else {
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
          if ((!have_measured && ray_through) || absent_by_depth) {
            add_contradiction();
            ++result.free_space_votes;
          } else if (occluded_by_depth) {
            ++result.occluded_votes;
          }
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

bool RayVerificator::hasStablePrefix(const DynamicSceneGraph& dsg) const {
  return hasStablePosePrefix(dsg) && hasStableMeshPrefix(dsg) &&
         hasStableObjectSet(dsg);
}

bool RayVerificator::hasStablePosePrefix(const DynamicSceneGraph& dsg) const {
  const auto agents_key = dsg.getLayerKey(DsgLayers::AGENTS);
  const auto* agents =
      agents_key ? dsg.findLayer(agents_key->layer, config.prefix.key) : nullptr;
  if (!agents) {
    return indexed_poses_.empty();
  }
  if (agents->numNodes() < indexed_poses_.size()) {
    return false;
  }

  const uint64_t latest_indexed_stamp = timestamps_.empty() ? 0 : timestamps_.back();
  const NodeId latest_indexed_id = node_ids_.empty() ? 0 : node_ids_.back();
  for (const auto& [node_id, snapshot] : indexed_poses_) {
    const auto* node = agents->findNode(node_id);
    if (!node) {
      return false;
    }
    const auto& attrs = node->attributes<spark_dsg::AgentNodeAttributes>();
    if (snapshot.timestamp != static_cast<uint64_t>(attrs.timestamp.count()) ||
        (snapshot.position.array() != attrs.position.cast<float>().array()).any() ||
        (snapshot.orientation.coeffs().array() !=
         attrs.world_R_body.coeffs().array())
            .any()) {
      return false;
    }
  }

  // Appending a measurement older than the indexed suffix would invalidate
  // the source indices already stored in every ray. Treat it as a prefix
  // violation and rebuild with a freshly sorted pose sequence.
  std::vector<uint64_t> new_timestamps;
  for (const auto& [node_id, node] : agents->nodes()) {
    if (indexed_poses_.count(node_id)) {
      continue;
    }
    const auto stamp = static_cast<uint64_t>(
        node->attributes<spark_dsg::AgentNodeAttributes>().timestamp.count());
    if (!timestamps_.empty() &&
        std::tie(stamp, node_id) <
            std::tie(latest_indexed_stamp, latest_indexed_id)) {
      return false;
    }
    new_timestamps.emplace_back(stamp);
  }

  // A pose added inside an already indexed vertex's observation interval can
  // change kLast/kMiddle/kAll source selection (or give a previously rayless
  // vertex its first source) even though the mesh bytes are unchanged. This is
  // not an append-only ray prefix, so use the full oracle path.
  std::sort(new_timestamps.begin(), new_timestamps.end());
  if (!new_timestamps.empty()) {
    const uint64_t active_window_offset =
        config.active_window_duration > 0
            ? static_cast<uint64_t>(config.active_window_duration * 1e9)
            : 0;
    for (size_t i = 0; i < indexed_vertex_first_seen_.size(); ++i) {
      const uint64_t first_seen = indexed_vertex_first_seen_[i];
      const uint64_t raw_last_seen = indexed_vertex_last_seen_[i];
      const uint64_t last_seen = raw_last_seen > active_window_offset
                                     ? raw_last_seen - active_window_offset
                                     : 0;
      const auto candidate =
          std::lower_bound(new_timestamps.begin(), new_timestamps.end(), first_seen);
      if (candidate != new_timestamps.end() && *candidate <= last_seen) {
        return false;
      }
    }
  }
  return true;
}

bool RayVerificator::hasStableMeshPrefix(const DynamicSceneGraph& dsg) const {
  if (!dsg.hasMesh()) {
    return indexed_vertex_positions_.empty();
  }
  const auto& mesh = *dsg.mesh();
  if (mesh.numVertices() < indexed_vertex_positions_.size() ||
      mesh.first_seen_stamps.size() < indexed_vertex_first_seen_.size() ||
      mesh.stamps.size() < indexed_vertex_last_seen_.size()) {
    return false;
  }
  // A valid ray target has all three fields. Do not accept a partially
  // appended mesh as a stable prefix.
  if (mesh.numVertices() != mesh.first_seen_stamps.size() ||
      mesh.numVertices() != mesh.stamps.size()) {
    return false;
  }
  for (size_t i = 0; i < indexed_vertex_positions_.size(); ++i) {
    if ((indexed_vertex_positions_[i].array() != mesh.points[i].array()).any() ||
        indexed_vertex_first_seen_[i] != mesh.first_seen_stamps[i] ||
        indexed_vertex_last_seen_[i] != mesh.stamps[i]) {
      return false;
    }
  }
  return true;
}

bool RayVerificator::hasStableObjectSet(const DynamicSceneGraph& dsg) const {
  if (indexed_objects_.empty()) {
    return true;
  }
  if (!dsg.hasLayer(DsgLayers::OBJECTS)) {
    return false;
  }
  const auto& objects = dsg.getLayer(DsgLayers::OBJECTS).nodes();
  if (objects.size() < indexed_objects_.size()) {
    return false;
  }
  for (const auto& [node_id, _] : indexed_objects_) {
    if (!objects.count(node_id)) {
      return false;
    }
  }
  return true;
}

RayVerificator::UpdateMode RayVerificator::setDsg(
    std::shared_ptr<const DynamicSceneGraph> dsg) {
  const auto start = std::chrono::steady_clock::now();
  const bool can_reuse = dsg_ && dsg && hasStablePrefix(*dsg);
  statistics_.last_prefix_check_us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start)
          .count());

  if (can_reuse) {
    dsg_ = std::move(dsg);
    ++statistics_.incremental_rebinds;
    statistics_.last_mode = UpdateMode::kIncremental;
    return UpdateMode::kIncremental;
  }

  resetState(std::move(dsg), dsg_ != nullptr);
  return UpdateMode::kFullReset;
}

void RayVerificator::resetState(std::shared_ptr<const DynamicSceneGraph> dsg,
                                bool rejected_prefix) {
  const auto start = std::chrono::steady_clock::now();
  dsg_ = std::move(dsg);
  rays_.clear();
  timestamps_.clear();
  node_ids_.clear();
  block_seen_by_rays_.clear();
  vertices_in_block_.clear();
  objects_in_block_.clear();
  indexed_poses_.clear();
  indexed_objects_.clear();
  object_blocks_.clear();
  indexed_vertex_positions_.clear();
  indexed_vertex_first_seen_.clear();
  indexed_vertex_last_seen_.clear();
  // Random policies must produce the same rays after a conservative reset as
  // an uninterrupted incremental run. A deterministic seed makes the full
  // path a valid oracle instead of changing semantics on every rebuild.
  seed_ = 0;
  previous_vertex_index_ = 0;
  reobserved_vertices_.clear();
  reobserved_objects_.clear();

  statistics_.last_new_poses = addPoseNodes();
  const size_t previous_rays = rays_.size();
  addVertices();
  statistics_.last_new_vertices = previous_vertex_index_;
  statistics_.last_new_rays = rays_.size() - previous_rays;
  rebuildObjectsInHash();

  ++statistics_.full_resets;
  if (rejected_prefix) {
    ++statistics_.rejected_prefixes;
  }
  statistics_.last_mode = UpdateMode::kFullReset;
  statistics_.indexed_poses = indexed_poses_.size();
  statistics_.indexed_vertices = previous_vertex_index_;
  statistics_.indexed_objects = indexed_objects_.size();
  statistics_.rays = rays_.size();
  statistics_.last_reobserved_vertices = 0;
  statistics_.last_reobserved_objects = 0;
  statistics_.last_update_us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
}

RayVerificator::UpdateMode RayVerificator::updateDsg() {
  const auto start = std::chrono::steady_clock::now();
  if (!dsg_) {
    return UpdateMode::kFullReset;
  }

  const auto prefix_start = std::chrono::steady_clock::now();
  const bool stable_prefix = hasStablePrefix(*dsg_);
  statistics_.last_prefix_check_us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - prefix_start)
          .count());
  if (!stable_prefix) {
    resetState(dsg_, true);
    return UpdateMode::kFullReset;
  }

  // Add only the new parts to the measurements and hash.
  const size_t previous_rays = rays_.size();
  statistics_.last_new_poses = addPoseNodes();
  const size_t previous_vertices = previous_vertex_index_;
  BlockIndexSet observed_blocks = addVertices();
  statistics_.last_new_vertices = previous_vertex_index_ - previous_vertices;
  statistics_.last_new_rays = rays_.size() - previous_rays;

  // Compute the newly re-observed vertices and objects.
  reobserved_vertices_.clear();
  reobserved_objects_.clear();
  updateObjectsInHash();
  for (const auto& index : observed_blocks) {
    const auto it = vertices_in_block_.find(index);
    if (it != vertices_in_block_.end()) {
      reobserved_vertices_.insert(it->second.begin(), it->second.end());
    }
    const auto it2 = objects_in_block_.find(index);
    if (it2 != objects_in_block_.end()) {
      reobserved_objects_.insert(it2->second.begin(), it2->second.end());
    }
  }

  ++statistics_.incremental_updates;
  statistics_.last_mode = UpdateMode::kIncremental;
  statistics_.indexed_poses = indexed_poses_.size();
  statistics_.indexed_vertices = previous_vertex_index_;
  statistics_.indexed_objects = indexed_objects_.size();
  statistics_.rays = rays_.size();
  statistics_.last_reobserved_vertices = reobserved_vertices_.size();
  statistics_.last_reobserved_objects = reobserved_objects_.size();
  statistics_.last_update_us = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
  return UpdateMode::kIncremental;
}

void RayVerificator::forceReset() { resetState(dsg_, false); }

size_t RayVerificator::addPoseNodes() {
  if (!dsg_) {
    return 0;
  }

  const auto agents_key = dsg_->getLayerKey(DsgLayers::AGENTS);
  if (!agents_key) {
    return 0;
  }
  const auto agents = dsg_->findLayer(agents_key->layer, config.prefix.key);
  if (!agents) {
    return 0;
  }

  using PendingPose = std::tuple<uint64_t, NodeId, PoseSnapshot>;
  std::vector<PendingPose> pending;
  const auto& nodes = agents->nodes();
  for (const auto& [node_id, node] : nodes) {
    if (indexed_poses_.count(node_id)) {
      continue;
    }

    const auto& attrs = node->attributes<spark_dsg::AgentNodeAttributes>();
    PoseSnapshot snapshot;
    snapshot.timestamp = static_cast<uint64_t>(attrs.timestamp.count());
    snapshot.position = attrs.position.cast<float>();
    snapshot.orientation = attrs.world_R_body;
    pending.emplace_back(snapshot.timestamp, node_id, snapshot);
  }

  // computeVertexSources() uses lower_bound, so this ordering is a hard
  // invariant rather than an assumption about SceneGraphLayer iteration.
  std::sort(pending.begin(), pending.end(), [](const auto& lhs, const auto& rhs) {
    return std::tie(std::get<0>(lhs), std::get<1>(lhs)) <
           std::tie(std::get<0>(rhs), std::get<1>(rhs));
  });
  for (const auto& [timestamp, node_id, snapshot] : pending) {
    timestamps_.emplace_back(timestamp);
    node_ids_.emplace_back(node_id);
    indexed_poses_.emplace(node_id, snapshot);
  }
  return pending.size();
}

BlockIndexSet RayVerificator::addVertices() {
  BlockIndexSet observed_blocks;

  // Check if DSG and mesh exist and have vertices
  if (!dsg_ || !dsg_->hasMesh() || dsg_->mesh()->numVertices() == 0) {
    return observed_blocks;
  }

  // For all vertices, compute the sources they belong to and add them to the library of rays.
  const auto& vertices = dsg_->mesh()->points;
  const auto& first_seen = dsg_->mesh()->first_seen_stamps;
  auto last_seen = dsg_->mesh()->stamps;

  // Ensure all mesh arrays have consistent sizes
  if (vertices.size() != first_seen.size() || vertices.size() != last_seen.size()) {
    LOG(WARNING) << "Mesh arrays have inconsistent sizes: vertices=" << vertices.size()
                 << ", first_seen=" << first_seen.size()
                 << ", last_seen=" << last_seen.size();
    return observed_blocks;
  }
  if (config.active_window_duration > 0) {
    const uint64_t offset_ns = config.active_window_duration * 1e9;
    for (auto& stamp : last_seen) {
      stamp = stamp > offset_ns ? stamp - offset_ns : 0;
    }
  }

  for (size_t i = previous_vertex_index_; i < vertices.size(); ++i) {
    // Add the vertex to the hash.
    vertices_in_block_[grid_.toIndex(vertices.at(i))].insert(i);

    // Keep an immutable copy of precisely the fields that define a ray target.
    // The next cloned DSG can then prove that this entire prefix is unchanged.
    indexed_vertex_positions_.emplace_back(vertices.at(i));
    indexed_vertex_first_seen_.emplace_back(first_seen.at(i));
    indexed_vertex_last_seen_.emplace_back(dsg_->mesh()->stamps.at(i));

    // Compute which measurements this vertex belongs to.
    const auto source_indices = computeVertexSources(first_seen.at(i), last_seen.at(i));
    if (source_indices.empty()) {
      // A vertex with no pose that actually observed it simply produces no measurement ray. It is
      // still a valid change-detection *target*: the evidence about it comes from other rays that
      // pass through its position (addRayToHash registers a ray in every block along its path), not
      // from a ray of its own. Pairing it with a pose that never saw it would fabricate a
      // measurement.
      continue;
    }

    // Create the rays and add them to the hash.
    observed_blocks.merge(emitVertexRays(i, source_indices));
  }
  previous_vertex_index_ = vertices.size();

  return observed_blocks;
}

BlockIndexSet RayVerificator::emitVertexRays(const size_t vertex_index,
                                             const std::unordered_set<size_t>& source_indices) {
  BlockIndexSet observed_blocks;
  for (const size_t source_index : source_indices) {
    if (source_index >= timestamps_.size() || source_index >= node_ids_.size()) {
      LOG(WARNING) << "Invalid source index: " << source_index
                   << " (timestamps size=" << timestamps_.size()
                   << ", node_ids size=" << node_ids_.size() << ")";
      continue;
    }
    rays_.emplace_back(
        timestamps_.at(source_index), node_ids_.at(source_index), vertex_index);
    observed_blocks.merge(addRayToHash(rays_.size() - 1));
  }
  return observed_blocks;
}

std::unordered_set<size_t> RayVerificator::computeVertexSources(const size_t first_seen,
                                                                const size_t last_seen) {
  std::unordered_set<size_t> result;

  // A mesh vertex may only create a measurement ray from a pose that actually falls inside the
  // interval in which that vertex was observed.  This is particularly important at a session
  // boundary: prior-session vertices are deliberately retained as change-detection *targets*, but
  // the new session does not contain their old camera poses.  Choosing the nearest new-session pose
  // for such a vertex fabricates a measurement and can make old geometry validate or delete itself.
  if (timestamps_.empty() || last_seen < first_seen) {
    return result;
  }

  const auto first = std::lower_bound(timestamps_.begin(), timestamps_.end(), first_seen);
  const auto after_last = std::upper_bound(timestamps_.begin(), timestamps_.end(), last_seen);
  if (first == timestamps_.end() || first >= after_last) {
    return result;
  }

  // Compute which source points (indicated by timestamps) are relevant for this vertex.
  if (config.ray_policy == Config::RayPolicy::kFirst ||
      config.ray_policy == Config::RayPolicy::kFirstAndLast) {
    result.insert(first - timestamps_.begin());
  }
  if (config.ray_policy == Config::RayPolicy::kLast ||
      config.ray_policy == Config::RayPolicy::kFirstAndLast) {
    result.insert(std::prev(after_last) - timestamps_.begin());
  }
  if (config.ray_policy == Config::RayPolicy::kMiddle) {
    const size_t stamp = first_seen + (last_seen - first_seen) / 2;
    auto candidate = std::lower_bound(first, after_last, stamp);
    if (candidate == after_last) {
      candidate = std::prev(after_last);
    } else if (candidate != first) {
      const auto previous = std::prev(candidate);
      if (stamp - *previous <= *candidate - stamp) {
        candidate = previous;
      }
    }
    result.insert(candidate - timestamps_.begin());
  }
  if (config.ray_policy == Config::RayPolicy::kAll) {
    for (auto it = first; it != after_last; ++it) {
      result.insert(it - timestamps_.begin());
    }
  }
  if (config.ray_policy == Config::RayPolicy::kRandom ||
      config.ray_policy == Config::RayPolicy::kRandom3) {
    const size_t range = std::distance(first, after_last);
    const size_t start = std::distance(timestamps_.begin(), first);
    if (range > 0) {
      for (size_t i = 0; i < (config.ray_policy == Config::RayPolicy::kRandom3 ? 3 : 1); ++i) {
        size_t index = start + rand_r(&seed_) % range;
        if (index < timestamps_.size()) {
          result.insert(index);
        }
      }
    }
  }
  return result;
}

void RayVerificator::recomputeHash() {
  // Populate the block hash for lookup of poses, vertices, and objects later.
  block_seen_by_rays_.clear();
  vertices_in_block_.clear();
  objects_in_block_.clear();
  if (dsg_ && dsg_->hasMesh()) {
    for (size_t i = 0; i < dsg_->mesh()->numVertices(); ++i) {
      vertices_in_block_[grid_.toIndex(dsg_->mesh()->pos(i))].insert(i);
    }
  }
  for (size_t i = 0; i < rays_.size(); ++i) {
    addRayToHash(i);
  }
  rebuildObjectsInHash();
  statistics_.indexed_vertices = dsg_ && dsg_->hasMesh()
                                     ? dsg_->mesh()->numVertices()
                                     : 0;
  statistics_.indexed_objects = indexed_objects_.size();
  statistics_.rays = rays_.size();
}

BlockIndexSet RayVerificator::addRayToHash(const size_t ray_index) {
  BlockIndexSet observed_blocks;
  const Ray& ray = rays_.at(ray_index);
  // TODO(lschmid): This is a bit inefficient, we could cache the lookup.
  const RayLookup lookup(*dsg_, config);
  const Point source = lookup.getSource(ray);
  const Point target = lookup.getTarget(ray);
  const float max_depth = (target - source).norm();
  if (max_depth <= std::numeric_limits<float>::epsilon()) {
    const BlockIndex index = grid_.toIndex(target);
    block_seen_by_rays_[index].insert(ray_index);
    observed_blocks.insert(index);
    return observed_blocks;
  }
  const Point direction = (target - source) / max_depth;
  const float ray_step = config.block_size / 4;
  float ray_distance = 0.f;

  // NOTE(lschmid): This ray marching may miss some corner case blocks but that's fine, this is a
  // preliminary implementation.
  while (ray_distance <= max_depth) {
    ray_distance += ray_step;
    const Point ray_point = source + ray_distance * direction;
    const BlockIndex index = grid_.toIndex(ray_point);
    block_seen_by_rays_[index].insert(ray_index);
    observed_blocks.insert(index);
  }
  return observed_blocks;
}

RayVerificator::ObjectSnapshot RayVerificator::makeObjectSnapshot(
    const KhronosObjectAttributes& attrs) const {
  ObjectSnapshot result;
  result.points.assign(attrs.mesh.points.begin(), attrs.mesh.points.end());
  result.center = attrs.bounding_box.world_P_center.cast<float>();
  const auto instance_iter = attrs.details.find("instance_id");
  if (instance_iter != attrs.details.end() && instance_iter->second.size() == 1 &&
      instance_iter->second.front() != 0) {
    result.instance_id = instance_iter->second.front();
  }
  result.first_seen.assign(attrs.mesh.first_seen_stamps.begin(),
                           attrs.mesh.first_seen_stamps.end());
  result.last_seen.assign(attrs.mesh.stamps.begin(), attrs.mesh.stamps.end());
  result.first_observed = attrs.first_observed_ns;
  result.last_observed = attrs.last_observed_ns;
  result.observation_first = observationFirstStamp(attrs);
  result.observation_last = observationLastStamp(attrs);
  return result;
}

bool RayVerificator::objectSnapshotsEqual(const ObjectSnapshot& lhs,
                                          const ObjectSnapshot& rhs) {
  if ((lhs.center.array() != rhs.center.array()).any() ||
      lhs.points.size() != rhs.points.size() ||
      lhs.instance_id != rhs.instance_id ||
      lhs.first_seen != rhs.first_seen || lhs.last_seen != rhs.last_seen ||
      lhs.first_observed != rhs.first_observed ||
      lhs.last_observed != rhs.last_observed ||
      lhs.observation_first != rhs.observation_first ||
      lhs.observation_last != rhs.observation_last) {
    return false;
  }
  for (size_t i = 0; i < lhs.points.size(); ++i) {
    if ((lhs.points[i].array() != rhs.points[i].array()).any()) {
      return false;
    }
  }
  return true;
}

BlockIndexSet RayVerificator::computeObjectBlocks(
    const KhronosObjectAttributes& attrs) const {
  BlockIndexSet result;
  const Point center = attrs.bounding_box.world_P_center.cast<float>();
  for (const auto& vertex : attrs.mesh.points) {
    // Private object vertices are local to their bounding-box center. This is
    // the same world-space point queried by RayObjectChangeDetector.
    result.insert(grid_.toIndex(vertex + center));
  }
  return result;
}

void RayVerificator::updateObjectsInHash() {
  if (!dsg_ || !dsg_->hasLayer(DsgLayers::OBJECTS)) {
    return;
  }

  for (const auto& [id, node] : dsg_->getLayer(DsgLayers::OBJECTS).nodes()) {
    const auto& attrs = node->attributes<KhronosObjectAttributes>();
    ObjectSnapshot snapshot = makeObjectSnapshot(attrs);
    const auto previous = indexed_objects_.find(id);
    if (previous != indexed_objects_.end() &&
        objectSnapshotsEqual(previous->second, snapshot)) {
      continue;
    }

    if (previous != indexed_objects_.end()) {
      // Existing object geometry or observation bounds changed. Remove the old
      // spatial membership and force its change record to be recomputed.
      const auto blocks_it = object_blocks_.find(id);
      if (blocks_it != object_blocks_.end()) {
        for (const auto& index : blocks_it->second) {
          auto hash_it = objects_in_block_.find(index);
          if (hash_it == objects_in_block_.end()) {
            continue;
          }
          hash_it->second.erase(static_cast<size_t>(id));
          if (hash_it->second.empty()) {
            objects_in_block_.erase(hash_it);
          }
        }
      }
      reobserved_objects_.insert(static_cast<size_t>(id));
    }

    BlockIndexSet block_indices = computeObjectBlocks(attrs);
    for (const auto& index : block_indices) {
      objects_in_block_[index].insert(static_cast<size_t>(id));
    }
    object_blocks_[id] = std::move(block_indices);
    indexed_objects_[id] = std::move(snapshot);
  }
}

void RayVerificator::rebuildObjectsInHash() {
  objects_in_block_.clear();
  object_blocks_.clear();
  indexed_objects_.clear();
  if (!dsg_ || !dsg_->hasLayer(DsgLayers::OBJECTS)) {
    return;
  }
  for (const auto& [id, node] : dsg_->getLayer(DsgLayers::OBJECTS).nodes()) {
    const auto& attrs = node->attributes<KhronosObjectAttributes>();
    BlockIndexSet block_indices = computeObjectBlocks(attrs);
    for (const auto& index : block_indices) {
      objects_in_block_[index].insert(static_cast<size_t>(id));
    }
    object_blocks_[id] = std::move(block_indices);
    indexed_objects_[id] = makeObjectSnapshot(attrs);
  }
}

}  // namespace khronos
