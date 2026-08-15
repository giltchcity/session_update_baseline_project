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

#pragma once

#include <optional>
#include <vector>

#include <khronos/common/common_types.h>

namespace khronos {

inline constexpr char kObservationFirstStampDetail[] = "observation_first_stamp_ns";
inline constexpr char kObservationLastStampDetail[] = "observation_last_stamp_ns";

// Number of frames whose observations actually fed the current private
// reconstruction (written by MeshObjectExtractor). Physical-ID canonicalization
// gates geometric takeover on this support value: an established current mesh is
// never regressed by a weaker re-observation (Invariant 1).
inline constexpr char kReconstructionFramesDetail[] = "reconstruction_frames";

// Whether the segment carries tracker-measured motion evidence
// (displacement >= min_dynamic_displacement). A moved segment's reconstruction
// is the object's new current pose and legitimately takes over geometry
// (Invariant 2).
inline constexpr char kHasDynamicHistoryDetail[] = "has_dynamic_history";

Point computeSurfaceCentroid(const KhronosObjectAttributes& attrs);

/**
 * @brief Compose the canonical current scene geometry.
 *
 * Khronos deliberately excludes object-labelled surfaces from the background
 * TSDF. A complete current scene is therefore the background mesh plus every
 * currently materialized private object mesh, transformed from its bounding-box
 * frame into the world frame. Trajectory history does not suppress an object's
 * current mesh.
 */
spark_dsg::Mesh::Ptr composeCurrentSceneMesh(
    const DynamicSceneGraph& dsg,
    std::optional<TimeStamp> query_time = std::nullopt);

/**
 * @brief Whether the object carries a materialized mesh for its current pose.
 *
 * This is intentionally independent of trajectory metadata: a physical object
 * that moved and then settled has both a trajectory history and a current mesh.
 */
bool hasCurrentObjectMesh(const KhronosObjectAttributes& attrs);

/**
 * @brief Whether the object carries a valid trajectory history.
 */
bool hasTrajectoryHistory(const KhronosObjectAttributes& attrs);

/**
 * @brief Number of timestamp/position pairs safe to consume.
 */
size_t trajectoryHistorySize(const KhronosObjectAttributes& attrs);

/**
 * @brief First/last timestamps at which the object's current surface was
 * actually measured.
 *
 * These are deliberately distinct from first_observed_ns/last_observed_ns.
 * Reconciliation turns the latter into estimated presence intervals (possibly
 * ending at max()), while hidden-change detection in either the same session or
 * a later deserialized session must query evidence relative to the real sensor
 * observation bounds.
 */
TimeStamp observationFirstStamp(const KhronosObjectAttributes& attrs);
TimeStamp observationLastStamp(const KhronosObjectAttributes& attrs);

/** Persist the actual sensor bounds before presence reconciliation rewrites the
 * estimated presence interval. */
void persistObservationBounds(KhronosObjectAttributes& attrs);

/** Set the actual sensor bounds associated with the current materialization. */
void setObservationBounds(KhronosObjectAttributes& attrs,
                          TimeStamp first,
                          TimeStamp last);

/**
 * @brief Merge timestamped trajectory samples while preserving optional point
 * frames at the same indices.
 *
 * The resulting timestamps are sorted and unique. If any input contains
 * visualization point frames, the output vector is aligned one-to-one with the
 * timestamp/position vectors and missing frames are represented by empty point
 * sets.
 */
void mergeTrajectoryHistory(
    const std::vector<const KhronosObjectAttributes*>& sources,
    KhronosObjectAttributes& output);

/**
 * @brief Get the latest time the object appeared before query time, if such a time exists.
 * Assumes that first/last seen stamps are sorted.
 * @param attrs Object attributes to check.
 * @param query_time Time to check for.
 * @return The latest time the object appeared before query time, if such a time exists.
 */
std::optional<TimeStamp> lastAppearedBefore(const KhronosObjectAttributes& attrs,
                                            const TimeStamp query_time);

/**
 * @brief Get the latest time the object disappeared before query time, if such a time exists.
 * Assumes that first/last seen stamps are sorted.
 * @param attrs Object attributes to check.
 * @param query_time Time to check for.
 * @return The latest time the object disappeared before query time, if such a time exists.
 */
std::optional<TimeStamp> lastDisappearedBefore(const KhronosObjectAttributes& attrs,
                                               const TimeStamp query_time);

/**
 * @brief Check if object is considered present at query time. Assumes that first/last seen stamps
 * are sorted.
 * @param attrs Object attributes to check.
 * @param query_time Time to check for.
 * @return True if object is present at query time.
 */
bool isPresent(const KhronosObjectAttributes& attrs, const TimeStamp query_time);

/**
 * @brief Check if object has appeared at query time. Assumes that first/last seen stamps
 * are sorted. A first seen value of 0 assumes the object was always present and does not count as
 * appeared.
 * @param attrs Object attributes to check.
 * @param query_time Time to check for.
 * @return True if object has appeared before query time.
 */
bool hasAppeared(const KhronosObjectAttributes& attrs, const TimeStamp query_time);

/**
 * @brief Check if object has disappeared at query time. Assumes that first/last seen stamps
 * are sorted.
 * @param attrs Object attributes to check.
 * @param query_time Time to check for.
 * @return True if object has disappeared before query time.
 */
bool hasDisappeared(const KhronosObjectAttributes& attrs, const TimeStamp query_time);

/**
 * @brief Add a presence duration from t_start to t_end to the object attributes' presence time,
 * represented by paired last/first seen stamps. Existing intervals and the new
 * interval are validated, sorted, and unioned; touching intervals are treated
 * as continuous. Invalid or unpaired intervals throw std::invalid_argument.
 * @param attrs Object attributes to add presence duration to.
 * @param t_start Start of presence duration.
 * @param t_end End of presence duration.
 */
void addPresenceDuration(KhronosObjectAttributes& attrs,
                         const TimeStamp t_start,
                         const TimeStamp t_end);
}  // namespace khronos
