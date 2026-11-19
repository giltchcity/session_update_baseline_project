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
#include <set>
#include <mutex>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <config_utilities/config_utilities.h>
#include <hydra/common/global_info.h>
#include <hydra/reconstruction/mesh_integrator.h>

#include "khronos/active_window/integration/object_integrator.h"
#include "khronos/active_window/object_extraction/move_trajectories.h"
#include "khronos/active_window/object_extraction/object_extractor.h"

namespace khronos {

/**
 * @brief An object extractor that performs mesh-based 3D reconstruction of objects to extract.
 */
class MeshObjectExtractor : public ObjectExtractor {
 public:
  struct Config {
    int verbosity = hydra::GlobalInfo::instance().getConfig().default_verbosity;

    // Extract only objects with an existence confidence larger than this value [0, 1].
    float min_object_allocation_confidence = 0.5f;

    // Extract only objects with a volume larger than this value in m3.
    float min_object_volume = 0.1f;

    // Extract only objects with a volume smaller than this value in m3.
    float max_object_volume = 4.0f;

    // Only extract objects if the reconstruction is not empty.
    bool only_extract_reconstructed_objects = false;

    // Dynamic objects are only valid if trajectory is longer than specified length.
    float min_dynamic_displacement = 0.2f;

    // Accept tracks carrying a configured dynamic semantic label even when their
    // observed displacement is below min_dynamic_displacement.
    bool accept_semantic_dynamic_tracks = false;

    // For a physical object that moved and then settled, preserve its D1
    // trajectory as metadata while materializing a static mesh at the settled
    // current pose.
    bool preserve_settled_dynamic_history = true;

    // The trajectory of a physical object's move, written to its node where the hand-over finds
    // the end of the move (MoveTrajectories): an output only.
    MoveTrajectories::Config move_trajectory;

    // Geometric compatibility is independent of the D1 trajectory threshold.
    // A newer surface observed as free in an older RGB-D frame (or vice versa)
    // starts a separate reconstruction state. Unknown/occluded pixels do not.
    float static_consistency_tolerance = 0.05f;
    int static_consistency_min_pixels = 20;

    // Only add vertices with a confidence larger than this to the object
    // reconstruction.
    float min_object_reconstruction_confidence = 0.5f;

    // Only add vertices with more observations than this to the object reconstruction.
    int min_object_reconstruction_observations = 10;

    // Resolution at which extracted objects are reconstructed. Positive values
    // indicate the voxel size in meters. Negative values indicate a fraction of
    // the extent. A value of 0 skips reconstruction.
    float object_reconstruction_resolution = -0.02f;

    // Minimum limit to override small reconstruction resolutions computed
    // from fraction of extent. Has no effect when `object_reconstruction_resolution`
    // is positive.
    float min_reconstruction_resolution = 0.0f;

    // If true, don't prune the mesh to only the relevant parts but isntead color it by confidence
    // for debugging.
    bool visualize_classification = false;

    hydra::ProjectiveIntegrator::Config projective_integrator;
    hydra::MeshIntegratorConfig mesh_integrator;
  } const config;

  // Construction.
  explicit MeshObjectExtractor(const Config& config);
  virtual ~MeshObjectExtractor() = default;

  KhronosObjectAttributes::Ptr extractObject(const Track& track,
                                             const FrameDataBuffer& frame_data) override;

  bool integrateFrame(const FrameData::Ptr& data,
                      const MeasurementCluster& cluster,
                      Track& track,
                      const FrameDataBuffer& earlier_frames) override;
  // One frame's reading of an object into a segment's volume.
  void fuseInto(Track::Segment& target, FrameData& data, int cluster_id, const BoundingBox& box);
  // The volume a segment's frames are integrated in: an absolute voxel size, so the segments of
  // one object share a grid.
  static hydra::VolumetricMap::Config segmentVolume(float voxel_size);
  // The volume an object is integrated in as its frames arrive: that grid, with the map's
  // configured truncation distance (setFrameTruncation). The band of two voxels is too narrow
  // for a surface seen at a slant: its far voxels get no reading and the surface comes out
  // short of where it is. (The buffered reconstruction the object reasoning gets keeps its own.)
  hydra::VolumetricMap::Config frameVolume() const;
  void setFrameTruncation(float truncation_distance) override { frame_truncation_ = truncation_distance; }

  KhronosObjectAttributes::Ptr reconstructVoted(
      const Track& track,
      const std::vector<std::pair<FrameData::Ptr, int>>& frames,
      std::shared_ptr<hydra::VolumetricMap>& rebuilt) const;
  bool knownObject(int instance) const override;

  /**
   * @brief Extract a dynamic object from the given track.
   * @param track The track representing the object.
   * @returns The extracted object attributes if valid, nullptr otherwise.
   */
  KhronosObjectAttributes::Ptr extractDynamicObject(const Track& track) const;

  void appendDynamicHistory(const Track& track, KhronosObjectAttributes& object) const;

  /**
   * @brief Append the track's motion history (the DynamicCluster measured in each of its
   * observations, see Observation::dynamic_centroid) to the object's trajectory.
   */
  void appendTrajectory(const Track& track, KhronosObjectAttributes& object) const;

  /**
   * @brief Compute the largest displacement from the first motion sample.
   */
  static float computeDynamicDisplacement(const Track& track);

  /**
   * @brief Extract a static object from the given track.
   * @param track The track representing the object.
   * @param frame_data The frame data buffer to extract the object from.
   * @returns The extracted object attributes if valid, nullptr otherwise.
   */
  KhronosObjectAttributes::Ptr extractStaticObject(const Track& track,
                                                   const FrameDataBuffer& frame_datam) const;
  /**
   * @brief Check if a track meets the minimum criteria for object extraction.
   */
  bool trackIsValid(const Track& track) const;

  /**
   * @brief Get a human readable name for the given track for printing.
   */
  static std::string getTrackName(const Track& track);

  /**
   * @brief Collect all frames that have a semantic measurement for the given track.
   * @param track The track to collect frames for.
   * @param frame_data The frame data buffer to collect frames from.
   * @returns A list of frames and the corresponding semantic cluster ID.
   */
  static std::vector<std::pair<FrameData::Ptr, int>> collectSemanticFrames(
      const Track& track,
      const FrameDataBuffer& frame_data,
      std::optional<TimeStamp> after_stamp = std::nullopt);

  // Retain the latest geometrically compatible observation segment. This
  // method does not assign D1 labels or alter physical identity/history.
  std::vector<std::pair<FrameData::Ptr, int>> selectStaticFrames(
      const Track& track, const FrameDataBuffer& frame_data,
      std::optional<TimeStamp> after_stamp = std::nullopt, bool report = true,
      TimeStamp* rejected_stamp = nullptr, bool learn = true) const;

  /**
   * @brief Compute tje maximal spatial extent covered by all frames.
   * @param frames  A list of frames and the corresponding semantic cluster ID to get the extent
   * for.
   */
  BoundingBox computeExtent(const std::vector<std::pair<FrameData::Ptr, int>>& frames) const;

  /**
   * @brief Compute the confidence of a voxel belonging to the object of interest.
   */
  virtual float computeConfidence(const TsdfVoxel& tsdf_voxel,
                                  const hydra::SemanticVoxel& confidence_voxel) const;

 private:
  // The instances the object tests told to be objects.
  mutable std::mutex judged_mutex_;
  mutable std::set<int> judged_objects_;
  float frame_truncation_ = 0.f;  // 0: the segment volume's own

  // Project physical-track policy: session_core/src/adapters/static_surface_selection.cpp.
  std::optional<Track> preparePhysicalTrack(const Track& track) const;

  hydra::MeshIntegrator mesh_integrator_;
  // Integrate arriving frames into the tracks' segments (the active window's thread only): a
  // segment of few blocks on the calling thread, a larger one on the configured threads.
  ObjectIntegrator frame_integrator_;
  ObjectIntegrator small_frame_integrator_;

  inline static const auto registration_ =
      config::RegistrationWithConfig<ObjectExtractor,
                                     MeshObjectExtractor,
                                     MeshObjectExtractor::Config>("MeshObjectExtractor");
};

void declare_config(MeshObjectExtractor::Config& config);

}  // namespace khronos
