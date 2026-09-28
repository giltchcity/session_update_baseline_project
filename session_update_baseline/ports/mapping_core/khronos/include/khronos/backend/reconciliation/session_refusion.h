#pragma once

#include <array>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "khronos/backend/reconciliation/frame_archive.h"
#include "khronos/common/common_types.h"

namespace khronos {

/**
 * @brief Session-end update of the final map from this session's own frames.
 *
 * One principle: the final map shows what this session measured, where and
 * when it measured it; memory (the previous session's final map) speaks only
 * where this session's evidence does not contradict it; which object a surface
 * belongs to, and where objects are, is the object reasoning's.
 *
 * Evidence. Every processed frame (range, pose, physical instance id). A frame
 * is evidence only for the state it saw: for an object whose current state
 * began within this session at t_L (the object reasoning's first sighting of
 * that state), frames before t_L lose the object's own pixels and the free
 * space they saw through the object's current surface (its online mesh and
 * every voxel its own pixels measured from t_L on).
 *
 * A surface element (half a voxel h of its layer) is known along a ray at range
 * q to tau = max(h, sigma(q)), sigma the sensor's depth noise measured on the
 * present. A frame hits it (a reading's point inside its ball of radius tau),
 * sees through it (every pixel of its footprint, radius focal * tau / z, reads
 * beyond it by more than tau, with free space the evidence counts) or is
 * blocked in front of it (within its layer's truncation: blocked_band).
 *
 * The final map:
 *  - present: one TSDF of the evidence at the object resolution (Open3D
 *    ScalableTSDFVolume semantics, see PresentTsdf), marching cubes; where it
 *    extracts no surface but a frame measured the session's own online surface
 *    (e.g. grazing angles), that surface stays;
 *  - identity: a present face belongs to the object whose surface in the final
 *    map (the object reasoning's geometry) it re-measures -- the current object
 *    mesh within one voxel of it -- otherwise to the background;
 *  - memory: a face of the previous final map stays unless this session's
 *    evidence places its surface elsewhere: the frames see through it more
 *    often than they hit it; or none hits or sees through it and most of its
 *    blocked views are blocked within the truncation band (no second surface
 *    that close behind the observed one); or it is a displaced copy of the
 *    present -- a present surface on the viewing side, farther than one voxel
 *    of its layer but within the position error the sessions' measured depth
 *    scales explain, tau + (s + s_prev) q at the nearest range q a frame
 *    reached it; or it lies inside the same object's present surface (INSIDE).
 *    An object's memory shows only while its node is current and its state did
 *    not begin in this session.
 * A node never ends empty. Object identities, states, boxes, presence and every
 * other snapshot are untouched; the map is edited only once everything is
 * computed.
 */
class SessionRefusion {
 public:
  struct Config {
    int num_threads = 4;
    // Depth noise estimator settings.
    double range_bin = 0.5;
    size_t num_bins = 16;
    size_t min_bin_samples = 1000;
    double histogram_resolution = 0.0005;
  };

  // The map's own resolutions (from the active window config).
  struct Scales {
    float background_voxel = 0.f;
    float background_truncation = 0.f;
    float object_voxel = 0.f;
    float object_min_voxel = 0.f;
  };

  // A map's surface in world coordinates with the physical id of every face
  // (0 = background).
  struct Surface {
    std::vector<Eigen::Vector3f> vertices;
    std::vector<std::array<uint32_t, 3>> faces;
    std::vector<uint32_t> face_physical;
  };

  struct Inputs {
    const std::vector<FrameArchive::Frame>* frames = nullptr;
    FrameArchive::Camera camera;
    Scales scales;
    // Physical id -> start stamp of its current state, for every object whose
    // current state began within this session.
    std::map<size_t, TimeStamp> state_starts;
    // Whether a point of the final map is memory (the loaded state).
    std::function<bool(const Eigen::Vector3f&)> is_memory;
    TimeStamp final_stamp = 0;
    std::string dump_dir;  // optional diagnostics
    // Memory as the previous session's final map showed it (optional).
    const Surface* shown = nullptr;
    // The measured depth scale of every earlier session (see Result).
    std::vector<float> previous_depth_scales;
  };

  struct Result {
    bool applied = false;
    // This session's depth scale: every reading scaled by (1 + s) makes the
    // session's frames agree best (exact depth: s = 0).
    float depth_scale = 0.f;
    std::string summary;
    std::string report_json;
  };

  explicit SessionRefusion(const Config& config) : config(config) {}

  Result apply(DynamicSceneGraph& dsg, const Inputs& inputs) const;

  const Config config;
};

}  // namespace khronos
