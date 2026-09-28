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
 * @brief Session-end update of the final map: the surface of the evidence of
 * the end-of-session scene, memory where that evidence does not place memory's
 * surface elsewhere, each surface with the object whose current state it
 * measures.
 *
 * 1. Valid evidence. Every processed frame (range, pose, physical instance id).
 *    A reading is evidence only where the scene has not changed since it was
 *    taken; the changes are the object reasoning's: an object whose current
 *    state began within this session at t_L (its first sighting) left the place
 *    its own pixels measured before t_L and entered the place they measure from
 *    t_L on. Before t_L, a reading whose endpoint lies within one truncation of
 *    where the object was (whatever the pixel's label) measured the object's
 *    earlier state and is removed; a reading's free space ends where it meets
 *    where the object is now (online mesh or current measurements).
 * 2. Present: one TSDF of the valid evidence at the object resolution (Open3D
 *    ScalableTSDFVolume semantics, see PresentTsdf), marching cubes. Where it
 *    extracts no surface (a cube corner never integrated), the session's own
 *    online surface stays where a valid reading measures it.
 * 3. Sensor model, measured on the session's own frames: depth noise sigma(q)
 *    per range bin on the present; the depth scale s (every reading scaled by
 *    1 + s makes the frames agree best). A surface element of half-voxel h is
 *    known along a ray at range q to tau = max(h, sigma(q)); two surfaces on one
 *    ray are told apart only farther than b(q) = max(T, tau + (s + s_prev) q):
 *    the layer's truncation, or the position error the two sessions' depth
 *    scales explain.
 * 4. Memory (the previous session's final map). An object's surface is memory
 *    only while the node of its physical id is current and its state did not
 *    begin in this session. Every other surface element is tested against the
 *    valid frames; each frame either hits it (a reading's point inside its ball
 *    of radius tau), sees through it (every pixel of its footprint, radius
 *    focal * tau / z, reads beyond it by more than tau) or is blocked in front
 *    of it (within b: its own surface displaced; beyond b: something else). It
 *    stays unless the frames that reached it saw it gone (through outnumbers
 *    hit) or, if none reached it, most blocked frames were blocked within b.
 *    Objects are solid: an object's memory inside the same object's present
 *    surface gives way.
 * 5. Identity: a present face belongs to one of the current objects whose
 *    current measurements lie within one truncation of it, voted as the object
 *    reasoning votes a voxel into an object: over the valid frames that observe
 *    the object and measure the face, the face's pixel is or is not the object;
 *    it belongs to the object with at least `membership_observations` such
 *    frames and a share of at least `membership_confidence` (the object
 *    extractor's own settings), the most voted if several; a face no observing
 *    frame measured belongs to the nearest such object; otherwise background.
 * 6. Compose: present faces go to the node of their identity, memory faces keep
 *    theirs; a node never ends empty. Object identities, states, boxes,
 *    presence and every other snapshot are untouched; the map is edited only
 *    once everything is computed.
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
    // The object extractor's rule for a voxel belonging to an object
    // (min_object_reconstruction_confidence / _observations).
    float membership_confidence = 0.5f;
    int membership_observations = 0;
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
