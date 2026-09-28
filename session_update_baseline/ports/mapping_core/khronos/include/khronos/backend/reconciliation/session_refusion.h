#pragma once

#include <array>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "khronos/backend/reconciliation/frame_archive.h"
#include "khronos/backend/reconciliation/session_consolidation.h"
#include "khronos/common/common_types.h"

namespace khronos {

/**
 * @brief Session-end re-integration of the present from the session's own
 * frames (after SessionConsolidation, on the final snapshot only).
 *
 * The final map's own geometry (surface this session built online) is
 * replaced by one joint TSDF of every processed frame of the session at the
 * object resolution, labelled per face with the physical instance the frames
 * measure there; the memory it shows is the previous session's final map
 * where this session's reasoning kept it. Steps:
 *  1  an object whose current state began within this session (state start
 *     t_L = the first sighting of that state) contributes no pixels from frames
 *     before t_L; 1b in those frames, a pixel whose ray passes through the
 *     object's current surface more than one truncation in front of its
 *     reading saw the place before the object got there: its free space from
 *     one truncation before that first hit on no longer counts (its surface
 *     band still does). The object's current surface is its online mesh and
 *     every voxel (object resolution) holding an endpoint of the object's own
 *     pixels from t_L on (where this session measured the object as it is);
 *  2  one TSDF of all remaining evidence (Open3D ScalableTSDFVolume semantics,
 *     see PresentTsdf), marching cubes;
 *  3  depth noise sigma(q) per range bin from every present vertex against
 *     the frames; tau(q) = max(v/2, sigma(q));
 *  4  face label = majority physical id over the frames that measure the face,
 *     propagated over edges to unmeasured faces;
 *  5  measured fill: an own online face stays where the present extracted no
 *     surface (a corner of its cube was never integrated) but some frame
 *     measured it (|reading - range| <= tau at its centroid), e.g. surfaces
 *     seen only at grazing angles;
 *  6  memory. Without `shown`: the memory faces of the final map (surface
 *     inherited from earlier sessions) stay as the online consolidation left
 *     them. With `shown` (the previous session's final map): a shown face is
 *     memory where this session's reasoning kept what it corresponds to in the
 *     loaded state (the nearest loaded point of the same physical id within one
 *     voxel of its layer is a vertex of a memory face the consolidated map
 *     keeps); an object's shown faces only while the node of its physical id is
 *     current and its state did not begin in this session; a shown face no
 *     decision of this session covers (no corresponding loaded point, or one
 *     the previous session's own consolidation retired) stays only where no
 *     frame's ray reached it (memory speaks where the session did not look).
 *     Then one principle, INSIDE: an object's memory vertex that the frames had
 *     in view and that lies inside the same object's present surface (64-ray
 *     first-hit orientation vote), farther than one voxel from the present,
 *     gives way to the present;
 *  7  compose: memory, the present replacing the own faces (label 0 ->
 *     background, label L -> the node with physical id L) and the measured
 *     fill; a node never ends empty.
 * Object identities, states, boxes, presence and every other snapshot are
 * untouched. The final map is edited only once everything is computed.
 */
class SessionRefusion {
 public:
  struct Config {
    int num_threads = 4;
    // Noise estimator settings (as SessionConsolidation).
    double range_bin = 0.5;
    size_t num_bins = 16;
    size_t min_bin_samples = 1000;
    double histogram_resolution = 0.0005;
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
    SessionConsolidation::Scales scales;
    // Physical id -> start stamp of its current state, for every object whose
    // current state began within this session.
    std::map<size_t, TimeStamp> state_starts;
    // Memory test (3 mm to the loaded state), the same the consolidation used.
    std::function<bool(const Eigen::Vector3f&)> is_memory;
    TimeStamp final_stamp = 0;
    std::string dump_dir;  // optional diagnostics
    // Memory as the previous session's final map showed it (optional). The
    // memory this map shows is then that surface, where this session's
    // reasoning kept the loaded state it corresponds to: `inherited` is the
    // loaded state (the previous session's unconsolidated final state that
    // object and change reasoning ran on) with the physical id of every
    // point, `chain_retired` tells the points the previous session's own
    // consolidation retired (not a decision of this session).
    const Surface* shown = nullptr;
    const std::vector<Eigen::Vector3f>* inherited = nullptr;
    const std::vector<uint32_t>* inherited_physical = nullptr;
    std::function<bool(const Eigen::Vector3f&)> chain_retired;
    float memory_match_distance = 0.003f;  // SessionConsolidation::Config
  };

  struct Result {
    bool applied = false;
    std::string summary;
    std::string report_json;
  };

  explicit SessionRefusion(const Config& config) : config(config) {}

  Result apply(DynamicSceneGraph& dsg, const Inputs& inputs) const;

  const Config config;
};

}  // namespace khronos
