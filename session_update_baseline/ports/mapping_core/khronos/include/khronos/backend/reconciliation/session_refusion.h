#pragma once

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
 * measure there. Steps:
 *  1  an object whose current state began within this session (state start
 *     t_L = the first sighting of that state) contributes no pixels from frames
 *     before t_L; 1b those frames also lose every pixel whose ray passes
 *     through the object's current mesh more than one truncation in front of
 *     the reading (free space seen before the object got there);
 *  2  one TSDF of all remaining pixels (Open3D ScalableTSDFVolume semantics,
 *     see PresentTsdf), marching cubes;
 *  3  depth noise sigma(q) per range bin from the present against the frames;
 *     tau(q) = max(v/2, sigma(q));
 *  4  face label = majority physical id over the frames that measure the face,
 *     propagated over edges to unmeasured faces;
 *  5  compose: memory (surface inherited from earlier sessions) stays as the
 *     online consolidation left it, own faces are replaced by the present
 *     (label 0 -> background, label L -> the node with physical id L); a node
 *     never ends empty.
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
    size_t noise_vertex_stride = 4;
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
