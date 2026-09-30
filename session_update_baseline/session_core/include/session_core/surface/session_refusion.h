#pragma once

#include <array>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "session_core/evidence/error_model.h"
#include "session_core/state/persistent_object_state.h"
#include "session_core/surface/frame_archive.h"
#include "khronos/common/common_types.h"

namespace khronos {

/** Session-end surface estimator. README (8)--(14).
 * State-authorized frames define a present TSDF. Historical and online-fill
 * elements share one view classification and one persistence loss.
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
    // README (10): T of the object representation layer, supplied by that layer.
    float object_truncation = 0.f;
    float object_min_voxel = 0.f;
  };

  // A map's surface in world coordinates with the physical id of every face
  // (0 = background).
  struct Surface {
    std::vector<Eigen::Vector3f> vertices;
    std::vector<std::array<uint32_t, 3>> faces;
    std::vector<uint32_t> face_physical;
    std::vector<float> face_error;  // README (11), metres; empty imports a legacy map.
    spark_dsg::Mesh::Timestamps stamps, first_seen_stamps;  // Zero is unknown.
    spark_dsg::Mesh::Colors colors;
    spark_dsg::Mesh::Labels labels;
  };

  struct Inputs {
    const std::vector<FrameArchive::Frame>* frames = nullptr;
    FrameArchive::Camera camera;
    Scales scales;
    // README (8a): physical ID -> active static-state start; nullopt is the
    // empty domain of a closed state. An omitted ID has no established current
    // state and therefore the empty domain.
    std::map<size_t, std::optional<TimeStamp>> state_starts;
    // README (8b): previous CURRENT no longer owns this identity's output surface.
    // Independent of birth_time, including old pending states promoted this session.
    std::set<size_t> replaced_states;
    TimeStamp final_stamp = 0;
    std::string dump_dir;  // optional diagnostics
    // Explicit predecessor surface; nullptr means an initial session.
    const Surface* shown = nullptr;
    // Legacy calibration diagnostics; face_error carries geometric provenance.
    std::vector<float> previous_depth_scales;
    // README (6m): the effective range error model of the previous round predicts this one.
    measurement::ErrorModel psi;
    // README (7): mean normal whole-round penetration fraction, the share of penetrating echoes a
    // surface in place normally shows (1/2 for the neutral start).
    double normal_fraction = 0.5;
    // README (5e): expected stationarity E[v] of each identity's current placement, and the
    // stationarity of the background, as the persistence prior of a historical face.
    std::map<size_t, double> stationarity;
    double background_stationarity = PersistentObjectState::kInitialMean;
  };

  struct Result {
    bool applied = false;
    // This session's depth scale: every reading scaled by (1 + s) makes the
    // session's frames agree best (exact depth: s = 0).
    float depth_scale = 0.f;
    std::string summary;
    std::string report_json;
    std::vector<float> surface_error;  // Final fromDsg face order.
    // README (9c): the session's effective residual scale per range bin [m] (0 = no estimate).
    std::vector<float> sigma;
  };

  explicit SessionRefusion(const Config& config) : config(config) {}

  Result apply(DynamicSceneGraph& dsg, const Inputs& inputs) const;

  static Surface fromDsg(const DynamicSceneGraph& dsg);
  static void saveSurfaceError(const std::string& path, const Surface& surface);
  static void loadSurfaceError(const std::string& path, Surface& surface);

  const Config config;
};

}  // namespace khronos
