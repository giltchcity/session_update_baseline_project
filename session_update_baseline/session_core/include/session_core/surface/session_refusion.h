#pragma once

#include <array>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "session_core/evidence/free_space_records.h"
#include "session_core/model/range_model.h"
#include "session_core/model/round_model.h"
#include "session_core/state/persistent_object_state.h"
#include "session_core/surface/frame_archive.h"
#include "khronos/common/common_types.h"

namespace khronos {

/** Session-end surface estimator. README principles 7, 9, 10, 11 (eqs. (8)--(14)).
 * State-authorized frames define a present TSDF. Historical and online-fill elements are retained
 * or dropped by one posterior: the prior of the persistence model Pi and the likelihood ratio of
 * the frames of the session, committed at the single decision level alpha.
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
    // README (8): physical ID -> active static-state start t_L; nullopt is the
    // empty domain of a closed state. An omitted ID has no established current
    // state and therefore the empty domain.
    std::map<size_t, std::optional<TimeStamp>> state_starts;
    // README (8): previous CURRENT no longer owns this identity's output surface.
    // Independent of birth_time, including old pending states promoted this session.
    std::set<size_t> replaced_states;
    TimeStamp final_stamp = 0;
    std::string dump_dir;  // optional diagnostics
    // Explicit predecessor surface; nullptr means an initial session.
    const Surface* shown = nullptr;
    // Audit of the depth scales of the earlier sessions.
    std::vector<float> previous_depth_scales;

    // README (6m): the parameters of the first-return model at the end of the session, estimated
    // online from the session's own static re-measurements.
    model::RangeModel psi;
    // README principle 6: the round model whose element-level likelihood ratio (7) and ended
    // distribution decide the retention of an element; null: neutral evidence.
    const model::RoundModel* rounds = nullptr;
    // README (7s): the hits of the construction of an element (minimum mesh weight).
    double construction_hits = 0.0;
    // README principle 9: the persistence prior q_e of a historical element. An element of a
    // placement has the placement's q^g at the start of the session; a background element the
    // background class's q^g; an element that coincides with the surface of a placement committed
    // changed takes that placement's closure odds.
    std::map<size_t, double> identity_change_prior;
    double background_change_prior = 0.5;
    // README (7s): the committed-round histories (hits k, see-throughs j) of the elements of each
    // placement at the start of the session, keyed by the cell of the map resolution.
    std::map<size_t, std::unordered_map<uint64_t, std::pair<float, float>>> element_histories;
    struct ClosedSurface {
      std::vector<Eigen::Vector3f> vertices;
      std::vector<std::array<uint32_t, 3>> faces;
      double odds = 0.0;  // the closure odds of the placement
    };
    std::vector<ClosedSurface> closed_surfaces;
    // README principles 9, 10: outcome statistics of the earlier committed decisions, the
    // beta-binomial data of pi_dup and of the completion prior.
    double dup_committed = 0.0, sep_committed = 0.0;
    double fill_confirmed = 0.0, fill_total = 0.0;
  };

  struct Result {
    bool applied = false;
    // This session's range scale zeta (psi.zeta): every reading scaled by (1 + zeta) makes the
    // session's frames agree best (exact depth: zeta = 0).
    float depth_scale = 0.f;
    std::string summary;
    std::string report_json;
    std::vector<float> surface_error;  // Final fromDsg face order.
    // README (9c): the session's residual scale of the present surface per range bin [m]
    // (0 = no estimate); sigma_table of (6s).
    std::vector<float> sigma;
    // README (15b): the free space this session observed.
    FreeSpaceRecords free_space;
    // The committed decisions of this run (Pi and the pair/completion statistics).
    double background_removed = 0.0, background_judged = 0.0;
    double dup_committed = 0.0, sep_committed = 0.0;
    double fill_confirmed = 0.0, fill_total = 0.0;
  };

  explicit SessionRefusion(const Config& config) : config(config) {}

  Result apply(DynamicSceneGraph& dsg, const Inputs& inputs) const;

  static Surface fromDsg(const DynamicSceneGraph& dsg);
  static void saveSurfaceError(const std::string& path, const Surface& surface);
  static void loadSurfaceError(const std::string& path, Surface& surface);

  const Config config;
};

}  // namespace khronos
