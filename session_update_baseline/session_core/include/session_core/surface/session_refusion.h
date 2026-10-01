#pragma once

#include <array>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "session_core/evidence/free_space_records.h"
#include "session_core/model/persistence_prior.h"
#include "session_core/model/range_model.h"
#include "session_core/model/round_model.h"
#include "session_core/state/persistent_object_state.h"
#include "session_core/surface/frame_archive.h"
#include "khronos/common/common_types.h"

namespace khronos {

/** Session-end surface estimator. README principles 5, 7, 9, 10, 11 (eqs. (8)--(14)).
 * State-authorized frames define a present TSDF: a reading without identity enters it iff the
 * surface it measured survives to the end of the session with probability >= 1/2 (the class's
 * learned survival). Historical (memory) and online-fill elements are shown, hidden or deleted by
 * one posterior of the persistence model Pi and the first-return likelihood: whether an element is
 * shown is a representation output (maximum a posteriori), whether its record is deleted a
 * commitment at the single decision level alpha; the records that are neither shown nor deleted
 * travel with the evidence state.
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
    // README principle 9, eq. (15b): the record of a memory element that the map does not show but
    // has not deleted: the hits k and see-throughs j of its committed history and the frames
    // whose verdict is still pending. Empty for a surface without such records.
    std::vector<float> face_hits, face_through, face_pending_hits, face_pending_through;

    /** Add the faces of `other` (a set of records) to this surface. */
    void append(const Surface& other);
    nlohmann::json toJson() const;
    static Surface fromJson(const nlohmann::json& value);
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
    // README principle 9: the group of a memory element is the class of the placement it belongs
    // to (objects, under a placement committed in place) or its own class (background); the class
    // of every identity of the registry.
    std::map<size_t, int> identity_class;
    // README principle 5: the predictive survival S_c of each semantic class at the end of the
    // session (the smoothed estimate from all events and exposure of the session). A reading
    // without identity enters the refusion iff S_c(t_end - t) >= 1/2.
    std::map<int, model::PersistencePrior::Hazard> class_hazards;
    // README (7s): the committed-round histories (hits k, see-throughs j) of the elements of each
    // placement at the start of the session, keyed by the cell of the map resolution.
    std::map<size_t, std::unordered_map<uint64_t, std::pair<float, float>>> element_histories;
    struct ClosedSurface {
      std::vector<Eigen::Vector3f> vertices;
      std::vector<std::array<uint32_t, 3>> faces;
      double odds = 0.0;  // the closure odds of the placement
    };
    std::vector<ClosedSurface> closed_surfaces;
    // README principle 9: the confirmed share of the earlier completion candidates (the prior of
    // a candidate), the beta-binomial data of the completion prior.
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
    // README (15b): the element record {k_e, j_e} of every output face that rests on a memory
    // element (zeros for the faces of the present surface and the completion faces), in the same
    // order: committed hits and see-throughs, and the frames whose verdict is still pending.
    struct FaceRecords {
      std::vector<float> hits, through, pending_hits, pending_through;
    } surface_records;
    // README (9c): the session's residual scale of the present surface per range bin [m]
    // (0 = no estimate); sigma_table of (6s).
    std::vector<float> sigma;
    // README (15b): the free space this session observed.
    FreeSpaceRecords free_space;
    // The completion statistics of this run: candidates judged and confirmed.
    double fill_confirmed = 0.0, fill_total = 0.0;
    // README (12d): the band-pair mixture fitted on this session's memory elements and present
    // surface (Delta_s, sigma_x, pi_dup); the estimate the next session starts from.
    bool pair_fit_valid = false;
    double pair_pi_dup = 0.5, pair_delta_s = 0.0, pair_sigma_x = 0.0;
    // README (15b): memory elements not shown but not deleted, with their evidence.
    Surface hidden;
  };

  explicit SessionRefusion(const Config& config) : config(config) {}

  Result apply(DynamicSceneGraph& dsg, const Inputs& inputs) const;

  static Surface fromDsg(const DynamicSceneGraph& dsg);
  static void saveSurfaceError(const std::string& path, const Surface& surface);
  static void loadSurfaceError(const std::string& path, Surface& surface);

  const Config config;
};

}  // namespace khronos
