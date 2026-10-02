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
 * State-authorized frames define a present TSDF (every reading outside the native motion mask,
 * without identity or with the identity of a still current state). A memory element (a face of the
 * previous map) is shown unless the majority of the frames that reach it see through it (10b),
 * unless it shares its zero crossing with a present surface in front of it (principle 11), is a
 * displaced copy of a present surface (12d) or lies inside the solid of its object (principle 10);
 * its record is deleted only when the frames as one look make the evidence of an ended surface
 * exceed ln((1 - alpha) / alpha). A face of the online map the present does not draw is completed
 * when a valid reading puts the surface on it (14f). The records that are neither shown nor
 * deleted travel with the evidence state.
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
    // README principle 12, eq. (15b): the record of a face: the frames that hit it (h_e) and saw
    // through it (v_e) in the session that kept it, the mean range rho_e of the measurements that
    // made it and the depth scale zeta_e of that session. Empty for a surface without records.
    std::vector<float> face_hits, face_through, face_rho, face_zeta;

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
    // online from the session's own data.
    model::RangeModel psi;
    // README principle 9: the in-place distributions of the look statistics decide the deletion of
    // a record; null: the cold-start distribution of the measurement model.
    const model::RoundModel* rounds = nullptr;
    // README principle 9: the group of a memory element is the class of the placement it belongs
    // to (objects, under a placement committed in place) or its own class (background).
    std::map<size_t, int> identity_class;
    // README principle 12: the depth scale of the previous session, zeta_e of a face whose record
    // is missing.
    double previous_zeta = 0.0;
  };

  struct Result {
    bool applied = false;
    // This session's range scale zeta (psi.zeta): every reading scaled by (1 + zeta) makes the
    // session's frames agree best (exact depth: zeta = 0).
    float depth_scale = 0.f;
    std::string summary;
    std::string report_json;
    std::vector<float> surface_error;  // Final fromDsg face order.
    // README (15b): the record {h_e, v_e, rho_e, zeta_e} of every output face, in the same order.
    struct FaceRecords {
      std::vector<float> hits, through, rho, zeta;
    } surface_records;
    // README (9c): the session's residual scale of the present surface per range bin [m]
    // (0 = no estimate); sigma_table of (6s).
    std::vector<float> sigma;
    // README (15b): the free space this session observed.
    FreeSpaceRecords free_space;
    // README (6m), principle 10: the range model of the session end -- sigma_table on the present
    // surface, and sigma_x estimated from the offsets of the memory elements from the present
    // surface of the same identity where the session had none -- which the next session starts from.
    model::RangeModel psi;
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
