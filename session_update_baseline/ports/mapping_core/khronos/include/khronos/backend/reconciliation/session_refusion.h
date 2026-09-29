#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "khronos/backend/reconciliation/frame_archive.h"
#include "khronos/backend/reconciliation/session_surface.h"
#include "khronos/common/common_types.h"

namespace khronos {

/**
 * @brief Session-end update of the final map: its surface after this session,
 * estimated from this session's valid readings and the surface of the map the
 * previous session left.
 *
 * Assumptions.
 *  A1 Entities. Every surface belongs to one entity: the background (identity
 *     0) or an object (its physical id). The object reasoning decides which
 *     objects exist, in which state, and since when: t_L, the first instant of
 *     an object's current state (Track::stateStart). An object moves only as a
 *     whole, at a state change; an object this session observes is re-estimated
 *     as a whole.
 *  A2 Measurement. A valid reading at pixel u measures the first surface along
 *     its ray at range r, with noise sigma(q) per range bin (measured on the
 *     present) and the session's depth scale s (a reading is off by s r). An
 *     estimate of a layer with voxel 2h locates its surface to h: a reading and
 *     an estimate at range q agree within tau(h, q) = max(h, sigma(q)). Every
 *     estimate is misplaced along its line of sight by up to its session's
 *     scale error: a reading at range r by max(s, 0) r; a surface by the error
 *     eps_f = max(s_k, 0) q_k the session k that measured it recorded (q_k the
 *     range of k's nearest view reaching it), carried while it persists
 *     (SessionSurface::error); a surface of this session's own readings by
 *     nothing beyond the reading's.
 *  A3 Time. A reading at time t is evidence about the current scene only where
 *     the scene did not change after t. For an object whose current state began
 *     at t_L within this session, a reading before t_L on its own pixels is
 *     void, and its free space ends one truncation before it meets where the
 *     object is now (its online mesh, else the voxels its own valid readings
 *     measured from t_L on).
 *  A4 Persistence. A surface the previous session left persists unless this
 *     session's valid readings contradict it: across the session boundary its
 *     prior is one view (p = 1); a surface of this session has none (p = 0).
 *  A5 Representation. A layer of voxel 2h and truncation T holds one surface
 *     within T along a ray and tells no two surfaces within one voxel apart.
 *     The stored map holds every entity at its layer (background: the
 *     background map; objects: the object maps). One set of readings yields one
 *     estimate, at the present's resolution: the present supersedes this
 *     session's online surface wherever it holds information about it;
 *     elsewhere an online face is a candidate location of that estimate.
 *
 * Present (A2, A3). One TSDF of the valid readings at the object resolution v,
 * T = 2v (Open3D ScalableTSDFVolume semantics, PresentTsdf), marching cubes:
 * position and existence of every surface the session measured.
 *
 * Elements (A1, A3, A5). The surfaces the present does not re-estimate: the
 * faces of the previous surface whose object has a current node and whose
 * current state began before this session (p = 1), at the layer of their node;
 * and the session's own online faces the present holds no information about,
 * that is whose centroid cube it never fully integrated and which it does not
 * pass through within one voxel along their normal (p = 0), candidate
 * locations of this session's estimate at the present's resolution. An element
 * is judged at its centroid x. The present coincides with x when its nearest
 * point lies within one voxel 2h of x.
 *
 * Views (A1, A2, A3, A5). A valid view of x is one of
 *   remeasured  x's line of sight ends in front of x beyond tau on x's own
 *               surface and the present does not coincide with x (within one
 *               voxel nothing is re-measured elsewhere, A5): the reading lies
 *               within T in front (A5: x's layer holds no second surface
 *               there), or the present lies on the view's side within the two
 *               estimates' position error eps(r) = tau(h, r) + max(s, 0) r +
 *               eps_f (A2; eps_f = 0 for this session's surfaces), or the
 *               reading is on x's own object (A1);
 *   support     otherwise a reading of x's footprint (radius f tau / z) lands
 *               within tau of x: the view measures x;
 *   free        otherwise every footprint pixel reads beyond x by more than
 *               tau, in free space the view may count (A3);
 *   occluded    otherwise x's line of sight ends in front of x beyond tau.
 *
 * Decision (A4). x stays iff p + S - F - max(0, R - O) > 0: its log-odds after
 * the views, in which re-measuring views count against x as far as they
 * outnumber the views occluded by another surface.
 *
 * Identity (A1). A face of the session's surface (present and own) takes the
 * object reasoning's decision: the object whose reconstructed current surface
 * coincides with it (within one voxel of the present's resolution); else the
 * reasoning's voxel membership rule on the frames that measure it (front-
 * facing, reading within tau): of the valid frames that observe object L and
 * measure the face, at least `membership_confidence` read it as L, with at
 * least `membership_observations` such frames (the most voted object if
 * several; else the background); else the majority identity of its nearest
 * decided faces along the surface (across the seam between the present and an
 * own face: the present face coinciding with it).
 *
 * Map. The present faces and the elements that stay, each in the node of its
 * identity (the background for identity 0 or an identity without a current
 * node; the map always has a background). An added vertex takes the attributes
 * of the nearest vertex of its mesh before the update; a mesh without vertices
 * gives it the session's final stamp and zero color and label. Object
 * identities, states, boxes, presence and every other snapshot are untouched;
 * the map is edited once everything is computed. Every face of the edited map
 * carries its position error: this session's, max(s, 0) q with q the range of
 * its nearest view reaching the face, for a present face or an admitted own
 * face; the recorded one for a kept previous face. The edited map with these
 * errors is the state the next session reads.
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
  };

  struct Inputs {
    const std::vector<FrameArchive::Frame>* frames = nullptr;
    FrameArchive::Camera camera;
    Scales scales;
    // Physical id -> t_L, the first instant of its current state, for every
    // object whose current state began within this session.
    std::map<size_t, TimeStamp> state_starts;
    // Whether a vertex of the final map was carried over from the loaded state
    // (the object reasoning's inherited geometry, superseded by `previous`).
    std::function<bool(const Eigen::Vector3f&)> carried;
    // The surface of the map the previous session left (may be null: first session).
    const SessionSurface* previous = nullptr;
    // The object extractor's voxel membership rule
    // (min_object_reconstruction_confidence / _observations).
    float membership_confidence = 0.5f;
    int membership_observations = 0;
    TimeStamp final_stamp = 0;
    std::string dump_dir;  // optional diagnostics
  };

  struct Result {
    bool applied = false;
    // This session's depth scale s (A2).
    float depth_scale = 0.f;
    // The position error of every face of the edited map, in the order of
    // SessionSurface::fromDsg (saved next to the map, SessionSurface::kErrorFileName).
    std::vector<float> surface_error;
    std::string summary;
    std::string report_json;
  };

  explicit SessionRefusion(const Config& config) : config(config) {}

  Result apply(DynamicSceneGraph& dsg, const Inputs& inputs) const;

  const Config config;
};

}  // namespace khronos
