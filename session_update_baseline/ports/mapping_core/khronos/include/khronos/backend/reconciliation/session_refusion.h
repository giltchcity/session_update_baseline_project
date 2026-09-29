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
 * @brief Session-end update of the final map: one estimate of the surface from
 * this session's valid evidence and the surface the previous session handed
 * over (SessionSurface). The object reasoning decides which objects exist, in
 * which state, since when; this update only estimates their surfaces and the
 * static structure's.
 *
 * State. The previous surface: faces with the identity of their state and the
 * measurement (q_f, s_f) that placed them. The object reasoning's current
 * states: for every object whose current state began within this session, the
 * start t_L of that state (its first sighting).
 *
 * Validity. A reading (frame t, pixel u, range r, instance id) counts only for
 * the scene it saw. For an object L whose current state began at t_L, a reading
 * before t_L on L's own pixels measured L's earlier state and is void, and its
 * free space ends one truncation before it meets where L is now (L's online
 * mesh, else the voxels L's own pixels measured from t_L on). A face of the
 * previous surface is void when its object has no current node or its current
 * state began within this session.
 *
 * Present. One TSDF of the valid readings at the object resolution v, T = 2v
 * (Open3D ScalableTSDFVolume semantics, PresentTsdf), marching cubes: the
 * session's estimate of every surface it measured. It supersedes the session's
 * own online surface wherever it is defined (both estimate the same readings);
 * where it is undefined, the own online faces fill it. A face of the session's
 * surface (present or own) takes its identity from the object reasoning, which
 * states membership by the surface it reconstructed for an object's current
 * state and by its pixel identities: a face that re-measures a current object's
 * surface (within one voxel) belongs to that object; elsewhere the reasoning's
 * voxel membership rule decides: of the valid frames that observe object L and
 * measure the face (front-facing, reading within tau), at least
 * `membership_confidence` read it as L, with at least `membership_observations`
 * such frames (the most voted object if several; else the static structure).
 * A face neither reconstructed nor measured takes the majority identity of its
 * nearest decided faces along the surface (across the seam between the present
 * and an own face: the present face nearest to it within one voxel).
 *
 * Sensor. sigma(q): the depth noise per range bin, measured on the present;
 * s: the session's depth scale. An element of half-voxel h is located along a
 * ray at range q to tau(q) = max(h, sigma(q)); the present confirms it where a
 * present surface lies within one voxel 2h of it (its layer cannot tell the two
 * apart), and holds a face where it passes through the face's centroid within
 * 2h along the face's normal (a present that ends beside the face, at a hole,
 * does not hold it). Every test on an element samples its centroid. The map
 * cannot tell the element from a surface this session measured
 * (at range r, in front of it) when that surface lies within the element
 * layer's truncation T_f in front of it along the view's ray (the layer's TSDF
 * holds no second surface closer behind), or when the present lies within the
 * two measuring sessions' position error
 * eps_f(r) = tau(r) + max(s, 0) r + max(s_f, 0) q_f of it, on the view's side
 * (the depth scales place one surface apart; the last term is 0 for this
 * session's own elements). The first is a distance along the ray, as the TSDF
 * integrates along rays; the second compares two surface estimates in space.
 *
 * Elements. The surface elements the present does not supersede: the valid
 * faces of the previous surface (prior p = 1) and the session's own online
 * faces where the present is undefined (a corner of their centroid cube never
 * integrated) and does not hold them (prior p = 0; a face the present holds is
 * the same readings' surface at a coarser resolution). A previous face lies in the layer of
 * its identity (static structure: the background map; objects: the object
 * maps); an own face fills the present and lies at the present's resolution.
 * A valid view of an element's centroid x is one of
 *   support     a reading of its footprint (radius f tau / z) lands in its ball
 *               of radius tau;
 *   free        every footprint pixel reads beyond it by more than tau, in free
 *               space the view may count;
 *   remeasured  its own line of sight ends in front of it on a surface the map
 *               cannot tell from it, or on its own object (a solid holds none of
 *               its own surface behind its surface), and the present does not
 *               confirm it;
 *   occluded    its own line of sight ends in front of it otherwise.
 * The element stays iff  p + S - F - max(0, R - O) > 0.
 * Occlusion never supports a surface; re-measuring views count only as far as
 * they outnumber occluded ones. A view blocked in front of a confirmed element
 * is occluded: a grazing view magnifies a confirming surface's offset along its
 * ray.
 *
 * Map. The present faces and the elements that stay, each in the node of its
 * identity (the background for identity 0 or an identity without a current
 * node). Object identities, states, boxes,
 * presence and every other snapshot are untouched; the map is edited only once
 * everything is computed. The resulting surface (SessionSurface) carries this
 * session's nearest reaching range and s for present faces and kept own
 * elements, and the records of kept previous faces.
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
    // Physical id -> start stamp of its current state, for every object whose
    // current state began within this session.
    std::map<size_t, TimeStamp> state_starts;
    // Whether a vertex of the final map was carried over from the loaded state
    // (the object reasoning's inherited geometry, superseded by `previous`).
    std::function<bool(const Eigen::Vector3f&)> carried;
    // The surface the previous session handed over (may be null: first session).
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
    float depth_scale = 0.f;
    // The final map's surface with identities and records (valid if applied).
    SessionSurface surface;
    std::string summary;
    std::string report_json;
  };

  explicit SessionRefusion(const Config& config) : config(config) {}

  Result apply(DynamicSceneGraph& dsg, const Inputs& inputs) const;

  const Config config;
};

}  // namespace khronos
