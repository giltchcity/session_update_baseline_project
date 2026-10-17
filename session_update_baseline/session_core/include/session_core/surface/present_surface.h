#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>
#include <spark_dsg/dynamic_scene_graph.h>

#include "khronos/common/common_types.h"
#include "session_core/surface/frame_archive.h"
#include "session_core/surface/present_tsdf.h"

namespace khronos {

struct FrameData;

/**
 * @brief The session's present surface, kept online (README section 5).
 *
 * Every processed frame is integrated into one TSDF as it arrives (eq. 9), the
 * background map's (its voxel and truncation: Khronos' surface layer), with
 * its legal contribution domain (eq. 8) under the current states. A state's
 * evidence counts only while the state holds. For a physical identity whose
 * current state began in this session at t_l, a frame older than t_l
 * contributes none of the identity's pixels, and the free space of each of its
 * other pixels stops one truncation before the pixel's first hit on the
 * identity's current surface within the ray's free part: the current object
 * mesh first, without a qualified hit the voxels holding an endpoint of the
 * identity's own pixels from t_l on. For a state of the identity that ended in
 * this session, a frame up to that end contributes no pixel that measured the
 * state's surface: labelled with the identity or unlabelled, ending within one
 * object voxel of the state's mesh (the correspondence identity uses below);
 * the identity's pixels elsewhere (a next state not yet known) stay. Each
 * change-detection round brings the states to the round's; every frame whose
 * domain changes is moved exactly to its new domain (PresentTsdf::update). The
 * TSDF is therefore always the one eq. (9) gives for the frames so far under
 * the current states.
 *
 * Each round shows this TSDF's surface: a face belongs to the current object
 * mesh within one object voxel of its centroid (the nearest), otherwise to the
 * background; an object that receives no face keeps its own mesh.
 *
 * Memory (README section 6): a face of the previous session's final map stays
 * unless the session's frames so far, each in its legal domain, place its
 * surface elsewhere (eqs. 10-13): seen through more often than hit, or neither
 * and mostly blocked within its layer's truncation band; a displaced copy of
 * the present within the error the sessions' depth scales explain; or inside
 * the same object's present surface. An object's memory shows only while its
 * node is current and its state did not begin in this session. A surface
 * element (half a voxel h of its layer) is known along a ray at range q to
 * tau = max(h, sigma(q)), sigma the depth residual measured on the present.
 *
 * Thread-safe: frames arrive from the active window, rounds from the backend.
 */
class PresentSurface {
 public:
  using Ptr = std::shared_ptr<PresentSurface>;

  /**
   * @param object_voxel The object reconstruction voxel [m]: the measured voxels of
   * a state and the face-to-object correspondence.
   * @param map_voxel, map_truncation The background map's voxel and truncation
   * distance [m]: the TSDF is the background map's (Khronos' surface layer).
   */
  PresentSurface(double object_voxel, double map_voxel, double map_truncation, int num_threads);

  /** A map's surface in world coordinates with the physical id of every face (0 = background). */
  struct Surface {
    std::vector<Eigen::Vector3f> vertices;
    std::vector<std::array<uint32_t, 3>> faces;
    std::vector<uint32_t> face_physical;
  };

  /** A round's inputs beyond its snapshot: the previous session's final map. */
  struct Inputs {
    // Memory as the previous session's final map showed it (its shown state).
    const Surface* shown = nullptr;
    // Without `shown`: whether a point of the round's snapshot is a copy of the loaded state.
    std::function<bool(const Eigen::Vector3f&)> is_memory;
    // The measured depth scale of every earlier session.
    std::vector<float> previous_depth_scales;
    // Optional diagnostics (KHRONOS_REFUSION_DUMP): the terminal round writes the frame
    // archive (archive.kfa), the present with its face identities (present.ply) and
    // its report (refusion_report.json) there, for refusion_replay.
    std::string dump_dir;
  };

  /**
   * Distance within which a vertex is a copy of `loaded` (P53): a copy passes
   * through at most two float32 rigid transforms (into an object's box frame and
   * back); each rounds a coordinate by at most six unit roundoffs of a magnitude
   * bounded by twice the scene extent R, so a copy lies within
   * sqrt(3) * 2 * 6 * u * 2R of its source.
   */
  static float copyTolerance(const std::vector<Eigen::Vector3f>& loaded);

  /** The states of one identity that ended in this session: their meshes (world) and latest end. */
  struct Ended {
    TimeStamp end = 0;
    std::vector<Eigen::Vector3f> triangles;  // 3 world vertices per face
  };

  /** Ended surfaces as the terminal round's dump stores them (for refusion_replay). */
  static bool saveEnded(const std::string& path, const std::map<size_t, Ended>& ended);
  static bool loadEnded(const std::string& path, std::map<size_t, Ended>& ended);

  struct Result {
    // This session's depth scale (every reading scaled by 1 + s makes the
    // session's frames agree best), when the round estimated it.
    std::optional<float> depth_scale;
    std::string report_json;
  };

  /** Keep and integrate one processed frame. */
  void integrate(const FrameData& data);

  /** Keep and integrate one archived frame (offline replay: refusion_replay). */
  void integrate(const FrameArchive::Camera& camera, FrameArchive::Frame frame);

  /**
   * One round: bring the TSDF to the round's states -- `state_starts` (physical
   * id -> t_l of its current state, begun in this session) and `ended` (physical
   * id -> its states ended in this session); an id in neither contributes in
   * every frame -- with the current object meshes of `dsg`, the round's snapshot at `stamp`,
   * and replace the surfaces of `dsg` by the TSDF's surface and the memory that
   * stays. The session's depth scale is estimated when the memory needs it and
   * in the `terminal` round.
   */
  Result compose(spark_dsg::DynamicSceneGraph& dsg,
                 TimeStamp stamp,
                 const std::map<size_t, TimeStamp>& state_starts,
                 const std::map<size_t, Ended>& ended,
                 const Inputs& inputs,
                 bool terminal);

  /** Stamp of the session's first kept frame (0 before any). */
  TimeStamp firstStamp() const;

 private:
  /**
   * An identity's states as eq. (8) uses them: its current state, if it began in
   * this session (`current`, at `start` = t_l), and its states ended in this
   * session (latest `end`, 0: none) with their surface voxels.
   */
  struct State {
    bool current = false;
    TimeStamp start = 0;                     // t_l
    std::vector<Eigen::Vector3f> triangles;  // current object mesh, 3 world vertices per face
    std::vector<uint64_t> voxels;            // measured voxels from t_l on (sorted keys)
    TimeStamp end = 0;
    std::vector<uint64_t> ended_voxels;      // voxels within one object voxel of the ended states' meshes (sorted keys)
    Eigen::AlignedBox3f box;                 // bounds of the triangles, voxels and ended voxels
    /** Whether the state restricts a frame at `t`: older than t_l, or not later than the end. */
    bool restricts(TimeStamp t) const { return (current && t < start) || (end > 0 && t <= end); }
    /** The latest frame stamp the state can restrict. */
    TimeStamp horizon() const { return std::max(current ? start : TimeStamp(0), end); }
    bool operator==(const State& other) const;
  };
  using States = std::map<uint16_t, State>;
  /**
   * What the states restricting one kept frame take from it: the pixels of the
   * `excluded` ids, the pixels (of that id or unlabelled) ending on an ended
   * voxel of a `voided` state, and (current states) the free space beyond
   * `limit` (empty: none).
   */
  struct Restriction {
    std::vector<uint16_t> excluded;
    std::vector<std::pair<uint16_t, const State*>> voided;
    std::vector<float> limit;
    bool prepared = false;
  };
  /** One frame's legal domain: depth without the excluded pixels, free limit (empty: none). */
  struct Domain {
    std::vector<float> depth, limit;
    bool operator==(const Domain& other) const { return depth == other.depth && limit == other.limit; }
    PresentTsdf::Domain view() const { return {&depth, limit.empty() ? nullptr : &limit}; }
  };
  struct Scratch;

  /** Integrate the frame just kept (decoded `range_mm`, `ids`) under the current states. */
  void integrateKept(const std::vector<uint16_t>& range_mm, const std::vector<uint16_t>& ids);
  /** Set `scratch` to kept frame `index` with its decoded range. */
  void prepare(size_t index, const std::vector<uint16_t>& range_mm, Scratch& scratch) const;
  /** Whether the surface of `state` reaches the image of kept frame `index`. */
  bool reaches(size_t index, const State& state) const;
  /**
   * Min into `limit` (+inf where none) the first hit of every valid pixel's ray
   * of the prepared frame on the surface of `state` within the ray's free part
   * [0, R - T]: on its mesh, else (no qualified mesh hit) on its measured voxels.
   */
  void limitBy(const State& state, Scratch& scratch, std::vector<float>& limit) const;
  /**
   * Add `state` of identity `id` to the restriction `r` of kept frame `index`
   * (decoded uncut `range`; `scratch` is prepared on first use); its surface
   * limits the free space only `with_limit`.
   */
  void restrictBy(uint16_t id, const State& state, size_t index, const std::vector<uint16_t>& range,
                  Scratch& scratch, Restriction& r, bool with_limit) const;
  /** The restriction of kept frame `index` by every state of `states` that restricts it. */
  void collect(const States& states, size_t index, const std::vector<uint16_t>& range,
               Scratch& scratch, Restriction& r, bool with_limit) const;
  /**
   * Apply `r` to the frame's decoded `range` (the pixels it takes become 0) and
   * return its free limit: +inf where no pixel remains, empty when no remaining
   * pixel is limited.
   */
  std::vector<float> apply(const Restriction& r, const Scratch& scratch, std::vector<uint16_t>& range,
                           const std::vector<uint16_t>& ids) const;
  /** The domain of a frame's counted `range` and free `limit`. */
  void domainOf(const std::vector<uint16_t>& range, std::vector<float> limit, Domain& domain) const;
  /**
   * Kept frame `index` (decoded `range`, `ids`) as the TSDF counts it under the
   * current states: `range` loses the pixels the states take; with `limit`, the
   * free limit per kept pixel (+inf: none; empty: no pixel limited).
   */
  void counted(size_t index, std::vector<uint16_t>& range, const std::vector<uint16_t>& ids,
               Scratch& scratch, std::vector<float>* limit) const;
  /** Record the endpoints of the frames kept since the last call. */
  void recordEndpoints();
  /** Bring the TSDF to `next`: move every kept frame whose domain differs to its new domain. */
  void updateStates(States next);

  const double voxel_;       // object voxel
  const double map_voxel_;   // the TSDF's voxel (background map)
  const double truncation_;  // the TSDF's truncation T (background map)
  const int num_threads_;
  mutable std::mutex mutex_;
  FrameArchive frames_;
  PresentTsdf tsdf_;
  PresentTsdf::Camera camera_;
  std::vector<float> ray_norm_;
  std::vector<Eigen::Vector3f> ray_dir_;  // unit ray of every pixel centre (camera frame)
  // The states every kept frame's contribution reflects.
  States states_;
  // Per physical id, the voxels holding an endpoint of its pixels and the
  // latest stamp of one, over the first `recorded_` kept frames.
  std::unordered_map<uint16_t, std::unordered_map<uint64_t, TimeStamp>> endpoints_;
  size_t recorded_ = 0;
};

}  // namespace khronos
