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
 * One causal surface per mapping snapshot, using native Khronos/Hydra fusion.
 * Frames enter a persistent geometry volume only through the snapshot stamp.
 * Current physical states define which observations remain eligible. Changed
 * domains rebuild affected blocks in observation order using the native
 * integrator, including its capped weights. Tracking keeps its short active
 * window and does not inherit the persistent volume's observation history.
 * Previous-session mesh memory is retained under the existing evidence rules.
 * Thread-safe: frames arrive from the active window, rounds from the backend.
 */
class PresentSurface {
 public:
  using Ptr = std::shared_ptr<PresentSurface>;

  // Reuse the active window's voxel and integrator; the volume integrates every
  // observation of the session, so its truncation is two voxels.
  PresentSurface(const hydra::VolumetricMap::Config& map,
                 const hydra::ProjectiveIntegrator::Config& integrator,
                 const hydra::MeshIntegratorConfig& mesh);

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
    // Whether a point of the round's snapshot is a copy of the loaded state.
    std::function<bool(const Eigen::Vector3f&)> is_memory;
    // World-space surfaces of the states the state layer closed this round. A shown
    // face of the previous final map that lies on one of them is the closed state's
    // own old surface: it follows the state atomically instead of the per-face test.
    std::vector<Eigen::Vector3f> closed_memory;
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

  struct Result {
    std::string report_json;
  };

  /** Enqueue a processed frame; compose consumes only frames through its stamp. */
  void integrate(const FrameData& data);

  /** Test/replay input: enqueue one archived frame through the same production path. */
  void integrate(const FrameArchive::Camera& camera, FrameArchive::Frame frame);

  /**
   * One round: bring the TSDF to the round's states -- `state_starts` (physical
   * id -> t_l of its current state, begun in this session; any other id
   * contributes in every frame) -- with the current object meshes of `dsg`, the
   * round's snapshot at `stamp`,
   * and replace the surfaces of `dsg` by the TSDF's surface and the memory that
   * stays. The session's depth scale is estimated when the memory needs it and
   * in the `terminal` round.
   */
  Result compose(spark_dsg::DynamicSceneGraph& dsg,
                 TimeStamp stamp,
                 const std::map<size_t, TimeStamp>& state_starts,
                 const Inputs& inputs,
                 bool terminal);

  /** Stamp of the session's first kept frame (0 before any). */
  TimeStamp firstStamp() const;

 private:
  /**
   * An identity's state as eq. (8) uses it: its current state, if it began in
   * this session (`current`, at `start` = t_l).
   */
  struct State {
    bool current = false;
    TimeStamp start = 0;                     // t_l
    std::vector<Eigen::Vector3f> triangles;  // current object mesh, 3 world vertices per face
    std::vector<uint64_t> voxels;            // measured voxels from t_l on (sorted keys)
    Eigen::AlignedBox3f box;                 // bounds of the triangles and voxels
    // The triangles (index of their first vertex) and voxels (index into `voxels`) by cell of the
    // volume's blocks, with each cell's bounds: a pixel only meets what lies in the cells it meets.
    struct Cell {
      Eigen::AlignedBox3f box;
      std::vector<uint32_t> triangles, voxels;
    };
    std::vector<Cell> cells;
    /** Whether the state restricts a frame at `t`: older than t_l. */
    bool restricts(TimeStamp t) const { return current && t < start; }
    /** The latest frame stamp the state can restrict. */
    TimeStamp horizon() const { return current ? start : TimeStamp(0); }
    bool operator==(const State& other) const;
  };
  using States = std::map<uint16_t, State>;
  /**
   * What the states restricting one kept frame take from it: the pixels of the
   * `excluded` ids and the free space beyond `limit` (empty: none).
   */
  struct Restriction {
    std::vector<uint16_t> excluded;
    std::vector<float> limit;
  };
  /** One frame's legal domain: depth without the excluded pixels, free limit (empty: none). */
  struct Domain {
    std::vector<float> depth, limit;
    PresentTsdf::Domain view() const { return {&depth, limit.empty() ? nullptr : &limit}; }
  };
  struct Scratch;
  using Window = PresentTsdf::Window;

  /** Advance the one volume through the snapshot stamp; never consume a future frame. */
  void advanceTo(TimeStamp stamp);
  /** Set `scratch` to kept frame `index` with its decoded range. */
  void prepare(size_t index, const std::vector<uint16_t>& range_mm, Scratch& scratch) const;
  /** The pixels of kept frame `index` whose centre ray can meet `box` (empty: none). */
  Window windowOf(size_t index, const Eigen::AlignedBox3f& box) const;
  /**
   * Whether kept frame `index` reads as far as `box`, or within `beyond` past its readings: the box
   * is in its image no farther than the readings there (bounded per tile) and its farthest reading.
   */
  bool reaches(size_t index, const Eigen::AlignedBox3f& box, float beyond) const;
  /** The triangles (3 world vertices each) and measured voxels one of two surfaces has and the other lacks. */
  struct Changed {
    Eigen::AlignedBox3f box;
    std::vector<Eigen::Vector3f> triangles;
    std::vector<uint64_t> voxels;
  };
  Changed changedPart(const State& before, const State& after) const;
  /** Mark in `mask` (and extend `touched` over) the pixels of kept frame `index` the `changed` primitives cover. */
  void markChanged(size_t index, const Changed& changed, std::vector<uint8_t>& mask, Window& touched) const;
  /**
   * Min into `limit` (+inf where none) the first hit of every valid pixel's ray
   * of the prepared frame, within `window` (only where `mask` is set, if given),
   * on the surface of `state` within the
   * ray's free part [0, R - T]: on its mesh, else (no qualified mesh hit) on its
   * measured voxels.
   */
  void limitBy(const State& state, Scratch& scratch, std::vector<float>& limit, const Window& window,
               const std::vector<uint8_t>* mask) const;
  /**
   * Apply `r` to the frame's decoded `range` (the pixels it takes become 0) and
   * return its free limit: +inf where no pixel remains, empty when no remaining
   * pixel is limited.
   */
  std::vector<float> apply(const Restriction& r, std::vector<uint16_t>& range,
                           const std::vector<FrameArchive::InstanceRun>& runs) const;
  /** The domain of a frame's counted `range` and free `limit`. */
  void domainOf(const std::vector<uint16_t>& range, std::vector<float> limit, Domain& domain) const;
  /**
   * Kept frame `index` (decoded `range`, instance `runs`) under `states`, computed: `range`
   * loses the pixels the states take; `limit` is the free limit per kept pixel
   * (+inf: none; empty: no pixel limited).
   */
  void measure(const States& states, size_t index, std::vector<uint16_t>& range,
               const std::vector<FrameArchive::InstanceRun>& runs, Scratch& scratch,
               std::vector<float>& limit) const;
  /**
   * Kept frame `index` as the TSDF counts it, decoded: `range` without the
   * pixels the current states take; with `limit`, the free limit it was
   * measured with, kept from its integration or its last move (it changes only
   * when it moves). False if the frame cannot be decoded.
   */
  bool counted(size_t index, std::vector<uint16_t>& range, std::vector<float>* limit) const;
  /** Record the endpoints of the frames kept since the last call. */
  void recordEndpoints();
  /** Bring the TSDF to `next`: move every kept frame whose domain differs to its new domain. */
  void updateStates(States next);

  const double voxel_;       // shared map voxel (background, objects, memory)
  const double truncation_;  // two voxels
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
  size_t consumed_ = 0;  // prefix visible to the current snapshot
};

}  // namespace khronos
