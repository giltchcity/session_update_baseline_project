#pragma once
// Online object front (11-21): physical ids from semantics inside the mapper, frame by frame, one front through the
// visits of a chain (its state in a file between visits). Two inputs, two modes:
//   SEMANTIC   annotated semantics: the TESSE route over visits, a C++ port of front_real3.py (tesse_semantic_front.py --3d,
//              the TESSE rows' semv5, with its fragments merged online as merge_fragments.py merged them offline): each
//              frame's object-class pixels cut into 3-D regions
//              (ConnectedSemantics: 0.1 m cells, 26-connected; pixels without depth take their image region's), named by
//              front.py's tracking (MaxIoUTracker over 0.2 m voxels in a 3 s window, then re-identification: a cluster whose
//              voxels lie by half in an earlier track's), an id written once its surface seen again is measured more often
//              than seen through; re-identification by half of the smaller of cluster and track, and a cluster
//              overlapping two tracks of its class by half of the smaller set makes them one (the older id stays). Pixel
//              by pixel the Python front's output.
//   PREDICTED  predicted semantics (a segmentation network's masks) only: SEMANTIC with online merging (a cluster overlapping
//              two tracks of its class by half of the smaller set makes them one; the confirmed track keeps its id), a
//              confirmation that also needs a new look at the track's own id and a surface that is not another class's
//              object seen at least as long (surface cells, see Cell), and the map's retired identities (<state>.retired).
// The input with annotated instances needs no front (OFF). KHRONOS_ONLINE_FRONT=off|semantic|predicted,
// KHRONOS_ONLINE_FRONT_STATE (state file), KHRONOS_ONLINE_FRONT_INTRINSICS (the recording's Intrinsics.txt) and
// KHRONOS_ONLINE_FRONT_STEP (its pixels per input pixel, 2 at image scale 0.5).
#include <cstdint>
#include <map>
#include <atomic>
#include <mutex>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace khronos {

/**
 * Identity merges of the online front (SEMANTIC, PREDICTED), process-wide: the front registers that output id `gone` is the
 * object of id `keep` (gone > keep) the moment it decides it; the map applies them at its next round (its states, its
 * nodes, its stored per-pixel identities read through root()). Empty unless the online front merges.
 */
class PhysicalIdAliases {
 public:
  static PhysicalIdAliases& instance();
  void add(size_t gone, size_t keep);
  size_t root(size_t id) const;
  // The merges registered since the last call, in order.
  std::vector<std::pair<size_t, size_t>> takeNew();
  bool empty() const;
  bool active() const { return active_.load(std::memory_order_acquire); }

 private:
  std::atomic<bool> active_{false};
  mutable std::mutex mutex_;
  std::map<size_t, size_t> alias_;
  std::vector<std::pair<size_t, size_t>> fresh_;
};

class OnlineFront {
 public:
  enum class Mode { OFF, SEMANTIC, PREDICTED };
  struct Intrinsics {
    int width = 0, height = 0;
    double fx = 0, fy = 0, cx = 0, cy = 0, offset = 0;
  };
  struct Config {
    Mode mode = Mode::OFF;
    std::vector<int> object_semantics;  // the label space's object labels without its dynamic ones
    double max_range = 5.0, min_range = 0.1;
    int min_cluster_size = 12;           // 50 px at full resolution, every 2nd pixel
    double voxel_size = 0.2, min_semantic_iou = 0.25, temporal_window_s = 3.0, min_reid_share = 0.5;
    double grid = 0.1;                   // ConnectedSemantics cells
    double surface_tolerance = 0.05;     // the map's voxel (naming: on the surface vs seen through)
    std::string state_path;
  };

  // From the environment (nullptr when KHRONOS_ONLINE_FRONT is unset or off); object classes from the label space.
  static std::unique_ptr<OnlineFront> fromEnvironment(const std::vector<int>& object_semantics, double map_voxel);
  OnlineFront(const Config& config, const Intrinsics& intrinsics);
  ~OnlineFront();  // saves the state

  // One frame, in time order: depth (z, metres, CV_32FC1), semantics (CV_32SC1), camera pose in the world.
  // Returns the frame's physical ids (CV_32SC1, 0 none) and writes each id pixel's class into `semantics`.
  cv::Mat process(const cv::Mat& depth, cv::Mat& semantics, const Eigen::Isometry3d& world_T_camera, int64_t stamp_ns);
  void save() const;
  const Config& config() const { return config_; }

 private:
  struct Track {
    int id = 0, semantic = 0;
    int64_t last_seen = 0, first_seen = 0;
    bool written = false;  // its id has reached the map: from then on it is never merged into another track
    std::unordered_set<int64_t> last_voxels, all_voxels;
    int observations = 0;
  };
  struct Cluster {
    int semantic = 0;
    std::vector<int> pixels;  // row-major indices
    std::unordered_set<int64_t> voxels;
  };
  using Points = std::vector<Eigen::Vector3d>;

  void judge(const cv::Mat& depth, const Eigen::Isometry3d& world_T_camera);
  // PREDICTED only (never called in the other modes)
  // Surface ownership (11-22). A surface cell (surface_tolerance) belongs to the first confirmed track whose written pixels
  // reach it. A track confirmed in an earlier visit keeps the cells it held then and takes no new ones (what is new on a
  // surface in a later visit is not the old object); a cell whose surface is seen through more often than measured again
  // belongs to no one (what stood there moved or went), the rule that names a track. A provisional track whose surface lies
  // by more than half on cells of a track of another class observed in at least as many frames is that track's surface
  // under another name and is not confirmed; a cell of an earlier visit is met within one cell (the visits are aligned to
  // a map voxel). A track confirmed over the cells of an object observed in fewer frames takes them (a first name seen
  // briefly gives way to the object seen longer). Kept between visits.
  struct Cell {
    int owner = -1;
    int visit = 0;                                    // the visit its owner took it in
    int hits = 0, through = 0;                        // its surface measured again / seen through
    Eigen::Vector3f point = Eigen::Vector3f::Zero();  // where the owner's surface was measured
  };

  cv::Mat processPredicted(const cv::Mat& depth, cv::Mat& semantics, const Eigen::Isometry3d& world_T_camera,
                           int64_t stamp_ns);
  void judgePredicted(const cv::Mat& depth, const cv::Mat& observed_ids, const Eigen::Isometry3d& world_T_camera);
  bool onOtherSurface(const Points& surface, int candidate) const;
  int64_t seen(int track) const;  // the frames it was observed in
  void release(const cv::Mat& depth, const Eigen::Isometry3d& world_T_camera);
  cv::Mat segments(const cv::Mat& depth, const cv::Mat& semantics, const Eigen::Isometry3d& world_T_camera) const;
  std::vector<Cluster> clusters(const cv::Mat& depth, const cv::Mat& segments, const cv::Mat& semantics,
                                const Eigen::Isometry3d& world_T_camera) const;
  std::vector<std::pair<int, int>> associate(const std::vector<Cluster>& clusters, int64_t stamp);
  int root(int t) const;
  Eigen::Vector3d backproject(int u, int v, float z, const Eigen::Isometry3d& world_T_camera) const;
  void load();

  Config config_;
  Intrinsics K_;
  std::set<int> objects_;
  std::vector<Track> tracks_;
  std::set<int> named_;
  std::vector<std::pair<int, Points>> pending_;          // insertion order, as the Python dict
  std::map<int, std::pair<int64_t, int64_t>> evidence_;  // track -> (supported, seen through)
  std::map<int, int> alias_;                             // merged track -> the track it joined
  // PREDICTED only
  std::unordered_map<int64_t, Cell> cells_;              // surface cell -> its owner
  std::map<int, int> confirmed_visit_;                   // confirmed track -> the visit it was confirmed in
  std::map<int, int64_t> observations_;                  // track -> frames it got a cluster in (merged ones summed)
  int visit_ = 0;                                        // this visit: the number of visits before it in the state
  // tracks the map closed at an earlier visit's end as another object's surface named wrongly (<state>.retired, written
  // by the map): never written again, never merged
  std::set<int> retired_;
};

}  // namespace khronos
