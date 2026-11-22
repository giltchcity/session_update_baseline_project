#include "khronos/active_window/online_front.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <array>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <sstream>

#include <iostream>

#include <glog/logging.h>

namespace khronos {
namespace {

// _place_keys of front.py: floor(p / voxel) + 2^20 per axis, packed in 21 bits each.
inline int64_t key(const Eigen::Vector3d& p, double voxel) {
  const int64_t x = static_cast<int64_t>(std::floor(p.x() / voxel)) + (int64_t(1) << 20);
  const int64_t y = static_cast<int64_t>(std::floor(p.y() / voxel)) + (int64_t(1) << 20);
  const int64_t z = static_cast<int64_t>(std::floor(p.z() / voxel)) + (int64_t(1) << 20);
  return (x << 42) | (y << 21) | z;
}

inline size_t intersection(const std::unordered_set<int64_t>& a, const std::unordered_set<int64_t>& b) {
  const auto& small = a.size() <= b.size() ? a : b;
  const auto& large = a.size() <= b.size() ? b : a;
  size_t n = 0;
  for (const auto k : small) n += large.count(k);
  return n;
}

// Connected components with ndimage.label's numbering: components numbered 1.. in the order of their first element
// in the scan order of the elements (`order` lists the elements in that order; `neighbours` yields an element's).
template <typename Neighbours>
std::vector<int> label(size_t n, const Neighbours& neighbours) {
  std::vector<int> parent(n);
  std::iota(parent.begin(), parent.end(), 0);
  std::function<int(int)> find = [&](int i) {
    while (parent[i] != i) i = parent[i] = parent[parent[i]];
    return i;
  };
  for (size_t i = 0; i < n; ++i) {
    neighbours(i, [&](int j) {
      const int a = find(static_cast<int>(i)), b = find(j);
      if (a != b) parent[std::max(a, b)] = std::min(a, b);
    });
  }
  std::vector<int> out(n, 0);
  std::unordered_map<int, int> number;
  for (size_t i = 0; i < n; ++i) {
    const int r = find(static_cast<int>(i));
    auto it = number.find(r);
    if (it == number.end()) it = number.emplace(r, static_cast<int>(number.size()) + 1).first;
    out[i] = it->second;
  }
  return out;
}

}  // namespace

PhysicalIdAliases& PhysicalIdAliases::instance() {
  static PhysicalIdAliases aliases;
  return aliases;
}

void PhysicalIdAliases::add(size_t gone, size_t keep) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (gone == keep || gone == 0 || keep == 0) return;
  alias_[gone] = keep;
  fresh_.emplace_back(gone, keep);
  active_.store(true, std::memory_order_release);
}

size_t PhysicalIdAliases::root(size_t id) const {
  if (!active()) return id;
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = alias_.find(id); it != alias_.end(); it = alias_.find(id)) id = it->second;
  return id;
}

std::vector<std::pair<size_t, size_t>> PhysicalIdAliases::takeNew() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::pair<size_t, size_t>> out;
  out.swap(fresh_);
  return out;
}

bool PhysicalIdAliases::empty() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return alias_.empty();
}

std::unique_ptr<OnlineFront> OnlineFront::fromEnvironment(const std::vector<int>& object_semantics, double map_voxel) {
  const char* mode = std::getenv("KHRONOS_ONLINE_FRONT");
  if (!mode || std::string(mode).empty() || std::string(mode) == "off") return nullptr;
  Config config;
  if (std::string(mode) == "semantic") {
    config.mode = Mode::SEMANTIC;
  } else if (std::string(mode) == "semantic_merge") {
    config.mode = Mode::SEMANTIC_MERGE;
  } else if (std::string(mode) == "predicted") {
    config.mode = Mode::PREDICTED;
  } else {
    LOG(FATAL) << "KHRONOS_ONLINE_FRONT must be off, semantic, semantic_merge or predicted, got '" << mode << "'";
  }
  config.object_semantics = object_semantics;
  config.surface_tolerance = map_voxel;
  if (const char* f = std::getenv("KHRONOS_ONLINE_FRONT_FUSE")) config.fuse = std::atoi(f) != 0;
  config.fuse_voxel = map_voxel;
  if (const char* m = std::getenv("KHRONOS_ONLINE_FRONT_MIN_OBS")) config.min_observations = std::atoi(m);
  if (const char* m = std::getenv("KHRONOS_ONLINE_FRONT_MIN_PLACES")) config.min_places = std::atoi(m);
  const char* state = std::getenv("KHRONOS_ONLINE_FRONT_STATE");
  const char* intrinsics = std::getenv("KHRONOS_ONLINE_FRONT_INTRINSICS");
  const char* step = std::getenv("KHRONOS_ONLINE_FRONT_STEP");
  CHECK(state && intrinsics) << "KHRONOS_ONLINE_FRONT needs KHRONOS_ONLINE_FRONT_STATE and _INTRINSICS";
  config.state_path = state;
  const int s = step ? std::atoi(step) : 1;
  config.min_cluster_size = 50 / (s * s);
  // Intrinsics.txt ("key: value" lines), subsampled by the step as frames.Intrinsics.scaled.
  std::map<std::string, double> values;
  std::ifstream in(intrinsics);
  CHECK(in) << "cannot read " << intrinsics;
  for (std::string line; std::getline(in, line);) {
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    auto name = line.substr(0, colon);
    name.erase(0, name.find_first_not_of(" \t"));
    name.erase(name.find_last_not_of(" \t") + 1);
    values[name] = std::stod(line.substr(colon + 1));
  }
  Intrinsics K;
  K.width = static_cast<int>(values.at("Res_x")) / s;
  K.height = static_cast<int>(values.at("Res_y")) / s;
  K.fx = values.at("f_x") / s;
  K.fy = values.at("f_y") / s;
  K.cx = values.at("u") / s;
  K.cy = values.at("v") / s;
  LOG(WARNING) << "ONLINE FRONT: mode " << mode << (config.fuse ? " + semantic fusion (decaying votes)" : "") << ", min observations " << config.min_observations << ", min places " << config.min_places << ", " << object_semantics.size() << " object classes, step " << s
               << ", state " << config.state_path;
  return std::make_unique<OnlineFront>(config, K);
}

OnlineFront::OnlineFront(const Config& config, const Intrinsics& intrinsics)
    : config_(config), K_(intrinsics), objects_(config.object_semantics.begin(), config.object_semantics.end()) {
  load();
}

OnlineFront::~OnlineFront() { save(); }

Eigen::Vector3d OnlineFront::backproject(int u, int v, float z, const Eigen::Isometry3d& T) const {
  const double zd = static_cast<double>(z);
  const Eigen::Vector3d cam((u + K_.offset - K_.cx) / K_.fx * zd, (v + K_.offset - K_.cy) / K_.fy * zd, zd);
  return T.linear() * cam + T.translation();
}

int OnlineFront::root(int t) const {
  for (auto it = alias_.find(t); it != alias_.end(); it = alias_.find(t)) t = it->second;
  return t;
}

void OnlineFront::judge(const cv::Mat& depth, const Eigen::Isometry3d& T) {
  // SemanticFront._judge: a pending track's surface (the mean point per 0.05 m place of what it got) seen again.
  const Eigen::Matrix3d Rt = T.linear().transpose();
  const Eigen::Vector3d t = T.translation();
  const double tol = config_.surface_tolerance;
  std::vector<int> done;
  for (auto& [r, P] : pending_) {
    std::map<int64_t, std::pair<Eigen::Vector3d, int>> places;
    for (const auto& p : P) {
      auto it = places.find(key(p, config_.surface_tolerance));
      if (it == places.end()) it = places.emplace(key(p, config_.surface_tolerance), std::make_pair(Eigen::Vector3d::Zero().eval(), 0)).first;
      auto& e = it->second;
      e.first += p;
      e.second += 1;
    }
    Points S;
    S.reserve(places.size());
    for (const auto& [k, e] : places) S.push_back(e.first / e.second);
    P = S;
    auto& ev = evidence_[r];
    for (const auto& s : S) {
      const Eigen::Vector3d cam = Rt * (s - t);
      const double z = cam.z();
      const double uf = std::floor(K_.fx * cam.x() / z + K_.cx - K_.offset + 0.5);
      const double vf = std::floor(K_.fy * cam.y() / z + K_.cy - K_.offset + 0.5);
      if (!(z > 0) || !std::isfinite(uf) || !std::isfinite(vf) || uf < 0 || uf >= K_.width || vf < 0 ||
          vf >= K_.height || !(z <= config_.max_range)) {
        continue;
      }
      const double d = depth.at<float>(static_cast<int>(vf), static_cast<int>(uf));
      if (!std::isfinite(d)) continue;
      if (std::abs(d - z) <= tol) ++ev.first;
      if (d > z + tol) ++ev.second;
    }
    if (std::getenv("KHRONOS_ONLINE_FRONT_DEBUG"))
      std::cerr << "judge track " << r << ": " << P.size() << " places, evidence " << ev.first << "/" << ev.second << "\n";
    int observed = 0;  // the track's frames with the tracks merged into it
    if (config_.min_observations > 0)
      for (size_t i = 0; i < tracks_.size(); ++i)
        if (root(static_cast<int>(i)) == r) observed += tracks_[i].observations;
    if (ev.first > ev.second && observed >= config_.min_observations && static_cast<int>(P.size()) >= config_.min_places) {
      named_.insert(r);
      done.push_back(r);
    }
  }
  for (const int r : done) {
    pending_.erase(std::remove_if(pending_.begin(), pending_.end(), [r](const auto& e) { return e.first == r; }),
                   pending_.end());
  }
}

void OnlineFront::judgePredicted(const cv::Mat& depth, const cv::Mat& observed_ids, const Eigen::Isometry3d& T) {
  // PREDICTED: the confirmation of SEMANTIC_MERGE, and a new look at the track's own id, and not on another class's surface.
  // SemanticFront._judge: a pending track's surface (the mean point per 0.05 m place of what it got) seen again.
  const Eigen::Matrix3d Rt = T.linear().transpose();
  const Eigen::Vector3d t = T.translation();
  const double tol = config_.surface_tolerance;
  std::vector<int> done;
  for (auto& [r, P] : pending_) {
    const int candidate = root(r);
    if (named_.count(candidate)) {
      done.push_back(r);
      continue;
    }
    std::map<int64_t, std::pair<Eigen::Vector3d, int>> places;
    for (const auto& p : P) {
      auto it = places.find(key(p, config_.surface_tolerance));
      if (it == places.end()) it = places.emplace(key(p, config_.surface_tolerance), std::make_pair(Eigen::Vector3d::Zero().eval(), 0)).first;
      auto& e = it->second;
      e.first += p;
      e.second += 1;
    }
    Points S;
    S.reserve(places.size());
    for (const auto& [k, e] : places) S.push_back(e.first / e.second);
    P = S;
    auto& ev = evidence_[r];
    size_t same_identity = 0, other_or_unassigned = 0;
    for (const auto& s : S) {
      const Eigen::Vector3d cam = Rt * (s - t);
      const double z = cam.z();
      const double uf = std::floor(K_.fx * cam.x() / z + K_.cx - K_.offset + 0.5);
      const double vf = std::floor(K_.fy * cam.y() / z + K_.cy - K_.offset + 0.5);
      if (!(z > 0) || !std::isfinite(uf) || !std::isfinite(vf) || uf < 0 || uf >= K_.width || vf < 0 ||
          vf >= K_.height || !(z <= config_.max_range)) {
        continue;
      }
      const double d = depth.at<float>(static_cast<int>(vf), static_cast<int>(uf));
      if (!std::isfinite(d)) continue;
      if (std::abs(d - z) <= tol) {
        ++ev.first;
        const int observed = observed_ids.at<int32_t>(static_cast<int>(vf), static_cast<int>(uf));
        if (observed == candidate + 1) {
          ++same_identity;
        } else {
          ++other_or_unassigned;
        }
      }
      if (d > z + tol) ++ev.second;
    }
    const bool other_surface = onOtherSurface(S, candidate);
    if (std::getenv("KHRONOS_ONLINE_FRONT_DEBUG"))
      std::cerr << "judge track " << r << " (class " << tracks_[candidate].semantic << "): " << P.size()
                << " places, evidence " << ev.first << "/" << ev.second << ", identity " << same_identity << "/"
                << other_or_unassigned << (other_surface ? ", on another object's surface" : "") << "\n";
    // A depth hit confirms geometry. Only a new look at this track can confirm its identity, and only on a surface that
    // no object of another class seen at least as long holds (there the track is that object under another name).
    if (ev.first > ev.second && same_identity > other_or_unassigned && !other_surface) {
      named_.insert(candidate);
      confirmed_visit_[candidate] = visit_;
      done.push_back(r);
      // the cells of its surface held by an object of another class seen less long are its now
      const int semantic = tracks_[candidate].semantic;
      for (const auto& s : S) {
        const auto it = cells_.find(key(s, tol));
        if (it == cells_.end() || it->second.owner < 0) continue;
        const int owner = root(it->second.owner);
        if (owner == candidate || tracks_[owner].semantic == semantic || seen(owner) >= seen(candidate)) continue;
        it->second.owner = candidate, it->second.visit = visit_, it->second.hits = 1, it->second.through = 0;
        it->second.writes = 1, it->second.rival = -1, it->second.rival_writes = 0;
      }
    }
  }
  for (const int r : done) {
    pending_.erase(std::remove_if(pending_.begin(), pending_.end(), [r](const auto& e) { return e.first == r; }),
                   pending_.end());
  }
}

int64_t OnlineFront::seen(int t) const {
  const auto it = observations_.find(t);
  return it == observations_.end() ? 0 : it->second;
}

bool OnlineFront::onOtherSurface(const Points& surface, int candidate) const {
  if (surface.empty() || cells_.empty()) return false;
  const int semantic = tracks_[candidate].semantic;
  const double tol = config_.surface_tolerance;
  const int64_t looks = seen(candidate);
  const auto other_class = [&](const Cell& c) {  // an object of another class, seen at least as long as the candidate
    if (c.owner < 0) return false;
    const int owner = root(c.owner);
    return owner != candidate && tracks_[owner].semantic != semantic && seen(owner) >= looks;
  };
  size_t other = 0;
  for (const auto& s : surface) {
    const auto it = cells_.find(key(s, tol));
    if (it != cells_.end() && it->second.owner >= 0) {
      other += other_class(it->second);
      continue;
    }
    // a free cell next to a cell an earlier visit gave to another class's object: that surface, seen through the
    // visits' alignment
    bool near = false;
    for (int dx = -1; dx <= 1 && !near; ++dx)
      for (int dy = -1; dy <= 1 && !near; ++dy)
        for (int dz = -1; dz <= 1 && !near; ++dz) {
          const auto n = cells_.find(key(s + Eigen::Vector3d(dx, dy, dz) * tol, tol));
          near = n != cells_.end() && n->second.visit < visit_ && other_class(n->second);
        }
    other += near;
  }
  return other > config_.min_reid_share * surface.size();
}

void OnlineFront::release(const cv::Mat& depth, const Eigen::Isometry3d& T) {
  // a held cell seen through more often than measured again belongs to no one any more
  const Eigen::Matrix3d Rt = T.linear().transpose();
  const Eigen::Vector3d t = T.translation();
  const double tol = config_.surface_tolerance;
  for (auto& [k, c] : cells_) {
    if (c.owner < 0) continue;
    const Eigen::Vector3d cam = Rt * (c.point.cast<double>() - t);
    const double z = cam.z();
    if (!(z > 0) || !(z <= config_.max_range)) continue;
    const double uf = std::floor(K_.fx * cam.x() / z + K_.cx - K_.offset + 0.5);
    const double vf = std::floor(K_.fy * cam.y() / z + K_.cy - K_.offset + 0.5);
    if (!(uf >= 0 && uf < K_.width && vf >= 0 && vf < K_.height)) continue;
    const float d = depth.at<float>(static_cast<int>(vf), static_cast<int>(uf));
    if (!std::isfinite(d)) continue;
    if (std::abs(d - z) <= tol) ++c.hits;
    if (d > z + tol) ++c.through;
    if (c.through > c.hits) c.owner = -1, c.hits = c.through = c.writes = c.rival_writes = 0, c.rival = -1;
  }
}

void OnlineFront::fuse(const cv::Mat& depth, cv::Mat& sem, const Eigen::Isometry3d& T, int64_t stamp) {
  // the votes before this frame decide (decayed to now, time constant fuse_tau_s); this frame's classes are added after
  const auto decay = [&](Votes& e) {
    if (e.stamp == stamp) return;
    const double f = std::exp(-static_cast<double>(stamp - e.stamp) * 1e-9 / config_.fuse_tau_s);
    for (auto& [cls, w] : e.weight) w *= f;
    e.stamp = stamp;
  };
  std::vector<std::pair<int64_t, int>> seen;
  for (int v = 0; v < depth.rows; ++v)
    for (int u = 0; u < depth.cols; ++u) {
      const float z = depth.at<float>(v, u);
      int32_t& c = sem.at<int32_t>(v, u);
      if (!std::isfinite(z) || !objects_.count(c)) continue;
      const int64_t k = key(backproject(u, v, z, T), config_.fuse_voxel);
      seen.emplace_back(k, c);
      const auto it = votes_.find(k);
      if (it == votes_.end()) continue;
      decay(it->second);
      double total = 0, own = 0, most = -1;
      int best = c;
      for (const auto& [cls, w] : it->second.weight) {
        total += w;
        if (cls == c) own = w;
        if (w > most || (w == most && cls < best)) best = cls, most = w;
      }
      if (total >= config_.fuse_min && best != c && most > own) c = best;
    }
  for (const auto& [k, c] : seen) {
    auto& e = votes_[k];
    if (e.weight.empty()) e.stamp = stamp; else decay(e);
    auto it = std::find_if(e.weight.begin(), e.weight.end(), [c = c](const auto& x) { return x.first == c; });
    if (it == e.weight.end()) e.weight.emplace_back(c, 1.0); else it->second += 1.0;
  }
}

cv::Mat OnlineFront::segments(const cv::Mat& depth, const cv::Mat& sem, const Eigen::Isometry3d& T) const {
  // segments_3d: per object class, its pixels with depth in 0.1 m cells, 26-connected; a pixel without depth takes the
  // region (8-connected in the image) its depth pixels got most.
  const int H = depth.rows, W = depth.cols;
  cv::Mat out = cv::Mat::zeros(H, W, CV_32SC1);
  int n = 0;
  for (const int c : config_.object_semantics) {
    std::vector<int> pix;
    std::vector<std::array<int64_t, 3>> cells;
    for (int v = 0; v < H; ++v) {
      for (int u = 0; u < W; ++u) {
        const float z = depth.at<float>(v, u);
        if (sem.at<int32_t>(v, u) != c || !std::isfinite(z)) continue;
        const Eigen::Vector3d p = backproject(u, v, z, T);
        pix.push_back(v * W + u);
        cells.push_back({static_cast<int64_t>(std::floor(p.x() / config_.grid)),
                         static_cast<int64_t>(std::floor(p.y() / config_.grid)),
                         static_cast<int64_t>(std::floor(p.z() / config_.grid))});
      }
    }
    if (pix.empty()) continue;
    // the occupied cells in C order (x, then y, then z), labelled as ndimage.label on the dense grid
    std::vector<std::array<int64_t, 3>> unique = cells;
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    std::map<std::array<int64_t, 3>, int> index;
    for (size_t i = 0; i < unique.size(); ++i) index[unique[i]] = static_cast<int>(i);
    const auto lab = label(unique.size(), [&](size_t i, const auto& join) {
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dz = -1; dz <= 1; ++dz) {
            if (!dx && !dy && !dz) continue;
            const auto it = index.find({unique[i][0] + dx, unique[i][1] + dy, unique[i][2] + dz});
            if (it != index.end()) join(it->second);
          }
    });
    int k = 0;
    for (size_t i = 0; i < pix.size(); ++i) {
      const int l = lab[index.at(cells[i])];
      k = std::max(k, l);
      out.at<int32_t>(pix[i] / W, pix[i] % W) = l + n;
    }
    // image regions of the class (8-connected), for its pixels without depth
    std::vector<int> cls;
    for (int i = 0; i < H * W; ++i)
      if (sem.at<int32_t>(i / W, i % W) == c) cls.push_back(i);
    std::unordered_map<int, int> where;
    for (size_t i = 0; i < cls.size(); ++i) where[cls[i]] = static_cast<int>(i);
    const auto reg = label(cls.size(), [&](size_t i, const auto& join) {
      const int v = cls[i] / W, u = cls[i] % W;
      for (int dv = -1; dv <= 1; ++dv)
        for (int du = -1; du <= 1; ++du) {
          if (!dv && !du) continue;
          const int vv = v + dv, uu = u + du;
          if (vv < 0 || vv >= H || uu < 0 || uu >= W) continue;
          const auto it = where.find(vv * W + uu);
          if (it != where.end()) join(it->second);
        }
    });
    std::map<int, std::map<int, int>> votes;  // region -> segment -> depth pixels
    std::map<int, std::vector<int>> holes;    // region -> pixels without depth
    for (size_t i = 0; i < cls.size(); ++i) {
      const int v = cls[i] / W, u = cls[i] % W;
      if (std::isfinite(depth.at<float>(v, u))) {
        votes[reg[i]][out.at<int32_t>(v, u)] += 1;
      } else {
        holes[reg[i]].push_back(cls[i]);
      }
    }
    for (const auto& [r, px] : holes) {
      const auto it = votes.find(r);
      if (it == votes.end()) continue;
      int best = 0, most = -1;
      for (const auto& [s, count] : it->second)  // ascending segment: bincount's argmax takes the smallest on ties
        if (count > most) best = s, most = count;
      for (const int p : px) out.at<int32_t>(p / W, p % W) = best;
    }
    n += k;
  }
  return out;
}

std::vector<OnlineFront::Cluster> OnlineFront::clusters(const cv::Mat& depth, const cv::Mat& seg, const cv::Mat& sem,
                                                        const Eigen::Isometry3d& T) const {
  // InstanceForwarding (front.py _clusters): one cluster per segment of an object class, pixels with depth.
  const int H = depth.rows, W = depth.cols;
  std::map<int, std::vector<int>> by_segment;
  for (int v = 0; v < H; ++v)
    for (int u = 0; u < W; ++u) {
      const int s = seg.at<int32_t>(v, u);
      const float z = depth.at<float>(v, u);
      if (s > 0 && std::isfinite(z) && z <= config_.max_range) by_segment[s].push_back(v * W + u);
    }
  std::vector<Cluster> out;
  for (auto& [s, px] : by_segment) {
    if (static_cast<int>(px.size()) < config_.min_cluster_size) continue;
    std::map<int, int> counts;
    for (const int p : px) counts[sem.at<int32_t>(p / W, p % W)] += 1;
    int c = 0, most = -1;
    for (const auto& [label, count] : counts)
      if (count > most) c = label, most = count;
    if (!objects_.count(c)) continue;
    Cluster cl;
    cl.semantic = c;
    cl.pixels = px;
    for (const int p : px)
      cl.voxels.insert(key(backproject(p % W, p / W, depth.at<float>(p / W, p % W), T), config_.voxel_size));
    out.push_back(std::move(cl));
  }
  return out;
}

std::vector<std::pair<int, int>> OnlineFront::associate(const std::vector<Cluster>& clusters, int64_t stamp) {
  // SEMANTIC: front.py steps 2-3 (front_real.py). SEMANTIC_MERGE: front_real3.py: re-identification by half of the
  // smaller of cluster and track, and online merging (a cluster overlapping two tracks of its class by half of the smaller
  // set makes them one; every merge is recorded with its stamp, for a fixed-lag output).
  // PREDICTED: SEMANTIC_MERGE, with re-identification by the cluster's own share in a track (a cluster that only covers half
  // of a smaller track is not that track) and merges in which the confirmed track keeps its id (joining never confirms).
  const bool predicted = config_.mode == Mode::PREDICTED;
  const bool merge = config_.mode == Mode::SEMANTIC_MERGE || predicted;
  std::vector<std::pair<int, int>> claimed;
  std::vector<char> taken(clusters.size(), 0);
  const int64_t window = stamp - static_cast<int64_t>(config_.temporal_window_s * 1e9);
  for (size_t ti = 0; ti < tracks_.size(); ++ti) {
    const auto& t = tracks_[ti];
    if (t.last_seen < window) continue;
    int best = -1;
    double best_iou = config_.min_semantic_iou;
    for (size_t ci = 0; ci < clusters.size(); ++ci) {
      if (taken[ci] || clusters[ci].semantic != t.semantic) continue;
      const double inter = static_cast<double>(intersection(clusters[ci].voxels, t.last_voxels));
      const double iou = inter / (clusters[ci].voxels.size() + t.last_voxels.size() - inter);
      if (iou > best_iou) best = static_cast<int>(ci), best_iou = iou;
    }
    if (best >= 0) {
      taken[best] = 1;
      claimed.emplace_back(best, static_cast<int>(ti));
    }
  }
  for (size_t ci = 0; ci < clusters.size(); ++ci) {
    if (taken[ci]) continue;
    const auto& c = clusters[ci];
    int best = -1;
    double best_share = config_.min_reid_share;
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
      const auto& t = tracks_[ti];
      if (t.semantic != c.semantic) continue;
      if (merge && t.all_voxels.empty()) continue;
      const double inter = static_cast<double>(intersection(c.voxels, t.all_voxels));
      const double share = merge && !predicted ? inter / std::min(c.voxels.size(), t.all_voxels.size()) : inter / c.voxels.size();
      if (share >= best_share) best = static_cast<int>(ti), best_share = share;
    }
    if (best < 0) {
      Track t;
      t.id = static_cast<int>(tracks_.size());
      t.semantic = c.semantic;
      t.last_seen = stamp;
      t.first_seen = stamp;
      tracks_.push_back(std::move(t));
      best = static_cast<int>(tracks_.size()) - 1;
    }
    taken[ci] = 1;
    claimed.emplace_back(static_cast<int>(ci), best);
  }
  if (!merge) return claimed;
  for (const auto& [ci, ti] : claimed) {
    const auto& c = clusters[ci];
    int r = root(ti);
    for (size_t ui = 0; ui < tracks_.size(); ++ui) {
      const int ru = root(static_cast<int>(ui));
      const auto& u = tracks_[ui];
      if (ru == r || u.semantic != c.semantic || u.all_voxels.empty()) continue;
      if (predicted && (retired_.count(r) || retired_.count(ru))) continue;
      if (intersection(c.voxels, u.all_voxels) >= config_.min_reid_share * std::min(c.voxels.size(), u.all_voxels.size())) {
        int keep = std::min(r, ru), gone = std::max(r, ru);
        if (predicted) {  // the confirmed track keeps its id, of two alike the older
          const bool r_named = named_.count(r) > 0, u_named = named_.count(ru) > 0;
          keep = r_named != u_named ? (r_named ? r : ru) : std::min(r, ru), gone = keep == r ? ru : r;
        }
        alias_[gone] = keep;
        merges_.push_back({stamp, gone, keep});
        PhysicalIdAliases::instance().add(static_cast<size_t>(gone) + 1, static_cast<size_t>(keep) + 1);
        const auto gone_voxels = tracks_[gone].all_voxels;
        tracks_[keep].all_voxels.insert(gone_voxels.begin(), gone_voxels.end());
        if (!predicted && named_.count(gone)) named_.insert(keep);
        if (predicted)
          if (const auto o = observations_.find(gone); o != observations_.end()) {
            observations_[keep] += o->second;
            observations_.erase(gone);
          }
        r = keep;
      }
    }
  }
  for (auto& [ci, ti] : claimed) ti = root(ti);
  return claimed;
}

cv::Mat OnlineFront::process(const cv::Mat& depth_in, cv::Mat& semantics, const Eigen::Isometry3d& T, int64_t stamp) {
  if (config_.mode == Mode::PREDICTED) return processPredicted(depth_in, semantics, T, stamp);
  CHECK(depth_in.type() == CV_32FC1 && semantics.type() == CV_32SC1);
  cv::Mat depth = depth_in.clone();  // FlatSession: depth outside [min_range, max_range] is no reading
  for (int v = 0; v < depth.rows; ++v)
    for (int u = 0; u < depth.cols; ++u) {
      float& z = depth.at<float>(v, u);
      if (!std::isfinite(z) || z < config_.min_range || z > config_.max_range) z = std::numeric_limits<float>::quiet_NaN();
    }
  judge(depth, T);
  cv::Mat fused = semantics;
  if (config_.fuse) {
    fused = semantics.clone();
    fuse(depth, fused, T, stamp);
  }
  const cv::Mat seg = segments(depth, fused, T);
  const auto cl = clusters(depth, seg, fused, T);
  const auto claimed = associate(cl, stamp);
  cv::Mat out = cv::Mat::zeros(depth.size(), CV_32SC1);
  const int W = depth.cols;
  for (const auto& [ci, ti] : claimed) {
    auto& t = tracks_[ti];
    t.last_seen = stamp;
    t.last_voxels = cl[ci].voxels;
    t.all_voxels.insert(cl[ci].voxels.begin(), cl[ci].voxels.end());
    t.observations += 1;
    for (const int p : cl[ci].pixels) out.at<int32_t>(p / W, p % W) = t.id + 1;
  }
  // SemanticFront.process: an unnamed track's measured points wait for its naming; only named tracks are written
  std::map<int, Points> fresh;
  for (int v = 0; v < depth.rows; ++v)
    for (int u = 0; u < W; ++u) {
      const int id = out.at<int32_t>(v, u);
      const float z = depth.at<float>(v, u);
      if (id <= 0 || !(z <= config_.max_range)) continue;
      if (named_.count(id - 1)) continue;
      fresh[id - 1].push_back(backproject(u, v, z, T));
    }
  for (auto& [t, P] : fresh) {
    auto it = std::find_if(pending_.begin(), pending_.end(), [t = t](const auto& e) { return e.first == t; });
    if (it == pending_.end()) {
      pending_.emplace_back(t, std::move(P));
    } else {
      it->second.insert(it->second.end(), P.begin(), P.end());
    }
  }
  for (int v = 0; v < depth.rows; ++v)
    for (int u = 0; u < W; ++u) {
      int32_t& id = out.at<int32_t>(v, u);
      if (id > 0 && !named_.count(id - 1)) id = 0;
      if (id > 0) {
        semantics.at<int32_t>(v, u) = tracks_[id - 1].semantic;  // the catalog contract: the id's class
        tracks_[id - 1].written = true;
      }
    }
  return out;
}

cv::Mat OnlineFront::processPredicted(const cv::Mat& depth_in, cv::Mat& semantics, const Eigen::Isometry3d& T,
                                      int64_t stamp) {
  CHECK(depth_in.type() == CV_32FC1 && semantics.type() == CV_32SC1);
  cv::Mat depth = depth_in.clone();  // FlatSession: depth outside [min_range, max_range] is no reading
  for (int v = 0; v < depth.rows; ++v)
    for (int u = 0; u < depth.cols; ++u) {
      float& z = depth.at<float>(v, u);
      if (!std::isfinite(z) || z < config_.min_range || z > config_.max_range) z = std::numeric_limits<float>::quiet_NaN();
    }
  release(depth, T);
  const cv::Mat seg = segments(depth, semantics, T);
  const auto cl = clusters(depth, seg, semantics, T);
  const auto claimed = associate(cl, stamp);
  cv::Mat out = cv::Mat::zeros(depth.size(), CV_32SC1);
  const int W = depth.cols;
  std::set<int> looked;
  for (const auto& [ci, ti] : claimed) {
    auto& t = tracks_[ti];
    looked.insert(ti);
    t.last_seen = stamp;
    t.last_voxels = cl[ci].voxels;
    t.all_voxels.insert(cl[ci].voxels.begin(), cl[ci].voxels.end());
    for (const int p : cl[ci].pixels) out.at<int32_t>(p / W, p % W) = t.id + 1;
  }
  for (const int t : looked) ++observations_[t];  // one look per frame, however many clusters it got
  judgePredicted(depth, out, T);  // out still contains provisional ids; fresh points enter pending_ below.
  // SemanticFront.process: an unnamed track's measured points wait for its naming; only named tracks are written
  std::map<int, Points> fresh;
  for (int v = 0; v < depth.rows; ++v)
    for (int u = 0; u < W; ++u) {
      const int id = out.at<int32_t>(v, u);
      const float z = depth.at<float>(v, u);
      if (id <= 0 || !(z <= config_.max_range)) continue;
      if (named_.count(id - 1)) continue;
      fresh[id - 1].push_back(backproject(u, v, z, T));
    }
  for (auto& [t, P] : fresh) {
    auto it = std::find_if(pending_.begin(), pending_.end(), [t = t](const auto& e) { return e.first == t; });
    if (it == pending_.end()) {
      pending_.emplace_back(t, std::move(P));
    } else {
      it->second.insert(it->second.end(), P.begin(), P.end());
    }
  }
  for (int v = 0; v < depth.rows; ++v)
    for (int u = 0; u < W; ++u) {
      int32_t& id = out.at<int32_t>(v, u);
      if (id > 0 && (!named_.count(id - 1) || retired_.count(id - 1))) id = 0;
      if (id <= 0) continue;
      const int t = id - 1;
      const float z = depth.at<float>(v, u);
      if (std::isfinite(z)) {
        const Eigen::Vector3d x = backproject(u, v, z, T);
        const int64_t k = key(x, config_.surface_tolerance);
        auto it = cells_.find(k);
        if (it != cells_.end() && it->second.owner >= 0) {
          Cell& cell = it->second;
          const int owner = root(cell.owner);
          if (owner == t || tracks_[owner].semantic == tracks_[t].semantic) {
            if (owner == t) ++cell.writes;
          } else {
            // a cell of an object of another class: whichever way the cluster reached this pixel (the window,
            // re-identification or a merge), the id is written here only once it was given here more often than the
            // owner's (the rival is the other-class track given it most, by majority vote)
            if (cell.rival >= 0 && root(cell.rival) == t) {
              ++cell.rival_writes;
            } else if (cell.rival_writes == 0) {
              cell.rival = t, cell.rival_writes = 1;
            } else {
              --cell.rival_writes;
            }
            if (cell.rival < 0 || root(cell.rival) != t || cell.rival_writes <= cell.writes) {
              id = 0;
              continue;
            }
            cell.owner = t, cell.visit = visit_, cell.writes = cell.rival_writes, cell.rival = -1, cell.rival_writes = 0;
          }
        } else {
          // a free surface cell is taken by the first track confirmed in this visit that writes it
          const auto confirmed = confirmed_visit_.find(t);
          if (confirmed != confirmed_visit_.end() && confirmed->second == visit_) {
            Cell& cell = it != cells_.end() ? it->second : cells_[k];
            cell.owner = t, cell.visit = visit_, cell.hits = 1, cell.through = 0, cell.point = x.cast<float>();
            cell.writes = 1, cell.rival = -1, cell.rival_writes = 0;
          }
        }
      }
      semantics.at<int32_t>(v, u) = tracks_[t].semantic;  // the catalog contract: the id's class
    }
  return out;
}

void OnlineFront::save() const {
  if (config_.state_path.empty()) return;
  std::ofstream f(config_.state_path, std::ios::binary | std::ios::trunc);
  auto w = [&](int64_t x) { f.write(reinterpret_cast<const char*>(&x), sizeof(x)); };
  auto wd = [&](double x) { f.write(reinterpret_cast<const char*>(&x), sizeof(x)); };
  w(static_cast<int64_t>(tracks_.size()));
  for (const auto& t : tracks_) {
    w(t.id), w(t.semantic), w(t.last_seen), w(t.observations), w(t.first_seen), w(t.written ? 1 : 0);
    w(static_cast<int64_t>(t.last_voxels.size()));
    for (const auto k : t.last_voxels) w(k);
    w(static_cast<int64_t>(t.all_voxels.size()));
    for (const auto k : t.all_voxels) w(k);
  }
  w(static_cast<int64_t>(named_.size()));
  for (const int n : named_) w(n);
  w(static_cast<int64_t>(pending_.size()));
  for (const auto& [t, P] : pending_) {
    w(t), w(static_cast<int64_t>(P.size()));
    for (const auto& p : P) wd(p.x()), wd(p.y()), wd(p.z());
  }
  w(static_cast<int64_t>(evidence_.size()));
  for (const auto& [t, e] : evidence_) w(t), w(e.first), w(e.second);
  w(static_cast<int64_t>(alias_.size()));
  for (const auto& [a, b] : alias_) w(a), w(b);
  if (config_.mode == Mode::PREDICTED) {
    // surface ownership (appended: a state without it reads as none)
    w(static_cast<int64_t>(visit_ + 1));
    w(static_cast<int64_t>(confirmed_visit_.size()));
    for (const auto& [t, v] : confirmed_visit_) w(t), w(v);
    w(static_cast<int64_t>(cells_.size()));
    for (const auto& [k, c] : cells_)
      w(k), w(c.owner), w(c.visit), w(c.hits), w(c.through), w(c.writes), w(c.rival), w(c.rival_writes), wd(c.point.x()),
          wd(c.point.y()), wd(c.point.z());
    w(static_cast<int64_t>(observations_.size()));
    for (const auto& [t, n] : observations_) w(t), w(n);
  }

  // the catalog the front made, for the record (id -> class of its named tracks)
  std::ofstream c(config_.state_path + ".catalog.json");
  c << "{\"schema\": \"session_update_physical_catalog/v1\", \"policy\": \"online front (OnlineFront)\", "
       "\"semantic_id_range\": [0, 149], \"entities\": [";
  bool first = true;
  for (const auto& t : tracks_) {
    if (!named_.count(t.id)) continue;
    c << (first ? "" : ", ") << "{\"physical_instance_id\": " << t.id + 1 << ", \"semantic_id\": " << t.semantic
      << ", \"name\": \"front_" << t.id + 1 << "\"}";
    first = false;
  }
  c << "]}\n";
}

void OnlineFront::load() {
  std::ifstream f(config_.state_path, std::ios::binary);
  if (!f) return;
  auto r = [&]() { int64_t x = 0; f.read(reinterpret_cast<char*>(&x), sizeof(x)); return x; };
  auto rd = [&]() { double x = 0; f.read(reinterpret_cast<char*>(&x), sizeof(x)); return x; };
  tracks_.resize(r());
  for (auto& t : tracks_) {
    t.id = r(), t.semantic = r(), t.last_seen = r(), t.observations = r(), t.first_seen = r(), t.written = r() != 0;
    for (int64_t n = r(); n > 0; --n) t.last_voxels.insert(r());
    for (int64_t n = r(); n > 0; --n) t.all_voxels.insert(r());
  }
  for (int64_t n = r(); n > 0; --n) named_.insert(static_cast<int>(r()));
  for (int64_t n = r(); n > 0; --n) {
    const int t = r();
    Points P(r());
    for (auto& p : P) {
      const double x = rd(), y = rd(), z = rd();
      p = Eigen::Vector3d(x, y, z);
    }
    pending_.emplace_back(t, std::move(P));
  }
  for (int64_t n = r(); n > 0; --n) {
    const int t = r();
    const int64_t a = r(), b = r();
    evidence_[t] = {a, b};
  }
  for (int64_t n = r(); n > 0; --n) {
    const int a = r(), b = r();
    alias_[a] = b;
  }
  if (config_.mode == Mode::PREDICTED) {
    visit_ = static_cast<int>(r());
    for (int64_t n = r(); n > 0; --n) {
      const int t = r();
      confirmed_visit_[t] = static_cast<int>(r());
    }
    for (int64_t n = r(); n > 0; --n) {
      const int64_t k = r();
      Cell c;
      c.owner = static_cast<int>(r());
      c.visit = static_cast<int>(r());
      c.hits = static_cast<int>(r());
      c.through = static_cast<int>(r());
      c.writes = static_cast<int>(r());
      c.rival = static_cast<int>(r());
      c.rival_writes = static_cast<int>(r());
      const double x = rd(), y = rd(), z = rd();
      c.point = Eigen::Vector3f(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
      cells_.emplace(k, c);
    }
    for (int64_t n = r(); n > 0; --n) {
      const int t = r();
      observations_[t] = r();
    }
  }

  // an earlier visit's merges: its map already carries the merged ids; stored identities of this visit never had them
  for (const auto& [a, b] : alias_) PhysicalIdAliases::instance().add(static_cast<size_t>(a) + 1, static_cast<size_t>(b) + 1);
  PhysicalIdAliases::instance().takeNew();
  if (config_.mode == Mode::PREDICTED) {
    std::ifstream retired(config_.state_path + ".retired");
    for (size_t id; retired >> id;)
      if (id > 0) retired_.insert(root(static_cast<int>(id) - 1));
    LOG(WARNING) << "ONLINE FRONT: " << retired_.size() << " tracks retired by the map (never written again)";
  }
  LOG(WARNING) << "ONLINE FRONT: state of " << tracks_.size() << " tracks (" << named_.size() << " named) from "
               << config_.state_path;
}

}  // namespace khronos
