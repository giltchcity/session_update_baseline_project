#include "khronos/active_window/predicted_site_handoff.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

namespace khronos {
namespace {

Eigen::Vector3d toWorld(int u, int v, float z, const OnlineFront::Intrinsics& k,
                        const Eigen::Isometry3d& pose) {
  return pose * Eigen::Vector3d((u + k.offset - k.cx) * z / k.fx,
                                (v + k.offset - k.cy) * z / k.fy, z);
}

template <typename Func>
void neighbours(const std::array<int64_t, 3>& c, Func fn) {
  for (int x = -1; x <= 1; ++x)
    for (int y = -1; y <= 1; ++y)
      for (int z = -1; z <= 1; ++z) fn(std::array<int64_t, 3>{c[0] + x, c[1] + y, c[2] + z});
}

}  // namespace

PredictedSiteHandoff::PredictedSiteHandoff(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("Cannot open predicted site handoff: " + path);
  const auto packet = nlohmann::json::parse(stream);
  if (packet.at("schema") != "predicted_site_handoff/v1")
    throw std::invalid_argument("Unsupported predicted site handoff");
  resolution_ = packet.at("resolution_m").get<double>();
  source_stamp_ = packet.at("source_stamp_ns").get<uint64_t>();
  if (!(resolution_ > 0) || !std::isfinite(resolution_) || !source_stamp_)
    throw std::invalid_argument("Invalid predicted site coordinate frame");

  for (const auto& row : packet.at("disputed_sites")) {
    Site s{};
    s.source = row.at("source_id");
    s.semantic = row.at("source_class");
    s.host = row.at("explaining_id");
    s.host_semantic = row.at("explaining_class");
    if (s.source <= 0 || s.host <= 0 || s.source == s.host || s.semantic == s.host_semantic)
      throw std::invalid_argument("Invalid disputed site identities");
    for (const auto& xyz : row.at("points_world_m")) {
      const auto p = xyz.get<std::array<double, 3>>();
      const Eigen::Vector3d point(p[0], p[1], p[2]);
      if (!point.allFinite() || (point / resolution_).cwiseAbs().maxCoeff() >= 9.e18)
        throw std::invalid_argument("Invalid handoff surface point");
      if (!s.cells.emplace(cell(point), s.samples.size()).second)
        throw std::invalid_argument("Repeated surface cell in a handoff site");
      s.samples.push_back({point});
    }
    auto setRole = [&](const char* field, Role role) {
      for (const auto& value : row.at(field)) {
        auto& sample = s.samples.at(value.get<size_t>());
        if (sample.role != Role::Unresolved) throw std::invalid_argument("Overlapping site assignments");
        sample.role = role;
      }
    };
    setRole("shared_indices", Role::Shared);
    setRole("independent_indices", Role::Independent);
    if (row.at("shared_indices").empty()) throw std::invalid_argument("Disputed site without shared geometry");
    by_source_[s.source].push_back(sites_.size());
    sites_.push_back(std::move(s));
  }
}

void PredictedSiteHandoff::validateSeed(uint64_t stamp, const std::map<int, int>& classes) const {
  if (stamp != source_stamp_) throw std::invalid_argument("Predicted site handoff timestamp differs from seed");
  for (const auto& s : sites_) {
    const auto source = classes.find(s.source);
    const auto host = classes.find(s.host);
    if (source == classes.end() || host == classes.end() ||
        source->second != s.semantic || host->second != s.host_semantic) {
      throw std::invalid_argument("Predicted handoff refers to identities absent from the source map");
    }
  }
}

PredictedSiteHandoff::Cell PredictedSiteHandoff::cell(const Eigen::Vector3d& p) const {
  return {static_cast<int64_t>(std::floor(p.x() / resolution_)),
          static_cast<int64_t>(std::floor(p.y() / resolution_)),
          static_cast<int64_t>(std::floor(p.z() / resolution_))};
}

bool PredictedSiteHandoff::keepInheritedVertex(int id, int semantic, const Eigen::Vector3d& p) const {
  const auto it = by_source_.find(id);
  if (it == by_source_.end() || !p.allFinite()) return true;
  bool covered = false;
  const auto key = cell(p);
  for (const auto index : it->second) {
    const auto& s = sites_[index];
    if (s.semantic != semantic) continue;
    const auto pos = s.cells.find(key);
    if (pos == s.cells.end()) continue;
    if (s.samples[pos->second].role == Role::Independent) return true;
    covered = true;
  }
  return !covered;
}

const PredictedSiteHandoff::Site* PredictedSiteHandoff::explanatorySite(
    int id, int semantic, const Eigen::Vector3d& p) const {
  const auto it = by_source_.find(id);
  if (it == by_source_.end()) return nullptr;
  const auto key = cell(p);
  double nearest = std::numeric_limits<double>::infinity();
  const Site* winner = nullptr;
  for (const auto index : it->second) {
    const auto& s = sites_[index];
    if (s.semantic != semantic) continue;
    neighbours(key, [&](const Cell& c) {
      const auto pos = s.cells.find(c);
      if (pos == s.cells.end()) return;
      const auto& sample = s.samples[pos->second];
      const double distance = (sample.point - p).cwiseAbs().maxCoeff();
      if (distance > resolution_ || distance > nearest) return;
      const bool shared = sample.role == Role::Shared && sample.balance >= 0;
      if (distance < nearest) {
        nearest = distance;
        winner = shared ? &s : nullptr;
      } else if (!shared || !winner || winner->host != s.host) {
        winner = nullptr;
      }
    });
  }
  return winner;
}

void PredictedSiteHandoff::observe(const cv::Mat& depth, const cv::Mat& raw,
                                   const cv::Mat& ids, const cv::Mat& semantics,
                                   const OnlineFront::Intrinsics& k, const Eigen::Isometry3d& pose) {
  const Eigen::Isometry3d camera_T_world = pose.inverse();
  for (auto& s : sites_) {
    for (auto& sample : s.samples) {
      if (sample.role != Role::Shared) continue;
      const Eigen::Vector3d p = camera_T_world * sample.point;
      if (!(p.z() > 0)) continue;
      const double x = std::floor(k.fx * p.x() / p.z() + k.cx - k.offset + .5);
      const double y = std::floor(k.fy * p.y() / p.z() + k.cy - k.offset + .5);
      if (!(x >= 0 && x < depth.cols && y >= 0 && y < depth.rows)) continue;
      const int u = static_cast<int>(x), v = static_cast<int>(y);
      const float z = depth.at<float>(v, u);
      if (!(z > 0) || !std::isfinite(z) || std::abs(z - p.z()) > resolution_) continue;
      const int label = raw.at<int32_t>(v,u);
      const int current = ids.at<int32_t>(v,u);
      if (label == s.semantic) --sample.balance;
      else if (current == s.host && label == s.host_semantic) ++sample.balance;
      else if (current > 0 && label == semantics.at<int32_t>(v,u)) --sample.balance;
    }
  }
}

void PredictedSiteHandoff::applyFrame(const cv::Mat& depth, const cv::Mat& raw,
                                      cv::Mat& ids, cv::Mat& semantics,
                                      const OnlineFront::Intrinsics& k,
                                      const Eigen::Isometry3d& pose, int64_t stamp) {
  if (sites_.empty()) return;
  if (stamp <= 0 || static_cast<uint64_t>(stamp) <= source_stamp_)
    throw std::invalid_argument("Predicted handoff requires observations from a later visit");
  if (stamp < last_stamp_) throw std::runtime_error("Out-of-order predicted site observations");
  struct Candidate { int u, v; const Site* site; Eigen::Vector3d point; };
  std::vector<Candidate> candidates;
  using HostIdentity = std::pair<int, int>;
  std::map<HostIdentity, std::map<Cell, std::vector<Eigen::Vector3d>>> observed;
  for (int v = 0; v < ids.rows; ++v) {
    const auto* row = ids.ptr<int32_t>(v);
    for (int u = 0; u < ids.cols; ++u) {
      if (by_source_.find(row[u]) == by_source_.end()) continue;
      const float z = depth.at<float>(v,u);
      if (!(z > 0) || !std::isfinite(z)) continue;
      const auto point = toWorld(u,v,z,k,pose);
      const auto* site = explanatorySite(row[u], raw.at<int32_t>(v,u), point);
      if (!site) continue;
      observed.try_emplace({site->host, site->host_semantic});
      candidates.push_back({u,v,site,point});
    }
  }

  // A host must independently show its own class at the same 3-D location in
  // this view. Stage all assignments so one routed pixel cannot support another.
  if (!candidates.empty()) {
    for (int v = 0; v < ids.rows; ++v) {
      for (int u = 0; u < ids.cols; ++u) {
        const auto host = observed.find({ids.at<int32_t>(v,u), raw.at<int32_t>(v,u)});
        if (host == observed.end()) continue;
        const float z = depth.at<float>(v,u);
        if (!(z > 0) || !std::isfinite(z)) continue;
        const auto point = toWorld(u,v,z,k,pose);
        host->second[cell(point)].push_back(point);
      }
    }
  }
  if (stamp != last_stamp_) {
    observe(depth, raw, ids, semantics, k, pose);
    last_stamp_ = stamp;
  }
  for (const auto& c : candidates) {
    bool supported = false;
    const auto& points = observed.at({c.site->host, c.site->host_semantic});
    neighbours(cell(c.point), [&](const Cell& key) {
      if (supported) return;
      const auto hit = points.find(key);
      if (hit == points.end()) return;
      supported = std::any_of(hit->second.begin(), hit->second.end(), [&](const Eigen::Vector3d& p) {
        return (p - c.point).cwiseAbs().maxCoeff() <= resolution_;
      });
    });
    if (!supported) continue;
    ids.at<int32_t>(c.v,c.u) = c.site->host;
    semantics.at<int32_t>(c.v,c.u) = c.site->host_semantic;
  }
}

}  // namespace khronos
