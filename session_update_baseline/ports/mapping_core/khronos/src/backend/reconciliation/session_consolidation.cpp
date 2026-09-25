#include "khronos/backend/reconciliation/session_consolidation.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <glog/logging.h>

#include "khronos/utils/khronos_attribute_utils.h"

namespace khronos {

namespace {

// One surface element of the final map: a background mesh vertex or a vertex
// of a current object mesh (world frame).
struct Element {
  Eigen::Vector3f position;
  float half_voxel = 0.f;
  float truncation = 0.f;
  uint8_t layer = 0;  // 0 background, 1 objects
  bool memory = false;
};

// Per-element evidence counters over the session's frames.
struct Evidence {
  uint32_t hit_any = 0;   // some pixel of the ball measures a surface within tau
  uint32_t hit_own = 0;   // the element's own pixel measures it within tau
  uint32_t through = 0;   // every pixel of the ball observed beyond it by > tau
  uint32_t through_band = 0;  // through, own pixel at most one truncation beyond
  uint32_t blocked = 0;       // own pixel in front of it by > tau
  uint32_t blocked_band = 0;  // blocked by a surface at most one truncation in front
};

constexpr float kHistogramResolution = 0.0005f;  // [m], estimator resolution

template <typename Fn>
void parallelFor(size_t n, int num_threads, Fn fn) {
  const size_t workers = std::max<size_t>(1, std::min<size_t>(num_threads, n));
  if (workers == 1) {
    fn(0, n);
    return;
  }
  std::vector<std::thread> threads;
  const size_t step = (n + workers - 1) / workers;
  for (size_t w = 0; w < workers; ++w) {
    const size_t begin = w * step;
    const size_t end = std::min(n, begin + step);
    if (begin >= end) break;
    threads.emplace_back([&fn, begin, end] { fn(begin, end); });
  }
  for (auto& thread : threads) thread.join();
}

// Erase vertices (and the faces that use them) from a mesh whose per-vertex
// attribute arrays may be shorter than its point array (e.g. an object mesh
// flagged with labels but carrying none). spark_dsg::Mesh::eraseVertices indexes
// every flagged array by vertex, so it must not be used on such meshes. An
// attribute array is re-indexed only when it has one entry per vertex and is
// otherwise kept as it was.
void eraseVerticesSafely(spark_dsg::Mesh& mesh, const std::vector<uint8_t>& erase) {
  const size_t n = mesh.numVertices();
  std::vector<int64_t> remap(n, -1);
  size_t kept = 0;
  for (size_t i = 0; i < n; ++i) {
    if (!erase[i]) remap[i] = static_cast<int64_t>(kept++);
  }
  auto compact = [&](auto& values) {
    if (values.size() != n) return;
    size_t j = 0;
    for (size_t i = 0; i < n; ++i) {
      if (!erase[i]) values[j++] = values[i];
    }
    values.resize(j);
  };
  compact(mesh.points);
  compact(mesh.colors);
  compact(mesh.stamps);
  compact(mesh.labels);
  compact(mesh.first_seen_stamps);
  spark_dsg::Mesh::Faces faces;
  faces.reserve(mesh.faces.size());
  for (const auto& face : mesh.faces) {
    if (face[0] >= n || face[1] >= n || face[2] >= n) continue;
    const int64_t a = remap[face[0]], b = remap[face[1]], c = remap[face[2]];
    if (a < 0 || b < 0 || c < 0) continue;
    faces.push_back({{static_cast<size_t>(a), static_cast<size_t>(b), static_cast<size_t>(c)}});
  }
  mesh.faces = std::move(faces);
}

}  // namespace

std::string SessionConsolidation::Result::summary() const {
  std::stringstream ss;
  ss << "frames=" << frames << " elements=" << elements << " memory=" << memory_elements
     << " retired_own=" << retired_own << " retired_memory_seen_through="
     << retired_memory_seen_through << " retired_memory_hidden=" << retired_memory_hidden
     << " retired_chain=" << retired_chain
     << " objects_kept_whole=" << objects_kept_whole
     << " erased_background=" << background_vertices_erased
     << " erased_object=" << object_vertices_erased << " sigma_bg_cm=[";
  for (size_t i = 0; i < sigma_background.size(); ++i) {
    ss << (i ? " " : "") << std::round(1000.f * sigma_background[i]) / 10.f;
  }
  ss << "] sigma_obj_cm=[";
  for (size_t i = 0; i < sigma_objects.size(); ++i) {
    ss << (i ? " " : "") << std::round(1000.f * sigma_objects[i]) / 10.f;
  }
  ss << "]";
  return ss.str();
}

void SessionConsolidation::setMemory(std::vector<Eigen::Vector3f> points) {
  memory_search_.reset();
  memory_points_ = std::move(points);
  if (!memory_points_.empty()) {
    memory_search_ = std::make_unique<hydra::PointNeighborSearch>(memory_points_);
  }
}

void SessionConsolidation::setChain(std::vector<Eigen::Vector3f> points) {
  chain_search_.reset();
  chain_points_ = std::move(points);
  if (!chain_points_.empty()) {
    chain_search_ = std::make_unique<hydra::PointNeighborSearch>(chain_points_);
  }
}

SessionConsolidation::Result SessionConsolidation::apply(
    DynamicSceneGraph& dsg, const PhysicalEvidenceStore::Snapshot& evidence) const {
  Result result;
  if (!evidence || scales_.background_voxel <= 0.f || scales_.background_truncation <= 0.f) {
    LOG(WARNING) << "[SessionConsolidation] no evidence or map scales; skipped.";
    return result;
  }

  // Gather the surface elements of the final map.
  std::vector<Element> elements;
  const auto mesh = dsg.hasMesh() ? dsg.mesh() : nullptr;
  const size_t num_background = mesh ? mesh->numVertices() : 0;
  elements.reserve(num_background);
  for (size_t i = 0; i < num_background; ++i) {
    Element e;
    e.position = mesh->pos(i);
    e.half_voxel = 0.5f * scales_.background_voxel;
    e.truncation = scales_.background_truncation;
    e.layer = 0;
    elements.push_back(e);
  }
  struct ObjectRange {
    NodeId id;
    size_t begin;
    size_t end;
  };
  std::vector<ObjectRange> objects;
  if (dsg.hasLayer(DsgLayers::OBJECTS)) {
    for (const auto& [id, node] : dsg.getLayer(DsgLayers::OBJECTS).nodes()) {
      const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
      if (!attrs || attrs->mesh.numVertices() == 0 || !hasCurrentObjectMesh(*attrs)) {
        continue;
      }
      float voxel = scales_.object_voxel;
      if (voxel < 0.f) {
        voxel = std::max(attrs->bounding_box.dimensions.maxCoeff() * -voxel,
                         scales_.object_min_voxel);
      }
      if (voxel <= 0.f) continue;
      ObjectRange range{id, elements.size(), elements.size()};
      for (size_t i = 0; i < attrs->mesh.numVertices(); ++i) {
        Element e;
        e.position = attrs->bounding_box.pointToWorldFrame(attrs->mesh.pos(i));
        e.half_voxel = 0.5f * voxel;
        e.truncation = 2.f * voxel;  // object truncation = 2 voxels (object extractor)
        e.layer = 1;
        elements.push_back(e);
      }
      range.end = elements.size();
      objects.push_back(range);
    }
  }
  result.elements = elements.size();
  if (elements.empty()) return result;

  // Memory: elements that coincide with the inherited state this session loaded.
  if (memory_search_) {
    const float max_sq = config.memory_match_distance * config.memory_match_distance;
    for (auto& e : elements) {
      float d_sq = std::numeric_limits<float>::max();
      size_t idx = 0;
      e.memory = memory_search_->search(e.position, d_sq, idx) && d_sq <= max_sq;
      result.memory_elements += e.memory;
    }
  }

  const auto stamps = evidence.timestamps(0, std::numeric_limits<TimeStamp>::max());
  result.frames = stamps.size();
  if (stamps.empty()) return result;

  // Depth noise model: this session's residuals on its own surfaces, per layer
  // and range bin (robust scale of the half-normal: 1.4826 * median |r|, r <= 0).
  const size_t num_bins = 16;
  float max_truncation = scales_.background_truncation;
  for (const auto& e : elements) max_truncation = std::max(max_truncation, e.truncation);
  const size_t hist_size = static_cast<size_t>(std::ceil(max_truncation / kHistogramResolution)) + 1;
  std::vector<std::vector<uint64_t>> hist(2 * num_bins, std::vector<uint64_t>(hist_size, 0));
  auto bin_of = [&](float range) {
    return std::min<size_t>(num_bins - 1, static_cast<size_t>(std::max(0.f, range) / config.range_bin));
  };

  uint32_t width = 0, height = 0;
  Eigen::Isometry3f sensor_T_world;
  hydra::Sensor::ConstPtr sensor;
  std::vector<uint16_t> range_mm;
  for (const auto stamp : stamps) {
    if (!evidence.denseRange(stamp, width, height, sensor_T_world, sensor, range_mm) || !sensor) {
      continue;
    }
    std::vector<std::vector<uint64_t>> local(2 * num_bins, std::vector<uint64_t>(hist_size, 0));
    for (const auto& e : elements) {
      if (e.memory) continue;
      const Eigen::Vector3f p = sensor_T_world * e.position;
      int u = -1, v = -1;
      if (!sensor->projectPointToImagePlane(p, u, v) || u < 0 || v < 0 ||
          static_cast<uint32_t>(u) >= width || static_cast<uint32_t>(v) >= height) {
        continue;
      }
      const uint16_t d_mm = range_mm[static_cast<size_t>(v) * width + u];
      if (!d_mm) continue;
      const float q = p.norm();
      const float r = 0.001f * d_mm - q;
      if (r > 0.f || -r > e.truncation) continue;
      const size_t h = std::min(hist_size - 1, static_cast<size_t>(-r / kHistogramResolution));
      ++local[e.layer * num_bins + bin_of(q)][h];
    }
    for (size_t b = 0; b < local.size(); ++b)
      for (size_t h = 0; h < hist_size; ++h) hist[b][h] += local[b][h];
  }
  std::vector<float> sigma(2 * num_bins, std::numeric_limits<float>::quiet_NaN());
  for (size_t b = 0; b < sigma.size(); ++b) {
    uint64_t total = 0;
    for (const auto c : hist[b]) total += c;
    if (total < config.min_bin_samples) continue;
    const double half = 0.5 * total;
    double cumulative = 0.0;
    for (size_t h = 0; h < hist_size; ++h) {
      if (cumulative + hist[b][h] >= half) {
        const double frac = hist[b][h] ? (half - cumulative) / hist[b][h] : 0.0;
        sigma[b] = 1.4826f * static_cast<float>((h + frac) * kHistogramResolution);
        break;
      }
      cumulative += hist[b][h];
    }
  }
  for (size_t layer = 0; layer < 2; ++layer) {  // empty bins take the nearest populated bin
    for (size_t k = 0; k < num_bins; ++k) {
      float& s = sigma[layer * num_bins + k];
      if (std::isfinite(s)) continue;
      for (size_t off = 1; off < num_bins && !std::isfinite(s); ++off) {
        if (k >= off && std::isfinite(sigma[layer * num_bins + k - off])) s = sigma[layer * num_bins + k - off];
        else if (k + off < num_bins && std::isfinite(sigma[layer * num_bins + k + off])) s = sigma[layer * num_bins + k + off];
      }
      if (!std::isfinite(s)) s = 0.f;
    }
  }
  result.sigma_background.assign(sigma.begin(), sigma.begin() + num_bins);
  result.sigma_objects.assign(sigma.begin() + num_bins, sigma.end());

  // Evidence pass: every stored frame, every element, over its localization ball.
  std::vector<Evidence> ev(elements.size());
  for (const auto stamp : stamps) {
    if (!evidence.denseRange(stamp, width, height, sensor_T_world, sensor, range_mm) || !sensor) {
      continue;
    }
    // Focal length in pixels from the sensor model (projection of two near-axis points).
    float u0 = 0, v0 = 0, u1 = 0, v1 = 0;
    if (!sensor->projectPointToImagePlane(Eigen::Vector3f(0.f, 0.f, 1.f), u0, v0) ||
        !sensor->projectPointToImagePlane(Eigen::Vector3f(0.01f, 0.f, 1.f), u1, v1)) {
      continue;
    }
    const float focal = std::abs(u1 - u0) / 0.01f;
    parallelFor(elements.size(), config.num_threads, [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        const auto& e = elements[i];
        const Eigen::Vector3f p = sensor_T_world * e.position;
        int u = -1, v = -1;
        if (p.z() <= 0.f || !sensor->projectPointToImagePlane(p, u, v) || u < 0 || v < 0 ||
            static_cast<uint32_t>(u) >= width || static_cast<uint32_t>(v) >= height) {
          continue;
        }
        const float q = p.norm();
        const float s = sigma[e.layer * num_bins + bin_of(q)];
        const float tau = std::max(e.half_voxel, s);
        auto& c = ev[i];
        const uint16_t d0_mm = range_mm[static_cast<size_t>(v) * width + u];
        const bool centre_valid = d0_mm > 0;
        const float r0 = centre_valid ? 0.001f * d0_mm - q : 0.f;
        if (centre_valid) {
          if (std::abs(r0) <= tau) ++c.hit_own;
          if (r0 < -tau) {
            ++c.blocked;
            if (r0 >= -e.truncation) ++c.blocked_band;
          }
        }
        // The ball: every pixel whose centre lies within focal * tau / z.
        const float rp = focal * tau / p.z();
        const int R = static_cast<int>(std::floor(rp));
        bool hit = false, all_in = true, all_valid = true, all_beyond = true;
        for (int dv = -R; dv <= R && !hit; ++dv) {
          for (int du = -R; du <= R; ++du) {
            if (du * du + dv * dv > rp * rp && (du || dv)) continue;
            const int x = u + du, y = v + dv;
            if (x < 0 || y < 0 || x >= static_cast<int>(width) || y >= static_cast<int>(height)) {
              all_in = false;
              continue;
            }
            const uint16_t d_mm = range_mm[static_cast<size_t>(y) * width + x];
            if (!d_mm) {
              all_valid = false;
              continue;
            }
            const float r = 0.001f * d_mm - q;
            if (std::abs(r) <= tau) {
              hit = true;
              break;
            }
            if (r <= tau) all_beyond = false;
          }
        }
        if (hit) {
          ++c.hit_any;
        } else if (all_in && all_valid && all_beyond) {
          ++c.through;
          if (centre_valid && r0 <= e.truncation) ++c.through_band;
        }
      }
    });
  }

  // Decide.
  std::vector<uint8_t> retire(elements.size(), 0);
  const float match_sq = config.memory_match_distance * config.memory_match_distance;
  for (size_t i = 0; i < elements.size(); ++i) {
    const auto& c = ev[i];
    if (elements[i].memory && chain_search_) {
      float d_sq = std::numeric_limits<float>::max();
      size_t idx = 0;
      if (chain_search_->search(elements[i].position, d_sq, idx) && d_sq <= match_sq) {
        retire[i] = 1;
        ++result.retired_chain;
        continue;
      }
    }
    if (!elements[i].memory) {
      if (c.through_band > c.hit_own) {
        retire[i] = 1;
        ++result.retired_own;
      }
    } else if (c.through > c.hit_any) {
      retire[i] = 1;
      ++result.retired_memory_seen_through;
    } else if (c.hit_any == 0 && c.through == 0 && 2 * c.blocked_band > c.blocked) {
      retire[i] = 1;
      ++result.retired_memory_hidden;
    }
  }
  // An object's surface is never removed entirely: its existence is decided by
  // change detection, not by the geometry consolidation.
  for (const auto& obj : objects) {
    bool all = obj.end > obj.begin;
    for (size_t i = obj.begin; i < obj.end && all; ++i) all = retire[i];
    if (all) {
      std::fill(retire.begin() + obj.begin, retire.begin() + obj.end, 0);
      ++result.objects_kept_whole;
    }
  }

  for (size_t i = 0; i < elements.size(); ++i) {
    if (retire[i]) result.retired_positions.push_back(elements[i].position);
  }

  // Apply.
  if (mesh) {
    std::vector<uint8_t> erase(retire.begin(), retire.begin() + num_background);
    result.background_vertices_erased = std::count(erase.begin(), erase.end(), 1);
    if (result.background_vertices_erased) eraseVerticesSafely(*mesh, erase);
  }
  for (const auto& obj : objects) {
    std::vector<uint8_t> erase(retire.begin() + obj.begin, retire.begin() + obj.end);
    const size_t count = std::count(erase.begin(), erase.end(), 1);
    if (!count) continue;
    auto& attrs = dsg.getNode(obj.id).attributes<KhronosObjectAttributes>();
    if (attrs.mesh.numVertices() != erase.size()) continue;  // defensive: mesh changed
    eraseVerticesSafely(attrs.mesh, erase);
    result.object_vertices_erased += count;
  }
  return result;
}

}  // namespace khronos
