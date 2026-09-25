#include "khronos/backend/reconciliation/session_consolidation.h"

#include <algorithm>
#include <array>
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

// Scale consistency of this session's depth with its trajectory. A surface
// point measured by one stored frame and re-measured by another is read at the
// same position whatever the two ranges when depth and trajectory agree in
// scale. A depth that is short (long) by a fixed fraction s of the range makes
// the far reading of every point shorter (longer) than the near one, so near
// and far views, and sessions that saw a surface from different distances,
// place the same surface at different positions. s is the scale that makes the
// stored frames agree best: every reading is scaled by (1 + s), and s minimises
// the median disagreement of all re-measured points (pixels of one frame
// projected into another frame; associated when the other frame's reading lies
// within one truncation of the projected point). The frame subset, pixel stride
// and search grid are estimator settings; exact depth gives s = 0.
float estimateDepthScale(const PhysicalEvidenceStore::Snapshot& evidence,
                         const std::vector<TimeStamp>& stamps,
                         float association,
                         int num_threads) {
  constexpr size_t kMaxFrames = 64;
  constexpr uint32_t kPixelStride = 16;
  struct Frame {
    uint32_t width = 0, height = 0;
    Eigen::Isometry3f sensor_T_world;
    hydra::Sensor::ConstPtr sensor;
    std::vector<uint16_t> range_mm;
  };
  std::vector<Frame> frames;
  const size_t step = std::max<size_t>(1, stamps.size() / kMaxFrames);
  for (size_t i = 0; i < stamps.size() && frames.size() < kMaxFrames; i += step) {
    Frame f;
    if (evidence.denseRange(stamps[i], f.width, f.height, f.sensor_T_world, f.sensor, f.range_mm) &&
        f.sensor) {
      frames.push_back(std::move(f));
    }
  }
  if (frames.size() < 2) return 0.f;
  // Pinhole parameters from the sensor model (projections of near-axis points).
  float cx = 0, cy = 0, ux = 0, vx = 0, uy = 0, vy = 0;
  const auto& sensor = *frames.front().sensor;
  if (!sensor.projectPointToImagePlane(Eigen::Vector3f(0.f, 0.f, 1.f), cx, cy) ||
      !sensor.projectPointToImagePlane(Eigen::Vector3f(0.01f, 0.f, 1.f), ux, vx) ||
      !sensor.projectPointToImagePlane(Eigen::Vector3f(0.f, 0.01f, 1.f), uy, vy)) {
    return 0.f;
  }
  const float fx = (ux - cx) / 0.01f, fy = (vy - cy) / 0.01f;
  if (std::abs(fx) < 1e-3f || std::abs(fy) < 1e-3f) return 0.f;

  struct Sample {
    Eigen::Vector3f origin;     // measuring sensor, world frame
    Eigen::Vector3f direction;  // its ray, world frame
    Eigen::Vector3f other;      // re-measuring sensor, world frame
    float range = 0.f;
    float other_range = 0.f;
  };
  std::vector<std::vector<Sample>> per_frame(frames.size());
  parallelFor(frames.size(), num_threads, [&](size_t begin, size_t end) {
    for (size_t g = begin; g < end; ++g) {
      const auto& a = frames[g];
      const Eigen::Isometry3f world_T_a = a.sensor_T_world.inverse();
      for (uint32_t v = 0; v < a.height; v += kPixelStride) {
        for (uint32_t u = 0; u < a.width; u += kPixelStride) {
          const uint16_t d_mm = a.range_mm[static_cast<size_t>(v) * a.width + u];
          if (!d_mm) continue;
          const Eigen::Vector3f direction =
              world_T_a.linear() *
              Eigen::Vector3f((u - cx) / fx, (v - cy) / fy, 1.f).normalized();
          const float range = 0.001f * d_mm;
          const Eigen::Vector3f point = world_T_a.translation() + direction * range;
          for (size_t f = 0; f < frames.size(); ++f) {
            if (f == g) continue;
            const auto& b = frames[f];
            const Eigen::Vector3f p = b.sensor_T_world * point;
            int pu = -1, pv = -1;
            if (p.z() <= 0.f || !b.sensor->projectPointToImagePlane(p, pu, pv) || pu < 0 ||
                pv < 0 || static_cast<uint32_t>(pu) >= b.width ||
                static_cast<uint32_t>(pv) >= b.height) {
              continue;
            }
            const uint16_t e_mm = b.range_mm[static_cast<size_t>(pv) * b.width + pu];
            if (!e_mm || std::abs(0.001f * e_mm - p.norm()) > association) continue;
            per_frame[g].push_back({world_T_a.translation(), direction,
                                    b.sensor_T_world.inverse().translation(), range,
                                    0.001f * e_mm});
          }
        }
      }
    }
  });
  std::vector<Sample> samples;
  for (auto& s : per_frame) samples.insert(samples.end(), s.begin(), s.end());
  if (samples.size() < 1000) return 0.f;

  std::vector<float> residual(samples.size());
  auto disagreement = [&](float s) {
    for (size_t i = 0; i < samples.size(); ++i) {
      const auto& x = samples[i];
      const Eigen::Vector3f point = x.origin + x.direction * (x.range * (1.f + s));
      residual[i] = std::abs(x.other_range * (1.f + s) - (point - x.other).norm());
    }
    auto mid = residual.begin() + residual.size() / 2;
    std::nth_element(residual.begin(), mid, residual.end());
    return *mid;
  };
  float best_s = 0.f, best = disagreement(0.f);
  for (int k = -50; k <= 50; ++k) {  // coarse: +-10 % in 0.2 % steps
    const float s = 0.002f * k, m = disagreement(s);
    if (m < best) best = m, best_s = s;
  }
  const float coarse = best_s;
  for (int k = -10; k <= 10; ++k) {  // fine: 0.02 % steps around the coarse optimum
    const float s = coarse + 0.0002f * k, m = disagreement(s);
    if (m < best) best = m, best_s = s;
  }
  return best_s;
}

// Closest point of triangle (a, b, c) to p (Ericson, Real-Time Collision Detection, 5.1.5).
Eigen::Vector3f closestPointOnTriangle(const Eigen::Vector3f& p,
                                       const Eigen::Vector3f& a,
                                       const Eigen::Vector3f& b,
                                       const Eigen::Vector3f& c) {
  const Eigen::Vector3f ab = b - a, ac = c - a, ap = p - a;
  const float d1 = ab.dot(ap), d2 = ac.dot(ap);
  if (d1 <= 0.f && d2 <= 0.f) return a;
  const Eigen::Vector3f bp = p - b;
  const float d3 = ab.dot(bp), d4 = ac.dot(bp);
  if (d3 >= 0.f && d4 <= d3) return b;
  const float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) return a + ab * (d1 / (d1 - d3));
  const Eigen::Vector3f cp = p - c;
  const float d5 = ab.dot(cp), d6 = ac.dot(cp);
  if (d6 >= 0.f && d5 <= d6) return c;
  const float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) return a + ac * (d2 / (d2 - d6));
  const float va = d3 * d6 - d5 * d4;
  if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
    return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
  }
  const float denom = 1.f / (va + vb + vc);
  return a + ab * (vb * denom) + ac * (vc * denom);
}

}  // namespace

std::string SessionConsolidation::Result::summary() const {
  std::stringstream ss;
  ss << "frames=" << frames << " elements=" << elements << " memory=" << memory_elements
     << " retired_own=" << retired_own << " retired_memory_seen_through="
     << retired_memory_seen_through << " retired_memory_displaced=" << retired_memory_displaced
     << " retired_memory_hidden=" << retired_memory_hidden << " retired_chain=" << retired_chain
     << " depth_scale_pct=" << std::round(10000.f * depth_scale) / 100.f
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

  // How far this session's depth displaces a surface per metre of range (only a
  // reading that is short relative to the trajectory places a surface in front
  // of where a nearer view puts it).
  result.depth_scale = estimateDepthScale(evidence, stamps, scales_.background_truncation,
                                          config.num_threads);
  const float displacement_per_metre = std::max(0.f, result.depth_scale);

  // Depth noise model: this session's residuals on its own surfaces, per layer
  // and range bin (robust scale of the half-normal: 1.4826 * median |r|, r <= 0).
  // Only an element that is the first own surface along its line of sight
  // contributes: an element hidden behind a nearer surface (within the
  // truncation band) sees that surface's depth, not noise. Visibility comes
  // from a coarse z-buffer of the own elements' ranges (cells of
  // kZBufferCell x kZBufferCell pixels; an element is visible when it lies
  // within one voxel of the nearest own element of its cell).
  constexpr int kZBufferCell = 4;
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
  std::vector<int32_t> pixel(elements.size(), -1);
  std::vector<float> range(elements.size(), 0.f);
  std::vector<float> zbuffer;
  for (const auto stamp : stamps) {
    if (!evidence.denseRange(stamp, width, height, sensor_T_world, sensor, range_mm) || !sensor) {
      continue;
    }
    const uint32_t cells_x = (width + kZBufferCell - 1) / kZBufferCell;
    const uint32_t cells_y = (height + kZBufferCell - 1) / kZBufferCell;
    zbuffer.assign(static_cast<size_t>(cells_x) * cells_y, std::numeric_limits<float>::max());
    for (size_t i = 0; i < elements.size(); ++i) {
      pixel[i] = -1;
      if (elements[i].memory) continue;
      const Eigen::Vector3f p = sensor_T_world * elements[i].position;
      int u = -1, v = -1;
      if (p.z() <= 0.f || !sensor->projectPointToImagePlane(p, u, v) || u < 0 || v < 0 ||
          static_cast<uint32_t>(u) >= width || static_cast<uint32_t>(v) >= height) {
        continue;
      }
      pixel[i] = v * static_cast<int32_t>(width) + u;
      range[i] = p.norm();
      float& z = zbuffer[static_cast<size_t>(v / kZBufferCell) * cells_x + u / kZBufferCell];
      z = std::min(z, range[i]);
    }
    std::vector<std::vector<uint64_t>> local(2 * num_bins, std::vector<uint64_t>(hist_size, 0));
    for (size_t i = 0; i < elements.size(); ++i) {
      if (pixel[i] < 0) continue;
      const auto& e = elements[i];
      const int u = pixel[i] % static_cast<int32_t>(width), v = pixel[i] / static_cast<int32_t>(width);
      const float nearest = zbuffer[static_cast<size_t>(v / kZBufferCell) * cells_x + u / kZBufferCell];
      if (range[i] > nearest + 2.f * e.half_voxel) continue;  // hidden behind a nearer own surface
      const uint16_t d_mm = range_mm[pixel[i]];
      if (!d_mm) continue;
      const float q = range[i];
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
  // For memory, also the nearest view that reached it: the element projects onto
  // a valid reading that is not more than one truncation in front of it.
  std::vector<Evidence> ev(elements.size());
  std::vector<float> nearest_range(elements.size(), std::numeric_limits<float>::infinity());
  std::vector<Eigen::Vector3f> nearest_camera(elements.size(), Eigen::Vector3f::Zero());
  for (const auto stamp : stamps) {
    if (!evidence.denseRange(stamp, width, height, sensor_T_world, sensor, range_mm) || !sensor) {
      continue;
    }
    const Eigen::Vector3f camera = sensor_T_world.inverse().translation();
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
          if (e.memory && r0 >= -e.truncation && q < nearest_range[i]) {
            nearest_range[i] = q;
            nearest_camera[i] = camera;
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
  // Memory that is a displaced second copy of a surface this session observed:
  // the present's own surface lies farther from it than one voxel (a separate
  // surface at this map resolution) but no farther than tau + s * range at the
  // nearest view that reached it (the displacement this session's depth scale
  // inconsistency produces at that range), on that view's side of it. With
  // exact depth (s = 0) the window (voxel, tau] is empty.
  if (displacement_per_metre > 0.f) {
    std::vector<std::array<size_t, 3>> own_faces;
    auto add_faces = [&](const spark_dsg::Mesh::Faces& faces, size_t offset, size_t count) {
      for (const auto& f : faces) {
        if (f[0] >= count || f[1] >= count || f[2] >= count) continue;
        const size_t a = offset + f[0], b = offset + f[1], c = offset + f[2];
        if (elements[a].memory || elements[b].memory || elements[c].memory) continue;
        if (retire[a] || retire[b] || retire[c]) continue;
        own_faces.push_back({a, b, c});
      }
    };
    if (mesh) add_faces(mesh->faces, 0, num_background);
    for (const auto& obj : objects) {
      const auto& attrs = dsg.getNode(obj.id).attributes<KhronosObjectAttributes>();
      add_faces(attrs.mesh.faces, obj.begin, obj.end - obj.begin);
    }
    // Samples on every own face (corners, edge midpoints, centroid) find the
    // nearest own face; the distance is then exact to that face.
    std::vector<Eigen::Vector3f> samples;
    std::vector<uint32_t> sample_face;
    samples.reserve(own_faces.size() * 7);
    sample_face.reserve(own_faces.size() * 7);
    for (size_t k = 0; k < own_faces.size(); ++k) {
      const Eigen::Vector3f& a = elements[own_faces[k][0]].position;
      const Eigen::Vector3f& b = elements[own_faces[k][1]].position;
      const Eigen::Vector3f& c = elements[own_faces[k][2]].position;
      const Eigen::Vector3f points[7] = {a, b, c, 0.5f * (a + b), 0.5f * (b + c), 0.5f * (a + c),
                                         (a + b + c) / 3.f};
      for (const auto& point : points) {
        samples.push_back(point);
        sample_face.push_back(static_cast<uint32_t>(k));
      }
    }
    if (!samples.empty()) {
      const hydra::PointNeighborSearch own_search(samples);
      for (size_t i = 0; i < elements.size(); ++i) {
        const auto& e = elements[i];
        if (!e.memory || retire[i] || !std::isfinite(nearest_range[i])) continue;
        float d_sq = 0.f;
        size_t idx = 0;
        if (!own_search.search(e.position, d_sq, idx)) continue;
        const auto& f = own_faces[sample_face[idx]];
        const Eigen::Vector3f closest =
            closestPointOnTriangle(e.position, elements[f[0]].position, elements[f[1]].position,
                                   elements[f[2]].position);
        const float d_own = (closest - e.position).norm();
        const float q = nearest_range[i];
        const float tau = std::max(e.half_voxel, sigma[e.layer * num_bins + bin_of(q)]);
        if (d_own <= 2.f * e.half_voxel || d_own > tau + displacement_per_metre * q) continue;
        if ((closest - e.position).dot(nearest_camera[i] - e.position) <= 0.f) continue;
        retire[i] = 1;
        ++result.retired_memory_displaced;
      }
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
