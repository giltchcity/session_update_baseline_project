#include "session_core/evidence/frame_pair_sampler.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <hydra/input/camera.h>

namespace khronos {
namespace {

// Computational sampling of the pixels of the newest frame (README table 5.1).
constexpr int kPixelStride = 16;

struct DenseFrame {
  uint32_t width = 0, height = 0;
  Eigen::Isometry3f sensor_T_world = Eigen::Isometry3f::Identity();
  hydra::Sensor::ConstPtr sensor;
  std::vector<uint16_t> range_mm;
  const hydra::Camera* camera = nullptr;
  bool load(const PhysicalEvidenceStore::Snapshot& evidence, TimeStamp stamp) {
    if (!evidence.denseRange(stamp, width, height, sensor_T_world, sensor, range_mm)) return false;
    camera = dynamic_cast<const hydra::Camera*>(sensor.get());
    return camera != nullptr;
  }
  // Camera-frame point of the reading at (u, v); false for an invalid pixel.
  bool point(int u, int v, Eigen::Vector3f& p) const {
    if (u < 0 || v < 0 || u >= static_cast<int>(width) || v >= static_cast<int>(height)) return false;
    const uint16_t code = range_mm[static_cast<size_t>(v) * width + u];
    if (!code) return false;
    const auto& c = camera->getConfig();
    const Eigen::Vector3f ray((u - c.cx) / c.fx, (v - c.cy) / c.fy, 1.f);
    p = ray.normalized() * (1e-3f * code);
    return true;
  }
};

}  // namespace

void accumulateFramePairs(const PhysicalEvidenceStore::Snapshot& evidence, TimeStamp stamp,
                          model::SensorCalibrator& calibrator, double* device_max_range) {
  if (!evidence) return;
  DenseFrame a;
  if (!a.load(evidence, stamp)) return;
  if (device_max_range) *device_max_range = a.sensor->max_range();
  const auto stamps = evidence.timestamps(0, stamp);
  if (stamps.size() < 2 || stamps.back() != stamp) return;
  const Eigen::Isometry3f world_T_a = a.sensor_T_world.inverse();
  const size_t newest = stamps.size() - 1;
  DenseFrame b;
  for (size_t k = 0, offset = 1; k < model::SensorCalibrator::kTimeBins && offset <= newest;
       ++k, offset *= 2) {
    const TimeStamp partner = stamps[newest - offset];
    if (!b.load(evidence, partner)) continue;
    const double dt = (static_cast<double>(stamp) - static_cast<double>(partner)) * 1e-9;
    const auto& cb = b.camera->getConfig();
    const Eigen::Isometry3f b_T_a = b.sensor_T_world * world_T_a;
    const Eigen::Vector3f origin_a = world_T_a.translation(), origin_b = b.sensor_T_world.inverse().translation();
    for (int v = kPixelStride / 2; v < static_cast<int>(a.height) - 1; v += kPixelStride) {
      for (int u = kPixelStride / 2; u < static_cast<int>(a.width) - 1; u += kPixelStride) {
        Eigen::Vector3f p, pl, pr, pu, pd;
        if (!a.point(u, v, p) || !a.point(u - 1, v, pl) || !a.point(u + 1, v, pr) ||
            !a.point(u, v - 1, pu) || !a.point(u, v + 1, pd)) {
          continue;
        }
        Eigen::Vector3f normal = (pr - pl).cross(pd - pu);
        const float length = normal.norm();
        if (!(length > 0.f)) continue;
        normal /= length;
        // The point in the earlier frame and the reading found there.
        const Eigen::Vector3f q = b_T_a * p;
        if (!(q.z() > 0.f)) continue;
        const int ub = static_cast<int>(std::floor(q.x() / q.z() * cb.fx + cb.cx + 0.5f));
        const int vb = static_cast<int>(std::floor(q.y() / q.z() * cb.fy + cb.cy + 0.5f));
        if (ub < 0 || vb < 0 || ub >= static_cast<int>(b.width) || vb >= static_cast<int>(b.height)) continue;
        const uint16_t code = b.range_mm[static_cast<size_t>(vb) * b.width + ub];
        if (!code) continue;
        const double reading = 1e-3 * code, predicted = q.norm();
        // Incidence of the ray of the earlier frame on the surface of the newest frame.
        const Eigen::Vector3f normal_b = b_T_a.linear() * normal;
        const double cosine = std::abs(static_cast<double>(normal_b.dot(q.normalized())));
        calibrator.addPair(k, dt, predicted, std::acos(std::min(1.0, cosine)), reading - predicted);
        RangePair pair;
        pair.origin = origin_a;
        pair.direction = (world_T_a.linear() * p.normalized()).normalized();
        pair.other = origin_b;
        pair.range = p.norm();
        pair.other_range = static_cast<float>(reading);
        calibrator.addScalePair(pair, dt);
      }
    }
  }
}

}  // namespace khronos
