#include "session_core/evidence/element_measurement.h"

#include <algorithm>
#include <cmath>

#include "session_core/surface/surface_sampling.h"

namespace khronos {

ElementRound measureElements(const PhysicalEvidenceStore::Snapshot& evidence, size_t physical_id,
                             const spark_dsg::Mesh& mesh, const BoundingBox& bbox,
                             const model::RangeModel& psi, const ElementSource& source,
                             TimeStamp earliest, TimeStamp latest,
                             model::SensorCalibrator* calibrator) {
  ElementRound round;
  if (!evidence) return round;
  const auto stamps = evidence.timestamps(earliest, latest);
  if (stamps.empty()) return round;
  const auto samples = sampleSurface(mesh, bbox, source.element_size);
  round.num_elements = samples.size();
  if (!psi.valid()) return round;
  const double h = source.element_size;
  round.verdicts.reserve(samples.size());
  for (const auto& sample : samples) {
    // t_e of (6s): the acquisition time the element rests on. A previous session's element is
    // aligned to this session at its start; drift grows from there (sigma_x carries the alignment).
    const TimeStamp element_time = sample.first_seen > 0 ? sample.first_seen : source.fallback_time;
    const bool cross_session = element_time < source.session_start;
    const TimeStamp t_e = cross_session ? source.session_start : element_time;
    ElementVerdict verdict;
    bool have = false;
    for (const auto stamp : stamps) {
      const auto p = evidence.project(stamp, sample.point);
      if (p.endpoint.type == EndpointClass::kUnavailable) continue;
      const double rho = p.query_range_m, reading = p.endpoint.measured_depth_m;
      // Normal incidence of the ray; an element without a normal is taken head-on.
      const double cosine = sample.has_normal
          ? std::abs(static_cast<double>(sample.normal.dot(p.view_direction_world))) : 1.0;
      const double incidence = std::acos(std::min(1.0, cosine));
      const double dt = std::abs(static_cast<double>(stamp) - static_cast<double>(t_e)) * 1e-9;
      const double sigma = psi.sigmaEff(rho, incidence, dt, h, cross_session);
      const auto kind = model::classifyRange(psi, reading, rho, sigma, p.sensor_min_range,
                                             p.sensor_max_range, cross_session);
      if (kind == model::RangeClass::kInvalid) {
        ++round.invalid;
        continue;
      }
      if (cross_session && calibrator) {
        const double z = reading - rho - psi.bias(rho, true);
        calibrator->addCrossSession(z, std::max(0.0, sigma * sigma - psi.sigma_x * psi.sigma_x));
      }
      if (kind == model::RangeClass::kOccluded) {
        ++round.occluded;
        continue;
      }
      verdict.key = packElementCell(sample.cell);
      verdict.through = kind == model::RangeClass::kThrough;
      verdict.stamp = stamp;
      verdict.label = ElementVerdict::kNoLabel;
      if (!verdict.through) {
        const auto& e = p.endpoint;
        if (e.type == EndpointClass::kPhysical) {
          verdict.label = static_cast<size_t>(e.physical_id) == physical_id ? ElementVerdict::kOwn
                                                                            : ElementVerdict::kForeign;
        }
        round.latest_hit = std::max(round.latest_hit, stamp);
      } else if (round.first_through == 0 || stamp < round.first_through) {
        round.first_through = stamp;
      }
      have = true;
    }
    if (have) round.verdicts.push_back(verdict);
  }
  return round;
}

std::vector<uint64_t> elementKeys(const spark_dsg::Mesh& mesh, const BoundingBox& bbox,
                                  float element_size) {
  std::vector<uint64_t> keys;
  for (const auto& sample : sampleSurface(mesh, bbox, element_size)) {
    keys.push_back(packElementCell(sample.cell));
  }
  return keys;
}

}  // namespace khronos
