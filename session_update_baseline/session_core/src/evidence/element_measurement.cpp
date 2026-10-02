#include "session_core/evidence/element_measurement.h"

#include <algorithm>
#include <cmath>

#include "session_core/surface/surface_sampling.h"

namespace khronos {
namespace {

struct Projection {
  bool valid = false;
  double rho = 0.0, reading = 0.0, incidence = 0.0;
  ProjectedEndpointEvidence evidence;
};

Projection project(const PhysicalEvidenceStore::Snapshot& evidence, TimeStamp stamp,
                   const SurfaceSample& sample) {
  Projection result;
  result.evidence = evidence.project(stamp, sample.point);
  if (result.evidence.endpoint.type == EndpointClass::kUnavailable) return result;
  result.valid = true;
  result.rho = result.evidence.query_range_m;
  result.reading = result.evidence.endpoint.measured_depth_m;
  // Normal incidence of the ray; an element without a normal is taken head-on.
  const double cosine = sample.has_normal
      ? std::abs(static_cast<double>(sample.normal.dot(result.evidence.view_direction_world))) : 1.0;
  result.incidence = std::acos(std::min(1.0, cosine));
  return result;
}

}  // namespace

ElementRound measureElements(const PhysicalEvidenceStore::Snapshot& evidence, size_t physical_id,
                             const spark_dsg::Mesh& mesh, const BoundingBox& bbox,
                             const model::RangeModel& psi, const ElementSource& source,
                             TimeStamp earliest, TimeStamp latest,
                             model::SensorCalibrator* calibrator) {
  ElementRound round;
  if (!evidence) return round;
  const auto stamps = evidence.timestamps(earliest, latest);
  if (stamps.empty()) return round;
  const auto samples = sampleSurface(mesh, bbox, source.element_size, kElementBudget);
  round.num_elements = samples.size();
  if (!psi.valid()) return round;
  const double h = source.element_size;

  // Per element: acquisition time (t_e of (6s)), cross-session flag, the recorded range rho_e.
  std::vector<uint8_t> cross(samples.size(), 0);
  std::vector<float> element_range(samples.size(), 0.f);
  bool any_cross = false;
  for (size_t i = 0; i < samples.size(); ++i) {
    const TimeStamp element_time = samples[i].first_seen > 0 ? samples[i].first_seen : source.fallback_time;
    cross[i] = element_time < source.session_start;
    any_cross = any_cross || cross[i];
    if (cross[i] && source.element_range) {
      const auto found = source.element_range->find(packElementCell(samples[i].cell));
      if (found != source.element_range->end()) element_range[i] = found->second;
    }
  }
  const auto biasOf = [&](size_t i, double rho, const model::RangeModel& model) {
    return model.bias(rho, element_range[i] > 0.f ? element_range[i] : rho, source.element_zeta);
  };
  // The offsets of the readings of a previous session's elements inside the comparison band
  // B = T + (|zeta_now| + |zeta_e|) rho, after the predicted scale displacement is removed.
  const auto addBandPair = [&](size_t i, const Projection& p, const model::RangeModel& model) {
    if (!calibrator || !(source.truncation > 0.0)) return;
    const double offset = p.reading - p.rho;
    const double band = source.truncation + (std::abs(model.zeta) + std::abs(source.element_zeta)) * p.rho;
    if (!(std::abs(offset) <= band)) return;
    const double base = model.sigmaBase(p.rho, p.incidence, h);
    calibrator->addBandPair({offset - biasOf(i, p.rho, model), base * base});
  };

  // sigma_x is estimated from the first overlap before the elements of a previous session are
  // judged; without an estimate they are not judged at all.
  model::RangeModel judged = psi;
  bool pairs_added = false;
  if (any_cross && !psi.sigma_x_known) {
    if (calibrator) {
      for (const auto stamp : stamps) {
        for (size_t i = 0; i < samples.size(); ++i) {
          if (!cross[i]) continue;
          const auto p = project(evidence, stamp, samples[i]);
          if (p.valid && std::isfinite(p.reading) && p.reading >= p.evidence.sensor_min_range &&
              p.reading <= p.evidence.sensor_max_range) {
            addBandPair(i, p, psi);
          }
        }
      }
      pairs_added = true;
      double sigma_x = 0.0;
      if (calibrator->estimateSigmaX(sigma_x)) {
        judged.sigma_x = sigma_x;
        judged.sigma_x_known = true;
      }
    }
  }

  struct Pending {
    bool have = false;
    ElementVerdict verdict;
  };
  std::vector<Pending> latest_verdict(samples.size());
  std::vector<ElementLearning> counts;
  if (source.placement_round) counts.resize(samples.size());
  std::vector<uint32_t> frame_own, frame_through;
  std::vector<float> frame_own_range;
  for (const auto stamp : stamps) {
    frame_own.clear();
    frame_through.clear();
    frame_own_range.clear();
    double frame_labelled_foreign = 0.0, frame_labelled = 0.0;
    for (size_t i = 0; i < samples.size(); ++i) {
      const bool is_cross = cross[i] != 0;
      if (is_cross && !judged.sigma_x_known) {
        if (stamp == stamps.front()) ++round.cross_session_skipped;
        continue;
      }
      const auto p = project(evidence, stamp, samples[i]);
      if (!p.valid) continue;
      const double rho = p.rho, reading = p.reading;
      const double sigma = judged.sigmaEff(rho, p.incidence, h, is_cross);
      const double bias = is_cross ? biasOf(i, rho, judged) : 0.0;
      const auto kind = model::classifyRange(judged, reading, rho, sigma, p.evidence.sensor_min_range,
                                             p.evidence.sensor_max_range, bias);
      if (kind == model::RangeClass::kInvalid) {
        ++round.invalid;
        continue;
      }
      if (is_cross && !pairs_added) addBandPair(i, p, judged);
      if (kind == model::RangeClass::kOccluded) {
        ++round.occluded;
        continue;
      }
      ElementVerdict verdict;
      verdict.key = packElementCell(samples[i].cell);
      verdict.through = kind == model::RangeClass::kThrough;
      verdict.stamp = stamp;
      verdict.predicted = static_cast<float>(
          judged.predictedSeeThrough(rho, sigma, p.evidence.sensor_max_range));
      bool own_hit = false;
      if (!verdict.through) {
        const auto& e = p.evidence.endpoint;
        if (e.type == EndpointClass::kPhysical) {
          own_hit = static_cast<size_t>(e.physical_id) == physical_id;
          verdict.label = own_hit ? ElementVerdict::kOwn : ElementVerdict::kForeign;
          frame_labelled += 1.0;
          if (!own_hit) frame_labelled_foreign += 1.0;
        }
        round.latest_hit = std::max(round.latest_hit, stamp);
      } else if (round.first_through == 0 || stamp < round.first_through) {
        round.first_through = stamp;
      }
      latest_verdict[i].have = true;
      latest_verdict[i].verdict = verdict;
      // README principle 3: the rays of the round, one per (frame, element): a ray that passes the
      // element (T) contradicts the placement, one that lands on its own identity (H) supports it.
      if (source.placement_round) {
        if (verdict.through) {
          ++round.contradict_rays;
        } else if (own_hit) {
          ++round.support_rays;
          round.latest_support = std::max(round.latest_support, stamp);
        }
      }
      if (source.placement_round) {
        if (own_hit) {
          frame_own.push_back(static_cast<uint32_t>(i));
          frame_own_range.push_back(static_cast<float>(rho));
        } else if (verdict.through) {
          frame_through.push_back(static_cast<uint32_t>(i));
        }
      }
    }
    // README principle 6 (2): the frame directly saw the placement in place.
    if (source.placement_round && source.k_min > 0 && frame_own.size() >= source.k_min &&
        frame_own.size() > frame_through.size()) {
      round.recognized = true;
      round.latest_recognized = std::max(round.latest_recognized, stamp);
      round.own_hits += frame_labelled - frame_labelled_foreign;
      round.foreign_hits += frame_labelled_foreign;
      for (size_t k = 0; k < frame_own.size(); ++k) {
        auto& c = counts[frame_own[k]];
        c.key = packElementCell(samples[frame_own[k]].cell);
        ++c.hits;
        c.range_sum += frame_own_range[k];
      }
      for (const auto i : frame_through) {
        auto& c = counts[i];
        c.key = packElementCell(samples[i].cell);
        ++c.through;
      }
    }
  }
  round.verdicts.reserve(samples.size());
  for (const auto& item : latest_verdict) {
    if (item.have) round.verdicts.push_back(item.verdict);
  }
  for (const auto& c : counts) {
    if (c.hits > 0 || c.through > 0) round.learning.push_back(c);
  }
  return round;
}

std::vector<uint64_t> elementKeys(const spark_dsg::Mesh& mesh, const BoundingBox& bbox,
                                  float element_size) {
  std::vector<uint64_t> keys;
  for (const auto& sample : sampleSurface(mesh, bbox, element_size, kElementBudget)) {
    keys.push_back(packElementCell(sample.cell));
  }
  return keys;
}

}  // namespace khronos
