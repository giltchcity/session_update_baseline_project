#include "session_core/evidence/element_measurement.h"

#include <algorithm>
#include <cmath>

#include "session_core/surface/surface_sampling.h"
#include "session_core/surface/triangle_grid.h"

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

// The entity of a measured mesh: its stored surface in world coordinates (for "lands on another part
// of the entity") and its extent, the box of the mesh (for "lands inside the entity's range").
class Entity {
 public:
  Entity(const spark_dsg::Mesh& mesh, const BoundingBox& bbox, float cell) : bbox_(bbox) {
    vertices_.reserve(mesh.numVertices());
    for (const auto& point : mesh.points) vertices_.push_back(bbox.pointToWorldFrame(point));
    faces_.reserve(mesh.faces.size());
    for (const auto& face : mesh.faces) {
      faces_.push_back({static_cast<uint32_t>(face[0]), static_cast<uint32_t>(face[1]),
                        static_cast<uint32_t>(face[2])});
    }
    if (!faces_.empty()) grid_ = std::make_unique<TriangleGrid>(vertices_, faces_, nullptr, cell);
  }

  /** The reading's 3D point lies within `margin` of the entity's stored surface. */
  bool onSurface(const Eigen::Vector3f& point, float margin) const {
    if (!grid_) return false;
    float distance = 0.f;
    Eigen::Vector3f nearest;
    uint32_t face = 0;
    return grid_->closest(point, margin, distance, nearest, face);
  }
  /** The reading's 3D point lies inside the entity's extent expanded by `margin`. */
  bool inExtent(const Eigen::Vector3f& point, float margin) const {
    return pointInExtent(bbox_, point, margin);
  }

 private:
  const BoundingBox& bbox_;
  std::vector<Eigen::Vector3f> vertices_;
  std::vector<TriangleGrid::Face> faces_;
  std::unique_ptr<TriangleGrid> grid_;
};

// What the newest informative frame of a sample said (principle 6: the look keeps the latest
// outcome of each sample).
enum class Outcome : uint8_t { kNone, kHitPlain, kHitOwn, kHitForeign, kThrough, kEntityOwn, kEntityOther };

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
  const auto samples = sampleSurface(mesh, bbox, source.element_size, model::kElementBudget);
  round.num_elements = samples.size();
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
  // The offsets of the readings of a previous session's elements inside the window of (12d),
  // W = tau + |zeta_now| rho + |zeta_e| rho_e, after the predicted scale displacement is removed.
  const auto addBandPair = [&](size_t i, const Projection& p, const model::RangeModel& model) {
    if (!calibrator || !(source.half_voxel > 0.0)) return;
    const double offset = p.reading - p.rho;
    // W of (12d): tau + |zeta_now| rho + |zeta_e| rho_e, tau = max(h, sigma_table(rho)).
    const double rho_e = element_range[i] > 0.f ? element_range[i] : p.rho;
    const double window = std::max(source.half_voxel, model.sigmaTable(p.rho)) +
                          std::abs(model.zeta) * p.rho + std::abs(source.element_zeta) * rho_e;
    if (!(std::abs(offset) <= window)) return;
    const double base = model.sigmaBase(p.rho, p.incidence, h);
    calibrator->addBandPair({offset - biasOf(i, p.rho, model), base * base, window});
  };

  // sigma_x is estimated from the first overlap before the elements of a previous session are
  // judged; without an estimate they are not judged at all.
  model::RangeModel judged = psi;
  bool pairs_added = false;
  if (any_cross && !psi.sigma_x_known && calibrator) {
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
    // The posterior of the prior (the previous session's sigma_x, else W / sqrt(3), as one pair)
    // and these pairs.
    double sigma_x = 0.0;
    if (calibrator->estimateSigmaX(source.prior_sigma_x, source.prior_sigma_x_known, sigma_x)) {
      judged.sigma_x = sigma_x;
      judged.sigma_x_known = true;
    }
  }

  const Entity entity(mesh, bbox, source.element_size);

  struct Pending {
    bool have = false;
    bool extent_own = false;  // some frame read the entity's own identity inside its extent
    ElementVerdict verdict;
    Outcome outcome = Outcome::kNone;
    TimeStamp outcome_stamp = 0;
    float outcome_range = 0.f;
  };
  std::vector<Pending> latest_sample(samples.size());
  double delta_plus_sum = 0.0;
  size_t delta_plus_count = 0;
  for (const auto stamp : stamps) {
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
      auto kind = model::classifyRange(judged, reading, rho, sigma, p.evidence.sensor_min_range,
                                       p.evidence.sensor_max_range, bias);
      if (kind == model::RangeClass::kInvalid) {
        ++round.invalid;
        continue;
      }
      if (is_cross && !pairs_added) addBandPair(i, p, judged);
      const auto& e = p.evidence.endpoint;
      const bool physical = e.type == EndpointClass::kPhysical;
      const bool own_identity = physical && static_cast<size_t>(e.physical_id) == physical_id;
      auto& slot = latest_sample[i];
      // The reading's 3D point, and whether it lies inside the entity's extent with the entity's
      // own identity (principle 4, assumption (6)).
      const auto bounds = judged.bounds(rho, sigma, p.evidence.sensor_max_range);
      delta_plus_sum += bounds.plus;
      ++delta_plus_count;
      const float margin = static_cast<float>(bounds.plus + std::abs(bias));
      const Eigen::Vector3f origin = samples[i].point - static_cast<float>(rho) * p.evidence.view_direction_world;
      const Eigen::Vector3f landed = origin + static_cast<float>(reading) * p.evidence.view_direction_world;
      if (own_identity && entity.inExtent(landed, margin)) slot.extent_own = true;
      if (kind == model::RangeClass::kThrough) {
        // README principle 4, assumption (6): a reading that passes the sample but lands on another
        // part of the entity's stored surface, or inside the entity's extent with its own identity,
        // says the entity is still there -- O for the sample.
        if (entity.onSurface(landed, margin) || (own_identity && entity.inExtent(landed, margin))) {
          ++round.entity_passed;
          slot.have = false;  // O for the sample: its newest informative frame carries no verdict
          if (own_identity) {
            slot.outcome = Outcome::kEntityOwn;
            slot.outcome_stamp = stamp;
            slot.outcome_range = static_cast<float>(rho);
          } else {
            slot.outcome = Outcome::kEntityOther;
            slot.outcome_stamp = stamp;
          }
          continue;
        }
      }
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
      slot.outcome_stamp = stamp;
      slot.outcome_range = static_cast<float>(rho);
      if (!verdict.through) {
        if (physical) {
          verdict.label = own_identity ? ElementVerdict::kOwn : ElementVerdict::kForeign;
          slot.outcome = own_identity ? Outcome::kHitOwn : Outcome::kHitForeign;
        } else {
          slot.outcome = Outcome::kHitPlain;
        }
        round.latest_hit = std::max(round.latest_hit, stamp);
      } else {
        slot.outcome = Outcome::kThrough;
        if (round.first_through == 0 || stamp < round.first_through) round.first_through = stamp;
      }
      slot.have = true;
      slot.verdict = verdict;
    }
  }

  if (delta_plus_count > 0) round.delta_plus = delta_plus_sum / static_cast<double>(delta_plus_count);
  // README principle 6 (2): the counts of the look, one per sample, from its newest informative frame.
  for (const auto& item : latest_sample) {
    if (item.have) round.verdicts.push_back(item.verdict);
    if (item.extent_own) ++round.extent_own;
    switch (item.outcome) {
      case Outcome::kHitOwn:
      case Outcome::kEntityOwn:
        ++round.own_samples;
        round.labelled_samples += 1.0;
        if (round.first_own == 0 || item.outcome_stamp < round.first_own) round.first_own = item.outcome_stamp;
        round.latest_own = std::max(round.latest_own, item.outcome_stamp);
        break;
      case Outcome::kHitForeign:
        ++round.foreign_samples;
        round.labelled_samples += 1.0;
        break;
      case Outcome::kThrough:
        ++round.through_samples;
        break;
      default:
        break;
    }
  }
  round.recognized = source.placement_round && source.k_min > 0 && round.own_samples >= source.k_min &&
                     round.own_samples > round.through_samples;
  if (round.recognized) {
    for (size_t i = 0; i < samples.size(); ++i) {
      const auto& item = latest_sample[i];
      // h_e counts the identity hits on the sample's own surface; a reading that passed it and
      // landed on another part of the entity is the entity's identity hit (it makes the look direct)
      // but neither a hit nor a see-through of this sample.
      const bool own = item.outcome == Outcome::kHitOwn;
      const bool vetoed = item.outcome == Outcome::kThrough || item.outcome == Outcome::kHitForeign;
      if (!own && !vetoed) continue;
      ElementLearning learning;
      learning.key = packElementCell(samples[i].cell);
      learning.own = own;
      learning.vetoed = vetoed;
      learning.range = item.outcome_range;
      round.learning.push_back(learning);
    }
  }
  return round;
}

std::vector<uint64_t> elementKeys(const spark_dsg::Mesh& mesh, const BoundingBox& bbox,
                                  float element_size) {
  std::vector<uint64_t> keys;
  for (const auto& sample : sampleSurface(mesh, bbox, element_size, model::kElementBudget)) {
    keys.push_back(packElementCell(sample.cell));
  }
  return keys;
}

bool pointInExtent(const BoundingBox& bbox, const Eigen::Vector3f& world_point, float margin) {
  const Eigen::Vector3f local = bbox.pointToBoxFrame(world_point);
  const Eigen::Vector3f half = 0.5f * bbox.dimensions;
  for (int axis = 0; axis < 3; ++axis) {
    if (std::abs(local[axis]) > half[axis] + margin) return false;
  }
  return true;
}

}  // namespace khronos
