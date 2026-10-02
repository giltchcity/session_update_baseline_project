#include "session_core/adapters/evidence_round.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

#include <glog/logging.h>

#include "session_core/evidence/element_measurement.h"
#include "session_core/surface/surface_sampling.h"

namespace khronos {
namespace {
// A placement's latest actual support as the registry exposes it.
TimeStamp latestSupportOf(const PersistentObjectState::FragmentView& view) {
  return std::max(view.last_support_time, view.last_confirmed_support);
}

// README principle 13 (f): the candidate's own look of this round -- its elements in the frames
// since its watermark (one look: the labels of a look are one independent unit) --, whether the
// current placement had support in the same frames (directly seen in place, or a reading of its own
// identity inside its extent), and how many samples of the candidate lie inside the extent of the
// current placement (its valid observations expanded by delta_*, principle 4).
void measureCandidateLook(const PhysicalEvidenceStore::Snapshot& evidence, ObservedAbsenceModel& calibration,
                          const model::RangeModel& psi, size_t id,
                          const PersistentObjectState::FragmentView& current,
                          const PersistentObjectState::FragmentView& candidate,
                          ElementSource candidate_source, ElementSource current_source,
                          const PersistentObjectState::RoundInput& current_input, size_t k_min,
                          TimeStamp stamp, float element_size, double max_range,
                          PersistentObjectState::PairInput& pair) {
  const ObservedAbsenceModel::StateKey key{id, candidate.evidence_key};
  const uint64_t first = std::max<uint64_t>(calibration.processed(key) + 1, candidate.birth_time);
  if (first > stamp) return;
  candidate_source.placement_round = true;
  candidate_source.k_min = k_min;
  pair.candidate_look = measureElements(evidence, id, *candidate.geometry, *candidate.bbox, psi,
                                        candidate_source, first, stamp, nullptr);
  // Principle 4: a candidate of a previous session is not judged before sigma_x is estimated.
  if (pair.candidate_look.cross_session_skipped > 0 && pair.candidate_look.verdicts.empty() &&
      pair.candidate_look.own_samples == 0) {
    return;
  }
  calibration.setProcessed(key, stamp);
  pair.look_measured = true;

  // The current placement in the same frames.
  ElementRound current_look;
  if (current_input.evidence_key != 0 && current_input.window_begin == first) {
    current_look = current_input.elements;
  } else if (current_input.evidence_key != 0 && current.geometry && current.geometry->numVertices() > 0) {
    current_source.placement_round = true;
    current_source.k_min = k_min;
    current_look = measureElements(evidence, id, *current.geometry, *current.bbox, psi, current_source,
                                   first, stamp, nullptr);
  }
  pair.current_known = current_input.evidence_key != 0 &&
      !(current_look.cross_session_skipped > 0 && current_look.verdicts.empty() &&
        current_look.own_samples == 0);
  pair.current_supported = current_look.recognized || current_look.extent_own > 0;

  // The candidate's surface against the extent of the current placement.
  if (current.inherited && !psi.sigma_x_known) return;
  const auto stamps = evidence.timestamps(first, stamp);
  if (stamps.empty() || !(max_range > 0.0)) return;
  const double rho = evidence.project(stamps.back(), candidate.bbox->world_P_center).query_range_m;
  if (!std::isfinite(rho) || !(rho > 0.0)) return;
  const double sigma = psi.sigmaEff(rho, 0.0, element_size, current.inherited);
  const double margin = psi.bounds(rho, sigma, max_range).plus +
                        (current.inherited ? std::abs(psi.zeta - current.zeta_e) * rho : 0.0);
  if (!std::isfinite(margin)) return;
  for (const auto& sample : sampleSurface(*candidate.geometry, *candidate.bbox, element_size,
                                          model::kElementBudget)) {
    ++pair.candidate_samples;
    if (pointInExtent(*current.bbox, sample.point, static_cast<float>(margin))) ++pair.candidate_in_extent;
  }
}
}  // namespace

// README (6d), (6m), (5e): one decision round. A frozen evidence snapshot gives every placement
// the verdicts of its elements in the inputs after its watermark; the registry resolves each
// identity (CUSUM (5r) with its identity channel, candidates) and learns from the looks that
// directly saw a placement in place.
size_t runEvidenceRound(PersistentObjectState& registry, ObservedAbsenceModel& calibration,
                        const PhysicalEvidenceStore::Snapshot* evidence_snapshot, TimeStamp stamp,
                        float element_size, double object_half_voxel) {
  if (!(element_size > 0.f)) {
    throw std::logic_error("The map resolution of the surface samples is not set");
  }
  const auto statistics = calibration.statistics();
  const TimeStamp session_start = calibration.sessionStart();
  // README (6m): the model held before this round predicts it.
  const auto psi = calibration.rangeModel();
  if (!evidence_snapshot || !*evidence_snapshot || !psi.valid() ||
      session_start == std::numeric_limits<TimeStamp>::max()) {
    // Nothing can be classified yet: no source is consumed.
    return 0;
  }
  const auto& evidence = *evidence_snapshot;
  const size_t k_min = registry.roundModel().minHits();
  // Principle 10: the previous session's sigma_x is the prior centre of this session's.
  const auto prior_model = calibration.priorRangeModel();
  const double max_range = statistics->max_range.load();

  size_t closed = 0;
  for (const size_t id : registry.trackedIds()) {
    const auto current = registry.currentFragment(id);
    PersistentObjectState::RoundInput input;
    input.session_start = session_start;
    std::vector<PersistentObjectState::PairInput> pairs;
    ElementSource placement_source;
    if (current && current->geometry && current->geometry->numVertices() > 0) {
      placement_source.element_size = element_size;
      placement_source.session_start = session_start;
      placement_source.fallback_time = current->birth_time;
      placement_source.half_voxel = object_half_voxel;
      placement_source.prior_sigma_x = prior_model.sigma_x;
      placement_source.prior_sigma_x_known = prior_model.sigma_x_known;
      placement_source.element_zeta = current->zeta_e;
      placement_source.element_range = registry.elementRanges(id);
      const ObservedAbsenceModel::StateKey key{id, current->evidence_key};
      // README (1), principle 1: the new sources of this placement lie after its watermark and
      // not before its birth.
      const uint64_t first = std::max<uint64_t>(calibration.processed(key) + 1, current->birth_time);
      input.evidence_key = current->evidence_key;
      input.geometry_revision = current->geometry_revision;
      input.window_begin = first;
      input.measured_through = stamp;
      if (first <= stamp) {
        ElementSource source = placement_source;
        source.placement_round = true;
        source.k_min = k_min;
        input.elements = measureElements(evidence, id, *current->geometry, *current->bbox, psi,
                                         source, first, stamp, &statistics->calibrator);
      }
      // Principle 4: a placement of a previous session is not judged before sigma_x is estimated;
      // its frames stay unconsumed and are measured again once it is.
      if (input.elements.cross_session_skipped > 0 && input.elements.verdicts.empty()) {
        input.evidence_key = 0;
      } else {
        calibration.setProcessed(key, stamp);
        LOG(INFO) << "STATE_EVIDENCE_ROUND inst=" << id << " key=" << current->evidence_key
                  << " through=" << stamp << " elements=" << input.elements.num_elements
                  << " verdicts=" << input.elements.verdicts.size()
                  << " recognized=" << input.elements.recognized
                  << " own=" << input.elements.own_samples
                  << " through=" << input.elements.through_samples
                  << " foreign=" << input.elements.foreign_samples
                  << " entity_passed=" << input.elements.entity_passed
                  << " occluded=" << input.elements.occluded << " invalid=" << input.elements.invalid
                  << " cusum=" << current->cusum;
      }
    }
    if (current) {
      // README (13): every pending candidate gets its own look of this round (f) and, where its
      // present geometry has not been evaluated against this placement, the evidence (5o).
      std::set<uint64_t> needing;
      for (const auto& view : registry.pendingNeedingExclusion(id)) needing.insert(view.evidence_key);
      for (const auto& candidate : registry.unresolvedCandidates(id)) {
        if (!candidate.geometry || candidate.geometry->numVertices() == 0) continue;
        PersistentObjectState::PairInput pair;
        pair.evidence_key = candidate.evidence_key;
        pair.geometry_revision = candidate.geometry_revision;
        pair.current_key = current->evidence_key;
        ElementSource candidate_source;
        candidate_source.element_size = element_size;
        candidate_source.session_start = session_start;
        candidate_source.fallback_time = candidate.birth_time;
        candidate_source.half_voxel = object_half_voxel;
        candidate_source.prior_sigma_x = prior_model.sigma_x;
        candidate_source.prior_sigma_x_known = prior_model.sigma_x_known;
        candidate_source.element_zeta = candidate.zeta_e;
        measureCandidateLook(evidence, calibration, psi, id, *current, candidate, candidate_source,
                             placement_source, input, k_min, stamp, element_size, max_range, pair);
        if (needing.count(candidate.evidence_key) == 0) {
          pairs.push_back(std::move(pair));
          continue;
        }
        pair.exclusion_evaluated = true;
        // E_{h->o}: frames of this session while the placement was in place and before the candidate.
        const TimeStamp from = std::max(current->presence_begin, session_start);
        const TimeStamp support = latestSupportOf(*current);
        const TimeStamp to = std::min(support, candidate.birth_time > 0 ? candidate.birth_time - 1 : 0);
        if (from <= to) {
          pair.exclusion = measureElements(evidence, id, *candidate.geometry, *candidate.bbox, psi,
                                           candidate_source, from, to, nullptr);
        }
        // The free space the placement's own earlier session observed (V_free, eq. (15b)).
        const auto& free_space = calibration.freeSpace();
        if (current->presence_begin < session_start && free_space.size() > 0) {
          std::set<uint64_t> known;
          for (const auto& v : pair.exclusion.verdicts) known.insert(v.key);
          const TimeStamp span_end = std::min(support, session_start - 1);
          for (const auto& sample : sampleSurface(*candidate.geometry, *candidate.bbox,
                                                  element_size, model::kElementBudget)) {
            const auto key = packElementCell(sample.cell);
            if (known.count(key) ||
                !free_space.freeDuring(sample.point, current->presence_begin, span_end)) {
              continue;
            }
            ElementVerdict verdict;
            verdict.key = key;
            verdict.through = true;
            verdict.stamp = span_end;
            pair.exclusion.verdicts.push_back(verdict);
          }
        }
        // E_{o->h}: the elements of the placement in the frames of the candidate.
        if (candidate.birth_time <= candidate.last_support_time && current->geometry->numVertices() > 0) {
          ElementSource pair_source = placement_source;
          pair_source.placement_round = false;
          pair.support = measureElements(evidence, id, *current->geometry, *current->bbox, psi,
                                         pair_source, candidate.birth_time, candidate.last_support_time,
                                         nullptr);
        }
        // (g): how far this session's own reconstruction lies from the inherited surface, in the
        // cross-session band of principle 4 (sigma_x and the scale displacement).
        if (current->inherited && !candidate.inherited && psi.sigma_x_known &&
            candidate.birth_time <= candidate.last_support_time) {
          const auto stamps = evidence.timestamps(candidate.birth_time, candidate.last_support_time);
          if (!stamps.empty()) {
            const auto p = evidence.project(stamps.back(), candidate.bbox->world_P_center);
            const double rho = p.query_range_m;
            if (std::isfinite(rho) && rho > 0.0 && max_range > 0.0) {
              const double sigma = psi.sigmaEff(rho, 0.0, element_size, true);
              const double band = psi.bounds(rho, sigma, max_range).plus +
                                  std::abs(psi.zeta - current->zeta_e) * rho;
              std::vector<Point> points;
              for (const auto& sample : sampleSurface(*candidate.geometry, *candidate.bbox,
                                                      element_size, model::kElementBudget)) {
                points.push_back(sample.point);
              }
              if (std::isfinite(band) && band > 0.0 && !points.empty()) {
                const auto agreement = surfaceAgreement(points, *current->geometry, *current->bbox,
                                                        static_cast<float>(band));
                pair.candidate_elements = agreement.total;
                pair.candidate_away = agreement.total - agreement.shared;
                pair.cross_session_band = true;
              }
            }
          }
        }
        pairs.push_back(std::move(pair));
      }
    }
    const auto round = registry.resolveRound(id, input, pairs, stamp);
    if (round.closed) ++closed;
    if (round.look.judged) {
      const auto& look = round.look;
      LOG(INFO) << "STATE_LOOK inst=" << id << " seen=" << look.directly_seen << " own=" << look.own_samples
                << " through=" << look.through_samples << " foreign=" << look.foreign_samples
                << " k_min=" << look.k_min << " eps=" << look.eps_hat << "(prior=" << look.eps_prior
                << ") all_vote=" << look.all_vote << " reliable=" << look.reliable_samples
                << " n=" << look.n << " f=" << look.f << " m0=" << look.predicted
                << " m=" << look.in_place.m << " c=" << look.in_place.c << "(prior=" << look.in_place_prior
                << ") lnLR=" << look.ln_lr << " w=" << look.weight << " id=" << look.ln_identity
                << " restart=" << look.coverage_restarted << " C=" << look.step.before << "->"
                << look.step.after << " exceeded=" << look.exceeded;
    }
    for (const auto& candidate : round.candidates) {
      LOG(INFO) << "STATE_CANDIDATE inst=" << id << " rule=" << static_cast<int>(candidate.rule)
                << " q=" << candidate.q << "(prior=" << candidate.q_prior << ") silent=" << candidate.silent
                << " lnLR=" << candidate.ln_lr << " ln_odds=" << candidate.ln_odds
                << " direct_looks=" << candidate.direct_looks << "/" << candidate.k_min
                << " in_extent=" << candidate.in_extent << "/" << candidate.tested
                << " h_supported=" << candidate.current_supported << " away=" << candidate.away << "/"
                << candidate.elements;
    }
  }
  calibration.retain(registry.liveEvidenceKeys());
  return closed;
}

}  // namespace khronos
