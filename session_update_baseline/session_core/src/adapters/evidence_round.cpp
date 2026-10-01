#include "session_core/adapters/evidence_round.h"

#include <algorithm>
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
}  // namespace

// README (6d), (6m), (5e): one decision round. A frozen evidence snapshot gives every placement
// the verdicts of its elements in the inputs after its watermark; the registry resolves each
// identity (Persistence Filter recursion, commitments, candidates) and learns from the rounds
// it committed; the rounds after a committed end teach the ended distribution.
size_t runEvidenceRound(PersistentObjectState& registry, ObservedAbsenceModel& calibration,
                        const PhysicalEvidenceStore::Snapshot* evidence_snapshot, TimeStamp stamp,
                        float element_size, double frame_interval) {
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
  // README principle 7: the truncation of the object layer, T = 2 h_o from the current estimate.
  const double truncation = 2.0 * psi.objectResolution(frame_interval);

  size_t closed = 0;
  for (const size_t id : registry.trackedIds()) {
    const auto current = registry.currentFragment(id);
    PersistentObjectState::RoundInput input;
    input.session_start = session_start;
    std::vector<PersistentObjectState::PairInput> pairs;
    if (current && current->geometry && current->geometry->numVertices() > 0) {
      const ObservedAbsenceModel::StateKey key{id, current->evidence_key};
      // README (1), principle 1: the new sources of this placement lie after its watermark and
      // not before its birth.
      const uint64_t first = std::max<uint64_t>(calibration.processed(key) + 1, current->birth_time);
      input.evidence_key = current->evidence_key;
      input.geometry_revision = current->geometry_revision;
      input.window_begin = first;
      input.measured_through = stamp;
      if (first <= stamp) {
        const ElementSource source{element_size, session_start, current->birth_time, truncation};
        input.elements = measureElements(evidence, id, *current->geometry, *current->bbox, psi,
                                         source, first, stamp, &statistics->calibrator);
      }
      calibration.setProcessed(key, stamp);
      size_t through = 0;
      for (const auto& v : input.elements.verdicts) through += v.through;
      LOG(INFO) << "STATE_EVIDENCE_ROUND inst=" << id << " key=" << current->evidence_key
                << " through=" << stamp << " elements=" << input.elements.num_elements
                << " verdicts=" << input.elements.verdicts.size() << " see_through=" << through
                << " occluded=" << input.elements.occluded << " invalid=" << input.elements.invalid
                << " odds_before=" << current->odds;
    }
    if (current) {
      // README (13), E_{h->o}: the exclusion evidence of each pending candidate whose present
      // geometry has not been evaluated against this placement.
      for (const auto& candidate : registry.pendingNeedingExclusion(id)) {
        if (!candidate.geometry || candidate.geometry->numVertices() == 0) continue;
        PersistentObjectState::PairInput pair;
        pair.evidence_key = candidate.evidence_key;
        pair.geometry_revision = candidate.geometry_revision;
        pair.current_key = current->evidence_key;
        // Frames of this session while the placement was in place and before the candidate.
        const TimeStamp from = std::max(current->presence_begin, session_start);
        const TimeStamp support = latestSupportOf(*current);
        const TimeStamp to = std::min(support, candidate.birth_time > 0 ? candidate.birth_time - 1 : 0);
        if (from <= to) {
          const ElementSource source{element_size, session_start, candidate.birth_time, truncation};
          pair.exclusion = measureElements(evidence, id, *candidate.geometry, *candidate.bbox, psi,
                                           source, from, to, nullptr);
        }
        // The free space the placement's own earlier session observed (V_free, eq. (15b)).
        const auto& free_space = calibration.freeSpace();
        if (current->presence_begin < session_start && free_space.size() > 0) {
          std::set<uint64_t> known;
          for (const auto& v : pair.exclusion.verdicts) known.insert(v.key);
          const TimeStamp span_end = std::min(support, session_start - 1);
          for (const auto& sample : sampleSurface(*candidate.geometry, *candidate.bbox,
                                                  element_size)) {
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
        pairs.push_back(std::move(pair));
      }
    }
    const auto round = registry.resolveRound(id, input, pairs, stamp);
    if (round.closed) ++closed;
  }


  // README principle 6: the rounds that follow a committed end teach the ended distribution.
  for (const auto& [id, view] : registry.endedWatch()) {
    const uint64_t first = std::max<uint64_t>(view.ended_processed, view.ended_since) + 1;
    if (first > stamp) continue;
    const ElementSource source{element_size, session_start, view.birth_time, truncation};
    const auto round = measureElements(evidence, id, *view.geometry, *view.bbox, psi, source,
                                       first, stamp, nullptr);
    registry.addEndedRound(id, view.evidence_key, round, stamp);
  }
  calibration.retain(registry.liveEvidenceKeys());
  return closed;
}

}  // namespace khronos
