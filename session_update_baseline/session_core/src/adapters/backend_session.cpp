/** -----------------------------------------------------------------------------
 * Copyright (c) 2024 Massachusetts Institute of Technology.
 * All Rights Reserved.
 *
 * AUTHORS:      Lukas Schmid <lschmid@mit.edu>, Marcus Abate <mabate@mit.edu>,
 *               Yun Chang <yunchang@mit.edu>, Luca Carlone <lcarlone@mit.edu>
 * AFFILIATION:  MIT SPARK Lab, Massachusetts Institute of Technology
 * YEAR:         2024
 * SOURCE:       https://github.com/MIT-SPARK/Khronos
 * LICENSE:      BSD 3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * -------------------------------------------------------------------------- */

#include "khronos/backend/backend.h"
#include "session_core/surface/closed_object_background.h"
#include "session_core/surface/triangle_grid.h"
#include "khronos/backend/update_khronos_objects_functor.h"
#include <spark_dsg/node_symbol.h>
#include <algorithm>
#include <string>
#include <cmath>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <malloc.h>
#include <glog/logging.h>
namespace khronos {
namespace {
// Return memory freed at session end (stored evidence) to the system: glibc
// keeps freed heap pages of the arenas otherwise.
void releaseFreedMemory() { malloc_trim(0); }
}  // namespace


void Backend::setPhysicalEvidenceStore(PhysicalEvidenceStore::Ptr store) {
  physical_evidence_store_ = store;
  change_detector_->setPhysicalEvidenceStore(std::move(store));
}

void Backend::setPresentSurface(PresentSurface::Ptr surface) {
  present_surface_ = std::move(surface);
}

void Backend::setLoadedMemory(std::vector<Eigen::Vector3f> points,
                              std::vector<std::array<uint32_t, 3>> faces) {
  loaded_memory_search_.reset();
  loaded_memory_ = std::move(points);
  loaded_memory_faces_ = std::move(faces);
  if (!loaded_memory_.empty()) {
    loaded_memory_search_ = std::make_unique<hydra::PointNeighborSearch>(loaded_memory_);
  }
  // A vertex of a round's snapshot is memory when it is a copy of a loaded vertex (P53).
  memory_copy_tolerance_ = PresentSurface::copyTolerance(loaded_memory_);
  LOG(INFO) << "[PresentSurface] memory copy tolerance m=" << memory_copy_tolerance_;
}

void Backend::setShownMemory(PresentSurface::Surface shown) {
  shown_memory_ = std::make_unique<PresentSurface::Surface>(std::move(shown));
}

void Backend::setObjectSurfaceResolution(const float resolution) {
  if (session_extensions_enabled_) persistent_objects_.setMapResolution(resolution);
  object_surface_resolution_ = resolution;
}

void Backend::setHighMobilitySemanticLabels(const std::vector<int>& labels) {
  if (session_extensions_enabled_) persistent_objects_.setHighMobilitySemanticLabels(labels);
}

size_t Backend::verifyCurrentObjectStates(const TimeStamp stamp) {
  const auto verificator = change_detector_->getRayVerificator();
  if (!verificator) {
    return 0;
  }
  // One frozen snapshot for the whole pass, so an asynchronous frame ingest cannot split a single
  // state decision across two store versions.
  const auto evidence = verificator->physicalEvidenceSnapshot();

  size_t closed = 0;
  for (const size_t id : persistent_objects_.trackedIds()) {
    const auto current = persistent_objects_.currentFragment(id);
    const auto session_current = persistent_objects_.sessionCurrentFragment(id);
    if ((!current || !current->geometry || current->geometry->numVertices() == 0) &&
        (!session_current || !session_current->geometry ||
         session_current->geometry->numVertices() == 0)) {
      continue;
    }

    // Surface evidence, not sparse-vertex evidence. Every sensor ray counts
    // once, no matter how many mesh samples it crosses.
    PersistentObjectState::SurfaceEvidence inherited_evidence;
    PersistentObjectState::SurfaceEvidence session_evidence;
    const auto copy_evidence =
        [&](PersistentObjectState::SurfaceEvidence& target,
            const RayVerificator::SurfaceEvidenceCounts& result, int state_slot,
            const uint64_t state_id) {
          target.latest_support_stamp = result.latest_support_stamp;
          target.absence_coverage_sufficient = result.absence_coverage_sufficient;
          target.support_rays = result.support_rays;
          target.contradiction_rays = result.contradiction_rays;
          target.surface_samples = result.surface_samples;
          target.supported_votes = result.supported_votes;
          target.free_space_votes = result.free_space_votes;
          target.replaced_by_other_votes = result.replaced_by_other_votes;
          target.replaced_by_background_votes =
              result.replaced_by_background_votes;
          target.occluded_votes = result.occluded_votes;
          target.unobserved_samples = result.unobserved_samples;
          target.reliable_in_view = result.reliable_in_view;
          target.reliable_seen_through = result.reliable_seen_through;
          target.reliable_samples = result.reliable_samples;
          const auto likelihood =
              physicalAbsenceLookLikelihood(verificator.get(), id, state_slot, stamp, state_id);
          target.measured_absence_log_ratio = likelihood.log_ratio;
          target.has_measured_absence_likelihood = likelihood.has_measurement;
          target.has_calibrated_absence_source = likelihood.calibrated_source;
        };
    const auto measure = [&](const PersistentObjectState::FragmentView& fragment,
                             const int state_slot) {
      bool projected = false;
      const double prior_log_odds = persistent_objects_.changePriorLogOdds(id, state_slot);
      auto counts = verificator->countCurrentPhysicalSurface(
          id, *fragment.geometry, *fragment.bbox, evidence,
          object_surface_resolution_,
          std::max(fragment.last_support_time, fragment.last_confirmed_support), stamp, &projected,
          state_slot, fragment.birth_time, prior_log_odds, fragment.uid);
      LOG(INFO) << "STATE_EVIDENCE_WINDOW inst=" << id
                << " after=" << std::max(fragment.last_support_time, fragment.last_confirmed_support)
                << " latest_measured_support=" << counts.latest_support_stamp << " through=" << stamp
                << " projected=" << projected
                << " support=" << counts.support_rays
                << " contradiction=" << counts.contradiction_rays
                << " reliable=" << counts.reliable_samples
                << " reliable_in_view=" << counts.reliable_in_view
                << " reliable_seen_through=" << counts.reliable_seen_through
                << " absence_llr=" << counts.absence_llr << " prior_log_odds=" << prior_log_odds
                << " absent_samples=" << counts.contradicted_surface_samples
                << " total_samples=" << counts.surface_samples
                << " absence_coverage_sufficient=" << counts.absence_coverage_sufficient;
      return counts;
    };
    if (current && current->geometry && current->geometry->numVertices() > 0) {
      copy_evidence(inherited_evidence, measure(*current, 0), 0, current->uid);
    }
    if (session_current && session_current->geometry &&
        session_current->geometry->numVertices() > 0) {
      copy_evidence(session_evidence, measure(*session_current, 1), 1, session_current->uid);
    }
    // Per-slice six-class evidence ledger (STATE_SLICE): every change
    // detection round records what the RGB-D actually measured at the old
    // site, per surface sample. This is the ground-truth trace that answers
    // "did the map follow the real world, and when".
    LOG(INFO) << "STATE_SLICE inst=" << id
              << " stamp=" << stamp
              << " inherited_sup=" << inherited_evidence.support_rays
              << " inherited_con=" << inherited_evidence.contradiction_rays
              << " inherited_samples=" << inherited_evidence.surface_samples
              << " inherited_supported=" << inherited_evidence.supported_votes
              << " inherited_free=" << inherited_evidence.free_space_votes
              << " inherited_repl_other="
              << inherited_evidence.replaced_by_other_votes
              << " inherited_repl_bg="
              << inherited_evidence.replaced_by_background_votes
              << " inherited_occ=" << inherited_evidence.occluded_votes
              << " inherited_unobs=" << inherited_evidence.unobserved_samples
              << " session_sup=" << session_evidence.support_rays
              << " session_con=" << session_evidence.contradiction_rays
              << " session_samples=" << session_evidence.surface_samples
              << " cur_verts="
              << (current && current->geometry
                      ? current->geometry->numVertices()
                      : 0)
              << " candidate_verts="
              << (session_current && session_current->geometry
                      ? session_current->geometry->numVertices()
                      : 0);
    if (persistent_objects_.resolveCurrentEvidence(
            id, inherited_evidence, session_evidence, stamp)) {
      ++closed;
    }
  }
  return closed;
}

void Backend::releaseSessionEvidence() {
  if (!session_extensions_enabled_) return;
  // The terminal change-detection pass has just written the final snapshot and
  // was the last reader of the stored evidence frames.
  if (physical_evidence_store_) physical_evidence_store_->clear();
  releaseFreedMemory();
}

void Backend::sessionBeforeReconcile(
    const DynamicSceneGraph::Ptr& dsg, Changes& changes, TimeStamp stamp, bool finalize_pending) {
  // One change-detection round is one observation event: every state decision
  // until the next round reads the persistence history resolved before it.
  persistent_objects_.beginObservationEvent(stamp);
  // Object CURRENT states must face the same measurements the background mesh does. Before the
  // reconciler touches any mesh, while the ray index still matches the geometry it was built from.
  const size_t closed = verifyCurrentObjectStates(stamp);
  if (closed > 0) {
    CLOG(3) << "[Backend] Closed " << closed
            << " current object fragment(s) contradicted by later free-space evidence.";
  }
  if (finalize_pending) {
    persistent_objects_.finalizePendingAbsences(stamp);
  }
  if (dsg->hasMesh()) {
    const auto closed_surfaces = closedStateSurfaces(persistent_objects_, stamp);
    markClosedObjectBackground(*dsg->mesh(), closed_surfaces,
                               object_surface_resolution_,
                               changes.background_changes);
  }
}
void Backend::sessionAfterReconcile(
    const DynamicSceneGraph::Ptr& dsg, TimeStamp stamp, bool finalize_pending) {
  // Change detection must see every visibility segment independently. In
  // particular, an old physical object can be cleared at its previous site
  // while a newer segment with the same stable ID materializes its current
  // geometry elsewhere. Folding the segments before the evidence transition
  // makes the old-site ray result incorrectly close the new current segment.
  // Reduce to one logical node per physical ID only after every segment has
  // been detected and reconciled.
  UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(*dsg, &persistent_objects_);
  if (finalize_pending) {
    // Canonicalization above ingests the last extractor segments. They did
    // not exist in the registry during the preceding state decision. Drain
    // them before the sole terminal snapshot, otherwise a late old segment
    // can remain CURRENT while a newer observed site waits for a next round
    // that will never happen (Synthetic A I69).
    // Reconciliation changed mesh indices: rebuild once here rather than
    // querying the stale pre-reconciliation ray index.
    change_detector_->setDsg(dsg);
    const size_t terminal_closed = verifyCurrentObjectStates(stamp);
    persistent_objects_.finalizePendingAbsences(stamp);
    Changes terminal_changes;
    if (dsg->hasMesh()) {
      const auto closed_surfaces = closedStateSurfaces(persistent_objects_, stamp);
      const auto background_closed = markClosedObjectBackground(
          *dsg->mesh(), closed_surfaces, object_surface_resolution_,
          terminal_changes.background_changes);
      if (background_closed) {
        // No object changes: only the newly confirmed background removals.
        reconciler_->reconcile(*dsg, terminal_changes, stamp);
      }
    }
    UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(*dsg, &persistent_objects_);
    LOG(INFO) << "TERMINAL_STATE_DRAIN stamp=" << stamp << " closed=" << terminal_closed;
  }
  if (!present_surface_) return;
  // The legal contribution domain (README section 5): an identity whose
  // current state began in this session contributes from that state's first
  // sighting on (the registry's track_first_seen, else its birth).
  std::map<size_t, TimeStamp> state_starts;
  const TimeStamp session_start = present_surface_->firstStamp();
  for (const size_t id : persistent_objects_.trackedIds()) {
    const auto current = persistent_objects_.currentFragment(id);
    if (!current || current->birth_time < session_start) continue;
    state_starts[id] =
        current->track_first_seen > 0 ? current->track_first_seen : current->birth_time;
    if (finalize_pending) {
      LOG(INFO) << "[PresentSurface] current state began in this session: id=" << id
                << " birth=" << current->birth_time << " first_sighting=" << state_starts[id]
                << " (session start " << session_start << ")";
    }
  }
  PresentSurface::Inputs inputs;
  inputs.shown = shown_memory_.get();
  // The state layer already knows which states closed. The shown memory reads the
  // same ownership, so a closed state's old-site faces follow its death atomically
  // instead of waiting for the per-face evidence to see through every pixel.
  for (const auto& state : closedStateSurfaces(persistent_objects_, stamp)) {
    inputs.closed_memory.insert(inputs.closed_memory.end(), state.points.begin(),
                                state.points.end());
  }
  if (finalize_pending) {
    if (const char* dump = std::getenv("KHRONOS_REFUSION_DUMP")) inputs.dump_dir = dump;
  }
  // Without a shown state, memory is what the snapshot copies of the loaded
  // state (P53). DIAG in the terminal round (P53 data, no decision reads it):
  // distances of the snapshot's vertices to the loaded memory, to its nearest
  // vertex and to its surface (exact point-triangle distance up to 1 m), in 20
  // bins per decade from 1e-9 m. The grid cell only sets the search cost.
  std::unique_ptr<TriangleGrid> memory_surface;
  std::vector<std::atomic<uint64_t>> vertex_hist(201), surface_hist(201);
  const auto bin_of = [](const float d) {
    if (!(d > 0.f)) return size_t{0};
    return static_cast<size_t>(std::clamp(std::floor((std::log10(d) + 9.f) * 20.f), 0.f, 200.f));
  };
  if (loaded_memory_search_) {
    if (finalize_pending && !loaded_memory_faces_.empty()) {
      memory_surface = std::make_unique<TriangleGrid>(loaded_memory_, loaded_memory_faces_, nullptr,
                                                      std::max(object_surface_resolution_, 0.02f));
    }
    inputs.is_memory = [&, this](const Eigen::Vector3f& p) {
      float d_sq = 0.f;
      size_t idx = 0;
      const bool found = loaded_memory_search_->search(p, d_sq, idx);
      if (finalize_pending) {
        if (found) vertex_hist[bin_of(std::sqrt(d_sq))].fetch_add(1, std::memory_order_relaxed);
        if (memory_surface) {
          float d = 0.f;
          Eigen::Vector3f q;
          uint32_t face = 0;
          surface_hist[memory_surface->closest(p, 1.f, d, q, face) ? bin_of(d) : 200].fetch_add(
              1, std::memory_order_relaxed);
        }
      }
      return found && d_sq <= memory_copy_tolerance_ * memory_copy_tolerance_;
    };
  }
  auto round = present_surface_->compose(*dsg, stamp, state_starts, inputs, finalize_pending);
  if (!finalize_pending) return;
  if (inputs.is_memory) {
    std::string v, s;
    for (size_t i = 0; i < vertex_hist.size(); ++i) {
      v += (i ? "," : "") + std::to_string(vertex_hist[i].load());
      s += (i ? "," : "") + std::to_string(surface_hist[i].load());
    }
    LOG(INFO) << "MEMORY_DISTANCE_DIAG bins_per_decade=20 from=1e-9 vertex=" << v << " surface=" << s;
  }
  refusion_report_ = std::move(round.report_json);
}

// The 4D history, a partition of its own (SpatioTemporalMap::kHistoryPartition): the places a physical
// identity left, moved or taken away. Its states at one place (overlapping surface boxes) are one place;
// following states there are one stay. A place whose last state there was closed is one history node: the
// surface last observed there, present over the identity's intervals within its stays there, each from its
// first sighting there (the identity's first state: from the start) until it was closed there. The objects
// layer is not touched; the query view shows the identity outside these intervals. Snapshot only: the
// working graph holds no history.
void Backend::sessionWriteHistory(DynamicSceneGraph& snapshot) const {
  if (!snapshot.hasLayer(DsgLayers::OBJECTS)) return;
  const auto box = [](const BoundingBox& bbox) {
    Eigen::AlignedBox3f world;
    for (const auto& corner : bbox.corners()) world.extend(corner);
    return world;
  };
  std::vector<std::pair<NodeId, size_t>> identities;
  for (const auto& [id, node] : snapshot.getLayer(DsgLayers::OBJECTS).nodes()) {
    const auto* attrs = node->tryAttributes<KhronosObjectAttributes>();
    const auto physical = attrs ? UpdateKhronosObjectsFunctor::physicalInstanceId(*attrs) : std::nullopt;
    if (physical && !attrs->first_observed_ns.empty()) identities.emplace_back(id, *physical);
  }
  for (const auto& [node_id, physical] : identities) {
    const auto& current = snapshot.getNode(node_id).attributes<KhronosObjectAttributes>();
    const auto& first = current.first_observed_ns;
    const auto& last = current.last_observed_ns;
    if (last.size() != first.size()) continue;
    const auto states = persistent_objects_.historyFragments(physical);
    // Places: the last state at each place, and the stays there [first sighting, closure).
    std::vector<std::pair<size_t, std::vector<std::pair<TimeStamp, TimeStamp>>>> places;
    size_t previous = 0;
    for (size_t i = 0; i < states.size(); ++i) {
      if (!states[i].bbox) continue;
      const TimeStamp closure = states[i].death_time.value_or(std::numeric_limits<TimeStamp>::max());
      const auto place = static_cast<size_t>(
          std::find_if(places.begin(), places.end(), [&](const auto& candidate) {
            return box(*states[candidate.first].bbox).intersects(box(*states[i].bbox));
          }) - places.begin());
      if (place < places.size() && place == previous) {
        places[place].second.back().second = closure;
      } else {
        const TimeStamp sighting = places.empty() ? 0
            : states[i].track_first_seen > 0 ? states[i].track_first_seen : states[i].birth_time;
        if (place == places.size()) places.emplace_back();
        places[place].second.emplace_back(sighting, closure);
      }
      places[place].first = i;
      previous = place;
    }
    for (auto& [index, stays] : places) {
      const auto& state = states[index];
      if (!state.death_time || !state.geometry || state.geometry->numVertices() == 0) continue;
      std::sort(stays.begin(), stays.end());
      std::vector<TimeStamp> place_first, place_last;
      for (size_t k = 0; k < first.size(); ++k) {
        for (const auto& [sighting, closure] : stays) {
          const TimeStamp begin = std::max(first[k], sighting);
          const TimeStamp end = std::min(last[k], closure);
          if (begin >= end) continue;
          if (!place_last.empty() && begin <= place_last.back()) {
            place_last.back() = std::max(place_last.back(), end);
          } else {
            place_first.push_back(begin);
            place_last.push_back(end);
          }
        }
      }
      if (place_first.empty()) continue;
      auto left = std::make_unique<KhronosObjectAttributes>(current);
      left->mesh = *state.geometry;
      left->bounding_box = *state.bbox;
      left->position = state.position;
      left->trajectory_timestamps.clear();
      left->trajectory_positions.clear();
      left->dynamic_object_points.clear();
      left->first_observed_ns = place_first;
      left->last_observed_ns = place_last;
      snapshot.emplaceNode(snapshot.getLayerKey(DsgLayers::OBJECTS)->layer, NodeSymbol('E', state.uid),
                           std::move(left), SpatioTemporalMap::kHistoryPartition);
    }
  }
}

void Backend::saveSessionState(const hydra::DataDirectory& log_setup) {
  const auto path = log_setup.path();
  saveAbsenceSensorStatistics((path / "sensor_statistics.txt").string());
    if (map_.numTimeSteps() > 0) {
      // One boundary surface: the consolidated final map is what the next
      // session reasons on and what it shows. It is the map without its 4D
      // history, so the history never enters the next session's mapping.
      const size_t final_index = map_.numTimeSteps() - 1;
      auto final_map = map_.rawDsg(final_index)->clone();
      if (const auto objects = final_map->getLayerKey(DsgLayers::OBJECTS)) {
        final_map->removeLayer(objects->layer, SpatioTemporalMap::kHistoryPartition);
      }
      SpatioTemporalMap chain(config.spatio_temporal_map);
      chain.update(final_map->clone(), map_.stamps()[final_index]);
      if (!chain.save(path / "chain_state.4dmap.zpk")) {
        LOG(ERROR) << "Failed to save the chain state to '" << path << "'.";
      }
      {
        // What this session's final map shows: the next session's memory.
        SpatioTemporalMap shown(config.spatio_temporal_map);
        shown.update(final_map, map_.stamps()[final_index]);
        if (!shown.save(path / "shown_state.4dmap.zpk")) {
          LOG(ERROR) << "Failed to save the shown state to '" << path << "'.";
        }
      }
      if (!refusion_report_.empty()) {
        std::ofstream(path / "refusion_report.json") << refusion_report_ << "\n";
      }
    }
}

}  // namespace khronos
