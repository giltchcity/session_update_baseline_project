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
#include "session_core/evidence/observed_absence.h"
#include "session_core/runtime/session_bundle.h"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <malloc.h>
#include <glog/logging.h>
namespace khronos {
namespace {
// Return memory freed at session end (frame archive, stored evidence, volumes)
// to the system: glibc keeps freed heap pages of the arenas otherwise.
void releaseFreedMemory() { malloc_trim(0); }
}  // namespace


void Backend::setPhysicalEvidenceStore(PhysicalEvidenceStore::Ptr store) {
  physical_evidence_store_ = store;
  change_detector_->setPhysicalEvidenceStore(std::move(store));
}

void Backend::setMapScales(const SessionRefusion::Scales& scales) { map_scales_ = scales; }

void Backend::setFrameArchive(FrameArchive::Ptr archive) { frame_archive_ = std::move(archive); }

void Backend::setShownMemory(SessionRefusion::Surface shown) {
  shown_memory_ = std::make_unique<SessionRefusion::Surface>(std::move(shown));
  inherited_current_keys_.clear();
  for (const auto id : persistent_objects_.trackedIds()) {
    const auto current = persistent_objects_.currentFragment(id);
    if (current) inherited_current_keys_.emplace(id, current->evidence_key);
  }
}

void Backend::setPreviousDepthScales(std::vector<float> scales) {
  previous_depth_scales_ = std::move(scales);
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
  const ObservedAbsenceBatch evidence_batch(verificator->observedAbsenceModel());

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
        [](PersistentObjectState::SurfaceEvidence& target,
           const RayVerificator::SurfaceEvidenceCounts& result) {
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
          target.reliable_points = result.reliable_points;
        };
    const auto measure = [&](const PersistentObjectState::FragmentView& fragment,
                             const int state_slot) {
      bool projected = false;
      auto counts = verificator->countCurrentPhysicalSurface(
          id, *fragment.geometry, *fragment.bbox, evidence,
          std::max(fragment.last_support_time, fragment.last_confirmed_support), stamp, &projected,
          state_slot, fragment.birth_time, fragment.evidence_key, fragment.inherited, fragment.last_support_time);
      LOG(INFO) << "STATE_EVIDENCE_WINDOW inst=" << id
                << " after=" << std::max(fragment.last_support_time, fragment.last_confirmed_support)
                << " latest_measured_support=" << counts.latest_support_stamp << " through=" << stamp
                << " projected=" << projected
                << " support=" << counts.support_rays
                << " contradiction=" << counts.contradiction_rays
                << " reliable=" << counts.reliable_samples
                << " reliable_in_view=" << counts.reliable_in_view
                << " reliable_seen_through=" << counts.reliable_seen_through
                << " absence_llr=" << counts.absence_llr
                << " absent_samples=" << counts.contradicted_surface_samples
                << " total_samples=" << counts.surface_samples
                << " absence_coverage_sufficient=" << counts.absence_coverage_sufficient;
      return counts;
    };
    if (current && current->geometry && current->geometry->numVertices() > 0) {
      copy_evidence(inherited_evidence, measure(*current, 0));
      inherited_evidence.evidence_key = current->evidence_key;
      inherited_evidence.geometry_revision = current->geometry_revision;
      inherited_evidence.measured_through = stamp;
    }
    if (session_current && session_current->geometry &&
        session_current->geometry->numVertices() > 0) {
      copy_evidence(session_evidence, measure(*session_current, 1));
      session_evidence.evidence_key = session_current->evidence_key;
      session_evidence.geometry_revision = session_current->geometry_revision;
      session_evidence.measured_through = stamp;
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
  verificator->observedAbsenceModel().retain(persistent_objects_.liveEvidenceKeys());
  return closed;
}

void Backend::updateFinalMap() {
  if (!session_extensions_enabled_) return;
  std::lock_guard<std::mutex> map_lock(map_mutex_);
  session_terminal_ready_ = false;
  if (map_.numTimeSteps() == 0) {
    throw std::runtime_error("Terminal session map is empty");
  }
  // The terminal change-detection pass has just written the final snapshot:
  // the object reasoning's final state, which the next session reasons on.
  // Update a copy of it and replace that snapshot at the same timestamp; every
  // earlier snapshot and every change-detection input stays as it was.
  const size_t last = map_.numTimeSteps() - 1;
  const TimeStamp stamp = map_.stamps()[last];
  const auto final_dsg = map_.rawDsg(last);
  if (!final_dsg) throw std::runtime_error("Terminal session snapshot is unavailable");
  unconsolidated_final_ = final_dsg->clone();
  unconsolidated_stamp_ = stamp;
  // The terminal change detection was the last reader of the stored evidence
  // frames; release them before the update.
  if (physical_evidence_store_) physical_evidence_store_->clear();
  releaseFreedMemory();
  if (!config.refuse_final_map) {
    final_surface_error_.assign(SessionRefusion::fromDsg(*final_dsg).faces.size(),0.f);
    session_terminal_ready_ = true;
    return;
  }
  if (!frame_archive_) throw std::runtime_error("Required session frame archive is unavailable");
  auto edited = final_dsg->clone();
  // A failed required estimator aborts terminal publication; the original snapshot
  // remains owned by map_ until the complete edited snapshot is available.
  refuseFinalMap(*edited, stamp);
  map_.update(edited, stamp);
  session_terminal_ready_ = true;
  // The frame archive and the update's volumes are gone now; hand the freed
  // memory back before the terminal save.
  releaseFreedMemory();
}

void Backend::refuseFinalMap(DynamicSceneGraph& edited, TimeStamp stamp) {
  const auto start = std::chrono::steady_clock::now();
  SessionRefusion::Inputs inputs;
  const auto frames = frame_archive_->release(&inputs.camera);
  if (frames.empty()) throw std::runtime_error("Required session frame archive is empty");
  inputs.frames = &frames;
  inputs.scales = map_scales_;
  inputs.final_stamp = stamp;
  inputs.shown = shown_memory_.get();
  inputs.previous_depth_scales = previous_depth_scales_;
  if (const char* dump = std::getenv("KHRONOS_REFUSION_DUMP")) inputs.dump_dir = dump;
  // README (8a)-(8b): measurement time domain and previous-surface ownership
  // come from the same current state, with separate time and identity meanings.
  for (const size_t id : persistent_objects_.trackedIds()) {
    const auto current = persistent_objects_.currentFragment(id);
    inputs.state_starts[id] = current ? std::optional<TimeStamp>(current->birth_time) : std::nullopt;
    const auto previous = inherited_current_keys_.find(id);
    if (!current || previous == inherited_current_keys_.end() ||
        previous->second != current->evidence_key)
      inputs.replaced_states.insert(id);
  }
  SessionRefusion::Config refusion_config;
  refusion_config.num_threads = config.session_end_threads;
  const SessionRefusion refusion(refusion_config);
  auto refused = refusion.apply(edited, inputs);
  if (!refused.applied) throw std::runtime_error("Required session surface update did not complete");
  refusion_report_ = std::move(refused.report_json);
  if (refused.applied) {
    session_depth_scale_ = refused.depth_scale;
    final_surface_error_ = std::move(refused.surface_error);
  }
  LOG(INFO) << "[SessionRefusion] applied=" << refused.applied << " " << refused.summary
            << " elapsed_s="
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// README (4b): a requested inference event completes before the next backend
// packet advances the source graph. Called after releasing the graph mutex.
void Backend::sessionCompleteUpdate() { waitForChangeDetection(); }

// README (6d): fix the sensor-time boundary before native and project queries.
void Backend::sessionBeforeDetect(TimeStamp stamp) {
  const auto verificator = change_detector_->getRayVerificator();
  if (!verificator) throw std::logic_error("Session evidence model is unavailable");
  verificator->setPhysicalEvidenceCutoff(stamp);
}

void Backend::sessionBeforeReconcile(
    const DynamicSceneGraph::Ptr& dsg, Changes& changes, TimeStamp stamp, bool finalize_pending) {
  persistent_objects_.ingestObjects(*dsg);
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
    const auto verificator = change_detector_->getRayVerificator();
    if (verificator) {
      const RayChangeDetector physical_changes(
          change_detector_->config.ray_change_detector);
      markClosedObjectBackground(*dsg->mesh(), persistent_objects_, *verificator,
                                 physical_changes, object_surface_resolution_,
                                 stamp, changes.background_changes);
    }
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
    // Canonicalization materializes resolved geometry. Re-measure that geometry
    // after the final settlement before publishing the terminal state.
    // Reconciliation changed mesh indices: rebuild once here rather than
    // querying the stale pre-reconciliation ray index.
    change_detector_->setDsg(dsg);
    const size_t terminal_closed = verifyCurrentObjectStates(stamp);
    persistent_objects_.finalizePendingAbsences(stamp);
    Changes terminal_changes;
    if (dsg->hasMesh()) {
      const auto verificator = change_detector_->getRayVerificator();
      if (verificator) {
        const RayChangeDetector physical_changes(change_detector_->config.ray_change_detector);
        const auto background_closed = markClosedObjectBackground(
            *dsg->mesh(), persistent_objects_, *verificator, physical_changes,
            object_surface_resolution_, stamp, terminal_changes.background_changes);
        if (background_closed) {
          // No object changes: only the newly confirmed background removals.
          reconciler_->reconcile(*dsg, terminal_changes, stamp);
        }
      }
    }
    UpdateKhronosObjectsFunctor::canonicalizePhysicalObjects(*dsg, &persistent_objects_);
    LOG(INFO) << "TERMINAL_STATE_DRAIN stamp=" << stamp << " closed=" << terminal_closed;
  }
}
void Backend::prepareSessionSave(const hydra::DataDirectory& log_setup) {
  if (!session_terminal_ready_ || !unconsolidated_final_)
    throw std::logic_error("Complete terminal processing before publishing a session");
  session_io::beginBundle(log_setup.path());
}

void Backend::saveSessionState(const hydra::DataDirectory& log_setup, bool primary_saved) {
  if (!primary_saved) throw std::runtime_error("Required final map save failed");
  if (!session_terminal_ready_ || !unconsolidated_final_)
    throw std::runtime_error("Session terminal state was not prepared");
  const auto path = log_setup.path();
  std::vector<std::string> members{"final.4dmap.zpk","chain_state.4dmap.zpk",
      "shown_state.4dmap.zpk","registry_state.cbor","evidence_state.cbor","sensor_statistics.txt","depth_scales.txt"};
  const auto verificator = change_detector_->getRayVerificator();
  if (!verificator) throw std::logic_error("Session evidence model is unavailable");
  auto& absence = verificator->observedAbsenceModel();
  absence.save((path / "evidence_state.cbor").string(),unconsolidated_stamp_,
               persistent_objects_.liveEvidenceKeys());
  if (!absence.saveSensorStatistics((path / "sensor_statistics.txt").string())) {
    throw std::runtime_error("Failed to save absence sensor statistics");
  }
  if (unconsolidated_final_) {
      // The next session reasons on the object reasoning's final state and
      // shows the updated final map as its memory.
      SpatioTemporalMap chain(config.spatio_temporal_map);
      chain.update(unconsolidated_final_->clone(), unconsolidated_stamp_);
      if (!chain.save(path / "chain_state.4dmap.zpk")) {
        throw std::runtime_error("Failed to save session reasoning state");
      }
      persistent_objects_.saveCheckpoint((path / "registry_state.cbor").string(),
          (path / "chain_state.4dmap.zpk").string(), unconsolidated_stamp_, *unconsolidated_final_);
      {
        // What this session's final map shows: the next session's memory.
        const size_t last = map_.numTimeSteps() - 1;
        SpatioTemporalMap shown(config.spatio_temporal_map);
        shown.update(map_.rawDsg(last)->clone(), map_.stamps()[last]);
        if (!shown.save(path / "shown_state.4dmap.zpk")) {
          throw std::runtime_error("Failed to save shown session state");
        }
        auto surface = SessionRefusion::fromDsg(*map_.rawDsg(last));
        surface.face_error = final_surface_error_;
        SessionRefusion::saveSurfaceError((path / "surface_error.bin").string(), surface);
        members.push_back("surface_error.bin");
      }
      if (!refusion_report_.empty()) {
        std::ofstream report(path / "refusion_report.json");
        report.exceptions(std::ios::badbit | std::ios::failbit);
        report << refusion_report_ << "\n";
        report.close();
        members.push_back("refusion_report.json");
      }
      {
        // Calibration audit. Per-face source error drives surface correspondence.
        std::ofstream scales(path / "depth_scales.txt");
        scales.exceptions(std::ios::badbit | std::ios::failbit);
        for (const float s : previous_depth_scales_) scales << s << "\n";
        if (session_depth_scale_) scales << *session_depth_scale_ << "\n";
        scales.close();
      }
    }
  session_io::publishBundle(path, unconsolidated_stamp_, members);
}

}  // namespace khronos
