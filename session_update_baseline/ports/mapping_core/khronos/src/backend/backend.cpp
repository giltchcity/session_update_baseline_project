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

#include <chrono>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <filesystem>
#include <iomanip>
#include <sstream>

#include <config_utilities/types/path.h>
#include <glog/logging.h>
#include <hydra/backend/mst_factors.h>
#include <hydra/common/global_info.h>
#include <hydra/common/pipeline_queues.h>
#include <hydra/utils/nearest_neighbor_utilities.h>
#include <hydra/utils/pgmo_mesh_traits.h>
#include <kimera_pgmo/utils/mesh_io.h>

#include "khronos/backend/change_state.h"
#include "khronos/backend/reconciliation/closed_object_background.h"
#include "khronos/common/common_types.h"
#include "khronos/utils/khronos_attribute_utils.h"

namespace khronos {

using hydra::UpdateInfo;
using spark_dsg::ObjectNodeAttributes;
using spark_dsg::PlaceNodeAttributes;
using spark_dsg::SemanticNodeAttributes;

void declare_config(Backend::Config& config) {
  using namespace config;
  name("Backend");
  base<hydra::BackendModule::Config>(config);
  field(config.verbosity, "verbosity");
  field(config.max_dt_merge_proposal, "max_dt_merge_proposal");
  field(config.optimize_on_new_merge, "optimize_on_new_merge");
  field(config.add_merge_factor, "add_merge_factor");
  field(config.pose_object_covariance, "pose_object_covariance");
  field(config.object_merge_covariance, "object_merge_covariance");
  field(config.pose_object_consistency_threshold, "pose_object_consistency_threshold");
  field(config.fix_input_pose_variance, "fix_input_pose_variance");
  field(config.fix_input_poses, "fix_input_poses");

  field(config.high_mobility_semantic_labels,
        "high_mobility_semantic_labels");

  {
    NameSpace ns("change_detection");
    field(config.run_change_detection_every_n_frames, "run_every_n_frames");
  }

  field(config.update_objects, "update_objects");
  field(config.spatio_temporal_map, "spatio_temporal_map");
  field(config.save_endpoint_snapshots_only, "save_endpoint_snapshots_only");
  field(config.reconciler, "reconciler");
  field(config.change_detection, "change_detection");

  check(config.pose_object_covariance, GT, 0, "pose_object_covariance");
  check(config.object_merge_covariance, GT, 0, "object_merge_covariance");
  check(config.pose_object_consistency_threshold, GE, 0, "pose_object_consistency_threshold");
}

Backend::Backend(const Config& config,
                 const hydra::SharedDsgInfo::Ptr& dsg,
                 const hydra::SharedModuleState::Ptr& state)
    : hydra::BackendModule(config::checkValid(config), dsg, state),
      config(config),
      map_(config.spatio_temporal_map) {
  change_detector_ = std::make_unique<SequentialChangeDetector>(config.change_detection);
  change_detector_->setDsg(unmerged_graph_);
  reconciler_ = std::make_unique<Reconciler>(config.reconciler);
  change_detection_worker_ = std::make_unique<LatestOnlyWorker>([this] {
    const auto clone_start = std::chrono::steady_clock::now();
    DynamicSceneGraph::Ptr dsg;
    RPGOMerges merges;
    TimeStamp stamp;
    bool had_loopclosure;
    {
      // Snapshot only when the single worker is ready. Requests accumulated
      // during the previous expensive reconciliation therefore cause neither
      // graph clones nor a queue of stale work.
      std::lock_guard<std::mutex> lock(mutex_);
      dsg = unmerged_graph_->clone();
      merges = proposed_merges_;
      stamp = last_timestamp_received_;
      // Consume the sticky OR-latch in the same state snapshot critical
      // section. A loop closure arriving after this exchange belongs to the
      // coalesced follow-up execution, never to this older DSG snapshot.
      had_loopclosure =
          pending_change_detection_had_loopclosure_.exchange(false,
                                                              std::memory_order_acq_rel);
    }
    const auto clone_done = std::chrono::steady_clock::now();
    change_detection_clone_time_ns_.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(clone_done - clone_start).count(),
        std::memory_order_relaxed);
    runChangeDetectionThread(
        std::move(dsg), std::move(merges), stamp, had_loopclosure);
    change_detection_compute_time_ns_.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - clone_done)
            .count(),
        std::memory_order_relaxed);
  });
}

Backend::~Backend() {
  // BackendModule stops its worker in its destructor, but that happens after
  // this derived destructor. Stop it here first so no new change-detection job
  // can be launched while we join the owned worker below.
  stop();
  waitForChangeDetection();
}

void Backend::setPhysicalEvidenceStore(PhysicalEvidenceStore::Ptr store) {
  change_detector_->setPhysicalEvidenceStore(std::move(store));
}

void Backend::setObjectSurfaceResolution(const float resolution) {
  persistent_objects_.setMapResolution(resolution);
  object_surface_resolution_ = resolution;
}

void Backend::setHighMobilitySemanticLabels(const std::vector<int>& labels) {
  persistent_objects_.setHighMobilitySemanticLabels(labels);
}

void Backend::start() { spin_thread_.reset(new std::thread(&Backend::spin, this)); }

void Backend::spin() {
  should_shutdown_ = false;
  bool should_shutdown = false;
  while (!should_shutdown) {
    auto& queue = hydra::PipelineQueues::instance().backend_queue;
    bool has_data = queue.poll();
    if (hydra::GlobalInfo::instance().force_shutdown() || !has_data) {
      // copy over shutdown request
      should_shutdown = should_shutdown_;
    }

    if (!has_data) {
      continue;
    }

    auto input = queue.pop();
    spinCallback(*input);
  }
}

void Backend::spinCallback(const hydra::BackendInput& input) {
  const uint64_t timestamp_ns = input.timestamp_ns;
  {
    // Serialize all backend state advancement and the asynchronous worker's
    // snapshot. Output sinks run below, after this lock is released: a sink may
    // synchronously wait for the just-requested change-detection job.
    std::lock_guard<std::mutex> lock(mutex_);
    status_log_.emplace_back(hydra::BackendModuleStatus{});
    last_timestamp_received_ = timestamp_ns;
    last_sequence_number_ = input.sequence_number;
    if (config.run_change_detection_every_n_frames > 0) {
      ++num_backend_updates_since_last_change_detection_;
    }

    Timer timer("backend/update", last_timestamp_received_);
    updateFactorGraph(input);
    if (config.fix_input_poses) {
      // TODO(lschmid): BROKEN Fix this, since the new hydra update we get a warning "adding node
      // measurement to a node aXXX not previously seen before"
      // fixInputPoses(input);
    }
    copyMeshDelta(input);
    updateFromLcdQueue();
    status_log_.back().total_loop_closures = num_loop_closures_;

    // TODO(lschmid): Update the hydra logic to not just drop packages in the counting.
    const bool private_dsg_updated = updatePrivateDsg(timestamp_ns, false);
    if (!private_dsg_updated) {
      CLOG(5) << "[Backend] skipping input @" << timestamp_ns << ".";
      // we only read from the frontend dsg if we've processed all the
      // factor graph update packets (as long as force_update is false)
      // we still log the status for each received frontend packet
      logStatus();
      return;
    }

    timer.reset("backend/spin");
    // Base optimization clears have_new_loopclosures_. Capture the genuine
    // input/update trigger before optimize() and use it for both scheduling and
    // the sticky OR-latch passed to change detection.
    const bool had_loopclosure = have_new_loopclosures_;
    const bool force_check_merge_proposals =
        last_timestamp_received_ >= last_merge_proposal_t_ + config.max_dt_merge_proposal * 1e9;

    if ((config.optimize_on_lc && had_loopclosure) ||
        (config.optimize_on_new_merge && !new_proposed_merges_.empty()) ||
        force_check_merge_proposals) {
      optimize(timestamp_ns, force_check_merge_proposals);
    } else {
      updateDsgMesh(timestamp_ns);
      UpdateInfo::ConstPtr info(new UpdateInfo{timestamp_ns});
      dsg_updater_->callUpdateFunctions(timestamp_ns, info);
    }

    CLOG(3) << "Proposed " << new_proposed_merges_.size() << " node merges.";

    // Run incremental change detection if desired. -1: disabled, 0: on LC only, >0: every n frames
    if (config.run_change_detection_every_n_frames >= 0) {
      const bool periodic_update = config.run_change_detection_every_n_frames > 0 &&
          num_backend_updates_since_last_change_detection_ >=
              config.run_change_detection_every_n_frames;
      if (had_loopclosure || periodic_update) {
        // The configured unit is backend packets. Reset after either trigger so
        // a loop closure is not immediately followed by a redundant periodic job.
        num_backend_updates_since_last_change_detection_ = 0;
        runChangeDetection(had_loopclosure);
      }
    }

    logStatus();
    have_new_loopclosures_ = false;
  }

  Timer sink_timer("backend/sinks", timestamp_ns);
  Sink::callAll(sinks_, timestamp_ns, *private_dsg_->graph, *deformation_graph_);
}

void Backend::runChangeDetection(bool had_loopclosure) {
  // Only true is published here; false is consumed exclusively by the worker
  // via exchange(). This is an atomic OR-latch across coalesced requests.
  if (had_loopclosure) {
    pending_change_detection_had_loopclosure_.store(true, std::memory_order_release);
  }
  change_detection_worker_->request();
}

void Backend::waitForChangeDetection() {
  change_detection_worker_->waitUntilIdle();
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
        };
    const auto measure = [&](const PersistentObjectState::FragmentView& fragment) {
      bool projected = false;
      auto counts = verificator->countCurrentPhysicalSurface(
          id, *fragment.geometry, *fragment.bbox, evidence,
          object_surface_resolution_,
          std::max(fragment.last_support_time, fragment.last_confirmed_support), stamp, &projected);
      LOG(INFO) << "STATE_EVIDENCE_WINDOW inst=" << id
                << " after=" << std::max(fragment.last_support_time, fragment.last_confirmed_support)
                << " latest_measured_support=" << counts.latest_support_stamp << " through=" << stamp
                << " projected=" << projected
                << " support=" << counts.support_rays
                << " contradiction=" << counts.contradiction_rays
                << " absent_samples=" << counts.contradicted_surface_samples
                << " total_samples=" << counts.surface_samples
                << " absence_coverage_sufficient=" << counts.absence_coverage_sufficient;
      return counts;
    };
    if (current && current->geometry && current->geometry->numVertices() > 0) {
      copy_evidence(inherited_evidence, measure(*current));
    }
    if (session_current && session_current->geometry &&
        session_current->geometry->numVertices() > 0) {
      copy_evidence(session_evidence, measure(*session_current));
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

void Backend::setInheritedGeometry(InheritedGeometry::Ptr geometry) {
  inherited_geometry_ = std::move(geometry);
}

void Backend::publishInheritedGeometry() {
  if (!inherited_geometry_ || inherited_horizon_ == 0) {
    return;
  }
  const auto mesh = private_dsg_->graph->mesh();
  if (!mesh || !mesh->has_timestamps || mesh->stamps.size() != mesh->numVertices()) {
    return;
  }
  auto& out = inherited_geometry_->mesh;
  out = spark_dsg::Mesh(mesh->has_colors, true, mesh->has_labels, mesh->has_first_seen_stamps);
  std::vector<int64_t> remap(mesh->numVertices(), -1);
  for (size_t i = 0; i < mesh->numVertices(); ++i) {
    if (mesh->stamps[i] > inherited_horizon_) continue;
    const size_t j = out.numVertices();
    remap[i] = j;
    out.resizeVertices(j + 1);
    out.setPos(j, mesh->pos(i));
    out.setTimestamp(j, mesh->stamps[i]);
    if (out.has_colors && i < mesh->colors.size()) out.setColor(j, mesh->colors[i]);
  }
  for (const auto& f : mesh->faces) {
    if (remap[f[0]] < 0 || remap[f[1]] < 0 || remap[f[2]] < 0) continue;
    out.faces.push_back({{static_cast<size_t>(remap[f[0]]),
                          static_cast<size_t>(remap[f[1]]),
                          static_cast<size_t>(remap[f[2]])}});
  }
  inherited_geometry_->horizon_ns = inherited_horizon_;
  inherited_geometry_->ready = out.numVertices() > 0;
  LOG(INFO) << "[InheritedPrior] published " << out.numVertices() << " vertices, "
            << out.numFaces() << " faces at horizon " << inherited_horizon_;
}

size_t Backend::replaceInheritedInArchivedBlocks(DynamicSceneGraph& dsg) {
  if (!inherited_geometry_ || !dsg.hasMesh()) {
    return 0;
  }
  std::vector<spatial_hash::BlockIndex> blocks;
  float block_size = 0.f;
  size_t seeded_blocks = 0, seeded_voxels = 0, pending = 0, unmeasured = 0;
  {
    std::lock_guard<std::mutex> lock(inherited_geometry_->mutex);
    blocks.swap(inherited_geometry_->archived_seeded);
    block_size = inherited_geometry_->block_size;
    seeded_blocks = inherited_geometry_->seeded_blocks;
    seeded_voxels = inherited_geometry_->seeded_voxels;
    pending = inherited_geometry_->seeded_pending.size();
    unmeasured = inherited_geometry_->unmeasured_archived;
  }
  LOG(INFO) << "[InheritedPrior] ledger: seeded_blocks=" << seeded_blocks
            << " seeded_voxels=" << seeded_voxels << " in_window=" << pending
            << " archived_now=" << blocks.size() << " kept_unmeasured=" << unmeasured;
  archived_seeded_all_.insert(blocks.begin(), blocks.end());
  if (archived_seeded_all_.empty() || block_size <= 0.f) {
    return 0;
  }
  const auto mesh = dsg.mesh();
  if (!mesh->has_timestamps || mesh->stamps.size() != mesh->numVertices()) {
    return 0;
  }
  const auto& archived = archived_seeded_all_;
  // Coverage guard: an inherited vertex inside an archived seeded block is
  // retired only where the re-integrated surface (this session's vertices)
  // already exists within one voxel of it. Where the prior did not reproduce
  // the surface, the inherited copy stays, so replacement can never lose
  // coverage; it can only remove a copy that the TSDF now represents.
  std::vector<Eigen::Vector3f> session_points;
  for (size_t i = 0; i < mesh->numVertices(); ++i) {
    if (mesh->stamps[i] > inherited_horizon_) session_points.push_back(mesh->pos(i));
  }
  if (session_points.empty()) {
    return 0;
  }
  const hydra::PointNeighborSearch session_search(session_points);
  const float guard = object_surface_resolution_;
  const float guard_sq = guard * guard;
  std::unordered_set<uint64_t> to_erase;
  size_t kept_uncovered = 0;
  for (size_t i = 0; i < mesh->numVertices(); ++i) {
    if (mesh->stamps[i] > inherited_horizon_) continue;
    const Eigen::Vector3f p = mesh->pos(i);
    const spatial_hash::BlockIndex idx(std::floor(p.x() / block_size),
                                       std::floor(p.y() / block_size),
                                       std::floor(p.z() / block_size));
    if (!archived.count(idx)) continue;
    float dist_sq = std::numeric_limits<float>::max();
    size_t nearest = 0;
    session_search.search(p, dist_sq, nearest);
    if (dist_sq <= guard_sq) {
      to_erase.insert(i);
    } else {
      ++kept_uncovered;
    }
  }
  if (!to_erase.empty()) {
    mesh->eraseVertices(to_erase);
  }
  LOG(INFO) << "[InheritedPrior] " << blocks.size() << " newly archived, "
            << archived.size() << " seeded block(s) archived in total; replaced "
            << to_erase.size() << " inherited vertices with TSDF-integrated surface, kept "
            << kept_uncovered << " the prior did not re-represent.";
  return to_erase.size();
}

void Backend::registerInheritedMemory(DynamicSceneGraph& dsg) {
  if (inherited_horizon_ == 0 || !dsg.hasMesh()) {
    return;
  }
  const auto mesh = dsg.mesh();
  if (!mesh->has_timestamps || mesh->stamps.size() != mesh->numVertices()) {
    return;
  }

  Points memory;
  Points session;
  std::vector<size_t> memory_indices;
  for (size_t i = 0; i < mesh->numVertices(); ++i) {
    if (mesh->stamps[i] <= inherited_horizon_) {
      memory.push_back(mesh->pos(i));
      memory_indices.push_back(i);
    } else {
      session.push_back(mesh->pos(i));
    }
  }

  MemoryRegistration::Config config;
  config.agreement = 0.5f * object_surface_resolution_;
  if (const auto verificator = change_detector_->getRayVerificator()) {
    config.association_tolerance = verificator->config.depth_tolerance;
  } else {
    return;
  }
  const auto result = MemoryRegistration::estimate(memory, session, config);
  if (!result.valid) {
    return;
  }

  // Apply to this round's graph, so change detection and reconciliation see
  // memory in the measured frame.
  for (size_t k = 0; k < memory_indices.size(); ++k) {
    mesh->setPos(memory_indices[k], result.memory_T_session * memory[k]);
  }

  // Apply to the live state as well, so the correction persists and the next
  // round refines it instead of re-deriving it. The deformation baseline moves
  // with the mesh: a registered seed is the new reference geometry.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto live = private_dsg_->graph->mesh();
    if (live && live->has_timestamps && live->stamps.size() == live->numVertices()) {
      for (size_t i = 0; i < live->numVertices(); ++i) {
        if (live->stamps[i] > inherited_horizon_) {
          continue;
        }
        const Point moved = result.memory_T_session * live->pos(i);
        live->setPos(i, moved);
        if (original_vertices_ && i < original_vertices_->size()) {
          auto& original = (*original_vertices_)[i];
          original.x = moved.x();
          original.y = moved.y();
          original.z = moved.z();
        }
      }
    }
  }

  memory_correction_ = result.memory_T_session * memory_correction_;
  LOG(INFO) << "MEMORY_REGISTRATION pairs=" << result.pairs
            << " rotation_deg=" << result.rotation_deg
            << " translation_m=" << result.translation_m
            << " median_residual_m=" << result.median_before_m << "->"
            << result.median_after_m;
}

void Backend::runChangeDetectionThread(DynamicSceneGraph::Ptr dsg,
                                       RPGOMerges rpgo_merges,
                                       TimeStamp stamp,
                                       bool had_loopclosure,
                                       bool finalize_pending) {
  // Currently just lock the entire map mutex to avoid threading headaches. Ideally 4D map updates
  // are not run at a fequency where this is blocking anyways.
  std::lock_guard<std::mutex> lock(map_mutex_);

  // TODO(lschmid): Decouple this for higher frequencey CD updates in the incremental visualization
  // version.
  // TODO(lschmid): Currently always reset the change detection to avoid rare (but possible) hiccups
  // from deleted active vertices. Fix this by only resetting the active window mesh.
  // Scene memory is registered into the measured frame before it is compared
  // against this session's measurements: otherwise the session-to-session
  // frame error is detected as change and stored as duplicate geometry.
  // Seeded blocks whose mesh has been archived now represent their region
  // through this session's TSDF; the frozen inherited copy is retired.
  replaceInheritedInArchivedBlocks(*dsg);
  registerInheritedMemory(*dsg);

  change_detector_->setDsg(dsg);
  auto changes =
      change_detector_->detectChanges(rpgo_merges, stamp, had_loopclosure);
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
  if (const auto verificator = change_detector_->getRayVerificator()) {
    reconciler_->setSurfaceScales(object_surface_resolution_,
                                  verificator->config.depth_tolerance);
    reconciler_->setMeasurementEvidence(verificator->physicalEvidenceSnapshot(), verificator);
  }
  reconciler_->reconcile(*dsg, changes, stamp);

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
  map_.update(dsg, stamp);

  ChangeSink::callAll(change_sinks_, stamp, changes);
  CLOG(3) << "Change detection completed: " << map_.numTimeSteps() << " time steps in 4D map.";
}

void Backend::finishProcessing() {
  std::lock_guard<std::mutex> finalize_lock(finalization_mutex_);
  if (final_processing_complete_) {
    return;
  }

  // Pipeline::stop drains active-window, frontend, and backend queues in that
  // order before this function is called. First join the last asynchronous map
  // update, then create exactly one terminal reconciled snapshot synchronously.
  waitForChangeDetection();
  std::lock_guard<std::mutex> lock(mutex_);
  updatePrivateDsg(last_timestamp_received_, true);
  have_new_loopclosures_ = true;
  optimize(last_timestamp_received_, true);
  if (config.run_change_detection_every_n_frames >= 0) {
    auto dsg = unmerged_graph_->clone();
    runChangeDetectionThread(dsg, proposed_merges_, last_timestamp_received_,
                             true, /*finalize_pending=*/true);
  } else {
    // With change detection explicitly disabled there is no reconciler output;
    // preserve the final optimized state as the sole honest fallback.
    std::lock_guard<std::mutex> map_lock(map_mutex_);
    map_.update(private_dsg_->graph->clone(), last_timestamp_received_);
  }
  const auto worker_stats = change_detection_worker_->stats();
  const auto& detector_stats = change_detector_->getRayVerificatorStatistics();
  LOG(INFO) << "[BackendPerformance] change_detection_unit=backend_update"
            << " requests=" << worker_stats.requests
            << " executions=" << worker_stats.executions
            << " coalesced=" << worker_stats.coalesced_requests
            << " clone_ms=" << change_detection_clone_time_ns_.load() / 1.0e6
            << " compute_ms=" << change_detection_compute_time_ns_.load() / 1.0e6
            << " full_resets=" << detector_stats.full_resets
            << " incremental_rebinds=" << detector_stats.incremental_rebinds
            << " incremental_updates=" << detector_stats.incremental_updates
            << " rejected_prefixes=" << detector_stats.rejected_prefixes
            << " last_prefix_check_us=" << detector_stats.last_prefix_check_us
            << " last_update_us=" << detector_stats.last_update_us
            << " indexed_poses=" << detector_stats.indexed_poses
            << " indexed_vertices=" << detector_stats.indexed_vertices
            << " indexed_objects=" << detector_stats.indexed_objects
            << " rays=" << detector_stats.rays;
  final_processing_complete_ = true;
}

void Backend::addChangeSink(const ChangeSink::Ptr& sink) {
  if (sink) {
    change_sinks_.push_back(sink);
  }
}

void Backend::optimize(size_t timestamp_ns, bool force_find_merge_proposals) {
  hydra::BackendModule::optimize(timestamp_ns, force_find_merge_proposals);
  if (have_new_loopclosures_ || force_find_merge_proposals) {
    last_merge_proposal_t_ = timestamp_ns;
  }
}

size_t Backend::findClosestNode(size_t timestamp_ns) {
  auto it = std::lower_bound(timestamps_.begin(), timestamps_.end(), timestamp_ns);
  if (it == timestamps_.end()) {
    return timestamps_.size() - 1;
  }
  return it - timestamps_.begin();
}

bool Backend::saveProposedMerges(const hydra::DataDirectory& log_setup) {
  const auto backend_path = log_setup.path("backend");
  std::lock_guard<std::mutex> lock(proposed_merges_mutex_);
  return proposed_merges_.save(backend_path / "proposed_merge.csv");
}

void Backend::save(const hydra::DataDirectory& log_setup) {
  // A periodic save may run immediately after an asynchronous change update was
  // requested. Never serialize the map while that update is still in flight.
  waitForChangeDetection();

  const auto path = log_setup.path();
  unmerged_graph_->save(path / "shared_dsg.json", false);
  private_dsg_->graph->save(path / "dsg.json", false);
  const auto mesh =
      composeCurrentSceneMesh(*private_dsg_->graph, last_timestamp_received_);
  if (mesh && !mesh->empty()) {
    // Object-labelled surfaces are stored in private meshes rather than the
    // background TSDF. The canonical saved scene always materializes both.
    kimera_pgmo::WriteMesh(path / "mesh.ply", *mesh, *mesh);
  }
  deformation_graph_->save(path / "deformation_graph.dgrf");

  // Save the proposed merges.
  saveProposedMerges(log_setup);

  saveMapAndChanges(log_setup, true);
}

void Backend::saveFinalMap(const hydra::DataDirectory& log_setup) {
  waitForChangeDetection();
  saveMapAndChanges(log_setup, false);
}

void Backend::saveMapAndChanges(const hydra::DataDirectory& log_setup,
                                bool save_individual_dsgs) {
  const auto path = log_setup.path();
  // Save the detected changes.
  const Changes& changes = change_detector_->getChanges();
  if (!changes.object_changes.empty()) {
    changes.object_changes.save(path / "object_changes.csv");
  }
  if (!changes.background_changes.empty()) {
    changes.background_changes.save(path / "background_changes.csv");
  }

  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    if (map_.numTimeSteps() == 0) {
      LOG(ERROR) << "Refusing to save an empty 4D map. Stop/drain the pipeline and call "
                    "finishProcessing() before the terminal save.";
      return;
    }
    if (config.save_endpoint_snapshots_only && map_.numTimeSteps() > 2) {
      SpatioTemporalMap endpoints(config.spatio_temporal_map);
      const auto stamps = map_.stamps();
      for (const TimeStamp stamp : {stamps.front(), stamps.back()}) {
        const auto dsg = map_.getDsgPtr(stamp);
        if (!dsg) {
          LOG(ERROR) << "Cannot materialize 4D map time step " << stamp << " for saving.";
          return;
        }
        endpoints.update(dsg->clone(), stamp);
      }
      if (endpoints.save(path / "final.4dmap")) {
        CLOG(1) << "Saved endpoint 4D map (2 of " << map_.numTimeSteps()
                << " time steps) to '" << path << "'.";
      }
    } else if (map_.save(path / "final.4dmap")) {
      CLOG(1) << "Saved 4D map with " << map_.numTimeSteps() << " time steps to '" << path << "'.";
    }

    if (!save_individual_dsgs) {
      return;
    }

    // Full/debug saves may additionally materialize every DSG. Production
    // checkpoints deliberately avoid these O(map history) JSON clones.
    const auto maps_path = path / "maps";
    if (!std::filesystem::exists(maps_path)) {
      std::filesystem::create_directories(maps_path);
    }

    const auto& timestamps = map_.stamps();
    for (size_t i = 0; i < map_.numTimeSteps(); ++i) {
      // Get the DSG for this timestamp
      auto dsg = map_.getDsgPtr(timestamps[i]);
      if (dsg) {
        // Save with timestamp as filename
        std::stringstream filename;
        filename << "dsg_" << std::setw(5) << std::setfill('0') << i
                 << "_" << timestamps[i] << ".json";
        dsg->save(maps_path / filename.str(), false);
      }
    }

    if (map_.numTimeSteps() > 0) {
      CLOG(1) << "Saved " << map_.numTimeSteps() << " individual DSGs to '" << maps_path << "'.";
    }
  }
}

void Backend::fixInputPoses(const hydra::BackendInput& input) {
  std::vector<std::pair<gtsam::Key, gtsam::Pose3>> prior_measurements;
  for (const auto& msg : input.agent_updates.pose_graphs) {
    status_log_.back().new_factors += msg.edges.size();

    for (const auto& node : msg.nodes) {
      if (node.key == 0) {
        continue;  // This should already be set by 'add_initial_prior'
      }

      prior_measurements.push_back(
          {gtsam::Symbol(kimera_pgmo::GetRobotPrefix(node.robot_id), node.key),
           gtsam::Pose3(node.pose.matrix())});
    }
  }
  deformation_graph_->processNodeMeasurements(prior_measurements, config.fix_input_pose_variance);
}

}  // namespace khronos
