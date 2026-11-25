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

#include <malloc.h>

#include "khronos_ros/khronos_pipeline.h"

#include <khronos/active_window/object_extraction/mesh_object_extractor.h>

#include <filesystem>
#include <memory>
#include <stdexcept>

#include <config_utilities/config_utilities.h>
#include <config_utilities/parsing/context.h>
#include <hydra/common/global_info.h>
#include <hydra/common/pipeline_queues.h>
#include <hydra_ros/backend/ros_backend_publisher.h>
#include <hydra_ros/frontend/ros_frontend_publisher.h>
#include <khronos/active_window/active_window.h>
#include <khronos/utils/khronos_attribute_utils.h>

#include "khronos_ros/utils/ros_namespaces.h"

namespace khronos {

KhronosPipeline::Config::Config() : hydra::HydraRosPipeline::Config() {
  active_window = ActiveWindow::Config{};
  verbosity = hydra::GlobalInfo::instance().getConfig().default_verbosity;
}

void declare_config(KhronosPipeline::Config& config) {
  using namespace config;
  name("KhronosPipeline::Config");
  base<hydra::HydraRosPipeline::Config>(config);
}

KhronosPipeline::KhronosPipeline(ianvs::NodeHandle nh)
    : hydra::HydraRosPipeline(0, 5),
      config(config::checkValid(config::fromContext<Config>())),
      nh_(nh),
      physical_evidence_store_(std::make_shared<PhysicalEvidenceStore>()),
      khronos_backend_(nullptr),
      khronos_active_window_(nullptr) {}

void KhronosPipeline::init() {
  HydraRosPipeline::init();
  khronos_active_window_ = dynamic_cast<ActiveWindow*>(active_window_.get());
  khronos_backend_ = dynamic_cast<session::MappingBackend*>(backend_.get());

  if (khronos_active_window_) {
    khronos_active_window_->setPhysicalEvidenceStore(physical_evidence_store_);
  }
  if (khronos_backend_) {
    session::MappingBackend::SessionInputs session;
    session.evidence = physical_evidence_store_;
    if (khronos_active_window_) {
      session.object_surface_resolution = khronos_active_window_->config.volumetric_map.voxel_size;
      if (khronos_backend_->usesPieceHistory()) {
        // The 4D map's history as versioned pieces, split by the active window's own voxel blocks.
        const auto& volume = khronos_active_window_->config.volumetric_map;
        const double block_edge = static_cast<double>(volume.voxel_size) * volume.voxels_per_side;
        khronos_backend_->usePieceHistory(block_edge);
        LOG(INFO) << "[4DMap] piece history, block edge " << block_edge << " m";
      }
    }
    // The persistent surface uses the same native reconstruction settings as
    // the active window; tracking retains its original temporal lifetime.
    if (khronos_active_window_) {
      const auto& window = khronos_active_window_->config;
      auto present = std::make_shared<PresentSurface>(
          window.volumetric_map, window.projective_integrator, window.mesh_integrator);
      khronos_active_window_->setPresentSurface(present);
      session.present = present;
      LOG(INFO) << "[PresentSurface] native Hydra voxel " << window.volumetric_map.voxel_size
                << " truncation " << window.volumetric_map.truncation_distance;
    }
    // The session takes what it shares with the active window.
    khronos_backend_->connectSession(session);
  }

  backend_->addSink(
      hydra::BackendModule::Sink::fromCallback([this](uint64_t timestamp_ns, const auto& dsg, const auto& dfg) {
        if (backend_evaluation_callback_) {
        backend_evaluation_callback_(timestamp_ns, dsg, dfg);
        }
      }));

  if (khronos_active_window_) {
    khronos_active_window_->addKhronosSink(ActiveWindow::KhronosSink::fromCallback(
        [this](const auto& frame_data, const auto& map, const auto& tracks) {
          if (aw_evaluation_callback_) {
          aw_evaluation_callback_(map, frame_data, tracks);
          }
        }));
  } else {
    LOG(ERROR) << "Active window is not khronos::ActiveWindow!";
  }

  if (!khronos_backend_) {
    LOG(ERROR) << "Backend is not the session's backend (SessionBackend)!";
  }
}

void KhronosPipeline::stop() {
  std::lock_guard<std::mutex> lock(stop_mutex_);
  if (stop_complete_) {
    return;
  }

  // First prevent new input and drain the ordinary active-window queue. Then
  // emit the active window's remaining mesh/object packet while frontend and
  // backend are still alive. HydraRosPipeline::stop subsequently drains those
  // downstream queues in order.
  if (input_module_) {
    input_module_->stop();
  }
  if (active_window_) {
    active_window_->stop();
  }
  if (khronos_active_window_) {
    khronos_active_window_->finishMapping();
  }

  hydra::HydraRosPipeline::stop();
  // Every queue is drained: the frames kept for object extraction are not read
  // any more. Release them before the session-end consolidation and save.
  if (khronos_active_window_) {
    khronos_active_window_->releaseFrameData();
  }
  malloc_trim(0);
  finishProcessing();
  stop_complete_ = true;
}

bool KhronosPipeline::save(const hydra::DataDirectory& log_setup, bool save_full_state) {
  if (!log_setup) {
    LOG(WARNING) << "[Khronos Pipeline] Could not save: Invalid log setup.";
    return false;
  }

  // Save state of all modules.
  if (save_full_state) {
    // HydraRosPipeline::save calls save on all modules including backend
    HydraRosPipeline::save(log_setup);
  } else {
    // Production checkpoint: the reconciled 4D map is authoritative. Do not
    // clone and serialize the same multi-hundred-MB DSG through frontend,
    // backend, JSON and evaluation paths when save_full_state is false.
    if (khronos_backend_) {
      khronos_backend_->saveFinalMap(log_setup);
    }
    return true;
  }

  // Full/debug save only.
  auto dsg = backend_dsg_->graph->clone();
  const auto backend_path = log_setup.path("backend");

  if (khronos_active_window_) {
    // Add all objects that are currently in the active window.
    auto aw_objects = khronos_active_window_->extractObjects();
    size_t current_object_id = 0;
    for (const auto& [id, attrs] : dsg->getLayer(DsgLayers::OBJECTS).nodes()) {
      current_object_id = std::max(current_object_id, NodeSymbol(id).categoryId());
    }

    for (auto& object : aw_objects) {
      NodeSymbol object_symbol('O', ++current_object_id);
      dsg->emplaceNode(DsgLayers::OBJECTS,
                       object_symbol,
                       std::make_unique<KhronosObjectAttributes>(std::move(*object)));
    }
  }

  dsg->save(backend_path / "dsg_with_mesh", true);
  CLOG(2) << "[Khronos Pipeline] Saved " << (save_full_state ? "full state" : "evaluation DSG")
          << " to '" << log_setup.path() << "'.";
  return true;
}

void KhronosPipeline::finishProcessing() {
  if (!khronos_backend_) {
    throw std::runtime_error(
        "Cannot finalize the pipeline: the backend is not the session's backend");
  }
  khronos_backend_->finishProcessing();
}

std::string KhronosPipeline::getConfigInfo() const {
  // TODO(lschmid): Find a more general way to do this when integrating with hydra.
  std::stringstream ss;
  ss << config::toString(hydra::GlobalInfo::instance().getConfig()) << "\n";
  ss << config::toString(config) << "\n";
  ss << config::toString(active_window_->config) << "\n";
  ss << config::toString(frontend_->config) << "\n";
  ss << config::toString(backend_->config);
  return ss.str();
}

void KhronosPipeline::setActiveWindowEvaluationCallback(const ActiveWindowEvaluationCallback& cb) {
  aw_evaluation_callback_ = cb;
}

void KhronosPipeline::setBackendEvaluationCallback(const BackendEvaluationCallback& callback) {
  backend_evaluation_callback_ = callback;
}

}  // namespace khronos
