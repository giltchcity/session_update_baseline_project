#pragma once
// The mapping pipeline's view of the session's backend (session_update's SessionBackend): how the
// 4D map keeps its history, what the backend shares with the active window, the terminal round
// once the pipeline drained its queues, and the saved map.
#include <memory>

#include <hydra/utils/data_directory.h>

namespace khronos {
class PhysicalEvidenceStore;
class PresentSurface;
}  // namespace khronos

namespace khronos::session {

class MappingBackend {
 public:
  // What the session takes from the pipeline: the evidence store and the present surface the active
  // window fills, and the voxel of the object reconstruction.
  struct SessionInputs {
    std::shared_ptr<PhysicalEvidenceStore> evidence;
    std::shared_ptr<PresentSurface> present;
    float object_surface_resolution = 0.05f;
  };

  virtual ~MappingBackend() = default;

  // Whether the 4D map keeps its history as versioned pieces (the session's config).
  virtual bool usesPieceHistory() const = 0;
  // block_edge: the active window's voxel block edge.
  virtual void usePieceHistory(double block_edge) = 0;
  virtual void connectSession(const SessionInputs& inputs) = 0;
  // One terminal round on the backend state; idempotent.
  virtual void finishProcessing() = 0;
  // The saved 4D map and the session state the next visit starts from.
  virtual void saveFinalMap(const hydra::DataDirectory& log_setup) = 0;
};

}  // namespace khronos::session
