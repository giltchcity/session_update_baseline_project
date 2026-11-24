#pragma once
// The latest static surface of a physical object's track (README M1, P33): the frames that show
// one common surface, the end of a move before them (D1), and the static fallback of a track that
// barely moved. Free functions of the session layer; the object extractor only supplies its
// frames and parameters.
#include <optional>
#include <utility>
#include <vector>

#include "khronos/active_window/data/frame_data.h"
#include "khronos/active_window/data/track.h"
#include "khronos/common/common_types.h"

namespace khronos::session {

// The comparison of two frames' views of one surface (the extractor's static_consistency_*).
struct StaticSurfaceConsistency {
  float tolerance = 0.05f;  // [m]
  int min_pixels = 20;      // free pixels a direction needs before it can speak for a move
};

// Of the track's frames (time order not required), the latest run that shows one static surface
// with the newest frame; earlier frames that a move separates from it are dropped, and `rejected`
// receives the stamp of the last dropped frame (the end of the move). Tracks without a physical
// identity, or with fewer than two frames, are returned as given.
std::vector<std::pair<FrameData::Ptr, int>> latestStaticSurface(
    std::vector<std::pair<FrameData::Ptr, int>> frames, const Track& track,
    const StaticSurfaceConsistency& consistency, TimeStamp* rejected = nullptr);

// A physical track flagged dynamic whose displacement stays below `min_displacement` is
// reconstructed as static: a copy without its motion bookkeeping; otherwise nothing.
std::optional<Track> staticFallback(const Track& track, float displacement, float min_displacement);

}  // namespace khronos::session
