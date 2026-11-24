#pragma once
// When a physical object begins to be present, from the session's own stored pixels: the observed-
// absence test run backwards over the rounds before its first real observation
// (SurfaceEvidence::presenceStart). A run of looks that saw the site empty dates the start between
// its latest measurement and the next measurement of the site taken: with nothing in the data to
// prefer a moment within that interval, its middle is the estimate of least expected error. Without
// such a run the site was never seen empty and the object stands from the start. Its end is the state
// layer's alone (PersistentObjectState::applyPhysicalGeometry): an identity stays present while it
// has a current state and leaves when its last state closes, so a moved object is never absent
// between its segments.
#include <cstdint>
#include <optional>
#include <vector>

#include "khronos/common/common_types.h"
#include "session_core/evidence/surface_evidence.h"

namespace khronos::session {

// Opens a node's presence (the state layer ends it), keeping its real observation bounds apart
// (persistObservationBounds); returns its first real observation.
TimeStamp openPresence(KhronosObjectAttributes& attrs);

// The dated start of the presence of a node of `physical_id` first observed at first_seen (0: from the
// start), or none when the session holds no measurement of that time (first observed before the
// session's first stored frame: an earlier session dated it) or the node has no surface to sample.
// Called once per round with the round's stamp as `latest`.
std::optional<TimeStamp> presenceStart(const SurfaceEvidence& evidence, const SurfaceEvidence::Snapshot& snapshot,
                                       const std::vector<uint64_t>& round_ends, size_t physical_id,
                                       const KhronosObjectAttributes& attrs, TimeStamp first_seen,
                                       TimeStamp latest);

}  // namespace khronos::session
