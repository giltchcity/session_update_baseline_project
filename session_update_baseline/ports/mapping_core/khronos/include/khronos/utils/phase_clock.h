#pragma once
// Where a session's time goes: one total (and a count) per phase, summed over every thread that
// runs it, reported in one line when the session ends (PHASE_TIMES). A phase nested in another
// (named in the report) is included in the outer one's total.
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>

namespace khronos {

enum class Phase : size_t {
  kObjectFrames,     // active window: each frame's object pixels into the objects' volumes
  kUnused,           // (no phase; keeps the report's columns)
  kExtraction,       // extraction workers: one hand-over of one track
  kShareModel,       //   in kExtraction: the buffered frame pairs teaching the share model
  kObjectTests,      //   in kExtraction: the object tests on the voted reconstruction
  kDetectChanges,    // layer round: change detector
  kBeforeReconcile,  // layer round: state decisions against the evidence
  kReconcile,        // layer round: reconciler
  kAfterReconcile,   // layer round: ingest of the handed-over segments, shown states
  kStateSurface,     //   in kAfterReconcile: a state's shown surface from its volume
  kMapUpdate,        // layer round: 4D map update and change sinks
  kEvidenceDecode,   // anywhere: unpacking a stored frame's ranges
  kCount
};

class PhaseClock {
 public:
  explicit PhaseClock(Phase phase) : phase_(static_cast<size_t>(phase)), start_(Clock::now()) {}
  ~PhaseClock() {
    totals()[phase_].ns += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_).count();
    ++totals()[phase_].calls;
  }
  PhaseClock(const PhaseClock&) = delete;
  PhaseClock& operator=(const PhaseClock&) = delete;

  static std::string report() {
    static const char* const names[] = {
        "object_frames", "unused", "extraction", "extraction.share_model",
        "extraction.object_tests", "round.detect_changes", "round.before_reconcile", "round.reconcile",
        "round.after_reconcile", "round.after_reconcile.state_surface", "round.map_update",
        "evidence_decode"};
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(1);
    for (size_t i = 0; i < static_cast<size_t>(Phase::kCount); ++i) {
      out << (i ? " " : "") << names[i] << "_s=" << totals()[i].ns.load() / 1.0e9 << "/"
          << totals()[i].calls.load();
    }
    return out.str();
  }

 private:
  using Clock = std::chrono::steady_clock;
  struct Total {
    std::atomic<int64_t> ns{0};
    std::atomic<uint64_t> calls{0};
  };
  static std::array<Total, static_cast<size_t>(Phase::kCount)>& totals() {
    static std::array<Total, static_cast<size_t>(Phase::kCount)> value;
    return value;
  }
  const size_t phase_;
  const Clock::time_point start_;
};

}  // namespace khronos
