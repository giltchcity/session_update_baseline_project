/** -----------------------------------------------------------------------------
 * A forward absence verdict must not survive a later confident presence.
 *
 * Scanning forward asks whether a surface has disappeared. The detector used to
 * return at the first window whose rays were mostly empty, so a grazing or noisy
 * stretch early in a session deleted surface that the same session went on
 * measuring for another minute. On the real room-18 stage-B run that removed
 * 5.0% of the background vertices, and the reference built from the very same
 * depth images still contained 81.1% of what we dropped.
 *
 * Scanning backward asks when a surface first appeared, where presence at later
 * times is expected and must not suppress an earlier absence. That direction is
 * unchanged.
 * -------------------------------------------------------------------------- */

#include <cstdlib>
#include <iostream>
#include <vector>

#include "khronos/backend/change_detection/ray_change_detector.h"
#include "khronos/backend/change_detection/ray_verificator.h"

namespace {

constexpr uint64_t kSecond = 1000000000ull;

khronos::RayChangeDetector::Config config() {
  khronos::RayChangeDetector::Config config;
  config.temporal_resolution = 5.f;
  config.window_size = 5;
  config.use_relative_confidence = true;
  config.absence_confidence = 0.6f;
  config.presence_confidence = 0.5f;
  return config;
}

/// Rays at one-second spacing over [from, to) seconds.
void rays(std::vector<uint64_t>& out, uint64_t from, uint64_t to) {
  for (uint64_t t = from; t < to; ++t) {
    out.push_back(t * kSecond);
  }
}

bool failed(const char* what) {
  std::cerr << "FAILED: " << what << "\n";
  return true;
}

}  // namespace

int main() {
  const khronos::RayChangeDetector detector(config());
  bool bad = false;

  // T1 A surface measured empty for 30 s and never seen again has disappeared.
  {
    khronos::RayVerificator::CheckResult check;
    rays(check.present, 0, 30);
    rays(check.absent, 30, 60);
    const auto result = detector.detectChanges(check, true);
    if (!result.closest_absent) {
      bad = failed("T1 a surface that stops being measured must be absent");
    }
  }

  // T2 The same absent stretch, but the session keeps measuring the surface for
  // another 60 s afterwards. The early emptiness is contradicted, not evidence.
  {
    khronos::RayVerificator::CheckResult check;
    rays(check.present, 0, 30);
    rays(check.absent, 30, 60);
    rays(check.present, 60, 120);
    const auto result = detector.detectChanges(check, true);
    if (result.closest_absent) {
      bad = failed("T2 an absence contradicted by later presence must not stand");
    }
    if (!result.furthest_persistent) {
      bad = failed("T2 the later measurements must leave a persistent verdict");
    }
  }

  // T3 Absence at the very end still wins: the last word is emptiness.
  {
    khronos::RayVerificator::CheckResult check;
    rays(check.present, 0, 60);
    rays(check.absent, 60, 90);
    const auto result = detector.detectChanges(check, true);
    if (!result.closest_absent) {
      bad = failed("T3 absence after the last presence must stand");
    }
    if (result.furthest_persistent &&
        *result.furthest_persistent >= *result.closest_absent) {
      bad = failed("T3 the persistent verdict must predate the absence");
    }
  }

  // T4 Two absent stretches with presence between them: the surviving verdict is
  // the later one, not the first window that looked empty.
  {
    khronos::RayVerificator::CheckResult check;
    rays(check.absent, 0, 30);
    rays(check.present, 30, 60);
    rays(check.absent, 60, 90);
    const auto result = detector.detectChanges(check, true);
    if (!result.closest_absent) {
      bad = failed("T4 the trailing absence must stand");
    } else if (*result.closest_absent < 30 * kSecond) {
      std::cerr << "T4 closest_absent=" << *result.closest_absent / kSecond << "s\n";
      bad = failed("T4 the contradicted leading absence must not be reported");
    }
  }

  // T5 Backward scanning is untouched: it looks for when a surface appeared, so
  // an absence before the presence is exactly what it must report.
  {
    khronos::RayVerificator::CheckResult check;
    rays(check.absent, 0, 30);
    rays(check.present, 30, 90);
    const auto forward = detector.detectChanges(check, true);
    const auto backward = detector.detectChanges(check, false);
    if (forward.closest_absent) {
      bad = failed("T5 forward must discard the absence the presence contradicts");
    }
    if (!backward.closest_absent) {
      bad = failed("T5 backward must still report the absence before appearance");
    }
  }

  if (bad) {
    return EXIT_FAILURE;
  }
  std::cout << "absence_requires_no_later_presence: all checks passed\n";
  return EXIT_SUCCESS;
}
