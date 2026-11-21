#pragma once

#include <cstdlib>
#include <sstream>
#include <string>

#include <glog/logging.h>

namespace khronos {

/**
 * Ablation switches of the mechanism's object and element levels, the TSDF counterpart of
 * update_layer/ablation.py's UL_ABLATION (3DGS), all off by default, read once from the
 * environment. SESSION_CORE_ABLATION is a comma-separated subset of
 *   no_object_rule   object states still begin and end (the state layer's decisions and their
 *                    logs), but no state end acts on the map: no closed state's old-site background
 *                    is removed (markClosedObjectBackground), the shown memory is not hidden for a
 *                    closed state, and an identity's node keeps the surfaces of its closed states
 *                    (with no current state it stays present with them; a succeeded state's surface
 *                    stays beside its successor's)
 *   no_element_rule  the element-level evidence on the earlier visits' map (every counted frame's
 *                    vote on a face of it, the INSIDE reads) is still accumulated, but its ends are not
 *                    applied: no earlier visit's face is hidden for being seen through, and INSIDE
 *                    retires no memory vertex; this visit's own fill and, without a majority either
 *                    way, the volume's one-surface rule are decided as before
 * A set switch logs "ABLATION: mechanism ..." once.
 */
struct MechanismAblation {
  bool no_object_rule = false;
  bool no_element_rule = false;
};

inline const MechanismAblation& mechanismAblation() {
  static const MechanismAblation ablation = [] {
    MechanismAblation a;
    if (const char* value = std::getenv("SESSION_CORE_ABLATION")) {
      std::stringstream list(value);
      for (std::string item; std::getline(list, item, ',');) {
        if (item == "no_object_rule") {
          a.no_object_rule = true;
        } else if (item == "no_element_rule") {
          a.no_element_rule = true;
        } else if (!item.empty()) {
          LOG(FATAL) << "SESSION_CORE_ABLATION: unknown switch '" << item << "'";
        }
      }
      LOG(WARNING) << "ABLATION: mechanism no_object_rule=" << a.no_object_rule
                   << " no_element_rule=" << a.no_element_rule;
    }
    return a;
  }();
  return ablation;
}

/**
 * SESSION_CORE_ROUND_HANDOVER=1 (a comparison, not a component of the method): the present hands its
 * surface over at every round as a visit hands its map to the next one. At the end of a round the
 * present's faces join the shown memory, decided by the later rounds' votes like an earlier
 * visit's faces, and the present volume starts again empty; no frame is fused twice, so valid
 * domains are off. Logs "ABLATION: round handover" once.
 */
inline bool roundHandover() {
  static const bool on = [] {
    const char* value = std::getenv("SESSION_CORE_ROUND_HANDOVER");
    const bool set = value && std::string(value) == "1";
    if (set) LOG(WARNING) << "ABLATION: round handover (the present's surface handed over every round)";
    return set;
  }();
  return on;
}

}  // namespace khronos
