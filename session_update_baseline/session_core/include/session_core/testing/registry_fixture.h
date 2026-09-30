#pragma once

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "session_core/state/persistent_object_state.h"
#include "session_core/testing/fixtures.h"

/** Crafted rounds for the checks of the registry (not used by the system): the verdicts an
 * evidence store would produce, written down directly. */
namespace khronos::testing {

constexpr TimeStamp kSecond = 1'000'000'000ULL;

inline void require(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << "\n";
    std::exit(EXIT_FAILURE);
  }
}

/** A registry that has learned, from a long quiet history, that a placement in place shows hits
 * (so one round of hits confirms it) and that a placement that ended shows see-through. */
inline void trainRegistry(PersistentObjectState& registry, double construction_hits = 5.0) {
  model::PersistencePrior prior;
  model::RoundModel rounds;
  trainedStatistics(prior, rounds, 9000, 3);
  registry.importStatistics(std::move(prior), std::move(rounds));
  registry.setConstructionHits(construction_hits);
  registry.setMapResolution(0.05f);
}

/** A round of the current placement of `id` at `stamp` with `hits` elements hit and `through`
 * elements passed (see-through); the elements are numbered from 1. */
inline PersistentObjectState::RoundInput craftRound(const PersistentObjectState& registry, size_t id,
                                                    TimeStamp stamp, size_t hits, size_t through,
                                                    TimeStamp session_start = 1) {
  const auto current = registry.currentFragment(id);
  require(current.has_value(), "the identity has a current placement to measure");
  PersistentObjectState::RoundInput input;
  input.evidence_key = current->evidence_key;
  input.geometry_revision = current->geometry_revision;
  input.window_begin = current->birth_time;
  input.measured_through = stamp;
  input.session_start = session_start;
  uint64_t key = 1;
  for (size_t i = 0; i < hits; ++i, ++key) {
    ElementVerdict v;
    v.key = key;
    v.stamp = stamp;
    v.label = ElementVerdict::kOwn;
    input.elements.verdicts.push_back(v);
  }
  for (size_t i = 0; i < through; ++i, ++key) {
    ElementVerdict v;
    v.key = key;
    v.through = true;
    v.stamp = stamp;
    input.elements.verdicts.push_back(v);
  }
  input.elements.num_elements = hits + through;
  if (hits > 0) input.elements.latest_hit = stamp;
  if (through > 0) input.elements.first_through = stamp;
  return input;
}

/** The round of a placement that is seen again: every element hit. */
inline PersistentObjectState::RoundResult confirm(PersistentObjectState& registry, size_t id,
                                                  TimeStamp stamp, size_t elements = 12) {
  return registry.resolveRound(id, craftRound(registry, id, stamp, elements, 0), {}, stamp);
}

/** The round of a placement that is seen through: every element passed. */
inline PersistentObjectState::RoundResult contradict(PersistentObjectState& registry, size_t id,
                                                     TimeStamp stamp, size_t elements = 12) {
  return registry.resolveRound(id, craftRound(registry, id, stamp, 0, elements), {}, stamp);
}

}  // namespace khronos::testing
