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

/** A registry that has learned, from a quiet history, that a placement in place shows hits (so
 * one look of hits confirms it) and that an object seen again after a gap has rarely changed. */
inline void trainRegistry(PersistentObjectState& registry) {
  model::PersistencePrior prior;
  model::RoundModel rounds;
  trainedStatistics(prior, rounds, 9000, 3);
  registry.importStatistics(std::move(prior), std::move(rounds));
  registry.setMapResolution(0.05f);
}

/** A look of the current placement of `id` at `stamp` with `hits` elements hit by the placement's
 * own identity and `through` elements passed (see-through); the elements are numbered from 1. A
 * look with more hits than see-throughs and at least k_min hits directly saw the placement in
 * place; a look that is only see-through carries one contradicting ray per element. */
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
  constexpr float kPredicted = 0.03f;  // the see-through rate the measurement model predicts
  uint64_t key = 1;
  for (size_t i = 0; i < hits; ++i, ++key) {
    ElementVerdict v;
    v.key = key;
    v.stamp = stamp;
    v.label = ElementVerdict::kOwn;
    v.predicted = kPredicted;
    input.elements.verdicts.push_back(v);
  }
  for (size_t i = 0; i < through; ++i, ++key) {
    ElementVerdict v;
    v.key = key;
    v.through = true;
    v.stamp = stamp;
    v.predicted = kPredicted;
    input.elements.verdicts.push_back(v);
  }
  auto& e = input.elements;
  e.num_elements = hits + through;
  if (hits > 0) e.latest_hit = stamp;
  if (through > 0) e.first_through = stamp;
  // The rays of the look, one per element.
  e.support_rays = hits;
  e.contradict_rays = through;
  if (hits > 0) e.latest_support = stamp;
  // A look with enough own hits, more than see-throughs, directly saw the placement in place: the
  // elements it hit are learned.
  if (hits >= 1 && hits > through) {
    e.recognized = true;
    e.latest_recognized = stamp;
    e.own_hits = static_cast<double>(hits);
    for (size_t i = 1; i <= hits; ++i) {
      ElementLearning learning;
      learning.key = i;
      learning.hits = 1;
      learning.range_sum = 1.0f;
      e.learning.push_back(learning);
    }
    for (size_t i = hits + 1; i <= hits + through; ++i) {
      ElementLearning learning;
      learning.key = i;
      learning.through = 1;
      e.learning.push_back(learning);
    }
  }
  return input;
}

/** The look of a placement that is seen again: every element hit. */
inline PersistentObjectState::RoundResult confirm(PersistentObjectState& registry, size_t id,
                                                  TimeStamp stamp, size_t elements = 20) {
  return registry.resolveRound(id, craftRound(registry, id, stamp, elements, 0), {}, stamp);
}

/** The look of a placement that is seen through: every element passed. */
inline PersistentObjectState::RoundResult contradict(PersistentObjectState& registry, size_t id,
                                                     TimeStamp stamp, size_t elements = 20) {
  return registry.resolveRound(id, craftRound(registry, id, stamp, 0, elements), {}, stamp);
}

}  // namespace khronos::testing
