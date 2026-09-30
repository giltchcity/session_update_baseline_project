/** -----------------------------------------------------------------------------
 * Unit checks of the unified model (README eq. (1)): the persistence prior Pi (principle 2), the
 * Persistence Filter recursion and the decision criterion D (principle 3), the round likelihood
 * of principle 6, the first-return model of principle 4, and the registry that joins them.
 * (Copied licence terms of the surrounding Khronos sources apply; see LICENSE.)
 * -------------------------------------------------------------------------- */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/node_symbol.h>

#include "session_core/model/change_filter.h"
#include "session_core/model/model_math.h"
#include "session_core/model/persistence_prior.h"
#include "session_core/model/range_model.h"
#include "session_core/model/round_model.h"
#include "session_core/state/persistent_object_state.h"
#include "session_core/testing/registry_fixture.h"
#include "khronos/backend/update_khronos_objects_functor.h"
#include "khronos/utils/khronos_attribute_utils.h"

namespace {

using khronos::TimeStamp;
using khronos::testing::kSecond;
using khronos::testing::require;
namespace model = khronos::model;

bool near(double a, double b, double tolerance) { return std::abs(a - b) <= tolerance; }

// README (5e): one alpha, the thresholds (1-alpha)/alpha and alpha/(1-alpha).
void testDecisionCriterion() {
  require(near(model::closeOdds(), 99.0, 1e-9) && near(model::confirmOdds(), 1.0 / 99.0, 1e-12),
          "alpha = 0.01 gives the thresholds 99 and 1/99 (the Wald ln 99)");
  require(model::decide(99.0) == model::Commitment::kCommitH, "odds 99 commit H");
  require(model::decide(98.9) == model::Commitment::kDefer, "odds just below 99 defer");
  require(model::decide(1.0 / 99.0) == model::Commitment::kCommitNotH, "odds 1/99 commit not H");
  require(model::decide(0.02) == model::Commitment::kDefer, "odds above 1/99 defer");
  require(model::decideLog(std::log(99.0)) == model::Commitment::kCommitH,
          "the log-odds decision agrees at the threshold");
  std::cout << "PASS decision criterion\n";
}

// README (5), (5a), (5c): Jeffreys cold start, Lomax survival, gap probability.
void testPersistencePrior() {
  model::PersistencePrior prior;
  require(!prior.hazard(1, 0).valid(), "no exposure: the Jeffreys predictive is not normalisable");
  require(prior.changeProbability(1, 0, 2.0) == 0.0, "no exposure: the round carries no prior change");
  require(near(prior.gapChangeProbability(1, 0), 0.5, 1e-3),
          "no judged gap: the Jeffreys Beta(1/2,1/2) gives q^g = 1/2");

  prior.addExposure(1, 0, 100.0);
  const auto h = prior.hazard(1, 0);
  require(h.valid() && near(h.shape, 0.5, 0.02) && near(h.rate, 100.0, 2.0),
          "after 100 s in place and no event the hazard is Gamma(1/2, 100)");
  const double q = prior.changeProbability(1, 0, 2.0);
  require(near(q, 1.0 - std::sqrt(100.0 / 102.0), 1e-3), "q = 1 - (B/(B+dt))^A, the Lomax form (5c)");
  // A placement that has been seen in place (uncredited exposure) is normal at once.
  model::PersistencePrior fresh;
  require(fresh.changeProbability(2, 0, 2.0, 50.0) > 0.0,
          "time observed in place makes the Jeffreys posterior normal");

  // Outcomes of session gaps: a class that changed is more likely to change again.
  model::PersistencePrior gaps;
  for (int i = 0; i < 4; ++i) {
    gaps.addGapOutcome(10 + i, /*cls=*/1, true);
    gaps.addGapOutcome(20 + i, /*cls=*/2, false);
  }
  gaps.addGapOutcome(30, 1, false);
  gaps.addGapOutcome(31, 2, true);
  require(gaps.gapChangeProbability(10, 1) > gaps.gapChangeProbability(20, 2),
          "the hierarchical estimate separates a class that changes from one that does not");
  const auto restored = model::PersistencePrior::fromJson(gaps.toJson());
  require(near(restored.gapChangeProbability(10, 1), gaps.gapChangeProbability(10, 1), 1e-9),
          "the statistics survive serialisation");
  std::cout << "PASS persistence prior\n";
}

// README (5r), (5t): the forward recursion in odds form and the change-time interval.
void testChangeFilter() {
  model::ChangeFilter fresh;
  const double q = 0.01;
  require(near(fresh.update(0, 10, q, 0.0), q / (1.0 - q), 1e-12), "Lambda_1 = q / (1-q) for LR = 1");
  const double second = fresh.update(10, 20, q, 0.0);
  require(near(second, (q / (1.0 - q) + q) / (1.0 - q), 1e-12), "Lambda_2 = (Lambda_1 + q)/(1 - q)");
  require(fresh.commitment() == model::Commitment::kDefer, "a small odds is neither committed");

  model::ChangeFilter inherited(1.0, 0, 10);
  require(near(inherited.odds(), 1.0, 1e-12), "an inherited placement starts at q^g/(1-q^g) = 1");
  inherited.update(10, 20, q, std::log(0.005));
  require(inherited.commitment() == model::Commitment::kCommitNotH,
          "strong support brings the odds under alpha/(1-alpha)");

  model::ChangeFilter closing;
  closing.update(0, 10, q, std::log(1.0e4));
  require(closing.commitment() == model::Commitment::kCommitH, "strong contradiction closes");

  // Two rounds, the second carrying all the contradiction: both cells hold the change.
  model::ChangeFilter interval;
  interval.update(0, 10, q, 0.0);
  interval.update(10, 20, q, std::log(1000.0));
  const auto span = interval.changeInterval();
  require(span.left == 0 && span.right == 20, "the change interval covers the cells that share the evidence");
  // A contradiction concentrated in the last round with nothing before it.
  model::ChangeFilter last_only;
  last_only.update(0, 10, 0.0, 0.0);
  last_only.update(10, 20, q, std::log(1000.0));
  const auto tight = last_only.changeInterval();
  require(tight.left == 10 && tight.right == 20, "a round without hazard cannot hold the change time");
  last_only.anchorAtConfirmation();
  require(last_only.changeInterval().left == 0 && last_only.changeInterval().right == 0,
          "a confirmation forgets the earlier rounds");
  std::cout << "PASS change filter\n";
}

// README (7), (7h), (7s): the round likelihood ratio and the element stability.
void testRoundModel() {
  model::RoundModel neutral;
  require(neutral.logLikelihoodRatio(1, {10, 0, 0, 0}) == 0.0, "no in-place statistic: LR = 1");

  model::PersistencePrior prior;
  model::RoundModel rounds;
  khronos::testing::trainedStatistics(prior, rounds, 100, 3);
  const auto parameters = rounds.parameters();
  require(parameters.see_through_available && parameters.psi_available,
          "the trained rounds identify the round and element parameters");
  require(rounds.logLikelihoodRatio(1, {12, 0, 0, 0}) < 0.0,
          "a round of hits supports the placement in place (ended : in place < 1)");
  require(rounds.logLikelihoodRatio(1, {12, 12, 0, 0}) > 10.0,
          "a round that is entirely see-through strongly supports a change");
  // The ended distribution learned from rounds after a committed end sharpens the contrast.
  model::RoundModel learned = rounds;
  for (int i = 0; i < 4; ++i) learned.addEndedRound({20, 18, 0, 0});
  require(learned.logLikelihoodRatio(1, {20, 18, 0, 0}) > rounds.logLikelihoodRatio(1, {20, 18, 0, 0}),
          "a learned ended distribution makes a mostly see-through round stronger evidence");

  const auto jeffreys = model::RoundModel().elementPrior(1);
  require(!model::RoundModel::elementStable(jeffreys, 0.0, 0.0, 0.0),
          "an element with no history is not stable");
  require(model::RoundModel::elementStable(jeffreys, 20.0, 0.0, 0.0),
          "an element built from many hits is stable, (7s)");
  require(!model::RoundModel::elementStable(jeffreys, 20.0, 0.0, 30.0),
          "an element that is repeatedly seen through is not");
  const auto restored = model::RoundModel::fromJson(rounds.toJson());
  require(near(restored.logLikelihoodRatio(1, {12, 6, 0, 0}), rounds.logLikelihoodRatio(1, {12, 6, 0, 0}), 1e-9),
          "the round statistics survive serialisation");
  std::cout << "PASS round model\n";
}

// README (6s), (6e): sigma_eff and the Bayes boundaries of hit against outlier.
void testRangeModel() {
  auto psi = khronos::testing::fixedRangeModel(0.02, 0.01);
  require(psi.valid(), "a model with outlier weights and scales is usable");
  const double rho = 2.0, range = 5.0;
  const double sigma = psi.sigmaEff(rho, 0.0, 0.0, 0.0, false);
  require(near(sigma, 0.02, 1e-12), "sigma_eff = sigma_s with no registration, broadening or alignment");
  const auto bounds = psi.bounds(rho, sigma, range);
  const double root = std::sqrt(2.0 * 3.14159265358979323846);
  const double expected_plus = sigma * std::sqrt(2.0 * std::log(0.98 * (range - rho) / (0.01 * sigma * root)));
  const double expected_minus = sigma * std::sqrt(2.0 * std::log(0.98 * rho / (0.01 * sigma * root)));
  require(near(bounds.plus, expected_plus, 1e-12) && near(bounds.minus, expected_minus, 1e-12),
          "the boundary is the point where hit and outlier densities are equal (6e)");
  using model::RangeClass;
  const auto classify = [&](double reading) {
    return model::classifyRange(psi, reading, rho, sigma, 0.1, range, false);
  };
  require(classify(rho) == RangeClass::kHit, "a reading at the element is a hit");
  require(classify(rho + bounds.plus + 1e-3) == RangeClass::kThrough, "beyond delta+ is see-through");
  require(classify(rho - bounds.minus - 1e-3) == RangeClass::kOccluded, "before -delta- is occluded");
  require(classify(0.0) == RangeClass::kInvalid && classify(range + 1.0) == RangeClass::kInvalid,
          "no reading or one outside the device range is invalid");

  // The registration variogram (9v), the alignment residual and the scale bias of (6s).
  psi.reg_dt = {0.1, 1.0};
  psi.reg_variance = {0.0, 4.0e-4};
  psi.sigma_x = 0.03;
  psi.delta_s = 0.01;
  require(near(psi.sigmaEff(rho, 0.0, 1.0, 0.0, false), std::sqrt(0.02 * 0.02 + 4.0e-4), 1e-12),
          "the registration variance grows with the time difference");
  require(near(psi.sigmaEff(rho, 0.0, 0.1, 0.0, true), std::sqrt(0.02 * 0.02 + 0.03 * 0.03), 1e-12),
          "across sessions the alignment residual is added");
  require(near(psi.bias(rho, true), 0.02, 1e-12) && psi.bias(rho, false) == 0.0,
          "b = Delta s * rho across sessions only");
  const double cross_sigma = psi.sigmaEff(rho, 0.0, 0.1, 0.0, true);
  require(model::classifyRange(psi, rho + 0.02, rho, cross_sigma, 0.1, range, true) == RangeClass::kHit,
          "the scale bias is removed before the reading is classified");
  const auto restored = model::RangeModel::fromJson(psi.toJson());
  require(near(restored.sigmaEff(rho, 0.0, 1.0, 0.0, true), psi.sigmaEff(rho, 0.0, 1.0, 0.0, true), 1e-12),
          "the model survives serialisation");
  std::cout << "PASS range model\n";
}

khronos::KhronosObjectAttributes::Ptr makeObject(TimeStamp first, TimeStamp last, size_t id,
                                                 bool open_end = false) {
  auto attrs = std::make_unique<khronos::KhronosObjectAttributes>();
  attrs->mesh = spark_dsg::Mesh(false, true, false, true);
  const khronos::Points points = {khronos::Point(0.f, 0.f, 0.f), khronos::Point(1.f, 0.f, 0.f),
                                  khronos::Point(0.f, 1.f, 0.f), khronos::Point(1.f, 1.f, 1.f)};
  for (const auto& point : points) {
    const size_t index = attrs->mesh.numVertices();
    attrs->mesh.resizeVertices(index + 1);
    attrs->mesh.setPos(index, point);
    attrs->mesh.setFirstSeenTimestamp(index, first);
    attrs->mesh.setTimestamp(index, last);
  }
  attrs->bounding_box = spark_dsg::BoundingBox(points);
  attrs->position = attrs->bounding_box.world_P_center.cast<double>();
  attrs->first_observed_ns = {first};
  attrs->last_observed_ns = {open_end ? std::numeric_limits<TimeStamp>::max() : last};
  khronos::setObservationBounds(*attrs, first, last);
  attrs->details["instance_id"] = {id};
  return attrs;
}

// README (5r), (5e), principle 12: a placement born in this session and an inherited one.
void testRegistryCommitments() {
  using khronos::PersistentObjectState;
  // --- a placement born in this session: committed in place, then seen through ----------------
  {
    auto dsg = std::make_shared<khronos::DynamicSceneGraph>();
    dsg->emplaceNode(spark_dsg::DsgLayers::OBJECTS, spark_dsg::NodeSymbol('O', 1),
                     makeObject(1 * kSecond, 5 * kSecond, 7));
    PersistentObjectState registry;
    khronos::testing::trainRegistry(registry, 20.0);
    registry.ingestObjects(*dsg);
    const auto first = khronos::testing::confirm(registry, 7, 8 * kSecond);
    require(first.confirmed && !first.closed, "a round of hits commits a born placement in place");
    require(registry.currentFragment(7)->odds <= model::confirmOdds(),
            "the odds of a confirmed placement stay under alpha/(1-alpha)");
    const auto statistics = registry.persistencePrior().objects().find(7);
    require(statistics != registry.persistencePrior().objects().end() &&
                statistics->second.exposure > 0.0 && statistics->second.events == 0.0,
            "the committed time in place is exposure, not an event");
    const auto second = khronos::testing::contradict(registry, 7, 12 * kSecond);
    require(second.closed, "a round entirely seen through commits the change");
    const auto history = registry.historyFragments(7);
    require(history.size() == 1 && history[0].death_time.has_value() &&
                history[0].change_left <= *history[0].death_time &&
                *history[0].death_time >= 8 * kSecond && *history[0].death_time <= 12 * kSecond,
            "the change time is the interval (5t) between the last confirmation and the commitment");
    require(registry.persistencePrior().objects().at(7).events == 1.0,
            "the committed change is an event of the object");
    require(registry.successionFloors().count(7) == 1,
            "the right end of the interval is published for the frame attribution (8)");
  }
  // --- an inherited placement: the gap outcome is judged once ---------------------------------
  for (const bool changed : {false, true}) {
    auto dsg = std::make_shared<khronos::DynamicSceneGraph>();
    dsg->emplaceNode(spark_dsg::DsgLayers::OBJECTS, spark_dsg::NodeSymbol('O', 2),
                     makeObject(1 * kSecond, 5 * kSecond, 8, true));
    PersistentObjectState registry;
    khronos::testing::trainRegistry(registry, 20.0);
    registry.initializeFromObjects(*dsg, 10 * kSecond);
    require(registry.currentFragment(8).has_value() && registry.currentFragment(8)->inherited,
            "the restored placement is current and inherited");
    const TimeStamp session_start = 20 * kSecond;
    if (changed) {
      const auto round = registry.resolveRound(
          8, khronos::testing::craftRound(registry, 8, 24 * kSecond, 0, 12, session_start), {},
          24 * kSecond);
      require(round.closed, "an inherited placement seen through closes");
    } else {
      const auto one = registry.resolveRound(
          8, khronos::testing::craftRound(registry, 8, 22 * kSecond, 12, 0, session_start), {},
          22 * kSecond);
      require(!one.closed, "one round of hits does not close an inherited placement");
      const auto two = registry.resolveRound(
          8, khronos::testing::craftRound(registry, 8, 24 * kSecond, 12, 0, session_start), {},
          24 * kSecond);
      require(one.confirmed || two.confirmed, "rounds of hits bring the inherited odds under alpha");
    }
    const auto& stats = registry.persistencePrior().objects().at(8);
    require(stats.gap_judged == 1.0 && stats.gap_changed == (changed ? 1.0 : 0.0),
            "the outcome of the gap is judged once and recorded");
  }
  std::cout << "PASS registry commitments\n";
}

}  // namespace

int main() {
  testDecisionCriterion();
  testPersistencePrior();
  testChangeFilter();
  testRoundModel();
  testRangeModel();
  testRegistryCommitments();
  std::cout << "ALL UNIFIED MODEL TESTS PASSED\n";
  return 0;
}
