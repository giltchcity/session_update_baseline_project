/** -----------------------------------------------------------------------------
 * Unit checks of the unified model (README eq. (1)): the persistence prior Pi (principle 2), the
 * CUSUM with its identity channel and the decision criterion D (principle 3), the look likelihood of
 * principle 6, the first-return model of principle 4, and the registry that joins them.
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
#include "session_core/model/sensor_calibrator.h"
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
  require(model::representationHolds(0.0) && !model::representationHolds(-1e-9),
          "a representation output (no deferral) holds at posterior odds 1 (the maximum a posteriori)");
  std::cout << "PASS decision criterion\n";
}

// README (5), (5a): the gap probabilities q_{c,g} by empirical Bayes, Jeffreys cold start.
void testPersistencePrior() {
  using model::Gap;
  model::PersistencePrior prior;
  require(near(prior.changeProbability(1, 0, Gap::kSession), 0.5, 1e-3),
          "no decided gap: the Jeffreys Beta(1/2,1/2) gives q = 1/2");

  // Outcomes of session gaps: a class that changed is more likely to change again.
  model::PersistencePrior gaps;
  for (int i = 0; i < 4; ++i) {
    gaps.addOutcome(10 + i, /*cls=*/1, Gap::kSession, true);
    gaps.addOutcome(20 + i, /*cls=*/2, Gap::kSession, false);
  }
  gaps.addOutcome(30, 1, Gap::kSession, false);
  gaps.addOutcome(31, 2, Gap::kSession, true);
  require(gaps.changeProbability(10, 1, Gap::kSession) > gaps.changeProbability(20, 2, Gap::kSession),
          "the hierarchical estimate separates a class that changes from one that does not");
  require(near(gaps.changeProbability(40, 3, Gap::kContinuous), 0.5, 1e-3),
          "a gap type without decided gaps stays at the Jeffreys start");
  const auto restored = model::PersistencePrior::fromJson(gaps.toJson());
  require(near(restored.changeProbability(10, 1, Gap::kSession),
               gaps.changeProbability(10, 1, Gap::kSession), 1e-9),
          "the statistics survive serialisation");
  std::cout << "PASS persistence prior\n";
}

// README (5r): the CUSUM C = max(0, C + w l^g + l^id) with the Wald boundary ln((1-alpha)/alpha).
void testCusum() {
  model::Cusum cusum;
  auto step = cusum.update(1.0, -2.0, 0.0);
  require(step.before == 0.0 && step.after == 0.0, "evidence for the placement keeps C at 0");
  step = cusum.update(0.5, 4.0, 0.0);
  require(near(step.after, 2.0, 1e-12), "the geometric log ratio enters weighted by w");
  require(!cusum.exceeded(), "2 nats are below ln 99");
  cusum.update(1.0, 3.0, 0.0);
  require(cusum.exceeded() && near(cusum.value(), 5.0, 1e-12), "5 nats exceed ln 99 = 4.595");
  // The identity channel: a look that directly saw the placement in place presses the statistic down
  // by |ln eps|; a weight of 0 (nothing newly judged) leaves only the identity term.
  step = cusum.update(0.0, 7.0, std::log(0.05));
  require(near(step.after, 5.0 + std::log(0.05), 1e-12) && !cusum.exceeded(),
          "the identity channel presses the accumulated end evidence down");
  cusum.update(1.0, -10.0, 0.0);
  require(cusum.value() == 0.0, "C returns to 0: a new accumulation");
  const auto restored = model::Cusum::fromJson(cusum.toJson());
  require(restored.value() == cusum.value(), "the statistic survives serialisation");
  std::cout << "PASS cusum\n";
}

// README (7), (7h): the look likelihood ratio, the in-place distribution and k_min.
void testRoundModel() {
  using model::RoundModel;
  RoundModel cold;
  // Cold start: the share the measurement model predicts and concentration 2.
  const auto start = cold.inPlace(1, 0.05);
  require(near(start.m, 0.05, 1e-12) && near(start.c, 2.0, 1e-9), "cold start: (m_0, 2)");
  // The numbers of the README: n = 14 all see-through crosses ln 99, n = 13 does not.
  require(near(RoundModel::logLikelihoodRatio(start, 14.0, 14.0), 4.63, 0.01) &&
              RoundModel::logLikelihoodRatio(start, 13.0, 13.0) < std::log(99.0),
          "n = 14 is the first decisive all-see-through look");
  require(RoundModel::decisiveSamples(start) == 14, "the decisive sample count is 14 at cold start");
  require(near(RoundModel::logLikelihoodRatio(start, 100.0, 100.0), 6.4, 0.05) &&
              near(RoundModel::logLikelihoodRatio(start, 100.0, 50.0), 2.2, 0.05),
          "n = 100: +6.4 nats all see-through, +2.2 half");
  require(RoundModel::logLikelihoodRatio(start, 100.0, 0.0) > -1.0,
          "one-sided: fewer see-throughs than usual count as the usual share");
  require(RoundModel::logLikelihoodRatio(start, 0.0, 0.0) == 0.0, "no judged sample: not a look");

  // Learning (7h): the looks of three objects give the total mean, the between-object and the
  // look-to-look concentrations; the object's own looks move its share and its predictive
  // concentration away from the prior.
  RoundModel learned;
  khronos::model::PersistencePrior prior;
  khronos::testing::trainedStatistics(prior, learned, 100, 3);
  require(learned.numInPlaceLooks(100) == 4, "the four looks of the object are its in-place looks");
  const auto object = learned.inPlace(100, 0.5);
  require(object.m < 0.1 && object.c > start.c && object.looks == 4,
          "the in-place distribution follows the learned shares and narrows with the object's own looks");
  require(object.mu < 0.1, "the total mean is learned from the looks, not the cold centre of the call");
  const auto fresh = learned.inPlace(777, 0.05);
  require(fresh.looks == 0 && near(fresh.c, 1.0 / (1.0 / (fresh.within + 1.0) + 1.0 / (fresh.kappa + 1.0)) - 1.0, 1e-9),
          "an object without looks has the population-level predictive concentration");
  require(RoundModel::logLikelihoodRatio(object, 20.0, 0.0) < 0.0,
          "a look of hits supports the placement in place (ended : in place < 1)");
  require(RoundModel::logLikelihoodRatio(object, 20.0, 20.0) > std::log(99.0),
          "a look entirely see-through is decisive");

  // k_min: the least k with eps^k <= alpha, eps the foreign-label share (Jeffreys smoothed).
  RoundModel labels;
  require(labels.minHits() == 7, "no labels: eps = 1/2 and k_min = 7");
  labels.addLabels(0.0, 1000.0);
  require(labels.minHits() == 1, "no foreign label in 1000: k_min = 1");
  RoundModel noisy;
  noisy.addLabels(23.0, 1000.0);
  require(noisy.minHits() == 2, "eps = 0.023 gives k_min = 2 (0.023^2 <= 0.01)");
  const auto restored = RoundModel::fromJson(learned.toJson());
  require(near(restored.inPlace(100, 0.5).m, object.m, 1e-12), "the statistics survive serialisation");
  std::cout << "PASS round model\n";
}

// README (6s), (6e): sigma_eff and the Bayes boundaries of hit against outlier.
void testRangeModel() {
  auto psi = khronos::testing::fixedRangeModel(0.02, 0.01);
  require(near(psi.w_plus, 0.01, 1e-12) && psi.w_estimated, "the fixture is a model with estimated outlier weights");
  const double rho = 2.0, range = 5.0;
  const double sigma = psi.sigmaEff(rho, 0.0, 0.0, false);
  require(near(sigma, 0.02, 1e-12), "sigma_eff = sigma_table with no broadening or alignment");
  const auto bounds = psi.bounds(rho, sigma, range);
  const double root = std::sqrt(2.0 * 3.14159265358979323846);
  const double expected_plus = sigma * std::sqrt(2.0 * std::log(0.98 * (range - rho) / (0.01 * sigma * root)));
  const double expected_minus = sigma * std::sqrt(2.0 * std::log(0.98 * rho / (0.01 * sigma * root)));
  require(near(bounds.plus, expected_plus, 1e-12) && near(bounds.minus, expected_minus, 1e-12),
          "the boundary is the point where hit and outlier densities are equal (6e)");
  using model::RangeClass;
  const auto classify = [&](double reading) {
    return model::classifyRange(psi, reading, rho, sigma, 0.1, range);
  };
  require(classify(rho) == RangeClass::kHit, "a reading at the element is a hit");
  require(classify(rho + bounds.plus + 1e-3) == RangeClass::kThrough, "beyond delta+ is see-through");
  require(classify(rho - bounds.minus - 1e-3) == RangeClass::kOccluded, "before -delta- is occluded");
  require(classify(0.0) == RangeClass::kInvalid && classify(range + 1.0) == RangeClass::kInvalid,
          "no reading or one outside the device range is invalid");
  require(near(psi.predictedSeeThrough(rho, sigma, range), 0.01 + 0.98 * 0.5 * std::erfc(bounds.plus / (sigma * std::sqrt(2.0))), 1e-12),
          "m_0 = w_+ + w_H (1 - Phi(delta_+ / sigma_eff))");

  // sigma_x counts only across sessions and only once it is estimated; b = zeta_now rho - zeta_e rho_e.
  psi.sigma_x = 0.03;
  require(near(psi.sigmaEff(rho, 0.0, 0.0, true), 0.02, 1e-12), "an unestimated sigma_x is not replaced by anything");
  psi.sigma_x_known = true;
  require(near(psi.sigmaEff(rho, 0.0, 0.0, true), std::sqrt(0.02 * 0.02 + 0.03 * 0.03), 1e-12),
          "across sessions the alignment residual is added");
  require(near(psi.sigmaEff(rho, 0.0, 0.0, false), 0.02, 1e-12), "within a session it is not");
  psi.zeta = 0.04;
  const double bias = psi.bias(rho, 2.2, 0.03);
  require(near(bias, 0.04 * 2.0 - 0.03 * 2.2, 1e-12), "b = zeta_now rho - zeta_e rho_e");
  const double cross_sigma = psi.sigmaEff(rho, 0.0, 0.0, true);
  require(model::classifyRange(psi, rho + bias, rho, cross_sigma, 0.1, range, bias) == RangeClass::kHit,
          "the scale displacement is removed before the reading is classified");
  const auto restored = model::RangeModel::fromJson(psi.toJson());
  require(near(restored.sigmaEff(rho, 0.0, 0.0, true), psi.sigmaEff(rho, 0.0, 0.0, true), 1e-12) &&
              restored.sigma_x_known,
          "the model survives serialisation");
  std::cout << "PASS range model\n";
}

// README principle 8: every quantity is the posterior of a prior and the data and is defined before
// the first frame: sigma_table at the quantisation scale (or the sensor curve), w_pm the mean of the
// Jeffreys prior Dir(1/2, 1/2, 1/2), per-bin weights a continuous posterior with no sample threshold.
void testColdStartPosterior() {
  using model::SensorCalibrator;
  const model::RangeModel none;  // a default model is the cold start
  require(near(none.w_plus, 1.0 / 3.0, 1e-12) && near(none.w_minus, 1.0 / 3.0, 1e-12) && !none.w_estimated,
          "a default model has the mean of the Jeffreys prior as outlier weights");
  SensorCalibrator calibrator;
  const auto cold = calibrator.estimate(none, {}, 5.0, 0.04);
  require(near(cold.w_plus, 1.0 / 3.0, 1e-12) && !cold.w_estimated &&
              near(cold.sigma_table.front(), 1e-3 / std::sqrt(12.0), 1e-12),
          "before any residual the posterior is the cold-start prior: w_pm 1/3, sigma_table u / sqrt(12)");
  const auto curved = calibrator.estimate(none, std::vector<double>(16, 0.02), 5.0, 0.04);
  require(near(curved.sigma_table[3], 0.02, 1e-12), "the sensor curve is the prior centre of sigma_table");
  require(near(curved.w_plus, 1.0 / 3.0, 1e-12), "the sensor curve gives nothing to w_pm");
  // Residuals of one range bin: its posterior moves, the other bins take the pooled posterior.
  for (int i = 0; i < 2000; ++i) calibrator.addResidual(1.2, 0.001 * static_cast<double>((i % 7) - 3) / 3.0);
  const auto data = calibrator.estimate(none, {}, 5.0, 0.04);
  require(data.w_estimated && data.w_plus < 0.01 && data.w_minus < 0.01,
          "residuals that are all hits make the outlier weights small");
  require(data.w_plus_bin.size() == 16 && near(data.w_plus_bin[10], data.w_plus, 1e-12),
          "a bin without residuals takes the pooled posterior");
  require(data.sigma_table[2] < 0.002, "the bin with residuals follows them");
  std::cout << "PASS cold start posterior\n";
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
  // --- a placement born in this session: directly seen in place, then seen through ------------
  {
    auto dsg = std::make_shared<khronos::DynamicSceneGraph>();
    dsg->emplaceNode(spark_dsg::DsgLayers::OBJECTS, spark_dsg::NodeSymbol('O', 1),
                     makeObject(1 * kSecond, 5 * kSecond, 7));
    PersistentObjectState registry;
    khronos::testing::trainRegistry(registry);
    registry.ingestObjects(*dsg);
    const auto first = khronos::testing::confirm(registry, 7, 8 * kSecond);
    require(first.confirmed && !first.closed, "a look of hits directly sees a born placement in place");
    require(registry.currentFragment(7)->cusum == 0.0, "the statistic of a confirmed placement stays at 0");
    require(registry.roundModel().numInPlaceLooks(7) == 1,
            "the look that directly saw it in place is learned (after it was judged)");
    const auto second = khronos::testing::contradict(registry, 7, 12 * kSecond);
    require(second.closed, "a look entirely seen through, with the rays against it, commits the end");
    const auto history = registry.historyFragments(7);
    require(history.size() == 1 && history[0].death_time.has_value() &&
                history[0].change_left <= *history[0].death_time &&
                *history[0].death_time >= 8 * kSecond && *history[0].death_time <= 12 * kSecond,
            "the end time is the interval (5t) between the last support and the commitment");
    require(registry.successionFloors().count(7) == 1,
            "the right end of the interval is published for the frame attribution (8)");
  }
  // --- the identity channel: a look that directly saw the placement in place presses C down -----
  {
    auto dsg = std::make_shared<khronos::DynamicSceneGraph>();
    dsg->emplaceNode(spark_dsg::DsgLayers::OBJECTS, spark_dsg::NodeSymbol('O', 3),
                     makeObject(1 * kSecond, 5 * kSecond, 9));
    PersistentObjectState registry;
    khronos::testing::trainRegistry(registry);
    registry.ingestObjects(*dsg);
    khronos::testing::confirm(registry, 9, 8 * kSecond);
    const auto part = registry.resolveRound(
        9, khronos::testing::craftRound(registry, 9, 12 * kSecond, 0, 4), {}, 12 * kSecond);
    require(part.look.judged && !part.look.directly_seen && part.look.ln_identity == 0.0,
            "a look that is only see-through is no support: the identity channel is silent");
    require(part.look.step.after > 0.0 && part.look.step.after == part.look.weight * part.look.ln_lr,
            "a see-through look accumulates the weighted log ratio");
    const auto seen = registry.resolveRound(
        9, khronos::testing::craftRound(registry, 9, 16 * kSecond, 15, 0), {}, 16 * kSecond);
    require(seen.look.directly_seen && seen.look.ln_identity < 0.0 && !seen.look.eps_prior,
            "a look of own-identity samples directly sees the placement in place: ln eps is negative");
    require(near(seen.look.step.after, std::max(0.0, seen.look.step.before + seen.look.weight * seen.look.ln_lr +
                                                  seen.look.ln_identity), 1e-12) &&
                seen.look.step.after < seen.look.step.before,
            "the support presses the accumulated end evidence down by the identity term");
    require(registry.currentFragment(9)->direct_since == 8 * kSecond,
            "t_L is the first time the placement was directly seen in place");
  }
  // --- an inherited placement: the gap outcome is decided once ---------------------------------
  for (const bool changed : {false, true}) {
    auto dsg = std::make_shared<khronos::DynamicSceneGraph>();
    dsg->emplaceNode(spark_dsg::DsgLayers::OBJECTS, spark_dsg::NodeSymbol('O', 2),
                     makeObject(1 * kSecond, 5 * kSecond, 8, true));
    PersistentObjectState registry;
    khronos::testing::trainRegistry(registry);
    registry.initializeFromObjects(*dsg, 10 * kSecond);
    require(registry.currentFragment(8).has_value() && registry.currentFragment(8)->inherited,
            "the restored placement is current and inherited");
    require(registry.currentFragment(8)->cusum == 0.0, "a restored placement starts this session at C = 0");
    const TimeStamp session_start = 20 * kSecond;
    if (changed) {
      const auto round = registry.resolveRound(
          8, khronos::testing::craftRound(registry, 8, 24 * kSecond, 0, 20, session_start), {},
          24 * kSecond);
      require(round.closed, "an inherited placement seen through closes");
    } else {
      const auto one = registry.resolveRound(
          8, khronos::testing::craftRound(registry, 8, 22 * kSecond, 20, 0, session_start), {},
          22 * kSecond);
      require(!one.closed && one.confirmed, "a look of hits directly sees an inherited placement in place");
      const auto two = registry.resolveRound(
          8, khronos::testing::craftRound(registry, 8, 24 * kSecond, 20, 0, session_start), {},
          24 * kSecond);
      require(two.confirmed && !two.closed, "a second look of hits keeps it");
    }
    const auto& stats = registry.persistencePrior().objects().at(8);
    const size_t session_gap = static_cast<size_t>(model::Gap::kSession);
    require(stats.judged[session_gap] == 1.0 && stats.changed[session_gap] == (changed ? 1.0 : 0.0),
            "the outcome of the session gap is decided once and recorded");
  }
  std::cout << "PASS registry commitments\n";
}

}  // namespace

int main() {
  testDecisionCriterion();
  testPersistencePrior();
  testCusum();
  testRoundModel();
  testRangeModel();
  testColdStartPosterior();
  testRegistryCommitments();
  std::cout << "ALL UNIFIED MODEL TESTS PASSED\n";
  return 0;
}
