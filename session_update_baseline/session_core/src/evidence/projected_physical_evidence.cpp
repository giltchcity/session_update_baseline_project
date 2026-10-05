#include "khronos/backend/change_detection/ray_verificator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace khronos {

ProjectedRelationProbabilities projectedRelationProbabilities(
    const ProjectedEndpointEvidence& p, const size_t id, const float tolerance) {
  ProjectedRelationProbabilities result;
  const auto& e = p.endpoint;
  if (e.type == EndpointClass::kUnavailable) {
    result.unavailable = 1.0;
    return result;
  }
  if (e.type == EndpointClass::kInvalid || !std::isfinite(e.measured_depth_m) ||
      !std::isfinite(p.query_range_m) || e.measured_depth_m <= 0 ||
      p.query_range_m <= 0) {
    result.invalid = 1.0;
    return result;
  }
  // PhysicalEvidenceStore stores lround(range * 1000). Integrate the unknown
  // phase of that millimetre cell; the matching band is not sensor variance.
  constexpr double quantum = 1.0 / 1000.0;
  const float delta = e.measured_depth_m - p.query_range_m;
  const bool same_identity = e.type == EndpointClass::kPhysical &&
      e.physical_id > 0 && static_cast<size_t>(e.physical_id) == id;
  const float near_boundary = same_identity
      ? -tolerance : -std::min(tolerance, static_cast<float>(quantum));
  const auto cdf = [&](const float boundary) {
    return std::max(0.0, std::min(1.0,
        (static_cast<double>(boundary) - delta) / quantum + 0.5));
  };
  result.occluded = cdf(near_boundary);
  const double through_upper = cdf(tolerance);
  result.free = 1.0 - through_upper;
  const double on_surface = through_upper - result.occluded;
  if (same_identity) {
    result.supported = on_surface;
  } else if (e.type == EndpointClass::kBackground) {
    result.background = on_surface;
  } else if (e.type == EndpointClass::kUnidentifiedObject) {
    result.unidentified = on_surface;
  } else if (e.type == EndpointClass::kPhysical) {
    result.other = on_surface;
  } else {
    result.invalid = on_surface;
  }
  return result;
}

namespace {

enum class Vote { Unavailable, Invalid, Occluded, Supported, Free,
                  Background, Other, Unidentified };

Vote classifyMeasurement(const ProjectedEndpointEvidence& p, size_t id, float tolerance) {
  const auto mass = projectedRelationProbabilities(p, id, tolerance);
  // Equal classification loss. Surface-first ties retain the closed matching
  // interval; a foreground return still cannot vote behind its occluder.
  Vote vote = Vote::Supported;
  double best = mass.supported;
  const auto consider = [&](const Vote candidate, const double probability) {
    if (probability > best) { vote = candidate; best = probability; }
  };
  consider(Vote::Background, mass.background);
  consider(Vote::Other, mass.other);
  consider(Vote::Unidentified, mass.unidentified);
  consider(Vote::Invalid, mass.invalid);
  consider(Vote::Unavailable, mass.unavailable);
  consider(Vote::Occluded, mass.occluded);
  consider(Vote::Free, mass.free);
  return vote;
}

// ---------------------------------------------------------------------------
// Observed absence (README eq. (7)).
//
// The samples of a state are the cells of its stored surface. Every stored frame finds a
// sample on the surface (with the object's own identity, another identity or none), seen
// through, or only in view. One reconciliation round is one look: the seen-through share f of
// the reliable samples it judged. Absent: f is uniform. Present: a Beta fitted to the object's
// own in-place looks, shrunk towards the population of objects. The sequential test is Page's
// maximum over candidate change times; crossing log 99 commits the absence, which the rays of
// the same round must confirm (C > S) before a state closes.
struct AbsenceSample {
  // Reliability record (README O10a, P22): outcomes in frames in which the object was
  // identified in place. A round's outcomes stay tentative and are committed only when the
  // round finds the object in place: frames seen after a move must not teach the record of
  // the state that has just ended.
  uint16_t identity_hits = 0, other_obs = 0;
  bool seen_through_while_identified = false;
  uint16_t tentative_hits = 0, tentative_other = 0;
  bool tentative_veto = false;
  // Latest verdicts. Occluded or unmeasured looks give no verdict.
  TimeStamp last_on_surface = 0;
  TimeStamp last_seen_through = 0;
  TimeStamp last_identity = 0;
  // Latest scored look (index into ObjectAbsenceState::page) that judged this sample, -1 for
  // none: looking again at the same surface is not new evidence about a change after it.
  int32_t last_look = -1;
  // DIAG (replay of per-sample models; no decision reads it): the latest measured outcome at
  // any incidence, +1 within the surface band, 2 beyond it, with |cos| of the view and range.
  TimeStamp diag_stamp = 0;
  int8_t diag_verdict = 0;
  float diag_cos = 0.f, diag_range = 0.f, diag_delta = 0.f;
};
using AbsenceCell = std::tuple<int64_t, int64_t, int64_t>;
struct ObjectAbsenceState {
  TimeStamp processed = 0;
  // False until the object is identified in a stored frame. An inherited state whose object
  // has left is never identified again, so its samples have no reliability record; then
  // every sample votes.
  bool ever_identified = false;
  // The state's surface was built from identity observations of an earlier session.
  bool inherited = false;
  // Seen-through shares of the looks in which the object stood in place: its own
  // missed-detection behaviour under this sensor (count, sum and the looks themselves).
  double history_n = 0, history_sum = 0;
  std::vector<double> looks;
  // Page's test since the last commitment: page[s] accumulates the log ratio absent : present
  // of the looks from scored look s on (the candidate change time); cusum = max(0, max page).
  std::vector<double> page;
  double cusum = 0;
  // The window still starts at this state's first look of the session, so page[0] is the
  // candidate "changed before this session": the cross-session relation its prior q is about.
  bool first_window = true;
  // The latest look as a finite-count ratio, read by the empty-interval test (M1e) and the
  // hand-over U term (README 1.1).
  const RayVerificator* likelihood_owner = nullptr;
  TimeStamp likelihood_stamp = 0;
  PhysicalAbsenceLookLikelihood likelihood;
  std::map<AbsenceCell, AbsenceSample> samples;
};
std::mutex absence_mutex;
// Key: (verificator lifetime token, identity, state key = the tested state's id, or the slot,
// marked by the top bit, for callers without a state id).
std::map<std::tuple<uint64_t, size_t, uint64_t>, std::shared_ptr<ObjectAbsenceState>> absence_states;
uint64_t absenceStateKey(const uint64_t state_id, const int state_slot) {
  return state_id != 0 ? state_id : (static_cast<uint64_t>(state_slot) | (uint64_t(1) << 63));
}
// Population of objects, the prior of an object without its own history: the means of the
// first kRobustLooks in-place looks of every object, and their scatter as carried over from the
// previous session.
double pooled_n = 0, pooled_sum = 0;
std::vector<double> pooled_geo_dev;
double loaded_geo_var = -1;

// Smallest sample whose median has a positive breakdown point, floor((n - 1) / 2) / n > 0: n = 3,
// the least sample that tolerates one bad look (Hampel's breakdown point; Huber & Ronchetti,
// Robust Statistics). The robust scatter of an object's looks needs it, an object joins the
// population with the mean of that many looks, and the population enters an object's model with
// the same weight (as many pseudo-looks). The empirical-Bayes weight (within / between robust
// variance) was tried once: in the L2_FINAL_20261005 replay it loses the GT-correct closure of the
// real C inst 2 start site (480.2 s).
constexpr size_t minimalRobustSample() {
  size_t n = 1;
  while ((n - 1) / 2 == 0) ++n;
  return n;
}
constexpr size_t kRobustLooks = minimalRobustSample();

// Numerical floor of a share variance (standard deviation 1 %). It acts (L2_FINAL replay: 210 real,
// 249 synthetic robust scatters below it); halving it changes no decision. The floor derived from
// the sample budget, the binomial variance of a 1500-sample share at its Jeffreys estimate
// (2.2e-7), was tried once and closes the GT-static synthetic basin_0001 at 205.6 s; kept.
constexpr double kShareVarianceFloor = 1e-4;

// Robust scale: 1.4826 * median absolute deviation. A few looks at a partly
// re-occupied old site must not widen the scatter of an otherwise exact sensor.
// With centre < 0 the median of the values is used as the centre.
double robustVariance(std::vector<double> values, double centre) {
  if (values.empty()) return kShareVarianceFloor;
  if (centre < 0) {
    std::vector<double> c(values);
    std::nth_element(c.begin(), c.begin() + c.size() / 2, c.end());
    centre = c[c.size() / 2];
  }
  for (auto& v : values) v = std::abs(v - centre);
  std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
  const double mad = values[values.size() / 2];
  return std::max(kShareVarianceFloor, (1.4826 * mad) * (1.4826 * mad));
}

// README M1 reliability (label part): a cell never seen through while the object stood in place
// is a surface of this object (R) or of something else carrying its label at times (P). In
// identified frames each class returns own label / other label with rates theta_R / theta_P;
// pi = P(R). Maximum likelihood by EM over the committed counts of every unvetoed cell of the
// process, once per reconciliation round; such a cell votes iff P(R | counts) >= 1/2. The
// one-look veto stays: it selects the never-seen-through subset whose in-place share the
// sequential test models (P22 trial 1). Without counts the classes are not identified and
// every observed cell votes.
struct CellModel {
  bool identified = false;
  double log_pi = std::log(0.5), log_1mpi = std::log(0.5);
  std::array<double, 2> log_r{{0.0, 0.0}}, log_p{{0.0, 0.0}};
  TimeStamp stamp = 0;
};
CellModel cell_model;

double cellLogOdds(const CellModel& m, const double own, const double other) {
  if (!m.identified) return 0.0;
  return m.log_pi - m.log_1mpi + own * (m.log_r[0] - m.log_p[0]) +
         other * (m.log_r[1] - m.log_p[1]);
}

// Caller holds absence_mutex.
void estimateCellModel(const TimeStamp stamp) {
  if (cell_model.stamp == stamp) return;
  cell_model.stamp = stamp;
  // Cells with equal counts have equal posteriors: the EM runs over the distinct count
  // pairs weighted by their multiplicity (same fixed point, a few hundred terms per pass).
  std::map<std::pair<uint32_t, uint32_t>, double> histogram;
  for (const auto& [key, st] : absence_states) {
    (void)key;
    for (const auto& [cell, sm] : st->samples) {
      (void)cell;
      if (sm.seen_through_while_identified) continue;
      if (sm.identity_hits + sm.other_obs > 0) histogram[{sm.identity_hits, sm.other_obs}] += 1.0;
    }
  }
  if (histogram.empty()) return;
  std::vector<std::array<double, 3>> cells;  // own, other, multiplicity
  double total = 0.0;
  for (const auto& [counts, m] : histogram) {
    cells.push_back({static_cast<double>(counts.first), static_cast<double>(counts.second), m});
    total += m;
  }
  // Initial split from the data: cells whose own-label share is at least one half.
  std::array<double, 2> sr{{0, 0}}, sp{{0, 0}};
  double nr = 0;
  for (const auto& c : cells) {
    const bool r = c[0] >= 0.5 * (c[0] + c[1]);
    for (int j = 0; j < 2; ++j) (r ? sr : sp)[j] += c[2] * c[j];
    nr += r ? c[2] : 0.0;
  }
  if (nr == 0 || nr == total) return;  // one class only: not identified
  double pi = nr / total;
  std::array<double, 2> tr, tp;
  const auto norm = [](const std::array<double, 2>& a, std::array<double, 2>& out) {
    const double tot = a[0] + a[1];
    // Jeffreys pseudo-count 1/2 per outcome keeps every rate inside (0, 1).
    for (int j = 0; j < 2; ++j) out[j] = (a[j] + 0.5) / (tot + 1.0);
  };
  norm(sr, tr); norm(sp, tp);
  for (int it = 0; it < 1000; ++it) {
    std::array<double, 2> ar{{0, 0}}, ap{{0, 0}};
    double w_sum = 0;
    const double lr0 = std::log(tr[0]), lr1 = std::log(tr[1]);
    const double lp0 = std::log(tp[0]), lp1 = std::log(tp[1]);
    for (const auto& c : cells) {
      const double a = std::log(pi) + c[0] * lr0 + c[1] * lr1;
      const double b = std::log1p(-pi) + c[0] * lp0 + c[1] * lp1;
      const double w = c[2] / (1.0 + std::exp(b - a));
      w_sum += w;
      for (int j = 0; j < 2; ++j) { ar[j] += w * c[j]; ap[j] += (c[2] - w) * c[j]; }
    }
    const double new_pi = std::min(std::max(w_sum / total, 0.5 / (total + 1)),
                                   1.0 - 0.5 / (total + 1));
    norm(ar, tr); norm(ap, tp);
    const bool done = std::abs(new_pi - pi) < 1e-12;
    pi = new_pi;
    if (done) break;
  }
  if (tr[0] < tp[0]) { std::swap(tr, tp); pi = 1.0 - pi; }  // R is the class with the higher own rate
  cell_model.identified = true;
  cell_model.log_pi = std::log(pi);
  cell_model.log_1mpi = std::log1p(-pi);
  for (int j = 0; j < 2; ++j) { cell_model.log_r[j] = std::log(tr[j]); cell_model.log_p[j] = std::log(tp[j]); }
}

// In-place gate (README P21): a frame (or a look) stands in place when the object's own identity
// outnumbers the samples seen through (the posterior sign of the symmetric vote for any common
// reliability above 1/2). It decides which frames and looks teach the in-place statistics, never a
// commitment by itself. (The semi-supervised identity-mixture gate, f42f3fe, was removed: in the
// full chain L2_P9 its look gate learned fewer, higher-share looks, widening the in-place model, and
// delayed real C closures by 11-107 s with one D2 miss.)

// Judged samples for a share to stand for the finite counts (P25). Two replacements were tried
// once and lose GT-correct closures, so 30 is kept: the effective sample size n / deff of a count
// model with the overdispersion of in-place looks (2026-10-05 19:18; closes the GT-static real A
// inst 7 and 10), and n_min = m(1 - m) / v from the look's present moments (L2_FINAL replay: 28
// changes, among them the real B inst 10/13/19 and real C inst 7/10/20 closures lost, because a
// tight in-place model raises n_min to about 200).
constexpr size_t kMinSamplesInView = 30;
// Computational budget of surface cells per look. With uniformly strided cells the share's sampling
// standard deviation is sqrt(m(1 - m) / 1500) (0.44 % at an in-place share m = 0.03, at most 1.3 %),
// against the in-place model's floor of 1 %. It reaches decisions only through the sample count:
// halving it (counts halved) changes 12 closures through the 30-sample gate.
constexpr size_t kMaxAbsenceSamples = 1500;

enum Verdict : int8_t { kNone = -2, kInViewOnly = -1, kSeenThrough = 0, kOnSurface = 1 };

struct AbsenceQuery {
  AbsenceCell cell;
  Point point;
  Eigen::Vector3f normal;
  bool has_normal;
};

// One query per spatial cell of the stored surface: face centroids with their normals (the
// vertices of a face-less mesh), subsampled evenly to the computation budget.
std::vector<AbsenceQuery> absenceQueries(const spark_dsg::Mesh& mesh, const BoundingBox& bbox,
                                         const float cell_size) {
  std::vector<AbsenceQuery> queries;
  std::set<AbsenceCell> cells;
  const auto add = [&](const Point& p, const Eigen::Vector3f& n, const bool has_normal) {
    if (!p.array().isFinite().all()) return;
    const auto c = (p / cell_size).array().floor().cast<int64_t>().eval();
    const AbsenceCell cell{c.x(), c.y(), c.z()};
    if (cells.insert(cell).second) queries.push_back({cell, p, n, has_normal});
  };
  if (mesh.faces.empty()) {
    for (const auto& p : mesh.points) add(bbox.pointToWorldFrame(p), Eigen::Vector3f::Zero(), false);
  } else {
    for (const auto& f : mesh.faces) {
      if (f[0] >= mesh.numVertices() || f[1] >= mesh.numVertices() || f[2] >= mesh.numVertices()) continue;
      const Point a = bbox.pointToWorldFrame(mesh.pos(f[0]));
      const Point b = bbox.pointToWorldFrame(mesh.pos(f[1]));
      const Point c = bbox.pointToWorldFrame(mesh.pos(f[2]));
      Eigen::Vector3f n = (b - a).cross(c - a);
      const float length = n.norm();
      const bool valid = std::isfinite(length) && length > 1e-12f;
      if (valid) n /= length;
      add((a + b + c) / 3.f, n, valid);
    }
  }
  if (queries.size() > kMaxAbsenceSamples) {
    std::vector<AbsenceQuery> reduced;
    const double stride = static_cast<double>(queries.size()) / kMaxAbsenceSamples;
    for (size_t i = 0; i < kMaxAbsenceSamples; ++i) reduced.push_back(queries[static_cast<size_t>(i * stride)]);
    queries.swap(reduced);
  }
  return queries;
}

// Every stored frame not yet processed, once. A seen-through verdict needs a ray below the
// grazing incidence limit. In frames in which the object is identified in place (its own
// identity outweighs the part of the surface seen through in the same frame: a moved object
// overlapping its old site is identified on a strip and seen through on the rest) the outcomes
// enter the tentative reliability record; another identity on the surface vetoes the sample
// like a seen-through ray, a background label is only a missing detection.
void classifyFrames(ObjectAbsenceState& state, const std::vector<AbsenceQuery>& queries,
                    const RayVerificator::PhysicalEvidenceSnapshot& snapshot, const size_t physical_id,
                    const float tolerance, const float min_cos, const TimeStamp latest,
                    const uint64_t record_key = 0) {
  std::vector<int8_t> observed(queries.size()), raw(queries.size());
  std::vector<float> raw_cos(queries.size()), raw_range(queries.size()), raw_delta(queries.size());
  std::vector<bool> identified(queries.size()), foreign(queries.size());
  for (const auto stamp : snapshot->timestamps(state.processed + 1, latest)) {
    size_t identified_samples = 0, seen_through_samples = 0;
    // DIAG (P21/P23/P24 data, no decision reads it): per |cos| decile of the measured samples
    // with a normal: on-surface count, residual sums, range sums, and beyond-band count at any angle.
    std::array<double, 10> d_on{}, d_rr{}, d_zz{}, d_z{}, d_beyond{}, d_front{};
    size_t d_other = 0;
    for (size_t i = 0; i < queries.size(); ++i) {
      observed[i] = kNone;
      raw[i] = 0;
      identified[i] = false;
      foreign[i] = false;
      const auto p = snapshot->project(stamp, queries[i].point);
      const auto& e = p.endpoint;
      if (e.type == EndpointClass::kUnavailable) continue;
      const float cos_view = queries[i].has_normal
          ? std::abs(queries[i].normal.dot(p.view_direction_world)) : 1.f;
      const bool facing = !queries[i].has_normal || cos_view >= min_cos;
      const bool measured = e.type != EndpointClass::kInvalid &&
          std::isfinite(e.measured_depth_m) && e.measured_depth_m > 0 &&
          std::isfinite(p.query_range_m) && p.query_range_m > 0;
      if (!measured) { if (facing) observed[i] = kInViewOnly; continue; }
      const float delta = e.measured_depth_m - p.query_range_m;
      if (delta >= -tolerance) {
        raw[i] = delta <= tolerance ? 1 : 2;
        raw_cos[i] = cos_view;
        raw_range[i] = p.query_range_m;
        raw_delta[i] = delta;
      }
      if (queries[i].has_normal) {
        const size_t bin = std::min<size_t>(9, static_cast<size_t>(cos_view * 10.f));
        if (std::abs(delta) <= tolerance) {
          d_on[bin] += 1; d_rr[bin] += delta * delta; d_z[bin] += p.query_range_m;
          d_zz[bin] += p.query_range_m * p.query_range_m;
        } else if (delta > tolerance) {
          d_beyond[bin] += 1;
        } else {
          d_front[bin] += 1;
        }
      }
      if (std::abs(delta) <= tolerance) {
        observed[i] = kOnSurface;
        const bool physical = e.type == EndpointClass::kPhysical && e.physical_id > 0;
        identified[i] = physical && static_cast<size_t>(e.physical_id) == physical_id;
        foreign[i] = physical && !identified[i];
        identified_samples += identified[i];
        d_other += !identified[i];
      } else if (facing) {
        observed[i] = delta > tolerance ? kSeenThrough : kInViewOnly;
        seen_through_samples += observed[i] == kSeenThrough;
      }
    }
    const bool in_place = identified_samples > seen_through_samples;
    if (in_place) state.ever_identified = true;
    if (identified_samples + seen_through_samples + d_other > 0) {
      const auto join = [](const std::array<double, 10>& a) {
        std::string out;
        char buf[32];
        for (size_t b = 0; b < a.size(); ++b) {
          std::snprintf(buf, sizeof(buf), b ? ",%.6g" : "%.6g", a[b]);
          out += buf;
        }
        return out;
      };
      LOG(INFO) << "FRAME_DIAG inst=" << physical_id << " record=" << record_key << " stamp=" << stamp
                << " inherited=" << state.inherited << " identified=" << identified_samples
                << " seen_through=" << seen_through_samples << " other=" << d_other
                << " in_place=" << in_place << " on=" << join(d_on) << " rr=" << join(d_rr)
                << " z=" << join(d_z) << " zz=" << join(d_zz) << " beyond=" << join(d_beyond)
                << " front=" << join(d_front);
    }
    for (size_t i = 0; i < queries.size(); ++i) {
      if (raw[i] != 0) {
        auto& diag = state.samples[queries[i].cell];
        diag.diag_stamp = stamp;
        diag.diag_verdict = raw[i];
        diag.diag_cos = raw_cos[i];
        diag.diag_range = raw_range[i];
        diag.diag_delta = raw_delta[i];
      }
      if (observed[i] == kNone || observed[i] == kInViewOnly) continue;
      auto& sample = state.samples[queries[i].cell];
      if (observed[i] == kOnSurface) sample.last_on_surface = stamp;
      if (identified[i]) sample.last_identity = stamp;
      if (observed[i] == kSeenThrough) sample.last_seen_through = stamp;
      if (!in_place) continue;
      if (identified[i] && sample.tentative_hits < UINT16_MAX) ++sample.tentative_hits;
      if (observed[i] == kSeenThrough || foreign[i]) sample.tentative_veto = true;
      if (observed[i] == kOnSurface && !identified[i] && sample.tentative_other < UINT16_MAX) {
        ++sample.tentative_other;
      }
    }
    state.processed = stamp;
  }
  state.processed = std::max<TimeStamp>(state.processed, latest);
}

// A sample votes if it was never seen through while the object stood in place (the
// homogeneous subset the share model describes) and, for an identified session-born state,
// P(surface of this object | own / other label counts) >= 1/2 (P22). pi is estimated on
// cells observed in identified frames, so such a cell that never was has no evidence;
// inherited and never-identified states have no record and all their cells vote.
bool reliableSample(const ObjectAbsenceState& state, const AbsenceSample& sample) {
  if (sample.seen_through_while_identified) return false;
  if (state.inherited || !state.ever_identified) return true;
  const double own = static_cast<double>(sample.identity_hits) + sample.tentative_hits;
  return own + sample.other_obs > 0.0 && cellLogOdds(cell_model, own, sample.other_obs) >= 0.0;
}

// One look: the latest verdict of every reliable sample judged in this round.
struct AbsenceLook {
  size_t reliable = 0, on_surface = 0, seen_through = 0, own_identity = 0;
  // DIAG (P28 replay; no decision reads it): samples whose latest verdict in the round carries the
  // object's own identity on the surface.
  size_t own_latest = 0;
  std::vector<AbsenceCell> judged;
  std::vector<int32_t> previous_look;  // of each judged sample
  size_t verdicts() const { return on_surface + seen_through; }
};

AbsenceLook summarizeLook(const ObjectAbsenceState& state, const std::vector<AbsenceQuery>& queries,
                          const TimeStamp round_start) {
  AbsenceLook look;
  for (const auto& query : queries) {
    const auto it = state.samples.find(query.cell);
    if (it == state.samples.end()) continue;
    const auto& sample = it->second;
    if (sample.last_identity >= round_start && sample.last_identity != 0) ++look.own_identity;
    if (sample.last_identity >= round_start && sample.last_identity != 0 &&
        sample.last_identity >= std::max(sample.last_on_surface, sample.last_seen_through)) {
      ++look.own_latest;
    }
    if (!reliableSample(state, sample)) continue;
    ++look.reliable;
    const TimeStamp last = std::max(sample.last_on_surface, sample.last_seen_through);
    if (last < round_start || last == 0) continue;
    ++(sample.last_seen_through > sample.last_on_surface ? look.seen_through : look.on_surface);
    look.judged.push_back(query.cell);
    look.previous_look.push_back(sample.last_look);
  }
  return look;
}

// DIAG: the round's latest outcome of every reliable sample at any incidence, "verdict:|cos|:range:delta".
std::string diagSamples(const ObjectAbsenceState& state, const std::vector<AbsenceQuery>& queries,
                        const TimeStamp round_start) {
  std::string out;
  char buf[64];
  for (const auto& query : queries) {
    const auto it = state.samples.find(query.cell);
    if (it == state.samples.end() || !reliableSample(state, it->second)) continue;
    const auto& d = it->second;
    if (d.diag_stamp < round_start || d.diag_stamp == 0) continue;
    std::snprintf(buf, sizeof(buf), "%s%d:%.3f:%.2f:%.3f", out.empty() ? "" : ";", d.diag_verdict,
                  d.diag_cos, d.diag_range, d.diag_delta);
    out += buf;
  }
  return out.empty() ? "-" : out;
}

// Cold start of the in-place share, used only before any population exists (the first session
// ever; later sessions load the carried population). Mean: the prior rate at which a surface
// sample of an object in place is judged seen through, i.e. the type-I error rate of the
// per-sample seen-through test, taken at the conventional significance level alpha = 0.05
// (Fisher 1925, Statistical Methods for Research Workers, ch. III sec. 12: "it is convenient to
// take this point as a limit in judging whether a deviation is to be considered significant or
// not"; Rosen, Mason, Leonard ICRA 2016 treat detector miss / false-alarm rates as innate detector
// characteristics, evaluated at .01, .05, .1, ...). Variance: that of the uniform share, 1/12
// (Bayes-Laplace prior Beta(1, 1)). Since m(1 - m) <= 1/4, any v >= 1/12 projects to the
// concentration floor 2 below, so the cold start is Beta(2 alpha, 2 (1 - alpha)) = Beta(0.1, 1.9);
// the former 0.09 projected to the same Beta (L2_FINAL replay: no decision changes).
constexpr double kColdStartSeenThroughRate = 0.05;
constexpr double kUniformShareVariance = 1.0 / 12.0;
// Total pseudo-count of the uniform prior Beta(1, 1): the in-place model never carries less
// information than the uniform prior (the same convention as the cold-start variance).
constexpr double kUniformPriorPseudoCount = 2.0;

// In-place model of a look's share (P25-P27 record why these moments stay). The object's own
// looks (robust scatter from kRobustLooks looks on) are shrunk towards the population of objects,
// which counts as kRobustLooks pseudo-looks, and the variance is never below the spread between
// objects: an object seen from a new viewpoint varies at least as much as objects vary among
// themselves. Before any population exists the cold start above stands in. Returns the mean and
// the second moment. Caller holds absence_mutex.
std::pair<double, double> presentMoments(const ObjectAbsenceState& state) {
  const double h = static_cast<double>(kRobustLooks);
  double v0 = pooled_geo_dev.size() >= kRobustLooks ? robustVariance(pooled_geo_dev, -1.0) : loaded_geo_var;
  double m0 = pooled_n >= h ? pooled_sum / pooled_n : 0.0;
  if (!(pooled_n >= h && v0 > 0)) { m0 = kColdStartSeenThroughRate; v0 = kUniformShareVariance; }
  const double own_n = state.history_n;
  const double own_m = own_n > 0 ? state.history_sum / own_n : 0.0;
  const double own_v = own_n >= h ? robustVariance(state.looks, own_m) : 0.0;
  const double k = h, use_n = own_n >= h ? own_n : 0.0;
  const double m = (use_n * own_m + k * m0) / (use_n + k);
  const double v = std::max(v0, (use_n * (own_v + (own_m - m) * (own_m - m)) +
                                 k * (v0 + (m0 - m) * (m0 - m))) / (use_n + k));
  return {m, m * m + v};
}

// Beta projection of the moments (concentration at least 2; README O10h for the bounds).
struct PresentBeta {
  double m = 0.0, a = 0.0, b = 0.0;
};

PresentBeta presentBeta(const double mean, const double second) {
  PresentBeta beta;
  // Numerical guard of the mean; it never acted in the acceptance data (L2_FINAL replay).
  beta.m = std::min(0.999, std::max(0.001, mean));
  const double v = std::max(kShareVarianceFloor, second - mean * mean);
  const double c = std::max(kUniformPriorPseudoCount, beta.m * (1 - beta.m) / v - 1);
  beta.a = beta.m * c + 1e-3;
  beta.b = (1 - beta.m) * c + 1e-3;
  return beta;
}

// Log density of a share under the present model. One-sided: only more seen-through than the
// object usually shows speaks for absence; a share below the mean is scored at the mean.
// Bounds of the scored share: 0.005 changes no decision (x0.5, x2). 0.995 acts for looks with every
// judged sample seen through (f > 0.995: 4 real, 28 synthetic looks in L2_FINAL); 0.99 loses the
// GT-correct real C inst 2 closure, and so does the Jeffreys share (k + 1/2) / (n + 1) (its decisive
// look has k = n = 31: 0.984 < 0.995), while the Beta-binomial predictive (2026-10-05 19:18) closes
// the GT-static real A inst 7 and 10. Kept as known item.
double presentLogDensity(const PresentBeta& beta, double f) {
  f = std::min(0.995, std::max(std::max(0.005, beta.m), f));
  return std::lgamma(beta.a + beta.b) - std::lgamma(beta.a) - std::lgamma(beta.b) +
         (beta.a - 1) * std::log(f) + (beta.b - 1) * std::log(1 - f);
}

// The same look as a finite-count ratio absent : present (uniform share against the
// Beta-binomial of the present model), one-sided.
double finiteCountLogRatio(const double k, const double n, const PresentBeta& beta) {
  const double log_present_count =
      std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0) +
      std::lgamma(k + beta.a) + std::lgamma(n - k + beta.b) - std::lgamma(n + beta.a + beta.b) -
      std::lgamma(beta.a) - std::lgamma(beta.b) + std::lgamma(beta.a + beta.b);
  double log_ratio = -std::log1p(n) - log_present_count;
  if (k / n <= beta.a / (beta.a + beta.b)) log_ratio = std::min(0.0, log_ratio);
  return log_ratio;
}

// Page's test with candidate-dependent weights. For a candidate change at scored look s, a look
// counts in proportion to the share of the reliable surface it judged for the first time since
// s: each sample speaks once about a change, so the total weight of a candidate is at most the
// object. With weights independent of s this is the recursion max(0, S + w llr); here the
// maximum over s is taken explicitly. Returns the weight of the look under the strongest
// candidate before it (a new candidate when none is positive).
double addLook(ObjectAbsenceState& state, const AbsenceLook& look, const double llr) {
  const size_t t = state.page.size();
  std::vector<size_t> first_since(t + 1, 0);
  for (const int32_t p : look.previous_look) ++first_since[static_cast<size_t>(p + 1)];
  for (size_t s = 1; s <= t; ++s) first_since[s] += first_since[s - 1];
  const auto weight = [&](const size_t s) {
    return std::min(1.0, static_cast<double>(first_since[s]) / std::max<size_t>(1, look.reliable));
  };
  size_t strongest = t;
  for (size_t s = 0; s < t; ++s) {
    if (state.page[s] > (strongest == t ? 0.0 : state.page[strongest])) strongest = s;
  }
  state.page.push_back(0.0);
  for (size_t s = 0; s <= t; ++s) state.page[s] += weight(s) * llr;
  for (const auto& cell : look.judged) state.samples[cell].last_look = static_cast<int32_t>(t);
  state.cusum = std::max(0.0, *std::max_element(state.page.begin(), state.page.end()));
  return weight(strongest);
}

// Judge, then learn: an in-place look joins the object's own statistics after it was scored,
// and the object joins the population with the mean of its first three looks (storage bounded).
// Caller holds absence_mutex.
void learnInPlaceLook(ObjectAbsenceState& state, const double f) {
  state.history_n += 1;
  state.history_sum += f;
  if (state.looks.size() < 256) state.looks.push_back(f);
  const double h = static_cast<double>(kRobustLooks);
  if (state.history_n == h && pooled_geo_dev.size() < 4096) {
    pooled_geo_dev.push_back(state.history_sum / h);
    pooled_n += 1;
    pooled_sum += state.history_sum / h;
  }
}

// The round's tentative reliability record is committed when the object stood in place.
void commitReliability(ObjectAbsenceState& state, const bool in_place) {
  for (auto& [cell, sample] : state.samples) {
    (void)cell;
    if (in_place) {
      sample.identity_hits = static_cast<uint16_t>(
          std::min<size_t>(UINT16_MAX, size_t{sample.identity_hits} + sample.tentative_hits));
      sample.other_obs = static_cast<uint16_t>(
          std::min<size_t>(UINT16_MAX, size_t{sample.other_obs} + sample.tentative_other));
      sample.seen_through_while_identified |= sample.tentative_veto;
    }
    sample.tentative_hits = 0;
    sample.tentative_other = 0;
    sample.tentative_veto = false;
  }
}

}  // namespace

PhysicalAbsenceLookLikelihood physicalAbsenceLookLikelihood(
    const RayVerificator* owner, const size_t physical_id, const int state_slot,
    const TimeStamp stamp, const uint64_t state_id) {
  const uint64_t state_key = absenceStateKey(state_id, state_slot);
  std::lock_guard<std::mutex> lock(absence_mutex);
  for (const auto& [key, state] : absence_states) {
    if (std::get<1>(key) == physical_id && std::get<2>(key) == state_key &&
        state->likelihood_owner == owner && state->likelihood_stamp == stamp) {
      return state->likelihood;
    }
  }
  return {};
}

// The population of in-place shares is a property of the sensor and the scene, part of the
// memory a session exports: a later session starts its inherited states with it. Format:
// count, sum of the per-object means, their robust variance (older files carry three more
// fields of a label channel that is not evidence, P28; they are ignored).
bool saveAbsenceSensorStatistics(const std::string& path) {
  std::lock_guard<std::mutex> lock(absence_mutex);
  std::ofstream out(path);
  if (!out) return false;
  out.precision(17);
  const double gv = pooled_geo_dev.size() >= kRobustLooks ? robustVariance(pooled_geo_dev, -1.0) : loaded_geo_var;
  out << pooled_n << ' ' << pooled_sum << ' ' << gv << '\n';
  return static_cast<bool>(out);
}

bool loadAbsenceSensorStatistics(const std::string& path) {
  std::ifstream in(path);
  double n = 0, sum = 0, gv = 0;
  if (!(in >> n >> sum >> gv)) return false;
  std::lock_guard<std::mutex> lock(absence_mutex);
  pooled_n += n; pooled_sum += sum; loaded_geo_var = gv;
  return true;
}

RayVerificator::CheckResult RayVerificator::checkProjectedPhysical(
    const Point& point, const size_t physical_id,
    const PhysicalEvidenceSnapshot& evidence_snapshot,
    const uint64_t earliest, const uint64_t latest) const {
  CheckResult result;
  if (!evidence_snapshot) return result;
  for (const auto stamp : evidence_snapshot->timestamps(earliest, latest)) {
    const auto p = evidence_snapshot->project(stamp, point);
    switch (classifyMeasurement(p, physical_id, config.depth_tolerance)) {
      case Vote::Supported:
        result.present.push_back(stamp); ++result.reasons.same_id; break;
      case Vote::Free:
        result.absent.push_back(stamp); ++result.reasons.free_space; break;
      case Vote::Background:
        result.absent.push_back(stamp); ++result.reasons.background_replacement; break;
      case Vote::Other:
        result.absent.push_back(stamp); ++result.reasons.different_id; break;
      case Vote::Occluded:
        result.inconclusive.push_back(stamp); ++result.reasons.geometric_occlusion; break;
      case Vote::Unidentified:
        result.inconclusive.push_back(stamp); ++result.reasons.unidentified_object; break;
      case Vote::Invalid:
        result.inconclusive.push_back(stamp); ++result.reasons.invalid; break;
      case Vote::Unavailable:
        break;  // Outside FOV / absent frame is not a coverage measurement.
    }
  }
  return result;
}

RayVerificator::SurfaceEvidenceCounts RayVerificator::countProjectedPhysicalSurface(
    const size_t physical_id, const spark_dsg::Mesh& mesh, const BoundingBox& bbox,
    const PhysicalEvidenceSnapshot& evidence_snapshot, const float map_resolution,
    const uint64_t earliest, const uint64_t latest) const {
  SurfaceEvidenceCounts result;
  if (!evidence_snapshot || !std::isfinite(map_resolution) || map_resolution <= 0) return result;

  // One actual surface sample per map cell keeps duplicate triangle storage
  // from multiplying evidence and bounds the cost of the fallback.
  Points samples;
  std::set<std::tuple<int64_t, int64_t, int64_t>> cells;
  const auto add_sample = [&](const Point& p) {
    if (!p.array().isFinite().all()) return;
    const auto cell = (p / map_resolution).array().floor().cast<int64_t>().eval();
    if (cells.emplace(cell.x(), cell.y(), cell.z()).second) samples.push_back(p);
  };
  if (mesh.faces.empty()) {
    for (const auto& p : mesh.points) add_sample(bbox.pointToWorldFrame(p));
  } else {
    for (const auto& f : mesh.faces) {
      if (f[0] >= mesh.numVertices() || f[1] >= mesh.numVertices() || f[2] >= mesh.numVertices()) continue;
      add_sample(bbox.pointToWorldFrame((mesh.pos(f[0])+mesh.pos(f[1])+mesh.pos(f[2]))/3.f));
    }
  }
  result.surface_samples = samples.size();
  const auto stamps = evidence_snapshot->timestamps(earliest, latest);
  for (const auto& point : samples) {
    bool coverage = false;
    TimeStamp sample_support = 0, sample_absence = 0;
    for (size_t frame = 0; frame < stamps.size(); ++frame) {
      const auto p = evidence_snapshot->project(stamps[frame], point);
      const auto vote = classifyMeasurement(p, physical_id, config.depth_tolerance);
      if (vote == Vote::Unavailable) continue;
      coverage = true;
      // Exact (frame, pixel), not (sample, frame): a pixel is one measurement
      // even when several surface triangles project onto it.
      const size_t key = (frame << 32) | static_cast<size_t>(p.pixel_index);
      switch (vote) {
        case Vote::Supported:
          sample_support = stamps[frame];
          result.latest_support_stamp = std::max(result.latest_support_stamp, stamps[frame]);
          result.support_indices.insert(key); ++result.supported_votes; break;
        case Vote::Free:
          sample_absence = stamps[frame];
          result.contradiction_indices.insert(key); ++result.free_space_votes; break;
        case Vote::Background:
          sample_absence = stamps[frame];
          result.contradiction_indices.insert(key); ++result.replaced_by_background_votes; break;
        case Vote::Other:
          sample_absence = stamps[frame];
          result.contradiction_indices.insert(key); ++result.replaced_by_other_votes; break;
        case Vote::Occluded:
          ++result.occluded_votes; break;
        default:
          break;
      }
    }
    if (!coverage) ++result.unobserved_samples;
    if (sample_absence > sample_support) ++result.contradicted_surface_samples;
  }
  // Compatibility layer (P20, kept pending, user 2026-10-05): the proxy-free coverage verdict of
  // the unit interface. Its only production caller, countCurrentPhysicalSurface, overwrites it at
  // once with the observed-absence test (7) (applyObservedAbsence).
  result.absence_coverage_sufficient = result.surface_samples > 0 &&
      result.contradicted_surface_samples > 0 &&
      static_cast<double>(result.contradicted_surface_samples) / result.surface_samples >=
          config.min_absent_surface_fraction;
  result.support_rays = result.support_indices.size();
  result.contradiction_rays = result.contradiction_indices.size();
  return result;
}

void RayVerificator::applyObservedAbsence(
    const size_t physical_id, const spark_dsg::Mesh& mesh, const BoundingBox& bbox,
    const PhysicalEvidenceSnapshot& evidence_snapshot, const uint64_t /*earliest*/,
    const uint64_t latest, SurfaceEvidenceCounts& counts, const int state_slot,
    const uint64_t state_birth, const double prior_log_odds, const uint64_t state_id) const {
  counts.absence_coverage_sufficient = false;
  if (!evidence_snapshot) return;
  const float tolerance = config.surface_match_tolerance;
  const float min_cos = std::cos(config.max_absence_incidence_deg * static_cast<float>(M_PI) / 180.f);
  const std::vector<AbsenceQuery> queries = absenceQueries(mesh, bbox, 0.5f * tolerance);

  std::shared_ptr<ObjectAbsenceState> state;
  {
    std::lock_guard<std::mutex> lock(absence_mutex);
    // The record belongs to the state it tests, identified by the state's id (a state handed over
    // from the session slot to the top slot keeps its record; a new state gets a new one; merges
    // that move a state's birth earlier keep it). Callers without a state id (offline tools,
    // fixtures) key by slot.
    const uint64_t state_key = absenceStateKey(state_id, state_slot);
    auto& slot = absence_states[{absence_owner_, physical_id, state_key}];
    if (!slot) {
      slot = std::make_shared<ObjectAbsenceState>();
      // Frames before a state exists are no evidence about it.
      if (state_birth > 0) slot->processed = state_birth - 1;
    }
    state = slot;
    // An allocator may reuse an address after a verificator is destroyed.
    // Keep ownership bound to its existing unique lifetime token.
    for (const auto& [old_key, old_state] : absence_states) {
      if (std::get<0>(old_key) != absence_owner_ && old_state->likelihood_owner == this) {
        old_state->likelihood_owner = nullptr;
      }
    }
    state->likelihood_owner = this;
    state->likelihood_stamp = latest;
    state->likelihood = {0.0, false, true};
  }
  if (latest < state->processed) {  // a new session restarts time
    state->processed = 0;
    state->ever_identified = false;
    state->samples.clear();
  }
  const TimeStamp round_start = state->processed + 1;
  {
    const auto all = evidence_snapshot->timestamps(0, latest);
    state->inherited = !all.empty() && state_birth != 0 && state_birth < all.front();
  }
  // Every surface cell is part of the object whether or not this round saw it:
  // the share a look judged is measured against the whole reliable surface.
  for (const auto& query : queries) state->samples.try_emplace(query.cell);
  classifyFrames(*state, queries, evidence_snapshot, physical_id, tolerance, min_cos, latest,
                 absenceStateKey(state_id, state_slot));
  {
    std::lock_guard<std::mutex> lock(absence_mutex);
    estimateCellModel(latest);
  }
  const AbsenceLook look = summarizeLook(*state, queries, round_start);
  const size_t verdicts = look.verdicts();
  counts.reliable_samples = look.reliable;
  counts.reliable_in_view = verdicts;
  counts.reliable_seen_through = look.seen_through;
  // In-place vote on like counts (P21): samples whose latest verdict in the round is the object's
  // own identity on the surface against samples whose latest verdict is seen through. Counting
  // identity at any time of the round taught a look that spans a move as in place (real C inst 2).
  const bool in_place = look.own_latest > look.seen_through;
  // A share stands for the finite counts once the look judged min(30, reliable) samples (P25);
  // partial views are weighted, not refused. A round without a judged reliable sample says
  // nothing about presence or absence and is neither scored nor learned from.
  if (verdicts > 0 && verdicts >= std::min<size_t>(kMinSamplesInView, look.reliable)) {
    const double f = static_cast<double>(look.seen_through) / verdicts;
    PresentBeta beta;
    {
      std::lock_guard<std::mutex> lock(absence_mutex);
      const auto [mean, second] = presentMoments(*state);
      beta = presentBeta(mean, second);
    }
    const double look_llr = -presentLogDensity(beta, f);
    {
      // DIAG (exact replay of eq. 7 under other present models; no decision reads it).
      std::vector<size_t> first_since(state->page.size() + 1, 0);
      for (const int32_t p : look.previous_look) ++first_since[static_cast<size_t>(p + 1)];
      std::string since;
      size_t cum = 0;
      for (size_t c = 0; c < first_since.size(); ++c) {
        cum += first_since[c];
        since += (c ? "," : "") + std::to_string(cum);
      }
      LOG(INFO) << "ABSENCE_LOOK inst=" << physical_id << " slot=" << state_slot
                << " record=" << absenceStateKey(state_id, state_slot) << " stamp=" << latest
                << " k=" << look.seen_through << " n=" << verdicts << " reliable=" << look.reliable
                << " own=" << look.own_identity << " own_latest=" << look.own_latest
                << " identified=" << in_place
                << " inherited=" << state->inherited << " llr=" << look_llr
                << " cusum_before=" << state->cusum << " first_since=" << since
                << " samples=" << diagSamples(*state, queries, round_start);
    }
    const double weight = addLook(*state, look, look_llr);
    {
      std::lock_guard<std::mutex> lock(absence_mutex);
      if (weight > 0.0) {
        const double log_ratio =
            finiteCountLogRatio(static_cast<double>(look.seen_through), static_cast<double>(verdicts), beta);
        state->likelihood = {weight * log_ratio, true, true};
      }
      if (in_place) learnInPlaceLook(*state, f);
    }
  } else if (verdicts > 0) {
    LOG(INFO) << "ABSENCE_UNSCORED inst=" << physical_id << " slot=" << state_slot << " stamp=" << latest
              << " k=" << look.seen_through << " n=" << verdicts << " reliable=" << look.reliable
              << " own=" << look.own_identity << " samples=" << diagSamples(*state, queries, round_start);
  }
  counts.absence_llr = static_cast<float>(state->cusum);
  VLOG(1) << "OBSERVED_ABSENCE inst=" << physical_id << " slot=" << state_slot
          << " queries=" << queries.size() << " reliable=" << look.reliable
          << " verdicts=" << verdicts << " seen_through=" << look.seen_through
          << " cusum=" << state->cusum;
  // README M4: posterior odds that the site is empty, committed at the declared 99:1 loss. The
  // prior odds q/(1-q) belong to the cross-session relation, a change before this session's
  // first look: they multiply that candidate's ratio alone (page[0] of the first window; before
  // any scored look the ratio is 1). A change inside the session carries no such prior and is
  // tested by the Page statistic alone.
  const double before_session =
      state->first_window ? prior_log_odds + (state->page.empty() ? 0.0 : state->page.front())
                          : -std::numeric_limits<double>::infinity();
  counts.absence_coverage_sufficient = std::max(before_session, state->cusum) > std::log(99.0);
  if (prior_log_odds != 0.0) {
    LOG(INFO) << "ABSENCE_COMMIT inst=" << physical_id << " slot=" << state_slot
              << " prior_log_odds=" << prior_log_odds << " before_session=" << before_session
              << " page_statistic=" << state->cusum
              << " committed=" << counts.absence_coverage_sufficient;
  }
  commitReliability(*state, in_place);
  // A commitment restarts the test with all its candidates and first-judgment bookkeeping.
  // The state may still continue when the rays do not confirm the absence (C <= S).
  if (counts.absence_coverage_sufficient) {
    state->cusum = 0;
    state->page.clear();
    state->first_window = false;
    for (auto& [cell, sample] : state->samples) {
      (void)cell;
      sample.last_look = -1;
    }
  }
}

RayVerificator::SurfaceEvidenceCounts RayVerificator::countCurrentPhysicalSurface(
    size_t physical_id, const spark_dsg::Mesh& mesh, const BoundingBox& bbox,
    const PhysicalEvidenceSnapshot& snapshot, float map_resolution,
    uint64_t last_support, uint64_t latest, bool* projected, const int state_slot,
    const uint64_t state_birth, const double prior_log_odds, const uint64_t state_id) const {
  if (projected) *projected = false;
  // Also prevents unsigned overflow and invalid inclusive intervals.
  if (last_support >= latest) {
    SurfaceEvidenceCounts none;
    applyObservedAbsence(physical_id, mesh, bbox, snapshot, latest, latest, none, state_slot, state_birth,
                         0.0, state_id);
    none.absence_coverage_sufficient = false;
    return none;
  }
  const uint64_t earliest = last_support + 1;
  if (!snapshot || snapshot->numFrames() == 0) {
    // Compatibility layer (P19, kept pending, user 2026-10-05): no stored pixels. In production
    // every verification round has the session's frame archive, so this branch is reached only by
    // offline tools and unit fixtures (test_hidden_change_session_equivalence); the sparse
    // mesh-ray proxy of the native verificator stands in, decided by its majority.
    auto counts = countPhysicalSurface(physical_id, mesh, bbox, snapshot, earliest, latest);
    counts.absence_coverage_sufficient = counts.contradiction_rays > counts.support_rays;
    return counts;
  }
  // The measurement is the stored pixels (README M1m): support and contradiction on the
  // state's own surface, and the observed-absence test (7) on the same frames. Its proxy-free
  // coverage verdict (P20) is replaced by that test.
  if (projected) *projected = true;
  auto measured = countProjectedPhysicalSurface(physical_id, mesh, bbox, snapshot, map_resolution,
                                                earliest, latest);
  applyObservedAbsence(physical_id, mesh, bbox, snapshot, earliest, latest, measured, state_slot,
                       state_birth, prior_log_odds, state_id);
  return measured;
}

}  // namespace khronos
