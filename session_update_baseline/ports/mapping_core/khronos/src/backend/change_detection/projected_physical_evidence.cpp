#include "khronos/backend/change_detection/ray_verificator.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <fstream>
#include <set>
#include <tuple>

namespace khronos {
namespace {

enum class Vote { Unavailable, Invalid, Occluded, Supported, Free,
                  Background, Other, Unidentified };

Vote classifyMeasurement(const ProjectedEndpointEvidence& p, size_t id, float tolerance) {
  const auto& e = p.endpoint;
  if (e.type == EndpointClass::kUnavailable) return Vote::Unavailable;
  if (e.type == EndpointClass::kInvalid || !std::isfinite(e.measured_depth_m) ||
      !std::isfinite(p.query_range_m) || e.measured_depth_m <= 0 || p.query_range_m <= 0) {
    return Vote::Invalid;
  }
  const float delta = e.measured_depth_m - p.query_range_m;
  if (delta < -tolerance) return Vote::Occluded;
  const bool same_identity = e.type == EndpointClass::kPhysical &&
      e.physical_id > 0 && static_cast<size_t>(e.physical_id) == id;
  // The matching tolerance permits shape/pose error for the same object;
  // it never permits a different, nearer surface to see through an occluder.
  // Synthetic A I49: measured 5.279 m versus queried 5.5785 m was previously
  // misclassified as background replacement inside the 0.3 m matching band.
  // PhysicalEvidenceStore quantizes range to millimetres; exact same-depth
  // replacement must survive that rounding, without a 30 cm occlusion band.
  if (!same_identity && delta < -1e-3f) return Vote::Occluded;
  if (delta > tolerance) return Vote::Free;
  if (e.type == EndpointClass::kBackground) return Vote::Background;
  if (e.type == EndpointClass::kUnidentifiedObject) return Vote::Unidentified;
  if (e.type == EndpointClass::kPhysical) {
    return e.physical_id > 0 && static_cast<size_t>(e.physical_id) == id
               ? Vote::Supported : Vote::Other;
  }
  return Vote::Invalid;
}


// ---------------------------------------------------------------------------
// Observed absence.
//
// A stored object surface is incomplete and partly wrong: depth on dark or
// specular material lands behind the object, and rays that graze a surface or
// pass its silhouette end on whatever lies behind it. Such samples are seen
// through while the object stands in place. Counting them let an object be
// closed although it never moved. Three conditions make "seen empty" an
// observation of the object rather than of those artefacts:
//   1. only reliable samples vote: built from the object's own identity in at
//      least three frames, and never seen through in a frame in which the
//      object itself was identified;
//   2. a seen-through ray counts only below the grazing incidence limit;
//   3. the seen-through samples are the majority of the reliable samples that
//      entered the view since the object was last identified. Occluded samples
//      are in view and not seen through: a sliver of an object cannot testify
//      for the whole of it.
// The state is incremental per object: every stored frame is evaluated once.
struct AbsenceSample {
  // Committed reliability record. Hits and vetoes observed in a round are only
  // committed when that round does not end with the verdict "empty": frames seen
  // after a move must not teach the model of the state that has just ended.
  uint16_t identity_hits = 0;
  bool seen_through_while_identified = false;
  uint16_t tentative_hits = 0;
  bool tentative_veto = false;
  // Latest verdicts of this sample. Occluded or unmeasured looks give no verdict.
  TimeStamp last_on_surface = 0;
  TimeStamp last_seen_through = 0;
  TimeStamp last_identity = 0;
  // On the surface but labelled as another object or background: the site is
  // occupied by something else. Only meaningful relative to the object's own
  // history of being mislabelled while it stands in place.
  TimeStamp last_foreign = 0;
  // Already contributed to the running accumulation: looking again at the same
  // surface is not new evidence.
  bool counted = false;
};
using AbsenceCell = std::tuple<int64_t, int64_t, int64_t>;
struct ObjectAbsenceState {
  TimeStamp processed = 0;
  // False until the object is identified in a stored frame. An inherited state whose
  // object has left is never identified again, so its samples have no reliability
  // record; then every sample votes.
  bool ever_identified = false;
  // Seen-through share of the looks in which the object was identified in place: the
  // object's own missed-detection behaviour under this sensor. Running moments.
  double history_n = 0, history_sum = 0, history_sq = 0;        // geometric: seen through
  double label_n = 0, label_sum = 0, label_sq = 0;              // label: on surface, foreign label
  std::vector<double> geo_looks, label_looks;                   // the looks themselves (robust scale)
  // The state's surface was built from identity observations of an earlier session.
  bool inherited = false;
  double cusum = 0;  // accumulated log-likelihood ratio absent : present
  std::map<AbsenceCell, AbsenceSample> samples;
};
std::mutex absence_mutex;
// Pooled over all objects: the prior for a state that was never identified in this
// process (an inherited state whose object has left).
double pooled_n = 0, pooled_sum = 0;
double pooled_label_n = 0, pooled_label_sum = 0, pooled_label_dev_n = 0, pooled_label_dev_sq = 0;
// Per-object means of the two statistics once an object has three looks: the prior
// for an object without its own history is the spread BETWEEN objects (an exact
// sensor makes every object alike; a real one makes dark, thin and shiny objects differ).
std::vector<double> pooled_geo_dev, pooled_label_dev;
double loaded_geo_var = -1, loaded_label_var = -1;      // scatter carried over from the previous session

// Robust scale: 1.4826 * median absolute deviation. A few looks at a partly
// re-occupied old site must not widen the scatter of an otherwise exact sensor.
// With centre < 0 the median of the values is used as the centre.
double robustVariance(std::vector<double> values, double centre) {
  if (values.empty()) return 1e-4;
  if (centre < 0) {
    std::vector<double> c(values);
    std::nth_element(c.begin(), c.begin() + c.size() / 2, c.end());
    centre = c[c.size() / 2];
  }
  for (auto& v : values) v = std::abs(v - centre);
  std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
  const double mad = values[values.size() / 2];
  return std::max(1e-4, (1.4826 * mad) * (1.4826 * mad));
}
// Within-object scatter of the looks (deviation from the object's own running mean):
// the spread a single object shows, not the spread between objects.
double pooled_dev_n = 0, pooled_dev_sq = 0;
std::map<std::tuple<uint64_t, size_t, int>, std::shared_ptr<ObjectAbsenceState>> absence_states;

constexpr size_t kMinIdentifiedSamples = 3;
constexpr uint16_t kMinIdentityHits = 3;
constexpr size_t kMinSamplesInView = 30;
constexpr size_t kMaxAbsenceSamples = 1500;

}  // namespace

// The learned scatter of "seen-through share" is a property of the sensor and
// the scene, part of the memory a session exports: a later session starts its
// inherited states with it instead of with no prior at all.
bool saveAbsenceSensorStatistics(const std::string& path) {
  std::lock_guard<std::mutex> lock(absence_mutex);
  std::ofstream out(path);
  if (!out) return false;
  out.precision(17);
  const double gv = pooled_geo_dev.size() >= 3 ? robustVariance(pooled_geo_dev, -1.0) : loaded_geo_var;
  const double lv = pooled_label_dev.size() >= 3 ? robustVariance(pooled_label_dev, -1.0) : loaded_label_var;
  out << pooled_n << ' ' << pooled_sum << ' ' << gv << ' ' << pooled_label_n << ' ' << pooled_label_sum << ' ' << lv << '\n';
  return static_cast<bool>(out);
}

bool loadAbsenceSensorStatistics(const std::string& path) {
  std::ifstream in(path);
  double n = 0, sum = 0, gv = 0, ln = 0, ls = 0, lv = 0;
  if (!(in >> n >> sum >> gv >> ln >> ls >> lv)) return false;
  std::lock_guard<std::mutex> lock(absence_mutex);
  pooled_n += n; pooled_sum += sum; loaded_geo_var = gv;
  pooled_label_n += ln; pooled_label_sum += ls; loaded_label_var = lv;
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
    const PhysicalEvidenceSnapshot& evidence_snapshot, const uint64_t earliest,
    const uint64_t latest, SurfaceEvidenceCounts& counts, const int state_slot,
    const uint64_t state_birth) const {
  counts.absence_coverage_sufficient = false;
  if (!evidence_snapshot) return;
  const float tolerance = config.surface_match_tolerance;
  const float min_cos = std::cos(config.max_absence_incidence_deg * static_cast<float>(M_PI) / 180.f);

  struct Query { AbsenceCell cell; Point point; Eigen::Vector3f normal; bool has_normal; };
  std::vector<Query> queries;
  std::set<AbsenceCell> cells;
  const float cell_size = 0.5f * tolerance;
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
    std::vector<Query> reduced;
    const double stride = static_cast<double>(queries.size()) / kMaxAbsenceSamples;
    for (size_t i = 0; i < kMaxAbsenceSamples; ++i) reduced.push_back(queries[static_cast<size_t>(i * stride)]);
    queries.swap(reduced);
  }

  std::shared_ptr<ObjectAbsenceState> state;
  {
    std::lock_guard<std::mutex> lock(absence_mutex);
    auto& slot = absence_states[{absence_owner_, physical_id, state_slot}];
    if (!slot) slot = std::make_shared<ObjectAbsenceState>();
    state = slot;
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
  enum : int8_t { kNone = -2, kInViewOnly = -1, kSeenThrough = 0, kOnSurface = 1 };
  std::vector<int8_t> observed(queries.size());
  std::vector<bool> identified(queries.size());
  std::vector<bool> foreign(queries.size());
  // Every surface cell is part of the object whether or not this round saw it:
  // the share a look judged is measured against the whole reliable surface.
  for (const auto& query : queries) state->samples.try_emplace(query.cell);
  for (const auto stamp : evidence_snapshot->timestamps(state->processed + 1, latest)) {
    size_t identified_samples = 0;
    for (size_t i = 0; i < queries.size(); ++i) {
      observed[i] = kNone;
      identified[i] = false;
      foreign[i] = false;
      const auto p = evidence_snapshot->project(stamp, queries[i].point);
      const auto& e = p.endpoint;
      if (e.type == EndpointClass::kUnavailable) continue;
      const bool facing = !queries[i].has_normal ||
          std::abs(queries[i].normal.dot(p.view_direction_world)) >= min_cos;
      const bool measured = e.type != EndpointClass::kInvalid &&
          std::isfinite(e.measured_depth_m) && e.measured_depth_m > 0 &&
          std::isfinite(p.query_range_m) && p.query_range_m > 0;
      if (!measured) { if (facing) observed[i] = kInViewOnly; continue; }
      const float delta = e.measured_depth_m - p.query_range_m;
      if (std::abs(delta) <= tolerance) {
        observed[i] = kOnSurface;
        if (e.type == EndpointClass::kPhysical && e.physical_id > 0 &&
            static_cast<size_t>(e.physical_id) == physical_id) {
          identified[i] = true;
          ++identified_samples;
        } else if (e.type == EndpointClass::kPhysical && e.physical_id > 0) {
          // Another identified object on this surface: a positive observation of
          // a replacement. A background label is only a missing detection.
          foreign[i] = true;
        }
      } else if (facing) {
        observed[i] = delta > tolerance ? kSeenThrough : kInViewOnly;
      }
    }
    size_t seen_through_samples = 0;
    for (size_t i = 0; i < queries.size(); ++i) seen_through_samples += observed[i] == kSeenThrough;
    // The object is identified in place only if its own identity on the stored
    // surface outweighs the part of that surface seen through in the same frame.
    // A moved object that still overlaps its old site is identified on a strip and
    // seen through on the rest: that frame testifies against the old state.
    const bool object_identified = identified_samples >= kMinIdentifiedSamples &&
                                   identified_samples > seen_through_samples;
    if (object_identified) state->ever_identified = true;
    for (size_t i = 0; i < queries.size(); ++i) {
      if (observed[i] == kNone || observed[i] == kInViewOnly) continue;
      auto& sample = state->samples[queries[i].cell];
      if (observed[i] == kOnSurface) sample.last_on_surface = stamp;
      if (identified[i]) sample.last_identity = stamp;
      if (foreign[i]) sample.last_foreign = stamp;
      if (observed[i] == kSeenThrough) sample.last_seen_through = stamp;
      if (object_identified) {
        if (identified[i] && sample.tentative_hits < UINT16_MAX) ++sample.tentative_hits;
        if (observed[i] == kSeenThrough || foreign[i]) sample.tentative_veto = true;
      }
    }
    state->processed = stamp;
  }
  state->processed = std::max<TimeStamp>(state->processed, latest);

  // One look = this reconciliation round. Verdicts are the latest per sample in the round.
  size_t on_surface = 0, seen_through = 0, foreign_on_surface = 0, own_identity = 0, fresh = 0;
  std::vector<AbsenceCell> fresh_cells;
  for (const auto& query : queries) {
    const auto it = state->samples.find(query.cell);
    if (it == state->samples.end()) continue;
    const auto& sample = it->second;
    if (sample.last_identity >= round_start && sample.last_identity != 0) ++own_identity;
    if (sample.seen_through_while_identified) continue;
    if (!state->inherited && state->ever_identified &&
        static_cast<size_t>(sample.identity_hits) + sample.tentative_hits < kMinIdentityHits) continue;
    ++counts.reliable_samples;
    const TimeStamp last = std::max(sample.last_on_surface, sample.last_seen_through);
    if (last < round_start || last == 0) continue;
    ++counts.reliable_in_view;
    if (sample.last_seen_through > sample.last_on_surface) {
      ++seen_through;
    } else {
      ++on_surface;
      if (sample.last_foreign == sample.last_on_surface &&
          sample.last_identity < sample.last_on_surface) ++foreign_on_surface;
    }
    if (!sample.counted) { ++fresh; fresh_cells.push_back(query.cell); }
  }
  counts.reliable_seen_through = seen_through;
  const size_t verdicts = on_surface + seen_through;
  // A look needs enough judged samples to estimate a share at all.
  // Partial views are handled by the weight of the look, not by refusing it.
  const size_t needed = std::min<size_t>(kMinSamplesInView, counts.reliable_samples);
  const bool identified_in_place = own_identity >= kMinIdentifiedSamples && own_identity > seen_through;
  // Log density of a look under "the object stands here": Beta fitted to the
  // object's own history of this statistic (or the pooled within-object scatter
  // of the sensor while the object has too little history). Absent: uniform.
  const auto log_present = [](double f, double n, double sum, double sq, bool* ok) {
    *ok = n >= 1;
    if (!*ok) return 0.0;
    const double m = std::min(0.999, std::max(0.001, sum / n));
    // One-sided: only MORE seen-through than the object usually shows speaks for
    // absence. A look below the usual share is scored at the usual share.
    f = std::min(0.995, std::max(std::max(0.005, m), f));
    const double v = std::max(1e-4, sq / n - (sum / n) * (sum / n));
    const double c = std::max(2.0, m * (1 - m) / v - 1);
    const double a = m * c + 1e-3, b = (1 - m) * c + 1e-3;
    return std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + (a - 1) * std::log(f) +
           (b - 1) * std::log(1 - f);
  };
  // Moments used for the Beta: own robust scatter with >= 3 looks, else the pooled
  // within-object scatter of the sensor (from this or the previous session).
  // Own looks shrunk towards the population of objects: the pooled between-object
  // mean and scatter count as three pseudo-looks, so a handful of own looks cannot
  // pretend to an exactness the sensor does not have.
  const auto prior = [&](double& n, double& sum, double& sq, const std::vector<double>& own,
                         double pn, double ps, const std::vector<double>& pdev, double loaded_var) {
    double v0 = pdev.size() >= 3 ? robustVariance(pdev, -1.0) : loaded_var;
    double pool_mean = pn >= 3 ? ps / pn : 0.0;
    bool have_pool = pn >= 3 && v0 > 0;
    if (!have_pool) {
      // Nothing known about this sensor yet: an uninformative population
      // (share near zero, spread 0.3). Only a surface seen through almost
      // entirely can then speak, which is what a fully vacated site shows.
      pool_mean = 0.05; v0 = 0.09; have_pool = true;
    }
    const double own_n = n;
    const double own_m = own_n > 0 ? sum / own_n : 0.0;
    const double own_v = own_n >= 3 ? robustVariance(own, own_m) : 0.0;
    if (!have_pool) { n = own_n; sq = own_n * (own_m * own_m + own_v); return; }
    const double m0 = pool_mean, k = 3.0;
    const double use_n = own_n >= 3 ? own_n : 0.0;
    const double m = (use_n * own_m + k * m0) / (use_n + k);
    // Never narrower than the spread between objects: how an object looks from a
    // new viewpoint varies at least as much as objects vary among themselves.
    const double v = std::max(v0, (use_n * (own_v + (own_m - m) * (own_m - m)) +
                                   k * (v0 + (m0 - m) * (m0 - m))) / (use_n + k));
    n = 1; sum = m; sq = m * m + v;
  };
  if (verdicts >= needed) {
    const double f_geo = static_cast<double>(seen_through) / verdicts;
    const double f_lab = on_surface > 0 ? static_cast<double>(foreign_on_surface) / on_surface : 0.0;
    double gn = state->history_n, gs = state->history_sum, gq = state->history_sq;
    double ln = state->label_n, ls = state->label_sum, lq = state->label_sq;
    {
      std::lock_guard<std::mutex> lock(absence_mutex);
      prior(gn, gs, gq, state->geo_looks, pooled_n, pooled_sum, pooled_geo_dev, loaded_geo_var);
      prior(ln, ls, lq, state->label_looks, pooled_label_n, pooled_label_sum, pooled_label_dev, loaded_label_var);
    }
    bool ok_geo = false, ok_lab = false;
    const double lp_geo = log_present(f_geo, gn, gs, gq, &ok_geo);
    // The observed-absence decision (absence_llr) is the sole authority for "seen empty";
    // the ray counts only report. Label disagreement is not used as evidence: on a real segmenter it is a
    // missing detection far more often than a replacement (kept as a statistic only).
    const double lp_lab = on_surface >= kMinIdentifiedSamples ? log_present(f_lab, ln, ls, lq, &ok_lab) : 0.0;
    ok_lab = false;
    if (ok_geo || ok_lab) {
      // The look counts in proportion to the share of the reliable surface judged for
      // the first time in this accumulation: the total weight is at most the object.
      const double weight = std::min(1.0, static_cast<double>(fresh) /
                                              std::max<size_t>(1, counts.reliable_samples));
      state->cusum = std::max(0.0, state->cusum - weight * ((ok_geo ? lp_geo : 0.0) + (ok_lab ? lp_lab : 0.0)));
      for (const auto& cell : fresh_cells) state->samples[cell].counted = true;
      if (state->cusum == 0.0) for (auto& [c2, s2] : state->samples) { (void)c2; s2.counted = false; }
    }
    if (identified_in_place) {
      std::lock_guard<std::mutex> lock(absence_mutex);
      state->history_n += 1; state->history_sum += f_geo; state->history_sq += f_geo * f_geo;
      if (state->geo_looks.size() < 256) state->geo_looks.push_back(f_geo);
      if (state->history_n == 3 && pooled_geo_dev.size() < 4096) {
        pooled_geo_dev.push_back(state->history_sum / 3); pooled_n += 1; pooled_sum += state->history_sum / 3;
      }
      if (on_surface >= kMinIdentifiedSamples) {
        state->label_n += 1; state->label_sum += f_lab; state->label_sq += f_lab * f_lab;
        if (state->label_looks.size() < 256) state->label_looks.push_back(f_lab);
        if (state->label_n == 3 && pooled_label_dev.size() < 4096) {
          pooled_label_dev.push_back(state->label_sum / 3); pooled_label_n += 1; pooled_label_sum += state->label_sum / 3;
        }
      }
    }
  }
  // A look at the object standing in place lowers the accumulation through its own
  // likelihood (f near the history); no separate reset.
  if (state->cusum == 0.0) for (auto& [c2, s2] : state->samples) { (void)c2; s2.counted = false; }
  counts.absence_llr = static_cast<float>(state->cusum);
  VLOG(1) << "OBSERVED_ABSENCE inst=" << physical_id << " slot=" << state_slot
          << " queries=" << queries.size() << " reliable=" << counts.reliable_samples
          << " verdicts=" << verdicts << " needed=" << needed << " seen_through=" << seen_through
          << " fresh=" << fresh << " cusum=" << state->cusum;
  // Wald threshold for 1 % false-closure and 1 % missed-closure probability.
  counts.absence_coverage_sufficient = state->cusum > std::log(99.0);
  for (auto& [cell, sample] : state->samples) {
    (void)cell;
    if (identified_in_place) {
      sample.identity_hits = static_cast<uint16_t>(
          std::min<size_t>(UINT16_MAX, static_cast<size_t>(sample.identity_hits) + sample.tentative_hits));
      sample.seen_through_while_identified |= sample.tentative_veto;
    }
    sample.tentative_hits = 0;
    sample.tentative_veto = false;
  }
  if (counts.absence_coverage_sufficient) state->cusum = 0;  // the state ends; a successor starts clean
}

RayVerificator::SurfaceEvidenceCounts RayVerificator::countCurrentPhysicalSurface(
    size_t physical_id, const spark_dsg::Mesh& mesh, const BoundingBox& bbox,
    const PhysicalEvidenceSnapshot& snapshot, float map_resolution,
    uint64_t last_support, uint64_t latest, bool* projected, const int state_slot,
    const uint64_t state_birth) const {
  if (projected) *projected = false;
  // Also prevents unsigned overflow and invalid inclusive intervals.
  if (last_support >= latest) {
    SurfaceEvidenceCounts none;
    applyObservedAbsence(physical_id, mesh, bbox, snapshot, latest, latest, none, state_slot, state_birth);
    none.absence_coverage_sufficient = false;
    return none;
  }
  const uint64_t earliest = last_support + 1;
  auto counts = countPhysicalSurface(physical_id, mesh, bbox, snapshot, earliest, latest);
  if (!snapshot || snapshot->numFrames() == 0) {
    // No pixel evidence at all (mesh-ray proxy only, as in offline tools and
    // unit fixtures): the proxy counts decide as before.
    counts.absence_coverage_sufficient = counts.contradiction_rays > counts.support_rays;
    return counts;
  }
  // A sparse mesh-ray subset can reverse the decision (Synthetic I108:
  // indexed support 3 / absence 4, measured pixels support 86 / absence 36).
  // Any proposed deletion must therefore be checked against the actual
  // sensor evidence; a mesh proxy alone cannot authorize disappearance.
  const bool proposed_absence = counts.contradiction_rays > counts.support_rays;
  if (proposed_absence || (counts.support_rays == 0 && counts.contradiction_rays == 0)) {
    auto measured = countProjectedPhysicalSurface(
        physical_id, mesh, bbox, snapshot, map_resolution, earliest, latest);
    if (proposed_absence || measured.support_rays || measured.contradiction_rays) {
      if (projected) *projected = true;
      applyObservedAbsence(physical_id, mesh, bbox, snapshot, earliest, latest, measured, state_slot, state_birth);
      return measured;
    }
  }
  applyObservedAbsence(physical_id, mesh, bbox, snapshot, earliest, latest, counts, state_slot, state_birth);
  return counts;
}

}  // namespace khronos
