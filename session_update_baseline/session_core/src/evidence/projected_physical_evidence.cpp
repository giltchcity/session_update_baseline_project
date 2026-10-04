#include "khronos/backend/change_detection/ray_verificator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <sstream>
#include <map>
#include <memory>
#include <mutex>
#include <filesystem>
#include <fstream>
#include <set>
#include <tuple>

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
  // README M1 reliability posterior: outcomes in frames where the object is identified in
  // place (own label / other label / seen through), committed like the hits above.
  uint16_t other_obs = 0, through_obs = 0, tentative_other = 0, tentative_through = 0;
  // Diagnostics only (exported per cell at session end): outcomes in frames where the
  // object is identified in place.
  uint16_t diag_own = 0, diag_other = 0, diag_through = 0;
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
  const RayVerificator* likelihood_owner = nullptr;
  TimeStamp likelihood_stamp = 0;
  PhysicalAbsenceLookLikelihood likelihood;
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

// README M1 reliability (label part): a cell never seen through while the object stood in place
// is a surface of this object (R) or of something else carrying its label at times (P). In
// identified frames each class returns own label / other label with rates theta_R / theta_P;
// pi = P(R). Maximum likelihood by EM over the committed counts of every unvetoed cell of the
// process, once per reconciliation round; such a cell votes iff P(R | counts) >= 1/2. The
// one-look veto stays: it selects the never-seen-through subset whose in-place share the
// sequential test models (P22 trial 1). No counts anywhere: the classes are not identified.
struct CellModel {
  bool identified = false;
  double log_pi = std::log(0.5), log_1mpi = std::log(0.5);
  std::array<double, 3> log_r{{0.0, 0.0, 0.0}}, log_p{{0.0, 0.0, 0.0}};
  TimeStamp stamp = 0;
};
CellModel cell_model;

double cellLogOdds(const CellModel& m, double own, double other, double through) {
  if (!m.identified) return 0.0;
  return m.log_pi - m.log_1mpi + own * (m.log_r[0] - m.log_p[0]) +
         other * (m.log_r[1] - m.log_p[1]) + through * (m.log_r[2] - m.log_p[2]);
}

// Caller holds absence_mutex.
void estimateCellModel(const TimeStamp stamp) {
  if (cell_model.stamp == stamp) return;
  cell_model.stamp = stamp;
  std::vector<std::array<double, 3>> cells;
  for (const auto& [key, st] : absence_states) {
    (void)key;
    for (const auto& [cell, sm] : st->samples) {
      (void)cell;
      if (sm.seen_through_while_identified) continue;
      const double o = sm.identity_hits, x = sm.other_obs;
      if (o + x > 0) cells.push_back({o, x, 0.0});
    }
  }
  if (cells.empty()) return;
  // Initial split from the data: cells whose own-label share is at least one half.
  std::array<double, 3> sr{{0, 0, 0}}, sp{{0, 0, 0}};
  double nr = 0;
  for (const auto& c : cells) {
    const bool r = c[0] >= 0.5 * (c[0] + c[1]);
    for (int j = 0; j < 3; ++j) (r ? sr : sp)[j] += c[j];
    nr += r;
  }
  if (nr == 0 || nr == cells.size()) return;  // one class only: not identified
  double pi = nr / cells.size();
  std::array<double, 3> tr, tp;
  const auto norm = [](const std::array<double, 3>& a, std::array<double, 3>& out) {
    const double tot = a[0] + a[1];
    // Jeffreys pseudo-count 1/2 per outcome keeps every rate inside (0, 1).
    for (int j = 0; j < 2; ++j) out[j] = (a[j] + 0.5) / (tot + 1.0);
    out[2] = 1.0;
  };
  norm(sr, tr); norm(sp, tp);
  for (int it = 0; it < 1000; ++it) {
    std::array<double, 3> ar{{0, 0, 0}}, ap{{0, 0, 0}};
    double w_sum = 0;
    const double lr0 = std::log(tr[0]), lr1 = std::log(tr[1]), lr2 = std::log(tr[2]);
    const double lp0 = std::log(tp[0]), lp1 = std::log(tp[1]), lp2 = std::log(tp[2]);
    for (const auto& c : cells) {
      const double a = std::log(pi) + c[0] * lr0 + c[1] * lr1 + c[2] * lr2;
      const double b = std::log1p(-pi) + c[0] * lp0 + c[1] * lp1 + c[2] * lp2;
      const double w = 1.0 / (1.0 + std::exp(b - a));
      w_sum += w;
      for (int j = 0; j < 3; ++j) { ar[j] += w * c[j]; ap[j] += (1 - w) * c[j]; }
    }
    const double new_pi = std::min(std::max(w_sum / cells.size(), 0.5 / (cells.size() + 1)),
                                   1.0 - 0.5 / (cells.size() + 1));
    norm(ar, tr); norm(ap, tp);
    const bool done = std::abs(new_pi - pi) < 1e-12;
    pi = new_pi;
    if (done) break;
  }
  if (tr[0] < tp[0]) { std::swap(tr, tp); pi = 1.0 - pi; }  // R is the class with the higher own rate
  cell_model.identified = true;
  cell_model.log_pi = std::log(pi);
  cell_model.log_1mpi = std::log1p(-pi);
  for (int j = 0; j < 3; ++j) { cell_model.log_r[j] = std::log(tr[j]); cell_model.log_p[j] = std::log(tp[j]); }
  LOG(INFO) << "CELL_MODEL stamp=" << stamp << " cells=" << cells.size() << " pi=" << pi
            << " theta_r=" << tr[0] << "," << tr[1] << " theta_p=" << tp[0] << "," << tp[1];
}

constexpr size_t kMinIdentifiedSamples = 3;
constexpr uint16_t kMinIdentityHits = 3;  // diagnostics classes only (ABSENCE_DIAG)
constexpr size_t kMinSamplesInView = 30;
constexpr size_t kMaxAbsenceSamples = 1500;

}  // namespace

PhysicalAbsenceLookLikelihood physicalAbsenceLookLikelihood(
    const RayVerificator* owner, const size_t physical_id, const int state_slot,
    const TimeStamp stamp) {
  std::lock_guard<std::mutex> lock(absence_mutex);
  for (const auto& [key, state] : absence_states) {
    if (std::get<1>(key) == physical_id && std::get<2>(key) == state_slot &&
        state->likelihood_owner == owner && state->likelihood_stamp == stamp) {
      return state->likelihood;
    }
  }
  return {};
}

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
  // Diagnostics: per-cell outcome counts for offline estimation of label and phantom rates.
  std::ofstream cells((std::filesystem::path(path).parent_path() / "absence_cells.csv").string());
  if (cells) {
    cells << "physical_id,slot,inherited,ever_identified,identity_hits,vetoed,own,other,through\n";
    for (const auto& [key, st] : absence_states) {
      for (const auto& [cell, sm] : st->samples) {
        (void)cell;
        if (!sm.diag_own && !sm.diag_other && !sm.diag_through && !sm.identity_hits) continue;
        cells << std::get<1>(key) << ',' << std::get<2>(key) << ',' << st->inherited << ',' << st->ever_identified
              << ',' << sm.identity_hits << ',' << sm.seen_through_while_identified << ',' << sm.diag_own << ','
              << sm.diag_other << ',' << sm.diag_through << '\n';
      }
    }
  }
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
  enum : int8_t { kNone = -2, kInViewOnly = -1, kSeenThrough = 0, kOnSurface = 1 };
  std::vector<int8_t> observed(queries.size());
  std::vector<bool> identified(queries.size());
  std::vector<bool> foreign(queries.size());
  // Every surface cell is part of the object whether or not this round saw it:
  // the share a look judged is measured against the whole reliable surface.
  for (const auto& query : queries) state->samples.try_emplace(query.cell);
  std::array<size_t, 10> angle_on{}, angle_beyond{};
  std::array<size_t, 9> cell_outcome{};  // [reliable|thin|vetoed] x [own label|other label|seen through]
  std::vector<float> incidence(queries.size(), -1.f);
  std::vector<int8_t> raw(queries.size(), 0);
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
      incidence[i] = queries[i].has_normal ? std::abs(queries[i].normal.dot(p.view_direction_world)) : -1.f;
      raw[i] = std::abs(delta) <= tolerance ? 1 : delta > tolerance ? 2 : 0;
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
    if (object_identified) {
      for (size_t i = 0; i < queries.size(); ++i) {
        if (raw[i] == 0) continue;
        const auto it = state->samples.find(queries[i].cell);
        if (it == state->samples.end()) continue;
        const size_t cls = it->second.seen_through_while_identified ? 2
            : (!state->inherited && static_cast<size_t>(it->second.identity_hits) < kMinIdentityHits) ? 1 : 0;
        if (raw[i] == 2 && incidence[i] >= 0.f && incidence[i] < min_cos) continue;  // grazing: counted below only
        ++cell_outcome[3 * cls + (raw[i] == 2 ? 2 : identified[i] ? 0 : 1)];
        auto& dc = raw[i] == 2 ? it->second.diag_through : identified[i] ? it->second.diag_own : it->second.diag_other;
        if (dc < UINT16_MAX) ++dc;
      }
      for (size_t i = 0; i < queries.size(); ++i) {
        if (raw[i] == 0 || incidence[i] < 0.f) continue;
        const auto it = state->samples.find(queries[i].cell);
        if (it == state->samples.end() || it->second.seen_through_while_identified) continue;
        if (!state->inherited && static_cast<size_t>(it->second.identity_hits) < kMinIdentityHits) continue;
        const size_t bin = std::min<size_t>(9, static_cast<size_t>(incidence[i] * 10.f));
        ++(raw[i] == 1 ? angle_on : angle_beyond)[bin];
      }
    }
    std::fill(raw.begin(), raw.end(), 0);
    std::fill(incidence.begin(), incidence.end(), -1.f);
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
        if (observed[i] == kSeenThrough) {
          if (sample.tentative_through < UINT16_MAX) ++sample.tentative_through;
        } else if (!identified[i] && sample.tentative_other < UINT16_MAX) {
          ++sample.tentative_other;
        }
      }
    }
    state->processed = stamp;
  }
  state->processed = std::max<TimeStamp>(state->processed, latest);

  {
    std::lock_guard<std::mutex> lock(absence_mutex);
    estimateCellModel(latest);
  }
  // One look = this reconciliation round. Verdicts are the latest per sample in the round.
  size_t on_surface = 0, seen_through = 0, foreign_on_surface = 0, own_identity = 0, fresh = 0;
  std::vector<AbsenceCell> fresh_cells;
  for (const auto& query : queries) {
    const auto it = state->samples.find(query.cell);
    if (it == state->samples.end()) continue;
    const auto& sample = it->second;
    if (sample.last_identity >= round_start && sample.last_identity != 0) ++own_identity;
    if (sample.seen_through_while_identified) continue;
    // Label part (README M1): P(surface of this object | own / other label counts) >= 1/2.
    // pi is estimated on cells observed in identified frames; a cell of an identified state
    // that never was observed there is outside that population and has no evidence.
    // Inherited and never-identified states have no such record: all their cells vote.
    if (!state->inherited && state->ever_identified) {
      const double own = static_cast<double>(sample.identity_hits) + sample.tentative_hits;
      if (own + sample.other_obs == 0.0 ||
          cellLogOdds(cell_model, own, sample.other_obs, 0.0) < 0.0) continue;
    }
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
  const auto log_present = [](double f, double n, double sum, double sq, bool* ok,
                              double* shape_a, double* shape_b) {
    *ok = n >= 1;
    if (!*ok) return 0.0;
    const double m = std::min(0.999, std::max(0.001, sum / n));
    // One-sided: only MORE seen-through than the object usually shows speaks for
    // absence. A look below the usual share is scored at the usual share.
    f = std::min(0.995, std::max(std::max(0.005, m), f));
    const double v = std::max(1e-4, sq / n - (sum / n) * (sum / n));
    const double c = std::max(2.0, m * (1 - m) / v - 1);
    const double a = m * c + 1e-3, b = (1 - m) * c + 1e-3;
    if (shape_a) *shape_a = a;
    if (shape_b) *shape_b = b;
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
  // A round in which no reliable sample was judged says nothing about presence or
  // absence: it is neither scored nor learned from. (With reliable_samples = 0, `needed`
  // is 0 and the share would be 0/0; one such round turned the pooled prior into NaN for
  // the rest of the session and was exported to the next one.)
  double diag_a = -1.0, diag_b = -1.0, diag_weight = 0.0, diag_increment = 0.0;
  bool diag_scored = false;
  if (verdicts > 0 && verdicts >= needed) {
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
    double shape_a = 0.0, shape_b = 0.0;
    const double lp_geo = log_present(f_geo, gn, gs, gq, &ok_geo, &shape_a, &shape_b);
    // The observed-absence decision (absence_llr) is the sole authority for "seen empty";
    // the ray counts only report. Label disagreement is not used as evidence: on a real segmenter it is a
    // missing detection far more often than a replacement (kept as a statistic only).
    const double lp_lab = on_surface >= kMinIdentifiedSamples ? log_present(f_lab, ln, ls, lq, &ok_lab, nullptr, nullptr) : 0.0;
    ok_lab = false;
    if (ok_geo || ok_lab) {
      // The look counts in proportion to the share of the reliable surface judged for
      // the first time in this accumulation: the total weight is at most the object.
      const double weight = std::min(1.0, static_cast<double>(fresh) /
                                              std::max<size_t>(1, counts.reliable_samples));
      // A view for S10; the original density/CUSUM calculation below is unchanged.
      if (ok_geo && weight > 0.0) {
        const double n = static_cast<double>(verdicts);
        const double k = static_cast<double>(seen_through);
        const double log_present_count =
            std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0) +
            std::lgamma(k + shape_a) + std::lgamma(n - k + shape_b) -
            std::lgamma(n + shape_a + shape_b) - std::lgamma(shape_a) -
            std::lgamma(shape_b) + std::lgamma(shape_a + shape_b);
        double log_ratio = -std::log1p(n) - log_present_count;
        if (f_geo <= shape_a / (shape_a + shape_b)) log_ratio = std::min(0.0, log_ratio);
        std::lock_guard<std::mutex> lock(absence_mutex);
        state->likelihood = {weight * log_ratio, true, true};
      }
      diag_scored = true; diag_a = shape_a; diag_b = shape_b; diag_weight = weight;
      diag_increment = -weight * ((ok_geo ? lp_geo : 0.0) + (ok_lab ? lp_lab : 0.0));
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
  if (verdicts > 0) {
    LOG(INFO) << "ABSENCE_LOOK inst=" << physical_id << " slot=" << state_slot << " stamp=" << latest
              << " k=" << seen_through << " n=" << verdicts << " reliable=" << counts.reliable_samples
              << " fresh=" << fresh << " needed=" << needed << " identified=" << identified_in_place
              << " inherited=" << state->inherited << " scored=" << diag_scored << " a=" << diag_a
              << " b=" << diag_b << " weight=" << diag_weight << " increment=" << diag_increment
              << " cusum=" << state->cusum;
  }
  // A look at the object standing in place lowers the accumulation through its own
  // likelihood (f near the history); no separate reset.
  if (state->cusum == 0.0) for (auto& [c2, s2] : state->samples) { (void)c2; s2.counted = false; }
  counts.absence_llr = static_cast<float>(state->cusum);
  if (std::accumulate(angle_on.begin(), angle_on.end(), size_t{0}) +
      std::accumulate(angle_beyond.begin(), angle_beyond.end(), size_t{0}) > 0) {
    std::ostringstream on, beyond, cells;
    for (size_t b = 0; b < 10; ++b) { on << (b ? "," : "") << angle_on[b]; beyond << (b ? "," : "") << angle_beyond[b]; }
    for (size_t b = 0; b < 9; ++b) cells << (b ? "," : "") << cell_outcome[b];
    LOG(INFO) << "ABSENCE_DIAG inst=" << physical_id << " slot=" << state_slot << " stamp=" << latest
              << " inherited=" << state->inherited << " angle_on=" << on.str() << " angle_beyond=" << beyond.str()
              << " cells=" << cells.str();
  }
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
      sample.other_obs = static_cast<uint16_t>(std::min<size_t>(UINT16_MAX, size_t{sample.other_obs} + sample.tentative_other));
      sample.through_obs = static_cast<uint16_t>(std::min<size_t>(UINT16_MAX, size_t{sample.through_obs} + sample.tentative_through));
    }
    sample.tentative_hits = 0;
    sample.tentative_veto = false;
    sample.tentative_other = 0;
    sample.tentative_through = 0;
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
