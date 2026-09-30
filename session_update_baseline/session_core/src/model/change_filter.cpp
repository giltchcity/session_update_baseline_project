#include "session_core/model/change_filter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace khronos::model {
namespace {
constexpr double kLogRange = 700.0;  // ln of the largest representable odds
}  // namespace

ChangeFilter::ChangeFilter(double gap_odds, uint64_t gap_begin, uint64_t start)
    : odds_(gap_odds), has_gap_(gap_odds > 0.0), gap_begin_(gap_begin), gap_end_(start) {
  if (!(gap_odds >= 0.0) || !std::isfinite(gap_odds)) throw std::invalid_argument("Invalid gap odds");
  gap_q_ = gap_odds / (1.0 + gap_odds);
}

double ChangeFilter::update(uint64_t begin, uint64_t end, double q, double log_lr) {
  if (!(q >= 0.0 && q < 1.0) || !std::isfinite(log_lr) || end < begin) {
    throw std::invalid_argument("Invalid round for the change filter");
  }
  // (5r), evaluated in the log domain so that an overwhelming likelihood ratio stays finite (the
  // cap is the range of a double, far beyond any threshold).
  const double numerator = odds_ + q;
  odds_ = numerator > 0.0
      ? std::exp(std::min(std::log(numerator) - std::log1p(-q) + log_lr, kLogRange))
      : 0.0;
  rounds_.push_back({begin, end, q, log_lr});
  return odds_;
}

void ChangeFilter::anchorAtConfirmation() {
  rounds_.clear();
  has_gap_ = false;
}

ChangeFilter::Interval ChangeFilter::changeInterval() const {
  // Cells of the change time: the gap (inherited placements) and every round since the anchor.
  struct Cell {
    uint64_t left, right;
    double log_weight;
  };
  std::vector<Cell> cells;
  // log pi_k + sum_{i >= k} ln LR_i, (5t). The probability of not having changed before the
  // cell is the product of (1 - q_j) of the cells before it (and (1 - q^gap)).
  std::vector<double> suffix(rounds_.size() + 1, 0.0);
  for (size_t i = rounds_.size(); i-- > 0;) suffix[i] = suffix[i + 1] + rounds_[i].log_lr;
  double log_survival = 0.0;
  if (has_gap_) {
    cells.push_back({gap_begin_, gap_end_, std::log(gap_q_) + suffix[0]});
    log_survival = std::log1p(-gap_q_);
  }
  for (size_t k = 0; k < rounds_.size(); ++k) {
    const auto& round = rounds_[k];
    if (round.q > 0.0) {
      cells.push_back({round.begin, round.end, log_survival + std::log(round.q) + suffix[k]});
    }
    log_survival += std::log1p(-round.q);
  }
  if (cells.empty()) {
    return has_gap_ ? Interval{gap_begin_, gap_end_}
                    : (rounds_.empty() ? Interval{} : Interval{rounds_.front().begin, rounds_.back().end});
  }
  std::vector<double> log_weights;
  log_weights.reserve(cells.size());
  for (const auto& cell : cells) log_weights.push_back(cell.log_weight);
  const double normaliser = logSumExp(log_weights);
  std::vector<double> mass(cells.size());
  for (size_t i = 0; i < cells.size(); ++i) mass[i] = std::exp(log_weights[i] - normaliser);

  // Shortest run of adjacent cells holding at least 1 - alpha of the mass.
  Interval best{cells.front().left, cells.back().right};
  uint64_t best_length = best.right - best.left;
  for (size_t i = 0; i < cells.size(); ++i) {
    double covered = 0.0;
    for (size_t j = i; j < cells.size(); ++j) {
      covered += mass[j];
      if (covered >= 1.0 - kAlpha) {
        const uint64_t length = cells[j].right - cells[i].left;
        if (length < best_length) {
          best_length = length;
          best = {cells[i].left, cells[j].right};
        }
        break;
      }
    }
  }
  return best;
}

nlohmann::json ChangeFilter::toJson() const {
  nlohmann::json rounds = nlohmann::json::array();
  for (const auto& round : rounds_) {
    rounds.push_back(nlohmann::json::array({round.begin, round.end, round.q, round.log_lr}));
  }
  return nlohmann::json{{"odds", odds_},     {"has_gap", has_gap_},       {"gap_q", gap_q_},
                        {"gap_begin", gap_begin_}, {"gap_end", gap_end_}, {"rounds", std::move(rounds)}};
}

ChangeFilter ChangeFilter::fromJson(const nlohmann::json& value) {
  ChangeFilter filter;
  filter.odds_ = value.at("odds").get<double>();
  filter.has_gap_ = value.at("has_gap").get<bool>();
  filter.gap_q_ = value.at("gap_q").get<double>();
  filter.gap_begin_ = value.at("gap_begin").get<uint64_t>();
  filter.gap_end_ = value.at("gap_end").get<uint64_t>();
  if (!(filter.odds_ >= 0.0) || !(filter.gap_q_ >= 0.0 && filter.gap_q_ < 1.0)) {
    throw std::invalid_argument("Invalid change filter state");
  }
  for (const auto& item : value.at("rounds")) {
    Round round{item.at(0).get<uint64_t>(), item.at(1).get<uint64_t>(), item.at(2).get<double>(),
                item.at(3).get<double>()};
    if (!(round.q >= 0.0 && round.q < 1.0) || round.end < round.begin || !std::isfinite(round.log_lr)) {
      throw std::invalid_argument("Invalid change filter round");
    }
    filter.rounds_.push_back(round);
  }
  return filter;
}

}  // namespace khronos::model
