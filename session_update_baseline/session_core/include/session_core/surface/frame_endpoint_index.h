#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace khronos {

// Exact acceleration of README (12)'s existential measured-endpoint query.
// Image rectangles define a fixed tree; endpoint bounds are rebuilt per frame.
// Bounds only reject empty intersections. Leaves retain the original predicate.
class FrameEndpointIndex {
 public:
  struct Rectangle { int x0, y0, x1, y1; };

  FrameEndpointIndex(int width, int height, double fx, double fy, double cx, double cy)
      : width_(width) {
    if (width <= 0 || height <= 0) throw std::invalid_argument("Empty endpoint image");
    directions_.resize(size_t(width) * height);
    points_.resize(directions_.size());
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        directions_[size_t(y) * width + x] =
            Eigen::Vector3d((double(x) - cx) / fx, (double(y) - cy) / fy, 1.).normalized();
      }
    }
    build({0, 0, width - 1, height - 1});
  }

  void bind(const std::vector<uint16_t>& ranges) {
    if (ranges.size() != points_.size()) throw std::invalid_argument("Endpoint image shape");
    ranges_ = &ranges;
    update(0);
  }

  template <typename Accept>
  bool any(const Eigen::Vector3d& centre, double radius_sq, Rectangle area,
           int preferred_x, int preferred_y, Accept&& accept) const {
    const auto hit = [&](int x, int y) {
      const size_t pixel = size_t(y) * width_ + x;
      return (*ranges_)[pixel] && (points_[pixel] - centre).squaredNorm() <= radius_sq &&
             accept(x, y);
    };
    if (preferred_x >= area.x0 && preferred_x <= area.x1 &&
        preferred_y >= area.y0 && preferred_y <= area.y1 && hit(preferred_x, preferred_y))
      return true;
    return search(0, centre, radius_sq, area, hit);
  }

 private:
  struct Node {
    Rectangle area;
    Eigen::AlignedBox3d bounds;
    int left = -1, right = -1;
  };
  static bool overlaps(Rectangle a, Rectangle b) {
    return a.x0 <= b.x1 && b.x0 <= a.x1 && a.y0 <= b.y1 && b.y0 <= a.y1;
  }
  int build(Rectangle area) {
    const int index = static_cast<int>(nodes_.size());
    nodes_.push_back({area, Eigen::AlignedBox3d(), -1, -1});
    const int nx = area.x1 - area.x0 + 1, ny = area.y1 - area.y0 + 1;
    // Leaf capacity is an indexing layout, with every pixel retained.
    if (nx * ny <= 16) return index;
    auto a = area, b = area;
    if (nx >= ny) { a.x1 = (area.x0 + area.x1) / 2; b.x0 = a.x1 + 1; }
    else { a.y1 = (area.y0 + area.y1) / 2; b.y0 = a.y1 + 1; }
    const int left = build(a), right = build(b);
    nodes_[index].left = left;
    nodes_[index].right = right;
    return index;
  }
  void update(int index) {
    auto& node = nodes_[index];
    node.bounds.setEmpty();
    if (node.left >= 0) {
      update(node.left); update(node.right);
      if (!nodes_[node.left].bounds.isEmpty()) node.bounds.extend(nodes_[node.left].bounds);
      if (!nodes_[node.right].bounds.isEmpty()) node.bounds.extend(nodes_[node.right].bounds);
      return;
    }
    for (int y = node.area.y0; y <= node.area.y1; ++y) {
      for (int x = node.area.x0; x <= node.area.x1; ++x) {
        const size_t pixel = size_t(y) * width_ + x;
        if (!(*ranges_)[pixel]) continue;
        points_[pixel] = (double((*ranges_)[pixel]) * 1e-3) * directions_[pixel];
        node.bounds.extend(points_[pixel]);
      }
    }
  }
  template <typename Hit>
  bool search(int index, const Eigen::Vector3d& centre, double radius_sq,
              Rectangle area, const Hit& hit) const {
    const auto& node = nodes_[index];
    if (!overlaps(node.area, area) || node.bounds.isEmpty()) return false;
    const Eigen::Vector3d nearest = centre.cwiseMax(node.bounds.min()).cwiseMin(node.bounds.max());
    if ((nearest - centre).squaredNorm() > radius_sq) return false;
    if (node.left >= 0)
      return search(node.left, centre, radius_sq, area, hit) ||
             search(node.right, centre, radius_sq, area, hit);
    for (int y = std::max(area.y0, node.area.y0); y <= std::min(area.y1, node.area.y1); ++y)
      for (int x = std::max(area.x0, node.area.x0); x <= std::min(area.x1, node.area.x1); ++x)
        if (hit(x, y)) return true;
    return false;
  }
  int width_;
  std::vector<Eigen::Vector3d> directions_, points_;
  std::vector<Node> nodes_;
  const std::vector<uint16_t>* ranges_ = nullptr;
};

}  // namespace khronos
