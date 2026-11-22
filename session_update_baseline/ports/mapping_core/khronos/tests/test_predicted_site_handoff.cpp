#include "khronos/active_window/predicted_site_handoff.h"

#include <cstdlib>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct View {
  cv::Mat depth{cv::Size(26, 1), CV_32FC1, cv::Scalar(1)};
  cv::Mat raw{cv::Size(26, 1), CV_32SC1, cv::Scalar(0)};
  cv::Mat ids{cv::Size(26, 1), CV_32SC1, cv::Scalar(0)};
  cv::Mat semantics{cv::Size(26, 1), CV_32SC1, cv::Scalar(0)};

  void see(int u, int id, int label) {
    ids.at<int>(0, u) = id;
    raw.at<int>(0, u) = semantics.at<int>(0, u) = label;
  }
  void apply(khronos::PredictedSiteHandoff& site, int stamp) {
    khronos::OnlineFront::Intrinsics camera;
    camera.fx = camera.fy = 100;
    site.applyFrame(depth, raw, ids, semantics, camera, Eigen::Isometry3d::Identity(), stamp);
  }
};

}  // namespace

int main(int argc, char** argv) {
  require(argc == 2, "Missing handoff fixture");
  khronos::PredictedSiteHandoff sites(argv[1]);
  require(sites.sourceStamp() == 10 && sites.size() == 1, "Invalid input metadata");
  sites.validateSeed(10, {{3, 8}, {9, 14}});
  bool rejected = false;
  try {
    sites.validateSeed(10, {{3, 8}});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "Source with no host was accepted");
  require(!sites.keepInheritedVertex(3, 8, {0, 0, 1}), "Duplicated source stayed active");
  require(!sites.keepInheritedVertex(3, 8, {.30, 0, 1}), "Unresolved source stayed active");
  require(sites.keepInheritedVertex(3, 8, {.20, 0, 1}), "Independent source disappeared");
  require(sites.keepInheritedVertex(3, 8, {.65, 0, 1}), "New location rejected");
  require(sites.keepInheritedVertex(9, 14, {0, 0, 1}), "Host geometry edited");
  View supported;
  supported.see(0, 3, 8);
  supported.see(1, 9, 14);
  supported.see(20, 3, 8);
  supported.apply(sites, 11);
  require(supported.ids.at<int>(0, 0) == 9 && supported.semantics.at<int>(0, 0) == 14,
          "Duplicate with a measured host kept separate");
  require(supported.ids.at<int>(0, 20) == 3, "Independently supported part lost");
  View regained;
  regained.see(0, 3, 8);
  regained.see(1, 9, 14);
  regained.apply(sites, 12);
  require(regained.ids.at<int>(0, 0) == 3, "Raw source evidence could not reverse the assignment");

  khronos::PredictedSiteHandoff fresh(argv[1]);
  View distant;
  distant.see(0, 3, 8);
  distant.see(20, 9, 14);
  distant.apply(fresh, 13);
  require(distant.ids.at<int>(0, 0) == 3, "Distant host took a local measurement");
  khronos::PredictedSiteHandoff another(argv[1]);
  View wrong_class;
  wrong_class.see(0, 3, 8);
  wrong_class.see(1, 9, 8);
  wrong_class.apply(another, 14);
  require(wrong_class.ids.at<int>(0, 0) == 3, "An unsupported host label established identity");

  setenv("KHRONOS_PREDICTED_SITE_HANDOFF", "/missing/handoff", 1);
  khronos::OnlineFront::Config config;
  config.mode = khronos::OnlineFront::Mode::SEMANTIC_MERGE;
  khronos::OnlineFront annotated(config, {});
  return 0;
}
