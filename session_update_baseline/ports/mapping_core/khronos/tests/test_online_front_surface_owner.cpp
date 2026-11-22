// The online front's surface ownership (PREDICTED): a confirmed object's surface named another class is not a new object, in the
// visit it was confirmed in and in a later one; a real object that appears on it in a later visit is one, also when it is
// first named as the surface under it; and it is one again where it is moved to, while its old place shows the surface.
// A first name seen for a few looks gives way to the object its surface is seen as for longer.
#include "khronos/active_window/online_front.h"

#include <unistd.h>

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
}  // namespace

int main() {
  constexpr int bed = 7, bag = 115, clothes = 92;
  const cv::Rect left(4, 4, 12, 12), right(48, 4, 12, 12);
  const khronos::OnlineFront::Intrinsics camera{64, 24, 150, 150, 32, 12, 0};
  khronos::OnlineFront::Config config;
  config.mode = khronos::OnlineFront::Mode::PREDICTED;
  config.object_semantics = {bed, bag, clothes};
  config.state_path = std::string("/tmp/test_online_front_surface_owner_") + std::to_string(::getpid()) + ".state";
  std::remove(config.state_path.c_str());
  int64_t stamp = 1000000000;
  // the bed at 2 m; `bump` raises a rectangle by 8 cm (an object lying on the bed)
  auto look = [&](khronos::OnlineFront& front, int left_class, int right_class, bool bump_left, bool bump_right) {
    cv::Mat depth(24, 64, CV_32FC1, cv::Scalar(2.0f));
    if (bump_left) depth(left).setTo(1.92f);
    if (bump_right) depth(right).setTo(1.92f);
    cv::Mat labels(depth.size(), CV_32SC1, cv::Scalar(bed));
    labels(left).setTo(left_class);
    labels(right).setTo(right_class);
    stamp += 33333333;
    return front.process(depth, labels, Eigen::Isometry3d::Identity(), stamp);
  };
  int bed_id = 0;
  {  // first visit
    khronos::OnlineFront front(config, camera);
    look(front, bed, bed, false, false);
    bed_id = look(front, bed, bed, false, false).at<int>(8, 8);
    require(bed_id > 0, "bed never confirmed");
    for (int i = 0; i < 12; ++i)
      require(look(front, clothes, bed, false, false).at<int>(8, 8) == 0, "the bed named clothes became an object");
    require(look(front, bed, bed, false, false).at<int>(8, 8) == bed_id, "the bed lost its id");
  }
  {  // a later visit: the bed is inherited
    khronos::OnlineFront front(config, camera);
    require(look(front, bed, bed, false, false).at<int>(8, 30) == bed_id, "the inherited bed lost its id");
    for (int i = 0; i < 12; ++i)
      require(look(front, clothes, bed, false, false).at<int>(8, 8) == 0, "the inherited bed named clothes became an object");
    for (int i = 0; i < 3; ++i) look(front, bed, bed, true, false);  // a bag on the bed, first named bed
    look(front, bag, bed, true, false);
    const int bag_id = look(front, bag, bed, true, false).at<int>(8, 8);
    require(bag_id > 0 && bag_id != bed_id, "the bag on the inherited bed was not confirmed");
    look(front, bed, bag, false, true);  // the bag moved to the right; its old place shows the bed again
    const auto moved = look(front, bed, bag, false, true);
    require(moved.at<int>(8, 50) > 0 && moved.at<int>(8, 50) != bed_id, "the moved bag is missing at its new place");
    require(moved.at<int>(8, 8) == bed_id, "the bed is missing where the bag was");
  }
  std::remove(config.state_path.c_str());
  std::remove((config.state_path + ".catalog.json").c_str());
  {  // a bag first named clothes for a few looks, then bag for longer: the bag is confirmed and takes its surface
    khronos::OnlineFront front(config, camera);
    for (int i = 0; i < 3; ++i) look(front, bed, clothes, false, true);
    const int clothes_id = look(front, bed, clothes, false, true).at<int>(8, 50);
    require(clothes_id > 0, "the first name never confirmed");
    int now = 0;
    for (int i = 0; i < 12; ++i) now = look(front, bed, bag, false, true).at<int>(8, 50);
    require(now > 0 && now != clothes_id, "a first name seen briefly kept the surface of the object seen longer");
  }
  std::remove(config.state_path.c_str());
  std::remove((config.state_path + ".catalog.json").c_str());
  {  // SEMANTIC_MERGE (annotated semantics) is the Python front: none of the rules for predicted semantics applies, the
     // bed named clothes over and over is an object there
    khronos::OnlineFront::Config original = config;
    original.mode = khronos::OnlineFront::Mode::SEMANTIC_MERGE;
    khronos::OnlineFront front(original, camera);
    for (int i = 0; i < 20; ++i) look(front, bed, bed, false, false);
    int patch = 0;
    for (int i = 0; i < 12; ++i) patch = look(front, clothes, bed, false, false).at<int>(8, 8);
    require(patch > 0, "SEMANTIC_MERGE applied a rule of the predicted front");
  }
  std::remove(config.state_path.c_str());
  std::remove((config.state_path + ".catalog.json").c_str());
  std::cout << "PASS online front surface ownership\n";
}
