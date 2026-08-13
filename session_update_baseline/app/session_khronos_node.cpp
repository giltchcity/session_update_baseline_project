#include <atomic>
#include <chrono>

#include <config_utilities/parsing/context.h>
#include <glog/logging.h>
#include <ianvs/node_init.h>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/u_int64.hpp>

#include "khronos_ros/experiments/experiment_manager.h"
#include "khronos_ros/khronos_pipeline.h"
#include "khronos_ros/utils/ros_namespaces.h"
#include "session_update_baseline/runtime/session_backend.h"

int main(int argc, char** argv) {
  config::initContext(argc, argv, true);
  config::setConfigSettingsFromContext();
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("khronos_node");
  ianvs::NodeHandle nh(*node);
  ianvs::CurrentNode::init(node);

  FLAGS_alsologtostderr = true;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  auto pipeline = std::make_shared<khronos::KhronosPipeline>(nh);
  auto processed_pub = node->create_publisher<std_msgs::msg::UInt64>(
      "/session_update/frame_processed", rclcpp::QoS(10).reliable());
  pipeline->setActiveWindowEvaluationCallback(
      [processed_pub](const auto&, const auto& data, const auto&) {
        std_msgs::msg::UInt64 message;
        message.data = data.input.timestamp_ns;
        processed_pub->publish(message);
      });
  const auto experiment_config =
      config::fromContext<khronos::ExperimentManager::Config>(
          khronos::RosNs::EXPERIMENT);
  khronos::ExperimentManager manager(experiment_config, nh, pipeline);

  // The official finish_mapping_and_save service is unreachable here: the ROS2
  // CLI cannot open sockets in this sandbox, so `ros2 service call` always
  // fails with EPERM. Topics are unaffected, so the player signals completion
  // over one. The callback saves synchronously, then ends the process itself --
  // no signals, no external kill.
  auto finish_sub = node->create_subscription<std_msgs::msg::Empty>(
      "/session_update/finish_and_save",
      rclcpp::QoS(1).reliable(),
      [&manager](const std_msgs::msg::Empty::SharedPtr) {
        LOG(INFO) << "[SessionNode] Playback finished; saving and shutting down.";
        manager.finishMappingAndSaveCallback(nullptr, nullptr);
        LOG(INFO) << "[SessionNode] Save complete.";
        rclcpp::shutdown();
      });
  manager.run();

  // Tear the ROS entities down here, while the DDS runtime is still loaded.
  // ianvs::CurrentNode keeps the node alive in a static unique_ptr, so leaving
  // it to __cxa_finalize destroys the node after FastDDS has finalized its own
  // globals and segfaults in rmw_destroy_service -- after a fully successful
  // save, which makes the run look like it crashed when it did not.
  finish_sub.reset();
  processed_pub.reset();
  ianvs::CurrentNode::clear();
  node.reset();

  return 0;
}
