#include <chrono>
#include <memory>
#include <stdexcept>

#include <config_utilities/parsing/context.h>
#include <glog/logging.h>
#include <ianvs/node_init.h>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/u_int64.hpp>
#include <std_srvs/srv/empty.hpp>

#include "khronos_ros/experiments/experiment_manager.h"
#include "khronos_ros/khronos_pipeline.h"
#include "khronos_ros/utils/ros_namespaces.h"

// Transport-only adapter for an unmodified upstream Khronos build. It adds the
// deterministic per-frame ACK and terminal save handshake used by our dataset
// player, but does not replace or subclass any mapping component.
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

  auto shutdown_client = node->create_client<std_srvs::srv::Empty>("shutdown");
  auto finish_ack_pub = node->create_publisher<std_msgs::msg::Empty>(
      "/session_update/finish_saved", rclcpp::QoS(1).reliable());
  auto finish_sub = node->create_subscription<std_msgs::msg::Empty>(
      "/session_update/finish_and_save",
      rclcpp::QoS(1).reliable(),
      [&manager, shutdown_client, finish_ack_pub](
          const std_msgs::msg::Empty::SharedPtr) {
        manager.finishMappingAndSaveCallback(nullptr, nullptr);
        finish_ack_pub->publish(std_msgs::msg::Empty());
        if (!finish_ack_pub->wait_for_all_acked(std::chrono::seconds(5))) {
          throw std::runtime_error("terminal save ACK was not delivered");
        }
        if (!shutdown_client->service_is_ready()) {
          throw std::runtime_error("ianvs shutdown service is unavailable");
        }
        shutdown_client->async_send_request(
            std::make_shared<std_srvs::srv::Empty::Request>());
      });

  manager.run();

  finish_sub.reset();
  finish_ack_pub.reset();
  shutdown_client.reset();
  processed_pub.reset();
  ianvs::CurrentNode::clear();
  node.reset();
  return 0;
}
