#include <iostream>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/u_int64.hpp>

#include "session_update_baseline/runtime/fastdds_ack_contract.h"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("fastdds_transaction_profile_test");
  const auto qos = rclcpp::QoS(10).reliable();
  auto ack = node->create_publisher<std_msgs::msg::UInt64>(
      "/session_update/frame_processed", rclcpp::QoS(10).reliable());
  auto rgb = node->create_publisher<sensor_msgs::msg::Image>(
      "/nss/rgb/image_raw", qos);
  auto depth = node->create_publisher<sensor_msgs::msg::Image>(
      "/nss/depth/image_raw", qos);
  auto semantic = node->create_publisher<sensor_msgs::msg::Image>(
      "/nss/semantic/image_raw", qos);

  const auto verify = [](const rclcpp::PublisherBase& publisher,
                         const std::string& topic) {
    const auto timing = session_update_baseline::verifyFastDdsWriterContract(
        publisher, topic);
    std::cout << "FAST_DDS_WRITER_CONTRACT_OK topic=" << topic
              << " reliability=RELIABLE history=KEEP_LAST depth=10"
              << " initial_heartbeat_ns="
              << timing.initial_heartbeat_nanoseconds
              << " heartbeat_ns=" << timing.heartbeat_nanoseconds
              << " nack_response_ns=" << timing.nack_response_nanoseconds
              << std::endl;
  };
  verify(*ack, "/session_update/frame_processed");
  verify(*rgb, "/nss/rgb/image_raw");
  verify(*depth, "/nss/depth/image_raw");
  verify(*semantic, "/nss/semantic/image_raw");
  std::cout << "FAST_DDS_TRANSACTION_PROFILE_OK writers=4" << std::endl;

  semantic.reset();
  depth.reset();
  rgb.reset();
  ack.reset();
  node.reset();
  rclcpp::shutdown();
  return 0;
}
