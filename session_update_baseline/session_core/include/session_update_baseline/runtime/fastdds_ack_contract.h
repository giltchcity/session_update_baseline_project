#pragma once

#include <cstdint>
#include <string>

#include <rclcpp/publisher_base.hpp>

namespace session_update_baseline {

struct FastDdsWriterTiming {
  int32_t initial_heartbeat_seconds = 0;
  uint32_t initial_heartbeat_nanoseconds = 0;
  int32_t heartbeat_seconds = 0;
  uint32_t heartbeat_nanoseconds = 0;
  int32_t nack_response_seconds = 0;
  uint32_t nack_response_nanoseconds = 0;
};

// Inspect the native Fast DDS writer behind a ROS publisher. This deliberately
// fails closed if another RMW implementation is selected: the production
// per-frame transport recovery contract is specific to rmw_fastrtps_cpp.
FastDdsWriterTiming inspectFastDdsWriterTiming(
    const rclcpp::PublisherBase& publisher);

// Require RELIABLE KEEP_LAST depth=10 and the exact native recovery timing
// configured for one transaction writer. Throws std::runtime_error on drift.
FastDdsWriterTiming verifyFastDdsWriterContract(
    const rclcpp::PublisherBase& publisher, const std::string& topic_name);

}  // namespace session_update_baseline
