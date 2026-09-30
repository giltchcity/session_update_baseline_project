#include "session_update_baseline/runtime/fastdds_ack_contract.h"

#include <stdexcept>

#include <rcl/publisher.h>
#include <rmw_fastrtps_cpp/get_publisher.hpp>

namespace session_update_baseline {
FastDdsWriterTiming inspectFastDdsWriterTiming(
    const rclcpp::PublisherBase& publisher) {
  auto* const rmw_publisher =
      rcl_publisher_get_rmw_handle(publisher.get_publisher_handle().get());
  auto* const writer = rmw_fastrtps_cpp::get_datawriter(rmw_publisher);
  if (!writer) {
    throw std::runtime_error(
        "publisher is not backed by rmw_fastrtps_cpp/Fast DDS");
  }

  const auto& times = writer->get_qos().reliable_writer_qos().times;
  FastDdsWriterTiming result;
  result.initial_heartbeat_seconds = times.initialHeartbeatDelay.seconds;
  result.initial_heartbeat_nanoseconds = times.initialHeartbeatDelay.nanosec;
  result.heartbeat_seconds = times.heartbeatPeriod.seconds;
  result.heartbeat_nanoseconds = times.heartbeatPeriod.nanosec;
  result.nack_response_seconds = times.nackResponseDelay.seconds;
  result.nack_response_nanoseconds = times.nackResponseDelay.nanosec;
  return result;
}

FastDdsWriterTiming verifyFastDdsWriterContract(
    const rclcpp::PublisherBase& publisher, const std::string& topic_name) {
  if (topic_name != publisher.get_topic_name()) {
    throw std::runtime_error("publisher topic mismatch: expected " +
                             topic_name + " got " +
                             publisher.get_topic_name());
  }
  // README (8): the per-frame ACK must arrive; that needs a reliable writer. The native writer
  // timing is reported, not checked against fixed values.
  if (publisher.get_actual_qos().reliability() != rclcpp::ReliabilityPolicy::Reliable) {
    throw std::runtime_error("publisher is not RELIABLE: " + topic_name);
  }
  const auto timing = inspectFastDdsWriterTiming(publisher);
  return timing;
}

}  // namespace session_update_baseline
