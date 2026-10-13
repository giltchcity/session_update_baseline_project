#include "session_update_baseline/runtime/fastdds_ack_contract.h"

#include <sstream>
#include <stdexcept>

#include <rcl/publisher.h>
#include <rmw_fastrtps_cpp/get_publisher.hpp>

namespace session_update_baseline {
namespace {

constexpr int32_t kExpectedInitialHeartbeatSeconds = 0;
constexpr uint32_t kExpectedInitialHeartbeatNanoseconds = 1'000'000;
constexpr int32_t kExpectedHeartbeatSeconds = 0;
constexpr uint32_t kExpectedHeartbeatNanoseconds = 10'000'000;
constexpr int32_t kExpectedNackResponseSeconds = 0;
constexpr uint32_t kExpectedNackResponseNanoseconds = 1'000'000;

}  // namespace

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
  const auto actual_qos = publisher.get_actual_qos();
  if (actual_qos.reliability() != rclcpp::ReliabilityPolicy::Reliable) {
    throw std::runtime_error("publisher is not RELIABLE: " + topic_name);
  }
  if (actual_qos.history() != rclcpp::HistoryPolicy::KeepLast ||
      actual_qos.get_rmw_qos_profile().depth != 10) {
    throw std::runtime_error(
        "publisher must retain RELIABLE KEEP_LAST depth=10: " +
        topic_name);
  }

  const auto timing = inspectFastDdsWriterTiming(publisher);
  const bool valid =
      timing.initial_heartbeat_seconds == kExpectedInitialHeartbeatSeconds &&
      timing.initial_heartbeat_nanoseconds ==
          kExpectedInitialHeartbeatNanoseconds &&
      timing.heartbeat_seconds == kExpectedHeartbeatSeconds &&
      timing.heartbeat_nanoseconds == kExpectedHeartbeatNanoseconds &&
      timing.nack_response_seconds == kExpectedNackResponseSeconds &&
      timing.nack_response_nanoseconds == kExpectedNackResponseNanoseconds;
  if (!valid) {
    std::ostringstream message;
    message << "Fast DDS writer timing contract mismatch for " << topic_name
            << ": initial=" << timing.initial_heartbeat_seconds << "s+"
            << timing.initial_heartbeat_nanoseconds << "ns heartbeat="
            << timing.heartbeat_seconds << "s+" << timing.heartbeat_nanoseconds
            << "ns nack_response=" << timing.nack_response_seconds << "s+"
            << timing.nack_response_nanoseconds
            << "ns; expected 0s+1000000ns, 0s+10000000ns, 0s+1000000ns";
    throw std::runtime_error(message.str());
  }
  return timing;
}

}  // namespace session_update_baseline
