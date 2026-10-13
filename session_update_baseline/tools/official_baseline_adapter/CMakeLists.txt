cmake_minimum_required(VERSION 3.22)
project(official_khronos_session_adapter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(gflags REQUIRED)
find_package(glog REQUIRED)
find_package(ianvs REQUIRED)
find_package(khronos REQUIRED)
find_package(khronos_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

add_executable(official_khronos_session_node official_khronos_session_node.cpp)
target_link_libraries(official_khronos_session_node PRIVATE
  khronos_ros::khronos_ros
  khronos::khronos
  glog::glog
  ${gflags_LIBRARIES}
  ianvs::ianvs
  rclcpp::rclcpp
  ${std_msgs_TARGETS}
  ${std_srvs_TARGETS}
)
