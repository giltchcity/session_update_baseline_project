cmake_minimum_required(VERSION 3.22)
project(session_update_baseline LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(khronos REQUIRED)
find_package(khronos_ros REQUIRED)
find_package(hydra REQUIRED)
find_package(glog REQUIRED)
find_package(gflags REQUIRED)
find_package(ianvs REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

add_subdirectory(ports/panoptic_core)

add_executable(run_session_update_baseline
  app/run_session_update_baseline.cpp
  src/base1/object_guided_map_reconciler.cpp
)

target_include_directories(run_session_update_baseline
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(run_session_update_baseline
  PRIVATE
    session_update_panoptic_core
    khronos::khronos
    ${hydra_LIBRARIES}
    Eigen3::Eigen
    glog::glog
    nlohmann_json::nlohmann_json
)

add_executable(session_khronos_node
  app/session_khronos_node.cpp
  src/runtime/session_backend.cpp
  src/runtime/streaming_spatio_temporal_map_save.cpp
)

target_include_directories(session_khronos_node
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(session_khronos_node
  PRIVATE
    khronos_ros::khronos_ros
    khronos::khronos
    ${hydra_LIBRARIES}
    glog::glog
    ${gflags_LIBRARIES}
    ianvs::ianvs
    rclcpp::rclcpp
    ${std_msgs_TARGETS}
    ${std_srvs_TARGETS}
)

add_executable(stream_4dmap_copy
  app/stream_4dmap_copy.cpp
  src/runtime/streaming_spatio_temporal_map_save.cpp
)

add_executable(compose_session_4dmap
  app/compose_session_4dmap.cpp
  src/runtime/streaming_spatio_temporal_map_save.cpp
)

target_link_libraries(compose_session_4dmap
  PRIVATE
    khronos::khronos
    ${hydra_LIBRARIES}
    glog::glog
)

target_link_libraries(stream_4dmap_copy
  PRIVATE
    khronos::khronos
    ${hydra_LIBRARIES}
    glog::glog
)

enable_testing()

add_executable(test_panoptic_core_in_khronos_env
  app/test_panoptic_core_in_khronos_env.cpp
)

target_link_libraries(test_panoptic_core_in_khronos_env
  PRIVATE session_update_panoptic_core
)

add_test(
  NAME panoptic_core_in_khronos_env
  COMMAND test_panoptic_core_in_khronos_env
)

add_executable(audit_base1_mesh_errors
  app/audit_base1_mesh_errors.cpp
)

target_include_directories(audit_base1_mesh_errors
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(audit_base1_mesh_errors
  PRIVATE
    khronos::khronos
    ${hydra_LIBRARIES}
    Eigen3::Eigen
    glog::glog
    ${PCL_LIBRARIES}
)

add_executable(export_4dmap_mesh_ply
  app/export_4dmap_mesh_ply.cpp
)

target_include_directories(export_4dmap_mesh_ply
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(export_4dmap_mesh_ply
  PRIVATE
    khronos::khronos
    ${hydra_LIBRARIES}
    Eigen3::Eigen
    glog::glog
)
