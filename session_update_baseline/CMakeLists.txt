cmake_minimum_required(VERSION 3.22)
project(session_update_baseline LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SESSION_UPDATE_CANONICAL_MAPPING_PREFIX "" CACHE PATH
    "Install prefix built from ports/mapping_core by scripts/build_canonical.sh")
if(NOT SESSION_UPDATE_CANONICAL_MAPPING_PREFIX)
  message(FATAL_ERROR
    "Use scripts/build_canonical.sh. Direct configuration must set "
    "-DSESSION_UPDATE_CANONICAL_MAPPING_PREFIX=<ports/mapping_core install prefix>.")
endif()

file(REAL_PATH "${SESSION_UPDATE_CANONICAL_MAPPING_PREFIX}"
     SESSION_UPDATE_CANONICAL_MAPPING_PREFIX_REAL)
set(_canonical_khronos_dir
    "${SESSION_UPDATE_CANONICAL_MAPPING_PREFIX_REAL}/lib/cmake/khronos")
set(_canonical_khronos_ros_dir
    "${SESSION_UPDATE_CANONICAL_MAPPING_PREFIX_REAL}/share/khronos_ros/cmake")
if(NOT EXISTS "${_canonical_khronos_dir}/khronosConfig.cmake" OR
   NOT EXISTS "${_canonical_khronos_ros_dir}/khronos_rosConfig.cmake")
  message(FATAL_ERROR
    "Canonical mapping install is incomplete: "
    "${SESSION_UPDATE_CANONICAL_MAPPING_PREFIX_REAL}")
endif()

# Never let a sourced external workspace select another Khronos build. Other
# research dependencies may currently come from an installed dependency prefix,
# but the mapper itself is always the copy under ports/mapping_core.
set(khronos_DIR "${_canonical_khronos_dir}" CACHE PATH "" FORCE)
set(khronos_ros_DIR "${_canonical_khronos_ros_dir}" CACHE PATH "" FORCE)
list(PREPEND CMAKE_PREFIX_PATH "${SESSION_UPDATE_CANONICAL_MAPPING_PREFIX_REAL}")

find_package(Eigen3 REQUIRED)
find_package(khronos REQUIRED)
find_package(khronos_ros REQUIRED)
find_package(hydra REQUIRED)
find_package(glog REQUIRED)
find_package(gflags REQUIRED)
find_package(ianvs REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(rclcpp REQUIRED)
find_package(rmw_fastrtps_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

file(REAL_PATH "${khronos_DIR}" _resolved_khronos_dir)
file(REAL_PATH "${khronos_ros_DIR}" _resolved_khronos_ros_dir)
string(FIND "${_resolved_khronos_dir}"
       "${SESSION_UPDATE_CANONICAL_MAPPING_PREFIX_REAL}/" _khronos_prefix_pos)
string(FIND "${_resolved_khronos_ros_dir}"
       "${SESSION_UPDATE_CANONICAL_MAPPING_PREFIX_REAL}/" _khronos_ros_prefix_pos)
if(NOT _khronos_prefix_pos EQUAL 0 OR NOT _khronos_ros_prefix_pos EQUAL 0)
  message(FATAL_ERROR
    "Resolved a non-canonical Khronos package: ${khronos_DIR}; ${khronos_ros_DIR}")
endif()

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
  src/runtime/fastdds_ack_contract.cpp
  src/runtime/session_backend.cpp
)

target_include_directories(session_khronos_node
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${rmw_fastrtps_cpp_INCLUDE_DIRS}
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
    ${rmw_fastrtps_cpp_LIBRARIES}
    ${std_msgs_TARGETS}
    ${std_srvs_TARGETS}
)

add_executable(stream_4dmap_copy
  app/stream_4dmap_copy.cpp
)

add_executable(inspect_session_state
  app/inspect_session_state.cpp
  src/runtime/session_state_fingerprint.cpp
)

target_include_directories(inspect_session_state
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(inspect_session_state
  PRIVATE
    khronos::khronos
    ${hydra_LIBRARIES}
    glog::glog
    nlohmann_json::nlohmann_json
)

add_executable(compose_session_4dmap
  app/compose_session_4dmap.cpp
)

target_include_directories(compose_session_4dmap
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
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

add_test(
  NAME fastdds_ack_xml_contract
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/scripts/verify_fastdds_ack_profile.py
)

add_executable(test_fastdds_ack_profile
  app/test_fastdds_ack_profile.cpp
  src/runtime/fastdds_ack_contract.cpp
)

target_include_directories(test_fastdds_ack_profile
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${rmw_fastrtps_cpp_INCLUDE_DIRS}
)

target_link_libraries(test_fastdds_ack_profile
  PRIVATE
    rclcpp::rclcpp
    ${rmw_fastrtps_cpp_LIBRARIES}
    ${sensor_msgs_TARGETS}
    ${std_msgs_TARGETS}
)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fastdds_ack_test_logs")
add_test(
  NAME fastdds_ack_runtime_contract
  COMMAND test_fastdds_ack_profile
)
set_tests_properties(fastdds_ack_runtime_contract PROPERTIES
  ENVIRONMENT
    "RMW_IMPLEMENTATION=rmw_fastrtps_cpp;FASTRTPS_DEFAULT_PROFILES_FILE=${CMAKE_CURRENT_SOURCE_DIR}/configs/fastdds_session_update_ack.xml;RMW_FASTRTPS_USE_QOS_FROM_XML=1;FASTDDS_DEFAULT_PROFILES_FILE=;ROS_DOMAIN_ID=229;ROS_HOME=${CMAKE_CURRENT_BINARY_DIR}/fastdds_ack_test_logs;ROS_LOG_DIR=${CMAKE_CURRENT_BINARY_DIR}/fastdds_ack_test_logs"
)

add_executable(test_session_state_semantics
  app/test_session_state_semantics.cpp
)

target_include_directories(test_session_state_semantics
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(test_session_state_semantics
  PRIVATE
    khronos::khronos
    ${hydra_LIBRARIES}
    Eigen3::Eigen
    glog::glog
)

add_test(
  NAME session_state_semantics
  COMMAND test_session_state_semantics
          "${CMAKE_CURRENT_BINARY_DIR}/session_state_semantics_test"
)

add_executable(test_session_state_fingerprint
  app/test_session_state_fingerprint.cpp
  src/runtime/session_state_fingerprint.cpp
)

target_include_directories(test_session_state_fingerprint
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(test_session_state_fingerprint
  PRIVATE
    khronos::khronos
    ${hydra_LIBRARIES}
    Eigen3::Eigen
    glog::glog
)

add_test(
  NAME session_state_fingerprint
  COMMAND test_session_state_fingerprint
          "${CMAKE_CURRENT_BINARY_DIR}/session_state_fingerprint_test"
)

add_executable(test_d3_provenance_and_presence
  app/test_d3_provenance_and_presence.cpp
)

target_include_directories(test_d3_provenance_and_presence
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(test_d3_provenance_and_presence
  PRIVATE
    khronos::khronos
    ${hydra_LIBRARIES}
    Eigen3::Eigen
    glog::glog
)

add_test(
  NAME d3_provenance_and_presence
  COMMAND test_d3_provenance_and_presence
          "${CMAKE_CURRENT_BINARY_DIR}/d3_provenance_and_presence_test"
)

add_executable(test_hidden_change_session_equivalence
  app/test_hidden_change_session_equivalence.cpp
)

target_include_directories(test_hidden_change_session_equivalence
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(test_hidden_change_session_equivalence
  PRIVATE
    khronos::khronos
    ${hydra_LIBRARIES}
    Eigen3::Eigen
    glog::glog
)

add_test(
  NAME hidden_change_session_equivalence
  COMMAND test_hidden_change_session_equivalence
          "${CMAKE_CURRENT_BINARY_DIR}/hidden_change_session_equivalence_test"
)

add_executable(test_physical_presence_recursion
  app/test_physical_presence_recursion.cpp
)

target_include_directories(test_physical_presence_recursion
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(test_physical_presence_recursion
  PRIVATE
    khronos::khronos
    ${hydra_LIBRARIES}
    Eigen3::Eigen
    glog::glog
)

add_test(
  NAME physical_presence_recursion
  COMMAND test_physical_presence_recursion
          "${CMAKE_CURRENT_BINARY_DIR}/physical_presence_recursion_test"
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

# Long-lived .4dmap mesh server: loads one or two maps once, serves
# per-time-step current-scene meshes over stdin/stdout as raw binary.
# Produces no files; the Python Open3D viewer drives it via a pipe.
add_executable(4dmap_mesh_server
  app/4dmap_mesh_server.cpp
)

target_include_directories(4dmap_mesh_server
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(4dmap_mesh_server
  PRIVATE
    khronos::khronos
    ${hydra_LIBRARIES}
    Eigen3::Eigen
    glog::glog
)
