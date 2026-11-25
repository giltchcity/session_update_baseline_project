cmake_minimum_required(VERSION 3.22.1)
project(khronos VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_compile_options(-Wall -Wextra)

option(BUILD_SHARED_LIBS "Build shared libs" ON)

find_package(config_utilities REQUIRED)
find_package(spatial_hash REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pose_graph_tools REQUIRED)
find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(spark_dsg REQUIRED)
find_package(kimera_pgmo REQUIRED)
find_package(hydra REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
find_library(KHRONOS_ZSTD_LIBRARY NAMES zstd REQUIRED)
find_library(KHRONOS_LZ4_LIBRARY NAMES lz4 REQUIRED)

include(GNUInstallDirs)

#############
# Libraries #
#############

add_library(${PROJECT_NAME}
        src/active_window/active_window.cpp
        src/active_window/data/frame_data_buffer.cpp
        src/active_window/data/track.cpp
        src/active_window/integration/object_integrator.cpp
        src/active_window/integration/tracking_integrator.cpp
        src/active_window/motion_detection/free_space_motion_detector.cpp
        src/active_window/object_detection/instance_forwarding.cpp
        src/active_window/object_extraction/mesh_object_extractor.cpp
        src/active_window/object_extraction/object_worker_pool.cpp
        src/active_window/tracking/external_tracker.cpp
        src/backend/update_khronos_objects_functor.cpp
        src/spatio_temporal_map/spatio_temporal_map.cpp
        src/spatio_temporal_map/piece_history.cpp
        src/utils/geometry_utils.cpp
        src/utils/khronos_attribute_utils.cpp
        )
target_include_directories(
        ${PROJECT_NAME}
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${OpenCV_INCLUDE_DIRS}
        ${kimera_pgmo_INCLUDE_DIRS}
        ${hydra_INCLUDE_DIRS}
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${tf2_eigen_INCLUE_DIRS}
)
target_link_libraries(
${PROJECT_NAME}
        PUBLIC ${OpenCV_LIBRARIES}
        config_utilities::config_utilities
        spark_dsg::spark_dsg
        gtsam
        spatial_hash::spatial_hash
        ${kimera_pgmo_LIBRARIES}
        ${hydra_LIBRARIES}
        PRIVATE ${tf2_eigen_LIBRARIES} ${KHRONOS_ZSTD_LIBRARY} ${KHRONOS_LZ4_LIBRARY} OpenSSL::Crypto
)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
add_library(khronos::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# piece_history.cpp instantiates spark_dsg's attribute readers, which call spark_dsg::Metadata(nlohmann::json). The
# json found for this build can be another release than the one spark_dsg was built with (the symbols carry the
# json ABI tag), so this one source compiles against the json next to spark_dsg's own headers when that holds one,
# else the system's. -I directories are searched before -isystem ones, so the flag takes effect wherever it lands.
find_path(KHRONOS_SPARK_DSG_JSON_DIR nlohmann/json.hpp
          HINTS ${spark_dsg_DIR}/../../../include /usr/include NO_DEFAULT_PATH)
if(KHRONOS_SPARK_DSG_JSON_DIR)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/spark_dsg_json)
  file(CREATE_LINK ${KHRONOS_SPARK_DSG_JSON_DIR}/nlohmann ${CMAKE_CURRENT_BINARY_DIR}/spark_dsg_json/nlohmann SYMBOLIC)
  set_source_files_properties(src/spatio_temporal_map/piece_history.cpp PROPERTIES
                              COMPILE_OPTIONS "-I${CMAKE_CURRENT_BINARY_DIR}/spark_dsg_json")
endif()

# Project algorithms and member adapters are owned by session_core.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../session_core"
                 "${CMAKE_CURRENT_BINARY_DIR}/session_core")

include(CTest)
if(BUILD_TESTING)
  add_executable(test_frame_data_buffer tests/test_frame_data_buffer.cpp)
  target_link_libraries(test_frame_data_buffer PRIVATE ${PROJECT_NAME})
  add_test(NAME frame_data_buffer COMMAND test_frame_data_buffer)

  add_executable(test_latest_only_worker tests/test_latest_only_worker.cpp)
  target_link_libraries(test_latest_only_worker PRIVATE ${PROJECT_NAME})
  add_test(NAME test_latest_only_worker COMMAND test_latest_only_worker)

  add_executable(test_object_worker_pool tests/test_object_worker_pool.cpp)
  target_link_libraries(test_object_worker_pool PRIVATE ${PROJECT_NAME})
  add_test(NAME test_object_worker_pool COMMAND test_object_worker_pool)

  add_executable(test_spatio_temporal_mesh_closure
    tests/test_spatio_temporal_mesh_closure.cpp
  )
  target_link_libraries(test_spatio_temporal_mesh_closure PRIVATE ${PROJECT_NAME})
  add_test(
    NAME spatio_temporal_mesh_closure
    COMMAND test_spatio_temporal_mesh_closure
  )

  add_executable(test_spatio_temporal_snapshot_storage
    tests/test_spatio_temporal_snapshot_storage.cpp
  )
  target_link_libraries(test_spatio_temporal_snapshot_storage PRIVATE ${PROJECT_NAME})
  add_test(
    NAME spatio_temporal_snapshot_storage
    COMMAND test_spatio_temporal_snapshot_storage
  )

  add_executable(test_spatio_temporal_amend tests/test_spatio_temporal_amend.cpp)
  target_link_libraries(test_spatio_temporal_amend PRIVATE ${PROJECT_NAME})
  add_test(NAME spatio_temporal_amend COMMAND test_spatio_temporal_amend)

  add_executable(test_explicit_labels_and_external_tracker
    tests/test_explicit_labels_and_external_tracker.cpp
  )
  target_link_libraries(test_explicit_labels_and_external_tracker PRIVATE ${PROJECT_NAME})
  add_test(
    NAME explicit_labels_and_external_tracker
    COMMAND test_explicit_labels_and_external_tracker
  )

  add_executable(test_ever_free_reset tests/test_ever_free_reset.cpp)
  target_link_libraries(test_ever_free_reset PRIVATE ${PROJECT_NAME})
  add_test(NAME ever_free_reset COMMAND test_ever_free_reset)


  add_executable(test_online_front_surface_owner tests/test_online_front_surface_owner.cpp)
  target_link_libraries(test_online_front_surface_owner PRIVATE ${PROJECT_NAME})
  add_test(NAME online_front_surface_owner COMMAND test_online_front_surface_owner)

  add_executable(test_physical_identity_evidence
    tests/test_physical_identity_evidence.cpp
  )
  target_link_libraries(test_physical_identity_evidence PRIVATE ${PROJECT_NAME})
  add_test(
    NAME physical_identity_evidence
    COMMAND test_physical_identity_evidence
  )

  add_executable(test_physical_low_motion_fallback
    tests/test_physical_low_motion_fallback.cpp
  )
  target_link_libraries(test_physical_low_motion_fallback PRIVATE ${PROJECT_NAME})
  add_test(
    NAME physical_low_motion_fallback
    COMMAND test_physical_low_motion_fallback
  )


  add_executable(test_physical_newest_segment_ownership
    tests/test_physical_newest_segment_ownership.cpp
  )
  target_link_libraries(test_physical_newest_segment_ownership PRIVATE ${PROJECT_NAME})
  add_test(
    NAME physical_newest_segment_ownership
    COMMAND test_physical_newest_segment_ownership
  )

  add_executable(test_static_surface_consistency tests/test_static_surface_consistency.cpp)
  target_link_libraries(test_static_surface_consistency PRIVATE ${PROJECT_NAME})
  add_test(NAME static_surface_consistency COMMAND test_static_surface_consistency)

  add_executable(test_after_motion_reconstruction_frames
    tests/test_after_motion_reconstruction_frames.cpp
  )
  target_link_libraries(test_after_motion_reconstruction_frames PRIVATE ${PROJECT_NAME})
  add_test(
    NAME after_motion_reconstruction_frames
    COMMAND test_after_motion_reconstruction_frames
  )


  add_executable(test_persistent_object_state
    tests/test_persistent_object_state.cpp
  )
  target_link_libraries(test_persistent_object_state PRIVATE ${PROJECT_NAME})
  add_test(
    NAME persistent_object_state
    COMMAND test_persistent_object_state
  )

  add_executable(test_present_surface tests/test_present_surface.cpp)
  target_link_libraries(test_present_surface PRIVATE ${PROJECT_NAME})
  add_test(NAME present_surface COMMAND test_present_surface)

  # Offline replay tool (no test): quick checks of a session's surface from its
  # dump (KHRONOS_REFUSION_DUMP) and against the offline reference.
  add_executable(refusion_replay tests/refusion_replay.cpp)
  target_link_libraries(refusion_replay PRIVATE ${PROJECT_NAME})

  add_executable(test_temporal_fragment_states
    tests/test_temporal_fragment_states.cpp
  )
  target_link_libraries(test_temporal_fragment_states PRIVATE ${PROJECT_NAME})
  add_test(
    NAME temporal_fragment_states
    COMMAND test_temporal_fragment_states
  )
endif()

##########
# Export #
##########

include(KhronosInstall)
