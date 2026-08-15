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
        src/active_window/object_detection/connected_semantics.cpp
        src/active_window/object_detection/instance_forwarding.cpp
        src/active_window/object_extraction/mesh_object_extractor.cpp
        src/active_window/object_extraction/object_worker_pool.cpp
        src/active_window/tracking/external_tracker.cpp
        src/active_window/tracking/max_iou_tracker.cpp
        src/backend/backend.cpp
        src/backend/change_detection/background/ray_background_change_detector.cpp
        src/backend/change_detection/objects/ray_object_change_detector.cpp
        src/backend/change_detection/physical_evidence_store.cpp
        src/backend/change_detection/ray_change_detector.cpp
        src/backend/change_detection/ray_verificator.cpp
        src/backend/change_detection/sequential_change_detector.cpp
        src/backend/change_state.cpp
        src/backend/latest_only_worker.cpp
        src/backend/reconciliation/mesh/change_merger.cpp
        src/backend/reconciliation/mesh/mesh_merger.cpp
        src/backend/reconciliation/mesh/overwrite_mesh.cpp
        src/backend/reconciliation/persistent_object_state.cpp
        src/backend/reconciliation/reconciler.cpp
        src/backend/update_khronos_objects_functor.cpp
        src/spatio_temporal_map/spatio_temporal_map.cpp
        src/spatio_temporal_map/incremental_mesh.cpp
        src/utils/geometry_utils.cpp
        src/utils/khronos_attribute_utils.cpp
        src/utils/output_file_utils.cpp
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
        PRIVATE ${tf2_eigen_LIBRARIES}
)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
add_library(khronos::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

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

  add_executable(test_explicit_labels_and_external_tracker
    tests/test_explicit_labels_and_external_tracker.cpp
  )
  target_link_libraries(test_explicit_labels_and_external_tracker PRIVATE ${PROJECT_NAME})
  add_test(
    NAME explicit_labels_and_external_tracker
    COMMAND test_explicit_labels_and_external_tracker
  )

  add_executable(test_incremental_change_detection_oracle
    tests/test_incremental_change_detection_oracle.cpp
  )
  target_link_libraries(test_incremental_change_detection_oracle PRIVATE ${PROJECT_NAME})
  add_test(
    NAME incremental_change_detection_oracle
    COMMAND test_incremental_change_detection_oracle
  )

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

  add_executable(test_backend_trigger_and_sink
    tests/test_backend_trigger_and_sink.cpp
  )
  target_link_libraries(test_backend_trigger_and_sink PRIVATE ${PROJECT_NAME})
  add_test(
    NAME backend_trigger_and_sink
    COMMAND test_backend_trigger_and_sink
            "${CMAKE_CURRENT_BINARY_DIR}/backend_trigger_and_sink_test"
  )

  add_executable(test_physical_newest_segment_ownership
    tests/test_physical_newest_segment_ownership.cpp
  )
  target_link_libraries(test_physical_newest_segment_ownership PRIVATE ${PROJECT_NAME})
  add_test(
    NAME physical_newest_segment_ownership
    COMMAND test_physical_newest_segment_ownership
  )

  add_executable(test_after_motion_reconstruction_frames
    tests/test_after_motion_reconstruction_frames.cpp
  )
  target_link_libraries(test_after_motion_reconstruction_frames PRIVATE ${PROJECT_NAME})
  add_test(
    NAME after_motion_reconstruction_frames
    COMMAND test_after_motion_reconstruction_frames
  )

  add_executable(test_d3_seed_vertex_later_ray
    tests/test_d3_seed_vertex_later_ray.cpp
  )
  target_link_libraries(test_d3_seed_vertex_later_ray PRIVATE ${PROJECT_NAME})
  add_test(
    NAME d3_seed_vertex_later_ray
    COMMAND test_d3_seed_vertex_later_ray
  )

  add_executable(test_persistent_object_state
    tests/test_persistent_object_state.cpp
  )
  target_link_libraries(test_persistent_object_state PRIVATE ${PROJECT_NAME})
  add_test(
    NAME persistent_object_state
    COMMAND test_persistent_object_state
  )
endif()

##########
# Export #
##########

include(KhronosInstall)
