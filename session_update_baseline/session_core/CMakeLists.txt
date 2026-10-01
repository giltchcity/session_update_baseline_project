# Project algorithm implementations, separate from the Khronos source target.
# Member adapters preserve the existing ABI; objects enter the final mapper once.
add_library(session_core OBJECT
  src/adapters/extraction_source.cpp
  src/runtime/ordered_completion.cpp
  src/adapters/physical_ray_queries.cpp
  src/adapters/physical_object_merge.cpp
  src/adapters/static_surface_selection.cpp
  src/adapters/backend_session.cpp
  src/adapters/evidence_round.cpp
  src/state/persistent_object_state.cpp
  src/state/state_checkpoint.cpp
  src/runtime/session_bundle.cpp
  src/surface/closed_object_background.cpp
  src/surface/element_posterior.cpp
  src/surface/memory_mixture.cpp
  src/surface/frame_archive.cpp
  src/surface/present_tsdf.cpp
  src/surface/session_refusion.cpp
  src/surface/range_calibration.cpp
  src/surface/surface_state.cpp
  src/surface/surface_sampling.cpp
  src/surface/triangle_grid.cpp
  src/evidence/physical_evidence_store.cpp
  src/evidence/projected_physical_evidence.cpp
  src/evidence/observed_absence.cpp
  src/evidence/element_measurement.cpp
  src/evidence/frame_pair_sampler.cpp
  src/evidence/free_space_records.cpp
  src/model/model_math.cpp
  src/model/duplicate_mixture.cpp
  src/model/persistence_prior.cpp
  src/model/change_filter.cpp
  src/model/hierarchical_beta.cpp
  src/model/motion_model.cpp
  src/model/round_model.cpp
  src/model/range_model.cpp
  src/model/sensor_calibrator.cpp
)
set_property(TARGET session_core PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(session_core PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../ports/mapping_core/khronos/include"
  ${OpenCV_INCLUDE_DIRS} ${kimera_pgmo_INCLUDE_DIRS} ${hydra_INCLUDE_DIRS})
target_link_libraries(session_core PRIVATE
  ${OpenCV_LIBRARIES} config_utilities::config_utilities spark_dsg::spark_dsg
  gtsam spatial_hash::spatial_hash ${kimera_pgmo_LIBRARIES} ${hydra_LIBRARIES}
  ${KHRONOS_ZSTD_LIBRARY} OpenSSL::Crypto)
target_sources(khronos PRIVATE $<TARGET_OBJECTS:session_core>)
target_include_directories(khronos PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
