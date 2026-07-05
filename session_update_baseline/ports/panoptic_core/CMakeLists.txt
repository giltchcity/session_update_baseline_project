cmake_minimum_required(VERSION 3.16)
project(session_update_panoptic_core LANGUAGES CXX)

find_package(Eigen3 REQUIRED)

add_library(session_update_panoptic_core
  src/tsdf_conflict_checker.cpp
)

target_compile_features(session_update_panoptic_core PUBLIC cxx_std_17)

target_include_directories(session_update_panoptic_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(session_update_panoptic_core
  PUBLIC Eigen3::Eigen
)

