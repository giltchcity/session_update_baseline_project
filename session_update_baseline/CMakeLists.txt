cmake_minimum_required(VERSION 3.22)
project(session_update_baseline LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(khronos REQUIRED)
find_package(hydra REQUIRED)
find_package(glog REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

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
    khronos::khronos
    ${hydra_LIBRARIES}
    Eigen3::Eigen
    glog::glog
    nlohmann_json::nlohmann_json
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
