# objects4d, built against the official Khronos build used by the Object F1 evaluation:
#   source FT/build_artifacts/official_khronos_63faadde_abc/environment.sh
#   cmake -S update_layer/eval/objects4d -B FT/build_artifacts/update_layer_objects4d && cmake --build FT/build_artifacts/update_layer_objects4d
cmake_minimum_required(VERSION 3.22)
project(objects4d LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(khronos REQUIRED)
add_executable(objects4d objects4d.cpp)
target_link_libraries(objects4d PRIVATE khronos::khronos)
