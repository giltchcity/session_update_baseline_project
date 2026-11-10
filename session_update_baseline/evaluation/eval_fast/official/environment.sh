# Source this file; it selects the isolated official Khronos libraries first.
source /opt/ros/jazzy/setup.bash
source /home/jixian/ros2_ws/install/setup.bash
source /home/jixian/Desktop/FT/eval_fast/official/install/local_setup.bash
export OFFICIAL_KHRONOS_BUILD=/home/jixian/Desktop/FT/eval_fast/official
export LD_LIBRARY_PATH="$OFFICIAL_KHRONOS_BUILD/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="$OFFICIAL_KHRONOS_BUILD/install${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
