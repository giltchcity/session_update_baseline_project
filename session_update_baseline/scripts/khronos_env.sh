#!/usr/bin/env bash

# Source this file from canonical runtime entry points. Khronos itself must come
# from ports/mapping_core via scripts/build_canonical.sh. The dependency overlay
# is currently still needed for Hydra/PGMO and related installed packages.
BASE1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE1_ROS_DISTRO_SETUP="${BASE1_ROS_DISTRO_SETUP:-/opt/ros/jazzy/setup.bash}"
BASE1_DEPENDENCY_SETUP="${BASE1_DEPENDENCY_SETUP:-/home/jixian/ros2_ws/install/setup.bash}"
BASE1_CANONICAL_PREFIX="${BASE1_CANONICAL_PREFIX:-${SESSION_UPDATE_CANONICAL_PREFIX:-${BASE1_ROOT}/.canonical_mapping/install}}"
BASE1_KHRONOS_SETUP="${BASE1_KHRONOS_SETUP:-${BASE1_CANONICAL_PREFIX}/local_setup.bash}"
BASE1_PYTHON="${BASE1_PYTHON:-/usr/bin/python3}"
BASE1_BUILD_DIR="${BASE1_BUILD_DIR:-${SESSION_UPDATE_CANONICAL_BUILD:-${BASE1_ROOT}/build_canonical}}"

if [[ ! -f "${BASE1_ROS_DISTRO_SETUP}" ]]; then
  echo "BASE1_ENV_ERROR missing ROS setup: ${BASE1_ROS_DISTRO_SETUP}" >&2
  return 2 2>/dev/null || exit 2
fi
if [[ ! -f "${BASE1_DEPENDENCY_SETUP}" ]]; then
  echo "BASE1_ENV_ERROR missing dependency setup: ${BASE1_DEPENDENCY_SETUP}" >&2
  return 2 2>/dev/null || exit 2
fi
if [[ ! -f "${BASE1_KHRONOS_SETUP}" ]]; then
  echo "BASE1_ENV_ERROR missing canonical mapping setup: ${BASE1_KHRONOS_SETUP}" >&2
  echo "Run ${BASE1_ROOT}/scripts/build_canonical.sh first." >&2
  return 2 2>/dev/null || exit 2
fi
if [[ ! -x "${BASE1_PYTHON}" ]]; then
  echo "BASE1_ENV_ERROR missing Python: ${BASE1_PYTHON}" >&2
  return 2 2>/dev/null || exit 2
fi

case "$-" in
  *u*) _base1_restore_nounset=1 ;;
  *) _base1_restore_nounset=0 ;;
esac
set +u
# shellcheck disable=SC1090
source "${BASE1_ROS_DISTRO_SETUP}"
# shellcheck disable=SC1090
source "${BASE1_DEPENDENCY_SETUP}"
# shellcheck disable=SC1090
source "${BASE1_KHRONOS_SETUP}"
if [[ "${_base1_restore_nounset}" == "1" ]]; then
  set -u
fi
unset _base1_restore_nounset

export BASE1_ROOT BASE1_PYTHON BASE1_BUILD_DIR BASE1_CANONICAL_PREFIX
export BASE1_KHRONOS_SETUP BASE1_DEPENDENCY_SETUP
