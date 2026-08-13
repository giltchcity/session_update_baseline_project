#!/usr/bin/env bash

# Source this file from every Base1 entry point. It keeps the baseline in the
# same ROS2/Jazzy runtime as the installed Khronos workspace.
BASE1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE1_ROS_DISTRO_SETUP="${BASE1_ROS_DISTRO_SETUP:-/opt/ros/jazzy/setup.bash}"
BASE1_KHRONOS_SETUP="${BASE1_KHRONOS_SETUP:-${BASE1_ROOT}/.official_khronos/install/setup.bash}"
BASE1_PYTHON="${BASE1_PYTHON:-/usr/bin/python3}"
BASE1_BUILD_DIR="${BASE1_BUILD_DIR:-${BASE1_ROOT}/build_official}"

if [[ ! -f "${BASE1_ROS_DISTRO_SETUP}" ]]; then
  echo "BASE1_ENV_ERROR missing ROS setup: ${BASE1_ROS_DISTRO_SETUP}" >&2
  return 2 2>/dev/null || exit 2
fi
if [[ ! -f "${BASE1_KHRONOS_SETUP}" ]]; then
  echo "BASE1_ENV_ERROR missing Khronos setup: ${BASE1_KHRONOS_SETUP}" >&2
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
source "${BASE1_KHRONOS_SETUP}"
if [[ "${_base1_restore_nounset}" == "1" ]]; then
  set -u
fi
unset _base1_restore_nounset

export BASE1_ROOT BASE1_PYTHON BASE1_BUILD_DIR BASE1_KHRONOS_SETUP
