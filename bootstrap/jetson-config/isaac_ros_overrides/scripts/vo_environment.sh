#!/usr/bin/env bash
# Source this in every camera, RViz and readiness shell. Never source the whole
# Isaac install: old RealSense message packages there are not ABI-compatible
# with the camera driver in the pinned image. Leave those files intact on disk.
_vio_prepare_environment() {
  local workspace="${ISAAC_ROS_WS:?ISAAC_ROS_WS is required}"
  local variable entry filtered
  local -a entries
  for variable in AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH \
                  LD_LIBRARY_PATH PYTHONPATH PKG_CONFIG_PATH PATH; do
    filtered=""
    IFS=: read -r -a entries <<< "${!variable-}"
    for entry in "${entries[@]}"; do
      case "$entry" in
        ""|"${workspace%/}/install"|"${workspace%/}/install/"*|*"/.vio-imu/"*) continue ;;
      esac
      filtered="${filtered:+$filtered:}$entry"
    done
    export "$variable=$filtered"
  done
  # ROS-generated setup scripts do not support nounset consistently.
  local restore_nounset=false
  [[ $- == *u* ]] && restore_nounset=true
  set +u
  source /opt/ros/humble/setup.bash || { $restore_nounset && set -u; return 1; }
  local overlay="${workspace%/}/install/isaac_ros_visual_slam/share/isaac_ros_visual_slam/package.bash"
  if [[ -f "$overlay" ]]; then
    # Package-local setup does not traverse the rest of this workspace.
    unset COLCON_CURRENT_PREFIX AMENT_CURRENT_PREFIX
    source "$overlay" || { $restore_nounset && set -u; return 1; }
  fi
  $restore_nounset && set -u
  return 0
}
_vio_prepare_environment
_vio_environment_result=$?
unset -f _vio_prepare_environment
return "$_vio_environment_result" 2>/dev/null || exit "$_vio_environment_result"
