#!/usr/bin/env bash
set -euo pipefail

ISAAC_WS="${ISAAC_ROS_WS:-$HOME/workspaces/isaac_ros-dev}"
CONTAINER_NAME="${ISAAC_CONTAINER_NAME:-isaac_ros_dev-aarch64-container}"
CAMERA_PITCH_DEG="${CAMERA_PITCH_DEG:--30.0}"
CAMERA_TO_BODY_X_M="${CAMERA_TO_BODY_X_M:-0.0}"
CAMERA_TO_BODY_Y_M="${CAMERA_TO_BODY_Y_M:-0.0}"
CAMERA_TO_BODY_Z_M="${CAMERA_TO_BODY_Z_M:-0.0}"

if ! docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -qx true; then
  echo "Container $CONTAINER_NAME is not running." >&2
  echo "Start it with $ISAAC_WS/src/isaac_ros_common/scripts/run_dev.sh, then retry." >&2
  exit 1
fi

pitch_rad=$(python3 -c "import math; print(math.radians(float('$CAMERA_PITCH_DEG')))" )
ros2 run tf2_ros static_transform_publisher \
  "$CAMERA_TO_BODY_X_M" "$CAMERA_TO_BODY_Y_M" "$CAMERA_TO_BODY_Z_M" \
  0 "$pitch_rad" 0 camera_link drone_link &
tf_pid=$!
trap 'kill "$tf_pid" 2>/dev/null || true' EXIT INT TERM

docker exec -u admin --workdir /workspaces/isaac_ros-dev "$CONTAINER_NAME" \
  bash -lc 'source /opt/ros/humble/setup.bash && \
    ros2 launch isaac_ros_examples isaac_ros_examples.launch.py \
      launch_fragments:=realsense_stereo_rect,visual_slam \
      interface_specs_file:=/workspaces/isaac_ros-dev/isaac_ros_assets/isaac_ros_visual_slam/quickstart_interface_specs.json \
      base_frame:=drone_link \
      enable_imu_fusion:=True \
      image_jitter_threshold_ms:=60.0 \
      camera_optical_frames:="['"'"'camera_infra1_optical_frame'"'"', '"'"'camera_infra2_optical_frame'"'"']"'
