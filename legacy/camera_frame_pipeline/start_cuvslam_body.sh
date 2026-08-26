#!/usr/bin/env bash
set -euo pipefail

ISAAC_WS="${ISAAC_ROS_WS:-$HOME/workspaces/isaac_ros-dev}"
CONTAINER_NAME="${ISAAC_CONTAINER_NAME:-isaac_ros_dev-aarch64-container}"
PID_FILE="/tmp/vio_px4_new_pipeline_cuvslam.pid"
OWNER_TAG="vio_px4_new_pipeline"

if ! docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -qx true; then
  echo "Container $CONTAINER_NAME is not running." >&2
  echo "Start it with $ISAAC_WS/src/isaac_ros_common/scripts/run_dev.sh, then retry." >&2
  exit 1
fi

if docker exec "$CONTAINER_NAME" bash -lc '
  for env_file in /proc/[0-9]*/environ; do
    { tr "\0" "\n" < "$env_file" | grep -qx "VIO_PIPELINE_OWNER=vio_px4_new_pipeline"; } 2>/dev/null && exit 0
  done
  exit 1
'; then
  echo "The VIO_PX4_NEW_PIPELINE cuVSLAM instance is already running in $CONTAINER_NAME." >&2
  exit 2
fi

docker exec -u admin --workdir /workspaces/isaac_ros-dev "$CONTAINER_NAME" \
  env VIO_CUVSLAM_PID_FILE="$PID_FILE" \
      VIO_PIPELINE_OWNER="$OWNER_TAG" \
  bash -lc 'set -e; source /opt/ros/humble/setup.bash; \
    echo $$ > "$VIO_CUVSLAM_PID_FILE"; \
    cleanup() { rm -f "$VIO_CUVSLAM_PID_FILE"; }; \
    trap cleanup EXIT INT TERM; \
    ros2 launch isaac_ros_examples isaac_ros_examples.launch.py \
      launch_fragments:=realsense_stereo_rect,visual_slam \
      interface_specs_file:=/workspaces/isaac_ros-dev/isaac_ros_assets/isaac_ros_visual_slam/quickstart_interface_specs.json \
      base_frame:=camera_link \
      image_jitter_threshold_ms:=60.0 \
      camera_optical_frames:="['"'"'camera_infra1_optical_frame'"'"', '"'"'camera_infra2_optical_frame'"'"']"'
