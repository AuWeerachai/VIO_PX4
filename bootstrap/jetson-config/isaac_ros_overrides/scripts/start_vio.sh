#!/usr/bin/env bash
set -euo pipefail

: "${ISAAC_ROS_WS:?ISAAC_ROS_WS must point to the Isaac ROS workspace}"
cd "${ISAAC_ROS_WS}/src/isaac_ros_common"

# Single authoritative vehicle-to-camera extrinsic in ROS FLU frames.
# Positive pitch rotates camera-forward +X toward downward -Z.
CAMERA_X_M="${CAMERA_X_M:-0.0}"
CAMERA_Y_M="${CAMERA_Y_M:-0.0}"
CAMERA_Z_M="${CAMERA_Z_M:-0.0}"
CAMERA_PITCH_RAD="${CAMERA_PITCH_RAD:-0.0}"

ros2 run tf2_ros static_transform_publisher \
  --x "$CAMERA_X_M" --y "$CAMERA_Y_M" --z "$CAMERA_Z_M" \
  --yaw 0 --pitch "$CAMERA_PITCH_RAD" --roll 0 \
  --frame-id drone_link --child-frame-id camera_link &
extrinsic_pid=$!
cleanup() { kill "$extrinsic_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# This experiment uses a project-owned launch file because NVIDIA's generic
# visual_slam fragment does not remap the RealSense combined IMU topic.
ros2 launch "$(dirname "${BASH_SOURCE[0]}")/cuvslam_vio.launch.py"
