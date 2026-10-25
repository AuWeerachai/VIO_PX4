#!/usr/bin/env bash
set -euo pipefail

: "${ISAAC_ROS_WS:?ISAAC_ROS_WS must point to the Isaac ROS workspace}"
cd "${ISAAC_ROS_WS}/src/isaac_ros_common"

# Single authoritative vehicle-to-camera extrinsic in ROS FLU frames.
# Positive pitch rotates camera-forward +X toward downward -Z.
CAMERA_X_M="${CAMERA_X_M:-0.0}"
CAMERA_Y_M="${CAMERA_Y_M:-0.0}"
CAMERA_Z_M="${CAMERA_Z_M:-0.0}"
CAMERA_ROLL_RAD="${CAMERA_ROLL_RAD:-0.0}"
CAMERA_PITCH_RAD="${CAMERA_PITCH_RAD:-0.0}"
CAMERA_YAW_RAD="${CAMERA_YAW_RAD:-0.0}"

mode_file="$ISAAC_ROS_WS/src/isaac_ros_common/scripts/vio_imu_mode"
mode="${VIO_MODE:-$(if [[ -f "$mode_file" ]]; then tr -d '\r\n' < "$mode_file"; else echo vio; fi)}"
case "$mode" in
  vo) fusion=False ;;
  vio) fusion=True ;;
  *) echo "Invalid camera mode: $mode (expected vio or vo)" >&2; exit 2 ;;
esac
if [[ "$mode" == vio ]]; then
  ./scripts/build_vio_imu_overlay.sh
  source ./scripts/vio_imu_release.env
  overlay="$ISAAC_ROS_WS/.vio-imu/$VIO_IMU_OVERLAY_ID"
  set +u
  source "$overlay/ros_install/local_setup.bash"
  set -u
  export LD_LIBRARY_PATH="$overlay/sdk/lib:${LD_LIBRARY_PATH:-}"
fi
echo "=== Camera mode: ${mode^^} (IMU fusion=$fusion) ==="

ros2 run tf2_ros static_transform_publisher \
  --x "$CAMERA_X_M" --y "$CAMERA_Y_M" --z "$CAMERA_Z_M" \
  --yaw "$CAMERA_YAW_RAD" --pitch "$CAMERA_PITCH_RAD" --roll "$CAMERA_ROLL_RAD" \
  --frame-id drone_link --child-frame-id camera_link &
extrinsic_pid=$!
cleanup() { kill "$extrinsic_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# image_jitter_threshold_ms is explicit so reproducing this pipeline never
# requires editing NVIDIA's visual_slam_node.cpp default.
setsid ros2 launch isaac_ros_examples isaac_ros_examples.launch.py \
  launch_fragments:=realsense_stereo_rect,visual_slam \
  interface_specs_file:="${ISAAC_ROS_WS}/isaac_ros_assets/isaac_ros_visual_slam/quickstart_interface_specs.json" \
  base_frame:=drone_link \
  enable_imu_fusion:="$fusion" \
  image_jitter_threshold_ms:=60.0 \
  camera_optical_frames:="['camera_infra1_optical_frame', 'camera_infra2_optical_frame']" &
launch_pid=$!
cleanup() {
  trap - EXIT INT TERM
  kill -INT -- "-$launch_pid" 2>/dev/null || true
  for _ in {1..50}; do
    kill -0 -- "-$launch_pid" 2>/dev/null || break
    sleep 0.1
  done
  kill -TERM -- "-$launch_pid" 2>/dev/null || true
  kill "$extrinsic_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "$mode" == vio ]]; then
  python3 ./scripts/stage_realsense_imu.py --configure-only
fi
wait "$launch_pid"
