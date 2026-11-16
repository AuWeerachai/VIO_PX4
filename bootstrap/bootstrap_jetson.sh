#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bootstrap_jetson.sh [--install-deps] [--isaac-ws PATH]

Prepare a Jetson that already has an Isaac ROS cuVSLAM workspace. This installs
project-owned Isaac launch overrides, builds vio_px4_bridge, and creates
~/vio-launch. --install-deps installs ROS/MAVLink runtime dependencies with apt
and pip; omit it for a read/check/build-only setup.
EOF
}

INSTALL_DEPS=false
ISAAC_WS="${ISAAC_ROS_WS:-$HOME/workspaces/isaac_ros-dev}"
while (($#)); do
  case "$1" in
    --install-deps) INSTALL_DEPS=true; shift ;;
    --isaac-ws) ISAAC_WS="${2:?--isaac-ws requires a path}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ISAAC_SCRIPTS="$ISAAC_WS/src/isaac_ros_common/scripts"
ISAAC_COMMON="$ISAAC_WS/src/isaac_ros_common"
ISAAC_VSLAM="$ISAAC_COMMON/isaac_ros_visual_slam"

[[ -f /opt/ros/humble/setup.bash ]] || {
  echo "ROS 2 Humble is required at /opt/ros/humble." >&2
  exit 1
}
[[ -x "$ISAAC_SCRIPTS/run_dev.sh" ]] || {
  echo "Isaac ROS run_dev.sh not found at $ISAAC_SCRIPTS/run_dev.sh" >&2
  echo "Install Isaac ROS cuVSLAM first or pass --isaac-ws PATH." >&2
  exit 1
}
[[ -f "$ISAAC_WS/isaac_ros_assets/isaac_ros_visual_slam/quickstart_interface_specs.json" ]] || {
  echo "Isaac ROS visual-slam interface spec is missing under $ISAAC_WS/isaac_ros_assets." >&2
  exit 1
}

if $INSTALL_DEPS; then
  sudo apt-get update
  sudo apt-get install -y \
    python3-pip python3-colcon-common-extensions tmux gnome-terminal
  python3 -m pip install --user --upgrade pymavlink
fi

python3 -c 'import pymavlink' 2>/dev/null || {
  echo "Missing pymavlink. Re-run with --install-deps." >&2
  exit 1
}
command -v tmux >/dev/null || { echo "Missing tmux; re-run with --install-deps." >&2; exit 1; }
command -v gnome-terminal >/dev/null || {
  echo "Missing gnome-terminal; re-run with --install-deps." >&2
  exit 1
}

backup_stamp="$(date +%Y%m%d-%H%M%S)"
install_override() {
  local source_file="$1"
  local destination="$2"
  if [[ -e "$destination" && ! -L "$destination" ]]; then
    cp -a "$destination" "${destination}.pre-vio-px4-${backup_stamp}"
  fi
  install -m 0755 "$source_file" "$destination"
}

install_override \
  "$SCRIPT_DIR/jetson-config/isaac_ros_overrides/scripts/start_vio.sh" \
  "$ISAAC_SCRIPTS/start_vio.sh"
install_override \
  "$SCRIPT_DIR/jetson-config/isaac_ros_overrides/scripts/start_rviz.sh" \
  "$ISAAC_SCRIPTS/start_rviz.sh"
for helper in check_camera_ready.py camera_selection.py vo_environment.sh check_vo_environment.py; do
  install_override "$SCRIPT_DIR/jetson-config/isaac_ros_overrides/scripts/$helper" "$ISAAC_SCRIPTS/$helper"
done
install_override "$SCRIPT_DIR/jetson-config/isaac_ros_overrides/scripts/vio-imu-mode" "$HOME/vio-imu-mode"
# Kept for older launchers; the current launcher always selects stereo VO.
if [[ ! -f "$ISAAC_SCRIPTS/vio_imu_mode" ]]; then
  printf 'vo\n' > "$ISAAC_SCRIPTS/vio_imu_mode"
fi
install -m 0644 \
  "$SCRIPT_DIR/jetson-config/isaac_ros_overrides/.isaac_ros_common-config" \
  "$ISAAC_SCRIPTS/.isaac_ros_common-config"

realsense_patch="$SCRIPT_DIR/jetson-config/isaac_ros_overrides/docker/Dockerfile.realsense.patch"
if grep -q 'ros-humble-isaac-ros-visual-slam' "$ISAAC_COMMON/docker/Dockerfile.realsense"; then
  echo "Isaac ROS RealSense image already contains the VIO runtime packages."
elif git -C "$ISAAC_COMMON" apply --check "$realsense_patch"; then
  git -C "$ISAAC_COMMON" apply "$realsense_patch"
  echo "Installed the versioned VIO runtime layer in Dockerfile.realsense."
else
  echo "Dockerfile.realsense does not match the pinned Isaac ROS base; refusing to patch it." >&2
  exit 1
fi

vslam_patch="$SCRIPT_DIR/jetson-config/isaac_ros_overrides/visual_slam/proven_jetson.patch"
if git -C "$ISAAC_VSLAM" apply --reverse --check "$vslam_patch" 2>/dev/null; then
  echo "Isaac ROS visual-slam source already matches the proven Jetson patch."
elif git -C "$ISAAC_VSLAM" apply --check "$vslam_patch"; then
  git -C "$ISAAC_VSLAM" apply "$vslam_patch"
  echo "Installed the proven Jetson visual-slam source patch."
else
  echo "isaac_ros_visual_slam does not match the pinned source; refusing to patch it." >&2
  exit 1
fi

# No IMU source fixes/driver overlay are required by the restored VO path.

set +u
source /opt/ros/humble/setup.bash
set -u
cd "$REPO_DIR"
# Incremental ament_python installs do not remove files deleted from source.
# Retire only the old Path B artifacts; never erase a workspace/build tree.
# Run deployment while stopped, as required for replacing the bridge itself.
python_version="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
for obsolete in \
  "$REPO_DIR/build/vio_px4_bridge/build/lib/vio_px4_bridge/mavros_ev_bridge.py" \
  "$REPO_DIR/install/vio_px4_bridge/lib/vio_px4_bridge/mavros_ev_bridge" \
  "$REPO_DIR/install/vio_px4_bridge/lib/python$python_version/site-packages/vio_px4_bridge/mavros_ev_bridge.py"; do
  if [[ -e "$obsolete" || -L "$obsolete" ]]; then
    mv -- "$obsolete" "${obsolete}.retired-path-b-${backup_stamp}"
    echo "Retired old Path B artifact (backup kept): $obsolete"
  fi
done
colcon build --packages-select vio_px4_bridge

ln -sfn "$REPO_DIR/scripts/vio-launch/vio-launch" "$HOME/vio-launch"
mkdir -p "$HOME/.local/bin"
ln -sfn "$REPO_DIR/scripts/vio-launch/vio-launch" "$HOME/.local/bin/vio-launch"

echo
echo "Jetson bootstrap complete."
echo "Repository:      $REPO_DIR"
echo "Isaac workspace: $ISAAC_WS"
echo "Launcher:        $HOME/vio-launch"
echo
echo "Start with: ~/vio-launch"
echo "Camera: VO (stereo only); RealSense IMU fusion disabled"
echo "Uses the base-image camera driver; no IMU overlay build is required."
