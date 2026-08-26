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
    python3-pip python3-colcon-common-extensions geographiclib-tools tmux gnome-terminal \
    ros-humble-mavros ros-humble-mavros-extras
  python3 -m pip install --user --upgrade pymavlink
  if [[ ! -f /usr/share/GeographicLib/geoids/egm96-5.pgm ]]; then
    sudo geographiclib-get-geoids egm96-5
  fi
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
  "$REPO_DIR/jetson/isaac_ros_overrides/scripts/start_vio.sh" \
  "$ISAAC_SCRIPTS/start_vio.sh"
install_override \
  "$REPO_DIR/jetson/isaac_ros_overrides/scripts/start_rviz.sh" \
  "$ISAAC_SCRIPTS/start_rviz.sh"
install -m 0644 \
  "$REPO_DIR/jetson/isaac_ros_overrides/.isaac_ros_common-config" \
  "$ISAAC_SCRIPTS/.isaac_ros_common-config"

realsense_patch="$REPO_DIR/jetson/isaac_ros_overrides/docker/Dockerfile.realsense.patch"
if grep -q 'ros-humble-isaac-ros-visual-slam' "$ISAAC_COMMON/docker/Dockerfile.realsense"; then
  echo "Isaac ROS RealSense image already contains the VIO runtime packages."
elif git -C "$ISAAC_COMMON" apply --check "$realsense_patch"; then
  git -C "$ISAAC_COMMON" apply "$realsense_patch"
  echo "Installed the versioned VIO runtime layer in Dockerfile.realsense."
else
  echo "Dockerfile.realsense does not match the pinned Isaac ROS base; refusing to patch it." >&2
  exit 1
fi

vslam_patch="$REPO_DIR/jetson/isaac_ros_overrides/visual_slam/proven_jetson.patch"
if git -C "$ISAAC_VSLAM" apply --reverse --check "$vslam_patch" 2>/dev/null; then
  echo "Isaac ROS visual-slam source already matches the proven Jetson patch."
elif git -C "$ISAAC_VSLAM" apply --check "$vslam_patch"; then
  git -C "$ISAAC_VSLAM" apply "$vslam_patch"
  echo "Installed the proven Jetson visual-slam source patch."
else
  echo "isaac_ros_visual_slam does not match the pinned source; refusing to patch it." >&2
  exit 1
fi

set +u
source /opt/ros/humble/setup.bash
set -u
cd "$REPO_DIR"
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
