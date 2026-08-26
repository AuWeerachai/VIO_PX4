#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bootstrap_isaac_ros.sh [--install-deps] [--isaac-ws PATH]

Provision the NVIDIA Isaac ROS/cuVSLAM base workspace needed by VIO_PX4, then
run bootstrap_jetson.sh to install this repository's launch configuration.

The exact Isaac ROS Common commit and NGC asset are pinned in
jetson/isaac_ros_release.env. The script rejects a different JetPack/L4T family
instead of silently installing an incompatible NVIDIA release.
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
COMMON_DIR="$ISAAC_WS/src/isaac_ros_common"
VSLAM_DIR="$COMMON_DIR/isaac_ros_visual_slam"
ASSET_FILE="$ISAAC_WS/isaac_ros_assets/isaac_ros_visual_slam/quickstart_interface_specs.json"
RELEASE_FILE="$REPO_DIR/jetson/isaac_ros_release.env"

[[ -r "$RELEASE_FILE" ]] || { echo "Missing release manifest: $RELEASE_FILE" >&2; exit 1; }
# shellcheck disable=SC1090
source "$RELEASE_FILE"

[[ "$(uname -m)" == "aarch64" ]] || {
  echo "This bootstrap is for the Jetson (aarch64), not $(uname -m)." >&2
  exit 1
}

L4T_RELEASE=""
if [[ -r /etc/nv_tegra_release ]]; then
  L4T_RELEASE="$(sed -n 's/^# R\([0-9][0-9]*\).*REVISION: \([0-9][0-9.]*\).*/\1.\2/p' /etc/nv_tegra_release | head -n1)"
fi

case "$L4T_RELEASE" in
  "$SUPPORTED_L4T_PREFIX"*) ;;
  *)
    echo "Unsupported L4T '${L4T_RELEASE:-unknown}'; expected $SUPPORTED_L4T_PREFIX.x." >&2
    echo "Do not bypass this check without validating JetPack/Isaac ROS compatibility." >&2
    exit 1
    ;;
esac

echo "Jetson Linux: ${L4T_RELEASE:-not reported}"
echo "Isaac ROS:    $ISAAC_ROS_RELEASE @ $ISAAC_COMMON_COMMIT"
echo "VSLAM asset:  $ISAAC_VSLAM_ASSET_VERSION"
echo "Workspace:    $ISAAC_WS"

if $INSTALL_DEPS; then
  sudo apt-get update
  sudo apt-get install -y git git-lfs curl tar
fi

for command_name in git git-lfs curl tar docker; do
  command -v "$command_name" >/dev/null || {
    echo "Missing $command_name. Install JetPack's Docker/NVIDIA container runtime" >&2
    echo "and re-run with --install-deps for the ordinary host tools." >&2
    exit 1
  }
done
docker info >/dev/null 2>&1 || {
  echo "Docker is not usable by $(id -un). Verify the Docker service and group membership." >&2
  exit 1
}

mkdir -p "$ISAAC_WS/src"
if [[ ! -d "$COMMON_DIR/.git" ]]; then
  if [[ -e "$COMMON_DIR" ]]; then
    echo "$COMMON_DIR exists but is not an Isaac ROS Common Git checkout." >&2
    exit 1
  fi
  git clone --branch "$ISAAC_ROS_RELEASE" --single-branch \
    https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git "$COMMON_DIR"
  git -C "$COMMON_DIR" checkout --detach "$ISAAC_COMMON_COMMIT"
else
  common_commit="$(git -C "$COMMON_DIR" rev-parse HEAD)"
  if [[ "$common_commit" != "$ISAAC_COMMON_COMMIT" ]]; then
    echo "Existing isaac_ros_common is $common_commit; expected $ISAAC_COMMON_COMMIT." >&2
    echo "Refusing to rewrite an existing NVIDIA checkout." >&2
    exit 1
  fi
fi

if [[ ! -d "$VSLAM_DIR/.git" ]]; then
  if [[ -e "$VSLAM_DIR" ]]; then
    echo "$VSLAM_DIR exists but is not the expected cuVSLAM source checkout." >&2
    exit 1
  fi
  git clone --branch "$ISAAC_ROS_RELEASE" --single-branch \
    https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git "$VSLAM_DIR"
  git -C "$VSLAM_DIR" checkout --detach "$ISAAC_VSLAM_COMMIT"
else
  vslam_commit="$(git -C "$VSLAM_DIR" rev-parse HEAD)"
  if [[ "$vslam_commit" != "$ISAAC_VSLAM_COMMIT" ]]; then
    echo "Existing isaac_ros_visual_slam is $vslam_commit; expected $ISAAC_VSLAM_COMMIT." >&2
    echo "Refusing to rewrite an existing NVIDIA checkout." >&2
    exit 1
  fi
fi

if [[ ! -f "$ASSET_FILE" ]]; then
  archive_url="https://api.ngc.nvidia.com/v2/resources/nvidia/isaac/isaac_ros_visual_slam_assets/versions/${ISAAC_VSLAM_ASSET_VERSION}/files/quickstart.tar.gz"
  temp_dir="$(mktemp -d)"
  trap 'rm -rf -- "$temp_dir"' EXIT
  curl --fail --location --show-error --output "$temp_dir/quickstart.tar.gz" "$archive_url"
  mkdir -p "$ISAAC_WS/isaac_ros_assets"
  tar -xzf "$temp_dir/quickstart.tar.gz" -C "$ISAAC_WS/isaac_ros_assets"
  printf '%s\n' "$ISAAC_VSLAM_ASSET_VERSION" > "$ISAAC_WS/isaac_ros_assets/.visual_slam_asset_version"
fi

[[ -f "$ASSET_FILE" ]] || {
  echo "Asset download completed but $ASSET_FILE is absent." >&2
  exit 1
}

jetson_args=(--isaac-ws "$ISAAC_WS")
if $INSTALL_DEPS; then
  jetson_args=(--install-deps "${jetson_args[@]}")
fi
"$REPO_DIR/scripts/bootstrap_jetson.sh" "${jetson_args[@]}"

echo
echo "Isaac ROS base and VIO_PX4 deployment are prepared."
echo "The first ./vio-launch start may pull/build the NVIDIA development image; later starts use Docker's cache."
