#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bootstrap_isaac_ros.sh [--install-deps] [--isaac-ws PATH]
                              [--isaac-release release-3.1|release-3.2]

Provision the NVIDIA Isaac ROS/cuVSLAM base workspace needed by VIO_PX4, then
run bootstrap_jetson.sh to install this repository's launch configuration.

The default Isaac ROS release is selected from the Jetson Linux (L4T) release:
  L4T 36.3.x -> Isaac ROS release-3.1 (JetPack 6.0)
  L4T 36.4.x -> Isaac ROS release-3.2 (JetPack 6.1/6.2)

An unknown platform is rejected. Use --isaac-release only after checking the
NVIDIA compatibility documentation for that JetPack version.
EOF
}

INSTALL_DEPS=false
ISAAC_WS="${ISAAC_ROS_WS:-$HOME/workspaces/isaac_ros-dev}"
ISAAC_RELEASE=""

while (($#)); do
  case "$1" in
    --install-deps) INSTALL_DEPS=true; shift ;;
    --isaac-ws) ISAAC_WS="${2:?--isaac-ws requires a path}"; shift 2 ;;
    --isaac-release) ISAAC_RELEASE="${2:?--isaac-release requires a value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMMON_DIR="$ISAAC_WS/src/isaac_ros_common"
ASSET_FILE="$ISAAC_WS/isaac_ros_assets/isaac_ros_visual_slam/quickstart_interface_specs.json"

[[ "$(uname -m)" == "aarch64" ]] || {
  echo "This bootstrap is for the Jetson (aarch64), not $(uname -m)." >&2
  exit 1
}

L4T_RELEASE=""
if [[ -r /etc/nv_tegra_release ]]; then
  L4T_RELEASE="$(sed -n 's/^# R\([0-9][0-9]*\).*REVISION: \([0-9][0-9.]*\).*/\1.\2/p' /etc/nv_tegra_release | head -n1)"
fi

if [[ -z "$ISAAC_RELEASE" ]]; then
  case "$L4T_RELEASE" in
    36.3*) ISAAC_RELEASE="release-3.1" ;;
    36.4*) ISAAC_RELEASE="release-3.2" ;;
    *)
      echo "Cannot safely select Isaac ROS for L4T '${L4T_RELEASE:-unknown}'." >&2
      echo "Check JetPack compatibility, then pass --isaac-release explicitly." >&2
      exit 1
      ;;
  esac
fi

case "$ISAAC_RELEASE" in
  release-3.1) ISAAC_MAJOR=3; ISAAC_MINOR=1 ;;
  release-3.2) ISAAC_MAJOR=3; ISAAC_MINOR=2 ;;
  *)
    echo "Supported Humble releases are release-3.1 and release-3.2; got $ISAAC_RELEASE." >&2
    exit 1
    ;;
esac

echo "Jetson Linux: ${L4T_RELEASE:-not reported}"
echo "Isaac ROS:    $ISAAC_RELEASE"
echo "Workspace:    $ISAAC_WS"

if $INSTALL_DEPS; then
  sudo apt-get update
  sudo apt-get install -y git curl jq tar
fi

for command_name in git curl jq tar docker; do
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
  git clone --branch "$ISAAC_RELEASE" --single-branch \
    https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git "$COMMON_DIR"
else
  common_branch="$(git -C "$COMMON_DIR" branch --show-current)"
  if [[ "$common_branch" != "$ISAAC_RELEASE" ]]; then
    echo "Existing isaac_ros_common branch is '$common_branch', expected '$ISAAC_RELEASE'." >&2
    echo "Refusing to rewrite an existing NVIDIA workspace." >&2
    exit 1
  fi
fi

if [[ ! -f "$ASSET_FILE" ]]; then
  versions_url="https://api.ngc.nvidia.com/v2/resources/nvidia/isaac/isaac_ros_visual_slam_assets/versions"
  available_versions="$(curl --fail --silent --show-error -H 'Accept: application/json' "$versions_url")"
  asset_version="$(jq -r ".recipeVersions[].versionId
    | select(test(\"^[0-9]+\\.[0-9]+\\.[0-9]+$\"))
    | split(\".\")
    | select(.[0] == \"$ISAAC_MAJOR\" and (.[1] | tonumber) <= $ISAAC_MINOR)
    | join(\".\")" <<<"$available_versions" | sort -V | tail -n1)"
  [[ -n "$asset_version" ]] || {
    echo "No compatible NVIDIA visual-slam asset was found for $ISAAC_RELEASE." >&2
    exit 1
  }
  archive_url="https://api.ngc.nvidia.com/v2/resources/nvidia/isaac/isaac_ros_visual_slam_assets/versions/${asset_version}/files/quickstart.tar.gz"
  temp_dir="$(mktemp -d)"
  trap 'rm -rf -- "$temp_dir"' EXIT
  curl --fail --location --show-error --output "$temp_dir/quickstart.tar.gz" "$archive_url"
  mkdir -p "$ISAAC_WS/isaac_ros_assets"
  tar -xzf "$temp_dir/quickstart.tar.gz" -C "$ISAAC_WS/isaac_ros_assets"
  printf '%s\n' "$asset_version" > "$ISAAC_WS/isaac_ros_assets/.visual_slam_asset_version"
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
