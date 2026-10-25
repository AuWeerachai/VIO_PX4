#!/usr/bin/env bash
# Run inside the Isaac container as admin. No system SDK replacement or flashing.
set -euo pipefail
: "${ISAAC_ROS_WS:?Set ISAAC_ROS_WS to the container workspace}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/vio_imu_release.env"
overlay="$ISAAC_ROS_WS/.vio-imu/$VIO_IMU_OVERLAY_ID"
signature="$VIO_IMU_SDK_COMMIT $VIO_IMU_ROS_COMMIT $VIO_IMU_OVERLAY_ID"
mkdir -p "$overlay"
exec 9>"$overlay/build.lock"
flock -w 2400 9 || { echo 'VIO setup failed: another overlay build is still running' >&2; exit 1; }
if [[ -f "$overlay/READY" && "$(cat "$overlay/READY")" == "$signature" &&
      -f "$overlay/sdk/lib/librealsense2.so.2.56.5" &&
      -f "$overlay/ros_install/local_setup.bash" &&
      -f "$overlay/ros_install/realsense2_camera/lib/librealsense2_camera.so" ]]; then
  echo 'VIO: using cached RealSense 2.56.5 / NVIDIA camera overlay.'
  exit 0
fi
trap 'echo "VIO setup failed: inspect the build output; no GPS output should start." >&2' ERR
echo 'VIO: building pinned IMU driver and ROS wrapper (first start only; may take 20–40 minutes).'
checkout_pinned() {
  local url="$1" commit="$2" destination="$3"
  if [[ ! -e "$destination" ]]; then
    mkdir -p "$destination"
    git -C "$destination" init -q
    git -C "$destination" remote add origin "$url"
    git -C "$destination" fetch --depth 1 origin "$commit"
    git -C "$destination" checkout --detach FETCH_HEAD
  fi
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$commit" ]] || {
    echo "Unexpected source revision at $destination; refusing to overwrite it." >&2; return 1;
  }
  [[ -z "$(git -C "$destination" status --porcelain --untracked-files=no)" ]] || {
    echo "Modified source at $destination; refusing to build an unpinned driver." >&2; return 1;
  }
}
checkout_pinned https://github.com/realsenseai/librealsense.git "$VIO_IMU_SDK_COMMIT" "$overlay/sdk_source"
checkout_pinned https://github.com/NVIDIA-ISAAC-ROS/realsense-ros.git "$VIO_IMU_ROS_COMMIT" "$overlay/ros_ws/src/realsense-ros"
cmake -S "$overlay/sdk_source" -B "$overlay/sdk_build" \
  -DFORCE_RSUSB_BACKEND=ON -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF \
  -DBUILD_TOOLS=OFF -DBUILD_WITH_CUDA=OFF -DBUILD_PYTHON_BINDINGS=OFF \
  -DBUILD_UNIT_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build "$overlay/sdk_build" --parallel 2
# DESTDIR also contains absolute install destinations; nothing goes into /usr/local.
DESTDIR="$overlay/stage" cmake --install "$overlay/sdk_build"
mkdir -p "$overlay/sdk"
cp -a "$overlay/stage/usr/local/." "$overlay/sdk/"
set +u
source /opt/ros/humble/setup.bash
set -u
cd "$overlay/ros_ws"
CMAKE_BUILD_PARALLEL_LEVEL=2 MAKEFLAGS=-j2 colcon build \
  --packages-select realsense2_camera --parallel-workers 1 \
  --install-base "$overlay/ros_install" \
  --cmake-args -Drealsense2_DIR="$overlay/sdk/lib/cmake/realsense2" -DCMAKE_BUILD_TYPE=Release
[[ -f "$overlay/ros_install/realsense2_camera/lib/librealsense2_camera.so" ]]
printf '%s\n' "$signature" > "$overlay/READY.tmp"
mv "$overlay/READY.tmp" "$overlay/READY"
echo 'VIO: compatible IMU driver ready. The original VO installation is unchanged.'
