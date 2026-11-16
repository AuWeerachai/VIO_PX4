# Proven Isaac ROS / cuVSLAM bootstrap

This deployment is intentionally frozen to the configuration observed on the
working `teamc-desktop` Jetson Orin Nano on 2026-08-26. It must not track the
current NVIDIA release: current Isaac ROS documentation has moved beyond this
ROS 2 Humble system.

For crash recovery, NVMe migration, prerequisites and separately saved vehicle
configuration, start with [Jetson recovery checklist](JETSON_RECOVERY.md).
Reviewed 2026-11-16: current camera operation is **stereo VO**, not the IMU experiment.

## Captured platform

| Component | Proven value |
|---|---|
| Jetson Linux / L4T | Earlier machine notes record `36.4.7`; bootstrap checks the `36.4` family. See version distinction below. |
| Ubuntu / ROS | Ubuntu 22.04 / ROS 2 Humble |
| Isaac ROS | `release-3.2` |
| `isaac_ros_common` | `fcf4d9e17f8f0a7f47f1d22d6a18421ce3768c01` |
| `isaac_ros_visual_slam` | `e31f4cc1d41a329a01946e5fe63669f8b15da677` |
| Visual-SLAM NGC asset | `3.1.0` (the asset used by the 3.2 documentation) |
| Active VO / base-image librealsense | `v2.55.1` |
| Inactive experimental IMU overlay | `v2.56.5`, commit `38a41441971387197193ad3aeae3cefe6a11f2cb`; not built/sourced by current startup |
| NVIDIA RealSense ROS branch | `release/4.51.1-isaac` |
| Inactive IMU wrapper commit | `c04f43308c00a6f495f14a47cc84cc293047cdff` |
| Docker image key | `aarch64.ros2_humble.realsense` |

NVIDIA's official JetPack **6.2.1** image contains L4T **36.4.4**. Do not treat
the earlier installed-system note `36.4.7` as an official image version or as
independently reverified hardware. Compare the actual disk's release/package
inventory and carrier requirements before choosing a flash image. Passing the
script's family check alone is not compatibility validation.
[NVIDIA JetPack 6.2.1](https://developer.nvidia.com/embedded/jetpack-sdk-621)

The authoritative machine-readable pins are in
`bootstrap/jetson-config/isaac_ros_release.env` (base dependencies) and
`bootstrap/jetson-config/isaac_ros_overrides/scripts/vio_imu_release.env`
(inactive experimental IMU overlay). Apt/pip packages and all container layers
are not fully locked by digest; these pins are not a complete offline image.

## Why the original setup was difficult to reproduce

The working workspace was assembled across several layers:

1. JetPack supplies the NVIDIA kernel/GPU platform. Docker and its NVIDIA
   runtime integration must be installed and working on the host.
2. `isaac_ros_common/scripts/run_dev.sh` selects and builds the Humble +
   RealSense development image.
3. A manual edit to NVIDIA's `Dockerfile.realsense` installs
   `isaac_ros_visual_slam`, `isaac_ros_examples`, and `isaac_ros_realsense`.
4. `isaac_ros_visual_slam` is a nested source checkout inside
   `isaac_ros_common`, rather than a normal sibling repository.
5. Historical source patches changed a launch default to 30 FPS, enabled IMU
   by default and increased the image-jitter threshold to 60 ms. These are not
   the active runtime settings: our current launch explicitly disables IMU and
   uses the RealSense stereo configuration (observed at 640x360, 90 Hz).
6. Custom `start_vio.sh` and `start_rviz.sh` files live inside NVIDIA's checkout.
7. The NGC interface-spec asset is outside Git.

Items 3–7 were therefore easy to lose when rebuilding or cloning only one
repository. They are now represented by versioned manifests, patches, and
launch overrides in VIO_PX4.

## Fresh installation

Start from a Jetson with the reviewed compatible 36.4-family BSP, host ROS
Humble and working Docker/NVIDIA runtime already provisioned. Follow the
[fresh-host requirements](JETSON_RECOVERY.md#51-provision-the-host-first),
including storage, device permissions and the desktop session. `--install-deps`
installs additional tools; it does not flash the module, install the full base
platform or recover vehicle-specific configuration:

```bash
cd ~/workspaces
git clone -b vio-as-gps git@github.com:AuWeerachai/VIO_PX4.git
cd VIO_PX4
./bootstrap/bootstrap_isaac_ros.sh --install-deps
./vio-launch
```

The bootstrap performs these operations in order:

1. Refuses non-aarch64 systems and an L4T family other than 36.4.x.
2. Checks that Docker responds. Actual GPU access must also be validated in the
   container; `docker info` alone does not establish that.
3. Clones both NVIDIA repositories at the captured commits.
4. Downloads the pinned NGC visual-SLAM asset.
5. Calls `bootstrap_jetson.sh`, which installs optional host dependencies.
6. Installs project-owned body-frame/extrinsic launch scripts and applies the
   Docker/cuVSLAM source patches. IMU fusion is explicitly disabled at launch;
   no experimental IMU overlay is built or loaded.
7. Builds the VIO/PX4 ROS package and exposes `~/vio-launch` and
   `~/.local/bin/vio-launch`. Saved CLI values must be restored separately from
   `~/.config/vio-launch/config.json`.

Camera validation is a **runtime** step, not performed by bootstrap: Path A
checks camera health during stationary spoofing. Restore body-to-camera extrinsics, serial and recovery settings
before use. An existing IMU-enabled session must be fully stopped.

Path B has been removed. Bootstrap no longer installs, patches or builds
MAVROS. Existing standalone workspaces are left untouched.

The first camera start through `./vio-launch` asks `run_dev.sh` to build/pull the large
Isaac ROS image. Docker caches unchanged layers, so subsequent launches should
reuse the image. Changing a relevant Dockerfile invalidates that layer and
causes a rebuild.

## Source edits versus launch configuration

The captured source patch is retained because the requirement is to reproduce
the proven machine exactly. For normal operation, VIO_PX4 also passes the
60 ms jitter threshold explicitly and selects **VO (stereo only)**.
The inactive experimental SDK and wrapper are a private workspace overlay: the Docker image
and `/usr/local` driver are not replaced. The mounted overlay survives container
recreation. Its build recipe, not generated binaries, is kept in Git.

The October 2026 bench trial confirmed that SDK 2.55.1 could not deliver motion
samples with D456 firmware 5.17.0.10, while the isolated SDK 2.56.5 could. The
newer SDK handles the firmware's newer IMU packet format. IMU delivery and
stationary tracking passed; this is not flight qualification. The camera's
missing-IMU-calibration warning still requires attention.

The current launcher ignores old IMU mode-selection files and always uses VO.
The compatibility helper `~/vio-imu-mode` reports VO and refuses to enable IMU.
Fully stop existing sessions before launching the restored baseline. A running
session with the wrong fusion mode or an explicitly sourced `.vio-imu` overlay
is rejected. That check is not a complete library-compatibility audit: do not
mix the base RealSense driver with newer `realsense2_camera_msgs` from another
overlay. See [deployment pitfalls](JETSON_RECOVERY.md#7-known-deployment-pitfalls).

Do not copy generated `build/`, `install/`, Docker images, the 11 GB asset
directory, logs, core dumps, swap files, or RViz-generated frame diagrams into
Git. The bootstrap reconstructs the required generated state.
