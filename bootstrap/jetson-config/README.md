# Jetson deployment files

This directory contains every project-owned override that otherwise lives
inside the external Isaac ROS workspace.

Run from a fresh clone on a Jetson that already has the Isaac ROS cuVSLAM
workspace:

```bash
cd ~/workspaces/VIO_PX4
./bootstrap/bootstrap_jetson.sh --install-deps
~/vio-launch
```

For a Jetson without the NVIDIA workspace, run
`./bootstrap/bootstrap_isaac_ros.sh --install-deps` instead. It selects only the
supported ROS 2 Humble Isaac ROS release for the detected JetPack/L4T version,
clones NVIDIA's `isaac_ros_common`, downloads the matching visual-SLAM assets,
and then invokes `bootstrap_jetson.sh`. It deliberately rejects unknown JetPack
versions rather than installing an incompatible current release.

The bootstrap script copies the reviewed overrides to:

```text
${ISAAC_ROS_WS}/src/isaac_ros_common/scripts/start_vio.sh
${ISAAC_ROS_WS}/src/isaac_ros_common/scripts/start_rviz.sh
${ISAAC_ROS_WS}/src/isaac_ros_common/scripts/camera_selection.py
${ISAAC_ROS_WS}/src/isaac_ros_common/scripts/check_camera_ready.py
${ISAAC_ROS_WS}/src/isaac_ros_common/scripts/vo_environment.sh
${ISAAC_ROS_WS}/src/isaac_ros_common/scripts/check_vo_environment.py
${ISAAC_ROS_WS}/src/isaac_ros_common/scripts/.isaac_ros_common-config
```

It creates timestamped backups before replacing existing files. The Git copies
under `bootstrap/jetson-config/isaac_ros_overrides/` remain authoritative.

All seven camera runtime files are covered by
`scripts/vio-launch/camera_release.json`. Bootstrap verifies source, manifest
and installed copies; the CLI blocks startup on a mismatch. Updating Git or
rebuilding the bridge alone is not a complete deployment. Stop the pipeline,
install the matching overrides, verify the deployment, and reopen the CLI.
The shared environment loads only the customized cuVSLAM package over the
base ROS installation, avoiding mixed RealSense driver/message versions.

The NVIDIA runtime workspace remains a standalone sibling:

```text
~/workspaces/isaac_ros-dev   # NVIDIA cuVSLAM/RealSense container workspace
~/workspaces/VIO_PX4         # this project
```

The full bootstrap creates the NVIDIA workspace and builds this project's
bridge. Generated build trees and Docker images are intentionally not committed.
MAVROS is no longer a dependency; existing installations are left untouched.

The active launch is stereo-only VO, using the base-image camera driver with
IMU fusion explicitly disabled. The experimental SDK 2.56.5 overlay is not
built or sourced. Camera extrinsics and bridge safety fixes are preserved.
`bootstrap/docs/STAGED_IMU_IMPLEMENTATION.md` records the inactive IMU experiment.

Before starting on a new vehicle, configure the rear D456 SDK serial in the
CLI's Camera extrinsic menu. An empty serial is an error, not automatic camera
selection. A temporary copy of the installed RealSense stereo configuration
adds `serial_no` and `device_type: d456`; it does not edit `/opt/ros` or change
stream settings. The actual camera identity is verified before accepting the
running session. See the [RealSense selection parameters](https://github.com/realsenseai/realsense-ros#parameters).

The exact NVIDIA/JetPack pins, original manual layers, and clean-room rebuild
sequence are documented in `bootstrap/docs/ISAAC_ROS_BOOTSTRAP.md`.
