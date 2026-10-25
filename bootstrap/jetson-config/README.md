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
${ISAAC_ROS_WS}/src/isaac_ros_common/scripts/.isaac_ros_common-config
```

It creates timestamped backups before replacing existing files. The Git copies
under `bootstrap/jetson-config/isaac_ros_overrides/` remain authoritative.

Both large runtime workspaces remain standalone siblings:

```text
~/workspaces/isaac_ros-dev   # NVIDIA cuVSLAM/RealSense container workspace
~/workspaces/mavros          # MAVROS source/build workspace for Path B
~/workspaces/VIO_PX4         # this project
```

The full bootstrap creates and builds the first two; generated build trees and
Docker images are intentionally not nested in this repository or committed.

The active launch is stereo-only VO, using the base-image camera driver with
IMU fusion explicitly disabled. The experimental SDK 2.56.5 overlay is not
built or sourced. Camera extrinsics and bridge safety fixes are preserved.
`bootstrap/docs/STAGED_IMU_IMPLEMENTATION.md` records the inactive IMU experiment.

The exact NVIDIA/JetPack pins, original manual layers, and clean-room rebuild
sequence are documented in `bootstrap/docs/ISAAC_ROS_BOOTSTRAP.md`.
